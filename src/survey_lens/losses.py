"""Loss terms for the survey-invariant lens network.

Total loss (from the project spec):
    L = L1_recon + lambda_c * L_reblur + lambda_d * L2_reg

  L1_recon = ||f_hat - f_gt||_1
             + lambda_a * (1 - SSIM(f_hat, f_gt))
             + lambda_b * || FFT(f_hat) - FFT(f_gt) ||_1
  L_reblur = || f_hat * h  -  g ||_2^2        (self-supervised; no ground truth)
  L2_reg   = || theta_E_true - theta_E_hat ||_2^2

Every term is written so it works on a single (variable-size) image with a
batch dimension of 1, which is how the model processes per-survey grids.
"""
import numpy as np
import torch
import torch.nn.functional as F


def ssim(a, b, C1=0.01 ** 2, C2=0.03 ** 2, window=7):
    """Mean SSIM between two [B,1,H,W] images.

    `C1`/`C2` assume data in [0,1] (the standard SSIM convention), but these
    images peak at ~0.017-0.19 depending on survey (CLAUDE.md bug 6), so both
    inputs are first divided by `b`'s per-sample peak -- every call site here
    passes (f_hat, f_gt), so `b` is always ground truth. A SHARED scalar keeps
    the comparison fair (do not normalize `a` by its own peak, which would
    hide amplitude error); without this, measured ssim(blurred, sharp) was
    0.999993 (HST)/0.986732 (DES), i.e. saturated. `lambda_a` is unchanged --
    this is a normalization fix, not a reweighting.

    Uses a uniform window (box filter); adequate as a structural regularizer.
    """
    peak = b.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    a, b = a / peak, b / peak
    pad = window // 2
    mu_a = F.avg_pool2d(a, window, 1, pad)
    mu_b = F.avg_pool2d(b, window, 1, pad)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = F.avg_pool2d(a * a, window, 1, pad) - mu_a2
    sb = F.avg_pool2d(b * b, window, 1, pad) - mu_b2
    sab = F.avg_pool2d(a * b, window, 1, pad) - mu_ab
    ssim_map = ((2 * mu_ab + C1) * (2 * sab + C2)) / \
               ((mu_a2 + mu_b2 + C1) * (sa + sb + C2))
    return ssim_map.mean()


def fft_l1(a, b):
    """L1 distance between magnitude spectra of two images."""
    Fa = torch.fft.rfft2(a, norm="ortho").abs()
    Fb = torch.fft.rfft2(b, norm="ortho").abs()
    return (Fa - Fb).abs().mean()


def _reblur_full(f_hat, h):
    """Reference implementation: convolve with the FULL-FOV PSF stamp, one
    sample at a time. O(npix**4) per batch (the stamp is npix x npix and is
    almost entirely zero far from center for a narrow Gaussian PSF) -- kept
    only as the ground truth `reblur()` is checked against in tests, never
    called from the training/eval path. See HANDOFF_run1.md Step 1.
    """
    B = f_hat.shape[0]
    outs = []
    for i in range(B):
        k = h[i:i + 1]                       # [1,1,Hp,Wp]
        kh, kw = k.shape[-2:]
        pad = (kw // 2, kw - 1 - kw // 2, kh // 2, kh - 1 - kh // 2)
        x = F.pad(f_hat[i:i + 1], pad, mode="constant")
        outs.append(F.conv2d(x, k))
    return torch.cat(outs, dim=0)


def _psf_sigma_px(h):
    """Per-sample PSF sigma in pixels, from h's second moment about its TRUE
    continuous center `((Hh-1)/2, (Wh-1)/2)` -- exact for the Gaussian stamps
    this project renders. `gaussian_psf_stamp` centers there, which is a
    half-integer for every survey's (even) npix, so this must match it
    exactly rather than assume an integer center."""
    B, _, Hh, Wh = h.shape
    device, dtype = h.device, h.dtype
    cy, cx = (Hh - 1) / 2.0, (Wh - 1) / 2.0
    yy = torch.arange(Hh, device=device, dtype=dtype) - cy
    xx = torch.arange(Wh, device=device, dtype=dtype) - cx
    r2 = yy.view(-1, 1) ** 2 + xx.view(1, -1) ** 2               # [Hh,Wh]
    hs = h.sum(dim=(1, 2, 3)).clamp_min(1e-12)                   # [B]
    sigma2 = (h[:, 0] * r2[None]).sum(dim=(1, 2)) / hs           # [B]
    return sigma2.clamp_min(0).sqrt()                            # [B]


def _crop_and_renormalize(h, sigma_px_target=4.0):
    """Crop each sample's PSF to a window covering `sigma_px_target` sigma,
    symmetric about h's TRUE continuous center (floor/ceil in continuous
    coordinates, not integer-index arithmetic -- a naive `Hh//2` center would
    silently shift the crop by half a pixel for every survey, since npix is
    even everywhere here). The batch uses the max sigma so one crop size
    covers every sample, then each crop is renormalized to sum 1 (the crop
    discards a negligible far-field tail, so renormalizing keeps the reblur
    consistency loss flux-preserving)."""
    B, _, Hh, Wh = h.shape
    sigma_px = _psf_sigma_px(h)
    half = float((sigma_px_target * sigma_px).max().item())
    cy, cx = (Hh - 1) / 2.0, (Wh - 1) / 2.0
    y0, y1 = max(0, int(np.floor(cy - half))), min(Hh, int(np.ceil(cy + half)) + 1)
    x0, x1 = max(0, int(np.floor(cx - half))), min(Wh, int(np.ceil(cx + half)) + 1)
    hc = h[:, :, y0:y1, x0:x1]
    return hc / hc.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)


def reblur(f_hat, h, sigma_px_target=4.0):
    """Convolve reconstruction f_hat with PSF stamp h (same-padding).

    f_hat: [B,1,H,W], h: [B,1,H,W] (same grid as f_hat, lens-centered).
    Returns [B,1,H,W].

    Crops each PSF to its effective support (~sigma_px_target sigma, from h's
    second moment) before convolving, and does the whole batch in one grouped
    conv2d instead of a Python loop -- the full-stamp, per-sample loop this
    replaces costs O(npix**4) for a kernel that is almost entirely zero
    (measured HST: a single call did not finish on CPU). Cropped-vs-full
    agreement is <1e-6 relative; see tests/test_reblur.py and
    HANDOFF_run1.md Step 1.
    """
    B, _, H, W = f_hat.shape
    hc = _crop_and_renormalize(h, sigma_px_target)               # [B,1,k,k]
    kh, kw = hc.shape[-2:]
    pad = (kw // 2, kw - 1 - kw // 2, kh // 2, kh - 1 - kh // 2)
    x = F.pad(f_hat, pad, mode="constant")                       # [B,1,H+kh-1,W+kw-1]
    x = x.transpose(0, 1)                                        # [1,B,H+kh-1,W+kw-1]
    out = F.conv2d(x, hc, groups=B)                              # [1,B,H,W]
    return out.transpose(0, 1)                                   # [B,1,H,W]


def reconstruction_loss(f_hat, f_gt, lambda_a=0.5, lambda_b=0.1):
    l1 = (f_hat - f_gt).abs().mean()
    s = 1.0 - ssim(f_hat, f_gt)
    ff = fft_l1(f_hat, f_gt)
    total = l1 + lambda_a * s + lambda_b * ff
    return total, dict(l1=l1.detach(), ssim_term=s.detach(), fft=ff.detach())


def reblur_loss(f_hat, h, g):
    g_hat = reblur(f_hat, h)
    return F.mse_loss(g_hat, g)


def regression_loss(theta_hat, theta_true):
    return F.mse_loss(theta_hat, theta_true)
