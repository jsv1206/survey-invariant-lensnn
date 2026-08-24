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


def ssim(a, b, C1=0.01 ** 2, C2=0.03 ** 2, window=7, mask=None):
    """Mean SSIM between two [B,1,H,W] images.

    `C1`/`C2` assume data in [0,1] (the standard SSIM convention), but these
    images peak at ~0.017-0.19 depending on survey (CLAUDE.md bug 6), so both
    inputs are first divided by `b`'s per-sample peak -- every call site here
    passes (f_hat, f_gt), so `b` is always ground truth. A SHARED scalar keeps
    the comparison fair (do not normalize `a` by its own peak, which would
    hide amplitude error); without this, measured ssim(blurred, sharp) was
    0.999993 (HST)/0.986732 (DES), i.e. saturated. `lambda_a` is unchanged --
    this is a normalization fix, not a reweighting.

    `mask` (optional [B,1,H,W] bool/float) restricts the mean to the given
    pixels, per sample, then averages across the batch -- used for the
    masked/normalized cross-survey metrics (HANDOFF_run1.md Step 3), since
    images are ~96-97% empty sky and an unmasked mean is mostly a trivial
    background-agreement number.

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
    if mask is None:
        return ssim_map.mean()
    mask = mask.to(ssim_map.dtype)
    denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    per_sample = (ssim_map * mask).sum(dim=(1, 2, 3)) / denom
    return per_sample.mean()


def fft_l1(a, b):
    """L1 distance between magnitude spectra of two images.

    Always computed in float32: torch.fft.rfft2 does not support bf16/fp16
    (RuntimeError: Unsupported dtype), and under torch.autocast the inputs
    here would otherwise arrive as whatever reduced-precision dtype the
    upstream linear/conv layers produced -- FFT isn't in autocast's
    auto-promoted op list, so it must be handled explicitly.
    """
    Fa = torch.fft.rfft2(a.float(), norm="ortho").abs()
    Fb = torch.fft.rfft2(b.float(), norm="ortho").abs()
    return (Fa - Fb).abs().mean()


def _reblur_full(f_hat, h):
    """LEGACY spatial-domain reblur: convolve with the FULL-FOV PSF stamp, one
    sample at a time. O(npix**4) per batch (the stamp is npix x npix and is
    almost entirely zero far from center for a narrow Gaussian PSF).

    **This carries a half-pixel offset and is NOT a correctness reference.**
    `gaussian_psf_stamp` centers the stamp on the image coordinate origin,
    index `(npix-1)/2`, which is a half-integer whenever npix is EVEN -- which
    it is for most survey/FOV combinations (at FOV 20" the sizes are
    400/200/182/100/76; at FOV 12" they are 240/120/109/60/46, so only Roman
    is odd). A convolution kernel's origin is zero
    LAG, and integer `pad` arithmetic cannot place zero lag on a half-integer
    index -- so this shifts the result by +0.5 px on both axes. Measured
    against the simulator's own forward model, that inflates
    ||f_gt * h - g||^2 / mean(g^2) from ~1e-4 to ~5e-2, i.e. 100-600x.

    Kept only to reproduce pre-fix runs and to pin the regression in
    tests/test_reblur.py. Use `reblur()`, which is exact for even and odd
    npix alike. See CLAUDE.md bug 11.
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
    """Convolve reconstruction f_hat with PSF stamp h. Returns [B,1,H,W].

    f_hat: [B,1,H,W], h: [B,1,H,W] (same grid as f_hat, lens-centered).
    `sigma_px_target` is accepted for backwards compatibility and unused.

    Done in the Fourier domain with an explicit phase ramp, because the stamp
    is sampled on the IMAGE coordinate convention -- lag `i - (npix-1)/2` --
    and `npix` is even for most survey/FOV combinations here (400/200/182/100/76
    at FOV 20"; 240/120/109/60/46 at FOV 12"), so the stamp's center of
    symmetry usually falls BETWEEN pixels -- and the parity varies with FOV, so
    it cannot be assumed either way. A spatial conv2d can only
    place zero lag on an integer index, so the previous cropped-kernel path
    shifted the result by +0.5 px on both axes for every sample (see
    `_reblur_full` and CLAUDE.md bug 11). The ramp
    `exp(2*pi*i*((H-1)/2*ky + (W-1)/2*kx))` moves the stamp's continuous center
    to zero lag exactly, for even and odd npix alike.

    Measured against the simulator's own convolve-then-resample forward model,
    ||reblur(f_gt,h) - g||^2 / mean(g^2) drops from ~5e-2 to 7.7e-5 (HST),
    1.3e-4 (Euclid), 1.2e-4 (Rubin), 2.8e-4 (DES) -- i.e. this is now a
    faithful data-consistency operator rather than a shift detector. Roman
    lands at 3.6e-3, 4.5x above the analytic-transfer-function floor of 8.0e-4,
    which is dataset bug 3 (PSF sigma = 0.46 native px, so the stamp aliases);
    that residual is a stamp-sampling limit, not a centering error.

    The phase ramp is exact for odd npix too, which matters because npix parity
    depends on the FOV: shrinking 20" -> 12" turns Roman odd (182 -> 109). The
    residuals are unchanged by that move -- 9.81e-6 / 1.61e-4 / 3.06e-5 /
    9.31e-5 for HST/Euclid/Rubin/DES at BOTH FOVs, with Roman 1.83e-3 -> 2.07e-3
    (its bug-3 aliasing, not a centering or wrap-around effect).

    Wrap-around is not an issue at these FOVs (the arc is central and the
    frames are ~97% empty sky): the FFT result matches a zero-padded spatial
    conv with the same centering to the printed precision above.
    """
    B, _, H, W = f_hat.shape
    # FFT in fp32: bf16 autocast is active during training and torch.fft has no
    # bf16 kernel; the reblur term is a small part of the step so this is cheap.
    dtype_in = f_hat.dtype
    f32 = f_hat.float()
    h32 = h.float()
    # PSF stamps must stay sum-normalized to 1, else the term silently rescales
    # the reconstruction (project convention; see CLAUDE.md).
    h32 = h32 / h32.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)

    ky = torch.fft.fftfreq(H, device=f_hat.device, dtype=torch.float32).view(-1, 1)
    kx = torch.fft.fftfreq(W, device=f_hat.device, dtype=torch.float32).view(1, -1)
    ramp = torch.exp(2j * torch.pi * (((H - 1) / 2.0) * ky + ((W - 1) / 2.0) * kx))

    out = torch.fft.ifft2(torch.fft.fft2(f32) * torch.fft.fft2(h32) * ramp).real
    return out.to(dtype_in)


def reconstruction_loss(f_hat, f_gt, lambda_a=0.5, lambda_b=0.1, mask_thresh=0.01):
    """L1 + lambda_a*(1-SSIM) + lambda_b*FFT, with the L1 term peak-normalized
    and half-weighted onto the arc.

    Two measured problems with the previous plain `(f_hat-f_gt).abs().mean()`:

    1. It was in RAW surface-brightness units, so its magnitude tracked survey
       brightness: mean|f_gt| is 2.5e-4 at HST but 7.0e-3 at DES, i.e. the
       pixel-L1 gradient was ~27x weaker on HST than on DES. For a model whose
       central claim is survey invariance, the loss itself was survey-biased.
    2. It was a global mean over images that are ~97% empty sky (measured
       signal fraction 2.1-3.0%), so the arc -- the only part anyone cares
       about -- contributed a few percent of the term.

    Now: divide both images by the per-sample f_gt peak (a shared scalar, so
    amplitude error is preserved), then average an arc-masked term with the
    global term. The masked half puts gradient where the signal is; the global
    half is retained deliberately as the anti-hallucination pressure that
    stopped Run 1's decoder from filling empty sky with junk (it emitted ~40%
    negative pixels and 1.9x the true total flux at HST).
    """
    peak = f_gt.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    a, b = f_hat / peak, f_gt / peak
    absdiff = (a - b).abs()
    m = (b > mask_thresh).to(absdiff.dtype)
    l1_masked = ((absdiff * m).sum(dim=(1, 2, 3))
                 / m.sum(dim=(1, 2, 3)).clamp_min(1.0)).mean()
    l1_global = absdiff.mean()
    l1 = 0.5 * (l1_masked + l1_global)
    s = 1.0 - ssim(f_hat, f_gt)
    ff = fft_l1(f_hat, f_gt)
    total = l1 + lambda_a * s + lambda_b * ff
    return total, dict(l1=l1.detach(), ssim_term=s.detach(), fft=ff.detach(),
                       l1_masked=l1_masked.detach(), l1_global=l1_global.detach())


def reblur_loss(f_hat, h, g):
    """Self-supervised data consistency ||f_hat * h - g||^2, made SCALE-FREE.

    As a raw MSE on surface brightness this term was ~1e-4 in absolute value
    and contributed 0.42% of the training loss (0.08% on validation) at
    lambda_c=1.0 -- i.e. the paper's headline sim-to-real contribution carried
    essentially no gradient, and its weight silently varied with survey
    brightness. Dividing by mean(g^2) makes it a relative residual, so one
    lambda_c means the same thing for every survey.
    """
    g_hat = reblur(f_hat, h)
    return F.mse_loss(g_hat, g) / (g ** 2).mean().clamp_min(1e-12)


def regression_loss(theta_hat, theta_true):
    return F.mse_loss(theta_hat, theta_true)
