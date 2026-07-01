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
import torch
import torch.nn.functional as F


def ssim(a, b, C1=0.01 ** 2, C2=0.03 ** 2, window=7):
    """Mean SSIM between two [B,1,H,W] images (values roughly in a shared range).

    Uses a uniform window (box filter); adequate as a structural regularizer.
    """
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


def reblur(f_hat, h):
    """Convolve reconstruction f_hat with PSF stamp h (same-padding, per-sample).

    f_hat: [B,1,H,W], h: [B,1,Hp,Wp] (odd kernel preferred). Returns [B,1,H,W].
    Done per-sample because each item may have its own grid/kernel size.
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
