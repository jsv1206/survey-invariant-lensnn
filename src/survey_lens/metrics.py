"""Frequency-domain reconstruction metrics.

Kept out of `losses.py` on purpose: these are EVAL-ONLY and non-differentiable
by intent, and they force float32 (`torch.fft` rejects bf16 -- the same
workaround `losses.fft_l1` already carries).

Why this module exists at all
-----------------------------
the project notes: "Global PSNR hides the deconvolution effect because it behaves like
the pixel-L1 term. Report high-frequency (top-octave) radial power-spectrum
error as the reconstruction metric." "Deconvolution
means beating `identity` in the top octave specifically." Run 1's
reconstruction verdict was adjudicated with masked L1/SSIM -- exactly the
metrics the docs say cannot decide it -- because this metric did not exist.

Why a power RATIO is not enough
-------------------------------
The literal Step-10 wording ("relative error integrated over the top octave")
describes a POWER ratio, which is phase-blind and therefore gameable. Measured
on the Run-1 checkpoint: Euclid's top-octave power ratio is a healthy-looking
0.86, but its `hf_complex_err` is 1.10 -- and since `f_hat = 0` scores exactly
1.0 by construction, that means the network is WORSE THAN EMITTING NOTHING in
the top octave. It puts roughly the right amount of power in the wrong place.
Reporting the ratio alone would have published "Euclid recovers 86% of
top-octave power", which is false in substance.

So `hf_complex_err` and `hf_coherence` are the adjudicating numbers;
`hf_power_ratio`/`hf_power_err` are retained because Step 10 names them and
because the signed ratio distinguishes over- from under-deconvolution.

Deviation from the Step-10 spec, deliberate
-------------------------------------------
Step 10 says to compute this on the "normalized, masked" images. Multiplying by
the binary arc mask (`f_gt > 0.01*peak`) is wrong for a power spectrum: a hard
edged indicator has its own spectrum falling only as k^-3, which injects
broadband top-octave power into BOTH images and dilutes the discriminant. We
use an apodized centered crop instead (fixed arcsec box + Hann window), and
keep the per-sample peak normalization (a harmless shared scalar that cancels
in every ratio here). `hf_*_full` variants on the uncropped frame are reported
alongside so the choice is auditable.
"""
import numpy as np
import torch

from . import env_setup  # noqa: F401


def _hann2d(h, w, device, dtype):
    wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
    wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
    return wy[:, None] * wx[None, :]


def _freq_grids(h, w, device):
    """Returns (fy, fx, u) where u = 2*|k| so that u == 1.0 IS the native
    Nyquist frequency by construction. f_hat and f_gt always live on the same
    native grid, so cross-survey comparability is automatic -- no resampling
    (forbids resampling surveys onto a common grid)."""
    fy = torch.fft.fftfreq(h, device=device)[:, None].expand(h, w)
    fx = torch.fft.fftfreq(w, device=device)[None, :].expand(h, w)
    return fy, fx, 2.0 * torch.sqrt(fy ** 2 + fx ** 2)


def patch_harmonic_mask(h, w, patch=4, tol_bins=1.5, device="cpu"):
    """Cells sitting on the patch-decoder grid harmonics (|fx| or |fy| = j/patch).

    A patch-independent decoder emits energy at the patch period; for patch=4
    that is 0.25 cyc/px == EXACTLY 0.5x Nyquist, i.e. the lower edge of the
    top octave we measure deconvolution in. So decoder artifacts contaminate
    the very band that adjudicates the paper's claim, and the contaminated
    fraction must be reported alongside the metric rather than hidden.

    The tolerance is in FREQUENCY units, not integer bin indices: Roman's
    npix=182 is not a multiple of `patch`, so the 0.25 cyc/px harmonic falls at
    bin index 45.5 and straddles two bins. Integer-index arithmetic silently
    misses it.
    """
    fy, fx, _ = _freq_grids(h, w, device)
    mask = torch.zeros(h, w, dtype=torch.bool, device=device)
    for j in range(1, patch // 2 + 1):
        f0 = j / patch
        mask |= (fx.abs() - f0).abs() < (tol_bins / w)
        mask |= (fy.abs() - f0).abs() < (tol_bins / h)
    return mask


def apodized_center_crop(x, s, half_arcsec=4.0):
    """Crop [B,1,H,W] to a fixed ARCSEC box about the image center, then apply a
    Hann window. The physical box is constant across surveys while the pixel
    count is not -- which is the point (native-scale handling).

    A single crop size must serve the whole batch, so we size it from the
    largest `s` present (the smallest pixel box), guaranteeing it is in-bounds
    for every sample. In evaluation every sample in a survey shares one `s`, so
    this is exact there.
    """
    B, _, H, W = x.shape
    s_max = float(torch.as_tensor(s).max())
    half = int(round(half_arcsec / max(s_max, 1e-9)))
    half = max(4, min(half, H // 2, W // 2))
    cy, cx = H // 2, W // 2
    out = x[:, :, cy - half:cy + half, cx - half:cx + half]
    win = _hann2d(out.shape[-2], out.shape[-1], out.device, out.dtype)
    return out * win


def _band_stats(F_hat, F_gt, sel):
    """Top-octave statistics over the boolean frequency selection `sel`.

    Returns (power_ratio, complex_err, coherence), each [B].
      power_ratio = sum|F_hat|^2 / sum|F_gt|^2      (phase-BLIND; Step-10 wording)
      complex_err = sum|F_hat-F_gt|^2 / sum|F_gt|^2 (phase-aware; f_hat=0 -> exactly 1.0)
      coherence   = |sum F_hat conj(F_gt)| / sqrt(sum|F_hat|^2 sum|F_gt|^2)
    """
    sel = sel[None, None]
    p_hat = (F_hat.abs() ** 2 * sel).sum(dim=(-3, -2, -1))
    p_gt = (F_gt.abs() ** 2 * sel).sum(dim=(-3, -2, -1)).clamp_min(1e-30)
    d = ((F_hat - F_gt).abs() ** 2 * sel).sum(dim=(-3, -2, -1))
    cross = ((F_hat * F_gt.conj()) * sel).sum(dim=(-3, -2, -1)).abs()
    coh = cross / (p_hat.clamp_min(1e-30) * p_gt).sqrt()
    return p_hat / p_gt, d / p_gt, coh


def _metrics_on(f_hat, f_gt, patch):
    """Core: FFT both, then top-octave stats with and without the patch harmonics."""
    F_hat = torch.fft.fft2(f_hat.float())
    F_gt = torch.fft.fft2(f_gt.float())
    H, W = f_hat.shape[-2:]
    _, _, u = _freq_grids(H, W, f_hat.device)
    top = (u >= 0.5) & (u <= 1.0)
    harm = patch_harmonic_mask(H, W, patch=patch, device=f_hat.device)

    ratio, cerr, coh = _band_stats(F_hat, F_gt, top)
    ratio_x, cerr_x, coh_x = _band_stats(F_hat, F_gt, top & ~harm)

    # what fraction of each image's top-octave power sits on the decoder grid
    sel, hsel = top[None, None], (top & harm)[None, None]
    ph = (F_hat.abs() ** 2 * sel).sum(dim=(-3, -2, -1)).clamp_min(1e-30)
    pg = (F_gt.abs() ** 2 * sel).sum(dim=(-3, -2, -1)).clamp_min(1e-30)
    hf_hat = (F_hat.abs() ** 2 * hsel).sum(dim=(-3, -2, -1)) / ph
    hf_gt = (F_gt.abs() ** 2 * hsel).sum(dim=(-3, -2, -1)) / pg
    return dict(hf_power_ratio=ratio, hf_power_err=(ratio - 1.0).abs(),
                hf_complex_err=cerr, hf_coherence=coh,
                hf_power_ratio_excl=ratio_x, hf_complex_err_excl=cerr_x,
                hf_coherence_excl=coh_x,
                hf_harm_frac_hat=hf_hat, hf_harm_frac_gt=hf_gt)


@torch.no_grad()
def hf_power_metrics(f_hat, f_gt, s, patch=4, half_arcsec=4.0):
    """Top-octave (0.5-1.0 x Nyquist) reconstruction metrics. [B,1,H,W] -> dict of [B].

    Computed on the apodized centered crop (see module docstring); `*_full`
    duplicates on the uncropped frame so the crop choice is auditable.
    """
    peak = f_gt.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    a, b = f_hat / peak, f_gt / peak
    out = _metrics_on(apodized_center_crop(a, s, half_arcsec),
                      apodized_center_crop(b, s, half_arcsec), patch)
    for k, v in _metrics_on(a, b, patch).items():
        out[k + "_full"] = v
    return out


@torch.no_grad()
def radial_power_profile(f_hat, f_gt, s, n_bins=32, half_arcsec=4.0):
    """Azimuthally-averaged power of f_hat and f_gt in `n_bins` equal radial
    bins over u in [0,1] (u=1 is native Nyquist). For Step 10's Fig 4a.

    Returns (centers[n_bins], P_hat[n_bins], P_gt[n_bins]) as numpy, averaged
    over the batch.
    """
    peak = f_gt.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    a = apodized_center_crop(f_hat / peak, s, half_arcsec)
    b = apodized_center_crop(f_gt / peak, s, half_arcsec)
    P_hat = torch.fft.fft2(a.float()).abs() ** 2
    P_gt = torch.fft.fft2(b.float()).abs() ** 2
    H, W = a.shape[-2:]
    _, _, u = _freq_grids(H, W, a.device)
    edges = torch.linspace(0, 1, n_bins + 1, device=a.device)
    ph, pg = [], []
    for i in range(n_bins):
        sel = (u >= edges[i]) & (u < edges[i + 1])
        if sel.sum() == 0:
            ph.append(0.0); pg.append(0.0); continue
        ph.append(float(P_hat[:, 0][:, sel].mean()))
        pg.append(float(P_gt[:, 0][:, sel].mean()))
    centers = (0.5 * (edges[:-1] + edges[1:])).cpu().numpy()
    return centers, np.array(ph), np.array(pg)


def wiener_deconvolve(g, h, K):
    """Classical Wiener deconvolution with the KNOWN PSF -- the baseline a
    reviewer will run. [B,1,H,W] -> [B,1,H,W].

    `sim_lens.gaussian_psf_stamp` centers the stamp at `(npix-1)/2`, a
    HALF-INTEGER for every survey's even npix, so a plain `ifftshift` leaves a
    half-pixel offset -> a linear phase ramp -> a half-pixel shifted output.
    That is invisible to power-ratio metrics but destroys phase coherence and
    inflates masked L1, so we de-center analytically with the exact phase ramp
    instead of shifting.
    """
    g = g.float(); h = h.float()
    H, W = h.shape[-2:]
    fy, fx, _ = _freq_grids(H, W, h.device)
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    ramp = torch.exp(2j * torch.pi * (fy * cy + fx * cx))
    Hf = torch.fft.fft2(h) * ramp
    G = torch.fft.fft2(g)
    return torch.fft.ifft2(Hf.conj() * G / (Hf.abs() ** 2 + K)).real
