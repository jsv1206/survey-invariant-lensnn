"""Regression tests for the reblur cost fix (HANDOFF_run1.md Step 1).

`losses.reblur` used to convolve with the FULL-FOV PSF stamp per-sample in a
Python loop: cost is npix**4, and the stamp is almost entirely zero far from
center for a narrow Gaussian PSF. A single HST-scale (npix=400) call did not
finish on CPU and had to be SIGKILLed. The fix crops each PSF to its
effective support (from its own second moment) before one batched conv2d.
These tests verify the crop preserves the loss value to <1e-6 relative
against the (still-available, but now unused in the hot path) full reference
implementation, and that the fix actually removes the cost blowup.
"""
import time

import numpy as np
import pytest
import torch

from survey_lens import env_setup  # noqa: F401
from survey_lens import losses as Lo
from survey_lens import sim_lens as S
from survey_lens import surveys as SV

FOV = 20.0


SMALL_NPIX = 48  # >> 2*4*sigma_px+1 for every survey (max sigma_px ~1.49 px, Rubin)


@pytest.mark.parametrize("name", list(SV.SURVEYS))
def test_cropped_reblur_matches_full_reference(name):
    """Cropped-kernel reblur must agree with the full-FOV reference to
    <1e-6 relative, for every survey's PSF width (sigma_px = fwhm/2.3548/scale).

    Deliberately does NOT use each survey's real FOV npix (400 for HST) --
    `_reblur_full` is O(npix**4) and a same-padded conv2d with an npix x npix
    kernel forces PyTorch's CPU conv into an im2col fallback whose intermediate
    is O(npix**4) elements. At HST scale that's ~100 GB and SIGKILLs the
    process (see HANDOFF_run1.md Step 1) -- it doesn't just run slow, it can
    take the whole machine down. `gaussian_psf_stamp(fwhm, scale, npix)` sets
    sigma_px from fwhm/scale alone, so a small synthetic npix reproduces each
    survey's exact PSF width -- and hence the exact crop window -- at safe cost.
    """
    cfg = SV.SURVEYS[name]
    npix = SMALL_NPIX
    torch.manual_seed(0)
    h = torch.from_numpy(
        S.gaussian_psf_stamp(cfg["fwhm"], cfg["scale"], npix)[None, None]
    ).float().repeat(3, 1, 1, 1)                     # batch of 3, same PSF
    f_hat = torch.rand(3, 1, npix, npix)

    g_full = Lo._reblur_full(f_hat, h)
    g_crop = Lo.reblur(f_hat, h)

    rel = ((g_crop - g_full).abs().max() / g_full.abs().max().clamp_min(1e-12)).item()
    assert rel < 1e-6, f"{name}: cropped reblur diverges from full reference, rel={rel:.2e}"


def test_hst_scale_reblur_loss_completes_under_a_second():
    """At HST scale (npix=400) the pre-fix full-FOV loop did not finish on
    CPU. The cropped/batched reblur_loss must complete comfortably fast."""
    cfg = SV.SURVEYS["HST"]
    npix = int(round(FOV / cfg["scale"]))
    torch.manual_seed(0)
    h = torch.from_numpy(
        S.gaussian_psf_stamp(cfg["fwhm"], cfg["scale"], npix)[None, None]
    ).float().repeat(8, 1, 1, 1)
    f_hat = torch.rand(8, 1, npix, npix, requires_grad=True)
    g = torch.rand(8, 1, npix, npix)

    t0 = time.perf_counter()
    loss = Lo.reblur_loss(f_hat, h, g)
    loss.backward()
    dt = time.perf_counter() - t0

    assert torch.isfinite(loss)
    assert dt < 1.0, f"HST-scale reblur_loss+backward took {dt:.3f}s (must be <1s)"


def test_measured_speedup_at_des_scale():
    """DES (npix=76) is small enough to run the O(npix**4) full reference
    path too; measure the actual speedup rather than assuming HANDOFF's
    quoted number (34x)."""
    cfg = SV.SURVEYS["DES"]
    npix = int(round(FOV / cfg["scale"]))
    torch.manual_seed(0)
    h = torch.from_numpy(
        S.gaussian_psf_stamp(cfg["fwhm"], cfg["scale"], npix)[None, None]
    ).float().repeat(8, 1, 1, 1)
    f_hat = torch.rand(8, 1, npix, npix)

    # Warm up both paths once: PyTorch's CPU conv2d pays a one-time
    # thread-pool/MKL-DNN setup cost on the first call, which at DES's small
    # 76x76 size is comparable to the whole timed loop and swamps the
    # algorithmic difference the test is trying to measure.
    Lo._reblur_full(f_hat, h)
    Lo.reblur(f_hat, h)

    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        Lo._reblur_full(f_hat, h)
    t_full = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        Lo.reblur(f_hat, h)
    t_crop = (time.perf_counter() - t0) / reps

    speedup = t_full / max(t_crop, 1e-9)
    print(f"\nDES measured speedup: {speedup:.1f}x (full={t_full * 1e3:.2f}ms, crop={t_crop * 1e3:.2f}ms)")
    assert speedup > 5, f"expected a meaningful speedup at DES scale, got {speedup:.1f}x"


def test_crop_stays_flux_normalized():
    """The cropped/renormalized kernel must still sum to 1 per sample."""
    cfg = SV.SURVEYS["Euclid"]
    npix = int(round(FOV / cfg["scale"]))
    h = torch.from_numpy(
        S.gaussian_psf_stamp(cfg["fwhm"], cfg["scale"], npix)[None, None]
    ).float().repeat(4, 1, 1, 1)
    hc = Lo._crop_and_renormalize(h)
    sums = hc.sum(dim=(1, 2, 3))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
