"""Regression tests for the reblur operator.

Two separate fixes are pinned here.

Step 1 (cost): `losses.reblur` used to convolve with the FULL-FOV PSF stamp
per-sample in a Python loop -- cost npix**4 for a kernel that is almost all
zeros. A single HST-scale (npix=400) call had to be SIGKILLed.

Bug 11 (centering): the replacement cropped-kernel conv2d was fast but WRONG.
`gaussian_psf_stamp` centers the stamp on the image coordinate origin, index
`(npix-1)/2`, which is a half-integer because npix is even for every survey
(400/200/182/100/76). A conv2d can only place zero lag on an integer index, so
every reblur was shifted by +0.5 px on both axes. `losses.reblur` is now an FFT
convolution with an explicit phase ramp, exact for even and odd npix alike.

`_reblur_full` inherits the same half-pixel offset and is therefore NOT a
correctness reference -- it is kept only to reproduce pre-fix runs and to time
the cost regression. The real reference is the SIMULATOR's own forward model:
simulate_one does convolve-then-resample, so `reblur(f_gt, h)` must reproduce
its `g`. That is what `test_reblur_matches_simulator_forward_model` checks.
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


# Measured Aug 2026: ||reblur(f_gt,h) - g||^2 / mean(g^2) against the
# simulator's own convolve-then-resample forward model, old spatial conv2d
# path vs the new FFT path. Roman is the outlier because its PSF sigma is
# 0.46 native px, so the stamp itself aliases (dataset bug 3) --
# that residual is a stamp-sampling limit, not a centering error.
#
#   survey   old conv    new fft    improvement
#   HST      3.18e-02    9.81e-06      3242x
#   Euclid   7.62e-02    1.61e-04       474x
#   Roman    1.04e-01    1.83e-03        57x
#   Rubin    4.75e-02    3.06e-05      1554x
#   DES      5.13e-02    9.31e-05       551x
FORWARD_MODEL_TOL = {
    "HST": 1e-4, "Euclid": 1e-3, "Roman": 1e-2, "Rubin": 1e-4, "DES": 1e-3,
}


@pytest.mark.parametrize("fov", [20.0, 12.0])
@pytest.mark.parametrize("name", list(SV.SURVEYS))
def test_reblur_matches_simulator_forward_model(name, fov):
    """`reblur(f_gt, h)` must reproduce the simulator's own blurred image `g`.

    This is the property the reblur consistency loss actually depends on: the
    term is ||f_hat * h - g||^2, so if the operator is mis-centered the loss
    penalises a perfect reconstruction and rewards a shifted one. The old
    cropped-conv path sat at ~5e-2 here -- a shift detector, not a data
    consistency term. Tolerances are ~10x the measured values above.

    Parameterized over FOV because `reblur` is an FFT convolution with NO
    zero-padding, i.e. a CIRCULAR one, and its correctness argument is
    explicitly conditional on the frame being mostly empty sky (`losses.py`).
    Shrinking the field eats that margin and lets the arc wrap around the
    boundary. Measured: at 12" the residual is essentially unchanged from 20",
    but at 5" it reaches 1.16e-1 -- worse than the half-pixel-shifted operator
    of bug 11 -- which is why the field was not shrunk further. This test is
    the guard on that, so keep both FOVs in it.
    """
    cfg = SV.SURVEYS[name]
    rng = np.random.default_rng(0)
    p = S.sample_params(rng)
    p.update(scale=cfg["scale"], fwhm=cfg["fwhm"])
    o = S.simulate_one(p, fov=fov)

    f_gt = torch.from_numpy(o["f_gt"][None, None]).float()
    g = torch.from_numpy(o["g"][None, None]).float()
    h = torch.from_numpy(o["h"][None, None]).float()

    rel = (((Lo.reblur(f_gt, h) - g) ** 2).mean()
           / (g ** 2).mean().clamp_min(1e-30)).item()
    assert rel < FORWARD_MODEL_TOL[name], (
        f"{name} @ fov={fov}\": reblur does not reproduce the simulator forward "
        f"model, rel={rel:.2e} (tol {FORWARD_MODEL_TOL[name]:.0e})")


@pytest.mark.parametrize("name", list(SV.SURVEYS))
def test_reblur_is_not_half_pixel_shifted(name):
    """Direct guard on bug 11: a centered delta must reblur to a stamp whose
    centroid lands on the image origin (npix-1)/2, not half a pixel past it.

    A delta at the coordinate origin convolved with a PSF centered on the same
    origin must return the PSF itself, in place. The old spatial path returned
    it shifted by +0.5 px on each axis.
    """
    cfg = SV.SURVEYS[name]
    npix = SMALL_NPIX
    h = torch.from_numpy(
        S.gaussian_psf_stamp(cfg["fwhm"], cfg["scale"], npix)[None, None]
    ).float()
    delta = torch.zeros(1, 1, npix, npix)
    c = (npix - 1) / 2.0
    # npix is even here, so the origin falls between pixels: split the delta
    # over the four neighbouring pixels to place it exactly on the origin.
    lo, hi = int(np.floor(c)), int(np.ceil(c))
    for iy in (lo, hi):
        for ix in (lo, hi):
            delta[0, 0, iy, ix] = 0.25

    out = Lo.reblur(delta, h)[0, 0]
    yy, xx = torch.meshgrid(torch.arange(npix).float(),
                            torch.arange(npix).float(), indexing="ij")
    tot = out.sum().clamp_min(1e-12)
    cy = (out * yy).sum() / tot
    cx = (out * xx).sum() / tot
    assert abs(cy.item() - c) < 0.02 and abs(cx.item() - c) < 0.02, (
        f"{name}: reblur centroid ({cy:.3f}, {cx:.3f}) != origin {c:.3f} "
        f"-- operator is shifted")


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
    path too; measure the actual speedup rather than assuming the
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
