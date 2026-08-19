"""Known-answer tests for the top-octave reconstruction metrics
(HANDOFF_run1.md Step 10, implemented in survey_lens/metrics.py).

The metric this file guards is the one the project docs designate as THE
reconstruction metric ("Deconvolution means beating `identity` in the top
octave specifically"), and which did not exist when Run 1's reconstruction
verdict was written. The phase-sensitivity test below is the load-bearing one:
a power-ratio-only metric is gameable, and the Run-1 checkpoint games it
(Euclid scores a healthy-looking 0.86 power ratio while being worse than
emitting nothing in phase-aware terms).
"""
import numpy as np
import pytest
import torch

from survey_lens import env_setup  # noqa: F401  (must precede lenstronomy)
from survey_lens import metrics as M
from survey_lens import sim_lens as S
from survey_lens import surveys as SV

N = 64


def _mode(n, fx, fy, phase=0.0):
    """A single real Fourier mode at exact bin-center frequencies (cyc/px)."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    a = np.cos(2 * np.pi * (fx * x + fy * y) + phase)
    return torch.from_numpy(a).float()[None, None]


def test_perfect_reconstruction_is_known_answer():
    """f_hat == f_gt -> ratio 1, complex error 0, coherence 1."""
    f = _mode(N, 0.3, 0.25)
    r = M._metrics_on(f, f, patch=4)
    assert float(r["hf_power_ratio"]) == pytest.approx(1.0, abs=1e-5)
    assert float(r["hf_complex_err"]) == pytest.approx(0.0, abs=1e-6)
    assert float(r["hf_coherence"]) == pytest.approx(1.0, abs=1e-5)


def test_zeros_scores_exactly_one_complex_error():
    """f_hat = 0 must score complex_err EXACTLY 1.0. This is what makes the
    metric self-calibrating: any value above 1.0 means the reconstruction is
    worse than emitting nothing in the top octave."""
    f = _mode(N, 0.3, 0.25)
    r = M._metrics_on(torch.zeros_like(f), f, patch=4)
    assert float(r["hf_complex_err"]) == pytest.approx(1.0, abs=1e-6)
    assert float(r["hf_power_ratio"]) == pytest.approx(0.0, abs=1e-9)


def test_power_ratio_is_phase_blind_but_complex_error_is_not():
    """THE decisive test. Two DIFFERENT top-octave modes of equal power give a
    power ratio of ~1.0 (metric fooled: "100% of the high-frequency power
    recovered!") while coherence ~0 and complex error ~2.0 correctly report
    that none of it is in the right place. This is exactly the Run-1 Euclid
    failure mode, and a ratio-only metric cannot see it."""
    f_gt = _mode(N, 0.3, 0.25)
    f_hat = _mode(N, 0.25, -0.3)          # orthogonal mode, same power
    r = M._metrics_on(f_hat, f_gt, patch=4)
    assert float(r["hf_power_ratio"]) == pytest.approx(1.0, abs=1e-3), "setup: powers must match"
    assert float(r["hf_coherence"]) < 0.05, "coherence must expose the phase/placement error"
    # orthogonal + equal power => |A-B|^2 = |A|^2 + |B|^2 => complex_err = 2
    assert float(r["hf_complex_err"]) == pytest.approx(2.0, rel=1e-3)
    assert float(r["hf_complex_err"]) > 1.0, "must register as worse than emitting nothing"


def test_metrics_invariant_to_joint_rescale():
    """A shared per-sample scalar cancels in every quantity reported here."""
    f_gt = _mode(N, 0.3, 0.25)
    f_hat = 0.7 * f_gt + 0.1 * _mode(N, 0.25, -0.3)
    a = M._metrics_on(f_hat, f_gt, patch=4)
    b = M._metrics_on(5.0 * f_hat, 5.0 * f_gt, patch=4)
    for k in ("hf_power_ratio", "hf_complex_err", "hf_coherence"):
        assert float(a[k]) == pytest.approx(float(b[k]), rel=1e-5), k


def test_patch_harmonic_mask_catches_checkerboard_at_non_multiple_npix():
    """A period-4 checkerboard is pure patch-grid artifact; the mask must catch
    it. Tested at npix=182 (Roman) -- NOT a multiple of patch=4 -- because the
    0.25 cyc/px harmonic then lands at bin index 45.5 and straddles two bins.
    An integer-index tolerance would silently miss it here."""
    n = 182
    checker = _mode(n, 0.25, 0.0) * _mode(n, 0.0, 0.25)
    smooth = _mode(n, 0.30, 0.27)
    r = M._metrics_on(checker, smooth, patch=4)
    assert float(r["hf_harm_frac_hat"]) > 0.9, (
        "period-4 checkerboard should be almost entirely on the patch harmonics, "
        f"got {float(r['hf_harm_frac_hat']):.3f}")


def test_wiener_recovers_truth_with_no_subpixel_shift():
    """Wiener with the known PSF at small K must return f_gt on noiseless data,
    AND with zero sub-pixel shift. gaussian_psf_stamp centers at (npix-1)/2 --
    a half-integer for even npix -- so a naive ifftshift would introduce a
    half-pixel offset that power-only metrics cannot see."""
    rng = np.random.default_rng(0)
    p = S.sample_params(rng)
    p.update({k: SV.SURVEYS["Rubin"][k] for k in SV.RENDER_KEYS})
    o = S.simulate_one(p, fov=8.0)
    g = torch.from_numpy(o["g"]).float()[None, None]
    h = torch.from_numpy(o["h"]).float()[None, None]
    f_gt = torch.from_numpy(o["f_gt"]).float()[None, None]

    rec = M.wiener_deconvolve(g, h, K=1e-6)[0, 0].numpy()
    truth = f_gt[0, 0].numpy()

    # zero-lag cross-correlation => no sub-pixel shift
    Fa = np.fft.fft2(rec - rec.mean())
    Fb = np.fft.fft2(truth - truth.mean())
    xc = np.real(np.fft.ifft2(Fa * np.conj(Fb)))
    peak = np.unravel_index(np.argmax(xc), xc.shape)
    assert peak == (0, 0), f"Wiener output is shifted by {peak} px relative to truth"

    # Wiener must be a large improvement over doing nothing. It cannot invert
    # exactly even at K->0: `h` is a Gaussian sampled on the NATIVE grid, but
    # `g` was produced by convolve-at-fine-scale-then-area-resample, so `h` is
    # not exactly the operator that generated `g` (CLAUDE.md bug 3, still open
    # -- Euclid/Roman PSF stamps are under-sampled). Assert the meaningful
    # property -- a large gain over identity -- not exact recovery.
    pk = truth.max(); m = truth > 0.01 * pk
    err_w = np.abs(rec[m] - truth[m]).mean() / pk
    err_i = np.abs(g[0, 0].numpy()[m] - truth[m]).mean() / pk
    assert err_w < 0.5 * err_i, (
        f"Wiener should substantially beat identity on noiseless data: "
        f"wiener={err_w:.4f} vs identity={err_i:.4f}")


def test_radial_profile_shape_and_monotone_bins():
    f = _mode(N, 0.3, 0.25)
    s = torch.tensor([0.2])
    centers, ph, pg = M.radial_power_profile(f, f, s, n_bins=16)
    assert centers.shape == ph.shape == pg.shape == (16,)
    assert np.all(np.diff(centers) > 0)
    assert np.allclose(ph, pg)
