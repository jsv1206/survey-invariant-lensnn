"""Regression tests for the trivial baselines (HANDOFF_run1.md Step 4).

A reviewer will ask what the trained numbers mean, so `test.py` reports the
same metrics for two null reconstruction models (zeros: f_hat=0; identity:
f_hat=g, i.e. no deconvolution at all) and one null regressor (prior_mean:
always predict mean(theta_E), whose RMSE equals the prior standard deviation
by construction). The deconvolution claim requires the trained model to beat
`identity` on the masked normalized metrics, not merely `zeros`.
"""
import numpy as np
import pytest

from survey_lens import env_setup  # noqa: F401  (must precede lenstronomy)
from survey_lens import data as D
from survey_lens import surveys as SV
from survey_lens.test import baseline_rows, _RECON_COLS, _THETA_COLS

FOV = 8.0


def _make_items(n_lenses=6, survey="DES"):
    ts = D.build_test_set_1({survey: SV.SURVEYS[survey]}, n_lenses=n_lenses,
                            seed=7, fov=FOV)
    return ts[survey]


def test_baseline_keys_present():
    items = _make_items(3)
    base = baseline_rows(items, batch_size=8, device="cpu")
    assert set(base) == {"zeros", "identity", "prior_mean"}
    for mode in ("zeros", "identity"):
        assert set(base[mode]) == set(_RECON_COLS)
    assert set(base["prior_mean"]) == set(_THETA_COLS)


def test_prior_mean_rmse_equals_population_std():
    """'Its RMSE equals the prior standard deviation' -- exact identity for
    a predictor that always outputs the sample mean."""
    items = _make_items(20)
    base = baseline_rows(items, batch_size=8, device="cpu")
    theta_true = np.array([float(it["theta_E"]) for it in items])
    assert base["prior_mean"]["theta_rmse"] == pytest.approx(theta_true.std(), rel=1e-6)
    assert base["prior_mean"]["theta_bias"] == pytest.approx(0.0, abs=1e-6)
    assert base["prior_mean"]["theta_scatter"] == pytest.approx(theta_true.std(), rel=1e-6)


def test_identity_beats_zeros_on_noise_free_data():
    """On Test Set 1 (no noise), doing nothing (identity = the blurred input)
    must reconstruct the arc far better than predicting all-zero -- if this
    flips, either the mask/normalization plumbing or the simulator's SNR is
    broken. This is also the specific comparison the deconvolution claim
    needs the TRAINED model to beat (not merely zeros)."""
    for survey in ("HST", "DES"):
        items = _make_items(6, survey=survey)
        base = baseline_rows(items, batch_size=8, device="cpu")
        assert base["identity"]["recon_l1_norm_masked"] < base["zeros"]["recon_l1_norm_masked"], survey
        assert base["identity"]["recon_ssim_norm"] > base["zeros"]["recon_ssim_norm"], survey


def test_zeros_masked_l1_exceeds_global_l1():
    """The zeros baseline scores well on the mostly-empty full frame (most
    pixels genuinely are ~0) but poorly restricted to the arc mask -- the
    dilution Step 3's mask_frac column exists to make visible. This is a
    structural sanity check on the masking, independent of any model."""
    items = _make_items(6)
    base = baseline_rows(items, batch_size=8, device="cpu")
    assert base["zeros"]["recon_l1_norm_masked"] > base["zeros"]["recon_l1_norm_global"]
