"""Regression tests for the trivial baselines.

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
    assert set(base) == {"zeros", "identity", "wiener", "wiener_hf", "prior_mean"}
    for mode in ("zeros", "identity"):
        assert set(base[mode]) == set(_RECON_COLS)
    # wiener additionally records which regularization K won the sweep,
    # which metric that K was selected on, and whether it pinned to a grid edge
    _wk = {"wiener_K", "wiener_K_selected_on", "wiener_K_at_grid_edge"}
    for row in ("wiener", "wiener_hf"):
        assert set(base[row]) == set(_RECON_COLS) | _wk
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


def test_wiener_is_tuned_on_the_metric_it_is_quoted_against():
    """A baseline quoted against a metric must have been SELECTED on that metric.

    The paper's deconvolution claim is `hf_complex_err` vs Wiener. That sweep
    used to pick K by masked L1 and then report the resulting `hf_complex_err`,
    i.e. Wiener was judged on a number it had never been tuned for -- which
    understates the baseline the whole claim rests on. Measured on Run 3, fixing
    this moved Wiener's top-octave error at Rubin from 0.315 to 0.057 and at DES
    from 0.328 to 0.078, flipping both surveys from "network wins ~4x" to
    "Wiener wins ~1.1x". A silent regression here would silently re-inflate the
    paper's headline, so pin it.
    """
    items = _make_items(3)
    base = baseline_rows(items, batch_size=8, device="cpu")

    assert base["wiener"]["wiener_K_selected_on"] == "recon_l1_norm_masked"
    assert base["wiener_hf"]["wiener_K_selected_on"] == "hf_complex_err"

    # each row must be at least as good as the other ON ITS OWN metric --
    # that is what "selected on" has to mean
    assert base["wiener_hf"]["hf_complex_err"] <= base["wiener"]["hf_complex_err"] + 1e-12
    assert (base["wiener"]["recon_l1_norm_masked"]
            <= base["wiener_hf"]["recon_l1_norm_masked"] + 1e-12)


def test_wiener_optimum_is_interior_to_the_sweep_grid():
    """An edge-pinned optimum means the baseline is under-tuned, and an
    under-tuned baseline flatters us. On noise-free data Wiener's optimum keeps
    sliding toward the naive inverse filter (K -> 0), so a grid that stops too
    early leaves real performance on the table; on noisy data it moves the
    other way and wants heavy regularization. The grid has to bracket both."""
    items = _make_items(3)
    base = baseline_rows(items, batch_size=8, device="cpu")
    for row in ("wiener", "wiener_hf"):
        assert not base[row]["wiener_K_at_grid_edge"], (
            f"{row}: best K={base[row]['wiener_K']:.0e} sits at a grid edge -- "
            f"widen the sweep in test.baseline_rows; the optimum may be outside it")
