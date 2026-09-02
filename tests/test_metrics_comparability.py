"""Regression tests for cross-survey metric comparability.

Two measured facts made the old `test.py` columns unusable for a cross-survey
table: (1) peak per-pixel brightness varies ~11x across surveys, so raw
`recon_l1` compares different physical quantities and makes coarse-pixel
surveys look worse by construction; (2) images are ~96-97% empty sky (only
2.7-4.0% of pixels exceed 1% of peak), so an unmasked mean is mostly a
trivial background-agreement number. `evaluate_survey` now also reports
peak-normalized, masked/global metrics: `recon_l1_norm_masked`,
`recon_l1_norm_global`, `recon_ssim_norm` (masked), and `mask_frac`.
"""
import numpy as np
import pytest
import torch

from survey_lens import env_setup  # noqa: F401  (must precede lenstronomy)
from survey_lens import data as D
from survey_lens import surveys as SV
from survey_lens.test import evaluate_survey

FOV = 8.0


class _PerfectModel:
    """Stands in for a trained checkpoint: returns the ground truth exactly,
    so the metrics computed against it are a known-answer check rather than
    depending on a real/random network."""

    def __call__(self, g, h, s):
        return dict(recon=g.new_zeros(g.shape), theta_E=s.new_zeros(s.shape[0]))


def _make_items(n_lenses=4):
    ts = D.build_test_set_1({"DES": SV.SURVEYS["DES"]}, n_lenses=n_lenses,
                            seed=7, fov=FOV)
    return ts["DES"]


def test_expected_metric_keys_present():
    items = _make_items(2)

    class _Identity:
        def __call__(self, g, h, s):
            return dict(recon=g, theta_E=torch.zeros(g.shape[0]))

    r = evaluate_survey(_Identity(), items, batch_size=8, device="cpu")
    for k in ("recon_l1", "recon_ssim", "recon_l1_norm_masked",
              "recon_l1_norm_global", "recon_ssim_norm", "mask_frac", "reblur_mse"):
        assert k in r, f"missing metric column: {k}"


def test_perfect_reconstruction_scores_near_zero_and_one():
    """f_hat == f_gt exactly must give ~0 masked/global normalized L1 and
    ~1 masked normalized SSIM -- the known-answer sanity check."""
    items = _make_items(4)

    class _Perfect:
        def __call__(self, g, h, s):
            f = torch.stack([it["f_gt"] for it in items[:g.shape[0]]])
            return dict(recon=f, theta_E=torch.zeros(g.shape[0]))

    r = evaluate_survey(_Perfect(), items, batch_size=8, device="cpu")
    assert r["recon_l1_norm_masked"] < 1e-5
    assert r["recon_l1_norm_global"] < 1e-5
    assert r["recon_ssim_norm"] == pytest.approx(1.0, abs=1e-4)


def test_mask_frac_matches_known_sparsity():
    """a known dataset issue: real lens images at FOV=20" are 2.7-4.0% signal above
    1% of peak (DES ~4.0%). mask_frac (same 1% threshold) must land in that
    ballpark at the same FOV, not e.g. near 0 (empty mask, metrics undefined)
    or near 1 (mask not actually restricting anything). Uses FOV=20 (not this
    file's usual FOV=8) specifically to match the conditions the project notes's
    figure was measured under -- a smaller FOV shrinks the empty-sky margin
    and inflates the fraction, which is not the regression this guards."""
    ts = D.build_test_set_1({"DES": SV.SURVEYS["DES"]}, n_lenses=6, seed=7, fov=20.0)
    items = ts["DES"]

    class _Identity:
        def __call__(self, g, h, s):
            return dict(recon=g, theta_E=torch.zeros(g.shape[0]))

    r = evaluate_survey(_Identity(), items, batch_size=8, device="cpu")
    assert 0.01 < r["mask_frac"] < 0.10, r["mask_frac"]


def test_masked_metric_ignores_error_outside_mask():
    """A reconstruction error placed ONLY in empty background (outside the
    arc mask) must move the global metric but leave the masked metric
    essentially unchanged -- the whole point of masking. Uses one real
    simulated item so the mask geometry is realistic."""
    items = _make_items(1)
    it = items[0]
    f = it["f_gt"][None]                                  # [1,1,H,W]
    peak = f.amax()
    mask = (f / peak) > 0.01
    err = torch.where(mask, torch.zeros_like(f), torch.full_like(f, 0.2 * peak))
    f_hat_outside_err = f + err

    class _OutsideErr:
        def __call__(self, g, h, s):
            return dict(recon=f_hat_outside_err, theta_E=torch.zeros(1))

    r = evaluate_survey(_OutsideErr(), [it], batch_size=8, device="cpu")
    assert r["recon_l1_norm_masked"] < 1e-5
    assert r["recon_l1_norm_global"] > 1e-3


def test_normalized_metrics_are_scale_invariant_but_raw_is_not():
    """Bug 7: raw L1 is not cross-survey comparable because peak brightness
    varies ~11x with pixel scale. Simulate that by comparing the SAME 10%
    relative reconstruction error applied at two different peak-brightness
    scalings of one image (a stand-in for two surveys' differing amplitude
    conventions). Peak-normalizing must make the metric agree; the raw
    (unnormalized) L1 must scale with the brightness instead."""
    items = _make_items(1)
    f0 = items[0]["f_gt"][None]                                    # [1,1,H,W]

    def _rel_error_item(brightness):
        f = brightness * f0[0]                                     # [1,H,W]
        return dict(items[0], f_gt=f)

    class _TenPercentHigh:
        """Returns 1.1x the TRUE f_gt for whatever item(s) it's given --
        a fixed 10% relative reconstruction error, independent of brightness."""
        def __init__(self, item):
            self.f = item["f_gt"][None]

        def __call__(self, g, h, s):
            return dict(recon=1.1 * self.f, theta_E=torch.zeros(g.shape[0]))

    dim_item, bright_item = _rel_error_item(1.0), _rel_error_item(5.0)
    r_dim = evaluate_survey(_TenPercentHigh(dim_item), [dim_item],
                            batch_size=8, device="cpu")
    r_bright = evaluate_survey(_TenPercentHigh(bright_item), [bright_item],
                               batch_size=8, device="cpu")
    assert r_bright["recon_l1"] == pytest.approx(5.0 * r_dim["recon_l1"], rel=0.05)
    assert r_bright["recon_l1_norm_global"] == pytest.approx(
        r_dim["recon_l1_norm_global"], rel=0.05)
