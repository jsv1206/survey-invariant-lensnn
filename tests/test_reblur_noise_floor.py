"""Regression tests for the Test Set 2 reblur noise-floor diagnostic.

On noisy input, a PERFECT reconstruction cannot drive reblur_mse to zero:
the residual `f_hat*h - g` still contains g's noise. The floor is
`mean(max(g,0)/gain + bkg_rms**2)`; report reblur_mse/floor since the raw
floor spans ~6x across surveys. Below 1.0 means the model is fitting noise,
not doing well.
"""
import numpy as np
import pytest
import torch

from survey_lens import env_setup  # noqa: F401  (must precede lenstronomy)
from survey_lens import data as D
from survey_lens import surveys as SV
from survey_lens.test import evaluate_survey

FOV = 8.0
BKG_RMS, GAIN = 0.002, 1.0e4


def _noisy_items(n_lenses=8, survey="DES"):
    ts = D.build_test_set_2({survey: SV.SURVEYS[survey]}, n_lenses=n_lenses,
                            seed=7, fov=FOV)
    return ts[survey]


def test_floor_matches_analytic_formula():
    """The reported reblur_floor_global must equal the mean of
    max(g,0)/gain + bkg_rms**2 computed directly from the same items."""
    items = _noisy_items(6)
    cfg = SV.SURVEYS["DES"]
    noise_cfg = dict(bkg_rms=cfg["bkg_rms"], gain=cfg["gain"])

    class _Identity:
        def __call__(self, g, h, s):
            return dict(recon=g, theta_E=torch.zeros(g.shape[0]))

    r = evaluate_survey(_Identity(), items, batch_size=8, device="cpu",
                        noise_cfg=noise_cfg)

    manual = []
    for it in items:
        g = it["g"]
        var = g.clamp_min(0.0) / cfg["gain"] + cfg["bkg_rms"] ** 2
        manual.append(var.mean().item())
    assert r["reblur_floor_global"] == pytest.approx(np.mean(manual), rel=1e-5)


def test_zeros_reconstruction_is_far_above_the_floor():
    """f_hat=0 is not a noise-fitting reconstruction -- its reblur residual
    (0*h - g = -g) is dominated by real signal, not just noise, so it must
    land WELL above the floor (large reblur_mse/floor), a structural sanity
    check that the ratio direction/magnitude makes sense."""
    items = _noisy_items(6)
    cfg = SV.SURVEYS["DES"]
    noise_cfg = dict(bkg_rms=cfg["bkg_rms"], gain=cfg["gain"])

    class _Zeros:
        def __call__(self, g, h, s):
            return dict(recon=torch.zeros_like(g), theta_E=torch.zeros(g.shape[0]))

    r = evaluate_survey(_Zeros(), items, batch_size=8, device="cpu",
                        noise_cfg=noise_cfg)
    assert r["reblur_mse_over_floor"] > 5.0, r["reblur_mse_over_floor"]


def test_floor_absent_without_noise_cfg():
    """Test Set 1 (no noise_cfg passed) must not report floor columns --
    they are only meaningful/computed for the noisy Test Set 2 path."""
    items = _noisy_items(3)

    class _Identity:
        def __call__(self, g, h, s):
            return dict(recon=g, theta_E=torch.zeros(g.shape[0]))

    r = evaluate_survey(_Identity(), items, batch_size=8, device="cpu")
    assert "reblur_floor_global" not in r
    assert "reblur_mse_over_floor" not in r


class _PerfectReconModel:
    """model(g,h,s) that returns the GROUND-TRUTH sharp image, i.e. the best any
    reconstruction can possibly do. Used to calibrate what `reblur_mse/floor`
    means: on noisy input a perfect f_hat still leaves g's noise in the reblur
    residual, so the ratio must land at ~1.0 by construction."""

    def __init__(self, items):
        self._f = {id(it["g"]): it["f_gt"] for it in items}

    def __call__(self, g, h, s):
        raise NotImplementedError  # replaced per-batch in the test below


@pytest.mark.parametrize("survey", ["DES", "Rubin"])
def test_perfect_reconstruction_scores_about_one(survey):
    """`reblur_mse_over_floor` must be ~1.0 for a PERFECT reconstruction.

    This is what makes the diagnostic interpretable at all: the floor is the
    residual an oracle cannot avoid, so an oracle scores 1.0, a model that
    fits the noise scores below 1.0, and a model inconsistent with the data
    scores above it.

    It is also a units regression. `losses.reblur_loss` is SCALE-FREE (it
    divides by mean(g**2)) while the floor `mean(max(g,0)/gain + bkg_rms**2)`
    is an absolute surface-brightness variance. Dividing the first by the
    second inflated the ratio by 1/mean(g**2) -- ~1e6 on this data, which is
    how a genuinely sub-floor 0.13x was reported as 126185x on Test Set 2.
    The ratio is now taken against `reblur_floor_rel`, the floor in the same
    scale-free units.
    """
    items = _noisy_items(n_lenses=6, survey=survey)

    class _Oracle:
        def __init__(self):
            self.i = 0

        def __call__(self, g, h, s):
            bs = g.shape[0]
            f = torch.stack([items[self.i + k]["f_gt"] for k in range(bs)]).to(g.device)
            self.i += bs
            return dict(recon=f, theta_E=torch.zeros(bs, device=g.device))

    cfg = SV.SURVEYS[survey]
    r = evaluate_survey(_Oracle(), items, batch_size=3, device="cpu",
                        noise_cfg=dict(bkg_rms=cfg["bkg_rms"], gain=cfg["gain"]))

    ratio = r["reblur_mse_over_floor"]
    assert 0.5 < ratio < 2.0, (
        f"{survey}: an oracle reconstruction should sit at the noise floor "
        f"(ratio ~1.0), got {ratio:.4g} -- the floor and reblur_mse are not "
        f"in the same units")
