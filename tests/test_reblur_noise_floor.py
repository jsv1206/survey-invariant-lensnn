"""Regression tests for the Test Set 2 reblur noise-floor diagnostic
(HANDOFF_run1.md Step 9b).

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
