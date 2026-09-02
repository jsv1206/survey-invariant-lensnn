"""Regression tests for dataset bugs 1 and 2.

Bug 1: a single global `fine_scale` quantized the requested pixel scale to a
       multiple of itself, collapsing Roman (0.11") onto Euclid's 0.10" grid and
       shifting DES 0.263" -> 0.275". The adaptive fine grid must render every
       survey's scale EXACTLY, giving 5 distinct grids.
Bug 2: HST and DES sat on the endpoints of the training priors, so the two
       extreme surveys had one-sided support. Every survey must now be interior.
"""
import numpy as np
import pytest

from survey_lens import env_setup  # noqa: F401  (must precede lenstronomy)
from survey_lens import sim_lens as S
from survey_lens import surveys as SV

FOV = 20.0


@pytest.mark.parametrize("name", list(SV.SURVEYS))
def test_adaptive_grid_renders_requested_scale_exactly(name):
    """Realized pixel scale == requested scale (no quantization)."""
    cfg = SV.SURVEYS[name]
    p = S.sample_params(np.random.default_rng(3))
    p.update({k: cfg[k] for k in SV.RENDER_KEYS})
    o = S.simulate_one(p, fov=FOV)                 # fine_scale=None -> adaptive
    assert float(o["s"]) == pytest.approx(cfg["scale"], rel=1e-6), (
        f"{name}: requested {cfg['scale']} but realized {float(o['s'])}")


def test_five_surveys_give_five_distinct_grids():
    """The degenerate-grid bug: Roman must not collapse onto Euclid."""
    p = S.sample_params(np.random.default_rng(3))
    scales = set()
    for cfg in SV.SURVEYS.values():
        q = dict(p); q.update({k: cfg[k] for k in SV.RENDER_KEYS})
        scales.add(round(float(S.simulate_one(q, fov=FOV)["s"]), 6))
    assert len(scales) == len(SV.SURVEYS), f"degenerate grids: {sorted(scales)}"


def test_legacy_global_fine_scale_still_quantizes():
    """Guard the documented legacy behaviour (explicit float => old semantics)."""
    p = S.sample_params(np.random.default_rng(3))
    p.update(dict(scale=0.11, fwhm=0.12))          # Roman
    o = S.simulate_one(p, fov=FOV, fine_scale=0.025)
    assert float(o["s"]) == pytest.approx(0.10, rel=1e-6)


def test_psf_stamp_is_unit_normalized():
    """Reblur loss is only self-consistent if the PSF sums to 1."""
    p = S.sample_params(np.random.default_rng(7))
    o = S.simulate_one(p, fov=10.0)
    assert float(o["h"].sum()) == pytest.approx(1.0, abs=1e-5)


def test_all_surveys_interior_to_training_priors():
    """Bug 2: no survey may sit on a prior endpoint."""
    rng = np.random.default_rng(0)
    draws = [S.sample_params(rng) for _ in range(4000)]
    fwhm = np.array([d["fwhm"] for d in draws])
    scale = np.array([d["scale"] for d in draws])
    for name, cfg in SV.SURVEYS.items():
        q_f = float((fwhm < cfg["fwhm"]).mean())
        q_s = float((scale < cfg["scale"]).mean())
        assert 0.01 < q_f < 0.99, f"{name}: FWHM {cfg['fwhm']} at prior edge (pctile {q_f:.3f})"
        assert 0.01 < q_s < 0.99, f"{name}: scale {cfg['scale']} at prior edge (pctile {q_s:.3f})"


def test_fixed_theta_E_gives_survey_invariant_ring_radius():
    """Physics: the same lens must subtend the same angle in every survey."""
    p = S.sample_params(np.random.default_rng(3))
    p["theta_E"] = 1.5
    radii = []
    for cfg in SV.SURVEYS.values():
        q = dict(p); q.update({k: cfg[k] for k in SV.RENDER_KEYS})
        o = S.simulate_one(q, fov=FOV)
        f = o["f_gt"]; n = f.shape[0]; c = (n - 1) / 2.0
        yy, xx = np.mgrid[0:n, 0:n]
        r = np.hypot(xx - c, yy - c) * float(o["s"])
        w = np.clip(f, 0, None)
        radii.append(float((r * w).sum() / w.sum()))
    assert np.ptp(radii) < 0.02, f"ring radius varies across surveys: {radii}"
