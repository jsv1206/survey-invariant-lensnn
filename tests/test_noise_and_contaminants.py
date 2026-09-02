"""Realistic-noise training and the unmodelled-astrophysics domain shift.

The first test here is the important one. It is not about noise at all -- it
pins that adding this machinery did not perturb the lens population that
every published Run 3 number was computed on.
"""
import numpy as np
import pytest
import torch

from survey_lens import env_setup  # noqa: F401
from survey_lens import losses as Lo
from survey_lens import sim_lens as S
from survey_lens import surveys as SV

# Captured from `sample_params` BEFORE the noise/contaminant work, seed 12345, first 4.
# `build_test_set_1` draws 200 lenses from ONE generator in a loop, so ANY extra
# draw inside sample_params -- even appended after the last existing one --
# shifts every sample from the second onward. Measured when this was attempted:
# all 14 lens fields moved. Every published Run 3 number would have silently
# stopped being reproducible from its seed, with no file changed.
_THETA_E_SEED12345 = [0.95467204493433933, 1.3836793323356256,
                      0.68332950308987184, 1.1617818609341093]
_SCALE_SEED12345 = [0.048827496407309857, 0.052357899800736074,
                    0.15945934799978675, 0.10005077467001249]


def test_sample_params_draw_sequence_is_unchanged():
    """The noise/contaminant draws must not consume from the lens-parameter generator.

    If this fails, noise or contaminant draws have leaked into `sample_params`
    and the Run 3 test sets no longer reproduce. Draw them from their own
    generator (`sample_noise`, `sample_contaminants`) instead.
    """
    rng = np.random.default_rng(12345)
    ps = [S.sample_params(rng) for _ in range(4)]
    for i, (want_t, want_s) in enumerate(zip(_THETA_E_SEED12345, _SCALE_SEED12345)):
        assert ps[i]["theta_E"] == pytest.approx(want_t, abs=1e-9), (
            f"lens draw {i} changed: Run 3's test sets no longer reproduce")
        assert ps[i]["scale"] == pytest.approx(want_s, abs=1e-9)


def test_noise_priors_leave_every_survey_interior():
    """Same rule that fixed dataset bug 2: a survey sitting ON a prior edge makes
    a prior-coverage artifact look like a generalization failure."""
    rng = np.random.default_rng(0)
    draws = [S.sample_noise(rng) for _ in range(4000)]
    for key, getter in (("bkg_rms", lambda c: c["bkg_rms"]),
                        ("gain", lambda c: c["gain"])):
        vals = np.array([getter(d) for d in draws])
        for nm, cfg in SV.SURVEYS.items():
            pct = 100.0 * (vals < cfg[key]).mean()
            assert 2.0 < pct < 98.0, (
                f"{nm} sits at the {pct:.1f}th percentile of the {key} prior "
                f"-- too close to an edge")


@pytest.mark.parametrize("corr_px", [0.5, 1.0, 1.5])
def test_correlated_noise_preserves_per_pixel_variance(corr_px):
    """The domain shift must isolate noise STRUCTURE from noise AMPLITUDE.

    Convolving unit-variance white noise with a sum-1 kernel reduces its
    variance to sum(k**2); `add_correlated_noise` divides by sqrt of that to
    restore it. Without the renormalization a correlation-length change would
    also change the depth, and any measured effect would confound the two.
    """
    bkg, gain = 1.5e-3, 1.0e4
    g = np.zeros((256, 256))                      # background-dominated
    n = S.add_correlated_noise(g, bkg, gain, np.random.default_rng(1), corr_px) - g
    assert n.std() == pytest.approx(bkg, rel=0.05), (
        "correlating the noise changed its amplitude")


def test_correlated_noise_actually_correlates():
    """Guard the other direction: renormalizing must not silently no-op."""
    g = np.zeros((256, 256))
    white = S.add_correlated_noise(g, 1.5e-3, 1e4, np.random.default_rng(2), 0.0)
    corr = S.add_correlated_noise(g, 1.5e-3, 1e4, np.random.default_rng(2), 1.5)
    a_white = float(np.mean(white[:, :-1] * white[:, 1:]) / white.var())
    a_corr = float(np.mean(corr[:, :-1] * corr[:, 1:]) / corr.var())
    assert abs(a_white) < 0.05, "corr_px=0 should be uncorrelated"
    assert a_corr > 0.5, f"corr_px=1.5 barely correlated (lag-1 = {a_corr:.3f})"


def test_chi2_reblur_scores_one_for_a_perfect_reconstruction():
    """Calibration by construction, mirroring tests/test_reblur_noise_floor.py.

    That file's oracle test is what caught the bug-12 units error (a ratio off by
    ~1e6). The same trick applies here: feed the TRUTH through the term on noisy
    input and the reduced chi^2 must be ~1. If it is not, the weighting is wrong
    and training on noise would be optimizing a mis-scaled objective.
    """
    cfg = SV.SURVEYS["Euclid"]
    rng = np.random.default_rng(0)
    p = S.sample_params(rng)
    p.update(scale=cfg["scale"], fwhm=cfg["fwhm"])
    o = S.simulate_one(p, fov=12.0)
    bkg, gain = cfg["bkg_rms"], cfg["gain"]

    g_noisy = S.add_noise(o["g"], bkg, gain, np.random.default_rng(1))
    var = np.maximum(g_noisy, 0.0) / gain + bkg ** 2

    f = torch.from_numpy(o["f_gt"][None, None]).float()
    h = torch.from_numpy(o["h"][None, None]).float()
    gn = torch.from_numpy(g_noisy[None, None]).float()
    v = torch.from_numpy(var[None, None]).float()

    chi2 = Lo.reblur_loss_chi2(f, h, gn, v).item()
    assert 0.7 < chi2 < 1.4, (
        f"reduced chi^2 for a PERFECT reconstruction is {chi2:.3f}, not ~1 -- "
        "the variance weighting is mis-scaled")


def test_contaminants_land_in_both_f_gt_and_g():
    """`model.py` computes recon = g + delta, a residual on the INPUT, so a
    contaminant present only in `g` would appear in the output by construction
    and scoring it against a clean `f_gt` would measure the deflector's
    brightness rather than anything about the reconstruction."""
    rng = np.random.default_rng(0)
    p = S.sample_params(rng)
    p.update(scale=0.10, fwhm=0.16)
    clean = S.simulate_one(p, fov=12.0)
    contam = S.sample_contaminants(p, np.random.default_rng(7))
    dirty = S.simulate_one(p, fov=12.0, contam=contam)

    d_gt = dirty["f_gt"] - clean["f_gt"]
    d_g = dirty["g"] - clean["g"]
    assert d_gt.sum() > 0 and d_g.sum() > 0, "contaminant missing from one of them"
    # same light, same flux, blurred by the same path
    assert d_g.sum() == pytest.approx(d_gt.sum(), rel=0.02), (
        "contaminant flux differs between f_gt and g -- it was not passed "
        "through convolve-then-resample identically")
    assert d_g.max() < d_gt.max(), "the contaminant in `g` should be BLURRED"


def test_contaminant_brightness_is_set_on_the_rendered_peak():
    """lenstronomy's Sersic `amp` is a surface-brightness normalization whose
    relation to the PEAK depends steeply on n_sersic and R_sersic. Setting the
    deflector's amp as a multiple of the source's gave a scene peak ~1e5x the
    arc; the ratio must be applied to the rendered peak instead."""
    rng = np.random.default_rng(0)
    p = S.sample_params(rng)
    p.update(scale=0.10, fwhm=0.16)
    clean = S.simulate_one(p, fov=12.0)
    for seed in (7, 11, 23):
        contam = S.sample_contaminants(p, np.random.default_rng(seed))
        dirty = S.simulate_one(p, fov=12.0, contam=contam)
        ratio = dirty["f_gt"].max() / clean["f_gt"].max()
        # Upper bound measured, not guessed: over 24 draws the realized
        # scene/arc peak ratio ran 1.00-12.28 (median 2.77), correlating 0.84
        # with the requested deflector peak_ratio. It scatters around the
        # request because R_sersic varies -- an extended deflector loses peak
        # to PSF blurring, a compact one keeps it, and a companion can land on
        # the peak. 15.0 leaves headroom above the observed tail.
        assert 0.8 < ratio < 15.0, (
            f"scene/arc peak ratio {ratio:.1f} is outside anything the "
            f"logU(1,10) deflector prior can produce")

def test_correlated_noise_returns_float32():
    """`_to_tensor` does not cast, so a float64 return reaches the model as a
    Double tensor and blows up in the first Linear. Caught only at evaluation
    time because the training path casts at its own call site."""
    g = np.zeros((16, 16), dtype=np.float32)
    out = S.add_correlated_noise(g, 1.5e-3, 1e4, np.random.default_rng(0), 1.0)
    assert out.dtype == np.float32, f"got {out.dtype}, must match add_noise"
    assert S.add_noise(g, 1.5e-3, 1e4, np.random.default_rng(0)).dtype == np.float32
