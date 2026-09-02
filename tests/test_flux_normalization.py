"""Regression tests for dataset bug 9 (see the project notes / ).

Bug: `sim_lens._area_resample` used `.mean(axis=(1,3))` to collapse the fine
render grid to the native grid, but `lenstronomy`'s `ImageModel.image()`
returns flux PER PIXEL, not surface brightness -- so mean-pooling divided
brightness by `factor**2` instead of conserving it. Because
`choose_supersample_factor` picks `factor` adaptively from the pixel scale
`s`, this was not a uniform offset: it was a per-sample, scale-correlated
amplitude error that could let the network read `s` off overall brightness
instead of learning scale-invariant structure -- undermining the paper's
central claim. Fix: sum instead of average. Apply to `f_gt`/`g` only, never
to the PSF stamp `h` (which must stay `sum(h) == 1`).
"""

import numpy as np
import pytest

from survey_lens import env_setup  # noqa: F401  (must precede lenstronomy)
from survey_lens import sim_lens as S
from survey_lens import surveys as SV

# 61s of the suite's 114s -- five real lenstronomy renders at survey resolution.
# Bug 9 is fixed and sim_lens.py is stable, so these are gated behind --runslow
# rather than paid on every unrelated edit. See tests/conftest.py.
pytestmark = pytest.mark.slow

FOV = 20.0


def _arc_snr(f_gt, g_clean, bkg_rms, gain):
    """Analytic aperture SNR of the arc region under the Test-Set-2 noise model
    (var = max(signal,0)/gain + bkg_rms**2), evaluated on the clean signal so
    the result is deterministic (no noise realization needed)."""
    mask = f_gt > 0.01 * f_gt.max()
    var = np.maximum(g_clean, 0.0) / gain + bkg_rms ** 2
    return float(g_clean[mask].sum() / np.sqrt(var[mask].sum()))


def test_factor_invariance_sum_f_gt_is_flux_conserved():
    """The test that would have caught the bug: the SAME lens rendered at the
    SAME requested native scale but different supersampling factors must give
    the same total flux sum(f_gt) (HST-like s=0.05" via legacy explicit
    fine_scale, factors 4/6/8 -- the >= factor_min range choose_supersample_
    factor actually produces in production; factor=2 is never used live and
    measurably carries more light-profile discretization noise for a
    cuspy/compact source, which is a rendering-resolution effect, not a
    flux-conservation one).

    Tolerance is set from measurement, not from a nominal
    "<1e-3 relative": on a randomly-drawn compact high-Sersic-index source,
    factor in {4,6,8,10} gives a real spread of ~1.6e-3, not <1e-3 -- still >
    100x tighter than the pre-fix factor**2 error (a spread of ~4x for
    factors 2 vs 5), so it remains a strong regression guard."""
    rng = np.random.default_rng(3)
    p = S.sample_params(rng)
    p.update(dict(scale=0.05, fwhm=0.10))
    sums = []
    for factor in (4, 6, 8):
        fine_scale = 0.05 / factor
        o = S.simulate_one(p, fov=FOV, fine_scale=fine_scale)
        assert float(o["s"]) == pytest.approx(0.05, rel=1e-6)
        sums.append(float(o["f_gt"].sum()))
    sums = np.array(sums)
    rel_spread = np.ptp(sums) / sums.mean()
    assert rel_spread < 5e-3, f"flux not factor-invariant: {sums} (rel_spread={rel_spread:.2e})"


def test_cross_survey_flux_invariance():
    """Same lens through all five surveys (paired design) must give
    approximately equal sum(f_gt) -- total flux is a physical invariant of
    the lens, not of the observing grid. Per-pixel PEAK may legitimately
    differ (coarser pixels collect more flux each); total flux must not.

    Tolerance (3%) is set from measurement: surveys render the same compact
    source on very different native/fine grids (e.g. DES native 0.263" vs a
    ~0.1-0.35" source), so a few % of legitimate cross-grid discretization
    spread is expected (measured ~1.6%) -- this is >10x tighter than the
    pre-fix bug, which produced an ~11x brightness spread across surveys."""
    rng = np.random.default_rng(3)
    p = S.sample_params(rng)
    sums = {}
    for name, cfg in SV.SURVEYS.items():
        q = dict(p)
        q.update({k: cfg[k] for k in SV.RENDER_KEYS})
        o = S.simulate_one(q, fov=FOV)
        sums[name] = float(o["f_gt"].sum())
    vals = np.array(list(sums.values()))
    rel_spread = (vals.max() - vals.min()) / vals.mean()
    assert rel_spread < 0.03, f"cross-survey flux not invariant: {sums} (rel_spread={rel_spread:.2e})"


def test_psf_stays_unit_normalized_after_flux_fix():
    """The flux fix touches f_gt/g only. h must still sum to 1 for every
    survey -- verifies the fix was NOT accidentally applied to the PSF path."""
    rng = np.random.default_rng(3)
    p = S.sample_params(rng)
    for name, cfg in SV.SURVEYS.items():
        q = dict(p)
        q.update({k: cfg[k] for k in SV.RENDER_KEYS})
        o = S.simulate_one(q, fov=FOV)
        assert abs(float(o["h"].sum()) - 1.0) < 1e-6, f"{name}: h.sum() != 1"


def test_amplitude_no_longer_leaks_pixel_scale():
    """Before the fix, sum(f_gt) ~ 1/factor**2 and factor is chosen from `s`
    by choose_supersample_factor, so brightness correlated with pixel scale --
    a shortcut around scale-invariant structure. After the fix,
    log(sum(f_gt)) must show no such correlation with log(s) across the
    training prior (small FOV for speed; the bug's mechanism doesn't depend
    on FOV)."""
    rng = np.random.default_rng(42)
    n = 200
    log_sum, log_s = [], []
    for _ in range(n):
        p = S.sample_params(rng)
        o = S.simulate_one(p, fov=8.0)
        log_sum.append(np.log(o["f_gt"].sum()))
        log_s.append(np.log(float(o["s"])))
    corr = float(np.corrcoef(log_sum, log_s)[0, 1])
    assert abs(corr) < 0.3, f"amplitude leaks pixel scale: corr={corr:.3f}"


def test_snr_ordering_is_space_over_ground():
    """Bug 9 inverted the Test Set 2 SNR ordering (measured pre-fix: HST arc
    SNR 0.86, DES 5.00 -- backwards). After the fix, deep/low-background space
    surveys (HST, Roman) must have higher aperture SNR than shallow/
    high-background ground surveys (Rubin, DES), matching README's claim
    (HST arc-SNR ~11, Rubin ~2, DES ~1.6). Averaged over several paired lenses
    so no single random draw can flip the ordering by chance."""
    rng = np.random.default_rng(11)
    snrs = {name: [] for name in SV.SURVEYS}
    for _ in range(8):
        p = S.sample_params(rng)
        for name, cfg in SV.SURVEYS.items():
            q = dict(p)
            q.update({k: cfg[k] for k in SV.RENDER_KEYS})
            o = S.simulate_one(q, fov=FOV)
            snrs[name].append(_arc_snr(o["f_gt"], o["g"], cfg["bkg_rms"], cfg["gain"]))
    med = {name: float(np.median(v)) for name, v in snrs.items()}
    assert med["HST"] > med["DES"], med
    assert med["HST"] > med["Rubin"], med
    assert med["Roman"] > med["DES"], med
    assert med["Roman"] > med["Rubin"], med
