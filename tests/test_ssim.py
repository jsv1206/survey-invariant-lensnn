"""Regression tests for SSIM saturation (HANDOFF_run1.md Step 2 / CLAUDE.md bug 6).

`losses.ssim` used `C1=0.01**2`, `C2=0.03**2`, which assume data in [0,1], but
images peak at ~0.017 (HST) to ~0.19 (DES) -- so the constants swamped local
variances and the metric saturated near 1 regardless of how different the two
images actually were (measured pre-fix: ssim(blurred, sharp) = 0.999993 HST,
0.986732 DES -- barely discriminative, and `lambda_a=0.5` bought almost no
gradient). Fix: divide both inputs by a SHARED per-sample scalar (`b`'s peak,
since every call site passes (f_hat, f_gt)) before computing SSIM. `lambda_a`
itself is unchanged -- this is a normalization fix, not a reweighting.
"""
import numpy as np
import pytest
import torch

from survey_lens import env_setup  # noqa: F401  (must precede lenstronomy)
from survey_lens import losses as Lo
from survey_lens import sim_lens as S
from survey_lens import surveys as SV

FOV = 20.0


def test_ssim_identity_is_one():
    """ssim(x, x) == 1 must still hold after normalization (peak-dividing an
    image by itself cancels, so this is a no-op check on the normalization math)."""
    rng = np.random.default_rng(0)
    p = S.sample_params(rng)
    p.update(dict(scale=0.05, fwhm=0.10))
    o = S.simulate_one(p, fov=FOV)
    x = torch.from_numpy(o["f_gt"])[None, None].float()
    assert Lo.ssim(x, x).item() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", list(SV.SURVEYS))
def test_ssim_no_longer_saturated(name):
    """ssim(blurred, sharp) must be measurably below 1 for every survey --
    pre-fix, HST/Euclid/Roman were saturated at >0.99995 regardless of the
    real blur/resample difference. This is the test that would have caught
    the saturation: it fails against the pre-fix C1/C2 (data in [0,1]
    assumption) because those constants dominate the local variances of
    images that peak at ~0.02-0.2."""
    rng = np.random.default_rng(3)
    p = S.sample_params(rng)
    cfg = SV.SURVEYS[name]
    q = dict(p)
    q.update({k: cfg[k] for k in SV.RENDER_KEYS})
    o = S.simulate_one(q, fov=FOV)
    f_gt = torch.from_numpy(o["f_gt"])[None, None].float()
    g = torch.from_numpy(o["g"])[None, None].float()
    val = Lo.ssim(g, f_gt).item()
    assert val < 0.9999, f"{name}: ssim still saturated at {val:.6f}"


def test_ssim_orders_ground_worse_than_space():
    """Ground surveys (DES, Rubin: broader PSF, coarser pixels) must show
    lower blurred-vs-sharp SSIM than space surveys (HST, Euclid, Roman) --
    the ordering the pre-fix saturated metric could not resolve because it
    was pinned near 1.0 for every survey."""
    rng = np.random.default_rng(3)
    p = S.sample_params(rng)
    vals = {}
    for name, cfg in SV.SURVEYS.items():
        q = dict(p)
        q.update({k: cfg[k] for k in SV.RENDER_KEYS})
        o = S.simulate_one(q, fov=FOV)
        f_gt = torch.from_numpy(o["f_gt"])[None, None].float()
        g = torch.from_numpy(o["g"])[None, None].float()
        vals[name] = Lo.ssim(g, f_gt).item()
    assert vals["HST"] > vals["DES"], vals
    assert vals["HST"] > vals["Rubin"], vals
    assert vals["Euclid"] > vals["DES"], vals


def test_ssim_uses_shared_not_self_peak():
    """Normalizing must use a SHARED scalar (b's peak), not each image's own
    peak -- self-normalizing would hide amplitude error entirely. A pure
    amplitude error (f_hat = 3*f_gt, identical structure) must therefore NOT
    score ssim==1; if it did, the fix would have used the wrong (self) peak."""
    rng = np.random.default_rng(0)
    p = S.sample_params(rng)
    p.update(dict(scale=0.05, fwhm=0.10))
    o = S.simulate_one(p, fov=FOV)
    f_gt = torch.from_numpy(o["f_gt"])[None, None].float()
    val = Lo.ssim(3.0 * f_gt, f_gt).item()
    assert val < 0.99, f"amplitude-only error hidden by self-normalization: ssim={val:.6f}"
