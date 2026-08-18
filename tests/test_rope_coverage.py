"""RoPE positional-encoding frequency-coverage guard.

The 2D-RoPE bank must place its informative frequencies on the spatial scales
that actually carry lensing signal -- arcs and the Einstein ring, theta_E in
~0.5-2.5" across a 20" field -- rather than on scales far larger than the
field (where the phase barely changes and the band is wasted).

The finest representable wavelength is ~2*pi*theta_ref (since inv_freq[0]=1),
and band k has wavelength lambda_k = 2*pi*theta_ref / base**(-k/n_freq). These
tests assert the default (theta_ref, rope_base) keep enough bands in-field, and
would have FAILED on the original (theta_ref=1.0, base=1e4) defaults that put
only 3/16 bands in range with a 6.3" finest wavelength.
"""
import numpy as np
import torch

from survey_lens import env_setup  # noqa: F401
from survey_lens.model import SurveyInvariantLensNet, build_2d_rope


def _band_wavelengths(head_dim, theta_ref, base):
    """Wavelengths (arcsec) of each RoPE frequency band for one axis."""
    d_half = head_dim // 2
    n_freq = d_half // 2
    k = np.arange(n_freq)
    inv_freq = base ** (-k / n_freq)          # radians per (arcsec/theta_ref)
    return 2 * np.pi * theta_ref / inv_freq   # arcsec


def test_default_finest_wavelength_is_subarcsec_scale():
    net = SurveyInvariantLensNet(dim=64, n_heads=4)          # head_dim = 16
    head_dim = net.dim // net.blocks[0].img_self.n_heads
    wl = _band_wavelengths(head_dim, net.theta_ref, net.rope_base)
    # finest wavelength must resolve the Einstein-ring scale (<= a few arcsec)
    assert wl.min() <= 3.0, f"finest RoPE wavelength {wl.min():.2f}\" too coarse"


def test_default_bands_land_in_field():
    net = SurveyInvariantLensNet(dim=64, n_heads=4)
    head_dim = net.dim // net.blocks[0].img_self.n_heads
    wl = _band_wavelengths(head_dim, net.theta_ref, net.rope_base)
    field = 20.0
    finest_patch = 4 * 0.05                                  # HST patch spacing, arcsec
    in_field = ((wl >= 2 * finest_patch) & (wl <= field)).sum()
    # want a healthy fraction of bands informative, not 3/16 as in the old default
    assert in_field >= wl.size // 2, (
        f"only {in_field}/{wl.size} RoPE bands in [{2*finest_patch:.1f}, {field}]\"")


def test_old_default_would_fail_guard():
    # sanity: the pre-fix tuning really was under-resolved (documents the fix)
    wl = _band_wavelengths(16, theta_ref=1.0, base=1.0e4)
    assert wl.min() > 5.0            # 6.28" finest -- too coarse
    in_field = ((wl >= 0.4) & (wl <= 20.0)).sum()
    assert in_field <= 4            # only 3/16 informative


def test_rope_positions_are_per_sample_not_batch_shared():
    """Regression: forward() used to compute RoPE positions from `s[0]` only
    and apply them to the WHOLE batch, on the (false) assumption that
    same-npix bucketing implies identical `s`. It does not -- different
    continuous-prior draws can round to the same npix (measured up to ~1.6%
    relative `s` spread within a real training-cache bucket) -- so every
    sample but the first silently got the wrong physical pixel scale baked
    into its positional encoding.

    Two batch items with IDENTICAL image/PSF content but DIFFERENT `s` must
    produce DIFFERENT post-attention image-token representations: under the
    old bug, sample 1's positions would incorrectly reuse sample 0's `s`,
    making the two (deterministic, no dropout) token trajectories identical
    despite the differing scale.
    """
    torch.manual_seed(0)
    net = SurveyInvariantLensNet(patch=4, dim=32, depth=2, n_heads=2)
    net.eval()
    g0 = torch.randn(1, 1, 32, 32)
    h0 = torch.rand(1, 1, 32, 32)
    g = g0.repeat(2, 1, 1, 1)
    h = h0.repeat(2, 1, 1, 1)
    s = torch.tensor([0.10, 0.16])            # same npix bucket, different scale
    with torch.no_grad():
        out = net(g, h, s)
    assert not torch.allclose(out["img_tokens"][0], out["img_tokens"][1]), (
        "identical image content with different `s` gave identical token "
        "representations -- positions are not being computed per-sample")


def test_rope_base_is_threaded_into_forward():
    # two models differing only in rope_base must produce different positional
    # rotations -> different outputs on the same input (proves the wiring).
    torch.manual_seed(0)
    g = torch.randn(1, 1, 40, 40)
    h = torch.randn(1, 1, 40, 40).abs()
    s = torch.tensor([0.20])
    a = SurveyInvariantLensNet(dim=64, depth=2, n_heads=4, rope_base=100.0)
    b = SurveyInvariantLensNet(dim=64, depth=2, n_heads=4, rope_base=1.0e4)
    b.load_state_dict(a.state_dict())        # identical weights, only base differs
    a.eval(); b.eval()
    with torch.no_grad():
        ta = a(g, h, s)["theta_E"]
        tb = b(g, h, s)["theta_E"]
    assert not torch.allclose(ta, tb), "rope_base has no effect -- not threaded"
