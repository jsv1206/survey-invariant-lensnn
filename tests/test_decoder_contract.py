"""Contract tests for the reconstruction decoder.

Every assertion here corresponds to a defect that shipped undetected into Run 1
and was only found by measuring the trained checkpoint after the fact. Measured
on Run 1, full 200-lens Test Set 1, top-octave complex error (1.0 == emitting
nothing): HST 2.705, Euclid 1.354, Roman 0.949, Rubin 0.940, DES 0.937 --
`identity` beat the trained model in ALL FIVE surveys, and HST/Euclid were
worse than emitting nothing.
"""
import numpy as np
import pytest
import torch

from survey_lens import env_setup  # noqa: F401
from survey_lens.model import SurveyInvariantLensNet


def _inputs(B=2, n=32, seed=0):
    torch.manual_seed(seed)
    g = torch.rand(B, 1, n, n)                     # non-negative, like real data
    h = torch.rand(B, 1, n, n)
    h = h / h.sum(dim=(1, 2, 3), keepdim=True)     # sum-normalized PSF
    s = torch.tensor([0.1] * B)
    return g, h, s


def test_zero_init_residual_is_exact_identity():
    """At init the decoder must reproduce the input EXACTLY, so training starts
    at the `identity` baseline instead of having to rediscover it. Run 1 had no
    skip path at all and never found it."""
    net = SurveyInvariantLensNet(patch=4, dim=32, depth=2, n_heads=2).eval()
    g, h, s = _inputs()
    with torch.no_grad():
        recon = net(g, h, s)["recon"]
    assert torch.equal(recon, g), (
        "recon must equal g bit-exactly at init; max dev "
        f"{(recon - g).abs().max().item():.3e}")


def test_recon_is_non_negative():
    """f_gt >= 0 by construction, but the old head was a bare Linear with no
    activation and emitted ~40% negative pixels, dumping junk flux into empty
    sky (HST sum(f_hat)/sum(f_gt) = 1.906, which is why it lost to all-zeros
    on the global metrics)."""
    net = SurveyInvariantLensNet(patch=4, dim=32, depth=2, n_heads=2)
    # perturb away from the zero-init no-op so this tests the clamp, not init
    with torch.no_grad():
        for pmt in net.recon_head[-1].parameters():
            pmt.add_(torch.randn_like(pmt))
    g, h, s = _inputs()
    with torch.no_grad():
        recon = net(g, h, s)["recon"]
    assert (recon < 0).sum() == 0, f"{(recon < 0).float().mean():.1%} of pixels negative"


def test_recon_is_scale_equivariant():
    """Scaling the input must scale the output by the same factor. Run 1 was
    NOT equivariant: raw brightness went straight into patch_proj and a
    LayerNorm before the head divided out absolute amplitude, giving an 11x
    survey-dependent input SNR and peak ratios that fell monotonically with
    blur (0.97 HST -> 0.27 DES)."""
    net = SurveyInvariantLensNet(patch=4, dim=32, depth=2, n_heads=2).eval()
    with torch.no_grad():
        for pmt in net.recon_head[-1].parameters():
            pmt.add_(0.1 * torch.randn_like(pmt))
    g, h, s = _inputs()
    c = 7.3
    with torch.no_grad():
        a = net(g, h, s)["recon"]
        b = net(c * g, h, s)["recon"]
    rel = (b - c * a).abs().max() / (c * a).abs().max().clamp_min(1e-12)
    assert rel < 1e-4, f"not scale-equivariant: relative deviation {rel:.2e}"


def test_theta_E_is_scale_invariant():
    """theta_E is a geometric quantity in arcsec: it must not depend on how
    bright the image happens to be."""
    net = SurveyInvariantLensNet(patch=4, dim=32, depth=2, n_heads=2).eval()
    g, h, s = _inputs()
    with torch.no_grad():
        a = net(g, h, s)["theta_E"]
        b = net(5.0 * g, h, s)["theta_E"]
    assert torch.allclose(a, b, atol=1e-5), "theta_E must be invariant to input scaling"


def test_refinement_crosses_patch_seams():
    """The decoder emits disjoint p x p tiles, so seams are discontinuous. The
    refinement conv must actually see across them -- otherwise patch-grid energy
    lands at 0.25 cyc/px == exactly 0.5x Nyquist, inside the band that
    adjudicates the deconvolution claim."""
    net = SurveyInvariantLensNet(patch=4, dim=32, depth=2, n_heads=2)
    # a delta that is non-zero in exactly one tile must leak into its neighbours
    with torch.no_grad():
        for pmt in net.refine[-1].parameters():
            pmt.add_(0.5 * torch.randn_like(pmt))
        d = torch.zeros(1, 1, 16, 16)
        d[..., 4:8, 4:8] = 1.0
        out = net.refine(d)
    assert out[..., 3, 5].abs() > 0, "refinement does not cross the patch boundary"
