"""Fast CPU smoke tests: simulator physics + a forward/backward pass.

These run in CI without a GPU and without downloading anything. They check
that the package is wired correctly, not that the model is trained.
"""
import numpy as np
import torch

# env_setup must be imported before lenstronomy/astropy touch the filesystem.
from survey_lens import env_setup  # noqa: F401
from survey_lens import sim_lens as S
from survey_lens import surveys as SV
from survey_lens.model import SurveyInvariantLensNet


def test_surveys_table():
    # the five reference surveys are present with sane scales/FWHM
    assert {"HST", "Euclid", "Roman", "Rubin", "DES"} <= set(SV.SURVEYS)
    for name, cfg in SV.SURVEYS.items():
        assert 0.01 < cfg["scale"] < 0.5
        assert 0.0 < cfg["fwhm"] < 2.0


def test_simulate_one_shapes_and_physics():
    rng = np.random.default_rng(0)
    p = S.sample_params(rng)
    out = S.simulate_one(p, fov=10.0, fine_scale=0.05)
    # all maps square and same size; pixel scale positive
    for k in ("f_gt", "g", "h"):
        a = out[k]
        assert a.ndim == 2 and a.shape[0] == a.shape[1]
    assert out["g"].shape == out["f_gt"].shape == out["h"].shape
    assert out["s"] > 0 and out["theta_E"] > 0
    # PSF stamp is (approximately) unit-normalized -> reblur loss is consistent
    assert abs(out["h"].sum() - 1.0) < 1e-3
    # blurring conserves total flux to good accuracy
    assert abs(out["g"].sum() - out["f_gt"].sum()) / out["f_gt"].sum() < 0.05


def test_model_forward_backward():
    torch.manual_seed(0)
    H = W_ = 32
    net = SurveyInvariantLensNet(patch=4, dim=32, depth=2, n_heads=2)
    g = torch.randn(2, 1, H, W_)
    h = torch.rand(2, 1, H, W_)
    s = torch.tensor([0.1, 0.1])
    out = net(g, h, s)
    assert "recon" in out and "theta_E" in out
    assert out["recon"].shape == g.shape
    assert out["theta_E"].shape[0] == 2
    loss = out["recon"].abs().mean() + out["theta_E"].pow(2).mean()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(gr).all() for gr in grads)


def test_add_noise_signal_dependent():
    # noise variance = signal/gain + bkg_rms^2 : background RMS matches bkg_rms,
    # and a brighter constant field gets more noise (shot term).
    rng = np.random.default_rng(0)
    bkg_rms, gain = 0.002, 1.0e4
    zero = np.zeros((128, 128), dtype=np.float32)
    n0 = S.add_noise(zero, bkg_rms, gain, rng)
    assert abs(n0.std() - bkg_rms) < 0.15 * bkg_rms          # background floor recovered
    bright = np.full((128, 128), 0.5, dtype=np.float32)
    nb = S.add_noise(bright, bkg_rms, gain, rng)
    assert (nb - bright).std() > n0.std()                     # shot noise adds on signal


def test_simulate_one_noisy_keeps_truth_clean():
    rng = np.random.default_rng(1)
    p = S.sample_params(rng)
    p.update(SV.SURVEYS["DES"])                               # noisiest survey
    o = S.simulate_one_noisy(p, fov=8.0, fine_scale=0.08,
                             rng=np.random.default_rng(2))
    # input g is noised; g_clean and f_gt are preserved; h stays a valid PSF
    assert "g_clean" in o and not np.allclose(o["g"], o["g_clean"])
    assert abs(o["h"].sum() - 1.0) < 1e-3                     # PSF still unit-normalized
    assert o["f_gt"].shape == o["g"].shape


def test_cross_attention_changes_param_count():
    full = SurveyInvariantLensNet(patch=4, dim=32, depth=2, n_heads=2,
                                  use_cross_attention=True)
    abl = SurveyInvariantLensNet(patch=4, dim=32, depth=2, n_heads=2,
                                 use_cross_attention=False)
    n_full = sum(p.numel() for p in full.parameters())
    n_abl = sum(p.numel() for p in abl.parameters())
    assert n_abl < n_full  # removing cross-attn drops parameters
