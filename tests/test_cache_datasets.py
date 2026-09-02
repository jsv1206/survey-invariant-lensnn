"""Regression tests for the dataset caching script.

Covers the properties that matter: a given --seed must
reproduce byte-identical output (in the sense of identical array content --
the .npz zip container itself is not required to match byte-for-byte, since
zip entries carry a timestamp), the cache must not depend on worker count,
the PSF `h` must NOT be stored on disk (it's regenerated from scalars), and
a cached sample must agree with a freshly-simulated one at fp16 precision.
"""
import json

import numpy as np
import pytest
import torch

from survey_lens import env_setup  # noqa: F401  (must precede lenstronomy)
from survey_lens import sim_lens as S
from survey_lens.data import SimLensDataset, LensDataModule, draw_params_sequence
from survey_lens.cache_datasets import cache_flat_split, cache_split, write_manifest

FOV = 8.0


def test_manifest_schema(tmp_path):
    cache_flat_split("train", n=3, seed=0, fov=FOV, fine_scale=None,
                     n_workers=2, out_root=str(tmp_path))
    with open(tmp_path / "train" / "manifest.json") as f:
        manifest = json.load(f)
    for key in ("seed", "n", "fov", "fine_scale", "git_commit", "samples"):
        assert key in manifest
    assert manifest["seed"] == 0 and manifest["n"] == 3
    assert len(manifest["samples"]) == 3
    for rec in manifest["samples"]:
        for key in ("file", "npix", "scale", "fwhm", "theta_E"):
            assert key in rec


def test_psf_not_stored_on_disk(tmp_path):
    """the project notes / Step 5: h is a pure function of (fwhm, s, npix) and must
    NOT be written to disk -- only f_gt and g."""
    cache_flat_split("train", n=2, seed=0, fov=FOV, fine_scale=None,
                     n_workers=2, out_root=str(tmp_path))
    with open(tmp_path / "train" / "manifest.json") as f:
        manifest = json.load(f)
    for rec in manifest["samples"]:
        d = np.load(tmp_path / "train" / rec["file"])
        assert set(d.files) == {"f_gt", "g"}, d.files
        assert d["f_gt"].dtype == np.float16 and d["g"].dtype == np.float16


def test_same_seed_reproduces_identical_arrays(tmp_path):
    """Two independent cache runs at the same seed must produce numerically
    identical samples -- the test that would catch any accidental
    nondeterminism (e.g. an RNG touched inside the parallel render step)."""
    cache_flat_split("train", n=4, seed=3, fov=FOV, fine_scale=None,
                     n_workers=2, out_root=str(tmp_path / "run_a"))
    cache_flat_split("train", n=4, seed=3, fov=FOV, fine_scale=None,
                     n_workers=2, out_root=str(tmp_path / "run_b"))
    ma = json.load(open(tmp_path / "run_a" / "train" / "manifest.json"))
    mb = json.load(open(tmp_path / "run_b" / "train" / "manifest.json"))
    for ra, rb in zip(ma["samples"], mb["samples"]):
        assert ra["npix"] == rb["npix"]
        assert ra["scale"] == pytest.approx(rb["scale"], rel=1e-12)
        assert ra["fwhm"] == pytest.approx(rb["fwhm"], rel=1e-12)
        assert ra["theta_E"] == pytest.approx(rb["theta_E"], rel=1e-12)
        da = np.load(tmp_path / "run_a" / "train" / ra["file"])
        db = np.load(tmp_path / "run_b" / "train" / rb["file"])
        assert np.array_equal(da["f_gt"], db["f_gt"])
        assert np.array_equal(da["g"], db["g"])


def test_cache_independent_of_worker_count(tmp_path):
    """simulate_one is a pure function of its params dict (no internal RNG),
    and params are drawn on a single sequential stream BEFORE the pool farms
    out the renders -- so the cache must be identical regardless of how many
    workers rendered it. This is the specific property requires ('seeding each worker from the master seed so results are
    independent of worker count')."""
    cache_flat_split("train", n=6, seed=5, fov=FOV, fine_scale=None,
                     n_workers=1, out_root=str(tmp_path / "w1"))
    cache_flat_split("train", n=6, seed=5, fov=FOV, fine_scale=None,
                     n_workers=3, out_root=str(tmp_path / "w3"))
    m1 = json.load(open(tmp_path / "w1" / "train" / "manifest.json"))
    m3 = json.load(open(tmp_path / "w3" / "train" / "manifest.json"))
    for r1, r3 in zip(m1["samples"], m3["samples"]):
        assert r1["npix"] == r3["npix"]
        d1 = np.load(tmp_path / "w1" / "train" / r1["file"])
        d3 = np.load(tmp_path / "w3" / "train" / r3["file"])
        assert np.array_equal(d1["g"], d3["g"])


def test_cached_matches_fresh_simulation(tmp_path):
    """A cached sample, loaded through SimLensDataset(cache_dir=...), must
    agree with a fresh in-memory simulate_one() call at the same seed/index
    to fp16 round-trip precision (~3e-4 relative to peak, per the project notes)."""
    n, seed = 5, 11
    cache_flat_split("train", n=n, seed=seed, fov=FOV, fine_scale=None,
                     n_workers=2, out_root=str(tmp_path))
    ds = SimLensDataset(n=n, seed=seed, fov=FOV, cache_dir=str(tmp_path / "train"))
    idx = 2
    cached = ds[idx]

    params = draw_params_sequence(n, seed)[idx]
    fresh = S.simulate_one(params, fov=FOV, fine_scale=None)
    g_fresh = torch.from_numpy(fresh["g"])[None]
    f_fresh = torch.from_numpy(fresh["f_gt"])[None]

    assert cached["g"].shape == g_fresh.shape
    rel_g = ((cached["g"] - g_fresh).abs().max() / g_fresh.abs().max()).item()
    rel_f = ((cached["f_gt"] - f_fresh).abs().max() / f_fresh.abs().max()).item()
    assert rel_g < 5e-4, rel_g
    assert rel_f < 5e-4, rel_f
    assert abs(float(cached["theta_E"]) - float(fresh["theta_E"])) < 1e-5
    # h is regenerated, not cached -- must still be unit-normalized
    assert abs(float(cached["h"].sum()) - 1.0) < 1e-6


def test_size_bucket_sampler_does_not_load_pixels(tmp_path, monkeypatch):
    """SizeBucketSampler must bucket cached items using manifest metadata
    only -- loading every sample's array just to read its shape would defeat
    the point of caching."""
    from survey_lens.data import SizeBucketSampler
    cache_flat_split("train", n=4, seed=0, fov=FOV, fine_scale=None,
                     n_workers=2, out_root=str(tmp_path))
    ds = SimLensDataset(n=4, seed=0, fov=FOV, cache_dir=str(tmp_path / "train"))

    calls = []
    orig = np.load
    def _tracking_load(*a, **k):
        calls.append(a)
        return orig(*a, **k)
    monkeypatch.setattr(np, "load", _tracking_load)

    sampler = SizeBucketSampler(ds, batch_size=2)
    assert len(sampler) >= 1
    assert calls == [], f"SizeBucketSampler triggered {len(calls)} np.load() call(s)"


def test_dataloader_workers_only_with_cache_dir(tmp_path):
    """num_workers/pin_memory/persistent_workers must
    be enabled ONLY for the cached path -- with in-memory SimLensDataset, a
    DataLoader worker would fork (and duplicate) the whole already-materialized
    dataset instead of reading small per-sample files."""
    cache_flat_split("train", n=2, seed=0, fov=FOV, fine_scale=None,
                     n_workers=2, out_root=str(tmp_path))
    cache_flat_split("val", n=2, seed=1, fov=FOV, fine_scale=None,
                     n_workers=2, out_root=str(tmp_path))

    dm_mem = LensDataModule(n_train=2, n_val=2, fov=FOV)
    assert dm_mem._loader_kwargs() == {}

    dm_cached = LensDataModule(n_train=2, n_val=2, fov=FOV, cache_dir=str(tmp_path))
    kwargs = dm_cached._loader_kwargs()
    assert kwargs.get("num_workers") == 4
    assert kwargs.get("pin_memory") is True
    assert kwargs.get("persistent_workers") is True


def test_lens_data_module_rejects_size_mismatch(tmp_path):
    """If the cache has a different n than requested, fail loudly rather
    than silently training on the wrong split size."""
    cache_flat_split("train", n=3, seed=0, fov=FOV, fine_scale=None,
                     n_workers=2, out_root=str(tmp_path))
    cache_flat_split("val", n=2, seed=1, fov=FOV, fine_scale=None,
                     n_workers=2, out_root=str(tmp_path))
    dm = LensDataModule(n_train=999, n_val=2, fov=FOV, cache_dir=str(tmp_path))
    with pytest.raises(ValueError):
        dm.setup()


def test_fov_mismatch_against_cache_raises(tmp_path):
    """A cache built at one FOV must not be silently consumed as another.

    The .npz files hold only `f_gt`/`g` -- no field of view, no coordinate
    metadata -- so without this guard `train.py --fov 12 --cache-dir <20"
    cache>` trains on 20" images and reports a 12" run, with every downstream
    number (token counts, mask_frac, the fixed-arcsec metric crops) quietly
    describing the wrong geometry. Now that caches at several FOVs coexist on
    disk this is a wrong-experiment bug, so it must be a hard error.
    """
    cache_flat_split("train", n=2, seed=0, fov=FOV, fine_scale=None,
                     n_workers=2, out_root=str(tmp_path))
    cache_dir = str(tmp_path / "train")

    # matching FOV: fine
    ds = SimLensDataset(2, seed=0, fov=FOV, cache_dir=cache_dir)
    assert len(ds) == 2

    # mismatched FOV: must raise, and must name both values
    with pytest.raises(ValueError, match="FOV mismatch"):
        SimLensDataset(2, seed=0, fov=FOV * 1.5, cache_dir=cache_dir)


def test_fov_mismatch_raises_through_the_datamodule(tmp_path):
    """The guard must fire on the path training actually uses, not just on a
    directly-constructed dataset."""
    cache_flat_split("train", n=2, seed=0, fov=FOV, fine_scale=None,
                     n_workers=2, out_root=str(tmp_path))
    cache_flat_split("val", n=2, seed=1, fov=FOV, fine_scale=None,
                     n_workers=2, out_root=str(tmp_path))
    dm = LensDataModule(n_train=2, n_val=2, batch_size=1, fov=FOV * 1.5,
                        cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="FOV mismatch"):
        dm.setup()
