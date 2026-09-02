"""Freeze a dataset split to disk.

`SimLensDataset` re-simulates in memory at construction; running every
ablation that way confounds "changed the model" with "changed the data"
(the project notes: "freeze one dataset to disk and reuse it for every ablation").
This script simulates once, in parallel, and writes:

  <out>/<split>/manifest.json          -- seed, n, fov, git commit, and one
                                           record per sample (file, npix,
                                           scale, fwhm, theta_E)
  <out>/<split>/samples/000000.npz ...  -- one file per sample: f_gt, g only

Format choice (one .npz per sample, not a dense array or a single HDF5 file):
  * `npix` varies per sample (5-15625 tokens across the training prior;
    a known dataset issue), so a single dense array does not fit.
  * Writes are then naturally parallel-safe across worker processes: each
    sample owns an independent file, so there is no single-writer lock or
    merge step to coordinate, unlike one shared HDF5 file.
  * Loading stays O(1) per item: one `np.load()` call, no scanning.

Does NOT store the PSF `h` on disk: it is a pure function of
`(fwhm, scale, npix)` (`sim_lens.gaussian_psf_stamp`) and regenerates to
2.5e-9 -- storing the scalars and rebuilding on load (see `data.py`'s
`_load_cached_sample`) saves roughly a third of the per-sample bytes for
free. `f_gt`/`g` are stored as float16 (round-trip error 3.1e-4 relative to
peak, well under the noise floor).

Reproducibility: parameter draws happen SEQUENTIALLY on a single RNG stream,
via `data.draw_params_sequence` -- the exact order `SimLensDataset` /
`data.build_test_set_1` use in memory -- so the cached lens population is
independent of `--workers` and matches a fresh in-memory run at the same
seed byte-for-byte (see tests/test_cache_datasets.py). Only the expensive
`simulate_one` RENDER step (no internal RNG -- a pure function of its params
dict) is farmed out to a process pool; the pool's scheduling order therefore
cannot change the output, only the wall time.

Usage:
  PYTHONPATH=src python -m survey_lens.cache_datasets --root /path/to/data
  PYTHONPATH=src python -m survey_lens.cache_datasets --root /path/to/data \
      --n-train 10000 --n-val 2000 --n-test1 200 --workers 32
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import time

from . import env_setup  # noqa: F401  (must precede lenstronomy/astropy imports)

# env_setup pins OMP/MKL/OpenBLAS threads to the cgroup CPU quota for a
# SINGLE process (e.g. training). This script instead parallelizes across
# `--workers` PROCESSES, each running one single-threaded simulate_one call
# -- without this override every worker would ALSO try to use the full
# quota's worth of BLAS threads internally, oversubscribing the quota by a
# factor of `--workers` (e.g. 32 processes x 32 threads on a 32-core quota).
# Must happen before any BLAS-heavy numpy/scipy call in this process (numpy
# is imported below, after this) and before the worker Pool is created, so
# forked children inherit "1" instead of the cgroup-quota value.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

from . import sim_lens as S
from . import surveys as SV
from .data import draw_params_sequence


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _init_worker():
    from . import env_setup  # noqa: F401  each process needs the same redirects


def _render_one(args):
    idx, params, fov, fine_scale = args
    o = S.simulate_one(params, fov=fov, fine_scale=fine_scale)
    return idx, o


def _dir_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            total += os.path.getsize(os.path.join(dirpath, fn))
    return total


def cache_split(params_list, out_dir, fov, fine_scale, n_workers):
    """Simulate every params dict in `params_list` (in parallel) and write
    one .npz per sample under `out_dir/samples/`. Returns (manifest_samples,
    wall_seconds); does NOT write manifest.json itself (caller adds
    split-level metadata like `survey`).

    Streams results to disk as they arrive (`imap_unordered`) rather than
    collecting them all in memory first -- holding 10k fp64 (f_gt, g) pairs
    at up to 500x500 in memory at once would be ~40GB, most of this sandbox's
    45GB cgroup memory limit (see env_setup.py); streaming bounds memory to
    roughly `n_workers` samples in flight.
    """
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    tasks = [(i, p, fov, fine_scale) for i, p in enumerate(params_list)]
    samples = [None] * len(params_list)
    t0 = time.time()
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    with ctx.Pool(processes=n_workers, initializer=_init_worker) as pool:
        for idx, o in pool.imap_unordered(_render_one, tasks):
            fname = f"samples/{idx:06d}.npz"
            np.savez(os.path.join(out_dir, fname),
                     f_gt=o["f_gt"].astype(np.float16),
                     g=o["g"].astype(np.float16))
            samples[idx] = dict(file=fname, npix=int(o["npix"]),
                                scale=float(o["s"]), fwhm=float(o["params"]["fwhm"]),
                                theta_E=float(o["theta_E"]))
    wall = time.time() - t0
    assert all(s is not None for s in samples), "a worker task never returned"
    return samples, wall


def write_manifest(out_dir, seed, n, fov, fine_scale, samples, **extra):
    manifest = dict(seed=seed, n=n, fov=fov, fine_scale=fine_scale,
                    git_commit=_git_commit(), samples=samples, **extra)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)


def cache_flat_split(name, n, seed, fov, fine_scale, n_workers, out_root,
                     survey_configs=None):
    """Cache a flat n-sample split (train, or a --survey-val-style val set)."""
    params_list = draw_params_sequence(n, seed, survey_configs)
    out_dir = os.path.join(out_root, name)
    samples, wall = cache_split(params_list, out_dir, fov, fine_scale, n_workers)
    write_manifest(out_dir, seed, n, fov, fine_scale, samples)
    size = _dir_size(out_dir)
    print(f"  {name}: n={n} wall={wall:.1f}s ({wall / max(n, 1) * 1000:.1f}ms/sample) "
          f"size={size / 1e6:.1f}MB -> {out_dir}")
    return wall, size


def cache_test_set_1(n_lenses, seed, fov, fine_scale, n_workers, out_root):
    """Cache Test Set 1's paired design: draw `n_lenses` base lens/source
    params ONCE (exactly `data.build_test_set_1`'s draw order), then render
    each through every survey as its own flat sub-split, so it loads via the
    same `SimLensDataset(cache_dir=...)` path as train/val, per survey."""
    rng = np.random.default_rng(seed)
    base_params = [S.sample_params(rng) for _ in range(n_lenses)]
    total_wall, total_size = 0.0, 0
    for nm, cfg in SV.SURVEYS.items():
        params_list = [dict(bp, **cfg) for bp in base_params]   # override scale+fwhm(+noise keys, unused here)
        out_dir = os.path.join(out_root, "test_set_1", nm)
        samples, wall = cache_split(params_list, out_dir, fov, fine_scale, n_workers)
        write_manifest(out_dir, seed, n_lenses, fov, fine_scale, samples, survey=nm)
        size = _dir_size(out_dir)
        total_wall += wall
        total_size += size
        print(f"  test_set_1/{nm}: n={n_lenses} wall={wall:.1f}s size={size / 1e6:.1f}MB")
    return total_wall, total_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="output root, e.g. $ROOT/data")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-test1", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0,
                    help="matches train.py's --seed default; val uses seed+9999, "
                         "the same offset LensDataModule uses for --survey-val")
    ap.add_argument("--test-seed", type=int, default=12345,
                    help="matches test.py's --seed default for Test Set 1")
    ap.add_argument("--fov", type=float, default=20.0)
    ap.add_argument("--fine-scale", type=float, default=None)
    ap.add_argument("--workers", type=int, default=env_setup._n_cpu,
                    help="default = this environment's actual cgroup CPU quota "
                         "(env_setup._n_cpu), NOT os.cpu_count()")
    ap.add_argument("--skip", nargs="*", default=[], choices=["train", "val", "test1"],
                    help="splits to skip, e.g. --skip test1 for a train/val-only run")
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    print(f"caching to {args.root} with {args.workers} workers "
          f"(fov={args.fov}, fine_scale={args.fine_scale})")

    total_wall, total_size = 0.0, 0
    if "train" not in args.skip:
        w, sz = cache_flat_split("train", args.n_train, args.seed, args.fov,
                                 args.fine_scale, args.workers, args.root)
        total_wall += w; total_size += sz
    if "val" not in args.skip:
        # matches LensDataModule.setup() under train.py's --survey-val (the
        # flag Step 8's actual run command uses): fixed survey configs, seed+9999
        w, sz = cache_flat_split("val", args.n_val, args.seed + 9999, args.fov,
                                 args.fine_scale, args.workers, args.root,
                                 survey_configs=SV.SURVEYS)
        total_wall += w; total_size += sz
    if "test1" not in args.skip:
        w, sz = cache_test_set_1(args.n_test1, args.test_seed, args.fov,
                                 args.fine_scale, args.workers, args.root)
        total_wall += w; total_size += sz

    print(f"\nTOTAL: wall={total_wall:.1f}s ({total_wall / 60:.1f}min) "
          f"size={total_size / 1e6:.1f}MB ({total_size / 1e9:.3f}GB)")


if __name__ == "__main__":
    main()
