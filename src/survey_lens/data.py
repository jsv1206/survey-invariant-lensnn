"""LightningDataModule for simulated strong-lens images.

Because each survey's pixel scale gives a different image size (token count),
a mini-batch must contain images of ONE size. We therefore bucket samples by
their realized pixel scale and yield batches drawn from a single bucket.

A physicist editing this file:
  * change priors            -> edit sim_lens.sample_params
  * change FOV / fine grid   -> edit `fov` below; fine_scale=None means
    ADAPTIVE per-sample supersampling (exact pixel-scale fidelity; recommended)
  * use fixed survey configs -> pass `survey_configs=surveys.SURVEYS`
"""
import functools
import json
import multiprocessing as mp
import os

from . import env_setup  # noqa: F401  (must precede lenstronomy/astropy AND
                          # lightning imports -- lightning transitively pulls
                          # in an ABI-broken torchvision in this sandbox)
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

from . import sim_lens as S


def _to_tensor(o):
    """Pack one simulator output dict into model-ready tensors."""
    g = torch.from_numpy(o["g"])[None]          # [1,H,W]
    h = torch.from_numpy(o["h"])[None]
    f = torch.from_numpy(o["f_gt"])[None]
    return dict(g=g, h=h, f_gt=f,
                s=torch.tensor(float(o["s"]), dtype=torch.float32),
                theta_E=torch.tensor(float(o["theta_E"]), dtype=torch.float32))


def draw_params_sequence(n, seed, survey_configs=None):
    """The exact sequential parameter-draw order `SimLensDataset` uses: one
    RNG stream consumed in order via `sim_lens.sample_params`, optionally
    overridden by a survey config cycled by index. Factored out (rather than
    inlined in `SimLensDataset.__init__`) so `cache_datasets.py` can
    reproduce a BYTE-IDENTICAL lens population without re-simulating in this
    process -- it draws the same params here, then farms the expensive
    `simulate_one` renders out to a process pool.
    """
    rng = np.random.default_rng(seed)
    survey_names = list(survey_configs) if survey_configs else None
    out = []
    for i in range(n):
        p = S.sample_params(rng)
        if survey_configs:
            nm = survey_names[i % len(survey_names)]
            p.update(survey_configs[nm])
        out.append(p)
    return out


class _ShapeStub:
    """Just enough of a tensor's interface (`.shape`) for `SizeBucketSampler`,
    without loading the sample's pixel data from disk."""
    def __init__(self, shape):
        self.shape = shape


class _CachedMeta:
    """One cached sample's metadata (from the split manifest): file path,
    npix/scale/fwhm/theta_E -- everything `SizeBucketSampler` (needs `.shape`
    to bucket by image size) and `_load_cached_sample` (needs the scalars to
    rebuild `h` and the tensors) require, with NO pixel array loaded yet."""
    __slots__ = ("path", "npix", "s", "fwhm", "theta_E")

    def __init__(self, path, npix, s, fwhm, theta_E):
        self.path, self.npix = path, npix
        self.s, self.fwhm, self.theta_E = s, fwhm, theta_E

    def __getitem__(self, key):
        return _ShapeStub((1, self.npix, self.npix))


def _load_cached_sample(meta):
    """Load one cached sample's tensors from its .npz. `h` is NOT stored on
    disk (the project notes: it's a pure function of (fwhm, s, npix), regenerating
    to 2.5e-9) -- rebuild it here from the manifest scalars instead."""
    d = np.load(meta.path)
    g = torch.from_numpy(d["g"].astype(np.float32))[None]
    f = torch.from_numpy(d["f_gt"].astype(np.float32))[None]
    h = torch.from_numpy(
        S.gaussian_psf_stamp(meta.fwhm, meta.s, meta.npix).astype(np.float32))[None]
    return dict(g=g, h=h, f_gt=f,
                s=torch.tensor(float(meta.s), dtype=torch.float32),
                theta_E=torch.tensor(float(meta.theta_E), dtype=torch.float32))


def load_manifest_items(cache_dir, expect_fov=None):
    """Read one split's manifest.json into a list of `_CachedMeta` (cheap:
    no pixel arrays touched). See `cache_datasets.py` for the manifest schema.

    `expect_fov`, if given, is checked against the manifest's own `fov` and a
    mismatch RAISES. The cached .npz files hold only `f_gt`/`g` -- no field of
    view, no coordinate metadata -- so nothing downstream can notice that the
    images are on a different angular grid than the caller asked for. Before
    this check, `train.py --fov 12 --cache-dir <a 20" cache>` trained happily
    on 20" data and reported it as a 12" run. With caches at several FOVs
    coexisting on disk that is a silent wrong-experiment bug, not an
    inconvenience, so it is a hard error rather than a warning.
    """
    with open(os.path.join(cache_dir, "manifest.json")) as f:
        manifest = json.load(f)
    cached_fov = manifest.get("fov")
    if expect_fov is not None and cached_fov is not None:
        if abs(float(cached_fov) - float(expect_fov)) > 1e-6:
            raise ValueError(
                f"FOV mismatch: cache {cache_dir!r} was built at fov="
                f"{float(cached_fov)}\" but fov={float(expect_fov)}\" was requested. "
                f"Pass --fov {float(cached_fov)} to use this cache, or point "
                f"--cache-dir at one built at {float(expect_fov)}\".")
    items = []
    for rec in manifest["samples"]:
        items.append(_CachedMeta(
            path=os.path.join(cache_dir, rec["file"]), npix=rec["npix"],
            s=rec["scale"], fwhm=rec["fwhm"], theta_E=rec["theta_E"]))
    return items


class SimLensDataset(Dataset):
    """Samples bucketed by image size. Either simulates in memory at
    construction (default), or -- if `cache_dir` is given -- reads a split
    frozen to disk by `cache_datasets.py`: cheap
    metadata is loaded upfront for bucketing, and each sample's pixel data is
    loaded from its .npz lazily, in `__getitem__`, one file per item (O(1)).
    """

    def __init__(self, n, seed, fov=20.0, fine_scale=None,
                 survey_configs=None, cache_dir=None, train_noise=False,
                 noise_seed=0):
        self.fov, self.fine_scale, self.cache_dir = fov, fine_scale, cache_dir
        # Apply CORRELATED noise on the fly. Deliberately not baked into
        # the cache -- a fixed realization would be memorized, and rebuilding
        # the frozen 12" cache costs ~1.4h and would make noise-free and noisy runs train
        # on different bytes. Drawing here gives a fresh level AND realization
        # per sample per epoch (free augmentation) while the cache stays
        # byte-identical across every arm.
        self.train_noise = bool(train_noise)
        self.noise_seed = int(noise_seed)
        self.epoch = 0
        if cache_dir is not None:
            self.items = load_manifest_items(cache_dir, expect_fov=fov)
        else:
            self.items = []
            for p in draw_params_sequence(n, seed, survey_configs):
                o = S.simulate_one(p, fov=fov, fine_scale=fine_scale)
                self.items.append(_to_tensor(o))

    def __len__(self):
        return len(self.items)

    def set_epoch(self, e):
        """Advance the noise stream so each epoch sees a new realization."""
        self.epoch = int(e)

    def __getitem__(self, idx):
        it = self.items[idx]
        if self.cache_dir is not None:
            it = _load_cached_sample(it)
        if not self.train_noise:
            return it
        # Seeded on (noise_seed, epoch, idx): reproducible, independent across
        # dataloader WORKERS (which each hold a copy of this dataset, so a
        # shared generator would repeat), and different every epoch.
        rng = np.random.default_rng((self.noise_seed, self.epoch, idx))
        nz = S.sample_noise(rng)
        g = it["g"].numpy()[0]
        g_noisy = S.add_correlated_noise(g, nz["bkg_rms"], nz["gain"], rng,
                                         nz["corr_px"])
        out = dict(it)
        out["g"] = torch.from_numpy(g_noisy.astype(np.float32))[None]
        # the per-pixel variance the chi^2 reblur term divides by. Computed from
        # the NOISY g, which is what a real pipeline has access to.
        var = np.maximum(g_noisy, 0.0) / nz["gain"] + nz["bkg_rms"] ** 2
        out["var"] = torch.from_numpy(var.astype(np.float32))[None]
        return out


def _pool_worker_init():
    """Runs once in each worker process before any task. Caps BLAS/OMP
    threads to 1 -- parallelism already comes from the process pool itself,
    so without this every worker would ALSO try to use the full cgroup CPU
    quota's threads internally, oversubscribing it by a factor of n_workers
    (same issue as cache_datasets.py; see env_setup.py). Scoped to the child
    process only (via Pool's initializer), so the main process's own thread
    settings (e.g. for GPU eval running concurrently) are untouched."""
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"
    from . import env_setup  # noqa: F401


def _sim_with_contam(arg, fov=None, fine_scale=None):
    """starmap-able wrapper: `arg` is a (params, contam) pair. Module-level so
    it is picklable by the process pool."""
    p, c = arg
    return S.simulate_one(p, fov=fov, fine_scale=fine_scale, contam=c)


def _render_params(params_list, fov, fine_scale, n_workers=1, contams=None):
    """simulate_one(p, fov, fine_scale) for every p in params_list, in the
    SAME ORDER as params_list. simulate_one has no internal RNG, so this is
    exactly reproducible regardless of n_workers -- only wall time changes.
    n_workers<=1 stays fully sequential, in-process (the default, and the
    only path exercised by existing tests -- zero behavior change there)."""
    if contams is None:
        # unchanged path -- every existing caller lands here, bit-for-bit
        if n_workers <= 1 or len(params_list) < 2:
            return [S.simulate_one(p, fov=fov, fine_scale=fine_scale) for p in params_list]
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        fn = functools.partial(S.simulate_one, fov=fov, fine_scale=fine_scale)
        with ctx.Pool(processes=n_workers, initializer=_pool_worker_init) as pool:
            return pool.map(fn, params_list)

    if len(contams) != len(params_list):
        raise ValueError("contams must align 1:1 with params_list")
    pairs = list(zip(params_list, contams))
    if n_workers <= 1 or len(pairs) < 2:
        return [_sim_with_contam(a, fov=fov, fine_scale=fine_scale) for a in pairs]
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    fn = functools.partial(_sim_with_contam, fov=fov, fine_scale=fine_scale)
    with ctx.Pool(processes=n_workers, initializer=_pool_worker_init) as pool:
        return pool.map(fn, pairs)


def build_test_set_1(survey_configs, n_lenses=200, seed=12345,
                     fov=20.0, fine_scale=None, n_workers=1):
    """Test Set 1: fixed survey configs, evaluated on the SAME underlying lenses.

    Draws `n_lenses` lens/source parameter sets once, then renders each lens
    through every survey's (pixel scale, PSF FWHM). This isolates cross-survey
    generalization from lens-population differences (paired comparison): a
    survey gap reflects the telescope, not a different lens sample.

    `n_workers>1` parallelizes the (expensive, CPU-bound) render step across
    a process pool -- see `_render_params`; the parameter draw itself stays
    sequential, so the returned population is identical regardless of
    `n_workers`, only faster.

    Returns: dict survey_name -> list of model-ready sample dicts (see _to_tensor),
             each also carrying 'theta_E_true' (float) for convenience.
    """
    rng = np.random.default_rng(seed)
    base_params = [S.sample_params(rng) for _ in range(n_lenses)]
    out = {}
    for nm, cfg in survey_configs.items():
        params_list = [dict(bp, **cfg) for bp in base_params]   # override scale + fwhm
        sims = _render_params(params_list, fov, fine_scale, n_workers)
        items = []
        for o in sims:
            t = _to_tensor(o)
            t["theta_E_true"] = float(o["theta_E"])
            items.append(t)
        out[nm] = items
    return out


def build_test_set_2(survey_configs, n_lenses=200, seed=12345,
                     fov=20.0, fine_scale=None, n_workers=1):
    """Test Set 2: Test Set 1 + realistic per-survey noise on the INPUT.

    Identical paired design to Test Set 1 (same underlying lens population sent
    through every survey), but each survey's blurred input `g` is corrupted by
    that survey's noise model (`sim_lens.add_noise`, using the `bkg_rms`/`gain`
    in `survey_configs`). The ground-truth sharp image `f_gt` and the PSF stamp
    `h` stay clean -- so this set tests robustness to a noise distribution shift
    the model never saw at train time (train data is noise-free).

    The noise RNG is derived deterministically from `seed` (offset by +1) so
    Test Set 2 is reproducible and uses a different stream than the lens-param
    draw. Each returned item also carries 'theta_E_true'.

    `n_workers>1` parallelizes only the expensive render step (`simulate_one`,
    no internal RNG -- see `_render_params`); the noise draw stays sequential
    in this process, in the same (survey, base_params) order as before, so
    the noised output is identical regardless of `n_workers`.

    Returns: dict survey_name -> list of model-ready sample dicts.
    """
    rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 1)
    base_params = [S.sample_params(rng) for _ in range(n_lenses)]
    out = {}
    for nm, cfg in survey_configs.items():
        if "bkg_rms" not in cfg or "gain" not in cfg:
            raise ValueError(
                f"survey '{nm}' has no noise config (bkg_rms/gain); "
                "Test Set 2 needs surveys.SURVEYS with noise keys.")
        params_list = [dict(bp, **cfg) for bp in base_params]
        sims = _render_params(params_list, fov, fine_scale, n_workers)
        items = []
        for o in sims:
            o = dict(o)
            o["g"] = S.add_noise(o["g"], cfg["bkg_rms"], cfg["gain"], noise_rng)
            t = _to_tensor(o)
            t["theta_E_true"] = float(o["theta_E"])
            items.append(t)
        out[nm] = items
    return out


class SizeBucketSampler(torch.utils.data.Sampler):
    """Yields batches whose items all share the same image size (token count)."""

    def __init__(self, dataset, batch_size, shuffle=True, seed=0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        buckets = {}
        for idx, it in enumerate(dataset.items):
            key = tuple(it["g"].shape[-2:])
            buckets.setdefault(key, []).append(idx)
        self.buckets = buckets

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for idxs in self.buckets.values():
            idxs = list(idxs)
            if self.shuffle:
                rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batches.append(idxs[i:i + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        n = 0
        for idxs in self.buckets.values():
            n += (len(idxs) + self.batch_size - 1) // self.batch_size
        return n


def collate(items):
    """Stack a list of same-size samples into batched tensors.

    `var` is present only when the dataset applies train-time noise; its
    presence is what `lit_model` keys on to select the chi^2 reblur term, so the
    noise-free arms take exactly the same path as before.
    """
    out = dict(
        g=torch.stack([x["g"] for x in items]),
        h=torch.stack([x["h"] for x in items]),
        f_gt=torch.stack([x["f_gt"] for x in items]),
        s=torch.stack([x["s"] for x in items]),
        theta_E=torch.stack([x["theta_E"] for x in items]),
    )
    if "var" in items[0]:
        out["var"] = torch.stack([x["var"] for x in items])
    return out


class LensDataModule(L.LightningDataModule):
    def __init__(self, n_train=512, n_val=128, batch_size=8,
                 fov=20.0, fine_scale=None, seed=0, survey_configs=None,
                 cache_dir=None, train_noise=False):
        """`cache_dir`, if given, must point at a directory `cache_datasets.py`
        wrote, containing `train/` and `val/`
        subdirs with their own manifest.json -- loads the frozen split
        instead of re-simulating in memory. `n_train`/`n_val`/`seed` are then
        used only to sanity-check the manifest matches what was requested
        (see `setup`), not to draw new samples; `survey_configs` similarly
        must match what the cache was built with (cache_datasets.py always
        caches val against the full 5-survey config, i.e. --survey-val)."""
        super().__init__()
        self.save_hyperparameters(ignore=["survey_configs"])
        self.survey_configs = survey_configs
        self.cache_dir = cache_dir

    def setup(self, stage=None):
        h = self.hparams
        if self.cache_dir is not None:
            train_dir = os.path.join(self.cache_dir, "train")
            val_dir = os.path.join(self.cache_dir, "val")
            self.train_ds = SimLensDataset(h.n_train, seed=h.seed, fov=h.fov,
                                           fine_scale=h.fine_scale, cache_dir=train_dir,
                                           train_noise=h.train_noise,
                                           noise_seed=h.seed)
            # val gets noise too when training does -- otherwise val/loss is
            # measured on a different distribution than the model is fitting
            # and is not a usable early-stopping/monitoring signal.
            self.val_ds = SimLensDataset(h.n_val, seed=h.seed + 9999, fov=h.fov,
                                         fine_scale=h.fine_scale, cache_dir=val_dir,
                                         train_noise=h.train_noise,
                                         noise_seed=h.seed + 9999)
            if len(self.train_ds) != h.n_train or len(self.val_ds) != h.n_val:
                raise ValueError(
                    f"cache_dir={self.cache_dir!r} has {len(self.train_ds)} train / "
                    f"{len(self.val_ds)} val samples, requested n_train={h.n_train} / "
                    f"n_val={h.n_val} -- pass matching --n-train/--n-val or rebuild the cache.")
            return
        self.train_ds = SimLensDataset(h.n_train, seed=h.seed,
                                       fov=h.fov, fine_scale=h.fine_scale)
        self.val_ds = SimLensDataset(h.n_val, seed=h.seed + 9999,
                                     fov=h.fov, fine_scale=h.fine_scale,
                                     survey_configs=self.survey_configs)

    def _loader_kwargs(self):
        # num_workers only pays off once data is cached:
        #  with the in-memory path, SimLensDataset already holds every
        # sample's pixels in RAM, so each DataLoader worker process would
        # FORK (and duplicate) the whole in-memory dataset instead of reading
        # small per-sample files -- expensive and pointless. With cache_dir
        # set, each worker just opens its own .npz files.
        if self.cache_dir is None:
            return {}
        return dict(num_workers=4, pin_memory=True, persistent_workers=True)

    def train_dataloader(self):
        sampler = SizeBucketSampler(self.train_ds, self.hparams.batch_size,
                                    shuffle=True, seed=self.hparams.seed)
        return DataLoader(self.train_ds, batch_sampler=sampler, collate_fn=collate,
                          **self._loader_kwargs())

    def val_dataloader(self):
        sampler = SizeBucketSampler(self.val_ds, self.hparams.batch_size,
                                    shuffle=False, seed=self.hparams.seed)
        return DataLoader(self.val_ds, batch_sampler=sampler, collate_fn=collate,
                          **self._loader_kwargs())
