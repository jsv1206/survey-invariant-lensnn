"""LightningDataModule for simulated strong-lens images.

Because each survey's pixel scale gives a different image size (token count),
a mini-batch must contain images of ONE size. We therefore bucket samples by
their realized pixel scale and yield batches drawn from a single bucket.

A physicist editing this file:
  * change priors            -> edit sim_lens.sample_params
  * change FOV / fine grid   -> edit `fov`, `fine_scale` below
  * use fixed survey configs -> pass `survey_configs=surveys.SURVEYS`
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

from . import env_setup  # noqa: F401  (must precede lenstronomy/astropy imports)
from . import sim_lens as S


def _to_tensor(o):
    """Pack one simulator output dict into model-ready tensors."""
    g = torch.from_numpy(o["g"])[None]          # [1,H,W]
    h = torch.from_numpy(o["h"])[None]
    f = torch.from_numpy(o["f_gt"])[None]
    return dict(g=g, h=h, f_gt=f,
                s=torch.tensor(float(o["s"]), dtype=torch.float32),
                theta_E=torch.tensor(float(o["theta_E"]), dtype=torch.float32))


class SimLensDataset(Dataset):
    """Pre-simulated, in-memory dataset. Samples are bucketed by image size."""

    def __init__(self, n, seed, fov=20.0, fine_scale=0.025,
                 survey_configs=None):
        self.fov, self.fine_scale = fov, fine_scale
        rng = np.random.default_rng(seed)
        self.items = []
        survey_names = list(survey_configs) if survey_configs else None
        for i in range(n):
            p = S.sample_params(rng)
            if survey_configs:                  # fixed survey grid + PSF
                nm = survey_names[i % len(survey_names)]
                p.update(survey_configs[nm])
            o = S.simulate_one(p, fov=fov, fine_scale=fine_scale)
            self.items.append(_to_tensor(o))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_test_set_1(survey_configs, n_lenses=200, seed=12345,
                     fov=20.0, fine_scale=0.025):
    """Test Set 1: fixed survey configs, evaluated on the SAME underlying lenses.

    Draws `n_lenses` lens/source parameter sets once, then renders each lens
    through every survey's (pixel scale, PSF FWHM). This isolates cross-survey
    generalization from lens-population differences (paired comparison): a
    survey gap reflects the telescope, not a different lens sample.

    Returns: dict survey_name -> list of model-ready sample dicts (see _to_tensor),
             each also carrying 'theta_E_true' (float) for convenience.
    """
    rng = np.random.default_rng(seed)
    base_params = [S.sample_params(rng) for _ in range(n_lenses)]
    out = {}
    for nm, cfg in survey_configs.items():
        items = []
        for bp in base_params:
            p = dict(bp)
            p.update(cfg)                      # override scale + fwhm only
            o = S.simulate_one(p, fov=fov, fine_scale=fine_scale)
            t = _to_tensor(o)
            t["theta_E_true"] = float(o["theta_E"])
            items.append(t)
        out[nm] = items
    return out


def build_test_set_2(survey_configs, n_lenses=200, seed=12345,
                     fov=20.0, fine_scale=0.025):
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
        items = []
        for bp in base_params:
            p = dict(bp)
            p.update(cfg)                      # scale + fwhm + bkg_rms + gain
            o = S.simulate_one_noisy(p, fov=fov, fine_scale=fine_scale,
                                     rng=noise_rng)
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
    """Stack a list of same-size samples into batched tensors."""
    return dict(
        g=torch.stack([x["g"] for x in items]),
        h=torch.stack([x["h"] for x in items]),
        f_gt=torch.stack([x["f_gt"] for x in items]),
        s=torch.stack([x["s"] for x in items]),
        theta_E=torch.stack([x["theta_E"] for x in items]),
    )


class LensDataModule(L.LightningDataModule):
    def __init__(self, n_train=512, n_val=128, batch_size=8,
                 fov=20.0, fine_scale=0.025, seed=0, survey_configs=None):
        super().__init__()
        self.save_hyperparameters(ignore=["survey_configs"])
        self.survey_configs = survey_configs

    def setup(self, stage=None):
        h = self.hparams
        self.train_ds = SimLensDataset(h.n_train, seed=h.seed,
                                       fov=h.fov, fine_scale=h.fine_scale)
        self.val_ds = SimLensDataset(h.n_val, seed=h.seed + 9999,
                                     fov=h.fov, fine_scale=h.fine_scale,
                                     survey_configs=self.survey_configs)

    def train_dataloader(self):
        sampler = SizeBucketSampler(self.train_ds, self.hparams.batch_size,
                                    shuffle=True, seed=self.hparams.seed)
        return DataLoader(self.train_ds, batch_sampler=sampler, collate_fn=collate)

    def val_dataloader(self):
        sampler = SizeBucketSampler(self.val_ds, self.hparams.batch_size,
                                    shuffle=False, seed=self.hparams.seed)
        return DataLoader(self.val_ds, batch_sampler=sampler, collate_fn=collate)
