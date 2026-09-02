# survey-invariant-lensnn

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A **telescope-invariant neural network** for strong gravitational lens images.
The network takes a blurred lens image, its PSF stamp, and the pixel scale, and produces
a PSF-invariant, deblurred embedding from which it (a) reconstructs the sharp
image and (b) regresses the Einstein radius theta_E. The architecture is robust
to PSF blurring and to variable pixel scales across surveys (HST, Euclid, Roman,
Rubin, DES), so a single model generalizes across telescopes. 

## Method in one paragraph

The image and PSF stamp are split into fixed `p x p` patches, linearly
projected to tokens, and tagged with (i) a 2D RoPE positional embedding in the
**shared arcsecond frame centered on the lens** (normalized by `theta_ref`) and
(ii) a learned type embedding (image vs PSF). A transformer block runs image
self-attention, PSF self-attention, and **image -> PSF cross-attention**: every
image token queries the full PSF representation, so the relative-offset RoPE
phase encodes the deconvolution structure directly into the attention. Two heads
read the shared embedding: a decoder that reconstructs the sharp image at the
*same* pixel scale (deconvolution, not super-resolution) and an MLP that
regresses theta_E. The loss is
`L = L_recon + lambda_c x L_reblur + lambda_d x L2_reg`, where the reconstruction loss
`L_recon  = || f_hat - f_gt || + lambda_a x (1 - SSIM(f_hat, f_gt)) + lambda_b x || F(f_hat) - F(f_gt)||`, the
self-supervised re-blurring term `L_reblur = || f_hat * h - g ||^2`, and regression term `L2_reg = || theta - theta_hat||^2`
<!-- 
## Install

```bash
# CPU PyTorch (CI / laptop); for GPU install the CUDA build of torch instead
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

This installs the `survey_lens` package and three console commands:
`survey-lens-train`, `survey-lens-test`, `survey-lens-diagnose`.
(The equivalent module forms `python -m survey_lens.train` etc. also work.)

## Quick start

```bash
# proposed method: full model, joint loss, PSF cross-attention on
survey-lens-train --name full --epochs 30
```

Metrics stream to `runs/<name>/version_*/metrics.csv`; the best checkpoint by
`val/loss` is saved automatically. -->

<!-- ## Evaluating on Test Set 1 (simulated telescope data) -->

<!-- ```bash
survey-lens-test --ckpt runs/full/version_0/checkpoints/epoch=29-step=...ckpt
``` -->
## Test Set 1 and Test Set 2(simulated telescope data)

Test Set 1 = the 5 fixed survey configs (HST, Euclid, Roman, Rubin, DES),
each applied to the **same** underlying lens population (paired comparison, so a
survey gap reflects the telescope, not a different lens sample without noise. 
<!-- Reports per survey: `theta_rmse / theta_bias / theta_scatter`
(Einstein radius), `recon_l1 / recon_ssim` (reconstruction vs sharp truth), and
`reblur_mse` (self-supervised consistency — the primary diagnostic on real data).
Outputs `test_set1_metrics.csv` (one row per survey + an `ALL` aggregate) and
`test_set1_summary.png`. Runs with random weights if `--ckpt` is omitted (smoke
test only — metrics not meaningful). -->

<!-- ## Evaluating on Test Set 2 (per-survey noise) -->

<!-- ```bash
survey-lens-test --test-set 2 --ckpt runs/full/version_0/checkpoints/epoch=...ckpt
survey-lens-diagnose --test-set 2 --ckpt runs/full/version_0/checkpoints/epoch=...ckpt
``` -->

Test Set 2 reuses the **same paired lens population** as Test Set 1 but adds a
realistic per-survey noise model to the input image `g` only — the sharp truth
`f_gt`, the PSF `h`, and `theta_E_true` are kept clean, and a noise-free copy is
retained as `g_clean`. Noise is signal-dependent: `var = max(g, 0)/gain +
bkg_rms²` (a Gaussian approximation to Poisson shot noise plus a background
floor), applied at the native pixel scale after convolve-then-resample. Each
survey carries its own `bkg_rms` and `gain` in `surveys.py`. 
<!-- Space surveys stay
high-SNR (HST arc-SNR ≈ 11); ground surveys are noise-dominated (Rubin ≈ 2, DES
≈ 1.6 — heaviest blur + shallowest depth). This is the primary distribution-shift
robustness test. Outputs are written with a `test_set2_` / `diag_set2_` prefix. -->

<!-- ## Visual diagnostics

```bash
survey-lens-diagnose --ckpt runs/full/version_0/checkpoints/epoch=...ckpt --n-examples 3
```

Produces, per example lens, a per-survey panel figure with five columns —
sharp `f_gt`, PSF-convolved input `g`, reconstruction `f_hat`, recon residual
(`f_hat - f_gt`, supervised), reblur residual (`f_hat * h - g`, self-supervised
— the one that also works on real data). A well-trained model gives a deblurred
ring in column 3 and structureless residuals in columns 4-5. Also writes a
true-vs-predicted theta_E scatter colored by survey (`diag_regression.png`).
Runs with random weights if `--ckpt` is omitted (smoke test only). -->

<!-- ## The three experiments (proposal)

**(1) Does PSF cross-attention help Einstein-radius recovery?**
```bash
survey-lens-train --name full      --epochs 30
survey-lens-train --name no_cross  --epochs 30 --no-cross-attention
```
Compare `val/theta_rmse`.

**(2) Joint vs single-task training**
```bash
survey-lens-train --name joint       --epochs 30 --loss-mode joint
survey-lens-train --name recon_only  --epochs 30 --loss-mode recon_only
survey-lens-train --name reg_only    --epochs 30 --loss-mode reg_only
```

**(3) Native pixel scale vs resampling to a common grid**
The native-scale path is the default (variable token count per survey, handled
by RoPE positional embeddings). The resampled baseline is a data-side change:
resample every image to one fixed scale before tokenizing. Hook this in
`data.py` (a `resample_to` arg on the dataset) — not yet wired, flagged as the
next step. -->

## Code layout

```
survey-invariant-lensnn/
├── src/survey_lens/
│   ├── env_setup.py   # import FIRST: redirects config/cache, configures numba
│   ├── sim_lens.py    # simulator: sample_params, simulate_one (convolve-then-resample)
│   ├── surveys.py     # fixed survey configs (HST/Euclid/Roman/Rubin/DES)
│   ├── model.py       # SurveyInvariantLensNet: patch tokenizer, 2D-RoPE, tri-attention
│   ├── losses.py      # reconstruction (L1+SSIM+FFT), reblur, regression
│   ├── data.py        # LightningDataModule + size-bucket sampler
│   ├── lit_model.py   # LitLensNet — ablation switches are constructor flags
│   ├── train.py       # CLI: survey-lens-train
│   ├── test.py        # CLI: survey-lens-test  (Test Set 1 / 2)
│   └── diagnose.py    # CLI: survey-lens-diagnose (visual panels + regression)
├── tests/             # fast CPU smoke tests (simulator physics + fwd/bwd pass)
├── .github/workflows/ # CI (lint + tests on py3.10/3.11)
├── pyproject.toml
├── LICENSE
└── README.md
```

<!-- ## Editing guide for a physicist

- **Priors** (theta_E, ellipticity, shear, PSF FWHM, pixel scale): `sim_lens.sample_params`.
- **Field of view / fine grid**: `fov`, `fine_scale` args (default 20", 0.025"/px fine).
- **Loss weights** lambda_a (SSIM), lambda_b (FFT), lambda_c (reblur), lambda_d (regression):
  `LitLensNet` constructor / `train` flags.
- **Model size** (dim, depth, heads, patch): `LitLensNet` constructor / flags.
- **Add a survey**: one line in `surveys.SURVEYS`.

## Physics conventions

- **Convolve-then-resample**: sharp image rendered on a fine grid -> Gaussian PSF
  convolution on that fine grid -> area (flux-mean) resampling to the survey
  native scale. This is the physically correct order.
- **Shared coordinate origin**: image and PSF stamp are both centered on the
  lens center (0,0) at the same pixel scale, so the *same* RoPE positional
  pipeline applies to both — this is what makes the cross-attention encode the
  deconvolution as a relative-offset phase.
- **Unit-normalized PSF**: the PSF stamp sums to 1, so the re-blurring
  consistency loss is self-consistent and flux is conserved.

## Roadmap

- [x] Test Set 2: per-survey noise model (`sim_lens.add_noise` /
      `simulate_one_noisy`, `--test-set 2`).
- [ ] Experiment 3: `resample_to` data-side switch for the common-grid baseline.
- [ ] Test Set 3: real lenses (COSMOS HST, DES, Euclid Q1) — reblur loss as the
      no-ground-truth diagnostic.
- [ ] GPU training run + checkpoints. -->

## License

MIT — see [LICENSE](LICENSE).
