# survey-invariant-lensnn

Survey-invariant transformer embedding for strong gravitational lens images:
robust to PSF blurring and variable pixel scale across HST/Euclid/Roman/Rubin/DES.
Dual heads: image reconstruction (deconvolution) + Einstein-radius regression.

Read `README.md` for architecture and `src/survey_lens/` for the implementation.
Deeper references, read on demand (do NOT auto-load):
- `dataset_spec.md` — verified dataset spec + the four known bugs below
- `plan_sim2science.md` — week plan and paper outline

## Current objective (time-boxed)

Submitting a **5-page paper to the Sim2Science NeurIPS workshop, deadline Aug 29 2026**
(NeurIPS template; 5 pp excluding references/appendix).

The workshop theme is *leveraging imperfect scientific simulators against real data*.
This determines emphasis: **the headline contribution is the sim-to-real angle** — the
self-supervised reblur consistency loss (works without ground truth on real data) and
survey-invariance (robustness to per-instrument simulator mis-specification). The
architecture ablations are supporting evidence, not the headline. When helping with
writing or experiment priority, weight the noise-shift test (Test Set 2) and the real-data
demo (Test Set 3) above the architecture ablations.

Test Set 3 stays **qualitative** for this deadline: 2–3 real lenses, reblur residual as
the diagnostic. Do not scope a full real-data benchmark.

## State as of Aug 2026

Pipeline is fully scaffolded and tests pass (smoke + RoPE coverage). **Nothing has been
trained end-to-end yet** — no checkpoints, no cached dataset, no `runs/`. The first
real milestone is a GPU smoke run that descends.

## Four known dataset bugs — 1 and 2 FIXED, 3 and 4 still open

These were found by computing the realized values, not by reading the code. They are
not visible from the source. Regression tests for 1 and 2 live in
`tests/test_scale_fidelity.py` — keep them green.

1. **FIXED (Aug 2026) — pixel scales were silently quantized.** `simulate_one` rounded the
   requested scale to a multiple of a single global `fine_scale` (0.025″), which collapsed
   Roman 0.11″ → **0.10″, identical to Euclid** (4 distinct grids, not 5) and shifted
   DES 0.263″ → 0.275″. **Fix implemented: adaptive per-sample supersampling.**
   `fine_scale=None` is now the default everywhere; `choose_supersample_factor(s, fwhm)`
   picks an integer factor and renders on `fine = s/factor`, so the requested scale is
   reproduced **exactly** for every survey. Note this is *not* the `fine_scale=0.005`
   global-grid fix an earlier draft of this file proposed — that was measured at **27–54×**
   the per-sample cost (10k samples: 27.8 h vs 0.68 h single-core), whereas the adaptive
   grid costs only ~2× (1.4 h) and holds render/PSF quality roughly constant across
   surveys instead of over-rendering the coarse ones. Passing an explicit float restores
   the legacy quantizing behaviour (kept for reproducing old runs; documented in `--help`).

2. **FIXED (Aug 2026) — HST and DES sat exactly on the training prior boundaries.** HST
   FWHM 0.10″/scale 0.05″ were the lower endpoints and DES FWHM 0.90″ the upper endpoint,
   giving the two extreme surveys one-sided support — an apparent generalization failure
   could have been a prior-coverage artifact. **Fix implemented:** priors widened to
   FWHM `logU(0.08, 1.1)″` and scale `logU(0.04, 0.32)″/px`. Verified all five surveys are
   now interior (percentiles 8.4%–92.3%). Do not narrow these back to the survey range.

3. **Euclid and Roman PSF stamps are under-sampled.** PSF σ in native pixels: HST 0.85,
   Euclid 0.68, Roman 0.46, Rubin 1.49, DES 1.45 (recomputed after fix 1). Below ~0.8 px the Gaussian stamp barely
   resolves its own kernel, so `h` misrepresents the blur and the reblur loss
   ‖f̂*h − g‖² inherits that error. Fix: sample `h` on a finer grid than the image (the PSF
   token stream has its own coordinates), or document as a limitation.

4. **PARTLY RESOLVED — token counts vary 30×.** At `patch=4`, FOV 20″: HST npix=400 →
   **10,000 image + 10,000 PSF tokens**; Euclid 2,500; Roman 2,025; Rubin 625; DES 361;
   training prior edge (s=0.04″) 15,625. Measured peak is 8.0 GB at batch 8, which FITS
   the 20 GB GPU (see Hardware section) — so this is a budgeting fact, not a blocker.
   **Always smoke-test at the prior edge (s=0.04″), not a random draw** — a random draw
   usually yields a cheap grid and hides the worst case.

## Hardware and GPU status (Aug 2026)

Target: a single **20 GB GPU**. Simulation is CPU-only (lenstronomy); training is GPU.

- `train.py` is GPU-ready: `accelerator="auto"`, model builds tensors on `g.device`,
  attention uses `F.scaled_dot_product_attention` (memory-efficient path available).
- **`test.py` and `diagnose.py` default to `--device cpu`** — pass `--device cuda`.
- Dataloader sets no `num_workers`/`pin_memory`; with a frozen cached dataset, add them.
- **Measured peak memory** (batch=8, dim=128, depth=4, fp32, patch=4), by token count:
  361 tok 0.5 GB · 2.5k tok 2.0 GB · 10k tok (HST) 5.2 GB · **15.6k tok 8.0 GB**.
  The 15.6k figure is the *training prior edge* (s=0.04″), not HST — 10.3% of training
  draws exceed HST's token count, so the prior sets the ceiling. 8 GB at batch 8 fits
  20 GB with headroom; if OOM, drop batch size before touching patch/FOV.
  `patch=8` cuts peak 2.7× (8.0 → 3.0 GB) if more headroom is needed.
- **Decision (Aug 2026): keep HST in both training prior and test set.** Dropping it
  would remove 42% of the log pixel-scale ladder (5.3× → 2.6×, leaving Euclid and Roman
  only 1.1× apart) while saving nothing, since the prior edge dominates memory anyway.

## Frozen datasets — NOT YET BUILT

There is **no caching script yet**; `SimLensDataset` re-simulates in memory at
construction. Build `cache_datasets.py` before running ablations (see the frozen-dataset
convention below). Measured sizing at FOV=20″:
- training, n=10k, `f_gt`+`g` as fp16: **~2.2 GB** (fp32 + PSF = 6.6 GB, wasteful)
- **do not store the PSF**: `h` is a pure function of `(fwhm, s, npix)` and regenerates
  to 2.5e-9 — store the scalars, rebuild `h` on load
- fp16 is safe for the images: round-trip error 3.1e-4 relative to peak
- Test Set 1 = 0.19 GB, Test Set 2 = 0.28 GB (200 lenses × 5 surveys)

## Known implementation bugs found Aug 2026 (fix before Run 1)

5. **`losses.reblur` convolves with the FULL-FOV PSF stamp** -> cost is `npix**4`,
   not `npix**2 * k**2`. Measured: HST = 2.6e10 MACs/sample (410 GFLOP at batch 8);
   a single CPU call was SIGKILLed. The PSF is a narrow Gaussian (sigma ~0.7 px at
   HST) so the stamp is almost all zeros. **Crop the kernel to ~4 sigma support and
   batch the conv** (currently a Python loop over B). Same math; speedups DES 34x,
   Euclid 816x, HST 3265x.
6. **`losses.ssim` is saturated — `lambda_a=0.5` buys nothing.** `C1=0.01**2`,
   `C2=0.03**2` assume data in [0,1] but images peak at ~0.017, so the constants
   swamp the local variances. Measured `ssim(blurred, sharp)` = 0.999993 (HST),
   0.9867 (DES). After normalizing both images by the same `f_gt` peak: 0.9995 and
   0.9677 — discriminative. Fix the normalization, NOT lambda_a.
7. **Cross-survey reconstruction metrics are not comparable as written.** Peak
   per-pixel brightness varies ~11x across surveys (0.0165 HST -> 0.1896 DES), so
   raw per-pixel L1 compares different physical quantities. Normalize per sample
   before any cross-survey recon metric.
8. **Images are sparse: only 2.7% (HST) to 4.0% (DES) of pixels exceed 1% of peak.**
   Global L1/SSIM are ~96% empty-sky agreement. Report metrics inside a signal mask
   (`f_gt > 0.01*peak`) as the headline, keeping global as a hallucination check.
9. **CONFIRMED BUG — flux normalization. `_area_resample` averages where it must sum.**
   `lenstronomy`'s `ImageModel.image()` returns flux PER PIXEL, so mean-pooling the
   fine grid leaves brightness proportional to `1/factor**2`. Proof: same lens, same
   survey, same output grid, varying only `fine_scale` -> `f_gt.max()` = 0.0701
   (factor 2), 0.0165 (4), 0.0102 (5), while `sum(f_gt)*factor**2*s**2` = 82.0 in all
   three cases. Fix: `.sum(axis=(1,3))` in `_area_resample`. Apply to `f_gt`/`g` only,
   NOT to the PSF stamp (`h` must keep `sum(h)==1`).
   **Why severe:** `choose_supersample_factor` picks the factor adaptively from `s`
   and `fwhm`, so amplitude is arbitrary per sample AND correlated with pixel scale —
   the network can read `s` off overall brightness instead of learning scale-invariant
   structure, which would undermine the paper's central claim. Also inverts the TS2
   SNR ordering (HST arc SNR 0.86 vs DES 5.00 — backwards). This is the origin of the
   ~11x brightness spread and ~27x flux variation in items 7/8.
   **Note the interaction:** under the old fixed `fine_scale` the factor was
   near-constant, so this was a roughly uniform offset and largely invisible; making
   supersampling adaptive turned it into a per-sample, `s`-correlated error.
   Items 7 and 8 above must be re-measured after this fix.
10. **Test Set 2 reblur diagnostic has a noise floor.** A perfect reconstruction
   cannot reach `reblur_mse == 0` on noisy input; the floor is
   `mean(max(g,0)/gain + bkg_rms**2)`. Report `reblur_mse / floor`; the raw floor
   spans 6.4x across surveys. **Below 1.0 means the model is fitting noise** — a
   failure mode, not a win.

## Run 1 scope (current task)

See `HANDOFF_run1.md` (11 steps). Full loss, cross-attention on, frozen dataset.
Test Set 1 (no noise) AND Test Set 2 (per-survey noise) on the SAME checkpoint —
TS2 is evaluation-only, no retraining and no noise augmentation, or it stops being
a robustness result. Plus paper figures 1-5. Outputs to
`/deepskieslab/stronglensing/survey-invariant-lensnn/`.
Architecture/loss ablations and Test Set 3 (real data) deferred.
Fix bug 9 FIRST — every pixel value depends on it.

## Conventions and rationale (differ from defaults / easy to get wrong)

- **Convolve-then-resample** is the physically correct order: render sharp on the fine grid →
  convolve with the PSF at fine scale → area-resample to native scale. Never resample first.
- **PSF stamps must stay sum-normalized to 1**, otherwise the reblur loss is not
  self-consistent (it silently rescales the reconstruction).
- **RoPE coordinates are in arcseconds from the lens center**, not pixels. This is the whole
  mechanism of survey invariance — a 0.1″ offset must rotate identically for HST and Rubin.
  Do not "simplify" positional encoding to pixel indices.
- **Freeze one dataset to disk and reuse it for every ablation.** `SimLensDataset` currently
  re-simulates in memory at construction; running ablations that way confounds "changed the
  model" with "changed the data". Cache once, then all configs train on identical bytes.
- **Test Set 1/2 use a paired design**: N lens parameter sets drawn once, each rendered
  through all 5 surveys. So a cross-survey gap reflects the telescope, not a different lens
  sample. Preserve this when editing; state it explicitly in the paper.
- **Test Set 2 noise is applied to the model input only** — `f_gt` and `h` stay clean. Training
  is noise-free, so Test Set 2 is a genuine distribution shift.
- **Global PSNR hides the deconvolution effect** because it behaves like the pixel-L1 term.
  Report **high-frequency (top-octave) radial power-spectrum error** as the reconstruction
  metric, alongside θ_E RMSE and the reblur residual.
- **Reblur loss is nearly blind in the blur operator's null space** — re-blurring discards the
  high frequencies that deconvolution must recover. It is a data-consistency /
  anti-hallucination term, *not* a substitute for the SSIM and Fourier terms. Keep this in
  mind before "simplifying" the loss; if dropping terms, run it as a measured ablation.

## Ablation axes (all flag-driven in `train.py`)

1. PSF cross-attention on/off → `--no-cross-attention`
2. joint vs single-task → `--loss-mode {joint,recon_only,reg_only}`
3. native scale vs resampled-to-common-grid → data-side switch
4. loss-term ablation (drop SSIM+FFT) → `--lambda-a 0 --lambda-b 0`

Metrics go to CSV under `runs/<name>/` (gitignored, as are `*.ckpt` and `diag_*.png`).
