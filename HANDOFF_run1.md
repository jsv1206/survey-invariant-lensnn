# Handoff: Run 1 — survey invariance on Test Set 1 (no noise)

**Goal.** One trained model, full loss, evaluated on the five survey configs — first
without noise (Test Set 1), then the same checkpoint under per-survey noise
(Test Set 2) as a distribution-shift robustness test. Deliverables: a per-survey table of Einstein-radius regression and PSF-deconvolution
metrics, plus qualitative reconstruction panels. Test Set 1 isolates pixel-scale and PSF
invariance; Test Set 2 adds the sim-to-real noise-shift evidence at no extra training
cost. The loss/architecture ablations and real-data Test Set 3 are out of scope.

**Read `CLAUDE.md` first.** It carries the physics conventions, the four dataset bugs
(1 and 2 fixed, 3 and 4 characterized), and the measured GPU memory table.

**Storage root.** Everything — cached data, checkpoints, logs, figures — goes under
`$ROOT = /deepskieslab/stronglensing/survey-invariant-lensnn/`. Keep the git repo
separate; write only outputs there. Create `$ROOT/{data,runs,figures}`.

**Hardware.** One 20 GB GPU for training; simulation is CPU-only (lenstronomy), 10 cores.

**Rules.** Do the steps in order and STOP at the first failure — report rather than
work around. Report measured numbers (wall time, disk, GPU peak), never estimates.
Do not change simulator physics, priors, `choose_supersample_factor`, or any loss
weight (`lambda_a/b/c/d`). Do not resample surveys onto a common grid — native-scale
handling is the paper's central claim.

---

## Step 0 — FIX THE FLUX NORMALIZATION BUG (hard blocker, do this first)

**This is now confirmed, with a root cause and a one-line fix. It was `UNVERIFIED`
in an earlier draft of this document; it has since been diagnosed. Nothing else in
this handoff is valid until it is fixed, because it changes every pixel value.**

### The bug

`sim_lens._area_resample` uses `.mean(axis=(1,3))` to go from the fine render grid
to the native grid. But `lenstronomy`'s `ImageModel.image()` returns **flux per
pixel**, not surface brightness. On a fine grid each pixel collects less flux, so
mean-pooling keeps the deficit instead of summing it back: brightness comes out
proportional to `1 / factor**2`.

Measured, same lens, same survey (HST), same output grid (`npix=400`, `s=0.05"`),
varying only the supersampling factor:

| `fine_scale` | factor | `f_gt.max()` | `sum(f_gt)*s^2` | `sum(f_gt)*factor^2*s^2` |
|---|---|---|---|---|
| 0.025 | 2 | 0.07014 | 0.05128 | 82.0 |
| 0.0125 | 4 | 0.01650 | 0.01281 | 82.0 |
| 0.01 | 5 | 0.01021 | 0.00820 | 82.0 |

The last column is constant, which is the proof: total flux on the fine grid is
invariant (as it must be), and `.mean()` divides it by `factor**2`.

### Why this is severe, not cosmetic

`choose_supersample_factor` picks the factor **adaptively per sample** from `s` and
`fwhm`. So:

1. **Image amplitude is arbitrary and varies sample to sample** by whatever factor
   was chosen — a scaling the physics does not contain.
2. **Amplitude correlates with pixel scale**, because the adaptive factor is a
   function of `s`. This hands the network a shortcut: it can read `s` off the
   overall brightness instead of learning scale-invariant structure. **This
   directly undermines the paper's central claim** — an apparently
   survey-invariant result could be the network exploiting a normalization
   artifact.
3. It is the origin of most of the ~11x cross-survey brightness spread, and of the
   ~27x non-monotonic total-flux variation flagged earlier.
4. It **inverts the Test Set 2 noise ordering** (see Step 9b): with fixed
   per-survey `bkg_rms`, arc SNR came out 0.86 for HST and 5.00 for DES — backwards.
   Space telescopes must have the better SNR.

Note this interacts with the adaptive-supersampling change (`fine_scale=None`
default): under the old fixed `fine_scale` the factor was near-constant, so the
error was a roughly uniform amplitude offset and largely hidden. Making the factor
adaptive turned a constant offset into a per-sample, `s`-correlated one.

### The fix

In `_area_resample`, **sum** instead of averaging:

```python
return img.reshape(m, factor, m, factor).sum(axis=(1, 3))   # flux-conserving
```

Then `sum(f_gt)` is the flux proxy and is invariant to `factor`; `sum(f_gt)*s**2`
is not the right quantity and should not be used as one.

Apply to `f_gt` and `g` (the sharp and blurred images). **Do NOT apply it to the
PSF stamp** — `h` must stay normalized to `sum(h) == 1` for `reblur` to be
flux-preserving. Verify `gaussian_psf_stamp` still satisfies that after the change.

### Verify (add as regression tests)

1. **Factor invariance:** same lens and survey rendered at several `fine_scale`
   values gives `sum(f_gt)` equal to within resampling error (<1e-3 relative).
   This is the test that would have caught the bug.
2. **Cross-survey flux invariance:** the same lens through all five surveys gives
   equal `sum(f_gt)` to within a few 1e-3. Per-pixel *peak* may still legitimately
   differ (coarse pixels collect more flux — that is real), but total flux must not.
3. **PSF normalization:** `abs(h.sum() - 1) < 1e-6` for every survey.
4. **Amplitude no longer leaks scale:** across ~200 training draws, correlation
   between `log(sum(f_gt))` and `log(s)` must be consistent with zero. Report it.
5. Re-measure and report the per-survey brightness table and the Test Set 2 SNR
   table (Step 9b) after the fix. **Confirm the SNR ordering is now
   HST/Roman best, DES/Rubin worst.** If it is not, stop and report.

Write the before/after numbers to `$ROOT/step0_flux_fix.md`.

**All noise, SNR, and metric numbers quoted elsewhere in this document were
measured with the bug present.** Treat them as order-of-magnitude context only and
re-measure after fixing. The `reblur` cost numbers (Step 1) are unaffected — they
depend on array sizes, not pixel values.

## Step 1 — Fix `reblur` cost (REQUIRED; changes no math)

`losses.reblur` convolves `f_hat` with the **full-FOV** PSF stamp (`npix x npix`),
so cost is `npix**4`, and it loops over the batch in Python. Measured arithmetic:

| survey | npix | MACs/sample | batch-8 GFLOP |
|---|---|---|---|
| DES | 76 | 3.3e7 | 0.5 |
| Euclid | 200 | 1.6e9 | 25.6 |
| HST | 400 | 2.6e10 | 409.6 |
| prior edge (s=0.04") | 500 | 6.3e10 | 1000.0 |

A single HST `reblur_loss` call on CPU did not finish and the process was
SIGKILLed. The PSF is a narrow Gaussian (sigma ~0.7 px at HST): the stamp is
almost entirely zeros.

**Fix, in `losses.reblur`:**
1. Crop the kernel to its actual support before `conv2d` — a centered
   `k x k` window with `k = 2*ceil(4*sigma_px) + 1` (odd, clipped to `npix`),
   `sigma_px = fwhm / 2.3548 / s`. Renormalize the crop to sum 1. If `fwhm`/`s`
   are not available inside the function, derive the support from `h` itself
   (smallest centered odd window retaining >= 1 - 1e-6 of `h.sum()`).
2. Replace the per-sample Python loop with one batched call. A batch is
   size-bucketed, so all items share `npix`; use grouped `conv2d`
   (`groups=B`, kernel `[B,1,k,k]`) or `unfold`.

Expected speedups from kernel cropping alone: DES 34x, Euclid 816x, HST 3265x.

**Verify:** on random inputs at each survey size, cropped-vs-full `reblur_loss`
must agree to `<1e-6` relative, and report the measured speedup. Add a regression
test asserting both the agreement and that HST completes in under a second.

---

## Step 2 — Fix SSIM saturation (REQUIRED; one line)

`losses.ssim` uses `C1 = 0.01**2`, `C2 = 0.03**2`, which assume data in [0,1].
The images peak at ~0.017 (HST), so `C1, C2` dominate the local variances and SSIM
saturates. Measured `ssim(blurred, sharp)` — a visibly degraded pair:

| survey | as-is | after peak-normalizing |
|---|---|---|
| HST | 0.999993 | 0.9995 |
| Euclid | 0.999952 | 0.9989 |
| Rubin | 0.994096 | 0.9780 |
| DES | 0.986732 | 0.9677 |

So `lambda_a = 0.5` currently contributes almost no gradient.

**Fix:** normalize both images by the SAME scalar before the SSIM computation —
use `f_gt.amax(dim=(-2,-1), keepdim=True)` per sample (clamped to a small floor).
Using a shared scalar keeps it a fair comparison; do NOT normalize each
image by its own peak, which would discard amplitude error.

**Do not change `lambda_a`.** Verify SSIM now separates blurred from sharp, and
that `ssim(x, x) == 1` still holds.

---

## Step 3 — Cross-survey metric comparability (REQUIRED for the paper table)

Two measured facts:
- Peak per-pixel brightness varies ~11x across surveys (Step 0), so raw `recon_l1`
  compared across surveys measures different physical quantities and will make
  coarse-pixel surveys look worse by construction.
- Images are sparse: only **2.7% (HST) to 4.0% (DES)** of pixels exceed 1% of peak.
  Global metrics are ~96% empty-sky agreement, which is trivially satisfied.

**In `test.py`'s `evaluate_survey`, report each reconstruction metric twice:**
1. **Normalized** — divide `f_hat` and `f_gt` by the per-sample `f_gt` peak before
   computing L1/SSIM, so surveys are comparable.
2. **Masked and global** — masked over `f_gt > 0.01 * f_gt.max()` (the arc region;
   this is the headline number) and over the full frame (the hallucination check).

Add columns: `recon_l1_norm_masked`, `recon_l1_norm_global`, `recon_ssim_norm`,
and keep the existing raw columns for continuity. Also log the mask filling
fraction per survey so the dilution is visible in the table.

---

## Step 4 — Trivial baselines (REQUIRED; this is what makes the table publishable)

A reviewer will ask what the numbers mean. Compute and report, per survey, the
same metrics for two null models:
- **zeros:** `f_hat = 0`
- **identity:** `f_hat = g` (do nothing — no deconvolution at all)

and for Einstein radius:
- **prior mean:** always predict `mean(theta_E)`. Its RMSE equals the prior
  standard deviation. Compute it from the actual prior draws and report it;
  **any `theta_rmse` at or above that value means the model learned nothing.**

Write these as extra rows in the metrics CSV (`survey`, `model` in {`trained`,
`zeros`, `identity`, `prior_mean`}). The deconvolution claim requires the trained
model to beat `identity` on the masked normalized metrics — not merely beat zeros.

---

## Step 5 — `cache_datasets.py`: freeze the data

Simulate once so every later run trains on byte-identical data.

- Reuse `sim_lens.sample_params` / `simulate_one` and `data.build_test_set_1`.
  Do NOT reimplement the simulator. Leave `fine_scale=None` (adaptive; exact scales).
- Store `f_gt` and `g` as **float16** (round-trip error 3.1e-4 relative to peak).
- **Do not store the PSF `h`** — it is a pure function of `(fwhm, s, npix)` and
  regenerates via `gaussian_psf_stamp` to 2.5e-9. Store the scalars; rebuild on load.
- `npix` varies per sample, so a single dense array will not work. Use one `.npz`
  per sample, or HDF5 with ragged storage, or group by `npix`. Justify the choice
  in a docstring; loading must be O(1) per item.
- Write `manifest.json` per split: seed, n, FOV, git commit, and per-sample `npix`
  and `scale`. `SizeBucketSampler` reads `it["g"].shape`, so the cached path needs
  `npix` available WITHOUT loading pixels — take it from the manifest.
- Same `--seed` must reproduce byte-identical output; add a test.
- Parallelize across the 10 cores with `multiprocessing`, seeding each worker from
  the master seed so results are independent of worker count.

Then generate into `$ROOT/data/`:
- train n=10000, val n=2000, seed 0, FOV 20" — expect ~1.4 h single-core, so
  ~10 min across 10 cores; ~2.2 GB on disk
- Test Set 1, n_lenses=200, seed 12345 — ~0.19 GB

Report actual wall time and on-disk size. Add `cache_dir=` to `SimLensDataset`,
keeping in-memory simulation as the fallback.

---

## Step 6 — Verify before spending GPU time

- `PYTHONPATH=src python -m pytest tests/ -q` — the existing 20 tests plus the new
  ones from Steps 1, 2, 5 must pass.
- Assert every cached sample's realized `s` matches its requested scale to <1e-5,
  and that the five surveys give five distinct `npix` (400/200/182/100/76).
- Load one cached sample and one freshly simulated sample at the same seed/params;
  assert `g` agrees to fp16 precision (~3e-4 relative to peak).

---

## Step 7 — GPU smoke test (gate before the real run)

- Print `torch.cuda.is_available()`, device name, total memory.
- **Force the worst case explicitly** — a random prior draw usually lands on a cheap
  grid and hides OOM. One forward+backward at `s=0.04"` (npix=500, 15,625 tokens),
  batch 8, dim=128, depth=4. Expect ~8 GB; report `torch.cuda.max_memory_allocated()`.
- Then two epochs of the real loop on cached data. Confirm the loss decreases and
  that a CSV and checkpoint appear. If OOM: reduce batch size first, and say so —
  do not change `patch` or `fov` silently.
- Add `num_workers=4, pin_memory=True, persistent_workers=True` to both dataloaders
  **only now that data is cached** (with the in-memory path this forks the whole
  dataset per worker). Report the epoch-time change.

---

## Step 8 — The real run

Full loss, cross-attention on, joint mode — i.e. all defaults:

```
PYTHONPATH=src python -m survey_lens.train \
  --name run1_full --epochs 60 --batch-size 8 \
  --n-train 10000 --n-val 2000 --survey-val \
  --loss-mode joint --seed 0
```

- Point Lightning's logger and checkpoint directory at `$ROOT/runs/run1_full/`.
- `--survey-val` validates against the fixed survey configs, so cross-survey
  behaviour is visible during training rather than only at the end.
- Watch `val/reblur`, `val/ssim_term`, `val/theta_rmse` separately. If `ssim_term`
  is still flat after Step 2, stop and report — the fix did not take.
- Report wall-clock per epoch and peak GPU memory. If 60 epochs is not converging,
  report the curve rather than silently extending.

---

## Step 9 — Evaluate on Test Set 1

```
PYTHONPATH=src python -m survey_lens.test \
  --ckpt $ROOT/runs/run1_full/.../best.ckpt \
  --test-set 1 --n-lenses 200 --device cuda \
  --out-prefix $ROOT/figures/run1_test_set1
```

Note `test.py` and `diagnose.py` default to `--device cpu` — pass `--device cuda`.
`test.py` silently falls back to RANDOM weights if the checkpoint path is missing;
assert the checkpoint loaded before trusting any number.

Then `diagnose.py --ckpt <same> --n-examples 3 --device cuda` for the per-survey
image panels (input / reconstruction / truth / residual / reblur residual).

**Deliverables into `$ROOT/figures/`:**
1. `run1_test_set1_metrics.csv` — rows = survey x model {trained, zeros, identity,
   prior_mean}; columns = theta_rmse/bias/scatter, normalized+masked and global
   recon L1 and SSIM, reblur MSE, mask fraction.
2. A pred-vs-true `theta_E` scatter per survey, with the prior-mean RMSE drawn as a
   reference line.
3. Reconstruction panels for 3 example lenses across all five surveys.
4. `run1_summary.md` — the table, plus explicit answers to: does `theta_rmse` beat
   the prior-mean baseline in every survey? does the reconstruction beat `identity`
   on masked normalized metrics in every survey? how much does performance vary
   across the 5.3x pixel-scale range (this is the survey-invariance claim)?

---

---

---

## Step 9b — Evaluate on Test Set 2 (noise shift) — SAME checkpoint, no retraining

Training data is noise-free, so Test Set 2 is a genuine **distribution shift**, not
a harder test set. This is the sim-to-real evidence the venue cares about: an
imperfect simulator (no noise) evaluated against more realistic data. Evaluate the
**same `run1_full` checkpoint** — do not retrain, do not fine-tune, do not add noise
augmentation. If you do, it stops being a robustness result.

`test.py` already supports this (`--test-set 2`, `build_test_set_2`). Noise is applied
to the model input `g` only; `f_gt` and the PSF `h` stay clean.

```
PYTHONPATH=src python -m survey_lens.test \
  --ckpt $ROOT/runs/run1_full/.../best.ckpt \
  --test-set 2 --n-lenses 200 --device cuda \
  --out-prefix $ROOT/figures/run1_test_set2
```

Use the **same seed (12345) and the same lens population** as Test Set 1, so the
two sets are paired and per-survey differences are attributable to noise alone.
Report metrics as **TS1 -> TS2 deltas** per survey, not just TS2 absolutes.

### The reblur diagnostic has a noise floor — do not report it naively

On noisy data a *perfect* reconstruction cannot drive `reblur_mse` to zero: the
residual `(f_hat * h) - g` still contains the noise in `g`. The floor is the mean
noise variance. Measured per survey (**with the Step 0 bug present — recompute
after fixing**):

| survey | reblur floor (global) | floor (arc mask) |
|---|---|---|
| HST | 6.4e-7 | 6.6e-7 |
| Euclid | 1.2e-6 | 1.3e-6 |
| Roman | 8.1e-7 | 8.5e-7 |
| Rubin | 2.6e-6 | 3.6e-6 |
| DES | 4.1e-6 | 6.1e-6 |

Compute the floor analytically as `mean(max(g,0)/gain + bkg_rms**2)` per survey and
**plot it as a reference line on every reblur figure.** Interpretation:
- `reblur_mse` at the floor = as good as possible.
- **`reblur_mse` well BELOW the floor = the model is fitting noise** (over-fitting
  the noise realization into `f_hat`). Flag this loudly if it happens; it is a
  failure mode, not a success.
- Report `reblur_mse / floor` as a dimensionless ratio so surveys are comparable —
  the raw floor spans 6.4x across surveys.

### Also report

- `theta_E` RMSE per survey on TS2, against the **same prior-mean baseline** — the
  baseline is unchanged by noise, so it is the honest reference for how much
  degradation matters.
- All Step 4 baselines recomputed on TS2. `identity` now means "return the noisy
  input", so on noisy data the trained model should beat it by *more*, since it
  denoises as well as deconvolves. If it does not, that is the headline finding
  and must be reported plainly.
- Masked + normalized recon metrics (Step 3) and top-octave `hf_power_err` (Step 10).
  Expect high-frequency recovery to degrade most — noise dominates exactly the
  frequencies deconvolution amplifies. Quantify it rather than asserting it.
- The realized per-survey **arc SNR and peak SNR** actually present in TS2, so the
  degradation can be read against how hard each survey's data really is.

### Honest-reporting requirement

Do not tune anything on Test Set 2. If TS2 performance is poor, report it — "the
noise-free simulator does not transfer without augmentation" is a legitimate and
publishable Sim2Science result, and the reblur residual is the diagnostic that
demonstrates it. Concealing it by adding noise to training would answer a different
question than the one this run asks.

## Step 10 — Paper figures (produce these, not ad-hoc plots)

Write ONE script, `src/survey_lens/make_paper_figures.py`, that reads the frozen
Test Set 1 results and emits every figure below. It must be re-runnable end-to-end
from the checkpoint + cached data with no manual steps, because the numbers will
change once and you do not want to redraw by hand.

### Output contract (applies to every figure)

- Save **both** `figures/paper/figN_<slug>.pdf` (vector — this is what LaTeX uses)
  and `.png` at 200 dpi (for quick viewing) under `$ROOT/figures/paper/`.
- **Width:** NeurIPS text width is 5.5 in. Full-width figures = 5.5 in, half-width
  = 2.7 in. Do not draw at 12 in and scale down in LaTeX — fonts become illegible.
  Set the figure size so that **no text is below 6 pt at final size**.
- Alongside each figure write `figN_<slug>_data.csv` containing exactly the numbers
  plotted. The paper text must quote values from these CSVs, never re-read off a plot.
- Every panel: axis labels with **units** (arcsec, arcsec/px), a title stating the
  claim, and a caption sentence written into `figures/paper/captions.md`.
- Colour: use ONE consistent survey colour map across all figures (HST -> DES
  ordered by pixel scale) and reuse it everywhere. Declare it once at module top.
- Do NOT use a rainbow/jet colormap for images; use a perceptually uniform one.
  Images of the same lens must share vmin/vmax so panels are comparable by eye;
  residual panels use a diverging map centered on zero, with the limits stated.
- Run the geometric self-check before saving: render, then assert no text-text or
  text-spine overlap and that all text is inside the figure bounds. Fix and re-render
  if it fails; do not ship an overlapping figure.

### Required metric that does not exist yet

Implement **high-frequency power-spectrum error** and add it to `test.py`'s metrics
before drawing figures. Global PSNR/L1 is dominated by low frequencies and will hide
the deconvolution effect — the whole point of the paper.

- For each sample: 2D FFT of `f_hat` and `f_gt`, azimuthally average the power into
  radial bins in units of the native Nyquist frequency.
- Report `hf_power_err` = relative error integrated over the **top octave**
  (0.5 to 1.0 x Nyquist), plus the full radial profile for plotting.
- Compute it on the **normalized, masked** images (Step 3), and report it for the
  `identity` and `zeros` baselines too (Step 4). Deconvolution means beating
  `identity` in the top octave specifically.

### Fig 1 (hero) — cross-survey reconstruction grid

**Claim:** one network, five telescopes, five different pixel scales, no resampling.
This is the paper's most arresting asset and belongs on page 1.

- Rows = the five surveys ordered by pixel scale (HST 0.05" -> DES 0.263").
  Columns = `input g` | `reconstruction f_hat` | `truth f_gt` | `residual (f_hat - f_gt)`.
- Pick **one** lens. Choose it by median masked reconstruction error across surveys —
  NOT the best case. State in the caption how it was chosen. Save a second version
  for the appendix with 3 more lenses (including the worst case).
- Crop to a 4" box around the lens center so the arc fills the panel; the full 20"
  field is mostly empty sky (only 2.7-4.0% of pixels exceed 1% of peak).
- **Annotate each row** with `s` in arcsec/px, PSF FWHM, and the native grid size —
  the varying npix (400/200/182/100/76) IS the result, so make it visible. Draw a
  1" scale bar on each row to show the physical size is constant while pixels change.

### Fig 2 — architecture schematic

Not data-dependent; assemble by hand and check in. Source material is already in
`figures/source_slides/`: `slide_rope_intuition.png` (why rotation encodes relative
offset), `slide_conv_deconv.png` (forward blur / deconvolution), `slide_loss_terms.png`
(why L1 alone is insufficient).

Show: image + PSF stamp -> patch embedding -> RoPE positions **in arcsec, not pixels**
-> the three attention paths (image self, PSF self, image<-PSF cross) -> two heads.
The pixel-scale scalar `s` entering the positional encoding must be explicit — that
is the invariance mechanism. Keep it schematic; do not redraw the RoPE maths.

### Fig 3 — quantitative survey invariance

**Claim:** accuracy is flat across a 5.3x pixel-scale range.

- Panel (a): `theta_E` predicted vs true, one colour per survey, y=x line, per-survey
  RMSE in the legend. Draw the **prior-mean RMSE as a horizontal reference band** or
  annotate it — a reviewer must see the model beats the null.
- Panel (b): `theta_E` RMSE vs pixel scale (log x), with the `prior_mean` baseline as
  a dashed line. Flatness here is the survey-invariance claim; if it is not flat, say
  so in the caption and report the trend rather than hiding it.
- Panel (c): masked normalized reconstruction SSIM (or L1) vs pixel scale, with
  `identity` and `zeros` baselines drawn as dashed lines.
- Use error bars: bootstrap the 200 test lenses (resample with replacement, 1000 draws)
  and plot 16-84th percentile intervals. With n=200 the per-survey scatter is not
  negligible and a bare point estimate will invite the question.

### Fig 4 — deconvolution quality vs PSF

**Claim:** the network is doing real deconvolution, not just denoising or smoothing.

- Panel (a): radial power-spectrum profiles for `f_gt`, `g` (input), `f_hat`, and the
  `identity` baseline, for one survey — showing `f_hat` recovering power in the top
  octave where `g` has lost it. Mark the Nyquist frequency and the PSF cutoff.
- Panel (b): top-octave `hf_power_err` vs PSF FWHM in **units of native pixels**
  (`sigma_px = fwhm / 2.3548 / s`), one point per survey, with the `identity` baseline.
  This is the honest difficulty axis — Euclid and Roman PSFs are under-sampled on
  their own grids (bug 3 in CLAUDE.md), so expect them to be harder; note it.
- Panel (c): reblur residual `(f_hat * h) - g` for one lens per survey. Small and
  structureless = good. This is the diagnostic that transfers to real data with no
  ground truth, so it is the sim2real hook — label it as such.

### Fig 5 — sim-to-real robustness under noise shift (Test Set 2)

**Claim:** a model trained on a noise-free simulator degrades gracefully (or does
not) on realistically noisy data, and the reblur residual detects it **without
ground truth** — so the same diagnostic works on real images.

- Panel (a): `theta_E` RMSE per survey, TS1 vs TS2 as paired bars, with the
  prior-mean baseline as a dashed line. Same lens population, so the pairing is
  meaningful — annotate the delta.
- Panel (b): `reblur_mse / noise_floor` per survey (dimensionless), with a line at
  1.0 marking the floor. Annotate that below 1.0 means fitting noise. Use the ratio,
  not the raw MSE — the floor spans 6.4x across surveys.
- Panel (c): reblur residual maps for one lens per survey on TS2, shared diverging
  colour scale. Structureless = good; residual that still traces the arc means the
  deconvolution failed there. This panel is the sim2real hook — it is the only
  diagnostic that transfers to real data, so caption it as such.
- Panel (d): top-octave `hf_power_err`, TS1 vs TS2 per survey, showing how much
  high-frequency recovery the noise costs.

Order figures for the paper as: Fig 1 hero grid, Fig 2 architecture, Fig 3
invariance, Fig 4 deconvolution, **Fig 5 noise shift**. Fig 5 carries the venue's
theme, so if space forces a cut, cut from Fig 4's panels rather than Fig 5's.

### Deferred (do NOT draw in Run 1)

Ablation bar charts (cross-attention on/off, loss modes, native vs resampled) and
the real-data Test Set 3 panel — they need runs that are out of scope. Leave
placeholders in `captions.md` so the paper skeleton is complete.

### Deliverable

`$ROOT/figures/paper/` containing figs 1, 3, 4 as PDF+PNG, their `_data.csv` files,
`captions.md`, and a `figure_manifest.json` recording for each figure: the checkpoint
path, the data manifest hash, the git commit, and the script version. A figure whose
provenance cannot be reconstructed does not go in the paper.

## Definition of done

- All tests pass; Steps 0-4 fixes committed with the verification numbers in the
  commit messages.
- Frozen dataset under `$ROOT/data/` with manifests.
- One trained checkpoint, cross-survey metrics CSV including all baselines
  AND the high-frequency power-spectrum error metric.
- Test Set 2 evaluated on the SAME checkpoint, with TS1->TS2 deltas and
  `reblur_mse / noise_floor` per survey.
- Paper figures 1, 3, 4, 5 under `$ROOT/figures/paper/` as PDF+PNG with per-figure
  CSVs, `captions.md`, and `figure_manifest.json`.
- `run1_summary.md` answers the three questions above with numbers.

**Do not proceed to any ablation or to Test Set 3 (real data).** Report back first.

Report Step 0's before/after numbers before anything else — if the flux fix changes
the picture, later steps may need rethinking.
