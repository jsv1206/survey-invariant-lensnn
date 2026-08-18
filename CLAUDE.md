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

4. **Token counts vary 30× and dominate the GPU budget.** At `patch=4`, FOV 20″:
   HST npix=400 → **10,000 image + 10,000 PSF tokens** (~1e8 attention scores/head);
   Euclid 2,500; Roman 2,025; Rubin 625; DES 361. HST may OOM without memory-efficient attention.
   Options: FOV 10″ (4× fewer tokens), larger patch at fine scales, or confirm PyTorch SDPA
   flash path is active.
   **Always smoke-test at HST scale explicitly** — a random prior draw usually yields a
   cheap grid and hides this.

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
