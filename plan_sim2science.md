# Sim2Science 2026 — Survey-Invariant Lens Network: 1-Week Analysis Plan & Paper Outline

**Deadline:** Aug 29, 2026 (23:59 AoE) · **Format:** 5 pages, NeurIPS template, refs/appendix excluded
**Workshop theme:** leveraging *imperfect* scientific simulators against real-world data

## Repo state (verified Aug 14)
- FULLY SCAFFOLDED & WIRED: model.py, losses.py, data.py, lit_model.py, train.py, test.py,
  diagnose.py, sim_lens.py, surveys.py. Tests pass (smoke, RoPE coverage).
- Three ablation axes are flag-driven:
  (1) PSF cross-attention  -> --no-cross-attention
  (2) joint vs single-task -> --loss-mode {joint,recon_only,reg_only}
  (3) native vs resampled  -> data-side switch
- NOT DONE: no cached dataset, no checkpoints, no runs/. Nothing trained yet.
- Dataset currently generated in-memory at construction (seeded) => re-simulates every run.

## Framing (do not miss)
Headline contribution for THIS venue = the sim-to-real angle:
  * reblur consistency loss  = self-supervised sim2real bridge (no GT on real data)
  * survey-invariance        = robustness to simulator mis-specification across telescopes
Architecture ablations prove the method works; Test Set 2 (noise shift) + Test Set 3
(real cutouts, qualitative) prove it's a Sim2Science paper.

## Day 1 — SUBSTRATE (the first thing)
1. Freeze ONE dataset to disk (run simulator ONCE): cache train/val + Test Sets 1&2 as .npz.
   => every ablation trains/evals on byte-identical data (controlled comparison).
2. Refactor DataModule to load from cache.
3. GPU smoke run: full model, --n-train 256 --epochs 2. Confirm curves descend, recon
   is ring-shaped, theta_E tracks. DO NOT launch ablations until clean.

## Days 2-3 — Core result + architecture ablations (run concurrently on GPUs)
- full        : proposed (cross-attn + joint + native)
- no_cross    : --no-cross-attention                (experiment 1)
- recon_only / reg_only : --loss-mode               (experiment 2)
- resampled   : common-grid data switch             (experiment 3)
- no_ssim_fft : drop SSIM+FFT loss terms            (loss ablation, if slot free)

## Day 4 — Sim-to-real robustness (Sim2Science core)
- Evaluate all checkpoints on Test Set 2 (per-survey noise shift).
- Reblur-residual diagnostic.
- Pull 2-3 real cutouts (COSMOS HST / DES / Euclid Q1) for qualitative Test Set 3 figure.

## Day 5 — Evaluation + metrics
- test.py / diagnose.py across 5 surveys x all checkpoints.
- Metrics: theta_E RMSE per survey; recon PSNR/SSIM + HIGH-FREQ power-spectrum error
  (top-octave; global PSNR hides the deconvolution effect); reblur residual.

## Day 6 — Figures + result tables (freeze numbers)

## Days 7+ (Aug 22-29) — Writing
Draft -> internal read -> tighten to 5 pp -> anonymize -> submit.
Buffer: all compute front-loaded into Days 1-5.

## Paper outline (5 pp)
1. Title/Abstract  — lead with sim2real hook
2. Intro (3/4 p)   — imperfect per-survey simulators; 4 contribution bullets
3. Method (1.5 p)  — arch (patch+RoPE arcsec, PSF cross-attn, dual heads) + loss breakdown;
                     coordinate-mapping & RoPE figures here
4. Experiments (2 p) — setup; cross-survey theta_E; 3 ablations; noise shift (TS2);
                     real-data qualitative (TS3) w/ reblur residual
5. Discussion/Limitations/Conclusion (1/2 p) — Gaussian-PSF & spatial-invariance assumptions
6. Appendix — hyperparams, full per-survey tables, extra reconstructions

## Four figures that carry the paper
1. Architecture schematic (reuse coordinate-mapping / RoPE slides)
2. Cross-survey reconstruction grid (one lens through 5 telescopes: blurred -> recovered)
3. Ablation bar charts (theta_E RMSE + high-freq error)
4. Real-data panel with reblur residual
