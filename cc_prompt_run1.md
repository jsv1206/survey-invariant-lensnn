# Paste-ready prompts for Claude Code — Run 1

Run from the repo root (`/Users/jarugula/Research/survey-invariant-lensnn`, or wherever
you clone it on the GPU box) so `CLAUDE.md` auto-loads.

---

## PROMPT 1 — Step 0 only (paste this first)

```
Read HANDOFF_run1.md. That document is the spec for this work; CLAUDE.md (already in
your context) carries the physics conventions and the known-bug list. Read both fully
before writing any code.

Do STEP 0 ONLY, then stop and report. Do not proceed to Step 1.

Step 0 is a confirmed flux-normalization bug in sim_lens._area_resample: it averages
the fine render grid where it must sum, because lenstronomy returns flux per pixel.
The fix is one line, but it changes every pixel value in the project, so I want to
see the before/after numbers before anything else runs.

Deliver:
1. The one-line fix, plus the five regression tests listed in Step 0's "Verify"
   subsection, in a new tests/test_flux_normalization.py.
2. $ROOT/step0_flux_fix.md with the before/after tables actually measured, not
   copied from the handoff.
3. The re-measured per-survey brightness table and the Test Set 2 SNR table.
   CONFIRM the SNR ordering is now HST/Roman best and Rubin/DES worst. If it is
   not, stop and tell me — something deeper is wrong in the renderer.
4. Confirmation that the PSF stamp still satisfies sum(h) == 1 (the fix must NOT
   be applied to h).
5. `PYTHONPATH=src python -m pytest tests/ -q` fully green, including the 20
   pre-existing tests. If the fix breaks an existing test, show me the failure and
   your diagnosis before changing that test.

Set ROOT=/deepskieslab/stronglensing/survey-invariant-lensnn and create
$ROOT/{data,runs,figures} if absent. Write only outputs there; keep the git repo clean.

Working agreements for this whole task:
- Every metric number quoted in HANDOFF_run1.md was measured WITH the bug present.
  Treat them as order-of-magnitude context and re-measure. The Step 1 reblur cost
  numbers are the exception: they depend on array sizes, not pixel values.
- Do not change simulator physics, parameter priors, choose_supersample_factor, or
  any loss weight (lambda_a/b/c/d). Do not resample surveys onto a common grid.
- Report measured numbers — wall time, disk, GPU peak. Never estimate and never
  present an estimate as a measurement.
- Stop at the first failure and report it rather than working around it.
- Commit with the verification numbers in the commit message.
```

---

## PROMPT 2 — Steps 1-7 (only after you have read the Step 0 report)

```
Step 0 is accepted. Now do Steps 1 through 7 of HANDOFF_run1.md, in order.

Gates — stop and report at each:
- After Step 4: show me the trivial-baseline numbers (zeros / identity / prior_mean)
  before any GPU time is spent. If the identity baseline is hard to beat, I want to
  know now.
- After Step 5: report actual wall time and on-disk size of the frozen dataset, and
  confirm the same seed reproduces byte-identical output.
- After Step 7: report the forced worst-case GPU memory (s=0.04", npix=500,
  batch 8) against the 20 GB budget, and confirm the 2-epoch smoke loss descends.

Do not launch the real training run (Step 8). Same working agreements as before.
```

---

## PROMPT 3 — Steps 8, 9, 9b, 10 (the run and the figures)

```
Smoke test is accepted. Run Steps 8, 9, 9b and 10 of HANDOFF_run1.md.

Reminders that are easy to get wrong:
- Step 9b (Test Set 2) uses the SAME checkpoint as Step 9. No retraining, no
  fine-tuning, no noise augmentation — otherwise it stops being a robustness result.
- test.py and diagnose.py default to --device cpu. Pass --device cuda.
- test.py silently falls back to RANDOM weights if the checkpoint path is wrong.
  Assert the checkpoint loaded before trusting any number.
- Report reblur_mse / noise_floor as a ratio, not raw MSE. Below 1.0 means the model
  is fitting noise — flag it loudly, it is a failure mode and not the best number
  in the table.
- Implement the top-octave high-frequency power-spectrum error (Step 10) before
  drawing figures. Global PSNR hides the deconvolution effect, which is the point
  of the paper.
- Figures go through ONE re-runnable script, src/survey_lens/make_paper_figures.py.
  Honour the output contract: PDF+PNG, 5.5in text width, no text below 6pt, a
  _data.csv per figure, and figure_manifest.json with checkpoint + data hash + commit.

Then write $ROOT/figures/run1_summary.md answering, with numbers:
1. Does theta_E RMSE beat the prior-mean baseline in every survey?
2. Does the reconstruction beat the identity baseline on masked normalized metrics,
   and specifically in the top frequency octave, in every survey?
3. How much does performance vary across the 5.3x pixel-scale range? (This is the
   survey-invariance claim.)
4. How much does TS1 -> TS2 cost, per survey, and is any survey's reblur_mse below
   its noise floor?

Do not proceed to any ablation or to Test Set 3. Report back.
```

---

## Notes

- Commit `HANDOFF_run1.md`, the modified `CLAUDE.md`, and `figures/source_slides/`
  before starting, so Claude Code's own commits stay clean and the handoff is
  reproducible from the repo alone.
- `figures/source_slides/` holds the five explanatory figures for Fig 2's assembly
  (`slide_rope_intuition.png`, `slide_conv_deconv.png`, `slide_loss_terms.png`).
- `dataset_spec.md` and `plan_sim2science.md` are background reading, referenced from
  `CLAUDE.md` — do not force them into context.
- Splitting into three prompts is deliberate: Step 0 changes every pixel value, and
  Step 4's baselines can invalidate the plan. Both deserve a human read before
  compute is spent.
