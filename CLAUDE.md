# survey-invariant-lensnn

Survey-invariant transformer embedding for strong gravitational lens images:
robust to PSF blurring and variable pixel scale across HST/Euclid/Roman/Rubin/DES.
Dual heads: image reconstruction (deconvolution) + Einstein-radius regression.

Read `README.md` for architecture and `src/survey_lens/` for the implementation.
Deeper references, read on demand (do NOT auto-load):
- `dataset_spec.md` — verified dataset spec + the four known bugs below
- `plan_sim2science.md` — **the per-figure spec for BOTH venues** + paper outline

## Current objective — TWO papers, one model (updated Aug 23 2026)

The work now targets **two** workshop submissions from the **same checkpoint and the same
numbers** (Run 3 joint arm, FOV 12″). They differ in framing and in exactly **one**
main-text figure. `plan_sim2science.md` holds the full per-figure spec; read it before
touching any figure.

| | **Sim2Science** | **Representations** |
|---|---|---|
| deadline | **Aug 29 2026** | **TBD — not yet known** |
| main text | figs 1–4 + **noise shift** | figs 1–4 + **loss ablation** |
| appendix | loss ablation, A1 | noise shift, A1 |

**Sim2Science** theme is *leveraging imperfect simulators against real data*: the headline
is the sim-to-real angle — the self-supervised reblur consistency loss (works without
ground truth on real data) and survey-invariance (robustness to per-instrument simulator
mis-specification). Weight the noise-shift test (Test Set 2) and the real-data demo
(Test Set 3) above the architecture ablations.

**Representations** theme is symmetry/geometry in learned representations: the headline is
the *representation* — one encoder invariant to pixel scale and PSF across a 5.3× range,
with **RoPE positions in arcsec rather than pixel indices** as the mechanism. There the
loss ablation is promoted to main text, and its **asymmetry** is the result: the
reconstruction objective helps θ_E by 16% (0.0216″ vs 0.0256″), while the θ_E objective
costs deconvolution 94% (0.037 vs 0.019). The two objectives are not interchangeable.

**Figure slugs are frozen; figure NUMBERS are per-venue.** `fig5_noise_shift` is Fig 5 for
Sim2Science and Fig A2 for Representations; `fig6_loss_ablation` is the reverse. The digit
inside a slug reflects Sim2Science ordering only. Never rename or regenerate an artifact to
match a venue's numbering — use `\label`/`\ref` in LaTeX.

Test Set 3 stays **qualitative** for this deadline: 2–3 real lenses, reblur residual as
the diagnostic. Do not scope a full real-data benchmark.

## State as of Aug 20 2026 (Run 2 — SUPERSEDED by Run 3, kept as provenance)

**Run 2 is trained (80 epochs) and fully evaluated on Test Set 1 AND Test Set 2**, and
the paper figures exist. Run 1 is superseded — its reconstruction conclusions were
retracted (measured with masked L1/SSIM, the wrong metric) and its checkpoint no longer
loads against the current `model.py`. Everything durable lives under
`/deepskieslab/stronglensing/survey-invariant-lensnn/` (`data/`, `runs/`, `figures/`);
see `RESTORE.md`.

Run 2 headline, top-octave complex error (`f_hat=0` scores exactly 1.0):

| survey | Run 1 | Run 2 TS1 | Run 2 TS2 | identity | Wiener |
|---|---|---|---|---|---|
| HST | 2.705 | 0.113 | **1.125** | 0.589 | 0.026 |
| Euclid | 1.354 | 0.093 | 0.111 | 0.411 | 0.036 |
| Roman | 0.949 | 0.209 | 0.204 | 0.166 | 0.093 |
| Rubin | 0.940 | 0.050 | 0.075 | 0.961 | 0.315 |
| DES | 0.937 | 0.084 | 0.113 | 0.959 | 0.328 |

Run 2 beats `identity` on TS1 at four of five surveys (Roman is the exception, and
that is bug 3). **Under the TS2 noise shift HST collapses to 1.125 — worse than
emitting nothing** — while the other four degrade gracefully; theta_E RMSE degrades
1.0x-8.5x but stays far below the prior-mean null (0.618") everywhere. Wiener with the
known PSF still beats the network 3-4x on noise-free sharp surveys; that gap is the
open question for the paper's deconvolution framing.

Paper figures: `figures/paper/` (mirrored to `$ROOT/figures/paper/`), produced by
`survey_lens/make_paper_figures.py` — one re-runnable script, each figure with its
`_data.csv`, `captions.md` and `figure_manifest.json`.

**THREE scripts now write into `figures/run3/paper/` and their outputs must not collide.**
`make_paper_figures.py` owns the generic names (`captions.md`, `figure_manifest.json`);
`make_ablation_figures.py` writes `captions_ablation.md` / `figure_manifest_ablation.json`;
`make_fig2_architecture.py` writes `figure_manifest_fig2.json`. Before the suffixes,
whichever script ran second **silently destroyed** the other's captions and provenance
manifest. Any new figure script added to this directory must take a suffixed name too.

`fig2_architecture` is drawn programmatically (`make_fig2_architecture.py`), NOT hand-
assembled as HANDOFF Step 10 proposed — the constants in its boxes are read from
`model.py` and its token counts from `surveys.py`, so the drawing cannot drift from the
code. The slides in `figures/source_slides/` are talk material, not this figure.

## Run 3 (Aug 21 2026) — THE PAPER'S RESULT (supersedes Run 2)

**The paper leads with Run 3.** Run 2 (FOV 20″, `figures/paper/`) is kept as provenance
and is NOT deleted, regenerated or moved; its manifest legitimately records
`script_version 1.0.0`. Run 3's figures are `figures/run3/paper/` (figs 1, 3, 4, 5, A1)
plus `figures/run3/fig6_loss_ablation`, mirrored under `$ROOT/figures/run3/`.

### The Wiener comparison — corrected, and NOT a clean sweep

`RESTORE.md` flagged "does the network beat classical Wiener?" as the question deciding
the deconvolution framing. The answer is **yes on the well-sampled surveys, no on the
seeing-limited ones, and yes almost everywhere once there is noise.**

Top-octave complex error, trained vs Wiener with the KNOWN PSF, K swept over
`1e+1 … 1e-14` and selected **on this same metric** (all optima interior, none
edge-pinned):

| survey | TS1 trained | TS1 Wiener | verdict | TS2 trained | TS2 Wiener | verdict |
|---|---|---|---|---|---|---|
| HST | 0.0168 | 0.0258 | net 1.53× | 1.029 | 0.914 | Wiener 1.13× (both fail, ≳1.0) |
| Euclid | 0.0098 | 0.0368 | net 3.74× | 0.039 | 0.309 | net 7.84× |
| Roman | 0.0082 | 0.0520 | net 6.32× | 0.020 | 0.079 | net 3.98× |
| Rubin | 0.0641 | 0.0573 | **Wiener 1.12×** | 0.090 | 0.782 | net 8.65× |
| DES | 0.0881 | 0.0782 | **Wiener 1.13×** | 0.113 | 0.728 | net 6.42× |

The defensible claim is about **noise robustness**, not raw noise-free inversion: an
oracle-tuned Wiener is excellent on clean data (as theory says it must be) and collapses
once its noise assumption is violated. Do NOT write "beats Wiener at every survey" — that
was true only against an under-tuned Wiener.

**The trap that produced the wrong claim, now guarded.** `baseline_rows` selected K by
`recon_l1_norm_masked` and then reported that K's `hf_complex_err` — judging the baseline
on a metric it was never tuned for. Fixing it moved Wiener at Rubin 0.315 → 0.057 and at
DES 0.328 → 0.078, flipping both surveys. The grid also has to bracket the optimum in
BOTH directions: noise-free data pushes K toward 0 (naive inverse filtering, no noise to
amplify) while noisy data wants heavy regularization. `tests/test_baselines.py` now pins
both rules — selected-on-the-quoted-metric, and optimum interior to the grid.

**General convention this generalizes to:** a baseline quoted against a metric must have
been tuned on that metric, and its hyperparameter optimum must be interior to the search
grid. An under-tuned baseline flatters us, and it is the first thing a referee checks.

### Limitation to state in the paper

The network learned the simulator's *exact* generative distribution (Sérsic source,
EPL + shear). Beating a generic-prior filter on data drawn from the prior it trained on
is easier than on real sky; Wiener's weaker assumptions transfer more honestly. At a
workshop about *imperfect simulators* this objection will be raised if we do not raise it.

## Run 3 configuration — loss-mode ablation at FOV 12″

Ablation axis 2, finally run: three arms of the SAME architecture on the SAME frozen
12″ dataset, differing only in `--loss-mode` (`joint` / `recon_only` / `reg_only`),
80-epoch cosine, seed 0. Run 2 could NOT serve as the joint arm — it was trained
before the bug-11 reblur fix, so it differs in code as well as loss mode. Outputs are
self-contained in `figures/run3/` (`make_ablation_figures.py`); **Run 2 at 20″ is
frozen and stays the paper's headline.**

**FOV 12″ instead of 20″** buys 2.78x fewer tokens (HST 10,000 -> 3,600; prior edge
15,625 -> 5,625) and ~7.7x less attention compute, at a measured cost of **0.088%
worst-case flux** (median ~0.0001%). It is NOT a memory fix — 10.73 GB of 20.9 GB
already fits. Why 12″ and not smaller, all measured on the frozen test set:

| | 20″ | **12″** | 10″ | 5″ |
|---|---|---|---|---|
| worst-case flux discarded | — | **0.088%** | 0.24% | **25–46%** |
| DES `mask_frac` vs the `0.01<f<0.10` test bound | 0.0277 ✅ | **0.0756 ✅** | 0.1108 ❌ | 0.363 ❌ |
| ±4″ metric crop (`metrics.py`) | fits | **fits, 2″ margin** | 1″ margin | **silently clamps to frame** |
| FFT `reblur` residual vs 20″ | — | **unchanged (4/5 bit-for-bit)** | unchanged | **1.16e-1, worse than bug 11** |
| RoPE bands in field | 2/4 ✅ | **2/4 ✅** | 2/4 ✅ | 1/4 ❌ |

5″ fails four independent ways: it clips real arc flux for the upper third of the
θ_E prior (θ_E ~ U(0.5,2.5)″, so a 2.5″ ring is 5″ across), it breaks the circular-FFT
`reblur`'s empty-sky precondition, it silently redefines the top-octave metric, and it
starves the RoPE bank. **Do not go below 12″.**

Two consequences of 12″ that are handled but easy to trip over again:
- **npix parity is FOV-dependent.** At 12″ the sizes are 240/120/**109**/60/46 —
  Roman is ODD. The FFT `reblur` is exact either way, but any doc asserting "npix is
  even for every survey" is FOV-specific. Measured residuals are unchanged from 20″
  for HST/Euclid/Rubin/DES; Roman 1.83e-3 -> 2.07e-3 (its bug-3 aliasing).
- **`test.py`/`diagnose.py`/`make_ablation_figures.py` must be passed `--fov 12.0`.**
  The cache guard below cannot help: those scripts build test sets on the fly, and the
  checkpoint does not record its FOV (`fov` is a DataModule arg, not a model hparam).

**Cache FOV guard (new).** `data.load_manifest_items(cache_dir, expect_fov=...)` now
raises on mismatch. The .npz files hold only `f_gt`/`g` — no field of view, no
coordinate metadata — so before this, `train.py --fov 12 --cache-dir <20″ cache>`
trained on 20″ data and reported it as a 12″ run. With caches at two FOVs now
coexisting (`$ROOT/data/` at 20″, `$ROOT/data_fov12/` at 12″) that is a
wrong-experiment bug, so it is a hard error. Tests in `tests/test_cache_datasets.py`.

## Ablation axis 1 — PSF cross-attention (Aug 24 2026): the Representations headline

One flag from the joint arm (`run3_nocross_fov12`; verified the two checkpoints differ in
`use_cross_attention` and **nothing else**). Cross-attention is the ONLY path from the PSF
to either head — with it off, feeding a 9x-wider WRONG PSF at inference changes the output
by **exactly 0.00e+00**. Test Set 1, top-octave complex error:

| survey | sigma_px | joint | no-cross | cost | 68% CI |
|---|---|---|---|---|---|
| Roman | 0.46 | 0.0082 | 0.0822 | **9.98x** | [6.92, 13.18] |
| Euclid | 0.68 | 0.0098 | 0.0944 | **9.59x** | [6.05, 13.25] |
| HST | 0.85 | 0.0168 | 0.0687 | **4.09x** | [3.58, 4.59] |
| Rubin | 1.49 | 0.0641 | 0.1244 | 1.94x | [1.77, 2.13] |
| DES | 1.45 | 0.0881 | 0.1445 | 1.64x | [1.51, 1.78] |
| **mean** | | 0.0374 | 0.1028 | **2.75x** | |

theta_E RMSE over the same lenses: **0.0216" (joint) vs 0.0207" (no-cross)** — no-cross
marginally BETTER, better at 3/5 surveys. **The heads dissociate**: PSF conditioning is
required for high-frequency recovery and irrelevant for the Einstein radius.

**The effect tracks PSF sampling: Spearman rho = -0.90 (p = 0.037)** vs sigma in native px.
Under-sampled PSF -> spectral cutoff near Nyquist, poorly resolved, blur not inferable from
the image, so being told it matters enormously. Wide well-sampled PSF -> the falloff is
obvious in the image, the network infers it, conditioning adds little.

**The capacity confound is controlled by the design.** No-cross has 266,240 fewer params
(13.9%), so "less capacity" would explain the deconvolution gap — except the same reduction
leaves regression unharmed. A generic capacity deficit degrades both heads. Always present
the theta_E panel alongside; it is the control, not padding.

**Do NOT call the no-cross arm "blind deconvolution."** Training-time PSF information cannot
be removed from a supervised deconvolution task: the recon target `g -> f_gt` IS the inverse
of `h`. `h` has TWO consumers — `self(g,h,s)` (the model) and `Lo.reblur_loss(f_hat,h,g)`
(the loss, computed outside the model). The loss exists only during training, so at
inference the no-cross net is a function of `(g, s)` alone; training used `h` to choose
WHICH function to be. The ablation isolates *inference-time, per-sample* conditioning, and
the claim holds only inside the training PSF family.

Figure: `fig7_cross_attention` (3 panels) — Representations main text, Sim2Science appendix.
`fig6_loss_ablation` is demoted to the appendix in BOTH versions.

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

4. **RESOLVED (memory budget) — token counts vary 30×.** At `patch=4`, FOV 20″: HST
   npix=400 → **10,000 image + 10,000 PSF tokens**; Euclid 2,500; Roman 2,025; Rubin 625;
   DES 361; training prior edge (s=0.04″) 15,625. Measured peak is **10.7 GB** at batch 8
   (Step 7, real GPU run — see Hardware section; supersedes the 8.0 GB figure below, which
   predated the Step 1 reblur-cost fix), which FITS the 20 GB GPU — so this is a budgeting
   fact, not a blocker. **Always smoke-test at the prior edge (s=0.04″), not a random
   draw** — a random draw usually yields a cheap grid and hides the worst case.

## Hardware and GPU status (Aug 2026)

Target: a single **20 GB GPU**. Simulation is CPU-only (lenstronomy); training is GPU.
Confirmed Aug 2026 (Step 7): this sandbox's actual GPU is an **NVIDIA A100 80GB PCIe MIG
2g.20gb** slice — 20.9 GB, matching the target exactly.

- `train.py` is GPU-ready: `accelerator="auto"`, model builds tensors on `g.device`,
  attention uses `F.scaled_dot_product_attention` (memory-efficient path available).
- **`test.py` and `diagnose.py` default to `--device cpu`** — pass `--device cuda`.
- **Dataloaders now set `num_workers=4, pin_memory=True, persistent_workers=True`
  automatically when `LensDataModule(cache_dir=...)` is used** (Step 7); the in-memory
  path (`cache_dir=None`) still sets none of these, since a worker would fork (and
  duplicate) the whole in-memory dataset. Measured epoch-time effect on a 500-sample
  smoke run: 130.0s/epoch (no workers) -> 123.8s/epoch (`num_workers=4`), a modest ~5% —
  at this model/image scale GPU compute (long attention sequences, up to 15.6k tokens)
  dominates over data-loading I/O, so this is a smaller lever than expected. Kept anyway
  since it's free.
- **Measured peak memory, Step 7, real GPU run** (`LitLensNet` + `AdamW`, batch=8,
  dim=128, depth=4, fp32, patch=4) at the training prior edge (s=0.04″, npix=500,
  15,625 tokens/stream): **10.73 GB** allocated (10.79 GB reserved) —
  `torch.cuda.max_memory_allocated()`, two full optimizer steps. Comfortably fits the
  20.9 GB MIG GPU (~10 GB headroom). This **supersedes** an earlier, lower estimate of
  8.0 GB (measured pre-Step-1, when the reblur term's cost/behavior was different) — do
  not use 8.0 GB for capacity planning. If OOM, drop batch size before touching
  patch/FOV. (A prior draft claimed `patch=8` "cuts peak 2.7× (8.0 -> 3.0 GB)" —
  that was from the same superseded pre-Step-1 estimate; re-measure before relying on it.)
  Earlier (superseded) table by token count, batch=8: 361 tok 0.5 GB · 2.5k tok 2.0 GB ·
  10k tok (HST) 5.2 GB · 15.6k tok 8.0 GB (now 10.73 GB, see above).
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

## Known implementation bugs found Aug 2026 (5-10 fixed before Run 1; 11-12 found Aug 20)

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
   **RE-MEASURED after the bug-9 flux fix (Aug 2026), as this item required:** the
   spread PERSISTS and is slightly larger — `f_gt` peak 0.0626 (HST) -> 0.7697 (DES)
   = **12.3x** on one lens through all five surveys (absolute values moved ~4x, the
   ratio did not). It is physical, not a bug: coarser pixels collect more flux per
   pixel. Per-sample normalization is still mandatory. Source:
   `figures/paper/fig1_hero_grid_data.csv`, column `f_gt_peak`.
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
11. **FIXED (Aug 20 2026) — `losses.reblur` was shifted by half a pixel on both axes.**
   `gaussian_psf_stamp` centers the stamp on the IMAGE coordinate origin, index
   `(npix-1)/2`, which is a half-integer because npix is even for every survey
   (400/200/182/100/76). A `conv2d` can only place zero lag on an integer index, so
   the cropped-kernel path introduced in Step 1 convolved correctly but *off-center*.
   Measured against the simulator's own convolve-then-resample forward model,
   `||reblur(f_gt,h) - g||^2 / mean(g^2)`: HST 3.18e-2 -> 9.81e-6, Euclid 7.62e-2 ->
   1.61e-4, Roman 1.04e-1 -> 1.83e-3, Rubin 4.75e-2 -> 3.06e-5, DES 5.13e-2 -> 9.31e-5
   (57x-3242x). **Fix implemented:** FFT convolution with an explicit phase ramp
   `exp(2*pi*i*((H-1)/2*ky + (W-1)/2*kx))`, exact for even and odd npix alike.
   `_reblur_full` inherits the same offset and is therefore **not** a correctness
   reference — the reference is the simulator's `g` (`tests/test_reblur.py`).
   Roman stays 1.8e-3 because of bug 3 (PSF sigma 0.46 native px, the stamp aliases);
   that is a stamp-sampling limit, not a centering error.
   **Runs 1 and 2 were both TRAINED with the shifted operator** — their reblur term
   was penalizing a correctly-centered reconstruction. Retraining is not required for
   the current results (the reblur weight is lambda_c=0.1 and its budget share was
   <1%), but any claim about the reblur loss *doing work* must be made on a run
   trained after this fix.
12. **FIXED (Aug 20 2026) — `reblur_mse / floor` was a units error, off by ~1e6.**
   `losses.reblur_loss` is scale-free (it divides by `mean(g**2)`); the Step 9b floor
   `mean(max(g,0)/gain + bkg_rms**2)` is an absolute surface-brightness variance.
   Dividing one by the other inflated the ratio by `1/mean(g**2)`. Test Set 2 first
   reported HST at **126185x** the floor; the correct value is **1.04x**. Fixed by
   also accumulating the floor in the loss's units (`reblur_floor_rel`) and taking the
   ratio against that. Calibrated by a test that feeds a PERFECT reconstruction
   (`f_hat = f_gt`) through the diagnostic and asserts it scores ~1.0 by construction
   (`tests/test_reblur_noise_floor.py`).

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
  **IMPLEMENTED (Aug 2026) in `survey_lens/metrics.py`**, wired into `test.py`. Run 1's
  reconstruction verdict was written before this existed, using masked L1/SSIM, and was
  wrong — see the correction banner in `figures/run1_summary.md`.
  **A power RATIO alone is not sufficient: it is phase-blind and gameable.** Run 1 scored a
  healthy-looking 0.86 top-octave power ratio at Euclid while being *worse than emitting
  nothing* in phase-aware terms. Always report `hf_complex_err` (normalized so `f_hat=0`
  scores exactly **1.0** — above 1.0 means worse than emitting nothing) and `hf_coherence`
  alongside it. Also report `hf_harm_frac_*`: the patch decoder emits energy at the patch
  period, which for `patch=4` is 0.25 cyc/px = **exactly 0.5× Nyquist**, i.e. inside the
  measurement band, so the decoder's own grid can inflate the metric that judges it.
  Compute on an **apodized centered crop, never a binary arc mask** — a hard-edged mask's
  own spectrum falls only as k⁻³ and injects broadband top-octave power into both images.
- **Reblur loss is nearly blind in the blur operator's null space** — re-blurring discards the
  high frequencies that deconvolution must recover. It is a data-consistency /
  anti-hallucination term, *not* a substitute for the SSIM and Fourier terms. Keep this in
  mind before "simplifying" the loss; if dropping terms, run it as a measured ablation.

## Ablation axes (all flag-driven in `train.py`)

1. PSF cross-attention on/off → `--no-cross-attention`
2. joint vs single-task → `--loss-mode {joint,recon_only,reg_only}`
3. native scale vs resampled-to-common-grid → data-side switch
   **(NOT IMPLEMENTED — there is no resampling code anywhere in `src/`.)**
4. loss-term ablation (drop SSIM+FFT) → `--lambda-a 0 --lambda-b 0`

**Axis 2 gives only TWO meaningful numbers, not six.** Each single-task arm's
*other* head is degenerate BY CONSTRUCTION, not merely untrained-and-weak, and
both were measured, not assumed:

- `recon_only`: `theta_hat` never enters `total` (`lit_model._step`), so
  `reg_head` receives **no gradient at all** (verified: 6/6 params report
  `.grad is None`; AdamW then skips them). It stays at random init — the
  `run3_recon_fov12` run logged `val/theta_rmse` ≈ **1.78** against a prior-mean
  null of 0.618", i.e. worse than guessing the mean. That number measures an
  untrained head, nothing about the representation.
- `reg_only`: symmetric. `recon_head[-1]` and `refine[-1]` are zero-init
  (`model.py`) and get no gradient, so `delta` stays exactly 0 and
  **`recon == g` bit-for-bit** — that arm's "reconstruction" IS the `identity`
  baseline, and its `hf_complex_err` should equal identity's to numerical
  precision. `make_ablation_figures.py` asserts exactly this as a self-check;
  if it ever fails, gradient is leaking into `recon_head`.

So read axis 2 as: **deconvolution** = joint vs `recon_only`; **regression** =
joint vs `reg_only`. Never present a degenerate cell as a measurement.

Related trap, now fixed: `train.py` hardcoded `ModelCheckpoint(monitor="val/l1")`,
but `val/l1` is only logged in the joint/recon_only branch, so `--loss-mode
reg_only` raised `MisconfigurationException` before saving anything. The monitor
is now mode-aware. Because the arms therefore select "best" by different
criteria, the ablation analysis uses **`last.ckpt` for every arm** — a uniform
rule — and treats `save_top_k` as a safety net.

Metrics go to CSV under `runs/<name>/` (gitignored, as are `*.ckpt` and `diag_*.png`).

## Running the tests (fast by default, full on demand)

```bash
PYTHONPATH=src python -m pytest tests/ -p no:warnings              # ~43s: 89 passed, 7 skipped
PYTHONPATH=src python -m pytest tests/ -p no:warnings --runslow    # ~115s: all 96 -- FINAL CHECK
```

Seven tests (`test_flux_normalization.py` x5, `test_build_test_set_parallel.py` x2) drive
real `lenstronomy` renders and are **61% of the suite's runtime** (70s of 114s). They carry
the `slow` marker and are skipped unless `--runslow` is passed, so routine edits do not pay
for them. Everything that guards current work still runs every time — the reblur centering
fix (bug 11), the reblur noise floor (bug 12), the decoder contract, and the Wiener baseline
fairness rules.

**Use the fast run while iterating; use `--runslow` before committing or when reporting
results.** Skips are reported by pytest, never silent — but "all tests pass" is a claim
about a `--runslow` run, so do not use that phrase for a default run. Config lives in
`tests/conftest.py`.
