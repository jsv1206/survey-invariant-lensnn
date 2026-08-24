# Paper plan & figure spec — Survey-Invariant Lens Network (TWO venues)

**Filename note:** this file is still `plan_sim2science.md` because `CLAUDE.md` references
it by that path. It now covers **two** submissions.

| | **Sim2Science** | **Representations** |
|---|---|---|
| deadline | **Aug 29 2026**, 23:59 AoE | **TBD — fill this in** |
| format | 5 pp, NeurIPS template, refs/appendix excluded | assume same until confirmed |
| theme | leveraging *imperfect* simulators against real data | symmetry / geometry / structure in learned representations |

Both papers are the **same model, same checkpoint, same numbers** (Run 3, joint arm,
FOV 12″). They differ in framing and in **one** main-text figure.

## Status (verified Aug 22–23 2026)

- Run 3 is trained (3 arms × 80 epochs, seed 0, frozen 12″ cache) and evaluated on
  Test Set 1 and Test Set 2. `$ROOT = /deepskieslab/stronglensing/survey-invariant-lensnn/`.
- **All seven figures now live in `figures/run3/paper/`**, mirrored byte-identically to
  `$ROOT/figures/run3/paper/`. Three scripts write there and their manifests coexist:
  `make_paper_figures.py` v1.2.0 (`captions.md`, `figure_manifest.json`),
  `make_ablation_figures.py` v1.0.0 (`captions_ablation.md`,
  `figure_manifest_ablation.json`), and `make_fig2_architecture.py` v1.0.0
  (`figure_manifest_fig2.json`). The suffixes are load-bearing: before them, whichever
  script ran second silently destroyed the other's captions and provenance.
- Run 2 (FOV 20″, `figures/paper/`) is **frozen provenance** — not regenerated, not moved.
- 96 tests green.

## Framing — what each venue's paper is *about*

**Sim2Science.** The headline is imperfect-simulator-to-real-data. The reblur consistency
loss is self-supervised — it needs no ground truth, so it transfers unchanged to real
images — and survey-invariance is robustness to per-instrument simulator
mis-specification. The noise shift (Test Set 2) is the evidence: trained noise-free,
evaluated on survey noise with no retraining and no augmentation.

**Representations.** The headline is the *representation*: one encoder that is invariant
to pixel scale and PSF across a 5.3× range of instruments, with **RoPE positions in
arcseconds rather than pixel indices** as the mechanism. Survey invariance is a symmetry
statement — a 0.1″ offset must rotate identically for HST and Rubin. The loss ablation is
the evidence for what the representation encodes, and its **asymmetry** is the result:
reconstruction supervision *helps* the geometric task, but geometric supervision *hurts*
reconstruction. The two objectives are not interchangeable.

## Figure allocation

| | **Sim2Science** main text | **Representations** main text |
|---|---|---|
| Fig 1 | hero grid | hero grid |
| Fig 2 | architecture | architecture |
| Fig 3 | survey invariance | survey invariance |
| Fig 4 | deconvolution | deconvolution |
| **Fig 5** | **noise shift** | **cross-attention ablation** |
| appendix | A1 extra lenses, A2 loss ablation, A3 cross-attention | A1 extra lenses, A2 noise shift, A3 loss ablation |

The two versions differ in exactly one main-text slot. Everything else is shared.

**Decided Aug 24 after measuring** (the effect size was allowed to drive the layout, not
the reverse): the **cross-attention** ablation takes the Representations main-text slot and
the loss ablation moves to that paper's appendix. Removing cross-attention costs 1.64-9.98x
in top-octave error at every survey (mean 2.75x) while leaving theta_E untouched, and the
effect tracks PSF sampling at Spearman rho = -0.90 (p = 0.037). That is a mechanism, not
just an ablation — and the theta_E null doubles as the control for the capacity confound.
The loss ablation, by contrast, yields only two meaningful numbers.

## Slug ↔ figure-number mapping (READ THIS BEFORE EDITING LaTeX)

Filenames are **frozen provenance** — they are referenced from `CLAUDE.md`,
`HANDOFF_run1.md`, and both `figure_manifest.json` files. Do **not** rename or regenerate
artifacts to match a venue's numbering.

| file slug | Sim2Science | Representations |
|---|---|---|
| `fig1_hero_grid` | Fig 1 | Fig 1 |
| `fig2_architecture` | Fig 2 | Fig 2 |
| `fig3_survey_invariance` | Fig 3 | Fig 3 |
| `fig4_deconvolution` | Fig 4 | Fig 4 |
| `fig5_noise_shift` | **Fig 5** | Fig A2 |
| `fig6_loss_ablation` | Fig A2 | Fig A3 |
| `fig7_cross_attention` | Fig A3 | **Fig 5** |
| `figA1_hero_grid_appendix` | Fig A1 | Fig A1 |

**The digit inside a slug reflects Sim2Science ordering and is NOT authoritative for the
representations version.** Use `\label`/`\ref` in LaTeX so numbering follows position in
the document, never the filename.

## Output contract (every figure, both venues)

Unchanged from `HANDOFF_run1.md` Step 10 — restated here because it is binding:

- Emit **both** `.pdf` (vector, what LaTeX uses) and `.png` @200 dpi.
- **Width 5.5 in** full / 2.7 in half (NeurIPS text width). Never draw large and scale
  down. No text below **6 pt** at final size.
- Alongside each figure, `figN_<slug>_data.csv` holding exactly the plotted numbers.
  **The paper quotes from the CSV, never off a plot.**
- One survey colour map everywhere (HST→DES by pixel scale), declared once at module top.
  Perceptually uniform image maps; diverging + zero-centred for residuals, limits stated.
- Shared vmin/vmax for panels of the same lens.
- The strict `geometry_check` in `save_figure` must pass — no text/text or text/spine
  overlap, nothing outside bounds, nothing landing on a legend box. It has caught real
  collisions; do not weaken it.
- `figure_manifest.json` records checkpoint path + hash, dataset digest, git commit,
  script version.

---

# Per-figure specification

## Fig 1 — hero grid (`fig1_hero_grid`) — BOTH, main text

**Claim:** one network, five telescopes, five pixel scales, no resampling.

Rows = 5 surveys by pixel scale (HST 0.05″ → DES 0.263″). Columns = `input g` |
`reconstruction f̂` | `truth f_gt` | `residual f̂ − f_gt`. One lens, chosen as the
**median** masked-L1 case across surveys (index 59 of 200) — not the best case; state
the selection rule in the caption. 4″ crop. Every panel in a row divided by that row's
`f_gt` peak, so rows are comparable despite the **12.3×** brightness spread across pixel
scales. Residual limits ±30% of truth peak.

Annotate each row with `s` (arcsec/px), PSF FWHM, and native grid size — the varying
npix (240/120/109/60/46 at 12″) **is** the result. 1″ scale bar per row.

- *Sim2Science:* emphasise that no resampling to a common grid occurs anywhere.
- *Representations:* emphasise that the same weights consume five different token-grid
  sizes, and that Roman's npix is **odd** (109) — the architecture has no parity assumption.

## Fig 2 — architecture (`fig2_architecture`) — BOTH, main text — **BUILT**

**Claim:** the pixel-scale scalar `s` enters the *positional encoding*, and that is the
invariance mechanism.

Drawn programmatically by `src/survey_lens/make_fig2_architecture.py`, **not** hand-assembled
as HANDOFF Step 10 originally proposed. Three reasons a hand-edited raster fails here: the
output contract (5.5″ width, ≥6 pt, PDF+PNG, `_data.csv`, manifest entry, `geometry_check`)
applies to every figure; the constants drawn in the boxes (dim 128, depth 4, heads 4,
patch 4, θ_ref 0.45″, base 100) are **read from `model.py`** so the drawing cannot drift
from the code it depicts; and the per-survey token counts are computed from `surveys.py`
at the stated FOV, which is FOV-dependent (HST 240²px → 3,600 tokens at 12″).

Shows: image `g` + PSF `h` → patchify 4×4 → shared linear (p²=16 → 128) + type embedding
→ TriAttention ×4 (image self-attn, PSF self-attn, image←PSF cross-attn, MLP) → recon head
(`f̂ = g + δ`, a **residual on the input**, zero-init) and regression head (mean-pooled
image tokens → `θ̂_E`). The `s` → arcsec-position path and the separate `log s` →
regression-head path are both drawn in accent, because those are the **only two places
`s` enters the network**.

The three slides in `figures/source_slides/` are pedagogical talk material (RoPE mechanics,
the forward/inverse problem, why L1 alone is insufficient) — **none of them is this block
diagram**, so nothing is duplicated. They stay useful for talks.

- *Representations:* this is the paper's central figure — the arcsec-vs-pixel contrast
  carries the symmetry claim and already has the most visual weight.

## Fig 3 — survey invariance (`fig3_survey_invariance`) — BOTH, main text

**Claim:** accuracy is flat across a 5.3× pixel-scale range.

- (a) θ_E predicted vs true, one colour per survey, y=x line, per-survey RMSE in legend.
- (b) θ_E RMSE vs pixel scale (log x), `prior_mean` as dashed reference.
- (c) masked normalized reconstruction L1 vs pixel scale, with `identity` and `zeros`
  baselines dashed.

Error bars = 16–84th percentile over 1000 bootstrap resamples of the 200 lenses.

Verified numbers (`fig3_survey_invariance_data.csv`): θ_E RMSE 0.0170 / 0.0164 / 0.0215 /
0.0227 / 0.0306″ for HST / Euclid / Roman / Rubin / DES against a prior-mean null of
**0.618″** — i.e. **20–38× better than the null**, varying only **1.9×** across a **5.3×**
pixel-scale range.

- *Representations:* this is the headline result. State the flatness as the invariance
  claim and report the residual trend honestly rather than calling it perfectly flat —
  DES is 1.9× worse than Euclid, and that trend correlates with pixel scale.

## Fig 4 — deconvolution (`fig4_deconvolution`) — BOTH, main text

**Claim:** the network performs real deconvolution, not smoothing or denoising.

- (a) Radial power spectra for `f_gt`, `g`, `f̂` at one survey (HST), top octave shaded,
  PSF cutoff marked.
- (b) Top-octave **complex** error vs PSF σ in native pixels, with `identity` and Wiener
  (known PSF, best `K`) reference curves.
- (c) Reblur residual `f̂ * h − g` per survey — the diagnostic needing no ground truth.

**Metric discipline (non-negotiable):** report `hf_complex_err`, normalized so `f̂=0`
scores exactly **1.0**; above 1.0 is worse than emitting nothing. A power *ratio* alone
is phase-blind and gameable. Computed on an apodized centred crop, never a binary arc
mask.

**Wiener must be quoted fairly.** Its `K` is swept `1e+1 … 1e-14` and selected **on the
metric it is quoted against**, with the optimum interior to the grid (never edge-pinned).
Verified TS1 (`fig4_deconvolution_data.csv`):

| survey | net | Wiener | verdict |
|---|---|---|---|
| HST | 0.0168 | 0.0258 | net 1.53× |
| Euclid | 0.0098 | 0.0368 | net 3.74× |
| Roman | 0.0082 | 0.0520 | net 6.32× |
| Rubin | 0.0641 | 0.0573 | **Wiener 1.12×** |
| DES | 0.0881 | 0.0782 | **Wiener 1.13×** |

**Do NOT write "beats Wiener at every survey."** That claim was true only against an
under-tuned baseline. On noise-free data an oracle-tuned Wiener wins at the two
seeing-limited surveys, as theory says it must.

## Fig 5 — noise shift (`fig5_noise_shift`) — **Sim2Science main text / Representations appendix**

**Claim:** a model trained on a noise-free simulator degrades gracefully under real survey
noise, and the reblur residual detects failure **without ground truth**.

- (a) θ_E RMSE per survey, TS1 vs TS2 paired bars, prior-mean dashed, delta annotated.
- (b) `reblur_mse / noise_floor`, line at 1.0. **Below 1.0 means fitting noise** — a
  failure mode, not a win.
- (c) Reblur residual maps on TS2, shared diverging scale. **This is the sim2real hook** —
  the only diagnostic that transfers unchanged to real images.
- (d) Top-octave error TS1 vs TS2, with a **black tick per bar marking Wiener**, swept
  **separately on each test set** (its noise-free optimum is up to 13 decades from its
  noisy one; carrying one `K` across both would fake a collapse).

Verified TS2 (`fig5_noise_shift_data.csv`): net beats Wiener **4.0–8.7×** at Euclid,
Roman, Rubin, DES. At **HST both fail** — net 1.029, Wiener 0.914, i.e. at or worse than
emitting nothing. Report it; panel (b) independently flags HST at 0.92× its floor, the
signature of a reconstruction that fitted the noise.

**Honest-reporting requirement:** nothing is tuned on Test Set 2. "The noise-free
simulator does not transfer without augmentation" is a legitimate and publishable
Sim2Science result. Adding noise to training would answer a different question.

## Fig A3 — loss ablation (`fig6_loss_ablation`) — **appendix in BOTH versions**

**Claim:** the two objectives are asymmetric — reconstruction supervision helps the
geometric task, but geometric supervision hurts reconstruction.

*Demoted from the Representations main text on Aug 24, deliberately:* this axis yields only
TWO meaningful numbers and each single-task arm has a degenerate head, whereas
`fig7_cross_attention` yields two genuine comparisons plus a mechanism. Kept in full — it is
a real result, just a smaller one.

Three arms, one architecture, one frozen 12″ dataset, 80-epoch cosine, seed 0; only
`--loss-mode` differs. Run 2 could **not** serve as the joint arm — it predates the
bug-11 reblur fix, so it differs in code as well as loss mode.

- (a) Regression: θ_E RMSE, joint vs `reg_only`, against the prior-mean null.
- (b) Deconvolution: top-octave complex error, joint vs `recon_only`, with `identity`
  and Wiener references.

Verified (`fig6_loss_ablation_data.csv`), mean over surveys:

| comparison | joint | single-task | effect |
|---|---|---|---|
| θ_E RMSE (vs `reg_only`) | 0.0216″ | 0.0256″ | recon loss **helps 16%** |
| top-octave err (vs `recon_only`) | 0.0374 | 0.0193 | θ_E loss **costs 94%** |

Consistent in sign at **all five surveys** in both directions.

**This axis yields exactly TWO meaningful numbers, not six.** Each single-task arm's
*other* head is degenerate **by construction**, and both were measured, not assumed:

- `recon_only`'s regression head receives **no gradient at all** (6/6 params report
  `.grad is None`); it sits at random init and logs θ_E RMSE ≈ **1.78″**, worse than the
  0.618″ prior-mean null. That measures an untrained head, nothing about the representation.
- `reg_only`'s `recon_head[-1]` and `refine[-1]` are zero-init and get no gradient, so
  `recon == g` **bit-for-bit** — that arm's "reconstruction" *is* the `identity` baseline.
  Verified: `reg_only` scores 0.588977 at HST, identical to `identity` to all printed
  digits. `make_ablation_figures.py` asserts this as a self-check; if it ever fails,
  gradient is leaking into `recon_head`.

Degenerate cells are drawn outlined and hatched and **must never be presented as
measurements**.

*Caveat to state in the caption:* the `reg_only` arm trains one head and gets the whole
gradient budget, while joint splits capacity across two objectives. Had `reg_only` won,
"undivided capacity" would have explained it equally well. It did not win — joint is
better at 5/5 surveys — so the confound works **against** the reported result, which makes
it stronger than the design alone could guarantee.

## Fig 7 — cross-attention ablation (`fig7_cross_attention`) — **Representations main text / Sim2Science appendix**

**Claim:** explicit per-sample PSF conditioning is required to recover high-frequency
structure and is irrelevant for recovering the Einstein radius — and the size of the effect
is set by how well the PSF is sampled.

Ablation axis 1: `--no-cross-attention`, joint loss, **one flag** from the reference arm
(verified: the checkpoints differ in `use_cross_attention` and nothing else). Cross-attention
is the ONLY path from the PSF to either head — with it off, feeding a 9x-wider wrong PSF at
inference changes the output by **exactly zero**.

- (a) Top-octave complex error, joint vs no-cross, per survey. Worse at all five:
  Roman 9.98x, Euclid 9.59x, HST 4.09x, Rubin 1.94x, DES 1.64x; mean **2.75x**.
- (b) theta_E RMSE, same lenses. **0.0216" vs 0.0207"** — no-cross marginally BETTER, and
  better at 3/5 surveys. This panel is not padding: it is the internal control for the
  capacity confound (below).
- (c) Benefit ratio vs PSF sigma in native pixels. **Spearman rho = -0.90, p = 0.037.**

**Mechanism to state:** an under-sampled PSF has its spectral cutoff near Nyquist where it
is poorly resolved, so the blur cannot be read off the image and being *told* it matters
enormously (Roman, sigma=0.46 px, 9.98x). A wide, well-sampled PSF announces itself in the
image's spectral falloff, so the network infers it and conditioning adds little
(Rubin, sigma=1.49 px, 1.94x).

**Confound, and why the design controls it:** removing cross-attention removes 266,240
params (13.9%). "Less capacity" would explain (a) — except the same reduction leaves (b)
untouched. A generic capacity deficit degrades both heads; only the reconstruction pathway
degrades. State this explicitly; it is the first thing a referee will probe.

**Bound the claim.** The no-cross arm is NOT "blind deconvolution": training-time PSF
information cannot be removed from a supervised deconvolution task, since the recon target
`g -> f_gt` IS the inverse of `h`. The ablation isolates *inference-time, per-sample*
conditioning. Within this PSF family (Gaussian, spatially invariant, `fwhm ~ logU(0.08,1.1)"`)
some blur is inferable from the image; for realistic multi-parameter, spatially varying PSFs
explicit conditioning should matter more, not less.

## Fig A1 — appendix hero grid (`figA1_hero_grid_appendix`) — BOTH, appendix

Three additional lenses including the **worst case** (index 10), same layout and
normalization as Fig 1.

---

# Limitations to state (both venues)

1. **The network learned the simulator's exact generative distribution** (Sérsic source,
   EPL + shear). Beating a generic-prior filter on data drawn from the prior it trained on
   is easier than on real sky; Wiener's weaker assumptions transfer more honestly. At a
   workshop about *imperfect simulators* this objection will be raised if we do not raise
   it first.
2. **Euclid and Roman PSF stamps are under-sampled** (σ = 0.68 and 0.46 native px). Below
   ~0.8 px the Gaussian stamp barely resolves its own kernel, so the reblur loss inherits
   that error. This is CLAUDE.md dataset bug 3, still open — it is why Roman's reblur
   residual is 2.07e-3 against HST's 9.81e-6.
3. **Gaussian, spatially-invariant PSF.** Real PSFs are neither.
4. **Test Set 3 (real data) is not in either paper** at this deadline.

# Remaining work

| # | item | status |
|---|---|---|
| 1 | Build Fig 2 | **DONE** — `make_fig2_architecture.py`, in `paper/` + mirrored |
| 2 | Move `fig6_loss_ablation` into `paper/` | **DONE** — suffixed manifest/captions so nothing is clobbered |
| 3 | Fix the stale "Deferred" caption block | **DONE** — now an accurate per-axis status list |
| 4 | *Optional:* run ablation axis 1 (`--no-cross-attention`) | **open**, ~2h15m train + ~10 min eval |
| 5 | Fill in the Representations deadline | **open** — gates (4) |
| 6 | Write the LaTeX | not started; no `.tex` exists yet |

On (4): Fig 6 is main-text for Representations but carries only **two** meaningful numbers,
and each single-task arm's other head is degenerate by construction. Axis 1 would add a
genuine architectural claim — does conditioning on the PSF token stream matter? — which is
squarely a representations question. Cost is measured, not estimated: the joint arm ran
80 epochs in ~2h15m (01:59:46 → 04:13) on the MIG A100. Gated on that venue's deadline.

**Ablation axis status:** axis 1 (cross-attention) not run · axis 2 (loss mode) **done** ·
axis 3 (native vs resampled) **has no implementation anywhere in `src/`** · axis 4
(drop SSIM+FFT) not run.

# Deferred (not in either submission)

- **Test Set 3 — real data.** 2–3 real cutouts (COSMOS HST / DES / Euclid Q1) with the
  reblur residual as the diagnostic. Deliberately qualitative; do not scope a full
  real-data benchmark for this deadline.
- Ablation axes 3 and 4.

# Paper outline (5 pp, both venues)

1. Title/Abstract — lead with the venue's hook (sim2real / representation)
2. Intro (¾ p) — 4 contribution bullets
3. Method (1½ p) — patch + RoPE-in-arcsec, PSF cross-attention, dual heads, loss breakdown
4. Experiments (2 p) — setup; cross-survey θ_E; deconvolution vs Wiener; **the venue's Fig 5**
5. Discussion / Limitations / Conclusion (½ p)
6. Appendix — hyperparameters, full per-survey tables, the other venue's Fig 5, extra reconstructions
