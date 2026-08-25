# Results summary — writing reference for both workshop papers

**Status:** Run 3, FOV 12″, joint arm is the headline model. Deadline **Aug 29 2026** for
both venues. Every number here is quoted from an emitted `_data.csv` or `_metrics.csv`
under `figures/run3/`, never read off a plot. Where a claim is bounded or fails, that is
stated inline — the bounds are part of the result.

Companion documents: `plan_sim2science.md` (per-figure spec, figure allocation),
`CLAUDE.md` (conventions, known bugs, hardware).

**Figure files:** all seven live in `figures/run3/paper/`, mirrored to
`$ROOT/figures/run3/paper/`. Slugs are frozen provenance; figure NUMBERS differ per venue
(see the mapping table in `plan_sim2science.md`) — resolve them with `\label`/`\ref`,
never from the filename.

---

## 1. The one-paragraph version

A single transformer embeds strong-lens images from five surveys spanning a **5.3× range
of pixel scale** (HST 0.05″/px → DES 0.263″/px) with **no resampling to a common grid**.
Positional encoding is 2D RoPE in **arcseconds from the lens centre**, not pixel indices,
so the same physical offset produces the same rotation at every instrument. Two heads:
image reconstruction (deconvolution) and Einstein-radius regression. A self-supervised
**reblur consistency** term `‖f̂ * h − g‖²` requires no ground truth and therefore
transfers unchanged to real data. Trained noise-free; evaluated both noise-free (Test
Set 1) and under each survey's own noise (Test Set 2) on the **same** checkpoint.

## 2. Setup facts that belong in every version

| item | value |
|---|---|
| surveys | HST 0.05″/0.10″ · Euclid 0.10/0.16 · Roman 0.11/0.12 · Rubin 0.20/0.70 · DES 0.263/0.90 (scale/FWHM, arcsec) |
| field of view | 12″ (0.088% worst-case flux discarded vs 20″; see `plan_sim2science.md`) |
| native grids | 240 / 120 / **109** / 60 / 46 px — Roman is **odd**; no parity assumption anywhere |
| tokens per stream | 3,600 / 900 / 784 / 225 / 144 |
| training set | 10,000 frozen samples, `fov=12.0`, cached to disk; every arm trains on identical bytes |
| test sets | 200 lens parameter sets drawn once, each rendered through **all five surveys** (paired design) |
| schedule | 80 epochs cosine, seed 0, batch 8, bf16-mixed, AdamW lr 3e-4 |
| model | 1.91 M params · dim 128 · depth 4 · 4 heads · patch 4 · θ_ref 0.45″ · RoPE base 100 |
| error bars | 16–84th percentile, 1000 bootstrap resamples of the 200 lenses |

**Paired design matters for every cross-survey statement:** a gap between surveys reflects
the telescope, not a different lens sample. Say so explicitly in the paper.

**Test Set 2 is evaluation-only** — no retraining, no noise augmentation, noise applied to
the model input while `f_gt` and `h` stay clean. It is a genuine distribution shift.

## 3. Metric discipline (the part a referee will probe)

- **Headline reconstruction metric is top-octave (0.5–1.0× Nyquist) *complex* error**,
  normalized so that emitting nothing (`f̂ = 0`) scores exactly **1.0**. Above 1.0 is
  worse than emitting nothing.
- **A power ratio alone is not sufficient** — it is phase-blind and gameable. Run 1 scored
  a healthy-looking 0.86 top-octave *power ratio* at Euclid while being worse than
  emitting nothing in phase-aware terms. Always report `hf_complex_err` and
  `hf_coherence`.
- Computed on an **apodized centred crop, never a binary arc mask** — a hard-edged mask's
  own spectrum falls only as k⁻³ and injects broadband top-octave power into both images.
- **Global PSNR hides the deconvolution effect** (it behaves like the pixel-L1 term).
- Images are sparse (2.7–4.0% of pixels above 1% of peak), so masked metrics are the
  headline and global ones are the hallucination check.
- Peak brightness spans **12.3×** across surveys (physical: coarser pixels collect more
  flux per pixel), so per-sample normalization is mandatory before any cross-survey
  reconstruction metric.

### Baseline fairness rule — state it, it is a strength

> A baseline quoted against a metric must have been **tuned on that metric**, and its
> hyperparameter optimum must be **interior to the search grid**.

Wiener's `K` is swept `1e+1 … 1e-14` and selected on the metric it is quoted against.
Every reported optimum is interior — none edge-pinned. An earlier version selected `K` by
masked L1 and then scored it on top-octave error; fixing that moved Wiener at Rubin
0.315 → 0.057 and DES 0.328 → 0.078, **flipping both surveys**. Two regression tests pin
the rule (`tests/test_baselines.py`).

---

## 4. Core results (shared by both papers)

### 4.1 Survey invariance — accuracy is NEARLY flat across a 5.3× pixel-scale range

theta_E error, Test Set 1 (`fig3_survey_invariance_data.csv`), reported as a **fraction of
the true theta_E** because a percentage is interpretable against the truth where an
absolute arcsec value needs the 0.5-2.5" prior to mean anything:

| | HST | Euclid | Roman | Rubin | DES |
|---|---|---|---|---|---|
| fractional RMSE | 1.37% | 1.34% | 1.72% | 2.37% | **3.69%** |
| absolute RMSE | 0.0170" | 0.0164" | 0.0215" | 0.0227" | 0.0306" |
| median \|frac\| | 0.66% | 0.67% | 0.79% | 1.01% | 1.38% |

**20-36x better than the prior-mean null** (0.618", §4.5). But say **"nearly flat"**, never
"flat": there is a monotonic **2.75x** trend with pixel scale. Note the fractional metric
makes that trend LARGER than absolute RMSE does (1.87x), because per-lens normalization
up-weights small-theta_E systems where a fixed positional error is a bigger fraction. That
is the honest presentation, and the trend was invisible in earlier drafts only because a
baseline line forced a 162x y-axis.

### 4.2 Deconvolution vs classical Wiener — NOT a clean sweep

Top-octave complex error, trained vs Wiener with the **known** PSF, both test sets:

| survey | TS1 net | TS1 Wiener | verdict | TS2 net | TS2 Wiener | verdict |
|---|---|---|---|---|---|---|
| HST | 0.0168 | 0.0258 | net 1.53× | 1.029 | 0.914 | Wiener 1.13× (**both fail**) |
| Euclid | 0.0098 | 0.0368 | net 3.74× | 0.039 | 0.309 | net **7.84×** |
| Roman | 0.0082 | 0.0520 | net 6.32× | 0.020 | 0.079 | net **3.98×** |
| Rubin | 0.0641 | 0.0573 | **Wiener 1.12×** | 0.090 | 0.782 | net **8.65×** |
| DES | 0.0881 | 0.0782 | **Wiener 1.13×** | 0.113 | 0.728 | net **6.42×** |

**Do NOT write "beats Wiener at every survey."** On noise-free data an oracle-tuned Wiener
wins at the two seeing-limited surveys, exactly as theory requires. The defensible claim is
**noise robustness**:

> An oracle-tuned Wiener filter is excellent on clean data — it edges us at the
> seeing-limited surveys — and collapses once its noise assumption is violated, where the
> network wins by 4.0–8.7× at four of five surveys.

Wiener's optimal `K` moves up to **13 decades** between the two test sets (1e-13 → 1e-3 at
Rubin), which is why it must be re-swept per test set.

### 4.3 Noise shift (Test Set 1 → Test Set 2)

θ_E RMSE degrades but stays far below the 0.618″ null everywhere. Top-octave error:
Euclid 0.010→0.039, Roman 0.008→0.020, Rubin 0.064→0.090, DES 0.088→0.113, and
**HST 0.017→1.029 — above 1.0, i.e. worse than emitting nothing.** Report it.

The reblur diagnostic independently flags HST **without ground truth**: it sits at
**0.92×** its own analytic noise floor. Below 1.0 means the model is fitting noise — a
failure mode, not a win. That is the sim2real hook: the same diagnostic runs unchanged on
real images.

### 4.4 The figure set, and what each figure is allowed to claim

**Three figures per paper, no appendix.** Everything else is still generated and kept,
but is not allocated to these submissions.

| | Sim2Science (5pp) | Representations (4pp) |
|---|---|---|
| Fig 1 | `fig1_hero_grid` | `fig1_hero_grid` |
| Fig 2 | `fig3_survey_invariance_noise` | `fig3_survey_invariance` |
| Fig 3 | `fig7_cross_attention` | `fig7_cross_attention` |

Generated but unallocated: `fig2_architecture`, `fig6_loss_ablation`,
`figA1_hero_grid_appendix`. Not emitted at all: `fig4_deconvolution`, `fig5_noise_shift`
(code retained behind `--with-fig4` / `--with-fig5`).

**Slugs are frozen provenance and their digits are historical — they match NEITHER
venue's numbering.** Resolve numbers with `\label`/`\ref`, never from a filename.

#### Fig 1 — one lens, five telescopes

COLUMNS are surveys ordered by pixel scale; ROWS are `input g` / `recon f_hat` /
`truth f_gt` / `residual`. Every panel of a COLUMN is divided by that column's `f_gt`
peak, printed in the header as `S_peak`, so columns are internally consistent and
comparable to each other despite a **15.3x** spread in flux per pixel.

* `S` is flux **per pixel** in arbitrary simulator units — there is no photometric
  zeropoint (`src_amp ~ U(2,6)`). It is NOT surface brightness: lenstronomy returns flux
  per pixel and `_area_resample` SUMS to conserve it, which is exactly why `S_peak` varies
  across surveys at all. True surface brightness would be roughly survey-independent.
* The input panel peaks BELOW 1.0 on purpose: the normalization is by the TRUTH's peak,
  and blurring spreads flux and lowers the peak.
* The residual row is drawn on **+/-0.05**, a 20x tighter scale than the image rows. It
  was +/-0.30 (inherited from Run 2, when the model was worse), which rendered
  HST/Euclid/Roman as blank white and hid the cross-survey ordering entirely. At 0.05 only
  ~0.5% of pixels saturate, at Rubin and DES, and that saturation is the point.
* One lens, chosen as the **median** masked-L1 case (index 59 of 200) — deliberately not
  the best.

#### Fig 2 — invariance, in two variants

One row, three panels. Both variants report identical numbers for the noise-free set.

| panel | metric | x axis |
|---|---|---|
| (a) | theta_E **fractional** RMSE (%) | pixel scale (log) |
| (b) | relative L1, % of arc flux, + `identity` | pixel scale (log) |
| (c) | high-freq. octave error, dimensionless, + `identity` + Wiener | **PSF sigma (native px)** |

* **Panel (c) uses a different x on purpose.** The effect tracks PSF SAMPLING: Spearman
  rho = +0.90 (p = 0.037) against sigma_px versus +0.60 (p = 0.285, not significant)
  against pixel scale. A one-row layout shares no axis, so mixed x costs nothing.
* **Panel (c) is deliberately NOT a percentage** while (a) and (b) are. It divides squared
  amplitudes, so x100 would be a fraction of POWER beside two fractions of FLUX: HST reads
  1.68% in power but 13.0% in amplitude. Adjacency alone would manufacture that
  misreading. The documented convention is that emitting nothing scores exactly 1.0.
* **Fractional, not absolute, theta_E error** — a percentage is directly interpretable
  where "0.017 arcsec" needs the 0.5-2.5" prior to mean anything. It is also a HARDER
  test: per-lens normalization up-weights small-theta_E systems, so the cross-survey
  spread is 2.75x rather than 1.87x. Say "nearly flat", never "flat".
* In the noise variant, solid = noise-free and hollow = survey noise, with the noisy
  points nudged along x so the error bars do not overprint. The nudge is presentational;
  both members of a pair are the same survey at the same true x.
* No prior-mean baseline is drawn: the 0.618" line forced a 162x y-span that left the data
  in the bottom 1.1% of the axis. The number is kept in 4.5 for referees.

#### Fig 3 — cross-attention, one panel

Deconvolution error, joint vs no-cross, per survey. **Its other two results are computed
and written to `fig7_cross_attention_data.csv` as `b_data` / `c_data`, and MUST appear in
the text** — see 5.1. Without them the panel does not answer the capacity objection.

### 4.5 How every baseline / reference number is computed

None of these is a fit, a separate trained model, or an interpolation.

**`identity`** — return the blurred input unchanged, `f_hat = g`. Same metric as the
trained curve with one substitution: normalize both by the same per-sample `f_gt` peak,
mask to `f_gt > 1% of peak`, take the relative L1, average over the 200 lenses. **The mask
is built from the TRUTH**, so trained and `identity` are scored on byte-identical pixels.

| survey | mask_frac | `identity` | trained | margin |
|---|---|---|---|---|
| HST | 5.8% | 8.80% | 1.17% | 7.5x |
| Euclid | 6.5% | 14.71% | 2.00% | 7.4x |
| Roman | 6.7% | 9.27% | 1.95% | 4.8x |
| Rubin | 7.7% | 56.71% | 9.40% | 6.0x |
| DES | 8.2% | 62.76% | 13.33% | 4.7x |

It is load-bearing, not decoration: trained L1 rises **22x** HST->DES, which alone reads as
the method degrading; `identity` rises **13x** over the same range, so the correct reading
is that the PROBLEM got harder while the margin held at 4.7-7.5x. It is also a genuinely
STRONG baseline here, because masked L1 is low-frequency dominated and `g` is already close
to `f_gt` there. CLAUDE.md: the deconvolution claim requires beating `identity`, **not
merely `zeros`**.

Caveats: the baseline curves carry **no error bars** (only the trained points are
bootstrapped), and `mask_frac` is not constant across surveys (5.8% -> 8.2%) because the
mask is defined on the truth in NATIVE pixels. That does not bias the within-survey
comparison, but a little of the cross-survey trend in the `identity` curve is that effect.

**`zeros`** — `f_hat = 0`. 2-12x worse than `identity`, so beating `identity` implies it.
In the CSVs; **not plotted**.

**`prior_mean`** — always predict the test set's mean theta_E (1.4716"), ignoring the
image. Its RMSE is therefore the sample's own standard deviation: **0.6184"**. The prior is
`theta_E ~ U(0.5, 2.5)"` (population sd 0.5774; this draw sits ~7% high, within scatter for
n=200). Identical at all five surveys — a consistency check, since the paired design sends
the SAME 200 lenses through every telescope. **Not plotted** (it forced a 162x y-span), but
Run 3 sits at 1.34-3.69% fractional error, i.e. **20-36x** better than this null.

**Masked SSIM** — measured, never plotted. It ranks the surveys **identically** to the L1
panel (Spearman rho = 1.00), so it adds no independent ordering. It is also the more
flattering metric: 1-SSIM sits 18-49x below `identity` against L1's 4.7-7.5x. Available in
the per-run metrics CSVs.

**reblur operator floor** — the residual `||f_hat*h - g||^2/mean(g^2)` cannot reach zero
even for a PERFECT reconstruction, because `losses.reblur` is not an exact inverse of the
simulator's convolve-then-resample forward model. Feeding `f_hat = f_gt` gives the floor,
which spans **211x** across surveys and tracks PSF sampling, not survey quality (Roman's
stamp aliases at sigma = 0.46 px, dataset bug 3). **Always report the RATIO, never the raw
residual**: raw, Roman looks worst by 3x when it is in fact sitting ON its floor and is the
best of the five. Per-sample ratios are heavy-tailed — quote the MEDIAN (HST 7.5) not the
mean (HST 119).

---

## 5. Ablations

### 5.1 Axis 1 — PSF cross-attention (Fig 3 in BOTH papers)

One flag apart from the joint arm; verified the checkpoints differ in
`use_cross_attention` **and nothing else**. Cross-attention is the **only** path from the
PSF to either output — with it off, feeding a 9× wider wrong PSF changes the output by
**exactly zero**.

Top-octave complex error, Test Set 1:

| survey | joint | no-cross | cost of removing |
|---|---|---|---|
| Roman | 0.0082 | 0.0822 | **9.98×** |
| Euclid | 0.0098 | 0.0944 | **9.59×** |
| HST | 0.0168 | 0.0687 | **4.09×** |
| Rubin | 0.0641 | 0.1244 | 1.94× |
| DES | 0.0881 | 0.1445 | 1.64× |
| **mean** | 0.0374 | 0.1028 | **2.75×** |

θ_E RMSE, same lenses: **0.0216 (joint) vs 0.0207 (no-cross)** — no-cross is marginally
*better*, and better at **3 of 5** surveys.

> PSF conditioning is required to recover high-frequency structure and is **irrelevant**
> for recovering the Einstein radius. The geometric quantity survives blur; the
> high-frequency content does not.

**The effect tracks PSF sampling: Spearman ρ = −0.90 (p = 0.037)** against σ in native
pixels. Every per-survey ratio excludes 1.0 with room to spare (68% bootstrap CIs:
Roman [6.92, 13.18], Euclid [6.05, 13.25], HST [3.58, 4.59], Rubin [1.77, 2.13],
DES [1.51, 1.78]), so the deconvolution effect is significant at all five. The benefit is largest where the PSF is most **under-sampled** (Roman σ=0.46 px →
9.98×) and smallest where it is wide and well-sampled (Rubin σ=1.49 px → 1.94×).
Mechanism: an under-sampled PSF has its spectral cutoff near Nyquist where it is poorly
resolved, so the blur cannot be inferred from the image and being *told* it matters
enormously; a wide PSF announces itself in the image's spectral falloff.

**This ranking is the OPPOSITE of the difficulty ranking in §4.4, and the two must not be
conflated.** Deconvolution is hardest at Rubin/DES (their PSF destroys the most
information); knowing the PSF helps most at Roman/Euclid (their PSF is hardest to infer
from the image). Different questions, opposite orderings.

**The capacity confound is controlled by the design.** Removing cross-attention removes
266,240 params (13.9%), so "less capacity" is a live alternative explanation for the
deconvolution gap — except the *same* reduction leaves regression unharmed. A generic
capacity deficit would degrade both heads; only the reconstruction pathway degrades.

**Bound the claim.** The no-cross arm is not "blind deconvolution": training-time PSF
information cannot be removed from a supervised deconvolution task, because the recon
target `g → f_gt` **is** the inverse of `h`. The ablation isolates *inference-time,
per-sample* conditioning. A weaker no-cross result therefore does not mean the PSF input
is unnecessary in general — only that within this PSF family (Gaussian, spatially
invariant, `fwhm ~ logU(0.08,1.1)″`) some of the blur is inferable from the image.

### 5.2 Axis 2 — loss mode (measured; NOT in either paper, text-only)

| comparison | joint | single-task | effect |
|---|---|---|---|
| θ_E RMSE vs `reg_only` | 0.0216″ | 0.0256″ | recon loss **helps 16%** |
| top-octave vs `recon_only` | 0.0374 | 0.0193 | θ_E loss **costs 94%** |

Consistent in sign at all five surveys both ways. The two objectives are **not symmetric**.

**This axis yields exactly TWO meaningful numbers, not six.** Each single-task arm's other
head is degenerate *by construction*, both measured not assumed: `recon_only`'s regression
head receives no gradient at all (6/6 params `.grad is None`) and sits at random init
(θ_E RMSE ≈ 1.78″, worse than the null); `reg_only`'s recon head is zero-init and receives
no gradient, so `recon == g` **bit-for-bit** — that arm's "reconstruction" *is* the
`identity` baseline (verified: 0.588977 at HST, identical to identity). Never present a
degenerate cell as a measurement.

Caveat that strengthens the regression result: `reg_only` trains one head and gets the
whole gradient budget while joint splits capacity. The confound runs **against** the
reported result, which did not stop joint winning at 5/5 surveys.

### 5.3 Axes not run

Axis 3 (native vs resampled) **has no implementation anywhere in `src/`**. Axis 4 (drop
SSIM+FFT) not run. Say "not evaluated", never imply they were.

---

## 6. Per-venue framing

Both papers: **three figures, no appendix.** Figs 1 and 3 are shared; only Fig 2 differs.

### 6.1 Sim2Science (5 pp) — *leveraging imperfect simulators against real data*

**Thesis.** The reblur consistency loss needs no ground truth, so the objective and the
diagnostic that train the model on simulations run unchanged on real images; survey
invariance is robustness to per-instrument simulator mis-specification.

**Lead with** the noise variant of Fig 2: trained noise-free, evaluated on realistic
per-survey noise with no retraining and no augmentation, and still beating an oracle-tuned
Wiener by 4.0-8.7x at four of five surveys once that filter's noise assumption breaks.

**Figures:** 1 hero · 2 invariance (noise variant) · 3 cross-attention.

**The honest-reporting angle is an asset here.** HST fails under noise (top-octave 1.029 >
1.0) and the paper says so; the reblur diagnostic catches it **without ground truth** at
0.92x its own floor. "The noise-free simulator does not transfer without augmentation at
HST" is a legitimate and publishable Sim2Science result.

### 6.2 Representations (4 pp) — *symmetry / geometry in learned representations*

**Thesis.** One encoder invariant to pixel scale and PSF across a 5.3x range, with **RoPE
positions in arcseconds rather than pixel indices** as the mechanism — a symmetry claim: a
0.1" offset must rotate identically for HST and Rubin.

**Lead with** Fig 2 (invariance) and Fig 3 (cross-attention). The cross-attention result
says what the representation NEEDS: explicit access to the instrument response for
high-frequency recovery, and nothing extra for the geometric quantity.

**Figures:** 1 hero · 2 invariance (noise-free) · 3 cross-attention.

**Strongest available sentence:** *the two heads dissociate.* The same architectural
removal destroys deconvolution (2.75x mean, up to 9.98x) and leaves Einstein-radius
regression untouched — a double dissociation that simultaneously controls the capacity
confound.

### 6.3 What is text-only in BOTH papers

No figure carries these any more. Each is measured and quotable from a CSV:

* **Reblur consistency** — on noisy input it flags HST at **0.92x its own analytic noise
  floor** (below 1.0 = fitting noise) **without using ground truth**. Caveat if quoted: the
  analytic floor omits the reblur operator's own error, which is 38% of Roman's true floor.
* **Top-octave coherence**, the answer to "is this just sharpening artifacts?" — trained vs
  `identity`: 0.993/0.900 HST, 0.975/0.615 Rubin, 0.958/0.587 DES. An artifact raises power
  without raising coherence.
* **theta_E is unaffected by removing cross-attention** — 0.0216" vs 0.0207", no-cross
  better at 3/5. This is the **capacity control** (§5.1) and must be stated, not omitted.
* **Why theta_E survives blur** — it is geometric, set by the ring's position and scale.
  The PSF attenuates high spatial frequencies without moving the ring, so what theta_E
  depends on survives convolution while the fine structure the reconstruction head needs
  does not. That is why the heads dissociate.
* **The loss-mode asymmetry** — reconstruction supervision helps theta_E by 16%
  (0.0216" vs 0.0256"); the theta_E objective costs deconvolution 94% (0.037 vs 0.019).
* **The architecture** — `fig2_architecture` is generated but unallocated, so patch
  embedding, arcsec RoPE, the three attention paths and the two heads must be described in
  prose. Unusual for a method paper; a referee may ask for the diagram.

## 7. Limitations to state (both venues)

1. **The network learned the simulator's exact generative distribution** (Sérsic source,
   EPL + shear). Beating a generic-prior filter on data drawn from the prior it trained on
   is easier than on real sky; Wiener's weaker assumptions transfer more honestly. At a
   workshop about *imperfect simulators* this objection will be raised if we do not raise
   it first.
2. **Euclid and Roman PSF stamps are under-sampled** (σ = 0.68 and 0.46 native px). Below
   ~0.8 px the Gaussian stamp barely resolves its own kernel, so the reblur operator
   inherits that error — Roman's reblur residual is 2.07e-3 against HST's 9.81e-6. Note
   the interaction with §5.1: these are also the surveys where cross-attention helps most.
3. **Gaussian, spatially invariant PSF.** Real PSFs are neither. This bounds §5.1 in the
   conservative direction — explicit conditioning should matter *more* for realistic
   multi-parameter, spatially varying PSFs.
4. **Test Set 3 (real data) is not in either paper** at this deadline.
5. **HST fails under the TS2 noise shift** (top-octave 1.029 > 1.0).
6. **The arcsec-vs-pixel RoPE claim is asserted, not measured.** The Representations paper
   states that arcsec coordinates *are* the invariance mechanism; no experiment
   demonstrates it. It is a ~10-line change plus one run — the obvious next experiment,
   and worth naming as such in the paper rather than leaving the gap silent.

## 8. Provenance and how to regenerate

Every figure is re-runnable from the checkpoint plus the cached dataset with no manual
steps, and ships `<slug>_data.csv` containing exactly the plotted numbers. **Quote the
CSV, never the plot.**

```bash
ROOT=/deepskieslab/stronglensing/survey-invariant-lensnn
CK=$ROOT/runs/run3_joint_fov12/version_0/checkpoints/last.ckpt

# Figs 1 and 2 (both variants) + figA1
PYTHONPATH=src python -m survey_lens.make_paper_figures \
    --ckpt $CK --out-dir figures/run3/paper --fov 12.0 --device cuda --workers 8

# Fig 3 (+ the unallocated loss ablation)
PYTHONPATH=src python -m survey_lens.make_ablation_figures \
    --joint-ckpt $CK \
    --recon-ckpt   $ROOT/runs/run3_recon_fov12/version_0/checkpoints/last.ckpt \
    --reg-ckpt     $ROOT/runs/run3_reg_fov12/version_0/checkpoints/last.ckpt \
    --nocross-ckpt $ROOT/runs/run3_nocross_fov12/version_0/checkpoints/last.ckpt \
    --out-dir figures/run3/paper --fov 12.0 --device cuda --workers 8

# architecture schematic (unallocated; reads its constants from model.py)
PYTHONPATH=src python -m survey_lens.make_fig2_architecture \
    --out-dir figures/run3/paper --fov 12.0
```

`--fov 12.0` is mandatory: the checkpoint does not record its FOV.

**Three scripts write into `figures/run3/paper/`** with non-colliding manifest names
(`figure_manifest{,_ablation,_fig2}.json`, `captions{,_ablation}.md`). Before the suffixes
existed, whichever ran second silently destroyed the other's captions and provenance. Any
new figure script added there needs the same treatment.

**Each figure seeds its own bootstrap generator** from `--boot-seed`, so intervals do not
depend on which other figures ran in the same process. They used to: a shared generator
meant gating one figure off silently shifted every later figure's error bars.

**Retained but not emitted:** `--with-fig4` (deconvolution) and `--with-fig5` (noise
shift). Both produce a caption explaining their status; fig 5's records that its
reblur/floor panel was wrong at Roman.

**Run 2** (FOV 20", `figures/paper/`) is frozen provenance — never regenerated or moved.
