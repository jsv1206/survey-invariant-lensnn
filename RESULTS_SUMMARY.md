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

### 4.1 Survey invariance — accuracy is flat across a 5.3× pixel-scale range

θ_E RMSE, Test Set 1 (`fig3_survey_invariance_data.csv`):

| HST | Euclid | Roman | Rubin | DES | prior-mean null |
|---|---|---|---|---|---|
| 0.0170″ | 0.0164″ | 0.0215″ | 0.0227″ | 0.0306″ | **0.618″** |

**20–38× better than the null**, varying only **1.9×** across a **5.3×** pixel-scale
range. Report the residual trend honestly: DES is 1.9× worse than Euclid and that trend
correlates with pixel scale — "flat" means "nearly flat", not "exactly flat".

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

---

## 5. Ablations

### 5.1 Axis 1 — PSF cross-attention (NEW; Representations main text)

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

### 5.2 Axis 2 — loss mode (Sim2Science appendix; Representations appendix)

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

### 6.1 Sim2Science — *leveraging imperfect simulators against real data*

**Thesis.** The reblur consistency loss needs no ground truth, so the same objective and
the same diagnostic that train the model on simulations also run unchanged on real images;
survey invariance is robustness to per-instrument simulator mis-specification.

**Lead with:** §4.3 (noise shift) and §4.2's TS2 column. The model is trained on a
noise-free simulator and evaluated on realistic per-survey noise with no retraining and no
augmentation — and it beats an oracle-tuned classical filter by 4.0–8.7× where that filter's
noise assumption breaks.

**Figure order:** 1 hero · 2 architecture · 3 invariance · 4 deconvolution · **5 noise
shift**. Appendix: A1 extra lenses, A2 loss ablation, A3 cross-attention.

**The honest-reporting angle is an asset here.** HST fails under noise (1.029 > 1.0) and we
report it; the reblur diagnostic catches it *without ground truth* at 0.92× its floor.
"The noise-free simulator does not transfer without augmentation at HST" is a legitimate
and publishable Sim2Science result.

### 6.2 Representations — *symmetry / geometry / structure in learned representations*

**Thesis.** One encoder, invariant to pixel scale and PSF across a 5.3× range, with **RoPE
positions in arcseconds rather than pixel indices** as the mechanism — a symmetry
statement: a 0.1″ offset must rotate identically for HST and Rubin.

**Lead with:** §4.1 (invariance) and §5.1 (cross-attention). The cross-attention result
says what the representation *needs*: explicit access to the instrument response for
high-frequency recovery, and nothing extra for the geometric quantity. The σ_px trend
(ρ = −0.90) turns that from an ablation into a mechanism.

**Figure order:** 1 hero · 2 architecture · 3 invariance · 4 deconvolution ·
**5 cross-attention**. Appendix: A1 extra lenses, A2 noise shift, A3 loss ablation.

**Strongest available sentence:** *the two heads dissociate.* The same architectural
removal destroys deconvolution (2.75× mean, up to 9.98×) and leaves Einstein-radius
regression untouched — a double dissociation that also controls the capacity confound.

---

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

## 8. Provenance

All figures re-runnable from checkpoint + cached data with no manual steps. Each figure
ships `figN_*_data.csv` with exactly the plotted numbers, plus a manifest recording
checkpoint path and hash, dataset digest, git commit and script version. Three scripts
write into `figures/run3/paper/` with non-colliding manifest names
(`make_paper_figures.py`, `make_ablation_figures.py`, `make_fig2_architecture.py`).
Run 2 (FOV 20″, `figures/paper/`) is frozen provenance and is never regenerated.
