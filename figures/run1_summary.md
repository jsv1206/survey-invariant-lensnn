> # ⚠ CORRECTION (2026-08-19) — read this before anything below
>
> **The reconstruction conclusions in this document are retracted.** Section 2 and
> bullet 2 of "Bottom line for the paper" are **SUPERSEDED**. The θ_E results
> (§1, §3, and the whole Test Set 2 section) are unaffected and stand.
>
> **What went wrong.** The deconvolution verdict was adjudicated with masked
> normalized L1 and SSIM — precisely the metrics `CLAUDE.md:187-189` and
> `HANDOFF_run1.md:418-430` say cannot decide it ("Global PSNR hides the
> deconvolution effect… Deconvolution means beating `identity` in the top octave
> specifically"). The prescribed metric, top-octave power-spectrum error, had
> never been implemented — it was specified in Step 10, which was never reached.
> It now exists in `survey_lens/metrics.py`.
>
> **The measured truth** (200-lens Test Set 1, top-octave complex error; `f_hat=0`
> scores exactly **1.0**, so anything above 1.0 is *worse than emitting nothing*):
>
> | survey | trained | identity | Wiener | trained coherence | identity coherence |
> |---|---|---|---|---|---|
> | HST | **2.705** | 0.589 | 0.026 | 0.347 | 0.900 |
> | Euclid | **1.354** | 0.411 | 0.036 | 0.434 | 0.932 |
> | Roman | 0.949 | 0.166 | 0.093 | 0.444 | 0.976 |
> | Rubin | 0.940 | 0.961 | 0.315 | 0.281 | 0.615 |
> | DES | 0.937 | 0.959 | 0.328 | 0.285 | 0.587 |
>
> - `identity` — doing nothing — beats the trained model in **all five** surveys.
> - HST and Euclid are **worse than emitting nothing**.
> - A three-line classical Wiener filter with the same known PSF beats the
>   network by 3-30x everywhere.
>
> **The most dangerous sentence in this document is not the one §2 flags.** §2's
> verdict happens to be directionally right. The genuinely *false* claims are in
> §2's interpretation and in "Bottom line" bullet 2: *"for Rubin/DES the network's
> deconvolution genuinely helps"* and *"the model deconvolves effectively where
> the PSF is a real obstacle"*. Measured at DES, top-octave coherence is 0.285
> for the network against 0.419-0.587 for identity. **The network
> under-deconvolves worst exactly where this document claims it succeeds.** That
> sentence was headed for the paper.
>
> The "Qualitative" section's inference of "genuine deconvolution, not just
> denoising" from eyeballing panels is likewise contradicted by the numbers.
>
> **Root causes** (all measured, see the plan file): the decoder had no residual
> path to `identity`, a LayerNorm before the head destroyed absolute amplitude,
> no non-negativity (~40% of output pixels negative), raw un-normalized input
> left image content at 0.6% of token energy at HST vs 6.5% at DES, the loss was
> 92% SSIM with the headline reblur term at 0.42%, and half the RoPE frequency
> bank was silently dead. Run 2 addresses these.

# Run 1 summary — Test Set 1 (no noise) + Test Set 2 (noise shift)

Checkpoint: `runs/run1_full/version_0/checkpoints/epoch=54-step=79585.ckpt`
(best `val/loss`, out of 60 epochs; training wall time ~6h22m on the
sandbox's A100 MIG 2g.20gb, bf16-mixed precision). Full loss, cross-attention
on, joint mode, native per-survey pixel scale (no resampling), trained on the
frozen `$ROOT/data/train` cache (n=10000) with `--survey-val` (n=2000).

Test Set 1: 200 lens/source parameter sets, each rendered through all 5
surveys (paired design), no noise. Metrics: `figures/run1_test_set1_metrics.csv`.
Figures: `run1_test_set1_summary.png` (θ_E scatter + RMSE/reblur bars),
`run1_diag_set1_images_lens{0,1,2}.png` (per-survey reconstruction panels),
`run1_diag_set1_regression.png`.

## 1. Does θ_E RMSE beat the prior-mean baseline in every survey?

**Yes, by 8.5x-17.4x.**

| survey | θ_E RMSE (") | prior-mean RMSE (") | improvement |
|---|---|---|---|
| HST    | 0.0356 | 0.6184 | 17.4x |
| Euclid | 0.0358 | 0.6184 | 17.3x |
| Roman  | 0.0359 | 0.6184 | 17.2x |
| Rubin  | 0.0504 | 0.6184 | 12.3x |
| DES    | 0.0728 | 0.6184 | 8.5x  |

The prior-mean baseline (always predict the population mean θ_E) has RMSE
equal to the prior's standard deviation by construction; every survey beats
it by nearly an order of magnitude or more. This is the paper's strongest
single result.

## 2. Does the reconstruction beat `identity` (masked, normalized) in every survey?

> **SUPERSEDED — see the Correction at the top.** Adjudicated with a metric the
> project docs disallow for this purpose. The interpretation below ("for
> Rubin/DES the network's deconvolution genuinely helps") is measurably FALSE.

**No — mixed, and the failure mode is not the one anticipated.** The
concern going in was that a network might merely denoise/smooth without
doing real deconvolution; what actually happened is closer to the opposite
for the least-blurred surveys.

Masked normalized **L1** (`recon_l1_norm_masked`, lower is better):

| survey | trained | identity | trained wins? |
|---|---|---|---|
| HST    | 0.0211 | 0.0064 | no (identity 3.3x better) |
| Euclid | 0.0321 | 0.0132 | no (identity 2.4x better) |
| Roman  | 0.0294 | 0.0087 | no (identity 3.4x better) |
| Rubin  | 0.0593 | 0.0672 | **yes** (trained 1.13x better) |
| DES    | 0.0764 | 0.0848 | **yes** (trained 1.11x better) |

Masked normalized **SSIM** (`recon_ssim_norm`, higher is better):

| survey | trained | identity | trained wins? |
|---|---|---|---|
| HST    | 0.902 | 0.987 | no |
| Euclid | 0.858 | 0.973 | no |
| Roman  | 0.877 | 0.989 | no |
| Rubin  | 0.636 | 0.651 | no (close, identity still ahead) |
| DES    | 0.582 | 0.573 | **yes** (barely) |

**Interpretation:** for the three sharp/space surveys (HST, Euclid, Roman),
the PSF blur is already mild (σ 0.46-0.85 native px), so `identity` — i.e.
literally returning the blurred input unchanged — is already very close to
`f_gt`, and the trained reconstruction adds a small amount of residual error
on top of that near-perfect starting point, landing slightly worse on the
pixel-level metrics even though the reconstruction panels (see below) look
visually excellent and θ_E is recovered accurately. For the two
seeing-limited ground surveys (Rubin, DES), where the input is visibly much
more degraded, the network's deconvolution genuinely helps and beats
`identity` on L1 (Rubin, DES) and narrowly on SSIM (DES only).

This means the reconstruction head is not yet earning its keep on the easy
cases — likely a multi-task balance issue (θ_E regression is clearly
dominating what the shared trunk learns to prioritize) rather than a
broken deconvolution mechanism, since the qualitative panels show the
network recovering ring structure well even at DES's SNR. Worth investigating
before claiming a blanket "deconvolution" result in the paper; the honest
framing is "the model deconvolves effectively where the PSF is a real
obstacle (ground-based, seeing-limited) and is roughly neutral where it
mostly isn't (space-based)."

## 3. Survey-invariance: how much does performance vary across the 5.3x pixel-scale range?

- **θ_E RMSE**: 0.0356" (HST) -> 0.0728" (DES), a **2.05x** spread across
  the 5.3x pixel-scale range (0.05"/px -> 0.263"/px) — not flat, but far
  more contained than the scale range itself, and every survey still beats
  the prior-mean baseline by 8x+. HST/Euclid/Roman (the three sharpest
  surveys, spanning only 0.050-0.110"/px) cluster tightly at 0.0356-0.0359",
  effectively invariant among themselves; the degradation is concentrated
  in the two seeing-limited ground surveys.
- **Reconstruction (masked normalized SSIM)**: degrades more than θ_E does,
  0.90 (HST) -> 0.58 (DES), tracking PSF FWHM in native pixels more than
  raw pixel scale (CLAUDE.md bug 3: Euclid/Roman PSF stamps are
  under-sampled, which likely also contributes to their reconstruction
  numbers being weaker than HST's despite similar pixel scale).
- **Reblur residual** (`reblur_mse`, self-supervised, no ground truth
  needed): rises monotonically with pixel scale, 7.3e-7 (HST) to 2.3e-4
  (DES) — small in absolute terms throughout (Test Set 1 is noise-free, so
  this reflects only reconstruction imperfection, not a noise floor; that
  comparison is Test Set 2's job).

## Qualitative (diagnose.py panels, 3 example lenses)

> **SUPERSEDED in part — see the Correction at the top.** The inference of
> "genuine deconvolution, not just denoising" is contradicted by the
> top-octave measurements.

All three example lenses show the same pattern: HST/Euclid/Roman
reconstructions are visually near-identical to the sharp truth with residual
error concentrated at the single brightest knot; Rubin/DES inputs are
visibly heavily blurred (the ring is barely distinguishable from a blob),
yet the reconstruction still recovers a recognizable ring/arc structure
close to the true morphology, and θ_E predictions stay close to truth in
every panel (e.g. lens #0: true 0.95" -> predicted 0.95-0.99" across all 5
surveys). This qualitatively supports "genuine deconvolution, not just
denoising" for the hard cases, consistent with the quantitative L1 result
for Rubin/DES above.

## Bottom line for the paper

- **Headline result: θ_E regression is strongly survey-invariant** (8.5-17x
  over baseline in every survey, RMSE varying only 2x across a 5.3x
  pixel-scale range). This is solid and should lead the results section.
- **[SUPERSEDED — see Correction] Reconstruction/deconvolution claim needs a qualified statement**, not a
  blanket one: real, measurable deconvolution gain shows up specifically
  where the PSF is a real obstacle (Rubin, DES); for the sharp surveys the
  network is architecturally capable (panels look excellent) but the
  reconstruction loss term isn't currently winning over doing nothing on
  the pixel metrics. Worth a `--loss-mode` ablation (recon_only vs joint)
  to check whether this is a multi-task weighting effect before the paper
  deadline, if time allows — flagged here rather than investigated further,
  since ablations are explicitly out of scope for Run 1.

---

# Test Set 2 — noise-shift robustness (Step 9b)

Same checkpoint, same 200-lens population (paired with Test Set 1), no
retraining, no noise augmentation. Noise applied to model input `g` only
(`f_gt`/`h` stay clean) via each survey's `bkg_rms`/`gain`. This is the
sim-to-real evidence the workshop theme cares about: a noise-free simulator
evaluated against more realistic data. Metrics:
`figures/run1_test_set2_metrics.csv`.

## θ_E RMSE: TS1 -> TS2 (") — the headline sim-to-real result

| survey | TS1 | TS2 | delta | % change |
|---|---|---|---|---|
| HST    | 0.0356 | 0.0364 | +0.0008 | +2.2% |
| Euclid | 0.0358 | 0.0363 | +0.0005 | +1.4% |
| Roman  | 0.0359 | 0.0362 | +0.0003 | +0.9% |
| Rubin  | 0.0504 | 0.0504 | -0.0000 | -0.02% |
| DES    | 0.0728 | 0.0722 | -0.0005 | -0.8% |

**θ_E regression is essentially unaffected by the noise shift** — every
survey stays within ~2% of its noise-free RMSE, still 8x+ better than the
prior-mean baseline (0.618") in every case. This is a strong, genuinely
surprising sim-to-real result: a model trained on a noise-free simulator
transfers its Einstein-radius recovery to noisy data with no measurable
degradation. Worth highlighting prominently — this is exactly the
"imperfect-simulator-against-real-data" story the venue is looking for.

## Reblur residual vs the noise floor

A perfect reconstruction cannot drive `reblur_mse` to zero on noisy data —
the floor is `mean(max(g,0)/gain + bkg_rms**2)`. Reporting the dimensionless
`reblur_mse / floor` (the raw floor spans ~8x across surveys: 6.6e-7 HST to
5.3e-6 DES):

| survey | reblur_mse | floor (global) | ratio |
|---|---|---|---|
| HST    | 1.37e-6 | 6.59e-7 | 2.08x |
| Euclid | 1.54e-5 | 1.28e-6 | 12.04x |
| Roman  | 3.08e-5 | 8.73e-7 | 35.31x |
| Rubin  | 9.95e-5 | 3.14e-6 | 31.66x |
| DES    | 2.31e-4 | 5.29e-6 | 43.67x |

**All ratios are well above 1.0 — the model is not fitting noise anywhere**
(the specific failure mode this diagnostic exists to catch). HST sits
closest to its floor (2.08x), consistent with its reconstruction already
being closest to ground truth; the other four surveys sit substantially
above their floor, meaning there is real, un-recovered high-frequency
structure in the reblur residual beyond what noise alone would explain —
consistent with the Test Set 1 finding that reconstruction quality (not
just θ_E) still has room to improve, particularly for Euclid/Roman (bug 3:
under-sampled PSF stamps) and Rubin/DES (genuinely harder deconvolution).

## Reconstruction (masked normalized SSIM): TS1 -> TS2

| survey | TS1 | TS2 | delta |
|---|---|---|---|
| HST    | 0.902 | 0.898 | -0.004 |
| Euclid | 0.858 | 0.857 | -0.001 |
| Roman  | 0.877 | 0.878 | +0.001 |
| Rubin  | 0.636 | 0.639 | +0.003 |
| DES    | 0.582 | 0.584 | +0.002 |

Negligible change, within noise of the TS1 measurement itself — the
reconstruction metrics are effectively flat under the noise shift too (in
either direction), consistent with `identity` (returning the noisy input
directly) also barely changing between TS1/TS2 on these surveys' SNR.

## Honest-reporting note

No tuning was done on Test Set 2 (per HANDOFF_run1.md's explicit rule). The
result reported here — near-total robustness to the noise shift — is the
actual measurement, not a cherry-picked one; it happens to be a strong
result rather than a cautionary one, but the same reblur-floor diagnostic
would have caught it plainly if the model had instead failed to transfer.

## Bottom line, updated

The **noise-shift robustness of θ_E regression is arguably this run's
strongest sim-to-real result** and should be promoted alongside (or ahead
of) the plain Test Set 1 numbers in the paper — it directly demonstrates
the workshop's "imperfect simulator against real data" theme: a model
trained with zero noise augmentation generalizes to realistic per-survey
noise with <2.2% RMSE degradation in the worst case.

**Per HANDOFF_run1.md: do not proceed to ablations or Test Set 3 (real
data).** Steps 0-9b are complete; Step 10 (paper figures) is next.
