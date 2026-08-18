# Dataset specification (verified against code; updated Aug 17 2026 after fixes 1-2)

## Training / validation set — continuous priors, NOISE-FREE
Per-sample draws (`sim_lens.sample_params`):
| quantity | prior |
|---|---|
| theta_E (label) | U(0.5, 2.5) arcsec |
| PSF FWHM | logU(0.08, 1.1) arcsec  (widened -- fix 2) |
| pixel scale s | logU(0.04, 0.32) arcsec/px  (widened -- fix 2) |
| mass | EPL (gamma=2) + SHEAR; q~U(0.7,0.95); shear~N(0,0.05) |
| source | elliptical Sersic; n~U(1,3); R_eff~U(0.1,0.35)"; offset~N(0,0.15)" |

FOV fixed 20 arcsec; lens center = origin = center of image AND PSF stamp.
Pipeline: render sharp on an ADAPTIVE fine grid (fine = s/factor, factor chosen
per sample by choose_supersample_factor: factor>=4 and PSF sigma>=3 fine px)
-> convolve Gaussian PSF at fine scale -> area-resample to native s
(convolve-then-resample = physically correct). The adaptive grid reproduces the
requested s EXACTLY; a global fine_scale float restores the legacy quantizing path.
Outputs: f_gt (sharp target), g (blurred input), h (PSF stamp), s (scalar), theta_E.

## Test Set 1 — fixed real-survey configs, noise-free
PAIRED design: 200 lens param sets drawn ONCE, each rendered through all 5 surveys
=> cross-survey differences attributable to the telescope, not the lens sample.

Measured with the adaptive grid (fix 1) -- realized == requested for all five:

| survey | scale (")| FWHM (")| realized scale | npix | image tokens (patch=4) | PSF sigma (native px) |
|---|---|---|---|---|---|---|
| HST    | 0.050 | 0.10 | 0.0500 | 400 | 10000 | 0.85 |
| Euclid | 0.100 | 0.16 | 0.1000 | 200 |  2500 | 0.68 |
| Roman  | 0.110 | 0.12 | 0.1100 | 182 |  2025 | 0.46 |
| Rubin  | 0.200 | 0.70 | 0.2000 | 100 |   625 | 1.49 |
| DES    | 0.263 | 0.90 | 0.2630 |  76 |   361 | 1.45 |

5 distinct grids (was 4). Cost: 1.7 s per lens for all 5 surveys (1 core).

## Test Set 2 — Test Set 1 + per-survey noise on INPUT only
var(pixel) = max(signal,0)/gain + bkg_rms^2 ; f_gt and h stay clean.
Training is noise-free => genuine distribution shift (the Sim2Science money plot).

## Test Set 3 — real cutouts (COSMOS HST / DES / Euclid Q1). NOT YET BUILT.
Keep qualitative for the workshop: 2-3 lenses, reblur residual as diagnostic.

---

# FOUR ISSUES — 1 and 2 FIXED (Aug 2026), 3 and 4 STILL OPEN
Regression tests: tests/test_scale_fidelity.py (20 tests pass).

1. [FIXED] PIXEL SCALES WERE SILENTLY QUANTIZED. simulate_one rounded s to a
   multiple of a global fine_scale=0.025. Roman 0.11 -> 0.10 (IDENTICAL to Euclid);
   DES 0.263 -> 0.275 (+4.6%). Only 4 distinct grids, not 5.
   FIX IMPLEMENTED: adaptive per-sample supersampling, now the default
   (fine_scale=None). choose_supersample_factor(s, fwhm) picks an integer factor
   subject to factor >= 4 and PSF sigma >= 3 fine px; the render grid is
   fine = s/factor, so the requested scale is reproduced EXACTLY. All 5 surveys
   verified at 0.00% error => 5 distinct grids.
   NOTE the earlier proposed fix (global fine_scale=0.005) was REJECTED on measured
   cost: 27-54x per sample (10k samples 27.8 h vs 0.68 h, 1 core) because cost
   scales as (FOV/fine)^2 and it over-renders the coarse surveys worst (DES 54x).
   Adaptive costs ~2x (10k = 1.4 h, 1 core) and holds quality constant instead.
   Legacy quantizing behaviour is preserved by passing an explicit float.

2. [FIXED] HST & DES SAT ON THE PRIOR BOUNDARIES. HST FWHM 0.10 = lower endpoint
   of logU(0.1,0.9) and scale 0.05 = lower endpoint of logU(0.05,0.30); DES FWHM
   0.90 = upper endpoint => one-sided training support at both extremes, so an
   apparent generalization failure could be a prior-coverage artifact.
   FIX IMPLEMENTED: priors widened to FWHM logU(0.08, 1.1)", scale
   logU(0.04, 0.32)"/px. Verified over 20k draws -- all five surveys interior:
   percentile (fwhm, scale) = HST (8.4, 10.8), Euclid (26.4, 43.9),
   Roman (15.3, 48.7), Rubin (82.8, 77.9), DES (92.3, 90.9).

3. EUCLID & ROMAN PSF STAMPS UNDER-SAMPLED. PSF sigma in native px:
   HST 0.85, Euclid 0.68, Roman 0.46, Rubin 1.49, DES 1.45
   (recomputed after fix 1; Roman/DES changed because their grids are now exact). Below ~0.8 px the
   Gaussian stamp barely resolves its own kernel, so h misrepresents the blur and
   the reblur loss ||f_hat*h - g||^2 inherits that error.
   FIX: sample h on a finer grid than the image (architecture allows: PSF token
   stream has its own coordinates), or document as a limitation.

4. TOKEN COUNTS DOMINATE GPU BUDGET. HST = 10k image + 10k PSF tokens (~1e8
   attention scores/head); 30x sequence-length range across surveys; may OOM
   without memory-efficient attention.
   FIX options: FOV 10" (4x fewer tokens), larger patch at fine scales, or confirm
   PyTorch SDPA flash path is active.
   >> SMOKE-TEST AT HST SCALE EXPLICITLY: a random draw usually gives a cheap grid
      and hides this.
