"""Evaluate a trained checkpoint on Test Set 1 (simulated telescope data).

Test Set 1 = the five fixed survey configurations (HST, Euclid, Roman, Rubin,
DES) from `surveys.SURVEYS`, each with its own pixel scale and PSF FWHM, all
applied to the SAME underlying lens population (paired cross-survey comparison).
No noise (that is Test Set 2).

For every survey we report:
  theta_rmse / theta_bias / theta_scatter   -- Einstein-radius regression
  recon_l1 / recon_ssim                      -- reconstruction vs ground-truth sharp image
  reblur_mse                                 -- self-supervised consistency (the
                                                primary diagnostic on real data)

Outputs:
  test_set1_metrics.csv   -- one row per survey + an "ALL" aggregate
  test_set1_summary.png   -- theta_E pred-vs-true and per-survey metric bars

Usage:
  python test.py --ckpt runs/full/version_0/checkpoints/epoch=29-step=...ckpt
  python test.py --ckpt <ckpt> --n-lenses 200 --fov 20
"""

import os
import argparse
import numpy as np
import torch
import pandas as pd

from . import env_setup  # noqa: F401
from .data import build_test_set_1, build_test_set_2
from .lit_model import LitLensNet
from . import losses as Lo
from . import metrics as MET
from . import surveys as SV

# Top-octave (0.5-1.0 x Nyquist) columns -- see survey_lens/metrics.py.
# `hf_complex_err` is the HEADLINE reconstruction number, not `hf_power_err`:
# the power ratio is phase-blind and gameable (a reconstruction can put the
# right amount of power in the wrong place and score ~1.0). `f_hat = 0` scores
# hf_complex_err == 1.0 exactly, so anything above 1.0 is worse than emitting
# nothing. `hf_harm_frac_*` exposes how much of the band is the patch-decoder
# grid artifact rather than signal.
_HF_KEYS = ("hf_complex_err", "hf_coherence", "hf_power_ratio", "hf_power_err",
            "hf_complex_err_excl", "hf_coherence_excl",
            "hf_harm_frac_hat", "hf_harm_frac_gt")


@torch.no_grad()
def evaluate_survey(model, items, batch_size=16, device="cpu", mask_thresh=0.01,
                    noise_cfg=None):
    """Run the model over one survey's samples (all share grid size).

    Reports each reconstruction metric two ways:
    raw (recon_l1/recon_ssim, kept for continuity -- NOT comparable across
    surveys, since peak brightness varies ~11x with pixel scale) and
    peak-normalized (both f_hat/f_gt divided by f_gt's per-sample peak before
    L1/SSIM, so surveys ARE comparable), each of the latter both masked to
    the arc region (`f_gt > mask_thresh * peak`, the headline number -- only
    2.7-4.0% of pixels clear it) and over the full frame (the hallucination
    check, ~96% trivial empty-sky agreement). `mask_frac` logs the masked
    pixel fraction per sample so the global-metric dilution is visible.

    `noise_cfg` (dict with 'bkg_rms'/'gain', Test Set 2 only): also computes
    the reblur noise floor -- on noisy input, a
    PERFECT reconstruction cannot drive reblur_mse to zero, since
    `f_hat*h - g` still contains g's noise. Floor = mean(max(g,0)/gain +
    bkg_rms**2), both globally and over the arc mask; report reblur_mse /
    floor as the comparable diagnostic (floor itself spans ~6x across
    surveys). Below 1.0 means the model is fitting noise, not a win.

    `reblur_mse_over_floor` divides by `reblur_floor_rel`, the floor in
    `reblur_loss`'s scale-free units, NOT by the absolute `reblur_floor_global`
    -- see the comment at the accumulation site.
    """
    theta_pred, theta_true = [], []
    l1_sum = ssim_sum = reblur_sum = 0.0
    l1_norm_masked_sum = l1_norm_global_sum = ssim_norm_masked_sum = 0.0
    mask_frac_sum = 0.0
    floor_global_sum = floor_masked_sum = floor_rel_sum = 0.0
    hf_vals = {k: [] for k in _HF_KEYS}          # per-sample: ratios are heavy-tailed,
    prof_acc = None                              # so we report mean AND median
    patch = getattr(getattr(model, "net", model), "patch", 4)
    n = len(items)
    for i in range(0, n, batch_size):
        chunk = items[i:i + batch_size]
        g = torch.stack([x["g"] for x in chunk]).to(device)
        h = torch.stack([x["h"] for x in chunk]).to(device)
        f = torch.stack([x["f_gt"] for x in chunk]).to(device)
        s = torch.stack([x["s"] for x in chunk]).to(device)
        th = torch.stack([x["theta_E"] for x in chunk]).to(device)

        out = model(g, h, s)
        f_hat, th_hat = out["recon"], out["theta_E"]

        theta_pred.append(th_hat.cpu())
        theta_true.append(th.cpu())
        bs = g.shape[0]
        l1_sum += (f_hat - f).abs().mean().item() * bs
        ssim_sum += Lo.ssim(f_hat, f).item() * bs
        reblur_sum += Lo.reblur_loss(f_hat, h, g).item() * bs

        peak = f.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        f_hat_n, f_n = f_hat / peak, f / peak
        mask = f_n > mask_thresh                                   # [B,1,H,W] bool
        mask_f = mask.to(f_n.dtype)
        denom = mask_f.sum(dim=(1, 2, 3)).clamp_min(1.0)
        absdiff = (f_hat_n - f_n).abs()
        l1_masked_per_sample = (absdiff * mask_f).sum(dim=(1, 2, 3)) / denom
        l1_norm_masked_sum += l1_masked_per_sample.sum().item()
        l1_norm_global_sum += absdiff.mean(dim=(1, 2, 3)).sum().item()
        ssim_norm_masked_sum += Lo.ssim(f_hat, f, mask=mask).item() * bs
        mask_frac_sum += (mask_f.sum(dim=(1, 2, 3)) / mask_f[0].numel()).sum().item()

        # top-octave metrics -- the numbers that actually
        # adjudicate the deconvolution claim
        hm = MET.hf_power_metrics(f_hat, f, s, patch=patch)
        for k in _HF_KEYS:
            hf_vals[k].extend(hm[k].detach().cpu().tolist())
        centers, ph, pg = MET.radial_power_profile(f_hat, f, s)
        prof_acc = (centers, ph * bs, pg * bs) if prof_acc is None else (
            centers, prof_acc[1] + ph * bs, prof_acc[2] + pg * bs)

        if noise_cfg is not None:
            var = g.clamp_min(0.0) / noise_cfg["gain"] + noise_cfg["bkg_rms"] ** 2
            floor_global_sum += var.mean(dim=(1, 2, 3)).sum().item()
            floor_masked_per_sample = (var * mask_f).sum(dim=(1, 2, 3)) / denom
            floor_masked_sum += floor_masked_per_sample.sum().item()
            # ... and the SAME floor expressed in reblur_loss's units. The
            # ratio below is the headline TS2 diagnostic, and `reblur_loss` is
            # scale-free (it divides by mean(g**2)) while `var` is an absolute
            # surface-brightness variance -- dividing one by the other is a
            # units error that inflates the ratio by 1/mean(g**2) (~1e6 here,
            # which is how a sub-floor 0.13x printed as 126185x). Normalize by
            # the batch-level mean(g**2), exactly the normalizer
            # `losses.reblur_loss` applies, so numerator and denominator are
            # the same quantity.
            gsq = (g ** 2).mean().clamp_min(1e-12)
            floor_rel_sum += (var.mean(dim=(1, 2, 3)) / gsq).sum().item()

    tp = torch.cat(theta_pred).numpy()
    tt = torch.cat(theta_true).numpy()
    err = tp - tt
    out = dict(
        theta_pred=tp, theta_true=tt,
        theta_rmse=float(np.sqrt(np.mean(err ** 2))),
        theta_bias=float(np.mean(err)),
        theta_scatter=float(np.std(err)),
        recon_l1=l1_sum / n,
        recon_ssim=ssim_sum / n,
        recon_l1_norm_masked=l1_norm_masked_sum / n,
        recon_l1_norm_global=l1_norm_global_sum / n,
        recon_ssim_norm=ssim_norm_masked_sum / n,
        mask_frac=mask_frac_sum / n,
        reblur_mse=reblur_sum / n,
        n=n,
    )
    for k in _HF_KEYS:
        v = np.asarray(hf_vals[k], dtype=float)
        out[k] = float(np.mean(v))
        out[k + "_median"] = float(np.median(v))
    if prof_acc is not None:
        out["radial_profile"] = (prof_acc[0], prof_acc[1] / n, prof_acc[2] / n)
    if noise_cfg is not None:
        floor_global = floor_global_sum / n
        floor_masked = floor_masked_sum / n
        floor_rel = floor_rel_sum / n
        out.update(
            reblur_floor_global=floor_global,
            reblur_floor_masked=floor_masked,
            # scale-free floor: directly comparable to `reblur_mse`
            reblur_floor_rel=floor_rel,
            reblur_mse_over_floor=out["reblur_mse"] / max(floor_rel, 1e-30),
        )
    return out


class _FixedReconModel:
    """model(g,h,s) stand-in that returns a FIXED reconstruction rule instead
    of a network forward pass -- lets the zeros/identity trivial baselines reuse `evaluate_survey` exactly, so baseline and
    trained numbers are computed identically and are directly comparable.
    theta_E output is unused for these (recon-only baseline rows).

    `wiener` is classical Wiener deconvolution with the KNOWN PSF -- the
    baseline a reviewer will run unprompted, and the one that decides whether
    the network's deconvolution is a real contribution or an under-performing
    substitute for three lines of signal processing.
    """
    def __init__(self, mode, K=1e-3):
        assert mode in ("zeros", "identity", "wiener")
        self.mode = mode
        self.K = K

    def __call__(self, g, h, s):
        if self.mode == "zeros":
            f_hat = torch.zeros_like(g)
        elif self.mode == "identity":
            f_hat = g
        else:
            f_hat = MET.wiener_deconvolve(g, h, self.K)
        return dict(recon=f_hat, theta_E=torch.zeros(g.shape[0], device=g.device))


_RECON_COLS = (("recon_l1", "recon_ssim", "recon_l1_norm_masked",
               "recon_l1_norm_global", "recon_ssim_norm", "mask_frac", "reblur_mse")
              + _HF_KEYS + tuple(k + "_median" for k in _HF_KEYS))
_THETA_COLS = ("theta_rmse", "theta_bias", "theta_scatter")


def baseline_rows(items, batch_size=16, device="cpu"):
    """Trivial baselines: a reviewer will ask what
    the trained numbers mean, so report the same metrics for two null recon
    models (zeros: f_hat=0; identity: f_hat=g, i.e. no deconvolution at all)
    and one null regressor (prior_mean: always predict mean(theta_E), whose
    RMSE equals the prior standard deviation by construction -- any trained
    theta_rmse at or above it means the model learned nothing).

    zeros/identity report ONLY the reconstruction columns (their theta_E
    output is meaningless, not "trained on nothing" -- that would conflate
    two different null hypotheses); prior_mean reports ONLY the theta
    columns. The deconvolution claim requires beating `identity`, not merely
    `zeros`, on the masked normalized metrics.
    """
    out = {}
    for mode in ("zeros", "identity"):
        r = evaluate_survey(_FixedReconModel(mode), items, batch_size, device)
        out[mode] = {k: r[k] for k in _RECON_COLS}

    # Wiener: sweep K and report the baseline at its BEST K for each metric it
    # gets quoted against -- not one K chosen on L1 and then scored on
    # everything else.
    #
    # The headline deconvolution claim is `hf_complex_err`, but K used to be
    # selected on `recon_l1_norm_masked` alone, so Wiener was being judged on a
    # metric it had not been tuned for. That understates the baseline, and the
    # baseline is exactly what the paper's claim rests on. Two rows now:
    #   `wiener`     -- K best by masked normalized L1 (unchanged, for continuity)
    #   `wiener_hf`  -- K best by top-octave complex error (the fair comparison
    #                   for the deconvolution claim)
    # Choosing K on the evaluation set at all is already an UPPER BOUND on
    # Wiener, i.e. deliberately generous to the baseline; say so in the caption
    # rather than quietly picking a flattering K.
    #
    # The grid spans more decades than looks necessary because the optimum was
    # pinning to the smallest K offered (1e-5) at Rubin and DES -- an
    # edge-pinned baseline is an under-tuned baseline, and a referee reading a
    # "we beat Wiener" claim will check precisely this.
    sweep = {}
    # Grid runs to 1e-12 because on NOISE-FREE data there is no noise to
    # amplify, so Wiener's optimum keeps sliding toward the naive inverse
    # filter (K -> 0) until float32 FFT conditioning takes over. Stopping at
    # 1e-5, as this sweep originally did, left Wiener badly under-tuned at the
    # seeing-limited surveys: Rubin's top-octave error falls 0.315 -> 0.059 and
    # DES's 0.328 -> 0.079 once the grid is allowed to reach the floor. Beyond
    # ~1e-13 the filter diverges numerically (values in the hundreds of
    # thousands); those K are never selected since we take the min, but they
    # bound the grid so the optimum is INTERIOR rather than edge-pinned.
    # The grid also runs UP to 1e+1 because on noisy data the optimum moves the
    # other way -- Wiener needs heavy regularization there, and on Test Set 2
    # HST/Euclid were pinning at the top of a grid that stopped at 1e-1.
    for K in (10.0 ** -e for e in range(-1, 15)):
        sweep[K] = evaluate_survey(_FixedReconModel("wiener", K=K), items,
                                   batch_size, device)
    for row, key in (("wiener", "recon_l1_norm_masked"),
                     ("wiener_hf", "hf_complex_err")):
        bestK = min(sweep, key=lambda K: sweep[K][key])
        out[row] = {k: sweep[bestK][k] for k in _RECON_COLS}
        out[row]["wiener_K"] = bestK
        out[row]["wiener_K_selected_on"] = key
        ks = sorted(sweep)
        out[row]["wiener_K_at_grid_edge"] = bool(bestK in (ks[0], ks[-1]))

    theta_true = np.array([float(it["theta_E"]) for it in items])
    theta_mean = float(theta_true.mean())
    err = theta_mean - theta_true
    out["prior_mean"] = dict(theta_rmse=float(np.sqrt(np.mean(err ** 2))),
                             theta_bias=float(np.mean(err)),
                             theta_scatter=float(np.std(err)))
    return out


def load_model(ckpt, device="cpu"):
    if ckpt and os.path.exists(ckpt):
        model = LitLensNet.load_from_checkpoint(ckpt, map_location=device)
        print(f"loaded checkpoint: {ckpt}")
    else:
        # fall back to a randomly-initialized model so the script is runnable
        # before training (metrics will be meaningless but the pipeline runs).
        model = LitLensNet()
        print("WARNING: no checkpoint given/found -> using RANDOM weights "
              "(metrics are not meaningful; pass --ckpt for a real evaluation).")
    model.eval().to(device)
    return model


def make_summary_figure(per_survey, path, suptitle=None):
    import matplotlib.pyplot as plt
    try:
        apply_figure_style()  # from figure-style skill, if present in kernel
    except Exception:
        pass
    names = list(per_survey)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # (a) theta_E predicted vs true, colored by survey
    ax = axes[0]
    lo, hi = 0.4, 2.6
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=1)
    cmap = plt.get_cmap("viridis")
    for k, nm in enumerate(names):
        r = per_survey[nm]
        ax.scatter(r["theta_true"], r["theta_pred"], s=10, alpha=0.5,
                   color=cmap(k / max(1, len(names) - 1)),
                   label=f"{nm} (RMSE {r['theta_rmse']:.3f})")
    ax.set_xlabel(r"true $\theta_E$ (arcsec)")
    ax.set_ylabel(r"predicted $\theta_E$ (arcsec)")
    ax.set_title("Einstein-radius recovery per survey")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.legend(fontsize=7, loc="upper left")

    # (b) per-survey metric bars (theta RMSE and reblur)
    ax = axes[1]
    x = np.arange(len(names))
    rmse = [per_survey[n]["theta_rmse"] for n in names]
    ax.bar(x - 0.2, rmse, width=0.4, label=r"$\theta_E$ RMSE (arcsec)")
    ax2 = ax.twinx()
    reb = [per_survey[n]["reblur_mse"] for n in names]
    ax2.bar(x + 0.2, reb, width=0.4, color="C1", label="reblur MSE")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel(r"$\theta_E$ RMSE (arcsec)")
    ax2.set_ylabel("reblur MSE")
    ax.set_title("Cross-survey metrics")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="path to a .ckpt; omit for random-weight smoke run")
    ap.add_argument("--n-lenses", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--fov", type=float, default=20.0)
    ap.add_argument("--fine-scale", type=float, default=None,
                    help="global fine render grid in arcsec (LEGACY). Default "
                         "None = adaptive per-sample supersampling, which "
                         "renders the requested pixel scale exactly.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--test-set", type=int, choices=[1, 2], default=1,
                    help="1 = no noise; 2 = realistic per-survey noise on the input")
    ap.add_argument("--out-prefix", default=None,
                    help="output file prefix (default test_set{N})")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallelize test-set simulation across N processes "
                         "(the simulate_one render step only; exactly reproduces "
                         "the --workers=1 result, just faster -- see data.py's "
                         "_render_params). Single-threaded default keeps this "
                         "script's behavior/tests unchanged.")
    args = ap.parse_args()

    out_prefix = args.out_prefix or f"test_set{args.test_set}"
    builder = build_test_set_1 if args.test_set == 1 else build_test_set_2

    model = load_model(args.ckpt, args.device)

    print(f"building Test Set {args.test_set}: {args.n_lenses} lenses x "
          f"{len(SV.SURVEYS)} surveys "
          f"({'no noise' if args.test_set == 1 else 'with per-survey noise'}) ...")
    test_set = builder(SV.SURVEYS, n_lenses=args.n_lenses,
                       seed=args.seed, fov=args.fov,
                       fine_scale=args.fine_scale, n_workers=args.workers)

    per_survey, rows = {}, []
    all_pred, all_true = [], []
    for nm, items in test_set.items():
        noise_cfg = (dict(bkg_rms=SV.SURVEYS[nm]["bkg_rms"], gain=SV.SURVEYS[nm]["gain"])
                    if args.test_set == 2 else None)
        r = evaluate_survey(model, items, args.batch_size, args.device, noise_cfg=noise_cfg)
        per_survey[nm] = r
        all_pred.append(r["theta_pred"]); all_true.append(r["theta_true"])
        # build from _RECON_COLS, never a hardcoded list: a hardcoded list here
        # silently omits any newly-added metric from the TRAINED row while the
        # baselines (which go through _RECON_COLS) still have it.
        row = dict(survey=nm, model="trained", scale=SV.SURVEYS[nm]["scale"],
                  fwhm=SV.SURVEYS[nm]["fwhm"], n=r["n"],
                  theta_rmse=r["theta_rmse"], theta_bias=r["theta_bias"],
                  theta_scatter=r["theta_scatter"],
                  **{k: r[k] for k in _RECON_COLS})
        if noise_cfg is not None:
            row.update(reblur_floor_global=r["reblur_floor_global"],
                      reblur_floor_masked=r["reblur_floor_masked"],
                      reblur_floor_rel=r["reblur_floor_rel"],
                      reblur_mse_over_floor=r["reblur_mse_over_floor"])
        rows.append(row)
        floor_msg = (f" reblur/floor={r['reblur_mse_over_floor']:.2f}x"
                    if noise_cfg is not None else "")
        print(f"  {nm:7s} theta_rmse={r['theta_rmse']:.4f} bias={r['theta_bias']:+.4f} "
              f"ssim_norm_masked={r['recon_ssim_norm']:.3f} "
              f"l1_norm_masked={r['recon_l1_norm_masked']:.3e} "
              f"mask_frac={r['mask_frac']:.3f} reblur={r['reblur_mse']:.2e}{floor_msg}")
        if noise_cfg is not None and r["reblur_mse_over_floor"] < 1.0:
            print(f"           WARNING: reblur_mse below the noise floor for {nm} -- "
                  f"model may be fitting noise, not a win.")

        # trivial baselines: same survey, same items
        base = baseline_rows(items, args.batch_size, args.device)
        for mode in ("zeros", "identity", "wiener", "wiener_hf", "prior_mean"):
            b = base[mode]
            rows.append(dict(survey=nm, model=mode, scale=SV.SURVEYS[nm]["scale"],
                             fwhm=SV.SURVEYS[nm]["fwhm"], n=r["n"],
                             wiener_K=b.get("wiener_K", np.nan),
                             wiener_K_selected_on=b.get("wiener_K_selected_on", ""),
                             wiener_K_at_grid_edge=b.get("wiener_K_at_grid_edge", ""),
                             **{k: b.get(k, np.nan) for k in _THETA_COLS + _RECON_COLS}))
        # the top-octave verdict, printed where it cannot be missed
        print(f"           TOP-OCTAVE hf_complex_err (1.0 == emitting nothing): "
              f"trained={r['hf_complex_err']:.3f} identity={base['identity']['hf_complex_err']:.3f} "
              f"wiener_hf={base['wiener_hf']['hf_complex_err']:.3f} "
              f"(K={base['wiener_hf']['wiener_K']:.0e}"
              f"{', AT GRID EDGE' if base['wiener_hf']['wiener_K_at_grid_edge'] else ''})")
        print(f"           coherence: trained={r['hf_coherence']:.3f} "
              f"identity={base['identity']['hf_coherence']:.3f} | "
              f"patch-harmonic frac: recon={r['hf_harm_frac_hat']:.3f} truth={r['hf_harm_frac_gt']:.3f} | "
              f"prior_mean theta_rmse={base['prior_mean']['theta_rmse']:.4f}")
        if r["hf_complex_err"] > 1.0:
            print(f"           WARNING: {nm} top-octave reconstruction is WORSE THAN "
                  f"EMITTING NOTHING (hf_complex_err={r['hf_complex_err']:.3f} > 1.0).")

    # aggregate row
    ap_, at_ = np.concatenate(all_pred), np.concatenate(all_true)
    err = ap_ - at_
    all_row = dict(survey="ALL", model="trained", scale=np.nan, fwhm=np.nan, n=len(ap_),
                   theta_rmse=float(np.sqrt(np.mean(err ** 2))),
                   theta_bias=float(np.mean(err)),
                   theta_scatter=float(np.std(err)),
                   # again from _RECON_COLS, so new metrics can never be dropped here
                   **{k: float(np.mean([per_survey[n][k] for n in test_set]))
                      for k in _RECON_COLS})
    if args.test_set == 2:
        all_row.update(
            reblur_floor_global=float(np.mean([per_survey[n]["reblur_floor_global"] for n in test_set])),
            reblur_floor_masked=float(np.mean([per_survey[n]["reblur_floor_masked"] for n in test_set])),
            reblur_floor_rel=float(np.mean([per_survey[n]["reblur_floor_rel"] for n in test_set])),
            reblur_mse_over_floor=float(np.mean([per_survey[n]["reblur_mse_over_floor"] for n in test_set])))
    rows.append(all_row)

    df = pd.DataFrame(rows)
    csv_path = f"{out_prefix}_metrics.csv"
    df.to_csv(csv_path, index=False)

    # radial power profiles are arrays, so they get their own file (Step 10 Fig 4a)
    prof_rows = []
    for nm in test_set:
        prof = per_survey[nm].get("radial_profile")
        if prof is None:
            continue
        centers, ph, pg = prof
        for u, a, b in zip(centers, ph, pg):
            prof_rows.append(dict(survey=nm, model="trained", u_over_nyquist=float(u),
                                  P_hat=float(a), P_gt=float(b)))
    if prof_rows:
        prof_path = f"{out_prefix}_radial_profiles.csv"
        pd.DataFrame(prof_rows).to_csv(prof_path, index=False)
        print(f"wrote {prof_path}")
    fig_path = f"{out_prefix}_summary.png"
    label = "Test Set 1 (no noise)" if args.test_set == 1 else "Test Set 2 (per-survey noise)"
    make_summary_figure(per_survey, fig_path, suptitle=label)
    print(f"\nwrote {csv_path} and {fig_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
