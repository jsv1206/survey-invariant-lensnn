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
from . import surveys as SV


@torch.no_grad()
def evaluate_survey(model, items, batch_size=16, device="cpu", mask_thresh=0.01):
    """Run the model over one survey's samples (all share grid size).

    Reports each reconstruction metric two ways (HANDOFF_run1.md Step 3):
    raw (recon_l1/recon_ssim, kept for continuity -- NOT comparable across
    surveys, since peak brightness varies ~11x with pixel scale) and
    peak-normalized (both f_hat/f_gt divided by f_gt's per-sample peak before
    L1/SSIM, so surveys ARE comparable), each of the latter both masked to
    the arc region (`f_gt > mask_thresh * peak`, the headline number -- only
    2.7-4.0% of pixels clear it) and over the full frame (the hallucination
    check, ~96% trivial empty-sky agreement). `mask_frac` logs the masked
    pixel fraction per sample so the global-metric dilution is visible.
    """
    theta_pred, theta_true = [], []
    l1_sum = ssim_sum = reblur_sum = 0.0
    l1_norm_masked_sum = l1_norm_global_sum = ssim_norm_masked_sum = 0.0
    mask_frac_sum = 0.0
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

    tp = torch.cat(theta_pred).numpy()
    tt = torch.cat(theta_true).numpy()
    err = tp - tt
    return dict(
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


class _FixedReconModel:
    """model(g,h,s) stand-in that returns a FIXED reconstruction rule instead
    of a network forward pass -- lets the zeros/identity trivial baselines
    (HANDOFF_run1.md Step 4) reuse `evaluate_survey` exactly, so baseline and
    trained numbers are computed identically and are directly comparable.
    theta_E output is unused for these two (recon-only baseline rows)."""
    def __init__(self, mode):
        assert mode in ("zeros", "identity")
        self.mode = mode

    def __call__(self, g, h, s):
        f_hat = torch.zeros_like(g) if self.mode == "zeros" else g
        return dict(recon=f_hat, theta_E=torch.zeros(g.shape[0], device=g.device))


_RECON_COLS = ("recon_l1", "recon_ssim", "recon_l1_norm_masked",
              "recon_l1_norm_global", "recon_ssim_norm", "mask_frac", "reblur_mse")
_THETA_COLS = ("theta_rmse", "theta_bias", "theta_scatter")


def baseline_rows(items, batch_size=16, device="cpu"):
    """Trivial baselines (HANDOFF_run1.md Step 4): a reviewer will ask what
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
    args = ap.parse_args()

    out_prefix = args.out_prefix or f"test_set{args.test_set}"
    builder = build_test_set_1 if args.test_set == 1 else build_test_set_2

    model = load_model(args.ckpt, args.device)

    print(f"building Test Set {args.test_set}: {args.n_lenses} lenses x "
          f"{len(SV.SURVEYS)} surveys "
          f"({'no noise' if args.test_set == 1 else 'with per-survey noise'}) ...")
    test_set = builder(SV.SURVEYS, n_lenses=args.n_lenses,
                       seed=args.seed, fov=args.fov,
                       fine_scale=args.fine_scale)

    per_survey, rows = {}, []
    all_pred, all_true = [], []
    for nm, items in test_set.items():
        r = evaluate_survey(model, items, args.batch_size, args.device)
        per_survey[nm] = r
        all_pred.append(r["theta_pred"]); all_true.append(r["theta_true"])
        rows.append(dict(survey=nm, model="trained", scale=SV.SURVEYS[nm]["scale"],
                         fwhm=SV.SURVEYS[nm]["fwhm"], n=r["n"],
                         theta_rmse=r["theta_rmse"], theta_bias=r["theta_bias"],
                         theta_scatter=r["theta_scatter"],
                         recon_l1=r["recon_l1"], recon_ssim=r["recon_ssim"],
                         recon_l1_norm_masked=r["recon_l1_norm_masked"],
                         recon_l1_norm_global=r["recon_l1_norm_global"],
                         recon_ssim_norm=r["recon_ssim_norm"],
                         mask_frac=r["mask_frac"],
                         reblur_mse=r["reblur_mse"]))
        print(f"  {nm:7s} theta_rmse={r['theta_rmse']:.4f} bias={r['theta_bias']:+.4f} "
              f"ssim_norm_masked={r['recon_ssim_norm']:.3f} "
              f"l1_norm_masked={r['recon_l1_norm_masked']:.3e} "
              f"mask_frac={r['mask_frac']:.3f} reblur={r['reblur_mse']:.2e}")

        # trivial baselines (HANDOFF_run1.md Step 4): same survey, same items
        base = baseline_rows(items, args.batch_size, args.device)
        for mode in ("zeros", "identity", "prior_mean"):
            b = base[mode]
            rows.append(dict(survey=nm, model=mode, scale=SV.SURVEYS[nm]["scale"],
                             fwhm=SV.SURVEYS[nm]["fwhm"], n=r["n"],
                             **{k: b.get(k, np.nan) for k in _THETA_COLS + _RECON_COLS}))
        print(f"           identity beats zeros (masked normalized L1)? "
              f"{base['identity']['recon_l1_norm_masked'] < base['zeros']['recon_l1_norm_masked']} | "
              f"prior_mean theta_rmse={base['prior_mean']['theta_rmse']:.4f}")

    # aggregate row
    ap_, at_ = np.concatenate(all_pred), np.concatenate(all_true)
    err = ap_ - at_
    rows.append(dict(survey="ALL", model="trained", scale=np.nan, fwhm=np.nan, n=len(ap_),
                     theta_rmse=float(np.sqrt(np.mean(err ** 2))),
                     theta_bias=float(np.mean(err)),
                     theta_scatter=float(np.std(err)),
                     recon_l1=float(np.mean([per_survey[n]["recon_l1"] for n in test_set])),
                     recon_ssim=float(np.mean([per_survey[n]["recon_ssim"] for n in test_set])),
                     recon_l1_norm_masked=float(np.mean([per_survey[n]["recon_l1_norm_masked"] for n in test_set])),
                     recon_l1_norm_global=float(np.mean([per_survey[n]["recon_l1_norm_global"] for n in test_set])),
                     recon_ssim_norm=float(np.mean([per_survey[n]["recon_ssim_norm"] for n in test_set])),
                     mask_frac=float(np.mean([per_survey[n]["mask_frac"] for n in test_set])),
                     reblur_mse=float(np.mean([per_survey[n]["reblur_mse"] for n in test_set]))))

    df = pd.DataFrame(rows)
    csv_path = f"{out_prefix}_metrics.csv"
    df.to_csv(csv_path, index=False)
    fig_path = f"{out_prefix}_summary.png"
    label = "Test Set 1 (no noise)" if args.test_set == 1 else "Test Set 2 (per-survey noise)"
    make_summary_figure(per_survey, fig_path, suptitle=label)
    print(f"\nwrote {csv_path} and {fig_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
