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
def evaluate_survey(model, items, batch_size=16, device="cpu"):
    """Run the model over one survey's samples (all share grid size)."""
    theta_pred, theta_true = [], []
    l1_sum = ssim_sum = reblur_sum = 0.0
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
        reblur_mse=reblur_sum / n,
        n=n,
    )


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
        rows.append(dict(survey=nm, scale=SV.SURVEYS[nm]["scale"],
                         fwhm=SV.SURVEYS[nm]["fwhm"], n=r["n"],
                         theta_rmse=r["theta_rmse"], theta_bias=r["theta_bias"],
                         theta_scatter=r["theta_scatter"],
                         recon_l1=r["recon_l1"], recon_ssim=r["recon_ssim"],
                         reblur_mse=r["reblur_mse"]))
        print(f"  {nm:7s} theta_rmse={r['theta_rmse']:.4f} bias={r['theta_bias']:+.4f} "
              f"ssim={r['recon_ssim']:.3f} reblur={r['reblur_mse']:.2e}")

    # aggregate row
    ap_, at_ = np.concatenate(all_pred), np.concatenate(all_true)
    err = ap_ - at_
    rows.append(dict(survey="ALL", scale=np.nan, fwhm=np.nan, n=len(ap_),
                     theta_rmse=float(np.sqrt(np.mean(err ** 2))),
                     theta_bias=float(np.mean(err)),
                     theta_scatter=float(np.std(err)),
                     recon_l1=float(np.mean([per_survey[n]["recon_l1"] for n in test_set])),
                     recon_ssim=float(np.mean([per_survey[n]["recon_ssim"] for n in test_set])),
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
