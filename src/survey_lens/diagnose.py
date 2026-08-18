"""Visual diagnostics for a trained checkpoint on Test Set 1.

Two kinds of output:

1) Per-survey image panels (one figure per example lens). For each survey, a row:
     sharp f_gt | PSF-convolved input g | reconstruction f_hat |
     recon residual (f_hat - f_gt) | reblur residual (f_hat * h - g)
   The recon residual is the supervised diagnostic (needs ground truth); the
   reblur residual is the self-supervised one that also applies to real data --
   small, structureless reblur residual = good reconstruction.

2) Regression scatter: true vs predicted Einstein radius across ALL test
   samples, colored by survey, with the y=x line and per-survey RMSE.

Usage:
  python diagnose.py --ckpt runs/full/version_0/checkpoints/epoch=...ckpt
  python diagnose.py --ckpt <ckpt> --n-examples 3 --n-lenses 200
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from . import env_setup  # noqa: F401
from .data import build_test_set_1, build_test_set_2
from . import losses as Lo
from . import surveys as SV
from .test import load_model, evaluate_survey   # reuse loader + evaluator


def _center_crop(img2d, s, half_arcsec):
    """Crop a 2D array to the central +-half_arcsec for display."""
    n = img2d.shape[0]
    half = int(round(half_arcsec / s))
    c = n // 2
    a, b = max(0, c - half), min(n, c + half)
    return img2d[a:b, a:b]


@torch.no_grad()
def _run_one(model, item, device):
    g = item["g"][None].to(device)
    h = item["h"][None].to(device)
    s = item["s"][None].to(device)
    out = model(g, h, s)
    f_hat = out["recon"]
    g_reblur = Lo.reblur(f_hat, h)            # f_hat * PSF
    return dict(
        f_gt=item["f_gt"][0].numpy(),
        g=item["g"][0].numpy(),
        f_hat=f_hat[0, 0].cpu().numpy(),
        reblur=g_reblur[0, 0].cpu().numpy(),
        s=float(item["s"]),
        theta_true=float(item["theta_E"]),
        theta_pred=float(out["theta_E"][0].cpu()),
    )


def image_panels(model, test_set, idx, device, crop_arcsec=4.0, set_label="Test Set 1"):
    """One figure for example lens `idx`: rows=surveys, cols=5 image views."""
    names = list(test_set)
    col_titles = ["sharp $f_{gt}$", "input $g$ (PSF-conv)", "recon $\\hat f$",
                  "recon resid $\\hat f - f_{gt}$", "reblur resid $\\hat f * h - g$"]
    fig, axes = plt.subplots(len(names), 5, figsize=(13.5, 2.5 * len(names)))
    if len(names) == 1:
        axes = axes[None, :]
    for r, nm in enumerate(names):
        d = _run_one(model, test_set[nm][idx], device)
        s = d["s"]
        sharp = _center_crop(d["f_gt"], s, crop_arcsec)
        inp = _center_crop(d["g"], s, crop_arcsec)
        rec = _center_crop(d["f_hat"], s, crop_arcsec)
        res = _center_crop(d["f_hat"] - d["f_gt"], s, crop_arcsec)
        rbl = _center_crop(d["reblur"] - d["g"], s, crop_arcsec)
        # shared positive scale for the three "image" panels; symmetric for residuals
        vmax = max(sharp.max(), inp.max(), rec.max(), 1e-6)
        rmax = max(abs(res).max(), abs(rbl).max(), 1e-6)
        panels = [(sharp, "magma", 0, vmax), (inp, "magma", 0, vmax),
                  (rec, "magma", 0, vmax), (res, "RdBu_r", -rmax, rmax),
                  (rbl, "RdBu_r", -rmax, rmax)]
        for c, (im, cmap, lo, hi) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(im, origin="lower", cmap=cmap, vmin=lo, vmax=hi)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{nm}\n{s:.3f}\"/px\nFWHM {SV.SURVEYS[nm]['fwhm']:.2f}\"",
                              fontsize=8)
        # annotate theta on the recon column
        axes[r, 2].set_title(
            (col_titles[2] + "\n" if r == 0 else "") +
            f"$\\theta_E$: {d['theta_true']:.2f}$\\to${d['theta_pred']:.2f}",
            fontsize=8)
    fig.suptitle(f"{set_label} diagnostics — example lens #{idx} "
                 f"(central {2*crop_arcsec:.0f}\")", fontsize=11)
    fig.tight_layout()
    return fig


def regression_panel(per_survey, set_label="Test Set 1"):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lo, hi = 0.4, 2.6
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=1, label="$y=x$")
    cmap = plt.get_cmap("viridis")
    names = list(per_survey)
    for k, nm in enumerate(names):
        r = per_survey[nm]
        ax.scatter(r["theta_true"], r["theta_pred"], s=12, alpha=0.5,
                   color=cmap(k / max(1, len(names) - 1)),
                   label=f"{nm}  RMSE={r['theta_rmse']:.3f}")
    ax.set_xlabel(r"true $\theta_E$ (arcsec)")
    ax.set_ylabel(r"predicted $\theta_E$ (arcsec)")
    ax.set_title(f"Einstein-radius regression ({set_label})")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-examples", type=int, default=3, help="# lenses to show as image panels")
    ap.add_argument("--n-lenses", type=int, default=200, help="# lenses for the regression scatter")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--fov", type=float, default=20.0)
    ap.add_argument("--fine-scale", type=float, default=None,
                    help="global fine render grid in arcsec (LEGACY). Default "
                         "None = adaptive per-sample supersampling, which "
                         "renders the requested pixel scale exactly.")
    ap.add_argument("--crop-arcsec", type=float, default=4.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--test-set", type=int, choices=[1, 2], default=1,
                    help="1 = no noise; 2 = realistic per-survey noise on the input")
    ap.add_argument("--out-prefix", default=None, help="default diag_set{N}")
    args = ap.parse_args()

    out_prefix = args.out_prefix or f"diag_set{args.test_set}"
    builder = build_test_set_1 if args.test_set == 1 else build_test_set_2

    model = load_model(args.ckpt, args.device)
    print(f"building Test Set {args.test_set}: {args.n_lenses} lenses x "
          f"{len(SV.SURVEYS)} surveys "
          f"({'no noise' if args.test_set == 1 else 'with per-survey noise'}) ...")
    test_set = builder(SV.SURVEYS, n_lenses=args.n_lenses,
                       seed=args.seed, fov=args.fov, fine_scale=args.fine_scale)

    set_label = f"Test Set {args.test_set}"
    # image panels for the first n_examples lenses
    saved = []
    for idx in range(min(args.n_examples, args.n_lenses)):
        fig = image_panels(model, test_set, idx, args.device, args.crop_arcsec,
                           set_label=set_label)
        path = f"{out_prefix}_images_lens{idx}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        print("wrote", path)

    # regression scatter across all surveys
    per_survey = {nm: evaluate_survey(model, items, args.batch_size, args.device)
                  for nm, items in test_set.items()}
    fig = regression_panel(per_survey, set_label=set_label)
    rpath = f"{out_prefix}_regression.png"
    fig.savefig(rpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", rpath)
    for nm, r in per_survey.items():
        print(f"  {nm:7s} theta_rmse={r['theta_rmse']:.4f} bias={r['theta_bias']:+.4f}")


if __name__ == "__main__":
    main()
