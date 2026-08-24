"""Fig 2 -- architecture schematic.

    PYTHONPATH=src python -m survey_lens.make_fig2_architecture \
        --out-dir figures/run3/paper \
        --mirror-dir $ROOT/figures/run3/paper

`HANDOFF_run1.md` Step 10 called this figure "not data-dependent; assemble by
hand". It is drawn programmatically here instead, for three reasons that all
bite a hand-assembled PNG:

  * the output contract (5.5in width, nothing below 6pt, PDF+PNG, a `_data.csv`,
    a manifest entry, the strict `geometry_check`) applies to EVERY figure, and a
    hand-edited raster satisfies none of it;
  * the architecture constants shown in the boxes (dim, depth, heads, patch,
    theta_ref, rope_base) must match `model.py`, and a drawing cannot be
    re-derived when they change -- here they are READ FROM THE MODEL, so the
    figure cannot silently drift from the code it depicts;
  * the per-survey token counts annotated at the bottom are computed from
    `surveys.py` at the stated FOV rather than typed in, and they are
    FOV-dependent (CLAUDE.md: npix parity and token count both change between
    20" and 12").

The three slides in `figures/source_slides/` are pedagogical talk material
(RoPE mechanics, the forward/inverse problem, why L1 alone is insufficient) --
none of them is this block diagram, so nothing is being duplicated.

The claim the figure has to land: **the pixel-scale scalar `s` enters the
POSITIONAL ENCODING**, converting patch indices to arcsec-from-lens-center.
That is the whole mechanism of survey invariance, so it is drawn on the
critical path and highlighted, not tucked into a caption.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import env_setup  # noqa: F401
from . import surveys as SV
from .make_paper_figures import (  # noqa: F401 -- shared, deliberately not duplicated
    FS_BASE,
    FS_SMALL,
    FS_TINY,
    INK,
    INK_2,
    INK_MUTED,
    SURVEY_COLORS,
    SURVEY_ORDER,
    TEXT_W,
    _git_state,
    _mpl,
    save_figure,
)
from .model import SurveyInvariantLensNet

SCRIPT_VERSION = "1.0.0"

# Accent for the one path that carries the paper's claim: s -> arcsec positions.
ACCENT = "#c2410c"
BOX_FC, BOX_EC = "#f5f4f1", "#c9c6bf"


def _box(ax, x, y, w, h, label, sub=None, fc=BOX_FC, ec=BOX_EC, lw=0.8,
         fs=FS_TINY, ec_text=INK):
    """One rounded block. Returns (cx, cy) so arrows can be anchored to it."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2 + (0.012 if sub else 0), label, ha="center",
            va="center", fontsize=fs, color=ec_text, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.026, sub, ha="center", va="center",
                fontsize=FS_TINY - 0.5, color=INK_2, zorder=3)
    return (x + w / 2, y + h / 2)


def _arrow(ax, p0, p1, color=INK_2, lw=0.9, style="-|>", rad=0.0, ls="-"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=7, linewidth=lw, color=color,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}",
        shrinkA=1.5, shrinkB=1.5, zorder=1))


def fig_architecture(net, fov, out_dir, manifest):
    plt = _mpl()
    fig = plt.figure(figsize=(TEXT_W, 3.35))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    dim = net.dim
    depth = len(net.blocks)
    heads = net.blocks[0].img_self.n_heads
    p = net.patch

    rows = []

    # ---- column 1: the two input streams -----------------------------------
    _box(ax, 0.015, 0.70, 0.135, 0.115, r"image $g$", sub=r"$[H\times W]$")
    _box(ax, 0.015, 0.40, 0.135, 0.115, r"PSF $h$", sub=r"$[H\times W]$")
    # the scalar, drawn as its own input because it is not an image
    _box(ax, 0.015, 0.135, 0.135, 0.115, r"scale $s$", sub=r"arcsec/px",
         fc="#fdf0e7", ec=ACCENT, ec_text=ACCENT)

    # ---- column 2: patchify + shared projection ----------------------------
    _box(ax, 0.185, 0.70, 0.145, 0.115, f"patchify {p}x{p}", sub=r"$\to$ tokens")
    _box(ax, 0.185, 0.40, 0.145, 0.115, f"patchify {p}x{p}", sub=r"$\to$ tokens")
    _arrow(ax, (0.150, 0.7575), (0.185, 0.7575))
    _arrow(ax, (0.150, 0.4575), (0.185, 0.4575))

    _box(ax, 0.365, 0.545, 0.150, 0.155,
         "shared linear", sub=f"$p^2$={p*p} $\\to$ {dim}")
    _arrow(ax, (0.330, 0.7575), (0.365, 0.66), rad=-0.15)
    _arrow(ax, (0.330, 0.4575), (0.365, 0.585), rad=0.15)
    ax.text(0.440, 0.520, "+ type embedding", ha="center", va="top",
            fontsize=FS_TINY - 0.5, color=INK_2)

    # ---- the claim: s -> arcsec positions ----------------------------------
    _box(ax, 0.185, 0.135, 0.330, 0.115,
         r"positions in ARCSEC:  $(i-c)\,s\,/\,\theta_{ref}$",
         sub=f"2D RoPE, $\\theta_{{ref}}$={net.theta_ref}\", base={net.rope_base:.0f}",
         fc="#fdf0e7", ec=ACCENT, ec_text=ACCENT, lw=1.15)
    _arrow(ax, (0.150, 0.1925), (0.185, 0.1925), color=ACCENT, lw=1.15)
    # RoPE feeds the attention stack's Q/K -- the critical path, in accent.
    # Aim at the LEFT EDGE beside the attention blocks, not at the MLP: an arrow
    # landing on the MLP box reads as "RoPE feeds the MLP", which is wrong.
    _arrow(ax, (0.520, 0.250), (0.5565, 0.700), color=ACCENT, lw=1.15, rad=-0.22)
    ax.text(0.523, 0.505, r"$Q,K$", ha="left", va="center", fontsize=FS_TINY,
            color=ACCENT)

    # ---- column 3: the attention stack -------------------------------------
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (0.560, 0.325), 0.245, 0.525,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        facecolor="none", edgecolor=INK_2, linewidth=0.9,
        linestyle=(0, (3, 2)), zorder=1))
    ax.text(0.6825, 0.874, f"TriAttention x {depth}   ({heads} heads, dim {dim})",
            ha="center", va="bottom", fontsize=FS_TINY, color=INK)

    _box(ax, 0.580, 0.760, 0.205, 0.088, "image self-attn")
    _box(ax, 0.580, 0.655, 0.205, 0.088, "PSF self-attn")
    _box(ax, 0.580, 0.550, 0.205, 0.088, r"image $\leftarrow$ PSF cross-attn",
         fc="#eef3fa", ec="#9db6d8")
    _box(ax, 0.580, 0.445, 0.205, 0.088, "MLP (x4)")
    _arrow(ax, (0.515, 0.6225), (0.580, 0.804), rad=-0.12)
    for y0, y1 in ((0.760, 0.743), (0.655, 0.638), (0.550, 0.533)):
        _arrow(ax, (0.6825, y0), (0.6825, y1))
    ax.text(0.6825, 0.315, "cross-attn: queries = image, keys/values = PSF",
            ha="center", va="top", fontsize=FS_TINY - 0.5, color=INK_2)

    # ---- column 4: the two heads -------------------------------------------
    _box(ax, 0.845, 0.700, 0.145, 0.130, r"recon head",
         sub=r"$\hat f = g + \delta$")
    _box(ax, 0.845, 0.470, 0.145, 0.130, r"regression head",
         sub=r"$\hat\theta_E$")
    _arrow(ax, (0.805, 0.640), (0.845, 0.765), rad=-0.15)
    _arrow(ax, (0.805, 0.560), (0.845, 0.535), rad=0.12)
    # log(s) ALSO enters the regression head directly (model.py concatenates it
    # onto the pooled embedding). Routed under the stack -- the earlier straight
    # run crossed the attention box and read as a connection into it.
    _arrow(ax, (0.515, 0.150), (0.963, 0.150), color=ACCENT, lw=0.9,
           style="-", ls=(0, (2.5, 1.8)))
    _arrow(ax, (0.963, 0.150), (0.963, 0.470), color=ACCENT, lw=0.9,
           ls=(0, (2.5, 1.8)))
    ax.text(0.700, 0.178, r"$\log s$", ha="center", va="bottom",
            fontsize=FS_TINY, color=ACCENT)

    ax.text(0.9175, 0.688, "residual on $g$",
            ha="center", va="top", fontsize=FS_TINY - 0.5, color=INK_2)
    ax.text(0.848, 0.458, "mean-pooled",
            ha="left", va="top", fontsize=FS_TINY - 0.5, color=INK_2)

    # ---- the claim sentence, bottom ---------------------------------------
    ax.text(0.015, 0.055,
            "Positions are arcsec from the lens centre, never pixel indices: a 0.1\" offset "
            "rotates identically at\nevery pixel scale. This is the survey-invariance "
            "mechanism -- the highlighted path is the only place $s$ enters.",
            ha="left", va="center", fontsize=FS_TINY, color=INK,
            linespacing=1.5)

    # ---- per-survey token counts at the stated FOV --------------------------
    parts = []
    for nm in SURVEY_ORDER:
        npix = int(round(fov / SV.SURVEYS[nm]["scale"]))
        ntok = (int(np.ceil(npix / p))) ** 2
        parts.append(f"{nm} {npix}$^2$px$\\to${ntok:,}")
        rows.append(dict(survey=nm, scale_arcsec_px=SV.SURVEYS[nm]["scale"],
                         fwhm_arcsec=SV.SURVEYS[nm]["fwhm"], fov_arcsec=fov,
                         npix=npix, patch=p, tokens_per_stream=ntok))
    ax.text(0.015, 0.952, "tokens/stream @ FOV %g\":  " % fov + " · ".join(parts),
            ha="left", va="center", fontsize=FS_TINY - 0.5, color=INK_2)

    # architecture constants go in the CSV too, so the figure's own numbers are
    # quotable without reading them off the drawing
    rows.append(dict(survey="__arch__", dim=dim, depth=depth, n_heads=heads,
                     patch=p, patch_dim=p * p, theta_ref=net.theta_ref,
                     rope_base=net.rope_base, fov_arcsec=fov))
    save_figure(fig, out_dir, "fig2_architecture", pd.DataFrame(rows), manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures/run3/paper")
    ap.add_argument("--mirror-dir", default=None)
    ap.add_argument("--fov", type=float, default=12.0,
                    help="only annotates token counts; must match the run being "
                         "described (Run 3 is 12\").")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--no-strict-layout", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    # Instantiate the real model so the drawn constants cannot drift from the
    # code. Cheap: no checkpoint, no forward pass, weights never used.
    net = SurveyInvariantLensNet(patch=args.patch, dim=args.dim,
                                 depth=args.depth, n_heads=args.heads)

    manifest = dict(script="survey_lens/make_fig2_architecture.py",
                    script_version=SCRIPT_VERSION,
                    generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    git=_git_state(), fov_arcsec=args.fov,
                    arch=dict(dim=args.dim, depth=args.depth, n_heads=args.heads,
                              patch=args.patch, theta_ref=net.theta_ref,
                              rope_base=net.rope_base),
                    figures={})

    global save_figure
    if args.no_strict_layout:
        from .make_paper_figures import save_figure as _sf
        save_figure = lambda *a, **k: _sf(*a, **{**k, "strict": False})  # noqa: E731

    fig_architecture(net, args.fov, args.out_dir, manifest)

    # Suffixed, like make_ablation_figures.py: three scripts now write into the
    # same paper directory and the generic names belong to make_paper_figures.
    mpath = os.path.join(args.out_dir, "figure_manifest_fig2.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("  wrote figure_manifest_fig2.json")

    if args.mirror_dir:
        os.makedirs(args.mirror_dir, exist_ok=True)
        import shutil
        for f in ("fig2_architecture.pdf", "fig2_architecture.png",
                  "fig2_architecture_data.csv", "figure_manifest_fig2.json"):
            src = os.path.join(args.out_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(args.mirror_dir, f))
        print(f"  mirrored to {args.mirror_dir}")

    print(f"done in {time.time() - t0:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
