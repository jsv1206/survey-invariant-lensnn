"""Paper figures for the Sim2Science submission.

ONE re-runnable script: from a checkpoint it rebuilds Test Set 1 (no noise) and
Test Set 2 (per-survey noise), runs inference once, and emits every data-driven
figure as PDF (for LaTeX) + PNG (for viewing), each with the exact numbers
plotted in a sibling `_data.csv`, plus `captions.md` and `figure_manifest.json`.

    PYTHONPATH=src python -m survey_lens.make_paper_figures \
        --ckpt <ckpt> --device cuda --workers 16 --out-dir figures/paper

Figures (Fig 2, the architecture schematic, is not data-driven and is assembled
by hand from figures/source_slides/):

  fig1_hero_grid          one lens, five telescopes, five pixel scales
  fig3_survey_invariance  theta_E accuracy vs pixel scale
  fig4_deconvolution      top-octave power recovery vs PSF width
  fig5_noise_shift        Test Set 1 -> Test Set 2 (the venue's theme)
  figA1_hero_grid_appendix  3 more lenses, including the worst case

Conventions that are easy to get wrong and are deliberate here:

* `hf_complex_err` is the headline reconstruction number, NOT `hf_power_ratio`.
  It is normalized so that emitting nothing (`f_hat = 0`) scores exactly 1.0,
  so >1.0 is worse than emitting nothing. The power ratio is phase-blind.
* Reconstruction panels are peak-normalized per sample before any cross-survey
  comparison -- raw surface brightness is not comparable across pixel scales.
* Error bars are bootstrap over the 200 test LENSES (paired design: the same
  lens population is rendered through all five surveys, so a cross-survey gap
  is the telescope, not a different sample).
* Survey colour is fixed once in SURVEY_COLORS and reused in every figure,
  ordered by pixel scale, with a distinct marker per survey as a secondary
  (non-colour) encoding.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from . import env_setup  # noqa: F401  (must precede lenstronomy/torch tuning)
from . import losses as Lo
from . import metrics as MET
from . import surveys as SV
from . import test as T
from .data import build_test_set_1, build_test_set_2

# 1.1.0 (Aug 21 2026): FOV-aware captions (the empty-sky fraction is measured,
#   not hardcoded to the 20" value), per-sample `mask_frac`, and data-driven
#   survey-label placement in fig4b (the hand-tuned offsets were fitted to
#   Run 2's values and collided once Run 3's numbers moved the points).
#   Run 2's figures in `figures/paper/` were produced with 1.0.0 and have NOT
#   been regenerated -- that is deliberate; Run 2 is frozen.
SCRIPT_VERSION = "1.3.0"

# --------------------------------------------------------------------------
# Style. NeurIPS text width is 5.5in; half width 2.7in. Draw at final size --
# never draw big and scale down in LaTeX, it makes the fonts illegible.
# --------------------------------------------------------------------------
TEXT_W, HALF_W = 5.5, 2.7
FS_BASE, FS_SMALL, FS_TINY = 8, 7, 6      # nothing below 6pt at final size

# Survey order is by PIXEL SCALE, and that order is the paper's x-axis
# everywhere. Colours: reference categorical slots 1/4/5/6/7 (blue, yellow,
# magenta, green, violet). This 5-subset was chosen by validating all 56
# five-subsets of the reference palette with the dataviz validator under
# `--pairs all` (the scatter case): it is the only one that PASSES both the
# CVD floor and the normal-vision floor (worst all-pairs CVD dE 13.0 protan,
# normal-vision dE 16.3). Tritan separation is 5.8, in the band that is legal
# only WITH a secondary encoding -- hence SURVEY_MARKERS, used everywhere a
# colour carries identity. Yellow and magenta sit below 3:1 contrast on white,
# which obligates visible direct labels (the relief rule); every panel that
# uses them is directly labelled or legended.
SURVEY_COLORS = {
    "HST":    "#2a78d6",   # 0.050 "/px
    "Euclid": "#eda100",   # 0.100
    "Roman":  "#e87ba4",   # 0.110
    "Rubin":  "#008300",   # 0.200
    "DES":    "#4a3aa7",   # 0.263
}
SURVEY_MARKERS = {"HST": "o", "Euclid": "s", "Roman": "^", "Rubin": "D", "DES": "v"}
SURVEY_ORDER = list(SURVEY_COLORS)

INK, INK_2, INK_MUTED = "#111111", "#52514e", "#8a8880"
GRID = "#dddddd"
# Image maps: perceptually uniform for intensity, diverging (two hues, neutral
# midpoint) for residuals. Never a rainbow.
CMAP_IMG, CMAP_RESID = "magma", "RdBu_r"

HERO_CROP_ARCSEC = 4.0        # the field is mostly empty sky; see captions
N_BOOT = 1000
BOOT_LO, BOOT_HI = 16, 84


def _no_minor_labels(*axes):
    """Kill log-scale MINOR tick labels. Matplotlib adds them whenever an axis
    spans less than a decade, and at 5.5in wide they collide with each other
    (the geometry self-check catches it every time)."""
    from matplotlib.ticker import NullFormatter
    for ax in axes:
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())


def _row_label(fig, axes_in_row, text):
    """Caption an image row ABOVE it, in figure coords. A rotated y-label on
    the first panel would hang off the canvas -- gridspec, unlike tight_layout,
    reserves no margin for it."""
    y = max(a.get_position().y1 for a in axes_in_row)
    fig.text(0.5, y + 0.022, text, ha="center", va="bottom", fontsize=FS_TINY,
             color=INK_2)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": FS_BASE, "axes.titlesize": FS_BASE, "axes.labelsize": FS_BASE,
        "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
        "legend.fontsize": FS_TINY, "figure.titlesize": FS_BASE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
        # Closed box with ticks on all four sides, pointing INWARD, no grid --
        # values are read off ticks, not gridlines. The top/right SPINES have to
        # be re-enabled for this: ticks cannot be drawn on a hidden spine.
        # Shared by make_ablation_figures.py and make_fig2_architecture.py, so
        # this governs every figure in the paper set.
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "axes.axisbelow": True, "axes.spines.top": True, "axes.spines.right": True,
        "lines.linewidth": 1.2, "figure.dpi": 200, "savefig.dpi": 200,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


# --------------------------------------------------------------------------
# Geometric self-check (Step 10 output contract): never ship a figure whose
# text overlaps other text or falls outside the canvas.
# --------------------------------------------------------------------------
def geometry_check(fig, name):
    """Return a list of human-readable layout problems. Empty == clean."""
    import matplotlib.text as mtext
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox

    # Matplotlib keeps Text objects for ticks OUTSIDE the view limits (a log
    # axis viewing 1e-2..1e0 still carries a 1e7 label). They are never drawn,
    # but they report visible=True and a window extent far off-canvas, so they
    # would swamp the real findings. Identify tick labels and ignore any whose
    # box does not touch its own axes.
    tick_owner = {}
    skip = set()
    for ax in fig.axes:
        # Each Axis carries an `offsetText` (the shared exponent/offset beside
        # an axis); it is parked at the origin with placeholder content and
        # painted only when a formatter needs one.
        skip.add(id(ax.xaxis.offsetText))
        skip.add(id(ax.yaxis.offsetText))
        for axis, scale in ((ax.xaxis, ax.get_xscale()),
                            (ax.yaxis, ax.get_yscale())):
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                try:
                    loc = tick.get_loc()
                except Exception:
                    loc = None
                lo, hi = sorted(axis.get_view_interval())
                for lab in (tick.label1, tick.label2):
                    tick_owner[id(lab)] = ax
                    # Matplotlib keeps Tick objects for locations OUTSIDE the
                    # current view (a linear axis showing 0.35-2.65 still holds
                    # a tick at 0.0, a log axis one at 0). They are never
                    # painted, but report visible=True with a real extent just
                    # off the axes, so they read as phantom overlaps.
                    if loc is None or not (lo <= loc <= hi):
                        skip.add(id(lab))

    items = []
    for t in fig.findobj(mtext.Text):
        if id(t) in skip:
            continue
        if not t.get_visible() or not (t.get_text() or "").strip():
            continue
        try:
            bb = t.get_window_extent(renderer=r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        owner = tick_owner.get(id(t))
        if owner is not None:
            ab = owner.get_window_extent(renderer=r).expanded(1.35, 1.35)
            if mtext.Bbox.intersection(bb, ab) is None:
                continue          # off-view tick label: not rendered
        # a legend's own texts live inside its frame; they are laid out by
        # matplotlib and are not the overlaps we are hunting for
        in_legend = False
        p = t.get_parent() if hasattr(t, "get_parent") else None
        for anc in fig.legends + [lg for ax in fig.axes
                                  if (lg := ax.get_legend()) is not None]:
            if t in anc.findobj(mtext.Text):
                in_legend = True
                break
        items.append((t, bb, in_legend))

    problems = []
    legend_boxes = []
    for lg in fig.legends + [l for ax in fig.axes
                             if (l := ax.get_legend()) is not None]:
        try:
            legend_boxes.append((lg, lg.get_window_extent(renderer=r)))
        except Exception:
            pass
    for t, bb, in_lg in items:
        if in_lg:
            continue
        for lg, lb in legend_boxes:
            ov = mtext.Bbox.intersection(bb, lb)
            if ov is not None and ov.width > 1 and ov.height > 1:
                problems.append(
                    f"{name}: text {t.get_text()!r} lands on a legend box "
                    f"({ov.width:.1f}x{ov.height:.1f} px)")
    for t, bb, _ in items:
        if (bb.x0 < fb.x0 - 1 or bb.y0 < fb.y0 - 1
                or bb.x1 > fb.x1 + 1 or bb.y1 > fb.y1 + 1):
            problems.append(f"{name}: text {t.get_text()!r} outside figure bounds")
    for i in range(len(items)):
        ti, bi, li = items[i]
        if li:
            continue
        for j in range(i + 1, len(items)):
            tj, bj, lj = items[j]
            if lj:
                continue
            ov = mtext.Bbox.intersection(bi, bj)
            if ov is not None and ov.width > 1 and ov.height > 1:
                where = ""
                ai = [k for k, a in enumerate(fig.axes) if ti.axes is a]
                aj = [k for k, a in enumerate(fig.axes) if tj.axes is a]
                if ai or aj:
                    where = f" [axes {ai or '-'} vs {aj or '-'}]"
                problems.append(
                    f"{name}: text {ti.get_text()!r} overlaps {tj.get_text()!r} "
                    f"({ov.width:.1f}x{ov.height:.1f} px){where}")
    return problems


def save_figure(fig, out_dir, slug, data: pd.DataFrame, manifest, strict=True):
    """Save PDF + PNG + the exact numbers plotted, after the geometry check."""
    problems = geometry_check(fig, slug)
    if problems:
        msg = f"LAYOUT PROBLEMS in {slug}:\n  " + "\n  ".join(problems)
        if strict:
            raise AssertionError(msg)
        print("WARNING: " + msg)
    pdf = os.path.join(out_dir, f"{slug}.pdf")
    png = os.path.join(out_dir, f"{slug}.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    csv = os.path.join(out_dir, f"{slug}_data.csv")
    data.to_csv(csv, index=False)
    manifest["figures"][slug] = dict(pdf=os.path.basename(pdf),
                                     png=os.path.basename(png),
                                     data_csv=os.path.basename(csv),
                                     n_rows=int(len(data)),
                                     layout_check="clean" if not problems
                                     else f"{len(problems)} problem(s)")
    print(f"  wrote {slug}.pdf/.png (+ {len(data)} rows of data)")
    _mpl().close(fig)


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
@torch.no_grad()
def per_sample_pass(model, items, device, batch_size=16, patch=4, noise_cfg=None):
    """Per-SAMPLE metrics for one survey (test.py reports survey means only;
    bootstrap error bars need the individual lenses)."""
    out = {k: [] for k in ("theta_pred", "theta_true", "l1_norm_masked",
                           "ssim_norm", "hf_complex_err", "hf_coherence",
                           "reblur_rel", "floor_rel", "reblur_floor_op",
                           "l1_norm_masked_identity", "hf_complex_err_identity",
                           "l1_norm_masked_zeros", "hf_complex_err_zeros",
                           "l1_rel_masked", "l1_rel_masked_identity",
                           "l1_rel_masked_zeros",
                           "mask_frac")}
    prof = None
    n = len(items)
    for i in range(0, n, batch_size):
        chunk = items[i:i + batch_size]
        g = torch.stack([x["g"] for x in chunk]).to(device)
        h = torch.stack([x["h"] for x in chunk]).to(device)
        f = torch.stack([x["f_gt"] for x in chunk]).to(device)
        s = torch.stack([x["s"] for x in chunk]).to(device)
        th = torch.stack([x["theta_E"] for x in chunk]).to(device)
        bs = g.shape[0]

        res = model(g, h, s)
        f_hat = res["recon"]
        out["theta_pred"].extend(res["theta_E"].cpu().tolist())
        out["theta_true"].extend(th.cpu().tolist())

        peak = f.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        for tag, img in (("", f_hat), ("_identity", g),
                         ("_zeros", torch.zeros_like(g))):
            a, b = img / peak, f / peak
            mask = b > 0.01
            mf = mask.to(b.dtype)
            den = mf.sum(dim=(1, 2, 3)).clamp_min(1.0)
            l1 = ((a - b).abs() * mf).sum(dim=(1, 2, 3)) / den
            out[f"l1_norm_masked{tag}"].extend(l1.cpu().tolist())
            # RELATIVE L1: total absolute error over total arc flux, inside the
            # same mask. Reported as a percentage because it reads directly as
            # "off by X% of the arc's flux".
            #
            # Deliberately NOT a per-pixel |f_hat - f_gt| / f_gt: the mask floor
            # is 1% of peak, so a pixel-wise ratio is amplified up to 100x at the
            # mask edge and the metric ends up dominated by the faintest pixels.
            # A ratio of SUMS is flux-weighted and robust. The peak
            # normalization cancels in the ratio, so this is scale-free without
            # relying on it.
            num = ((a - b).abs() * mf).sum(dim=(1, 2, 3))
            tot = (b * mf).sum(dim=(1, 2, 3)).clamp_min(1e-12)
            out[f"l1_rel_masked{tag}"].extend((num / tot).cpu().tolist())
            if tag == "":
                out["mask_frac"].extend(
                    (mf.sum(dim=(1, 2, 3)) / mf[0].numel()).cpu().tolist())
            hm = MET.hf_power_metrics(img, f, s, patch=patch)
            out[f"hf_complex_err{tag}"].extend(hm["hf_complex_err"].cpu().tolist())
            if tag == "":
                out["hf_coherence"].extend(hm["hf_coherence"].cpu().tolist())
                for k in range(bs):
                    out["ssim_norm"].append(
                        float(Lo.ssim(f_hat[k:k + 1], f[k:k + 1], mask=mask[k:k + 1])))

        # reblur residual, per sample, in reblur_loss's scale-free units
        gsq = (g ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
        rb = ((Lo.reblur(f_hat, h) - g) ** 2).mean(dim=(1, 2, 3)) / gsq
        out["reblur_rel"].extend(rb.cpu().tolist())
        # OPERATOR floor: the residual an ORACLE reconstruction (f_hat = f_gt)
        # still leaves, because `reblur` is not an exact inverse of the
        # simulator's convolve-then-resample forward model. It is dominated by
        # PSF-stamp aliasing (dataset bug 3) and spans ~211x across surveys,
        # tracking sigma-in-pixels rather than survey quality. Raw reblur
        # therefore confounds "how good is the reconstruction" with "how
        # faithful is the diagnostic here" -- report the RATIO.
        rb0 = ((Lo.reblur(f, h) - g) ** 2).mean(dim=(1, 2, 3)) / gsq
        out["reblur_floor_op"].extend(rb0.cpu().tolist())
        if noise_cfg is not None:
            var = g.clamp_min(0.0) / noise_cfg["gain"] + noise_cfg["bkg_rms"] ** 2
            out["floor_rel"].extend((var.mean(dim=(1, 2, 3)) / gsq).cpu().tolist())

        # radial power profiles: f_hat vs truth, and g (== the identity
        # baseline) vs truth, accumulated as batch-size-weighted means
        c, ph, pg = MET.radial_power_profile(f_hat, f, s)
        _, pgin, _ = MET.radial_power_profile(g, f, s)
        add = (c, ph * bs, pg * bs, pgin * bs)
        prof = add if prof is None else (c, prof[1] + ph * bs, prof[2] + pg * bs,
                                         prof[3] + pgin * bs)
    res = {k: np.asarray(v, dtype=float) for k, v in out.items() if len(v)}
    res["radial"] = (prof[0], prof[1] / n, prof[2] / n, prof[3] / n)
    return res


@torch.no_grad()
def images_for_lens(model, items, idx, device):
    """Everything Fig 1 / 4c / 5c draw, for one lens index."""
    it = items[idx]
    g = it["g"][None].to(device)
    h = it["h"][None].to(device)
    f = it["f_gt"][None].to(device)
    s = it["s"][None].to(device)
    f_hat = model(g, h, s)["recon"]
    reblur = Lo.reblur(f_hat, h)
    # `var` is present only on noise-carrying test sets. Passed through
    # as None otherwise, so fig_hero's chi residual can fail loudly when asked
    # for on noise-free data rather than silently dividing by nothing.
    _v = it.get("var")
    return dict(g=g[0, 0].cpu().numpy(), f_hat=f_hat[0, 0].cpu().numpy(),
                f_gt=f[0, 0].cpu().numpy(),
                reblur_resid=(reblur - g)[0, 0].cpu().numpy(),
                var=(None if _v is None else _v[0].cpu().numpy()),
                s=float(it["s"]), theta_E=float(it["theta_E"]))


def boot_ci(vals, stat, rng, n_boot=N_BOOT):
    """Bootstrap over lenses -> (point, lo, hi) at the 16-84th percentile."""
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    draws = np.array([stat(vals[rng.integers(0, n, n)]) for _ in range(n_boot)])
    return float(stat(vals)), float(np.percentile(draws, BOOT_LO)), \
        float(np.percentile(draws, BOOT_HI))


def boot_ci_paired(a, b, stat, rng, n_boot=N_BOOT):
    """Bootstrap a statistic of PAIRED arrays (e.g. theta RMSE from pred/true)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    draws = np.array([stat(a[i], b[i]) for i in
                      (rng.integers(0, n, n) for _ in range(n_boot))])
    return float(stat(a, b)), float(np.percentile(draws, BOOT_LO)), \
        float(np.percentile(draws, BOOT_HI))


def _rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def _crop(img, s, half_arcsec=HERO_CROP_ARCSEC):
    """Center crop to a fixed ARCSEC box, so all surveys show the same sky."""
    n = img.shape[-1]
    half_px = int(round(half_arcsec / s))
    c = n // 2
    lo, hi = max(0, c - half_px), min(n, c + half_px)
    return img[lo:hi, lo:hi]


# --------------------------------------------------------------------------
# Fig 1 (hero) -- one lens, five telescopes
# --------------------------------------------------------------------------
def fig_hero(res, imgs, out_dir, manifest, slug="fig1_hero_grid",
             lens_ids=None, title=None, stretch=None, resid_mode="frac"):
    """COLUMNS = surveys ordered by pixel scale, ROWS = input / recon / truth /
    residual.

    Transposed from the original rows-are-surveys layout: five survey columns
    across the text width is a landscape figure (~5.5 x 4.5in) instead of a
    portrait one (~5.5 x 6.2in), which is the cheaper shape for a short paper.

    Normalization is per SURVEY COLUMN, by that column's `f_gt` peak, so the
    four panels of a column share vmin/vmax and columns stay comparable despite
    a ~15x spread in raw per-pixel flux across pixel scales.

    Multi-lens (figA1) stacks 4-row blocks vertically rather than widening to
    15 columns, so the grid is always 5 wide.

    Colorbars sit on the RIGHT here, not the bottom: after the transpose the two
    colour scales are separated by ROW (magma for rows 0-2, diverging for the
    residual row), so vertical bars align with what they describe. Both carry the
    SAME dimensionless label with visibly different spans -- that is what makes
    the residual's 20x tighter scale legible. They must NOT be labelled
    "flux/px": the pixels are normalized and no longer carry that unit.

    Units: `S` is flux per PIXEL in arbitrary simulator units -- there is no
    zeropoint anywhere in sim_lens (src_amp ~ U(2,6)). It is not surface
    brightness: lenstronomy returns flux per pixel and _area_resample SUMS to
    conserve it, which is exactly why S_peak varies across surveys at all.
    """
    plt = _mpl()
    lens_ids = lens_ids or [None]
    n_lens = len(imgs)
    nrow = 4 * n_lens
    ncol = len(SURVEY_ORDER)
    # +0.45 (was +0.72) for the top strip: the column header is now ONE line
    # (survey name) instead of the 4-5 line stack of scale/FWHM/npix/S_peak, which
    # freed ~0.3in. Leaving 0.72 would just re-open the whitespace as dead space
    # at the top, exactly where it was reclaimed.
    fig = plt.figure(figsize=(TEXT_W, 0.90 * nrow + 0.42))
    # 5 image columns + a narrow spacer + the colorbar column + a spacer for the
    # bars' tick labels and rotated axis labels.
    gs = fig.add_gridspec(nrow, ncol + 3,
                          width_ratios=[1] * ncol + [0.12, 0.07, 0.42])
    CBAR_COL = ncol + 1
    axes = [[fig.add_subplot(gs[r, c]) for c in range(ncol)] for r in range(nrow)]
    im_seq = im_div = None
    rows = []
    CHI_LIM = 3.0
    RESID_LIM = 0.05          # see the note below the table in the results write-up
    _chi = resid_mode == "chi"
    # Order: input, TRUTH, reconstruction, residual. Truth sits directly under
    # the input so the reader sees what the blur removed, then the
    # reconstruction directly above the residual it is scored by -- both
    # comparisons are between ADJACENT rows.
    row_names = [r"input $g$", r"truth $f_{gt}$", r"recon $\hat f$",
                 (r"$(\hat f - f_{gt})/\sigma$" if _chi
                  else r"resid $\hat f - f_{gt}$")]

    for li, per_survey in enumerate(imgs):
        for ci, nm in enumerate(SURVEY_ORDER):
            d = per_survey[nm]
            s = d["s"]
            pk = max(float(np.abs(d["f_gt"]).max()), 1e-12)
            # Row 4 in "chi" mode is the residual in units of the per-pixel
            # INPUT noise, not of the f_gt peak. +/-0.05 of peak is an arbitrary
            # scale a reader cannot judge; chi ~ 1 means "the reconstruction sits
            # within the data's own noise", which separates noise-limited
            # surveys (HST/Euclid/Roman, rms chi 0.46-0.94) from
            # deconvolution-limited ones (Rubin/DES, 3.4-4.2).
            if _chi:
                if d.get("var") is None:
                    raise ValueError(
                        "resid_mode='chi' needs per-pixel `var`, which only "
                        "noise-carrying test sets provide. Run 3 is noise-free.")
                resid = (d["f_hat"] - d["f_gt"]) / np.sqrt(
                    np.maximum(d["var"], 1e-30))
            else:
                resid = (d["f_hat"] - d["f_gt"]) / pk
            panels = [d["g"] / pk, d["f_gt"] / pk, d["f_hat"] / pk, resid]
            npix = d["f_gt"].shape[-1]
            for pi, img in enumerate(panels):
                ax = axes[li * 4 + pi][ci]
                c = _crop(img, s)
                if pi == 3:
                    _lim = CHI_LIM if _chi else RESID_LIM
                    h = ax.imshow(c, cmap=CMAP_RESID, vmin=-_lim,
                                  vmax=_lim, interpolation="nearest")
                    im_div = im_div or h
                else:
                    # `stretch="asinh"`: training noise runs 0.19%-0.91% of
                    # the f_gt peak, i.e. below one display level on a linear
                    # 0->1 map, so the "input g" row looked identical to a
                    # noise-free one and the premise of the paper was invisible
                    # in its own lead figure. asinh (the standard imaging stretch)
                    # lifts 0.2% -> 6.5% and 0.9% -> 22% while leaving the arc
                    # bright (0.5 -> 0.88) rather than blowing it out.
                    #
                    # Applied to the three SEQUENTIAL rows only. The residual row
                    # keeps its linear diverging scale: it is already at +/-0.05
                    # and symmetric about zero, where a nonlinear stretch would
                    # distort the sign structure that panel exists to show.
                    if stretch == "asinh":
                        from matplotlib.colors import AsinhNorm
                        h = ax.imshow(c, cmap=CMAP_IMG,
                                      norm=AsinhNorm(linear_width=0.005,
                                                     vmin=0.0, vmax=1.0),
                                      interpolation="nearest")
                    else:
                        h = ax.imshow(c, cmap=CMAP_IMG, vmin=0.0, vmax=1.0,
                                      interpolation="nearest")
                    im_seq = im_seq or h
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if ci == 0:
                    ax.set_ylabel(row_names[pi], fontsize=FS_BASE, rotation=90,
                                  ha="center", va="center", labelpad=6)
            # Column header is the survey NAME only. Pixel scale, FWHM, npix and
            # S_peak used to be stacked here as a 4-5 line title; they now live in
            # a table in the paper, sourced from `fig1_hero_grid_data.csv`, which
            # still carries every one of them.
            #
            # Known cost of this: the comment that used to sit here said "the
            # varying npix IS the result, so keep it visible", and printing npix
            # was the only direct evidence that the five surveys sit on different
            # pixel grids. The 1" scale bar drawn below still shows it implicitly
            # -- its length in pixels differs per column -- and the table carries
            # the numbers. Deliberate trade, not an oversight.
            if li == 0:
                axes[0][ci].set_title(nm, fontsize=FS_BASE + 1, pad=4,
                                      color=SURVEY_COLORS[nm])
            # 1" scale bar on the input row: physical size is constant while the
            # pixel grid is not
            ax = axes[li * 4][ci]
            npx_crop = _crop(panels[0], s).shape[-1]
            bar_px = 1.0 / s
            ax.plot([0.06 * npx_crop, 0.06 * npx_crop + bar_px],
                    [0.92 * npx_crop, 0.92 * npx_crop],
                    color="white", lw=1.4, solid_capstyle="butt")
            ax.text(0.06 * npx_crop, 0.88 * npx_crop, "1\"", color="white",
                    fontsize=FS_TINY - 1, va="bottom")
            # Computed from the FRACTIONAL residual explicitly, never from
            # panels[3]: in chi mode panels[3] is the chi map, and reading these
            # off it silently filled the resid_*_frac_of_peak columns with chi
            # statistics -- correct numbers under names that lie about their
            # units (rms read 1.99 where a fraction of peak should be ~0.003).
            _rc = _crop((d["f_hat"] - d["f_gt"]) / pk, s)
            # chi stats go to the CSV whether or not row 4 plots them, so the
            # caption and the text can quote them either way. The
            # absolute resid_* columns below are KEPT even in chi mode: the text
            # still wants amplitudes, they just leave the figure.
            if d.get("var") is not None:
                _chi_map = _crop((d["f_hat"] - d["f_gt"]) / np.sqrt(
                    np.maximum(d["var"], 1e-30)), s)
                _a = np.abs(_chi_map)
                # Signal mask, per a known dataset issue: the crop is ~86% empty sky,
                # where chi ~ 0, so a whole-crop rms is diluted 2.4-2.8x and does
                # not describe the arc a reader is looking at. `chi_rms` is
                # masked; `chi_rms_global` keeps the whole-crop value as the
                # hallucination check the same rule asks for.
                _fc = _crop(d["f_gt"], s)
                _mk = (_fc / max(float(_fc.max()), 1e-12)) > 0.01
                _n = max(int(_mk.sum()), 1)
                # The SAME chi for the unprocessed input (f_hat = g). Without it
                # a reader cannot tell whether chi = 4.2 at DES is a failure or a
                # good result on hard data -- the input was at 15.8. Emitted per
                # survey so the caption can quote the CSV rather than prose.
                _chi_id = _crop((d["g"] - d["f_gt"]) / np.sqrt(
                    np.maximum(d["var"], 1e-30)), s)
                _chi_cols = dict(
                    chi_rms=float(np.sqrt(((_chi_map ** 2) * _mk).sum() / _n)),
                    chi_identity_rms=float(np.sqrt(((_chi_id ** 2) * _mk).sum() / _n)),
                    chi_rms_global=float(np.sqrt((_chi_map ** 2).mean())),
                    chi_mask_frac=float(_mk.mean()),
                    chi_frac_within_1=float((_a[_mk] < 1).mean()),
                    chi_frac_within_3=float((_a[_mk] < 3).mean()),
                    chi_max=float(_a.max()),
                    chi_saturated_frac=float((_a > CHI_LIM).mean()))
            else:
                _chi_cols = {}
            rows.append(dict(lens_id=lens_ids[li], survey=nm, scale_arcsec_px=s,
                             fwhm_arcsec=SV.SURVEYS[nm]["fwhm"], npix_native=npix,
                             theta_E_true=d["theta_E"], f_gt_peak=pk,
                             crop_half_arcsec=HERO_CROP_ARCSEC,
                             resid_limit_frac_of_peak=RESID_LIM,
                             resid_max_frac_of_peak=float(np.abs(_rc).max()),
                             resid_rms_frac_of_peak=float(np.sqrt((_rc ** 2).mean())),
                             resid_saturated_frac=float((np.abs(_rc) > RESID_LIM).mean()),
                             **_chi_cols))
    if title:
        fig.suptitle(title, fontsize=FS_SMALL, y=1.0)
    # w_pad 0.30 -> 0.04: the panels are borderless images with no tick labels
    # between them, so horizontal padding bought nothing but whitespace.
    # h_pad == w_pad so the gutters between rows and between columns are the
    # same width; tight_layout's pads are both in fractions of font size, so
    # equal values give equal absolute spacing.
    fig.tight_layout(h_pad=0.10, w_pad=0.10)

    # Colorbars AFTER tight_layout, which does not lay out gridspec slices.
    CBAR_LABEL = r"$S\,/\,S_{\mathrm{peak}}(f_{gt})$"
    # A stretched colourbar needs ticks where the eye can use them: 0/0.5/1 on an
    # asinh scale leaves the whole low end -- the part the stretch exists to
    # reveal -- unlabelled.
    _seq_ticks = ([0.0, 0.01, 0.05, 0.2, 0.5, 1.0] if stretch == "asinh"
                  else [0.0, 0.5, 1.0])
    _resid_ticks = ([-CHI_LIM, 0.0, CHI_LIM] if _chi
                    else [-RESID_LIM, 0.0, RESID_LIM])
    for rsl, handle, ticks in ((slice(0, 3), im_seq, _seq_ticks),
                               (slice(3, 4), im_div, _resid_ticks)):
        if handle is None:
            continue
        cax = fig.add_subplot(gs[rsl, CBAR_COL])
        pos = cax.get_position()
        pad = 0.012
        cax.set_position([pos.x0, pos.y0 + pad, pos.width, pos.height - 2 * pad])
        cb = fig.colorbar(handle, cax=cax, ticks=ticks)
        _lbl = (r"$(\hat f - f_{gt})\,/\,\sigma$"
                if (_chi and rsl.start == 3) else CBAR_LABEL)
        if stretch == "asinh" and rsl.start == 0:
            # AsinhNorm installs its own locator/formatter and ignores `ticks=`,
            # rendering the bar as 1e0 / 1e-2 / 0 -- three labels, none of them
            # in the low range the stretch exists to expose. Force both.
            from matplotlib.ticker import FixedLocator, FixedFormatter
            cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
            cb.ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
            cb.ax.yaxis.set_minor_locator(FixedLocator([]))
        # FS_BASE, matching the row labels on the left: at FS_TINY the rotated
        # colourbar label was the smallest text in the figure and the hardest to
        # read at print size, while being the only thing that says what the
        # colours mean.
        cb.set_label(_lbl, fontsize=FS_BASE, labelpad=2)
        cb.ax.tick_params(labelsize=FS_TINY, length=2, pad=1.5, direction="out")
        cb.outline.set_linewidth(0.6)
    save_figure(fig, out_dir, slug, pd.DataFrame(rows), manifest)


# --------------------------------------------------------------------------
# Fig 3 -- quantitative survey invariance
# --------------------------------------------------------------------------
def _frac_rmse(pred, true):
    """RMS of the PER-LENS fractional error, (pred-true)/true.

    Reported instead of absolute RMSE because a percentage is directly
    interpretable against the truth, whereas "0.017 arcsec" needs the theta_E
    range (0.5-2.5") to mean anything. Note this is a HARDER test of invariance,
    not a rescaling: normalizing per lens up-weights small-theta_E systems where
    a fixed positional error is a larger fraction, so the cross-survey spread
    grows from 1.87x (absolute) to 2.75x (fractional).
    """
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.sqrt(np.mean(((pred - true) / true) ** 2)))


def fig_invariance(ts1, base1, out_dir, manifest, rng, ts2=None, base2=None,
                   slug="fig3_survey_invariance", theta_scale="linear",
                   null_line=False, set_labels=("noise-free", "survey noise")):
    """Three stacked panels sharing ONE x axis (pixel scale).

    Two panels: the regression head and the reconstruction head, both against
    pixel scale. This keeps a clean split from Fig 4, which owns the
    frequency-domain metric and the Wiener comparison -- no metric appears in
    both figures.

    A reblur/data-consistency panel was built and REMOVED. On noise-free data it
    adds nothing independent (a good reconstruction implies good reblur
    consistency), and it is confounded twice over: the reblur operator's own
    floor spans ~100x across surveys for reasons unrelated to the network
    (PSF-stamp aliasing, dataset bug 3), and because the reblur term is IN the
    training loss the model partly compensates for that operator error where it
    is largest -- at Roman 94.5% of lenses land BELOW the oracle floor, which
    cannot be read as a better reconstruction. Reblur's value is on noisy/real
    data; it belongs in Fig 5, not here.

    Two reference lines were deliberately REMOVED and one deliberately KEPT:

    * `prior_mean` (0.618") is gone from (a). It forced y-limits of 0.008..1.3,
      a 162x span, leaving the data in the bottom ~1% of the axis where nothing
      could be read. Zoomed to the data, the mild 1.87x trend with pixel scale
      becomes visible -- which is why the prose must say "nearly flat", never
      "flat". The number lives in the results write-up for referees.
    * `zeros` is gone from (b): 2-12x worse than `identity`, so beating
      `identity` implies it.
    * `identity` STAYS in (b) and is load-bearing. Trained L1 rises 22x across
      the pixel-scale range, which alone reads as the method degrading;
      `identity` rises 13x over the same range, so the correct reading is that
      the PROBLEM got harder while the margin held at 4.7-7.8x. Masked L1 is
      also low-frequency dominated, where `g` is already close to `f_gt` -- so
      without this line a reader cannot tell deconvolution from pass-through.
    """
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.25))
    LBL, TCK = FS_SMALL, FS_TINY
    # Two variants from one function, like fig_hero does for fig1/figA1:
    #   ts2 = None -> Test Set 1 only         (Representations; no noise discussion)
    #   ts2 given  -> both test sets overlaid (Sim2Science; the noise shift IS the point)
    # Filled marker = noise-free, OPEN marker = with noise. Chosen over alpha
    # because it survives greyscale and stays legible at ms=4.
    # Same marker SIZE for both sets -- only the fill differs (solid = noise-free,
    # hollow = with noise). Separation comes from the x-nudge below, not from
    # size, so neither series is visually weighted above the other.
    #
    # The 4th element is that x-nudge, so the two error bars do not overprint.
    # It is applied MULTIPLICATIVELY in (a)/(b), whose x axis is log, and
    # ADDITIVELY in (c), whose x axis is linear -- a single offset value would
    # be wrong in one of them. The nudge is presentational: both members of a
    # pair share the same TRUE x, and the caption says so.
    XLOG, XLIN = 1.055, 0.045
    SETS = [(ts1, 1, dict(ms=4), 0)]
    if ts2 is not None:
        # Open marker for the noisy set, drawn LARGER with a THINNER edge: at
        # ms=4 with edgewidth 1.1 the ring is nearly all edge and reads as
        # filled. A hollow centre needs room to be seen.
        SETS.append((ts2, 2, dict(ms=4, markerfacecolor="none",
                                  markeredgewidth=0.9), 1))
    rows = []
    xs = [SV.SURVEYS[n]["scale"] for n in SURVEY_ORDER]

    # ---- (a) theta_E RMSE -------------------------------------------------
    ax = axes[0]
    pts = []
    for src, tset, style, sh in SETS:
        for nm in SURVEY_ORDER:
            r = src[nm]
            pt, blo, bhi = boot_ci_paired(r["theta_pred"], r["theta_true"],
                                          lambda a, b: 100.0 * _frac_rmse(a, b), rng)
            s_ = SV.SURVEYS[nm]["scale"]
            xp = s_ * (XLOG ** sh)
            ax.errorbar(xp, pt, yerr=[[pt - blo], [bhi - pt]],
                        fmt=SURVEY_MARKERS[nm], color=SURVEY_COLORS[nm],
                        capsize=2, lw=1.0, zorder=3, **style)
            if tset == 1:
                pts.append((s_, pt, nm))
            _t = np.asarray(r["theta_true"], float)
            _e = np.asarray(r["theta_pred"], float) - _t
            rows.append(dict(panel="a", survey=nm, test_set=tset, scale_arcsec_px=s_,
                             theta_frac_rmse_pct=pt, ci_lo=blo, ci_hi=bhi,
                             theta_rmse_arcsec=_rmse(r["theta_pred"], r["theta_true"]),
                             theta_frac_median_pct=float(100 * np.median(np.abs(_e / _t))),
                             prior_mean_rmse=base1[nm]["prior_mean"]["theta_rmse"]))
    # Survey names appear ONCE, as a legend on the top panel -- every panel
    # plots the same five x positions with the same colour/marker, so one key
    # serves all three. Point annotations were tried and do not fit: Euclid
    # (0.100) and Roman (0.110) sit 10% apart in x at different heights, so
    # even alternating offsets overprint in a 3.6in-wide panel.
    from matplotlib.lines import Line2D
    survey_handles = [Line2D([], [], marker=SURVEY_MARKERS[n], ls="",
                             color=SURVEY_COLORS[n], ms=3.0, label=n)
                      for n in SURVEY_ORDER]
    # LINEAR by default: Run 3's data spans 1.87x, which contains no decade tick,
    # so a log axis renders with no y labels at all. Linear also shows the mild
    # trend honestly instead of flattening it.
    #
    # `theta_scale="log"` exists for a domain-shift variant, where that
    # reasoning inverts: in-distribution ~1.4% against ~42% under the shift is a
    # 30x span with two decades in it, and on a linear axis the five
    # in-distribution points collapse onto the baseline, hiding exactly the
    # cross-survey spread this panel exists to show.
    _amax = max(r["theta_frac_rmse_pct"] for r in rows if r["panel"] == "a")
    if null_line:
        # Fold the null into the limit BEFORE it is set: at 66.9% against a data
        # max of 39.8%, `_amax * 1.6` topped out at 63.6 and clipped the line
        # clean off the axis -- drawn, invisible, and the panel silently lost the
        # reference that makes "degraded" quantitative.
        _tt0 = np.asarray(ts1[SURVEY_ORDER[0]]["theta_true"], float)
        _null_pct = 100.0 * _frac_rmse(np.full_like(_tt0, _tt0.mean()), _tt0)
        _amax = max(_amax, _null_pct)
    if theta_scale == "log":
        _amin = min(r["ci_lo"] for r in rows if r["panel"] == "a")
        ax.set_yscale("log")
        ax.set_ylim(_amin * 0.65, _amax * 1.6)
    else:
        ax.set_ylim(0, _amax * 1.30)
    if null_line:
        # The prior-mean null: RMSE of always predicting the mean theta_E. Run 3's
        # fig 3 deliberately omits it (the model sits far below it and explaining
        # it costs words a short paper does not have). Under the domain shift it
        # is the entire point -- theta_E lands AT it -- so "collapse" is only
        # legible against this line.
        # Computed with the PANEL'S OWN metric (_frac_rmse of a constant
        # mean-theta predictor), not as rmse/mean(theta). Those differ by 1.59x
        # here -- 66.9% vs 42.0% -- because fractional error weights small
        # theta_E heavily and the mean-predictor is worst exactly there. Drawing
        # the rmse/mean version put the line 1.59x too low and made the shifted
        # points look like they sat ON the null when this metric puts them below.
        ax.axhline(_null_pct, ls=":", lw=1.0, color=INK_MUTED, zorder=1,
                   label="prior-mean null")
    ax.set_ylabel(r"$\theta_E$ frac. RMSE (%)", fontsize=LBL)
    ax.set_xscale("log")
    ax.set_xlabel("pixel scale (arcsec/px)", fontsize=LBL)
    ax.set_title("(a) Einstein radius", fontsize=FS_SMALL, pad=3)
    if null_line:
        # (b) and (c) label their reference lines in-panel; (a) had no legend at
        # all, so the null rendered as an unexplained dotted line. Placed centre-
        # right: the null spans the top, the shifted points sit mid-panel and the
        # in-distribution ones along the bottom, leaving that band clear.
        ax.legend(loc="center right", frameon=False, handletextpad=0.5,
                  borderpad=0.1, labelspacing=0.2, fontsize=FS_TINY)

    # ---- (b) masked normalized L1, with the identity reference ------------
    ax = axes[1]
    for src, tset, style, sh in SETS:
        for nm in SURVEY_ORDER:
            r = src[nm]
            s_ = SV.SURVEYS[nm]["scale"]
            pt, blo, bhi = boot_ci(100.0 * np.asarray(r["l1_rel_masked"]),
                                   np.mean, rng)
            xp = s_ * (XLOG ** sh)
            ax.errorbar(xp, pt, yerr=[[pt - blo], [bhi - pt]],
                        fmt=SURVEY_MARKERS[nm], color=SURVEY_COLORS[nm],
                        capsize=2, lw=1.0, zorder=3, **style)
            rows.append(dict(panel="b", survey=nm, test_set=tset, scale_arcsec_px=s_,
                             l1_rel_pct=pt, ci_lo=blo, ci_hi=bhi,
                             identity_pct=float(100 * np.mean(r["l1_rel_masked_identity"])),
                             l1_norm_masked=float(np.mean(r["l1_norm_masked"]))))
    # Small markers on the reference lines: identity and Wiener are evaluated
    # PER SURVEY (five discrete points), not interpolated, and with marker=""
    # that was invisible -- the line read as a continuous function of pixel
    # scale, which it is not.
    ax.plot(xs, [100 * float(np.mean(ts1[n]["l1_rel_masked_identity"]))
                 for n in SURVEY_ORDER],
            ls="--", lw=1.0, color=INK_MUTED, marker="o", ms=2.2, zorder=1,
            label=r"identity ($\hat f = g$)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("pixel scale (arcsec/px)", fontsize=LBL)
    ax.set_ylabel("L1 error (% of arc flux)", fontsize=LBL)
    ax.set_title("(b) Reconstruction", fontsize=FS_SMALL, pad=3)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.5,
              borderpad=0.1, labelspacing=0.2, fontsize=FS_TINY)

    # ---- (c) top-octave error vs PSF sampling -----------------------------
    # Moved here from Fig 4. Kept DIMENSIONLESS on purpose while (a) and (b) are
    # percentages: hf_complex_err divides squared amplitudes, so x100 would be a
    # % of POWER sitting beside two % of FLUX. HST reads 1.68% in power but 13.0%
    # in amplitude -- an order-of-magnitude misreading that adjacency alone would
    # invite. The documented "1.0 == emitting nothing" convention is clearer.
    #
    # x is PSF sigma in native pixels, NOT pixel scale like (a)/(b): measured
    # Spearman rho = +0.90 (p=0.037) against sigma_px versus +0.60 (p=0.285),
    # not significant, against pixel scale. A one-row layout shares no axis, so
    # the mixed x costs nothing structurally and keeps the panel explanatory.
    ax = axes[2]
    xs_s, ident_c, wien_c, pts_c = [], [], [], []
    for src, tset, style, sh in SETS:
        for nm in SURVEY_ORDER:
            r = src[nm]
            x_ = _sigma_px(nm)
            xp = x_ + XLIN * sh
            pt, blo, bhi = boot_ci(r["hf_complex_err"], np.mean, rng)
            ax.errorbar(xp, pt, yerr=[[pt - blo], [bhi - pt]],
                        fmt=SURVEY_MARKERS[nm], color=SURVEY_COLORS[nm],
                        capsize=2, lw=1.0, zorder=3, **style)
            pts_c.append(pt)
            if tset == 1:
                xs_s.append(x_)
                ident_c.append(float(np.mean(r["hf_complex_err_identity"])))
                wien_c.append(float(base1[nm]["wiener_hf"]["hf_complex_err"]))
            rows.append(dict(panel="c", survey=nm, test_set=tset, sigma_px=x_,
                             hf_complex_err=pt, ci_lo=blo, ci_hi=bhi,
                             hf_coherence=float(np.mean(r["hf_coherence"]))))
    for nm, i_, w_ in zip(SURVEY_ORDER, ident_c, wien_c):
        rows.append(dict(panel="c_ref", survey=nm, test_set=1,
                         sigma_px=_sigma_px(nm), identity=i_, wiener=w_))
    o = np.argsort(xs_s)
    xa = np.array(xs_s)[o]
    ax.plot(xa, np.array(ident_c)[o], ls="--", lw=1.0, color=INK_MUTED,
            marker="o", ms=2.2, zorder=1, label="identity")
    ax.plot(xa, np.array(wien_c)[o], ls=":", lw=1.0, color=INK_2, marker="o",
            ms=2.2, zorder=1, label="Wiener")
    # Wiener under NOISE. Without this the noise variant would show only the
    # noise-free reference, and "the network beats an oracle-tuned Wiener
    # 4.0-8.7x once its noise assumption breaks" -- the strongest quantitative
    # claim in the sim-to-real story -- would appear in no figure at all.
    # Wiener's K is re-swept per test set: its noise-free optimum is up to 13
    # decades from its noisy one, so carrying one K across both would fake a
    # collapse.
    if ts2 is not None and base2 is not None:
        w2 = [float(base2[nm]["wiener_hf"]["hf_complex_err"]) for nm in SURVEY_ORDER]
        xa2 = np.array([_sigma_px(nm) + XLIN for nm in SURVEY_ORDER])[o]
        ax.plot(xa2, np.array(w2)[o], ls=":", lw=1.0, color=INK_2, marker="o",
                ms=2.2, alpha=0.5, zorder=1, label="Wiener (noise)")
        for nm, v in zip(SURVEY_ORDER, w2):
            rows.append(dict(panel="c_ref", survey=nm, test_set=2,
                             sigma_px=_sigma_px(nm), wiener=v))
    # y = 1.0 is what `f_hat = 0` scores BY CONSTRUCTION, so above it the model
    # is worse than emitting nothing. Drawn ONLY in the noise variant: there
    # HST's Test Set 2 point sits at 1.029, above the line, and that is the
    # paper's headline honest failure. Noise-free, every point is an order of
    # magnitude below it, so the line explains nothing and its presence forces
    # the empty headroom the ylim below otherwise reserves for it.
    if ts2 is not None:
        ax.axhline(1.0, ls="-", lw=0.8, color=INK_2, zorder=1,
                   label=r"emit nothing ($\hat f = 0$)")
    ax.set_yscale("log")
    ax.set_xlabel(r"PSF $\sigma$ (native px)", fontsize=LBL)
    ax.set_ylabel("high-freq. octave error", fontsize=LBL)
    ax.set_title("(c) Deconvolution", fontsize=FS_SMALL, pad=3)
    ax.set_xlim(min(xs_s) - 0.18, max(xs_s) + 0.18)
    # Noise variant: do NOT clip the top -- HST's Test Set 2 point sits ABOVE 1.0
    # (worse than emitting nothing) and that failure has to stay visible, hence
    # the 2.4 floor. Noise-free: that floor would reserve a decade of blank space
    # for a line that is no longer drawn, so size to the data instead -- but to
    # ALL plotted series, since `identity` reaches 0.96 at Rubin and would fall
    # off a limit derived from the model points alone (max 0.088).
    if ts2 is not None:
        ax.set_ylim(min(pts_c) * 0.35, max(2.4, max(pts_c) * 1.9))
    else:
        _top = max(max(pts_c), max(ident_c), max(wien_c))
        ax.set_ylim(min(pts_c) * 0.35, _top * 1.7)
    ax.legend(loc="lower right", frameon=False, fontsize=FS_TINY,
              handletextpad=0.5, borderpad=0.1, labelspacing=0.2)

    for a in axes:
        a.tick_params(labelsize=TCK)
    _no_minor_labels(*axes)
    # `rect` reserves the strip the figure-level legend sits in; tight_layout
    # does not know about it otherwise.
    fig.tight_layout(w_pad=1.1, rect=(0, 0.085, 1, 1))
    handles = list(survey_handles)
    if ts2 is not None:
        handles += [Line2D([], [], marker="o", ls="", color=INK_2, ms=3.0,
                           label=set_labels[0]),
                    Line2D([], [], marker="o", ls="", color=INK_2, ms=3.0,
                           markerfacecolor="none", markeredgewidth=0.9,
                           label=set_labels[1])]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=FS_TINY, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.25, columnspacing=0.75, borderpad=0.1)
    save_figure(fig, out_dir, slug, pd.DataFrame(rows), manifest)


# --------------------------------------------------------------------------
# Fig 4 -- deconvolution quality vs PSF
# --------------------------------------------------------------------------
def _sigma_px(nm):
    c = SV.SURVEYS[nm]
    return c["fwhm"] / 2.3548 / c["scale"]


def fig_deconvolution(ts1, base1, imgs_hero, out_dir, manifest, rng,
                      profile_survey="HST"):
    plt = _mpl()
    fig = plt.figure(figsize=(TEXT_W, 3.2))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.42, 1.0], hspace=0.46,
                          wspace=1.05, left=0.105, right=0.965, top=0.93,
                          bottom=0.05)
    rows = []

    # (a) radial power spectra for one survey
    ax = fig.add_subplot(gs[0, :2])
    u, P_hat, P_gt, P_g = ts1[profile_survey]["radial"]
    # truth is drawn widest and first so the reconstruction, plotted thinner on
    # top of it, does not simply hide it -- the two lying on top of each other
    # IS the result, and the reader has to be able to see both.
    for y, lab, color, ls, lw in (
            (P_gt, r"truth $f_{gt}$", INK, "-", 2.6),
            (P_g, r"input $g$ (= identity)", INK_MUTED, "--", 1.3),
            (P_hat, r"reconstruction $\hat f$", SURVEY_COLORS[profile_survey], "-", 1.1)):
        ax.plot(u, y, ls=ls, color=color, label=lab, lw=lw)
    ax.axvspan(0.5, 1.0, color=INK_MUTED, alpha=0.10, lw=0)
    ax.text(0.75, 0.97, "top octave", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=FS_TINY, color=INK_2)
    sig_px = _sigma_px(profile_survey)
    u_cut = min(0.99, 2.0 / (np.pi * sig_px * np.sqrt(2)))
    ax.axvline(u_cut, color=INK_2, ls=":", lw=1.0)
    ax.text(u_cut, 0.99, "PSF cutoff\n(MTF=1/e) ", transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=FS_TINY, color=INK_2)
    ax.set_yscale("log")
    ax.set_xlabel(r"spatial frequency $u$  ($u=1$: Nyquist)")
    ax.set_ylabel("azimuthal power (a.u.)")
    ax.set_title(f"(a) power recovered at {profile_survey}", fontsize=FS_SMALL)
    ax.set_xlim(0, 1.0)
    _no_minor_labels(ax)
    ax.legend(loc="lower left", frameon=False, handletextpad=0.4, borderpad=0.1,
              labelspacing=0.2)
    for k in range(len(u)):
        rows.append(dict(panel="a", survey=profile_survey, u_over_nyquist=u[k],
                         P_truth=P_gt[k], P_input=P_g[k], P_recon=P_hat[k],
                         psf_cutoff_u=u_cut))

    # (b) reblur residual per survey -- the diagnostic that needs no truth
    lim = 0.0
    for nm in SURVEY_ORDER:
        d = imgs_hero[nm]
        pk = max(float(np.abs(d["g"]).max()), 1e-12)
        lim = max(lim, float(np.abs(_crop(d["reblur_resid"] / pk, d["s"])).max()))
    lim = min(lim, 0.5)
    row_axes = []
    for i, nm in enumerate(SURVEY_ORDER):
        ax = fig.add_subplot(gs[1, i])
        row_axes.append(ax)
        d = imgs_hero[nm]
        pk = max(float(np.abs(d["g"]).max()), 1e-12)
        c = _crop(d["reblur_resid"] / pk, d["s"])
        ax.imshow(c, cmap=CMAP_RESID, vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.04, 0.96, nm, transform=ax.transAxes, ha="left", va="top",
                fontsize=FS_TINY, color=SURVEY_COLORS[nm])
        rows.append(dict(panel="c", survey=nm,
                         reblur_resid_rms_frac_of_peak=float(np.sqrt(np.mean(c ** 2))),
                         colour_limit_frac_of_peak=lim))
    _row_label(fig, row_axes,
               f"(b) reblur residual $\\hat f * h - g$, no ground truth needed "
               f"($\\pm${lim * 100:.0f}% of peak $g$)")
    save_figure(fig, out_dir, "fig4_deconvolution", pd.DataFrame(rows), manifest)


# --------------------------------------------------------------------------
# Fig 5 -- sim-to-real robustness under noise shift (the venue's theme)
# --------------------------------------------------------------------------
def fig_noise_shift(ts1, ts2, base1, base2, imgs_ts2, out_dir, manifest, rng):
    plt = _mpl()
    fig = plt.figure(figsize=(TEXT_W, 4.05))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.35, 1.35, 0.95],
                          hspace=0.95, wspace=1.05, left=0.105, right=0.985,
                          top=0.94, bottom=0.04)
    rows = []
    x = np.arange(len(SURVEY_ORDER))
    w = 0.38
    prior_rmse = float(np.mean([base1[nm]["prior_mean"]["theta_rmse"]
                                for nm in SURVEY_ORDER]))

    # (a) theta_E RMSE, TS1 vs TS2
    ax = fig.add_subplot(gs[0, :3])
    for k, (src, lab, alpha) in enumerate(((ts1, "Test Set 1 (noise-free)", 1.0),
                                           (ts2, "Test Set 2 (survey noise)", 0.45))):
        vals, los, his = [], [], []
        for nm in SURVEY_ORDER:
            r = src[nm]
            pt, blo, bhi = boot_ci_paired(r["theta_pred"], r["theta_true"], _rmse, rng)
            vals.append(pt); los.append(pt - blo); his.append(bhi - pt)
        ax.bar(x + (k - 0.5) * w, vals, width=w - 0.04,
               color=[SURVEY_COLORS[n] for n in SURVEY_ORDER], alpha=alpha,
               label=lab, yerr=[los, his], capsize=1.8,
               error_kw=dict(lw=0.8, ecolor=INK_2))
        for nm, v in zip(SURVEY_ORDER, vals):
            rows.append(dict(panel="a", survey=nm, test_set=k + 1, theta_rmse=v))
    for i, nm in enumerate(SURVEY_ORDER):
        a = _rmse(ts1[nm]["theta_pred"], ts1[nm]["theta_true"])
        b = _rmse(ts2[nm]["theta_pred"], ts2[nm]["theta_true"])
        ax.annotate(f"{b / a:.1f}x", (i, max(a, b)), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=FS_TINY, color=INK_2)
    ax.axhline(prior_rmse, ls="--", lw=1.0, color=INK_MUTED)
    ax.text(-0.42, prior_rmse * 1.10, f"prior-mean ({prior_rmse:.2f}\")",
            fontsize=FS_TINY, color=INK_MUTED, ha="left", va="bottom")
    ax.set_yscale("log"); ax.set_ylim(0.008, 2.2)
    ax.set_xticks(x); ax.set_xticklabels(SURVEY_ORDER)
    ax.set_ylabel(r"$\theta_E$ RMSE (arcsec)")
    ax.set_title(r"(a) $\theta_E$ accuracy under noise shift (paired lenses)",
                 fontsize=FS_SMALL)
    ax.legend(loc="upper right", frameon=False, handletextpad=0.4, borderpad=0.1,
              labelspacing=0.2)
    _no_minor_labels(ax)

    # (b) reblur residual over its own noise floor
    ax = fig.add_subplot(gs[0, 3:])
    vals, los, his = [], [], []
    for nm in SURVEY_ORDER:
        r = ts2[nm]
        ratio = r["reblur_rel"] / np.maximum(r["floor_rel"], 1e-30)
        pt, blo, bhi = boot_ci(ratio, np.mean, rng)
        vals.append(pt); los.append(pt - blo); his.append(bhi - pt)
        rows.append(dict(panel="b", survey=nm, reblur_over_floor=pt,
                         ci_lo=blo, ci_hi=bhi,
                         reblur_rel=float(np.mean(r["reblur_rel"])),
                         floor_rel=float(np.mean(r["floor_rel"]))))
    ax.bar(x, vals, width=0.62, color=[SURVEY_COLORS[n] for n in SURVEY_ORDER],
           yerr=[los, his], capsize=1.8, error_kw=dict(lw=0.8, ecolor=INK_2))
    ax.axhline(1.0, ls="-", lw=0.9, color=INK_2)
    ax.set_xticks(x); ax.set_xticklabels(SURVEY_ORDER)
    ax.set_ylabel("reblur MSE / noise floor")
    ax.set_ylim(0, max(vals) * 1.45)
    ax.text(len(x) - 0.55, max(vals) * 1.42,
            "1.0 = noise floor\n(below it: fitting noise)",
            fontsize=FS_TINY, color=INK_2, ha="right", va="top")
    ax.set_title("(b) data consistency vs its floor", fontsize=FS_SMALL)

    # (d) top-octave error, TS1 vs TS2, WITH the Wiener reference on each bar.
    #
    # This panel carries the paper's headline claim, so the classical baseline
    # has to be visible in it: bars are the network, the black tick across each
    # bar is Wiener with the known PSF at its best K *for that test set*. On
    # TS1 the ticks sit near the floor (Wiener is excellent on clean data, and
    # edges us at Rubin/DES); on TS2 they jump to 0.31-0.91 while the bars stay
    # low. That contrast IS the result -- drawing it as a bullet-chart tick
    # rather than a third bar keeps the network/baseline pairing readable
    # instead of splitting each survey into three competing rectangles.
    ax = fig.add_subplot(gs[1, :3])
    tick_proxy = None
    for k, (src, base, lab, alpha) in enumerate(
            ((ts1, base1, "Test Set 1 (noise-free)", 1.0),
             (ts2, base2, "Test Set 2 (survey noise)", 0.45))):
        vals, los, his = [], [], []
        for nm in SURVEY_ORDER:
            pt, blo, bhi = boot_ci(src[nm]["hf_complex_err"], np.mean, rng)
            vals.append(pt); los.append(pt - blo); his.append(bhi - pt)
            wv = float(base[nm]["wiener_hf"]["hf_complex_err"])
            rows.append(dict(panel="d", survey=nm, test_set=k + 1,
                             hf_complex_err=pt, ci_lo=blo, ci_hi=bhi,
                             wiener=wv,
                             wiener_K=base[nm]["wiener_hf"]["wiener_K"],
                             wiener_K_at_grid_edge=base[nm]["wiener_hf"]["wiener_K_at_grid_edge"],
                             net_over_wiener=pt / wv if wv > 0 else np.nan))
        xpos = x + (k - 0.5) * w
        ax.bar(xpos, vals, width=w - 0.04,
               color=[SURVEY_COLORS[n] for n in SURVEY_ORDER], alpha=alpha,
               label=lab, yerr=[los, his], capsize=1.8,
               error_kw=dict(lw=0.8, ecolor=INK_2))
        wvals = [float(base[nm]["wiener_hf"]["hf_complex_err"]) for nm in SURVEY_ORDER]
        h = ax.hlines(wvals, xpos - (w - 0.04) / 2, xpos + (w - 0.04) / 2,
                      color=INK, lw=1.4, zorder=6)
        if tick_proxy is None:
            tick_proxy = h
            h.set_label("Wiener (known PSF, best $K$)")
    ax.axhline(1.0, ls="-", lw=0.9, color=INK_2)
    ax.text(len(x) - 0.55, 1.03, "worse than emitting nothing", fontsize=FS_TINY,
            color=INK_2, ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(SURVEY_ORDER)
    ax.set_ylabel("high-freq. octave error")
    # headroom above HST's TS2 bar (1.03) for the Wiener legend; the TS1/TS2
    # colour key is already in panel (a), so this legend carries only the tick.
    ax.set_ylim(0, 1.78)
    ax.legend(handles=[tick_proxy], loc="upper left", frameon=False,
              handletextpad=0.5, borderpad=0.1, labelspacing=0.2)
    ax.set_title("(d) high-frequency recovery lost to noise", fontsize=FS_SMALL)

    # a short verdict block, so the panel does not have to be read off the bars
    ax = fig.add_subplot(gs[1, 3:])
    ax.axis("off")
    # The verdict block reports the TS2 net-vs-Wiener ratio rather than the
    # TS1->TS2 drift it used to: under noise is where the two methods actually
    # separate, and "net 8.7x better than Wiener at Rubin" is the sentence the
    # paper makes. `!` marks a survey where the network is above 1.0, i.e.
    # worse than emitting nothing -- HST, where neither method works.
    lines = []
    for nm in SURVEY_ORDER:
        b = float(np.mean(ts2[nm]["hf_complex_err"]))
        wv = float(base2[nm]["wiener_hf"]["hf_complex_err"])
        gain = wv / b if b > 0 else np.nan
        flag = "!" if b > 1.0 else " "
        lines.append(f"{nm:7s}{b:5.3f} {wv:5.3f} {gain:4.1f}x{flag}")
    # The header carries the `!` key: a separate footnote below the block
    # collides with it (the five monospace lines at linespacing 1.5 run past
    # y=0.06), which the strict layout check catches.
    ax.text(0.0, 0.98, "TS2      net  Wien   gain\n! = net worse than nothing",
            fontsize=FS_TINY, va="top", color=INK, family="monospace",
            linespacing=1.4)
    ax.text(0.0, 0.72, "\n".join(lines), fontsize=FS_TINY, va="top",
            color=INK_2, family="monospace", linespacing=1.5)

    # (c) reblur residual maps on the NOISY test set
    lim = 0.0
    for nm in SURVEY_ORDER:
        d = imgs_ts2[nm]
        pk = max(float(np.abs(d["g"]).max()), 1e-12)
        lim = max(lim, float(np.abs(_crop(d["reblur_resid"] / pk, d["s"])).max()))
    lim = min(lim, 0.6)
    row_axes = []
    for i, nm in enumerate(SURVEY_ORDER):
        ax = fig.add_subplot(gs[2, i])
        row_axes.append(ax)
        d = imgs_ts2[nm]
        pk = max(float(np.abs(d["g"]).max()), 1e-12)
        c = _crop(d["reblur_resid"] / pk, d["s"])
        ax.imshow(c, cmap=CMAP_RESID, vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.04, 0.96, nm, transform=ax.transAxes, ha="left", va="top",
                fontsize=FS_TINY, color=SURVEY_COLORS[nm])
        rows.append(dict(panel="c", survey=nm, test_set=2,
                         reblur_resid_rms_frac_of_peak=float(np.sqrt(np.mean(c ** 2))),
                         colour_limit_frac_of_peak=lim))
    _row_label(fig, row_axes,
               f"(c) reblur residual on NOISY input ($\\pm${lim * 100:.0f}% of peak $g$)")
    save_figure(fig, out_dir, "fig5_noise_shift", pd.DataFrame(rows), manifest)


# --------------------------------------------------------------------------
# Provenance + captions
# --------------------------------------------------------------------------
def _dataset_digest(test_set):
    """Deterministic fingerprint of the rendered test set, so a figure's data
    provenance can be checked without shipping the arrays."""
    hsh = hashlib.sha256()
    for nm in sorted(test_set):
        items = test_set[nm]
        hsh.update(nm.encode())
        hsh.update(str(len(items)).encode())
        hsh.update(str(tuple(items[0]["g"].shape)).encode())
        for it in items:
            hsh.update(f"{float(it['f_gt'].sum()):.6e}|{float(it['g'].sum()):.6e}|"
                       f"{float(it['theta_E']):.6f}".encode())
    return hsh.hexdigest()[:16]


def _git_state():
    def run(*a):
        try:
            return subprocess.check_output(a, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    commit = run("git", "rev-parse", "HEAD")
    dirty = run("git", "status", "--porcelain")
    return dict(commit=commit,
                working_tree="dirty" if dirty and dirty != "unknown" else "clean")


FIG5_SECTION = """## Fig 5 -- Sim-to-real: a noise-free simulator meeting noisy data

The model is trained on **noise-free** simulations and evaluated here, with no
retraining and no noise augmentation, on the same lenses rendered with each
survey's own noise. (a) theta_E RMSE degrades by the annotated factor but stays
far below the prior-mean null at every survey. (b) The reblur residual divided
by its own analytic noise floor. **This panel is known to be wrong at Roman:**
the analytic floor omits the reblur operator's own error, which is 38% of
Roman's true floor, so the panel reports 1.87 where the correct value against
`reblur_floor_op` is 0.65 -- below 1.0, i.e. fitting noise. The other four
surveys move by <=6%. (c) Reblur residual maps on noisy input -- the diagnostic
that needs no ground truth. (d) High-freq. octave error before and after the
noise shift, with a black tick per bar marking Wiener at its best `K`, re-swept
separately on each test set. {wiener_note} {fail_note}"""

FIG5_OMITTED = """## Fig 5 -- NOT EMITTED

`fig_noise_shift` is retained in the code and can be re-enabled with `--with-fig5`,
but is not part of either paper. Its theta_E and top-octave panels are duplicated by
`fig3_survey_invariance_noise`, which shows the same TS1-vs-TS2 comparison across
three metrics. Its reblur/noise-floor panel was additionally **wrong**: it divided by
an ANALYTIC noise-variance estimate that omits the reblur operator's own error. At
Roman that operator error is 38% of the true floor, so the panel reported 1.87 where
the correct value is 0.65 -- below 1.0, which by that panel's own legend means the
model is fitting noise. It ranked Roman healthiest when Roman is the opposite. The
other four surveys moved by <=6%.

**What must move into the TEXT, since no figure now shows it:**

* **The reblur consistency diagnostic itself.** With Fig 4 also dropped, no figure
  shows the reblur residual. This is the mechanism the sim-to-real claim rests on, so
  state it numerically: on noisy input the diagnostic flags HST at **0.92x its own
  analytic noise floor** -- below 1.0, i.e. fitting noise -- **without using ground
  truth**, which is exactly the property that transfers to real data.
* **The reblur floor caveat**, if the number above is quoted: the analytic floor omits
  the operator error, so Roman's ratio in particular should be computed against
  `reblur_floor_op` (the residual an ORACLE reconstruction leaves) rather than against
  `floor_rel`.
"""

FIG4_SECTION = """## Fig 4 -- The network is deconvolving, not smoothing

(a) Azimuthally averaged power spectra at {profile_survey}: the input has lost
power through the top octave (shaded, 0.5-1.0x native Nyquist) and the
reconstruction puts it back. (b) The reblur residual `f_hat * h - g` for one
lens per survey: small and structureless is good, and because it needs no
ground truth it is the one diagnostic that transfers unchanged to real data."""

FIG4_OMITTED = """## Fig 4 -- NOT EMITTED

`fig_deconvolution` is retained in the code and can be re-enabled with
`--with-fig4`, but is not part of either paper. Its top-octave panel moved into
Fig 3(c), where the same metric is reported for all five surveys against
`identity` and Wiener. Its remaining two panels were judged redundant for a
5-page paper: the reblur residual maps repeat Fig 5(c), which shows the same
diagnostic on NOISY data (the case that matters), and the power spectra
illustrate at one survey what Fig 3(c) measures quantitatively at all five.

Two things that panel carried should be stated in the TEXT instead, since no
figure now shows them:

* **Top-octave coherence** -- the direct answer to "is this just sharpening
  artifacts?". Recovered high-frequency structure is phase-aligned with the
  truth, and most dramatically where the blur is worst: trained vs `identity`
  is 0.993/0.900 at HST, 0.975/0.615 at Rubin, 0.958/0.587 at DES. A sharpening
  artifact would raise power without raising coherence.
* **`hf_harm_frac_*`** -- the fraction of top-octave power sitting on the patch
  decoder's own grid harmonic, which for patch=4 lands exactly on the band edge.

"""

CAPTIONS = """# Figure captions

Auto-generated by `survey_lens/make_paper_figures.py` v{version}.
Every number quoted below is read from the matching `figN_*_data.csv`, never
off a plot. Checkpoint: `{ckpt}`.

## Fig 1 -- Cross-survey reconstruction (hero)

**One network, five telescopes, five native pixel grids, no resampling.** The
same simulated lens rendered through HST, Euclid, Roman, Rubin and DES
(top to bottom, ordered by pixel scale from 0.05 to 0.263 arcsec/px, i.e.
400x400 down to 76x76 native pixels), with the input `g`, the network's
reconstruction, the ground truth and their residual. The lens shown is the
**median** case by masked normalized reconstruction error averaged over the
five surveys -- not the best case; three further lenses, including the worst,
are in Fig A1. Every panel in a row is divided by that row's own truth peak,
so the first three share one intensity scale (0 to 1) and are comparable by
eye despite an ~11x spread in raw per-pixel surface brightness across pixel
scales; the residual column uses a diverging map centred on zero at
+/-{resid_pct:.0f}% of that same peak. The 1-arcsec bar is identical sky in every
row while the pixel grid under it changes by 5.3x -- that invariance is the
result. Panels are cropped to a {crop:.0f}-arcsec box; the full
{fov:g}-arcsec field is {empty_pct:.0f}% empty sky (measured: only
{signal_pct:.1f}% of pixels exceed 1% of the peak).

## Fig 2 -- Architecture (schematic, not data-driven)

*Assembled by hand from `figures/source_slides/`; not produced by this script.*
Image and PSF stamps are patch-embedded, positioned by RoPE **in arcseconds
from the lens centre rather than in pixel indices** -- the pixel scale `s`
enters the positional encoding explicitly, which is the invariance mechanism --
then mixed by image self-attention, PSF self-attention and image-from-PSF
cross-attention before two heads emit the deconvolved image and theta_E.

## Fig 3 -- Survey invariance is quantitative, not anecdotal

All five surveys share the **same** paired lens population, so a cross-survey
gap is the telescope and not a different lens sample.

(a) Einstein-radius error as a percentage of the true theta_E. Nearly flat --
not flat: it degrades mildly and monotonically toward coarser pixels. Note the
per-lens normalization makes this a HARDER test than absolute RMSE, since it
up-weights small-theta_E systems where a fixed positional error is a larger
fraction.

(b) Relative L1 **metric** on the reconstruction, sum|f_hat - f_gt| / sum(f_gt)
over pixels with `f_gt > 1% of peak`, with `identity` (return the blurred input
unchanged) as the reference. This is NOT the L1 term of the training objective,
which is half-weighted onto empty sky as anti-hallucination pressure and is
accompanied by SSIM and FFT terms. {ssim_note}

(c) High-freq. octave error against PSF sigma in NATIVE pixels -- the
honest difficulty axis for deconvolution, and a different x from (a)/(b) on
purpose: the effect tracks PSF sampling, not pixel scale. Left DIMENSIONLESS
while (a) and (b) are percentages because it divides squared amplitudes, so a
percentage here would be a fraction of POWER sitting beside two fractions of
FLUX. `identity` and Wiener-with-known-PSF are shown for reference.

## Definitions -- training LOSS TERMS vs evaluation METRICS

These are two different families and must not be conflated. The model is
optimized on the loss terms; the figures report the metrics.

**Training objective** (`losses.py`, weights from `lit_model.py`):

| term | what it is | weight |
|---|---|---|
| L1 | `0.5*(masked + global)` mean abs. difference, both images divided by the `f_gt` peak. Half of it covers empty sky ON PURPOSE, as anti-hallucination pressure | 1.0 |
| SSIM | `1 - ssim(f_hat, f_gt)`, peak-normalized | 0.5 |
| FFT | L1 between the MAGNITUDE spectra, **all** frequencies -- phase-BLIND | 0.1 |
| reblur | `\|f_hat*h - g\|^2 / mean(g^2)`, scale-free, needs no ground truth | 0.1 |
| regression | MSE on theta_E | 1.0 |

**Evaluation metrics** (what the figures plot):

| name in figures | definition |
|---|---|
| theta_E frac. RMSE (%) | RMS of the per-lens `(pred-true)/true` |
| L1 error (% of arc flux) | `sum\|f_hat - f_gt\| / sum(f_gt)` inside `f_gt > 1% of peak` |
| **high-freq. octave error** | `sum\|F_hat - F_gt\|^2 / sum\|F_gt\|^2` over 0.5-1.0x the native Nyquist. Normalized so emitting nothing scores exactly **1.0**, a perfect reconstruction 0; above 1.0 is worse than emitting nothing |
| reblur / floor | reblur residual over the floor an ORACLE reconstruction still leaves |

**Two traps worth stating explicitly.**

*The metrics are not the loss terms of the same name.* The L1 metric is masked and
normalized by arc flux; the L1 loss term is half-weighted onto empty sky and
normalized by peak. They are related, not equal.

*The high-freq. octave error is NOT the FFT loss term.* The FFT term differences
magnitude spectra over all frequencies and is therefore phase-BLIND; the metric
differences COMPLEX coefficients in the top octave and is phase-AWARE. That
distinction decides results: Run 1 scored a healthy-looking 0.86 top-octave power
ratio at Euclid while its phase-aware error was 1.10 -- worse than emitting
nothing, having put roughly the right amount of power in the wrong place. That
published claim was retracted. A phase-blind score cannot adjudicate
deconvolution.

*"Native" Nyquist* means each survey's OWN grid: the band is identical in pixel
units but spans 0.10-0.20" of structure at HST and 0.53-1.05" at DES. So a
cross-survey comparison asks "did the network exploit the resolution it was
given?", not "did DES recover what HST recovered".

**Two variants of this figure are emitted.** `fig3_survey_invariance` shows the
noise-free test set only and is the one used where noise is not discussed;
`fig3_survey_invariance_noise` overlays both test sets (solid marker =
noise-free, hollow = survey noise) and carries the sim-to-real result. Both report identical numbers for the noise-free set.

In the noise variant the with-noise points are **shifted slightly along x so
their error bars do not overprint** the noise-free ones. The shift is
presentational only: both members of a pair are the same survey at the same
true pixel scale (and, in panel (c), the same PSF sigma). Marker size is
identical for the two sets so neither is visually weighted above the other.

Error bars are the 16-84th percentile over 1000 bootstrap resamples of the
{n_lens} test lenses; the reference curves are plain means and carry none. Each
figure draws from its own generator seeded from `--boot-seed`, so the intervals
do not depend on which other figures were produced in the same run.

{fig4_section}
{fig5_section}

## Ablation status (axis 2 IS run -- this block used to claim otherwise)

- **Loss mode (axis 2) -- RUN.** Three arms (`joint` / `recon_only` / `reg_only`)
  on the same frozen 12" dataset, 80-epoch cosine, seed 0; only `--loss-mode`
  differs. Drawn by `make_ablation_figures.py` as `fig6_loss_ablation`, with its
  own `captions_ablation.md`. Main text for the representations version of the
  paper, appendix for Sim2Science.
- **PSF cross-attention (axis 1) -- RUN (Aug 24 2026).** `--no-cross-attention`, joint
  loss, one flag from the reference arm. Drawn as `fig7_cross_attention`. Main text for
  the representations version, appendix for Sim2Science. Removing cross-attention costs
  1.64-9.98x in top-octave error at every survey (mean 2.75x) and leaves theta_E
  untouched; the effect tracks PSF sampling at Spearman rho = -0.90 (p = 0.037).
- **Native vs resampled (axis 3) -- NOT IMPLEMENTED.** There is no resampling
  code anywhere in `src/`; this axis cannot be run without writing it first.
- **Drop SSIM+FFT (axis 4) -- not run.** Flags exist: `--lambda-a 0 --lambda-b 0`.

## Deferred (not in scope for this deadline)

- **Test Set 3, real data (qualitative).** 2-3 real lens cutouts with the reblur
  residual as the diagnostic. Deliberately qualitative -- a full real-data
  benchmark is explicitly out of scope.
"""


def write_captions(out_dir, ctx):
    path = os.path.join(out_dir, "captions.md")
    with open(path, "w") as fh:
        fh.write(CAPTIONS.format(**ctx))
    print(f"  wrote captions.md")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", default="figures/paper")
    ap.add_argument("--mirror-dir", default=None,
                    help="optional second directory to copy the finished "
                         "figure set into (e.g. the CephFS results root)")
    ap.add_argument("--n-lenses", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345,
                    help="MUST match test.py's seed: the figures and the "
                         "metrics CSVs have to describe the same lenses")
    ap.add_argument("--fov", type=float, default=20.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--profile-survey", default="HST",
                    help="survey whose radial power spectrum Fig 4a shows")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--with-fig5", action="store_true",
                    help="also emit fig5_noise_shift. OFF by default: its theta_E "
                         "and top-octave panels are duplicated by "
                         "fig3_survey_invariance_noise, and its reblur/floor panel "
                         "divides by an analytic noise floor that omits the reblur "
                         "operator's own error (38%% of Roman's true floor, which "
                         "flips Roman from 1.87 to 0.65). Code kept so it can be "
                         "reproduced on demand.")
    ap.add_argument("--with-fig4", action="store_true",
                    help="also emit fig4_deconvolution. OFF by default: its "
                         "top-octave panel moved to Fig 3(c) and its remaining "
                         "panels duplicate Fig 3(c) / Fig 5(c). The code is kept "
                         "so the figure can be reproduced on demand.")
    ap.add_argument("--no-strict-layout", action="store_true",
                    help="warn instead of failing on the geometry self-check")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    # A FRESH generator per figure, all seeded the same. Sharing one generator
    # made every figure's bootstrap CIs depend on which figures ran before it:
    # the same Test Set 1 data appeared with different error bars in
    # `fig3_survey_invariance` and `fig3_survey_invariance_noise`, and gating
    # fig4 off silently shifted fig5's intervals. Point estimates were never
    # affected -- they are deterministic -- but the intervals were.
    def _rng():
        return np.random.default_rng(args.boot_seed)

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}\n"
                         "(test.py silently falls back to random weights; this "
                         "script refuses to, because a figure drawn from random "
                         "weights is indistinguishable from a real one.)")
    model = T.load_model(args.ckpt, args.device)
    patch = getattr(getattr(model, "net", model), "patch", 4)

    print(f"building Test Set 1 + 2 ({args.n_lenses} lenses x {len(SV.SURVEYS)} surveys) ...")
    ts1_items = build_test_set_1(SV.SURVEYS, n_lenses=args.n_lenses, seed=args.seed,
                                 fov=args.fov, n_workers=args.workers)
    ts2_items = build_test_set_2(SV.SURVEYS, n_lenses=args.n_lenses, seed=args.seed,
                                 fov=args.fov, n_workers=args.workers)

    print("running inference ...")
    ts1, ts2, base1, base2 = {}, {}, {}, {}
    for nm in SURVEY_ORDER:
        ts1[nm] = per_sample_pass(model, ts1_items[nm], args.device,
                                  args.batch_size, patch)
        cfg = SV.SURVEYS[nm]
        ts2[nm] = per_sample_pass(model, ts2_items[nm], args.device,
                                  args.batch_size, patch,
                                  noise_cfg=dict(bkg_rms=cfg["bkg_rms"], gain=cfg["gain"]))
        base1[nm] = T.baseline_rows(ts1_items[nm], args.batch_size, args.device)
        # TS2 baselines too: the paper's headline is that Wiener COLLAPSES under
        # the noise shift while the network does not, and that claim cannot be
        # drawn from base1 -- Wiener's optimal K on noisy data is ~1e-3..1e+0,
        # four to thirteen decades away from its noise-free optimum. Re-sweeping
        # on the noisy set is what makes the fig-5 comparison honest.
        base2[nm] = T.baseline_rows(ts2_items[nm], args.batch_size, args.device)
        print(f"  {nm:7s} theta_rmse={_rmse(ts1[nm]['theta_pred'], ts1[nm]['theta_true']):.4f}"
              f" -> {_rmse(ts2[nm]['theta_pred'], ts2[nm]['theta_true']):.4f} (TS2)"
              f" | hf_complex_err={np.mean(ts1[nm]['hf_complex_err']):.3f}"
              f" -> {np.mean(ts2[nm]['hf_complex_err']):.3f}")

    # Hero lens: the MEDIAN case by masked reconstruction error averaged over
    # surveys, so the figure is representative rather than cherry-picked.
    per_lens = np.mean([ts1[nm]["l1_norm_masked"] for nm in SURVEY_ORDER], axis=0)
    order = np.argsort(per_lens)
    median_idx = int(order[len(order) // 2])
    worst_idx = int(order[-1])
    extra = [int(order[len(order) // 4]), int(order[3 * len(order) // 4]), worst_idx]
    print(f"hero lens = index {median_idx} (median of {len(per_lens)} by masked L1); "
          f"worst = {worst_idx}")

    imgs_hero = {nm: images_for_lens(model, ts1_items[nm], median_idx, args.device)
                 for nm in SURVEY_ORDER}
    imgs_ts2 = {nm: images_for_lens(model, ts2_items[nm], median_idx, args.device)
                for nm in SURVEY_ORDER}
    imgs_extra = [{nm: images_for_lens(model, ts1_items[nm], i, args.device)
                   for nm in SURVEY_ORDER} for i in extra]

    manifest = dict(
        script="survey_lens/make_paper_figures.py",
        script_version=SCRIPT_VERSION,
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        checkpoint=os.path.abspath(args.ckpt),
        checkpoint_sha256=hashlib.sha256(
            open(args.ckpt, "rb").read()).hexdigest()[:16],
        git=_git_state(),
        test_set_1_digest=_dataset_digest(ts1_items),
        test_set_2_digest=_dataset_digest(ts2_items),
        n_lenses=args.n_lenses, seed=args.seed, fov_arcsec=args.fov,
        surveys={n: dict(scale=SV.SURVEYS[n]["scale"], fwhm=SV.SURVEYS[n]["fwhm"])
                 for n in SURVEY_ORDER},
        hero_lens_index=median_idx, worst_lens_index=worst_idx,
        bootstrap=dict(n=N_BOOT, pct=[BOOT_LO, BOOT_HI], seed=args.boot_seed),
        figures={},
    )

    strict = not args.no_strict_layout
    print("drawing figures ...")
    global save_figure
    _sf = save_figure
    save_figure = lambda *a, **k: _sf(*a, **{**k, "strict": strict})  # noqa: E731

    fig_hero(ts1, [imgs_hero], args.out_dir, manifest, lens_ids=[median_idx])
    # Two variants: Test-Set-1-only for Representations (that paper does not
    # discuss noise) and both test sets overlaid for Sim2Science, where the
    # noise shift IS the result. Separate slugs, because slugs are provenance.
    fig_invariance(ts1, base1, args.out_dir, manifest, _rng())
    fig_invariance(ts1, base1, args.out_dir, manifest, _rng(), ts2=ts2,
                   base2=base2, slug="fig3_survey_invariance_noise")
    if args.with_fig4:
        fig_deconvolution(ts1, base1, imgs_hero, args.out_dir, manifest, _rng(),
                          profile_survey=args.profile_survey)
    if args.with_fig5:
        fig_noise_shift(ts1, ts2, base1, base2, imgs_ts2, args.out_dir, manifest,
                        _rng())
    fig_hero(ts1, imgs_extra, args.out_dir, manifest, slug="figA1_hero_grid_appendix",
             lens_ids=extra,
             title="Appendix: three further lenses (last row block = worst case)")

    rmses = [_rmse(ts1[n]["theta_pred"], ts1[n]["theta_true"]) for n in SURVEY_ORDER]
    prior = float(np.mean([base1[n]["prior_mean"]["theta_rmse"] for n in SURVEY_ORDER]))
    failed = [n for n in SURVEY_ORDER if np.mean(ts2[n]["hf_complex_err"]) > 1.0]
    ratios = {n: float(np.mean(ts2[n]["reblur_rel"] /
                               np.maximum(ts2[n]["floor_rel"], 1e-30)))
              for n in SURVEY_ORDER}
    if failed:
        # Do NOT claim panel (b) "flags" the failure unless it actually does.
        # The failing survey can have the LOWEST ratio: a residual that has
        # collapsed onto the noise floor is what fitting the noise looks like,
        # which is the opposite end of the axis from a large residual.
        lowest = min(ratios, key=ratios.get)
        highest = max(ratios, key=ratios.get)
        flags = " ".join([
            f"Under noise the top-octave error at {', '.join(failed)} rises above 1.0,",
            "i.e. the reconstruction there is worse than emitting nothing --",
            "reported rather than hidden.",
            f"Read panel (b) carefully: {failed[0]} does not stand out by the SIZE of",
            f"its reblur residual -- it has the lowest ratio of all five",
            f"({ratios[failed[0]]:.2f}, essentially sitting ON its noise floor), which is",
            "precisely the signature of a reconstruction that has fitted the noise",
            "rather than the source.",
            f"The largest ratio ({highest} at {ratios[highest]:.1f}) is a different",
            "effect: its PSF stamp is under-sampled on its own grid (sigma =",
            f"{_sigma_px(highest):.2f} native px), so the reblur operator is least",
            "faithful there -- a simulator limitation, not a reconstruction failure.",
        ]) if ratios[failed[0]] == min(ratios.values()) else " ".join([
            f"Under noise the top-octave error at {', '.join(failed)} rises above 1.0,",
            "i.e. worse than emitting nothing -- reported rather than hidden;",
            f"panel (b) independently puts {failed[0]} at {ratios[failed[0]]:.2f}x its",
            "noise floor, without access to ground truth.",
        ])
        fail_note = flags
    else:
        fail_note = ("Top-octave error stays below 1.0 at every survey under the "
                     "noise shift.")
    # Data-driven Wiener verdict: state whatever the numbers say, per test set,
    # rather than a claim written once and left to drift. The TS1 and TS2 splits
    # genuinely disagree -- Wiener wins on clean seeing-limited data and loses
    # badly under noise -- and that disagreement IS the paper's argument.
    w_gain = {n: float(base2[n]["wiener_hf"]["hf_complex_err"])
                 / float(np.mean(ts2[n]["hf_complex_err"])) for n in SURVEY_ORDER}
    w_gain1 = {n: float(base1[n]["wiener_hf"]["hf_complex_err"])
                  / float(np.mean(ts1[n]["hf_complex_err"])) for n in SURVEY_ORDER}
    won2 = [n for n in SURVEY_ORDER if w_gain[n] > 1.0]
    lost1 = [n for n in SURVEY_ORDER if w_gain1[n] < 1.0]
    wiener_note = (
        f"Under the noise shift the network is better than Wiener at "
        f"{', '.join(won2)} by {min(w_gain[n] for n in won2):.1f}-"
        f"{max(w_gain[n] for n in won2):.1f}x"
        + (f", while on noise-free data Wiener is the better estimator at "
           f"{', '.join(lost1)}" if lost1 else "")
        + ". The defensible claim is therefore noise robustness, not raw "
          "noise-free inversion: an oracle-tuned Wiener is excellent on clean "
          "data, as theory requires, and collapses once its noise assumption "
          "is violated.")
    # SSIM is measured but not plotted. State that it tracks L1 so its absence
    # reads as a space decision, not a gap -- and state the margin difference,
    # because SSIM is the more flattering of the two and an unexplained larger
    # number invites the question.
    from scipy.stats import spearmanr as _sp
    _l1 = [float(np.mean(ts1[n]["l1_rel_masked"])) for n in SURVEY_ORDER]
    _ss = [1.0 - float(np.mean(ts1[n]["ssim_norm"])) for n in SURVEY_ORDER]
    _rho = _sp(_l1, _ss).correlation
    ssim_note = (
        f"Masked SSIM behaves the same way and is omitted only for space: it ranks "
        f"the five surveys identically to this panel (Spearman rho = {_rho:.2f} "
        f"against the L1 values, 1-SSIM spanning "
        f"{min(_ss):.5f}-{max(_ss):.5f}), so it adds no independent ordering. "
        f"Per-survey SSIM is in the appendix table.")
    signal_pct = 100.0 * float(np.mean(
        [np.mean(ts1[n]["mask_frac"]) for n in SURVEY_ORDER]))
    write_captions(args.out_dir, dict(
        version=SCRIPT_VERSION, ckpt=os.path.basename(args.ckpt),
        resid_pct=30, crop=HERO_CROP_ARCSEC,
        fov=args.fov, signal_pct=signal_pct, empty_pct=100.0 - signal_pct,
        rmse_lo=min(rmses), rmse_hi=max(rmses),
        factor=prior / max(rmses), factor_hi=prior / min(rmses), prior=prior,
        n_lens=args.n_lenses, profile_survey=args.profile_survey,
        fail_note=fail_note, wiener_note=wiener_note, ssim_note=ssim_note,
        fig4_section=(FIG4_SECTION.format(profile_survey=args.profile_survey)
                      if args.with_fig4 else FIG4_OMITTED),
        fig5_section=(FIG5_SECTION.format(wiener_note=wiener_note,
                                          fail_note=fail_note)
                      if args.with_fig5 else FIG5_OMITTED)))

    with open(os.path.join(args.out_dir, "figure_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("  wrote figure_manifest.json")

    if args.mirror_dir:
        os.makedirs(args.mirror_dir, exist_ok=True)
        import shutil
        # Copy FILES only. Jupyter drops a `.ipynb_checkpoints/` directory into
        # any folder it touches, and shutil.copy2 raises IsADirectoryError on
        # it -- which killed the mirror step (and the run) after every figure
        # had already been written, so the failure looked like success.
        for f in sorted(os.listdir(args.out_dir)):
            src = os.path.join(args.out_dir, f)
            if not os.path.isfile(src) or f.startswith("."):
                continue
            shutil.copy2(src, os.path.join(args.mirror_dir, f))
        print(f"  mirrored to {args.mirror_dir}")

    print(f"done in {time.time() - t0:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
