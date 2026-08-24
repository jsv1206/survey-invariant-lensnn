"""Paper figures for the Sim2Science submission (HANDOFF_run1.md Step 10).

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
SCRIPT_VERSION = "1.2.1"

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
        "font.size": FS_BASE, "axes.titlesize": FS_BASE, "axes.labelsize": FS_SMALL,
        "xtick.labelsize": FS_TINY, "ytick.labelsize": FS_TINY,
        "legend.fontsize": FS_TINY, "figure.titlesize": FS_BASE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
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
                           "reblur_rel", "floor_rel",
                           "l1_norm_masked_identity", "hf_complex_err_identity",
                           "l1_norm_masked_zeros", "hf_complex_err_zeros",
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
    return dict(g=g[0, 0].cpu().numpy(), f_hat=f_hat[0, 0].cpu().numpy(),
                f_gt=f[0, 0].cpu().numpy(),
                reblur_resid=(reblur - g)[0, 0].cpu().numpy(),
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
             lens_ids=None, title=None):
    """Rows = surveys ordered by pixel scale, cols = input / recon / truth /
    residual. Every panel of a given row is divided by that row's f_gt peak,
    so the four panels share vmin/vmax and the rows are comparable despite an
    ~11x spread in raw per-pixel surface brightness across pixel scales."""
    plt = _mpl()
    lens_ids = lens_ids or [None]
    n_lens = len(imgs)
    nrow = len(SURVEY_ORDER) * n_lens
    fig, axes = plt.subplots(nrow, 4, figsize=(TEXT_W, 1.12 * nrow + 0.35),
                             squeeze=False)
    rows = []
    cols = [r"input $g$", r"reconstruction $\hat f$", r"truth $f_{gt}$",
            r"residual $\hat f - f_{gt}$"]
    RESID_LIM = 0.30                      # +/- 30% of the truth's peak
    for li, per_survey in enumerate(imgs):
        for si, nm in enumerate(SURVEY_ORDER):
            d = per_survey[nm]
            row = li * len(SURVEY_ORDER) + si
            s = d["s"]
            pk = max(float(np.abs(d["f_gt"]).max()), 1e-12)
            panels = [d["g"] / pk, d["f_hat"] / pk, d["f_gt"] / pk,
                      (d["f_hat"] - d["f_gt"]) / pk]
            npix = d["f_gt"].shape[-1]
            for ci, img in enumerate(panels):
                ax = axes[row][ci]
                c = _crop(img, s)
                if ci == 3:
                    ax.imshow(c, cmap=CMAP_RESID, vmin=-RESID_LIM, vmax=RESID_LIM,
                              interpolation="nearest")
                else:
                    ax.imshow(c, cmap=CMAP_IMG, vmin=0.0, vmax=1.0,
                              interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if row == 0:
                    ax.set_title(cols[ci], fontsize=FS_TINY, pad=3)
            # row label: the varying npix IS the result, so make it visible
            axes[row][0].set_ylabel(
                f"{nm}\n{s:.3f}\"/px\nFWHM {SV.SURVEYS[nm]['fwhm']:.2f}\"\n"
                f"{npix}x{npix} px",
                fontsize=FS_TINY, rotation=0, ha="right", va="center",
                labelpad=18, color=SURVEY_COLORS[nm])
            # 1" scale bar: physical size is constant while the pixel grid is not
            ax = axes[row][0]
            npx_crop = _crop(panels[0], s).shape[-1]
            bar_px = 1.0 / s
            ax.plot([0.06 * npx_crop, 0.06 * npx_crop + bar_px],
                    [0.92 * npx_crop, 0.92 * npx_crop],
                    color="white", lw=1.6, solid_capstyle="butt")
            ax.text(0.06 * npx_crop, 0.87 * npx_crop, "1\"", color="white",
                    fontsize=FS_TINY, va="bottom")
            rows.append(dict(lens_id=lens_ids[li], survey=nm, scale_arcsec_px=s,
                             fwhm_arcsec=SV.SURVEYS[nm]["fwhm"], npix_native=npix,
                             theta_E_true=d["theta_E"],
                             f_gt_peak=pk,
                             crop_half_arcsec=HERO_CROP_ARCSEC,
                             resid_limit_frac_of_peak=RESID_LIM))
    if title:
        fig.suptitle(title, fontsize=FS_SMALL, y=1.0)
    # a visible gap between adjacent image fills: without it the three dark
    # panels of a row read as one wide band rather than three images
    fig.tight_layout(h_pad=0.45, w_pad=0.9)
    save_figure(fig, out_dir, slug, pd.DataFrame(rows), manifest)


# --------------------------------------------------------------------------
# Fig 3 -- quantitative survey invariance
# --------------------------------------------------------------------------
def fig_invariance(ts1, base1, out_dir, manifest, rng):
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.15))
    rows = []
    prior_rmse = float(np.mean([base1[nm]["prior_mean"]["theta_rmse"]
                                for nm in SURVEY_ORDER]))

    # (a) predicted vs true
    ax = axes[0]
    lo, hi = 0.35, 2.65
    ax.fill_between([lo, hi], [lo - prior_rmse, hi - prior_rmse],
                    [lo + prior_rmse, hi + prior_rmse],
                    color=INK_MUTED, alpha=0.13, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.9, color=INK_2, zorder=1)
    for nm in SURVEY_ORDER:
        r = ts1[nm]
        ax.scatter(r["theta_true"], r["theta_pred"], s=4, alpha=0.55, lw=0,
                   marker=SURVEY_MARKERS[nm], color=SURVEY_COLORS[nm],
                   label=f"{nm} {_rmse(r['theta_pred'], r['theta_true']):.3f}\"",
                   zorder=3)
    ax.set_xlabel(r"true $\theta_E$ (arcsec)")
    ax.set_ylabel(r"predicted $\theta_E$ (arcsec)")
    ax.set_title("(a) one network, 5 surveys", fontsize=FS_SMALL)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.legend(title="RMSE", loc="upper left", frameon=False, handletextpad=0.2,
              borderpad=0.1, labelspacing=0.18, title_fontsize=FS_TINY)
    ax.text(0.97, 0.06, "shaded: prior-mean\nRMSE band", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=FS_TINY, color=INK_MUTED)

    # (b) theta RMSE vs pixel scale
    ax = axes[1]
    for nm in SURVEY_ORDER:
        r = ts1[nm]
        pt, blo, bhi = boot_ci_paired(r["theta_pred"], r["theta_true"], _rmse, rng)
        s = SV.SURVEYS[nm]["scale"]
        ax.errorbar(s, pt, yerr=[[pt - blo], [bhi - pt]], fmt=SURVEY_MARKERS[nm],
                    ms=4, color=SURVEY_COLORS[nm], capsize=2, lw=1.0, zorder=3)
        dy = 7 if nm not in ("Roman", "DES") else -13
        ax.annotate(nm, (s, pt), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=FS_TINY, color=SURVEY_COLORS[nm])
        rows.append(dict(panel="b", survey=nm, scale_arcsec_px=s,
                         theta_rmse=pt, ci_lo=blo, ci_hi=bhi,
                         prior_mean_rmse=base1[nm]["prior_mean"]["theta_rmse"]))
    ax.axhline(prior_rmse, ls="--", lw=1.0, color=INK_MUTED, zorder=1)
    ax.text(0.055, prior_rmse * 0.90, f"prior-mean baseline ({prior_rmse:.2f}\")",
            fontsize=FS_TINY, color=INK_MUTED, va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("pixel scale (arcsec/px)")
    ax.set_ylabel(r"$\theta_E$ RMSE (arcsec)")
    ax.set_title(r"(b) $\theta_E$ vs pixel scale", fontsize=FS_SMALL)
    ax.set_ylim(0.008, 1.3)

    # (c) masked normalized recon L1 vs pixel scale, with the null models
    ax = axes[2]
    for nm in SURVEY_ORDER:
        r = ts1[nm]
        s = SV.SURVEYS[nm]["scale"]
        pt, blo, bhi = boot_ci(r["l1_norm_masked"], np.mean, rng)
        ax.errorbar(s, pt, yerr=[[pt - blo], [bhi - pt]], fmt=SURVEY_MARKERS[nm],
                    ms=4, color=SURVEY_COLORS[nm], capsize=2, lw=1.0, zorder=3)
        rows.append(dict(panel="c", survey=nm, scale_arcsec_px=s,
                         l1_norm_masked=pt, ci_lo=blo, ci_hi=bhi,
                         identity=float(np.mean(r["l1_norm_masked_identity"])),
                         zeros=float(np.mean(r["l1_norm_masked_zeros"]))))
    xs = [SV.SURVEYS[n]["scale"] for n in SURVEY_ORDER]
    for key, lab, ls in (("l1_norm_masked_identity", "identity ($\\hat f = g$)", "--"),
                         ("l1_norm_masked_zeros", "zeros ($\\hat f = 0$)", ":")):
        ax.plot(xs, [float(np.mean(ts1[n][key])) for n in SURVEY_ORDER],
                ls=ls, lw=1.0, color=INK_MUTED, marker="", zorder=1, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("pixel scale (arcsec/px)")
    ax.set_ylabel("masked normalized L1")
    ax.set_title("(c) recon vs pixel scale", fontsize=FS_SMALL)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4,
              borderpad=0.1, labelspacing=0.2)
    _no_minor_labels(axes[1], axes[2])

    for nm in SURVEY_ORDER:
        r = ts1[nm]
        pt, blo, bhi = boot_ci_paired(r["theta_pred"], r["theta_true"], _rmse, rng)
        rows.append(dict(panel="a", survey=nm, scale_arcsec_px=SV.SURVEYS[nm]["scale"],
                         theta_rmse=pt, ci_lo=blo, ci_hi=bhi,
                         prior_mean_rmse=base1[nm]["prior_mean"]["theta_rmse"]))
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, out_dir, "fig3_survey_invariance", pd.DataFrame(rows), manifest)


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

    # (b) top-octave complex error vs PSF width in NATIVE pixels
    ax = fig.add_subplot(gs[0, 2:])
    ax.axhline(1.0, color=INK_2, ls="-", lw=0.9, zorder=1)
    ax.text(0.02, 1.03, "worse than emitting nothing", transform=ax.get_yaxis_transform(),
            fontsize=FS_TINY, color=INK_2, va="bottom")
    xs, ident, wien, ylo, yhi, label_pts = [], [], [], [], [], []
    for nm in SURVEY_ORDER:
        r = ts1[nm]
        x = _sigma_px(nm)
        pt, blo, bhi = boot_ci(r["hf_complex_err"], np.mean, rng)
        ax.errorbar(x, pt, yerr=[[pt - blo], [bhi - pt]], fmt=SURVEY_MARKERS[nm],
                    ms=4.5, color=SURVEY_COLORS[nm], capsize=2, lw=1.0, zorder=4)
        label_pts.append((x, pt, nm))
        xs.append(x); ylo.append(blo); yhi.append(bhi)
        ident.append(float(np.mean(r["hf_complex_err_identity"])))
        wien.append(float(base1[nm]["wiener_hf"]["hf_complex_err"]))
        rows.append(dict(panel="b", survey=nm, sigma_px=x,
                         hf_complex_err=pt, ci_lo=blo, ci_hi=bhi,
                         identity=ident[-1], wiener=wien[-1],
                         wiener_K=base1[nm]["wiener_hf"]["wiener_K"],
                         wiener_K_at_grid_edge=base1[nm]["wiener_hf"]["wiener_K_at_grid_edge"],
                         hf_coherence=float(np.mean(r["hf_coherence"]))))
    o = np.argsort(xs)
    xs_a = np.array(xs)[o]
    ax.plot(xs_a, np.array(ident)[o], ls="--", lw=1.0, color=INK_MUTED,
            label="identity (no deconvolution)")
    ax.plot(xs_a, np.array(wien)[o], ls=":", lw=1.0, color=INK_2,
            label="Wiener (known PSF, best $K$)")
    ax.set_yscale("log")
    ax.set_xlabel(r"PSF $\sigma$ in native pixels")
    ax.set_ylabel("top-octave complex error")
    # Labels always go ABOVE their point, alternating between two heights in
    # x-order. Two rules, both load-bearing: alternating heights keeps
    # neighbours in x from colliding (Rubin and DES sit 0.04 px apart in PSF
    # sigma), and keeping every label above leaves the band beneath the lowest
    # datum clear for the legend, which no series can wander into.
    for rank, (px, py, nm) in enumerate(sorted(label_pts)):
        dy = 8 if rank % 2 == 0 else 20
        ax.annotate(nm, (px, py), textcoords="offset points", xytext=(0, dy),
                    ha="center", va="bottom",
                    fontsize=FS_TINY, color=SURVEY_COLORS[nm])
    ax.set_xlim(min(xs) - 0.22, max(xs) + 0.26)
    # open a clear band under the lowest datum so the legend has somewhere to
    # sit that no series can wander into
    ax.set_ylim(min(ylo + wien) * 0.16, max(yhi + ident + [1.0]) * 1.9)
    ax.set_title("(b) deconvolution vs PSF sampling", fontsize=FS_SMALL)
    ax.legend(loc="lower left", frameon=False, handletextpad=0.4, borderpad=0.1,
              labelspacing=0.2)
    _no_minor_labels(ax)

    # (c) reblur residual per survey -- the diagnostic that needs no truth
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
               f"(c) reblur residual $\\hat f * h - g$, no ground truth needed "
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
    ax.set_ylabel("top-octave complex error")
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

(a) Predicted vs true Einstein radius for all five surveys on the **same**
paired lens population; the shaded band is the prior-mean null model's RMSE.
(b) theta_E RMSE against pixel scale on log axes: {rmse_lo:.3f}-{rmse_hi:.3f}
arcsec across a 5.3x range of pixel scale, {factor:.0f}-{factor_hi:.0f}x below
the prior-mean baseline of {prior:.3f} arcsec. (c) Masked normalized L1
against pixel scale, with the `identity` (no deconvolution) and `zeros` null
models as reference lines. Error bars are the 16-84th percentile over 1000
bootstrap resamples of the {n_lens} test lenses.

## Fig 4 -- The network is deconvolving, not smoothing

(a) Azimuthally averaged power spectra at {profile_survey}: the input has lost
power through the top octave (shaded, 0.5-1.0x native Nyquist) and the
reconstruction puts it back. (b) Top-octave **complex** error -- phase-aware,
and normalized so that emitting nothing scores exactly 1.0 -- against PSF
sigma in native pixels, with the `identity` baseline and classical Wiener
deconvolution with the *known* PSF. Wiener's regularization `K` is swept and
chosen **on this same top-octave metric**, on the evaluation set itself -- so
the classical baseline is shown at its best on exactly the number it is being
judged by, which makes this a deliberately generous upper bound on it. (An
earlier version selected `K` by masked L1 and then scored it here, which
understated Wiener.) Caveat that belongs with this panel: the network learned
the simulator's exact generative distribution, so beating a generic-prior
filter on data drawn from that prior is easier than it would be on real sky --
Wiener's weaker assumptions transfer more honestly. (c) The reblur residual `f_hat * h - g` for one
lens per survey: small and structureless is good, and because it needs no
ground truth it is the one diagnostic that transfers unchanged to real data.

## Fig 5 -- Sim-to-real: a noise-free simulator meeting noisy data

The model is trained on **noise-free** simulations and evaluated here, with no
retraining and no noise augmentation, on the same lenses rendered with each
survey's own noise. (a) theta_E RMSE degrades by the annotated factor but
stays far below the prior-mean null at every survey. (b) The reblur residual
divided by its own analytic noise floor `mean(max(g,0)/gain + bkg_rms^2)`,
both in the scale-free units of the reblur loss: 1.0 is the best any
reconstruction can do on noisy input, and *below* 1.0 would mean the model is
fitting noise. (c) Reblur residual maps on noisy input. (d) Top-octave complex
error before and after the noise shift. The black tick on each bar is Wiener
deconvolution with the *known* PSF at its best `K`, re-swept **separately on
each test set** -- Wiener's noise-free optimum sits up to 13 decades away from
its noisy one, so carrying one `K` across both would understate the baseline.
{wiener_note} **{fail_note}**

## Ablation status (axis 2 IS run -- this block used to claim otherwise)

- **Loss mode (axis 2) -- RUN.** Three arms (`joint` / `recon_only` / `reg_only`)
  on the same frozen 12" dataset, 80-epoch cosine, seed 0; only `--loss-mode`
  differs. Drawn by `make_ablation_figures.py` as `fig6_loss_ablation`, with its
  own `captions_ablation.md`. Main text for the representations version of the
  paper, appendix for Sim2Science (see `plan_sim2science.md`).
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
    ap.add_argument("--no-strict-layout", action="store_true",
                    help="warn instead of failing on the geometry self-check")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.boot_seed)

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
    fig_invariance(ts1, base1, args.out_dir, manifest, rng)
    fig_deconvolution(ts1, base1, imgs_hero, args.out_dir, manifest, rng,
                      profile_survey=args.profile_survey)
    fig_noise_shift(ts1, ts2, base1, base2, imgs_ts2, args.out_dir, manifest, rng)
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
    signal_pct = 100.0 * float(np.mean(
        [np.mean(ts1[n]["mask_frac"]) for n in SURVEY_ORDER]))
    write_captions(args.out_dir, dict(
        version=SCRIPT_VERSION, ckpt=os.path.basename(args.ckpt),
        resid_pct=30, crop=HERO_CROP_ARCSEC,
        fov=args.fov, signal_pct=signal_pct, empty_pct=100.0 - signal_pct,
        rmse_lo=min(rmses), rmse_hi=max(rmses),
        factor=prior / max(rmses), factor_hi=prior / min(rmses), prior=prior,
        n_lens=args.n_lenses, profile_survey=args.profile_survey,
        fail_note=fail_note, wiener_note=wiener_note))

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
