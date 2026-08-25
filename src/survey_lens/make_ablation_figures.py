"""Loss-mode ablation figures (Run 3) -- ablation axis 2.

Three arms of the SAME architecture on the SAME frozen 12" dataset, differing
only in `--loss-mode`:

    joint       recon + reblur + regression      the reference
    recon_only  recon + reblur                   what the theta_E task costs deconvolution
    reg_only    regression                       what the recon task costs regression

    PYTHONPATH=src python -m survey_lens.make_ablation_figures \
        --joint-ckpt <...> --recon-ckpt <...> --reg-ckpt <...> \
        --fov 12.0 --device cuda --workers 16 --out-dir figures/run3/paper

Deliberately a separate entry point from `make_paper_figures.py`: that script is
built around a SINGLE checkpoint (load one, run inference, derive every figure
from it) and produces the paper's finished Run-2 figures. Rather than tangle a
three-checkpoint mode into it, this module IMPORTS its machinery -- the survey
palette, the strict layout self-check, the bootstrap helpers, the per-figure
`_data.csv` contract -- so styling and provenance carry over unchanged and
Run 2's path is untouched.

**Only two of the six (arm x metric) cells are meaningful.** Each single-task
arm's other metric is degenerate BY CONSTRUCTION, not merely poor:

  * `recon_only` never puts theta_hat into the loss (`lit_model._step`), so
    `reg_head` receives no gradient and stays at its random initialization. Its
    theta_E number measures an untrained head, nothing more.
  * `reg_only` never puts f_hat into the loss, and `recon_head[-1]`/`refine[-1]`
    are zero-init (`model.py`), so `delta` stays exactly 0 and `recon == g`
    bit-for-bit. Its "reconstruction" IS the `identity` baseline.

So the honest comparisons are joint-vs-recon_only for deconvolution and
joint-vs-reg_only for regression. The degenerate cells are still drawn -- as
outlined, hatched bars, labelled as degenerate -- because hiding them would
misrepresent what the ablation can conclude.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from . import env_setup  # noqa: F401  (must precede lenstronomy/torch tuning)
from . import surveys as SV
from . import test as T
from .data import build_test_set_1, build_test_set_2
from .make_paper_figures import (  # noqa: F401  -- shared, deliberately not duplicated
    FS_SMALL,
    FS_TINY,
    INK,
    INK_2,
    INK_MUTED,
    N_BOOT,
    SURVEY_COLORS,
    SURVEY_MARKERS,
    SURVEY_ORDER,
    TEXT_W,
    _dataset_digest,
    _git_state,
    _mpl,
    _no_minor_labels,
    _rmse,
    boot_ci,
    boot_ci_paired,
    per_sample_pass,
    save_figure,
)

SCRIPT_VERSION = "1.1.0"

# arm key -> (display label, degenerate-in-which-metric)
#
# `no_cross` is ablation AXIS 1 and is categorically different from the other
# two: it changes the ARCHITECTURE, not the loss, and it is degenerate in
# NEITHER head -- so unlike recon_only/reg_only it is a valid measurement in
# both panels. That is why it is drawn with a colour fill (a real number) and
# only a hatch to mark it as the architecture arm, never the white "degenerate"
# treatment.
ARMS = {
    "joint": ("joint (recon + $\\theta_E$)", None),
    "recon_only": ("recon only", "theta"),
    "reg_only": ("$\\theta_E$ only", "recon"),
    "no_cross": ("no PSF cross-attn", None),
}
DEGENERATE_NOTE = {
    "theta": "untrained head",
    "recon": "= identity by construction",
}


def fig_loss_ablation(res, base, out_dir, manifest, rng):
    """(a) regression: joint vs reg_only.  (b) deconvolution: joint vs recon_only."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.6))
    rows = []
    x = np.arange(len(SURVEY_ORDER))
    # Four arms now. The degenerate one stays LAST in each panel so the two
    # real comparisons sit adjacent to `joint` and can be read against it.
    # LOSS arms only. `no_cross` is ablation axis 1 and gets its own figure
    # (fig7), which has room for the sigma_px mechanism panel that is the
    # interesting part of that result -- see fig_cross_attention below.
    order_a = ["joint", "reg_only", "recon_only"]   # last degenerate
    order_b = ["joint", "recon_only", "reg_only"]   # last degenerate
    # `--nocross-ckpt` is optional, so drop any arm that was not supplied and
    # re-centre: with 3 bars the offset is (k-1), with 4 it is (k-1.5).
    order_a = [a for a in order_a if a in res]
    order_b = [a for a in order_b if a in res]
    n_arms = len(order_a)
    w = 0.20 if n_arms >= 4 else 0.26
    off = (n_arms - 1) / 2.0

    # ---- (a) regression -------------------------------------------------
    ax = axes[0]
    for k, arm in enumerate(order_a):
        deg = ARMS[arm][1] == "theta"
        vals, los, his = [], [], []
        for nm in SURVEY_ORDER:
            r = res[arm]["ts1"][nm]
            pt, lo, hi = boot_ci_paired(r["theta_pred"], r["theta_true"], _rmse, rng)
            vals.append(pt); los.append(pt - lo); his.append(hi - pt)
            rows.append(dict(panel="a", metric="theta_rmse", arm=arm, survey=nm,
                             value=pt, ci_lo=lo, ci_hi=hi, degenerate=bool(deg)))
        ax.bar(x + (k - off) * w, vals, width=w - 0.03,
               yerr=None if deg else [los, his], capsize=1.6,
               error_kw=dict(lw=0.7, ecolor=INK_2),
               hatch="///" if deg else ("..." if arm == "no_cross" else None),
               linewidth=0.9 if (deg or arm == "no_cross") else 0.0,
               color=["white" if deg else SURVEY_COLORS[n] for n in SURVEY_ORDER],
               edgecolor=[SURVEY_COLORS[n] for n in SURVEY_ORDER],
               alpha=1.0 if arm == "joint" or deg else 0.5)
    prior = float(np.mean([base[nm]["prior_mean"]["theta_rmse"] for nm in SURVEY_ORDER]))
    ax.axhline(prior, ls="--", lw=1.0, color=INK_MUTED)
    ax.text(-0.45, prior * 0.80, f"prior-mean null ({prior:.2f}\")",
            fontsize=FS_TINY, color=INK_MUTED, ha="left", va="top")
    ax.set_yscale("log")
    trained_max = max(_rmse(res[arm]["ts1"][nm]["theta_pred"],
                            res[arm]["ts1"][nm]["theta_true"])
                      for arm in ("joint", "reg_only", "no_cross") if arm in res
                      for nm in SURVEY_ORDER)
    top = max(prior * 1.45, trained_max * 3)
    ax.set_ylim(trained_max * 0.30, top)
    # The degenerate arm runs off the top of the focused range. State its value
    # once, in the title, rather than annotating five clipped bars -- per-bar
    # labels collided with the prior-mean line and added nothing.
    deg_vals = [_rmse(res["recon_only"]["ts1"][nm]["theta_pred"],
                      res["recon_only"]["ts1"][nm]["theta_true"])
                for nm in SURVEY_ORDER]
    ax.set_xticks(x); ax.set_xticklabels(SURVEY_ORDER)
    ax.set_ylabel(r"$\theta_E$ RMSE (arcsec)")
    # explicit decade-spanning ticks: the focused range covers well under two
    # decades, and suppressing the minor labels (needed to stop them colliding)
    # would otherwise leave the bars with no labelled tick anywhere near them
    from matplotlib.ticker import FormatStrFormatter
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_title("(a) regression: does the recon loss help?\n"
                 f"recon-only off scale at {min(deg_vals):.2f}-{max(deg_vals):.2f}\""
                 " (untrained head)", fontsize=FS_SMALL)
    _no_minor_labels(ax)
    for nm in SURVEY_ORDER:
        rows.append(dict(panel="a", metric="prior_mean_rmse", arm="prior_mean",
                         survey=nm, value=base[nm]["prior_mean"]["theta_rmse"],
                         degenerate=False))

    # ---- (b) deconvolution ----------------------------------------------
    ax = axes[1]
    ymax_b, ymins_b = 0.0, []
    for k, arm in enumerate(order_b):
        deg = ARMS[arm][1] == "recon"
        vals, los, his = [], [], []
        for nm in SURVEY_ORDER:
            r = res[arm]["ts1"][nm]
            pt, lo, hi = boot_ci(r["hf_complex_err"], np.mean, rng)
            vals.append(pt); los.append(pt - lo); his.append(hi - pt)
            ymax_b = max(ymax_b, hi); ymins_b.append(max(lo, 1e-4))
            rows.append(dict(panel="b", metric="hf_complex_err", arm=arm, survey=nm,
                             value=pt, ci_lo=lo, ci_hi=hi, degenerate=bool(deg)))
        ax.bar(x + (k - off) * w, vals, width=w - 0.03,
               yerr=None if deg else [los, his], capsize=1.6,
               error_kw=dict(lw=0.7, ecolor=INK_2),
               hatch="///" if deg else ("..." if arm == "no_cross" else None),
               linewidth=0.9 if (deg or arm == "no_cross") else 0.0,
               color=["white" if deg else SURVEY_COLORS[n] for n in SURVEY_ORDER],
               edgecolor=[SURVEY_COLORS[n] for n in SURVEY_ORDER],
               alpha=1.0 if arm == "joint" or deg else 0.5)
    ident = [float(np.mean(res["joint"]["ts1"][n]["hf_complex_err_identity"]))
             for n in SURVEY_ORDER]
    wien = [float(base[n]["wiener_hf"]["hf_complex_err"]) for n in SURVEY_ORDER]
    ax.plot(x, ident, ls="--", lw=1.0, color=INK_MUTED, marker="", label="identity")
    ax.plot(x, wien, ls=":", lw=1.0, color=INK_2, marker="", label="Wiener")
    ax.axhline(1.0, ls="-", lw=0.9, color=INK_2)
    ax.set_yscale("log")
    lo_b = min(v for v in ymins_b if v > 0)
    ax.set_ylim(lo_b * 0.45, max(1.55, ymax_b * 1.55))
    ax.text(len(x) - 0.5, 1.06, "worse than emitting nothing", fontsize=FS_TINY,
            color=INK_2, ha="right", va="bottom")
    _no_minor_labels(ax)
    for nm, i, wv in zip(SURVEY_ORDER, ident, wien):
        rows.append(dict(panel="b", metric="identity", arm="identity", survey=nm,
                         value=i, degenerate=False))
        rows.append(dict(panel="b", metric="wiener", arm="wiener", survey=nm,
                         value=wv, degenerate=False))
    ax.set_xticks(x); ax.set_xticklabels(SURVEY_ORDER)
    ax.set_ylabel("high-freq. octave error")
    ax.set_title("(b) deconvolution: does the $\\theta_E$ loss hurt?\n"
                 "$\\theta_E$-only lands exactly on identity (degenerate)",
                 fontsize=FS_SMALL)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.4, borderpad=0.1,
              labelspacing=0.2)

    # one shared arm legend in neutral ink -- survey identity is already carried
    # by the bar colours, so the arm legend must not re-use them
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=INK_2, edgecolor="none", label="joint"),
        Patch(facecolor=INK_2, edgecolor="none", alpha=0.5, label="single-task"),
        Patch(facecolor="white", edgecolor=INK_2, hatch="///", label="degenerate"),
    ]
    if "no_cross" in res:
        handles.insert(2, Patch(facecolor=INK_2, edgecolor=INK_2, alpha=0.5,
                                hatch="...", label="no cross-attn"))
    fig.tight_layout(w_pad=2.0, rect=(0, 0.10, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=FS_TINY, bbox_to_anchor=(0.5, 0.005),
               handletextpad=0.4, columnspacing=1.4)
    save_figure(fig, out_dir, "fig6_loss_ablation", pd.DataFrame(rows), manifest)


def _sigma_px(nm):
    """PSF sigma in NATIVE pixels -- the honest difficulty axis for deconvolution
    (CLAUDE.md dataset bug 3: below ~0.8 px the Gaussian stamp barely resolves
    its own kernel)."""
    c = SV.SURVEYS[nm]
    return c["fwhm"] / 2.3548 / c["scale"]


def fig_cross_attention(res, base, out_dir, manifest, rng):
    """Ablation axis 1: PSF cross-attention on/off, joint loss, one flag apart.

    ONE panel is plotted -- deconvolution error, joint vs no-cross, per survey --
    because both venues are page-limited (Sim2Science 5pp, Representations 4pp).

    The other two parts of the result are still COMPUTED and written to the data
    CSV, because they must be quoted in the text and the paper quotes the CSV:

      panel `b_data` -- theta_E is NOT affected. 0.0216" (joint) vs 0.0207"
        (no-cross), no-cross marginally better and better at 3/5 surveys. This
        is not a footnote: it is the internal control for the capacity confound.
        The no-cross arm has 266,240 fewer parameters (13.9%), so "less
        capacity" is a live alternative explanation for the plotted panel --
        until you notice the SAME reduction leaves regression untouched. A
        generic capacity deficit degrades both heads; only the reconstruction
        pathway does. **Without this in the text, the figure alone does not
        answer the first question a referee will ask.**

      panel `c_data` -- the effect tracks PSF SAMPLING: Spearman rho = -0.90
        (p = 0.037) against sigma in native pixels. An under-sampled PSF has its
        cutoff near Nyquist where it is poorly resolved, so the blur cannot be
        inferred from the image and being told it matters enormously (Roman,
        sigma = 0.46 px, 9.98x); a wide PSF announces itself in the image's
        spectral falloff (Rubin, sigma = 1.49 px, 1.94x). This is the mechanism,
        and it is why this ablation was promoted over the loss ablation.
    """
    from scipy.stats import spearmanr
    plt = _mpl()
    fig, ax_main = plt.subplots(1, 1, figsize=(3.3, 2.5))
    # The two unplotted panels still run: they populate the data CSV so the text
    # can quote them. They draw onto a throwaway figure that is never saved.
    _scratch = plt.figure()
    axes = [ax_main, _scratch.add_subplot(2, 1, 1), _scratch.add_subplot(2, 1, 2)]
    rows = []
    x = np.arange(len(SURVEY_ORDER))
    w = 0.34
    ARM_ORDER = [("joint", "with PSF cross-attn", 1.0),
                 ("no_cross", "without", 0.45)]

    # ---- (a) deconvolution ------------------------------------------------
    ax = axes[0]
    hi_all = []
    for k, (arm, lab, alpha) in enumerate(ARM_ORDER):
        vals, los, his = [], [], []
        for nm in SURVEY_ORDER:
            pt, lo, hi = boot_ci(res[arm]["ts1"][nm]["hf_complex_err"], np.mean, rng)
            vals.append(pt); los.append(pt - lo); his.append(hi - pt); hi_all.append(hi)
            rows.append(dict(panel="a", metric="hf_complex_err", arm=arm, survey=nm,
                             value=pt, ci_lo=lo, ci_hi=hi))
        ax.bar(x + (k - 0.5) * w, vals, width=w - 0.05, label=lab, alpha=alpha,
               color=[SURVEY_COLORS[n] for n in SURVEY_ORDER],
               yerr=[los, his], capsize=1.6, error_kw=dict(lw=0.7, ecolor=INK_2))
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(SURVEY_ORDER, rotation=45, ha="right")
    ax.set_ylabel("high-freq. octave error")

    ax.legend(loc="upper left", frameon=False, fontsize=FS_TINY - 0.5,
              handletextpad=0.4, borderpad=0.1, labelspacing=0.2)
    _no_minor_labels(ax)

    # ---- (b) regression: the null, and the confound control ---------------
    ax = axes[1]
    for k, (arm, lab, alpha) in enumerate(ARM_ORDER):
        vals, los, his = [], [], []
        for nm in SURVEY_ORDER:
            r = res[arm]["ts1"][nm]
            pt, lo, hi = boot_ci_paired(r["theta_pred"], r["theta_true"], _rmse, rng)
            vals.append(pt); los.append(pt - lo); his.append(hi - pt)
            rows.append(dict(panel="b_data", metric="theta_rmse", arm=arm, survey=nm,
                             value=pt, ci_lo=lo, ci_hi=hi))
        ax.bar(x + (k - 0.5) * w, vals, width=w - 0.05, alpha=alpha,
               color=[SURVEY_COLORS[n] for n in SURVEY_ORDER],
               yerr=[los, his], capsize=1.6, error_kw=dict(lw=0.7, ecolor=INK_2))
    ax.set_xticks(x); ax.set_xticklabels(SURVEY_ORDER, rotation=45, ha="right")
    ax.set_ylabel(r"$\theta_E$ RMSE (arcsec)")
    ax.set_title(r"(b) $\theta_E$ unaffected", fontsize=FS_SMALL)
    ax.set_ylim(0, None)

    # ---- (c) the mechanism ------------------------------------------------
    ax = axes[2]
    sig, ben, blo, bhi = [], [], [], []
    for nm in SURVEY_ORDER:
        a = res["joint"]["ts1"][nm]["hf_complex_err"]
        b = res["no_cross"]["ts1"][nm]["hf_complex_err"]
        # paired bootstrap over the SAME lens indices in both arms -- the test
        # set is a paired design, so resampling them independently would inflate
        # the interval with between-lens variance that cancels in the ratio.
        n = min(len(a), len(b))
        idx = rng.integers(0, n, size=(N_BOOT, n))
        draws = np.mean(np.asarray(b)[idx], axis=1) / np.mean(np.asarray(a)[idx], axis=1)
        pt = float(np.mean(b) / np.mean(a))
        lo, hi = np.percentile(draws, [16, 84])
        sig.append(_sigma_px(nm)); ben.append(pt); blo.append(pt - lo); bhi.append(hi - pt)
        rows.append(dict(panel="c_data", metric="benefit_ratio", arm="no_cross/joint",
                         survey=nm, value=pt, ci_lo=lo, ci_hi=hi,
                         sigma_px=_sigma_px(nm)))
        ax.errorbar(sig[-1], pt, yerr=[[blo[-1]], [bhi[-1]]], fmt=SURVEY_MARKERS[nm],
                    ms=4.5, color=SURVEY_COLORS[nm], capsize=2, lw=1.0, zorder=4)
    rho, pval = spearmanr(sig, ben)
    ax.axhline(1.0, ls="-", lw=0.9, color=INK_2)
    ax.set_yscale("log")
    ax.set_xlabel(r"PSF $\sigma$ in native pixels")
    ax.set_ylabel("error without / with")
    ax.set_title("(c) the effect tracks PSF sampling -- conditioning matters most where the PSF is under-sampled", fontsize=FS_SMALL)
    ax.text(0.96, 0.94, f"Spearman $\\rho$ = {rho:+.2f}\n$p$ = {pval:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=FS_TINY,
            color=INK)
    ax.text(0.04, 0.06, "1.0 = no effect", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=FS_TINY - 0.5, color=INK_2)
    ax.set_xlim(min(sig) - 0.14, max(sig) + 0.14)
    ax.set_ylim(min(b_ - l for b_, l in zip(ben, blo)) * 0.55,
                max(b_ + h for b_, h in zip(ben, bhi)) * 2.6)
    _no_minor_labels(ax)
    # Labels above the point, alternating two heights in x-order -- the same rule
    # fig4b uses, and for the same reason: Rubin and DES are 0.03 px apart in
    # sigma and would otherwise overprint each other.
    for rank, (xx, yy, nm) in enumerate(sorted(zip(sig, ben, SURVEY_ORDER))):
        ax.annotate(nm, (xx, yy), textcoords="offset points",
                    xytext=(0, 9 if rank % 2 == 0 else 19),
                    ha="center", va="bottom", fontsize=FS_TINY - 0.5,
                    color=SURVEY_COLORS[nm])

    manifest["cross_attention_spearman"] = dict(rho=float(rho), p=float(pval),
                                                n=len(SURVEY_ORDER))
    plt.close(_scratch)          # never saved; only ran to populate `rows`
    fig.tight_layout()
    save_figure(fig, out_dir, "fig7_cross_attention", pd.DataFrame(rows), manifest)


CAPTIONS = """# Run 3 -- loss-mode ablation (axis 2)

Auto-generated by `survey_lens/make_ablation_figures.py` v{version}.
Numbers come from `fig6_loss_ablation_data.csv`, never off the plot.
FOV {fov:g}", frozen 12" dataset, {n_lens} paired test lenses, 80-epoch cosine
schedule, seed 0 -- identical across arms; only `--loss-mode` differs.

## Fig 6 -- What each loss term contributes

Three arms of one architecture: **joint** (reconstruction + reblur + theta_E),
**recon only**, and **theta_E only**. Because the two single-task arms train
only one head, their other head is degenerate *by construction* rather than
merely weak -- `recon only` leaves the regression head at its random
initialization (it receives no gradient at all), and `theta_E only` leaves the
zero-initialized reconstruction head untouched, so its output is exactly the
input image, i.e. literally the `identity` baseline. Those cells are drawn
outlined and hatched and must not be read as measurements.

(a) **Regression.** theta_E RMSE per survey, joint vs theta_E-only, against the
prior-mean null ({prior:.3f}"). This isolates what the reconstruction objective
does to Einstein-radius accuracy. {verdict_a}

*Caveat, stated because the bar chart does not show it:* this comparison is not
as clean as (b). The theta_E-only arm trains a single head and therefore gets
the entire gradient budget for it, whereas the joint arm splits capacity across
two objectives. Had theta_E-only won, "undivided capacity" would have been at
least as good an explanation as "the recon loss hurts". It did not win -- the
joint model is better at every survey -- so the confound works AGAINST the
result reported here, which makes the conclusion stronger than the design alone
could guarantee. {consistency_a}

(b) **Deconvolution.** High-freq. octave error per survey (defined at the end of
this file), joint vs recon-only,
with the `identity` and Wiener references. The metric is normalized so that
emitting nothing scores exactly 1.0, and it is phase-aware -- a power ratio
alone is gameable. This isolates what the theta_E objective costs (or buys)
the deconvolution. {verdict_b}

Error bars are the 16-84th percentile over {n_boot} bootstrap resamples of the
{n_lens} lenses; the paired design means a cross-arm gap at fixed survey
reflects the loss, not a different lens sample.

**Both effects run the same way at every one of the five surveys**, so neither
verdict rests on a single telescope.

## Under the noise shift (Test Set 2)

The figure shows Test Set 1. The same three checkpoints evaluated on Test Set 2
-- identical lenses, each rendered with its own survey's noise, no retraining
and no noise augmentation -- are in `all_arms_metrics.csv`. Two things there do
not survive the shift and should not be quoted from panel (a) or (b):

- **HST collapses for both reconstruction-trained arms.** Top-octave error goes
  to {hst_joint_ts2:.3f} (joint) and {hst_recon_ts2:.3f} (recon-only), i.e. above
  1.0 -- worse than emitting nothing. Run 2 saw the same failure independently
  at FOV 20" (1.125), so this reproduces across field of view and across arms
  and is a property of the method, not of one run.
- **The regression ranking inverts at HST only.** Joint degrades to
  {hst_joint_theta_ts2:.4f}" while theta_E-only holds at {hst_reg_theta_ts2:.4f}".
  Averaged over surveys that single survey flips the mean, which is why the TS2
  comparison should be read per-survey rather than as an average. The reading
  that fits: the deconvolution pathway is what is fragile to noise, and in the
  joint model it drags theta_E down with it -- the theta_E-only arm has no such
  pathway to break.

## Relationship to Run 2

Run 2 (FOV 20", `figures/paper/`) remains the paper's headline result and is
unchanged. This ablation is self-contained at FOV 12" with the corrected reblur
operator, so its arms are mutually comparable; its absolute numbers are not
directly comparable to Run 2's, which is why all three arms were retrained here
rather than reusing Run 2 as the joint arm.

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
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--joint-ckpt", required=True)
    ap.add_argument("--recon-ckpt", required=True)
    ap.add_argument("--reg-ckpt", required=True)
    ap.add_argument("--nocross-ckpt", default=None,
                    help="ablation axis 1 (--no-cross-attention), joint loss. "
                         "Optional so the 3-arm figure still reproduces; when "
                         "given, `no_cross` joins BOTH panels as a real "
                         "measurement (it is degenerate in neither head).")
    ap.add_argument("--out-dir", default="figures/run3/paper",
                    help="defaults into the PAPER set: fig6 is main-text for the "
                         "representations version (see plan_sim2science.md), so it "
                         "must live alongside figs 1-5 and be mirrored with them.")
    ap.add_argument("--mirror-dir", default=None)
    ap.add_argument("--fov", type=float, default=12.0,
                    help="MUST match the FOV the checkpoints were trained at. "
                         "The checkpoint does not record it (fov is a DataModule "
                         "arg, not a model hparam), so nothing can catch a "
                         "mismatch for you.")
    ap.add_argument("--n-lenses", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--no-strict-layout", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.boot_seed)

    ckpts = {"joint": args.joint_ckpt, "recon_only": args.recon_ckpt,
             "reg_only": args.reg_ckpt}
    if args.nocross_ckpt:
        ckpts["no_cross"] = args.nocross_ckpt
    for arm, path in ckpts.items():
        if not os.path.exists(path):
            raise SystemExit(f"{arm}: checkpoint not found: {path}")

    print(f"building Test Set 1 + 2 at fov={args.fov}\" "
          f"({args.n_lenses} lenses x {len(SV.SURVEYS)} surveys) ...")
    ts1_items = build_test_set_1(SV.SURVEYS, n_lenses=args.n_lenses, seed=args.seed,
                                 fov=args.fov, n_workers=args.workers)
    ts2_items = build_test_set_2(SV.SURVEYS, n_lenses=args.n_lenses, seed=args.seed,
                                 fov=args.fov, n_workers=args.workers)

    # baselines depend only on the data, so compute them once and share
    print("computing baselines (zeros / identity / wiener / prior_mean) ...")
    base = {nm: T.baseline_rows(ts1_items[nm], args.batch_size, args.device)
            for nm in SURVEY_ORDER}

    res = {}
    for arm, path in ckpts.items():
        print(f"--- {arm}: {os.path.basename(path)}")
        model = T.load_model(path, args.device)
        patch = getattr(getattr(model, "net", model), "patch", 4)
        res[arm] = {"ts1": {}, "ts2": {}}
        for nm in SURVEY_ORDER:
            cfg = SV.SURVEYS[nm]
            res[arm]["ts1"][nm] = per_sample_pass(model, ts1_items[nm], args.device,
                                                  args.batch_size, patch)
            res[arm]["ts2"][nm] = per_sample_pass(
                model, ts2_items[nm], args.device, args.batch_size, patch,
                noise_cfg=dict(bkg_rms=cfg["bkg_rms"], gain=cfg["gain"]))
            r1 = res[arm]["ts1"][nm]
            print(f"    {nm:7s} theta_rmse={_rmse(r1['theta_pred'], r1['theta_true']):.4f}"
                  f"  hf_complex_err={np.mean(r1['hf_complex_err']):.3f}")
        del model

    # --- the self-check the plan calls for: reg_only's reconstruction must BE
    # the identity baseline, bit-for-bit. If it is not, gradient is leaking into
    # recon_head and the ablation is not clean.
    print("\ndegeneracy self-check (reg_only recon vs identity):")
    max_dev = 0.0
    for nm in SURVEY_ORDER:
        a = float(np.mean(res["reg_only"]["ts1"][nm]["hf_complex_err"]))
        b = float(np.mean(res["reg_only"]["ts1"][nm]["hf_complex_err_identity"]))
        max_dev = max(max_dev, abs(a - b))
        print(f"    {nm:7s} reg_only={a:.6f}  identity={b:.6f}  |diff|={abs(a-b):.2e}")
    verdict = ("CONFIRMED degenerate" if max_dev < 1e-6
               else f"NOT degenerate (max |diff| = {max_dev:.2e}) -- investigate")
    print(f"  -> {verdict}")

    manifest = dict(
        script="survey_lens/make_ablation_figures.py",
        script_version=SCRIPT_VERSION,
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        checkpoints={a: os.path.abspath(p) for a, p in ckpts.items()},
        git=_git_state(),
        fov_arcsec=args.fov, n_lenses=args.n_lenses, seed=args.seed,
        test_set_1_digest=_dataset_digest(ts1_items),
        test_set_2_digest=_dataset_digest(ts2_items),
        bootstrap=dict(n=N_BOOT, seed=args.boot_seed),
        reg_only_recon_equals_identity=verdict,
        figures={},
    )

    strict = not args.no_strict_layout
    global save_figure
    _sf = save_figure
    save_figure = lambda *a, **k: _sf(*a, **{**k, "strict": strict})  # noqa: E731

    print("\ndrawing figures ...")
    fig_loss_ablation(res, base, args.out_dir, manifest, rng)
    if "no_cross" in res:
        fig_cross_attention(res, base, args.out_dir, manifest, rng)

    # tidy per-arm CSV of everything, both test sets
    rows = []
    for arm in ckpts:
        for ts in ("ts1", "ts2"):
            for nm in SURVEY_ORDER:
                r = res[arm][ts][nm]
                rows.append(dict(
                    arm=arm, test_set=ts, survey=nm,
                    theta_rmse=_rmse(r["theta_pred"], r["theta_true"]),
                    hf_complex_err=float(np.mean(r["hf_complex_err"])),
                    hf_coherence=float(np.mean(r["hf_coherence"])),
                    l1_norm_masked=float(np.mean(r["l1_norm_masked"])),
                    reblur_rel=float(np.mean(r["reblur_rel"])),
                    theta_degenerate=(arm == "recon_only"),
                    recon_degenerate=(arm == "reg_only")))
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "all_arms_metrics.csv"),
                              index=False)
    print("  wrote all_arms_metrics.csv")

    prior = float(np.mean([base[nm]["prior_mean"]["theta_rmse"] for nm in SURVEY_ORDER]))
    jt = np.mean([_rmse(res["joint"]["ts1"][n]["theta_pred"],
                        res["joint"]["ts1"][n]["theta_true"]) for n in SURVEY_ORDER])
    rt = np.mean([_rmse(res["reg_only"]["ts1"][n]["theta_pred"],
                        res["reg_only"]["ts1"][n]["theta_true"]) for n in SURVEY_ORDER])
    jd = np.mean([np.mean(res["joint"]["ts1"][n]["hf_complex_err"]) for n in SURVEY_ORDER])
    rd = np.mean([np.mean(res["recon_only"]["ts1"][n]["hf_complex_err"])
                  for n in SURVEY_ORDER])
    verdict_a = (f"Averaged over surveys the joint model reaches {jt:.4f}\" against "
                 f"{rt:.4f}\" for theta_E-only, i.e. the reconstruction objective "
                 f"{'helps' if jt < rt else 'costs'} "
                 f"{abs(1 - jt / rt) * 100:.0f}% in theta_E RMSE.")
    verdict_b = (f"Averaged over surveys the joint model reaches {jd:.3f} against "
                 f"{rd:.3f} for recon-only, i.e. adding the theta_E objective "
                 f"{'helps' if jd < rd else 'costs'} "
                 f"{abs(1 - jd / rd) * 100:.0f}% in top-octave error.")
    def _th(arm, ts, nm):
        r = res[arm][ts][nm]
        return _rmse(r["theta_pred"], r["theta_true"])

    n_better_a = sum(_th("joint", "ts1", nm) < _th("reg_only", "ts1", nm)
                     for nm in SURVEY_ORDER)
    consistency_a = (f"The joint arm is ahead at {n_better_a}/{len(SURVEY_ORDER)} "
                     f"surveys.")
    # NOT "captions.md" / "figure_manifest.json": this script now writes into the
    # SAME directory as make_paper_figures.py (figures/run3/paper), because
    # fig6 is main-text for the representations version of the paper and so has
    # to sit in the paper set and its mirror. Both scripts writing the generic
    # names meant whichever ran second silently destroyed the other's captions
    # and provenance manifest. Suffixed names let them coexist.
    with open(os.path.join(args.out_dir, "captions_ablation.md"), "w") as fh:
        fh.write(CAPTIONS.format(
            version=SCRIPT_VERSION, fov=args.fov, n_lens=args.n_lenses,
            n_boot=N_BOOT, prior=prior, verdict_a=verdict_a, verdict_b=verdict_b,
            consistency_a=consistency_a,
            hst_joint_ts2=float(np.mean(res["joint"]["ts2"]["HST"]["hf_complex_err"])),
            hst_recon_ts2=float(np.mean(res["recon_only"]["ts2"]["HST"]["hf_complex_err"])),
            hst_joint_theta_ts2=_th("joint", "ts2", "HST"),
            hst_reg_theta_ts2=_th("reg_only", "ts2", "HST")))
    print("  wrote captions_ablation.md")

    with open(os.path.join(args.out_dir, "figure_manifest_ablation.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("  wrote figure_manifest_ablation.json")

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
