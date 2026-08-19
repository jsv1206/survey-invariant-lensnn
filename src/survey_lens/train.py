"""Train / run ablations for the survey-invariant lens network.

Examples
--------
# full model, joint loss, with PSF cross-attention (the proposed method)
python train.py --name full --epochs 30

# Ablation (1): no PSF cross-attention
python train.py --name no_cross --no-cross-attention --epochs 30

# Ablation (2): single-task baselines
python train.py --name recon_only --loss-mode recon_only --epochs 30
python train.py --name reg_only   --loss-mode reg_only   --epochs 30

The three experiments named in the project spec map to flags:
  (1) PSF cross-attention on/off    -> --no-cross-attention
  (2) joint vs single-task          -> --loss-mode {joint,recon_only,reg_only}
  (3) native-scale vs resampled     -> see note in README (data-side switch)

All metrics are written to CSV under runs/<name>/ via Lightning's CSVLogger.
"""

import argparse
import os

from . import env_setup  # noqa: F401  (must precede `lightning`: it transitively
                          # imports torchvision, which is ABI-broken in this
                          # sandbox -- see env_setup.py's _stub_broken_torchvision)
import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from .data import LensDataModule
from .lit_model import LitLensNet
from . import surveys as SV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--log-root", default="runs",
                    help="CSVLogger save_dir; checkpoints land under "
                         "<log-root>/<name>/version_N/checkpoints/")
    ap.add_argument("--version", type=int, default=0,
                    help="logger version dir. Pinned (not auto-incrementing) so a "
                         "resumed run appends to the same metrics.csv.")
    ap.add_argument("--resume", default=None,
                    help="path to a .ckpt to resume from, or 'auto' to pick up "
                         "<log-root>/<name>/version_<version>/checkpoints/last.ckpt "
                         "if it exists. Restores optimizer + LR schedule + epoch, so "
                         "pass the SAME --epochs as the original run.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fov", type=float, default=20.0)
    ap.add_argument("--fine-scale", type=float, default=None,
                    help="global fine render grid in arcsec (LEGACY). Default "
                         "None = adaptive per-sample supersampling, which "
                         "renders the requested pixel scale exactly.")
    ap.add_argument("--cache-dir", default=None,
                    help="directory written by cache_datasets.py (HANDOFF_run1.md "
                         "Step 5), containing train/ and val/ subdirs. If given, "
                         "loads the frozen split instead of re-simulating in memory.")
    # architecture
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--theta-ref", type=float, default=0.45,
                    help="arcsec normalization for RoPE coords; sets finest "
                         "positional wavelength (~2*pi*theta_ref). Must exceed "
                         "2*patch*s_max (=2.56\" for the s=0.32\"/px prior edge) "
                         "or the top band aliases; was 0.3 (=1.885\", aliased).")
    ap.add_argument("--rope-base", type=float, default=100.0,
                    help="RoPE frequency decay base (lower = tighter band on "
                         "in-field scales)")
    ap.add_argument("--no-cross-attention", action="store_true")
    # loss
    ap.add_argument("--loss-mode", default="joint",
                    choices=["joint", "recon_only", "reg_only"])
    ap.add_argument("--lambda-a", type=float, default=0.5,
                    help="SSIM weight inside the reconstruction loss. Exposed so "
                         "CLAUDE.md's documented ablation axis 4 (--lambda-a 0 "
                         "--lambda-b 0) is actually runnable; it previously was not.")
    ap.add_argument("--lambda-b", type=float, default=0.1, help="FFT weight inside recon")
    ap.add_argument("--lambda-c", type=float, default=0.1,
                    help="reblur weight. Was 1.0 against a RAW-MSE reblur term "
                         "that contributed 0.42%% of the loss; the term is now a "
                         "scale-free relative residual, so 0.1 puts it at ~20-25%%.")
    ap.add_argument("--lambda-d", type=float, default=1.0, help="regression weight")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--precision", default="bf16-mixed",
                    choices=["32-true", "bf16-mixed", "16-mixed"],
                    help="bf16-mixed (default) is a 6.9x measured speedup on the "
                         "worst-case (prior-edge, 15,625-token) batch over 32-true "
                         "on this sandbox's A100 -- fp32 attention at long sequence "
                         "length falls back to a much slower non-flash SDPA kernel "
                         "(measured backward pass 3.6x forward, vs the ~2x expected). "
                         "No numerics issue found switching (bf16 has fp32's dynamic "
                         "range, just less mantissa; losses.fft_l1 already forces "
                         "float32 internally since rfft2 doesn't support bf16). Pass "
                         "--precision 32-true to reproduce the old (slow) behavior.")
    # evaluate validation against fixed survey configs (Test Set 1)
    ap.add_argument("--survey-val", action="store_true")
    args = ap.parse_args()

    torch.set_float32_matmul_precision("high")  # enable TF32 on Ampere+ tensor cores
    L.seed_everything(args.seed)

    dm = LensDataModule(
        n_train=args.n_train, n_val=args.n_val, batch_size=args.batch_size,
        fov=args.fov, fine_scale=args.fine_scale, seed=args.seed,
        survey_configs=SV.SURVEYS if args.survey_val else None,
        cache_dir=args.cache_dir)

    lit = LitLensNet(
        patch=args.patch, dim=args.dim, depth=args.depth, n_heads=args.heads,
        theta_ref=args.theta_ref, rope_base=args.rope_base,
        use_cross_attention=not args.no_cross_attention,
        loss_mode=args.loss_mode, lambda_a=args.lambda_a, lambda_b=args.lambda_b,
        lambda_c=args.lambda_c, lambda_d=args.lambda_d,
        lr=args.lr, max_epochs=args.epochs)

    # `--resume` restores model AND optimizer AND LR-scheduler AND epoch
    # counter, so the cosine schedule continues correctly rather than
    # restarting. Pass the SAME --epochs as the original run or the cosine
    # T_max changes and the schedule is silently different.
    resume = args.resume
    if resume == "auto":
        cand = os.path.join(args.log_root, args.name, f"version_{args.version}",
                            "checkpoints", "last.ckpt")
        resume = cand if os.path.exists(cand) else None
        print(f"--resume auto: {'resuming from ' + resume if resume else 'no last.ckpt, starting fresh'}")

    # Pin the logger version so a resumed run writes into the SAME directory
    # instead of silently starting version_1 and splitting the training curve
    # across two files.
    #
    # BUT: Lightning's CSVLogger DELETES an existing metrics.csv in that
    # directory ("Previous log files in this directory will be deleted when the
    # new ones are saved!") and then writes only the epochs from THIS process.
    # Resuming would therefore destroy the pre-interruption training curve --
    # verified by actually resuming a run and watching metrics.csv disappear.
    # Snapshot it first; stitch the parts together when analysing.
    _mcsv = os.path.join(args.log_root, args.name, f"version_{args.version}", "metrics.csv")
    if resume and os.path.exists(_mcsv):
        import shutil, time as _t
        _bak = _mcsv.replace(".csv", f".pre_resume_{_t.strftime('%Y%m%dT%H%M%S')}.csv")
        shutil.copy2(_mcsv, _bak)
        print(f"backed up pre-resume training curve -> {_bak}")
    logger = CSVLogger(args.log_root, name=args.name, version=args.version)
    # monitor val/l1 rather than val/loss, keep 3 + always keep last.
    # Run 1 used monitor="val/loss", save_top_k=1, no save_last. val/loss is
    # dominated by the noisiest channel (SSIM: val log-residual std 0.571 vs
    # 0.116 for l1), so it selected a noise excursion -- and epoch 57, which
    # had the lowest val/l1 of the whole run, was overwritten and is
    # permanently unrecoverable. Do not repeat that.
    ckpt = ModelCheckpoint(monitor="val/l1", mode="min", save_top_k=3, save_last=True)
    trainer = L.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator, devices=1,
        precision=args.precision,
        logger=logger, callbacks=[ckpt], log_every_n_steps=10)
    trainer.fit(lit, dm, ckpt_path=resume)
    print("best checkpoint:", ckpt.best_model_path)
    print("metrics CSV:", logger.log_dir)


if __name__ == "__main__":
    main()
