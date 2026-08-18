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
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from . import env_setup  # noqa: F401
from .data import LensDataModule
from .lit_model import LitLensNet
from . import surveys as SV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--fov", type=float, default=20.0)
    ap.add_argument("--fine-scale", type=float, default=None,
                    help="global fine render grid in arcsec (LEGACY). Default "
                         "None = adaptive per-sample supersampling, which "
                         "renders the requested pixel scale exactly.")
    # architecture
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--theta-ref", type=float, default=0.3,
                    help="arcsec normalization for RoPE coords; sets finest "
                         "positional wavelength (~2*pi*theta_ref)")
    ap.add_argument("--rope-base", type=float, default=100.0,
                    help="RoPE frequency decay base (lower = tighter band on "
                         "in-field scales)")
    ap.add_argument("--no-cross-attention", action="store_true")
    # loss
    ap.add_argument("--loss-mode", default="joint",
                    choices=["joint", "recon_only", "reg_only"])
    ap.add_argument("--lambda-c", type=float, default=1.0, help="reblur weight")
    ap.add_argument("--lambda-d", type=float, default=1.0, help="regression weight")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accelerator", default="auto")
    # evaluate validation against fixed survey configs (Test Set 1)
    ap.add_argument("--survey-val", action="store_true")
    args = ap.parse_args()

    L.seed_everything(args.seed)

    dm = LensDataModule(
        n_train=args.n_train, n_val=args.n_val, batch_size=args.batch_size,
        fov=args.fov, fine_scale=args.fine_scale, seed=args.seed,
        survey_configs=SV.SURVEYS if args.survey_val else None)

    lit = LitLensNet(
        patch=args.patch, dim=args.dim, depth=args.depth, n_heads=args.heads,
        theta_ref=args.theta_ref, rope_base=args.rope_base,
        use_cross_attention=not args.no_cross_attention,
        loss_mode=args.loss_mode, lambda_c=args.lambda_c, lambda_d=args.lambda_d,
        lr=args.lr)

    logger = CSVLogger("runs", name=args.name)
    ckpt = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1)
    trainer = L.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator, devices=1,
        logger=logger, callbacks=[ckpt], log_every_n_steps=10)
    trainer.fit(lit, dm)
    print("best checkpoint:", ckpt.best_model_path)
    print("metrics CSV:", logger.log_dir)


if __name__ == "__main__":
    main()
