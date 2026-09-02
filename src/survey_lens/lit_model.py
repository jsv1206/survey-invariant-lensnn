"""LightningModule wrapping SurveyInvariantLensNet + the three-term loss.

Designed so a physicist runs experiments by flipping constructor flags:

    LitLensNet(use_cross_attention=False)        # ablation (1): no PSF cross-attn
    LitLensNet(loss_mode="recon_only")           # ablation (2): L1 only
    LitLensNet(loss_mode="reg_only")             # ablation (2): L2 only
    LitLensNet(loss_mode="joint")                # full L1 + L2 (+ reblur)
    LitLensNet(lambda_c=0.0)                      # turn off reblur consistency

All loss weights (lambda_a,b,c,d) and model size (dim, depth, patch) are
constructor args; nothing is hard-coded in the training loop.
"""
from . import env_setup  # noqa: F401  (must precede lightning -- see data.py)
import torch
import lightning as L

from .model import SurveyInvariantLensNet
from . import losses as Lo


class LitLensNet(L.LightningModule):
    def __init__(self,
                 # --- architecture ---
                 patch=4, dim=128, depth=4, n_heads=4,
                 theta_ref=0.3, rope_base=100.0,
                 use_cross_attention=True,
                 # --- loss configuration ---
                 loss_mode="joint",          # "joint" | "recon_only" | "reg_only"
                 lambda_a=0.5,               # SSIM weight inside recon
                 lambda_b=0.1,               # FFT weight inside recon
                 # lambda_c 1.0 -> 0.1 (Aug 2026), together with making
                 # `losses.reblur_loss` scale-free. Measured budget in Run 1 was
                 # SSIM 92.3% / L1 3.5% / FFT 1.1% / reblur 0.42% / reg 2.7%:
                 # the paper's headline self-supervised term had no gradient at
                 # all, and its weight drifted with survey brightness. As a
                 # relative residual it is ~170x larger, so lambda_c=1.0 would
                 # swing it to ~72% of the post-warmup budget -- which would be
                 # actively harmful: the project notes notes the reblur term is "nearly
                 # blind in the blur operator's null space", i.e. it is
                 # satisfied by ANY amount of over-smoothing, and
                 # under-deconvolution is already the measured failure mode.
                 # 0.1 puts it at ~20-25% of the reconstruction budget: a real
                 # data-consistency constraint, not a smoothing incentive.
                 lambda_c=0.1,               # reblur consistency weight (scale-free)
                 lambda_d=1.0,               # regression weight
                 # --- optimization ---
                 lr=3e-4, weight_decay=1e-4, max_epochs=60):
        super().__init__()
        self.save_hyperparameters()
        self.net = SurveyInvariantLensNet(
            patch=patch, dim=dim, depth=depth, n_heads=n_heads,
            theta_ref=theta_ref, rope_base=rope_base,
            use_cross_attention=use_cross_attention)

    def forward(self, g, h, s):
        return self.net(g, h, s)

    # -- shared step computing all loss terms; logging prefix distinguishes split
    def on_train_epoch_start(self):
        """Advance the training set's noise stream once per epoch.

        Nothing in the project called `set_epoch` before -- `SizeBucketSampler`
        defines one but it was never invoked, so batch ORDER is fixed across
        epochs. That is pre-existing behaviour and is deliberately left alone:
        changing it would alter training dynamics between noise-free and noisy runs for
        a reason unrelated to noise, and confound the comparison.

        Only the TRAIN set advances. Validation noise stays on epoch 0 on
        purpose, so `val/loss` is measured against a fixed realization and moves
        because the model changed, not because the noise did.
        """
        dm = getattr(self.trainer, "datamodule", None)
        ds = getattr(dm, "train_ds", None) if dm is not None else None
        if ds is not None and hasattr(ds, "set_epoch"):
            ds.set_epoch(self.current_epoch)

    def _step(self, batch, prefix):
        g, h, f_gt = batch["g"], batch["h"], batch["f_gt"]
        s, theta_true = batch["s"], batch["theta_E"]
        out = self(g, h, s)
        f_hat, theta_hat = out["recon"], out["theta_E"]

        mode = self.hparams.loss_mode
        total = g.new_zeros(())
        logs = {}

        if mode in ("joint", "recon_only"):
            rec, parts = Lo.reconstruction_loss(
                f_hat, f_gt, self.hparams.lambda_a, self.hparams.lambda_b)
            total = total + rec
            logs.update({f"{prefix}/recon": rec.detach(),
                         f"{prefix}/l1": parts["l1"],
                         f"{prefix}/ssim_term": parts["ssim_term"],
                         f"{prefix}/fft": parts["fft"]})
            if self.hparams.lambda_c > 0:
                # `var` is present only when the dataset applied train-time
                # noise. On noisy `g` the scale-free reblur has an irreducible
                # floor (6-102x its noise-free converged value) and its gradient
                # pushes f_hat*h toward f_gt*h + noise -- it would train the
                # model to reproduce the noise realization. The variance-weighted
                # form is a reduced chi^2: expectation 1 at the truth, so going
                # below 1 is visibly overfitting instead of silently rewarded.
                var = batch.get("var")
                if var is not None:
                    rb = Lo.reblur_loss_chi2(f_hat, h, g, var)
                else:
                    rb = Lo.reblur_loss(f_hat, h, g)
                total = total + self.hparams.lambda_c * rb
                logs[f"{prefix}/reblur"] = rb.detach()

        if mode in ("joint", "reg_only"):
            reg = Lo.regression_loss(theta_hat, theta_true)
            total = total + self.hparams.lambda_d * reg
            logs[f"{prefix}/reg_mse"] = reg.detach()

        # always log theta_E error (diagnostic), even if not optimized
        with torch.no_grad():
            logs[f"{prefix}/theta_rmse"] = torch.sqrt(
                torch.mean((theta_hat - theta_true) ** 2))
        logs[f"{prefix}/loss"] = total.detach()

        bs = g.shape[0]
        self.log_dict(logs, prog_bar=(prefix == "val"), batch_size=bs,
                      on_step=False, on_epoch=True)
        return total

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        """AdamW + cosine decay to ~0.

        Run 1 used a bare constant lr=3e-4 for 60 epochs with no schedule, and
        the consequence was measurable: val/loss over epochs 50-59 had a
        coefficient of variation of 43% (a 4.3x spread), and the mean
        epoch-to-epoch swing EXCEEDED the block-to-block improvement -- so the
        saved "best" checkpoint (epoch 54) was a downward noise excursion
        rather than a better model. Decaying the lr to zero makes "best" and
        "last" converge and makes checkpoint selection mean something.
        """
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, int(self.hparams.max_epochs)), eta_min=self.hparams.lr * 1e-2)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
