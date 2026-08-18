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
                 lambda_c=1.0,               # reblur consistency weight
                 lambda_d=1.0,               # regression weight
                 # --- optimization ---
                 lr=3e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.net = SurveyInvariantLensNet(
            patch=patch, dim=dim, depth=depth, n_heads=n_heads,
            theta_ref=theta_ref, rope_base=rope_base,
            use_cross_attention=use_cross_attention)

    def forward(self, g, h, s):
        return self.net(g, h, s)

    # -- shared step computing all loss terms; logging prefix distinguishes split
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
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                 weight_decay=self.hparams.weight_decay)
