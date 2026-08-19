"""
Survey-invariant strong-lens embedding network.

Inputs (per sample):
    g    : blurred image, [1, H, W]   (native pixel scale s)
    h    : PSF stamp,     [1, H, W]   (same scale, same FOV, lens-centered)
    s    : pixel scale scalar (arcsec/px)

Tokenization:
    - image and PSF are split into p x p patches -> linear patch_proj -> d-dim tokens
    - each token gets:
        * a 2D RoPE positional encoding using the PHYSICAL patch-center position
          in arcsec-from-lens-center, normalized by theta_ref:
              (i*s/theta_ref, j*s/theta_ref)
        * a learned type embedding (image vs PSF)
    - image and PSF share the SAME coordinate origin (lens center) and scale,
      so identical positional pipeline applies to both.

Attention block (one "tri-attention" layer):
    1. image self-attention   (image tokens attend to image tokens)
    2. PSF self-attention     (PSF tokens attend to PSF tokens)
    3. cross-attention        (image queries, PSF keys/values)
    All use 2D-RoPE on Q,K in the shared arcsec frame, so the image<->PSF score
    is modulated by the relative offset (deconvolution structure).

Heads:
    - reconstruction decoder: image tokens -> sharp image patches (same scale)
    - regression MLP: pooled image embedding -> theta_E

The `use_cross_attention` flag enables the cross-attention ablation.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2D RoPE
# ---------------------------------------------------------------------------
def build_2d_rope(positions, dim, base=10000.0):
    """
    positions: [..., N, 2] physical coords (x,y) in normalized arcsec units.
    Leading batch dims (e.g. [B, N, 2], per-sample positions) are supported
    via broadcasting -- see `apply_rope` and CLAUDE.md's note on per-sample
    pixel scale: a bucket of same-npix samples does NOT guarantee identical
    `s` (different continuous-prior draws can round to the same npix), so
    positions must be computed per-sample, not from one shared scalar.
    dim: per-head dimension (must be divisible by 4).
    Returns cos, sin each [..., N, dim], split half for x and half for y.
    """
    assert dim % 4 == 0, "head dim must be divisible by 4 for 2D RoPE"
    d_half = dim // 2                      # half the dims for x, half for y
    # frequencies for each axis
    n_freq = d_half // 2
    idx = torch.arange(n_freq, dtype=torch.float32, device=positions.device)
    inv_freq = base ** (-idx / n_freq)     # [n_freq]
    x = positions[..., 0:1]                # [...,N,1]
    y = positions[..., 1:2]
    ang_x = x * inv_freq                   # [...,N,n_freq] (broadcasts trailing dim)
    ang_y = y * inv_freq
    # INTERLEAVED layout: slot pair (2k, 2k+1) carries band k, matching how
    # `apply_rope` indexes with [..., 0::2] / [..., 1::2].
    #
    # This previously used `cat([ang_x, ang_x, ang_y, ang_y])` -- a
    # concatenate-duplicate layout -- while `apply_rope` used interleaved-pair
    # indexing. The two conventions are incompatible, and the result was that
    # only the EVEN-indexed bands were ever applied: verified numerically at
    # head_dim=32, slots consumed {x0,x2,x4,x6,y0,y2,y4,y6} and bands
    # {1,3,5,7} of each axis were computed and then silently discarded. Half
    # the positional frequency bank was dead, each surviving band applied
    # twice, leaving only three in-field wavelengths per axis. The rotation
    # itself stayed valid (both channels of a pair shared an angle), which is
    # why nothing downstream ever failed loudly.
    cos = torch.cat([ang_x.cos().repeat_interleave(2, dim=-1),
                     ang_y.cos().repeat_interleave(2, dim=-1)], dim=-1)
    sin = torch.cat([ang_x.sin().repeat_interleave(2, dim=-1),
                     ang_y.sin().repeat_interleave(2, dim=-1)], dim=-1)
    return cos, sin


def apply_rope(t, cos, sin):
    """
    t: [B, n_heads, N, head_dim]; cos/sin: [B, N, head_dim] (per-sample
    positions -- broadcasts over the head dimension). Rotates pairs (2k, 2k+1).
    """
    cos = cos.unsqueeze(1)   # [B,1,N,head_dim] -> broadcasts over n_heads
    sin = sin.unsqueeze(1)
    t1 = t[..., 0::2]
    t2 = t[..., 1::2]
    cos1 = cos[..., 0::2]
    sin1 = sin[..., 0::2]
    rot_even = t1 * cos1 - t2 * sin1
    rot_odd = t1 * sin1 + t2 * cos1
    out = torch.empty_like(t)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


# ---------------------------------------------------------------------------
# Multi-head attention with 2D RoPE on Q,K
# ---------------------------------------------------------------------------
class RopeAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def _split(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x_q, x_kv, cos_q, sin_q, cos_k, sin_k):
        B, Nq, _ = x_q.shape
        q = self._split(self.q(x_q))            # [B,h,Nq,hd]
        k = self._split(self.k(x_kv))
        v = self._split(self.v(x_kv))
        q = apply_rope(q, cos_q, sin_q)
        k = apply_rope(k, cos_k, sin_k)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.dim)
        return self.o(out)


class TriAttentionBlock(nn.Module):
    """One layer: image self-attn, PSF self-attn, image<-PSF cross-attn, + MLP."""
    def __init__(self, dim, n_heads, mlp_ratio=4.0, use_cross_attention=True):
        super().__init__()
        self.use_cross = use_cross_attention
        self.ln_img1 = nn.LayerNorm(dim)
        self.ln_psf1 = nn.LayerNorm(dim)
        self.img_self = RopeAttention(dim, n_heads)
        self.psf_self = RopeAttention(dim, n_heads)
        if use_cross_attention:
            self.ln_img_x = nn.LayerNorm(dim)
            self.ln_psf_x = nn.LayerNorm(dim)
            self.cross = RopeAttention(dim, n_heads)
        self.ln_img2 = nn.LayerNorm(dim)
        self.ln_psf2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp_img = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.mlp_psf = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, img, psf, cos_i, sin_i, cos_p, sin_p):
        # self-attention
        i = self.ln_img1(img)
        img = img + self.img_self(i, i, cos_i, sin_i, cos_i, sin_i)
        p = self.ln_psf1(psf)
        psf = psf + self.psf_self(p, p, cos_p, sin_p, cos_p, sin_p)
        # cross-attention: image queries, PSF keys/values
        if self.use_cross:
            iq = self.ln_img_x(img)
            pk = self.ln_psf_x(psf)
            img = img + self.cross(iq, pk, cos_i, sin_i, cos_p, sin_p)
        # MLP
        img = img + self.mlp_img(self.ln_img2(img))
        psf = psf + self.mlp_psf(self.ln_psf2(psf))
        return img, psf


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class SurveyInvariantLensNet(nn.Module):
    def __init__(self, patch=4, dim=128, depth=4, n_heads=4,
                 theta_ref=0.45, rope_base=100.0,
                 use_cross_attention=True, mlp_ratio=4.0):
        super().__init__()
        self.patch = patch
        self.dim = dim
        # theta_ref (arcsec) sets the finest RoPE wavelength (~2*pi*theta_ref);
        # rope_base sets how fast the frequency bank decays. Defaults tuned so
        # the informative bands land on sub-arcsec--to--few-arcsec structure
        # (arcs, Einstein ring theta_E~0.5-2.5"), not the >field-width scales
        # the original (theta_ref=1.0, base=1e4) wasted 13/16 bands on.
        #
        # theta_ref raised 0.30 -> 0.45 (Aug 2026): at 0.30 the finest
        # wavelength (2*pi*theta_ref = 1.885") was BELOW the Nyquist
        # requirement of 2*patch*s for DES (2.104") and for the training
        # prior's upper edge s=0.32"/px (2.560"), so the top band aliased --
        # scale-dependent positional degradation, which is exactly what a
        # survey-invariant model cannot afford. 0.45 gives 2.827", clear of
        # both, and is affordable only because the layout bug in
        # `build_2d_rope` was fixed in the same change: with all 8 bands per
        # axis live instead of 4, in-field bands go 3 -> 4 despite the coarser
        # finest wavelength. Strictly better on both axes.
        # See tests/test_rope_coverage.py for both assertions.
        self.theta_ref = theta_ref
        self.rope_base = rope_base
        self.use_cross = use_cross_attention
        self.patch_dim = patch * patch
        self.patch_proj = nn.Linear(self.patch_dim, dim)   # shared img/psf projection
        self.type_emb = nn.Embedding(2, dim)               # 0=image, 1=psf
        # Default nn.Embedding init is N(0,1), which SWAMPED the projected image
        # content: measured signal/embedding token-energy ratio was 0.0057 at HST
        # (i.e. the token was 99.4% constant) vs 0.065 at DES -- an 11x
        # survey-dependent input SNR, which is precisely what a survey-invariant
        # model must not have. Standard transformer embedding scale instead.
        nn.init.normal_(self.type_emb.weight, std=0.02)
        self.blocks = nn.ModuleList([
            TriAttentionBlock(dim, n_heads, mlp_ratio, use_cross_attention)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        # Reconstruction decoder: token -> patch pixels. Takes the raw patch
        # pixels alongside the normalized token, because `self.norm` (LayerNorm)
        # divides out per-token amplitude -- every post-norm token has norm
        # exactly sqrt(dim) -- while this head has to emit absolutely-scaled
        # flux. That mismatch showed up as peak ratios falling monotonically
        # with blur: 0.97 / 0.74 / 0.56 / 0.29 / 0.27 across the surveys.
        self.recon_head = nn.Sequential(
            nn.Linear(dim + self.patch_dim, dim), nn.GELU(),
            nn.Linear(dim, self.patch_dim))
        # Cross-seam refinement. Each token decodes its own disjoint p x p tile
        # with no overlap, so patch boundaries are discontinuous: measured excess
        # boundary-step energy 1.29-1.59x vs ground truth, and for patch=4 that
        # lands at 0.25 cyc/px == EXACTLY 0.5x Nyquist, i.e. inside the top
        # octave that adjudicates the deconvolution claim (48.9% of HST's
        # top-octave power was on those lines vs 21.1% for truth). 3x3 convs see
        # across the seams.
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 3, padding=1))
        # ZERO-INIT both output layers => delta == 0 at init => recon == g
        # exactly. The model therefore STARTS at the `identity` baseline and can
        # only improve on it, instead of having to rediscover it from scratch
        # (Run 1 never did: overfitting two images, the old decoder still landed
        # 2.5x worse than doing nothing).
        nn.init.zeros_(self.recon_head[-1].weight); nn.init.zeros_(self.recon_head[-1].bias)
        nn.init.zeros_(self.refine[-1].weight); nn.init.zeros_(self.refine[-1].bias)
        # regression head: pooled image embedding + log(s) -> theta_E
        self.reg_head = nn.Sequential(
            nn.Linear(dim + 1, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, 1))

    def _patchify(self, x):
        """x: [B,1,H,W] -> patches [B, nP, p*p], grid (gh, gw)."""
        B, _, H, W = x.shape
        p = self.patch
        gh, gw = H // p, W // p
        x = x[:, :, :gh * p, :gw * p]
        x = x.unfold(2, p, p).unfold(3, p, p)          # [B,1,gh,gw,p,p]
        x = x.contiguous().view(B, gh * gw, p * p)
        return x, gh, gw

    def _patch_positions(self, gh, gw, s, device):
        """Physical centers of patches in normalized arcsec, origin = image
        center, PER SAMPLE. `s`: [B] pixel scale -- a batch is bucketed by
        matching npix, which does NOT guarantee identical `s` (different
        continuous-prior draws can round to the same npix; measured up to
        ~1.6% relative spread within a bucket on the real training cache),
        so positions must vary per sample rather than use one shared scalar.
        """
        p = self.patch
        # patch center index in pixels (centered): (k+0.5)*p - H/2
        H = gh * p
        W = gw * p
        ii = (torch.arange(gh, device=device) + 0.5) * p - H / 2.0
        jj = (torch.arange(gw, device=device) + 0.5) * p - W / 2.0
        yy, xx = torch.meshgrid(ii, jj, indexing='ij')
        xx = xx.reshape(-1)                             # [nP]
        yy = yy.reshape(-1)
        s = (s / self.theta_ref).view(-1, 1)             # [B,1]
        x_arc = xx[None, :] * s                          # [B,nP]
        y_arc = yy[None, :] * s
        return torch.stack([x_arc, y_arc], dim=-1)       # [B,nP,2]

    def forward(self, g, h, s):
        """
        g,h: [B,1,H,W] (same H,W per batch element; batch must share grid).
        s:   [B] pixel scale.
        Returns dict with recon [B,1,H,W] and theta_E [B].
        """
        B = g.shape[0]
        device = g.device
        H0, W0 = g.shape[-2:]
        g_in = g                                   # ORIGINAL, unpadded, unscaled

        # --- per-sample input conditioning => exact scale equivariance ---
        # Raw surface brightness is ~0.017 at HST and ~0.19 at DES, and it went
        # straight into patch_proj, leaving image content at 0.6% of token
        # energy at HST vs 6.5% at DES. Normalizing by a statistic of `g` ALONE
        # (never f_gt: that would leak ground truth, and `g` is what exists on
        # real data) and undoing it on the output makes the network exactly
        # scale-equivariant: g -> c*g gives recon -> c*recon, by construction.
        gscale = g.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        g = g / gscale
        # PSF token stream only. `losses.reblur` still convolves with the
        # untouched, sum-normalized `h` from the batch, so the "PSF stamps must
        # stay sum-normalized to 1" convention is preserved where it matters.
        h = h / h.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)

        # pad both image and PSF to a whole number of patches (high side only)
        p = self.patch
        pad_h = (p - H0 % p) % p
        pad_w = (p - W0 % p) % p
        if pad_h or pad_w:
            g = F.pad(g, (0, pad_w, 0, pad_h), mode="reflect")
            h = F.pad(h, (0, pad_w, 0, pad_h), mode="reflect")
        # NOTE: batch shares one grid (we batch by identical native size)
        img_p, gh, gw = self._patchify(g)
        psf_p, _, _ = self._patchify(h)
        img_tok = self.patch_proj(img_p) + self.type_emb(torch.zeros(1, dtype=torch.long, device=device))
        psf_tok = self.patch_proj(psf_p) + self.type_emb(torch.ones(1, dtype=torch.long, device=device))

        # positions computed PER SAMPLE (bucketing by npix does not guarantee
        # identical s -- see _patch_positions); also avoids a host sync from
        # .item() on every forward call.
        pos = self._patch_positions(gh, gw, s, device)              # [B,nP,2]
        hd = self.dim // self.blocks[0].img_self.n_heads
        cos, sin = build_2d_rope(pos, hd, base=self.rope_base)
        cos_i, sin_i = cos, sin
        cos_p, sin_p = cos, sin   # identical pipeline for PSF (shared frame)

        img, psf = img_tok, psf_tok
        for blk in self.blocks:
            img, psf = blk(img, psf, cos_i, sin_i, cos_p, sin_p)
        img_n = self.norm(img)

        # --- reconstruction: a RESIDUAL on the input, not a from-scratch synthesis ---
        # `identity` (return g) is a strong baseline precisely because the PSF
        # attenuates high frequencies without destroying their phase, so the
        # information is still in `g`. The old head had no path to express it and
        # lost to identity in every survey. Predicting a correction to `g` makes
        # identity the starting point rather than a target to rediscover.
        # img_p is concatenated because self.norm() divides out the per-token
        # amplitude the head needs to emit absolute flux.
        delta = self._unpatchify(self.recon_head(torch.cat([img_n, img_p], dim=-1)),
                                 gh, gw)[..., :H0, :W0]
        delta = delta + self.refine(delta)                         # cross patch seams
        # undo the input scaling (equivariance) and impose f_gt >= 0. Both output
        # layers are zero-init, so delta == 0 and recon == g_in EXACTLY at init.
        recon = (g_in + gscale * delta).clamp_min(0.0)

        # regression: mean-pool image tokens + log(s)
        pooled = img_n.mean(dim=1)                                 # [B,dim]
        logs = torch.log(s).view(B, 1)
        theta_E = self.reg_head(torch.cat([pooled, logs], dim=-1)).squeeze(-1)
        return dict(recon=recon, theta_E=theta_E, img_tokens=img, psf_tokens=psf)

    def _unpatchify(self, patches, gh, gw):
        B = patches.shape[0]
        p = self.patch
        x = patches.view(B, gh, gw, p, p)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(B, 1, gh * p, gw * p)
        return x
