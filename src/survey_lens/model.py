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
    positions: [N, 2] physical coords (x,y) in normalized arcsec units.
    dim: per-head dimension (must be divisible by 4).
    Returns cos, sin each [N, dim], split half for x and half for y.
    """
    assert dim % 4 == 0, "head dim must be divisible by 4 for 2D RoPE"
    d_half = dim // 2                      # half the dims for x, half for y
    # frequencies for each axis
    n_freq = d_half // 2
    idx = torch.arange(n_freq, dtype=torch.float32, device=positions.device)
    inv_freq = base ** (-idx / n_freq)     # [n_freq]
    x = positions[:, 0:1]                  # [N,1]
    y = positions[:, 1:2]
    ang_x = x * inv_freq[None, :]          # [N, n_freq]
    ang_y = y * inv_freq[None, :]
    # interleave-free layout: [x_freqs, y_freqs] each duplicated for (even,odd) pair
    cos = torch.cat([ang_x.cos(), ang_x.cos(), ang_y.cos(), ang_y.cos()], dim=-1)
    sin = torch.cat([ang_x.sin(), ang_x.sin(), ang_y.sin(), ang_y.sin()], dim=-1)
    return cos, sin


def apply_rope(t, cos, sin):
    """
    t: [B, n_heads, N, head_dim]; cos/sin: [N, head_dim].
    Rotates pairs (2k, 2k+1).
    """
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
                 theta_ref=0.3, rope_base=100.0,
                 use_cross_attention=True, mlp_ratio=4.0):
        super().__init__()
        self.patch = patch
        self.dim = dim
        # theta_ref (arcsec) sets the finest RoPE wavelength (~2*pi*theta_ref);
        # rope_base sets how fast the frequency bank decays. Defaults tuned so
        # the informative bands land on sub-arcsec--to--few-arcsec structure
        # (arcs, Einstein ring theta_E~0.5-2.5"), not the >field-width scales
        # the original (theta_ref=1.0, base=1e4) wasted 13/16 bands on.
        # See tests/test_rope_coverage.py for the coverage assertion.
        self.theta_ref = theta_ref
        self.rope_base = rope_base
        self.use_cross = use_cross_attention
        self.patch_dim = patch * patch
        self.patch_proj = nn.Linear(self.patch_dim, dim)   # shared img/psf projection
        self.type_emb = nn.Embedding(2, dim)               # 0=image, 1=psf
        self.blocks = nn.ModuleList([
            TriAttentionBlock(dim, n_heads, mlp_ratio, use_cross_attention)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        # reconstruction decoder: token -> patch pixels (same scale)
        self.recon_head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, self.patch_dim))
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
        """Physical centers of patches in normalized arcsec, origin = image center."""
        p = self.patch
        # patch center index in pixels (centered): (k+0.5)*p - H/2
        H = gh * p
        W = gw * p
        ii = (torch.arange(gh, device=device) + 0.5) * p - H / 2.0
        jj = (torch.arange(gw, device=device) + 0.5) * p - W / 2.0
        yy, xx = torch.meshgrid(ii, jj, indexing='ij')
        x_arc = xx.reshape(-1) * s / self.theta_ref
        y_arc = yy.reshape(-1) * s / self.theta_ref
        return torch.stack([x_arc, y_arc], dim=-1)     # [nP,2]

    def forward(self, g, h, s):
        """
        g,h: [B,1,H,W] (same H,W per batch element; batch must share grid).
        s:   [B] pixel scale.
        Returns dict with recon [B,1,H,W] and theta_E [B].
        """
        B = g.shape[0]
        device = g.device
        H0, W0 = g.shape[-2:]
        # pad both image and PSF to a whole number of patches (reflect padding
        # keeps the lens-centered coordinate origin fixed: we pad on the high
        # side only, so patch (0,0) still starts at the original corner).
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

        # positions (shared across batch only if s shared; we assume per-batch equal s)
        s_val = float(s[0].item())
        pos = self._patch_positions(gh, gw, s_val, device)         # [nP,2]
        hd = self.dim // self.blocks[0].img_self.n_heads
        cos, sin = build_2d_rope(pos, hd, base=self.rope_base)
        cos_i = sin_i = None
        cos_i, sin_i = cos, sin
        cos_p, sin_p = cos, sin   # identical pipeline for PSF (shared frame)

        img, psf = img_tok, psf_tok
        for blk in self.blocks:
            img, psf = blk(img, psf, cos_i, sin_i, cos_p, sin_p)
        img = self.norm(img)

        # reconstruction
        recon_patches = self.recon_head(img)                       # [B,nP,p*p]
        recon = self._unpatchify(recon_patches, gh, gw)
        recon = recon[..., :H0, :W0]                               # crop to input grid

        # regression: mean-pool image tokens + log(s)
        pooled = img.mean(dim=1)                                   # [B,dim]
        logs = torch.log(s).view(B, 1)
        theta_E = self.reg_head(torch.cat([pooled, logs], dim=-1)).squeeze(-1)
        return dict(recon=recon, theta_E=theta_E, img_tokens=img, psf_tokens=psf)

    def _unpatchify(self, patches, gh, gw):
        B = patches.shape[0]
        p = self.patch
        x = patches.view(B, gh, gw, p, p)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(B, 1, gh * p, gw * p)
        return x
