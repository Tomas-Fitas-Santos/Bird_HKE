"""MambaViT Backbone (branch).

Sequential SSM-then-Attention blocks with 2D sine/cosine positional encoding.

This module implements the MambaViT branch. When combined with HRNet
it forms the full HR-MambaViT model (see pose_hr_mamba_vit.py).

Key design principles:
 - Each hybrid block runs Mamba SSM *then* multi-head self-attention
   sequentially, each with its own residual connection.
 - A fixed 2D sine/cosine positional encoding is added to tokens after
   patch embedding, so a single SSM scan direction is sufficient
   (no multi-directional LR/RL/TB/BT redundancy).
 - The backbone outputs a spatial feature map (B, embed_dim, Hp, Wp)
   ready for de-projection and fusion with HRNet.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Mamba import ──────────────────────────────────────────────────────
MambaBlockImpl = None
for _candidate in (
    'mamba_ssm.modules.Mamba',
    'mamba_ssm.models.Mamba',
    'mamba_ssm.Mamba',
):
    try:
        _module_path, _class_name = _candidate.rsplit('.', 1)
        _mod = __import__(_module_path, fromlist=[_class_name])
        MambaBlockImpl = getattr(_mod, _class_name)
        logger.info('MambaViT: using Mamba from %s', _candidate)
        break
    except Exception:
        MambaBlockImpl = None


# ── Drop-path (stochastic depth) ─────────────────────────────────────
class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) >= self.drop_prob
        return x / (1 - self.drop_prob) * keep


# ── 2D Sine / Cosine Positional Encoding ─────────────────────────────
class SinCos2DPositionalEncoding(nn.Module):
    """Fixed 2D sine/cosine positional encoding.

    Encodes row (height) position using sin/cos in the first half of
    channels and column (width) position in the second half.  This lets
    a single row-major SSM scan capture both horizontal and vertical
    spatial structure.
    """
    def __init__(self, embed_dim: int, max_h: int = 64, max_w: int = 64,
                 temperature: float = 10000.0):
        super().__init__()
        assert embed_dim % 4 == 0, 'embed_dim must be divisible by 4 for 2D sin/cos PE'
        self.embed_dim = embed_dim
        self.max_h = max_h
        self.max_w = max_w
        self.temperature = temperature
        pe = self._build_pe(max_h, max_w, embed_dim, temperature)
        self.register_buffer('pe', pe, persistent=False)  # (1, H, W, D)

    @staticmethod
    def _build_pe(max_h: int, max_w: int, embed_dim: int,
                  temperature: float) -> torch.Tensor:
        d = embed_dim // 4  # each of sin_h, cos_h, sin_w, cos_w gets d dims

        pos_h = torch.arange(max_h, dtype=torch.float32).unsqueeze(1)  # (H, 1)
        pos_w = torch.arange(max_w, dtype=torch.float32).unsqueeze(1)  # (W, 1)

        dim_t = torch.arange(d, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / d)

        pe_h_sin = torch.sin(pos_h / dim_t)  # (H, d)
        pe_h_cos = torch.cos(pos_h / dim_t)  # (H, d)
        pe_w_sin = torch.sin(pos_w / dim_t)  # (W, d)
        pe_w_cos = torch.cos(pos_w / dim_t)  # (W, d)

        # Broadcast to (H, W, d) each
        pe_h_sin = pe_h_sin.unsqueeze(1).expand(-1, max_w, -1)
        pe_h_cos = pe_h_cos.unsqueeze(1).expand(-1, max_w, -1)
        pe_w_sin = pe_w_sin.unsqueeze(0).expand(max_h, -1, -1)
        pe_w_cos = pe_w_cos.unsqueeze(0).expand(max_h, -1, -1)

        pe = torch.cat([pe_h_sin, pe_h_cos, pe_w_sin, pe_w_cos], dim=-1)  # (H, W, D)
        return pe.unsqueeze(0)  # (1, H, W, D)

    def forward(self, tokens: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
        """Add positional encoding to tokens (B, L, D)."""
        if Hp <= self.max_h and Wp <= self.max_w:
            pe = self.pe[:, :Hp, :Wp, :].reshape(1, Hp * Wp, self.embed_dim)
        else:
            # Rebuild for larger-than-expected spatial dims
            pe = self._build_pe(Hp, Wp, self.embed_dim, self.temperature)
            pe = pe.to(tokens.device, tokens.dtype)
            pe = pe[:, :Hp, :Wp, :].reshape(1, Hp * Wp, self.embed_dim)
        return tokens + pe


# ── Learned 2D Positional Encoding ───────────────────────────────────
class Learned2DPositionalEncoding(nn.Module):
    """Learnable positional encoding (following ViTPose / VHR-BirdPose).

    A single ``nn.Parameter`` of shape ``(1, H*W, embed_dim)`` is learned
    during training and added to the patch tokens.
    """
    def __init__(self, embed_dim: int, max_h: int = 64, max_w: int = 64,
                 **_kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_h = max_h
        self.max_w = max_w
        num_patches = max_h * max_w
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, tokens: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
        """Add positional encoding to tokens (B, L, D)."""
        L = Hp * Wp
        pe = self.pos_embed[:, :L, :]
        return tokens + pe


# ── Mamba SSM Branch ──────────────────────────────────────────────────
class MambaSSMBranch(nn.Module):
    """Single-scan Mamba SSM branch with LayerNorm."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if MambaBlockImpl is None:
            raise ImportError(
                'mamba_ssm is required for MambaViT. '
                'Install it with: pip install mamba-ssm'
            )
        self.mamba = MambaBlockImpl(d_model=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(self.norm(x))


# ── Multi-Head Self-Attention Branch ──────────────────────────────────
class AttentionBranch(nn.Module):
    """Standard multi-head self-attention branch with LayerNorm."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, 'embed_dim must be divisible by num_heads'

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        B, N, C = x.shape

        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ── Sequential Hybrid Block ─────────────────────────────────────────
class SequentialHybridBlock(nn.Module):
    """Single block with Mamba SSM *followed by* multi-head self-attention.

    Forward:
        x = x + DropPath(MambaSSMBranch(x))   # SSM with residual
        x = x + DropPath(AttentionBranch(x))   # Attention with residual
        x = x + DropPath(FFN(LN(x)))           # feed-forward with residual
    """
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.ssm_branch = MambaSSMBranch(dim)
        self.att_branch = AttentionBranch(
            dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Feed-forward network
        self.norm_ffn = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── SSM with residual ──
        x = x + self.drop_path(self.ssm_branch(x))

        # ── Attention with residual ──
        x = x + self.drop_path(self.att_branch(x))

        # ── FFN with residual ──
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x


# ── Full Hybrid Backbone ─────────────────────────────────────────────
class MambaViT(nn.Module):
    """MambaViT branch: sequential SSM → Attention with 2D PE.

    Args:
        img_size:  spatial size of the input feature map (from HRNet conv1).
        patch_size: non-overlapping patch size for tokenisation.
        in_chans:  input channels (64 from HRNet conv1).
        embed_dim: token embedding dimension.
        depth:     number of stacked SequentialHybridBlocks.
        num_heads: attention heads per block.
        mlp_ratio: FFN hidden-dim multiplier.
        drop_rate: dropout inside FFN / attention projection.
        attn_drop_rate: attention-map dropout.
        drop_path_rate: stochastic depth (linearly increases across blocks).
    """
    def __init__(self, img_size: int = 128, patch_size: int = 4,
                 in_chans: int = 64, embed_dim: int = 768,
                 depth: int = 8, num_heads: int = 8,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, drop_path_rate: float = 0.2,
                 pos_embed_type: str = 'sincos'):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_features = embed_dim

        # ── patch embedding ──
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        Hp = img_size // patch_size
        Wp = img_size // patch_size

        # ── positional encoding ──
        if pos_embed_type == 'learned':
            self.pos_enc = Learned2DPositionalEncoding(
                embed_dim, max_h=Hp, max_w=Wp,
            )
        else:
            self.pos_enc = SinCos2DPositionalEncoding(
                embed_dim, max_h=Hp, max_w=Wp,
            )

        # ── stochastic depth schedule ──
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # ── hybrid blocks ──
        self.blocks = nn.ModuleList([
            SequentialHybridBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    # ── forward ──────────────────────────────────────────────────
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)                   # (B, D, Hp, Wp)
        Hp, Wp = x.shape[2], x.shape[3]
        tokens = x.flatten(2).transpose(1, 2)      # (B, L, D)

        # 2D positional encoding
        tokens = self.pos_enc(tokens, Hp, Wp)

        # Hybrid blocks
        for blk in self.blocks:
            tokens = blk(tokens)

        # Final norm + reshape to spatial
        tokens = self.norm(tokens)
        feat = tokens.transpose(1, 2).reshape(B, self.embed_dim, Hp, Wp)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    # ── weight initialisation ────────────────────────────────────
    def init_weights(self, pretrained: Optional[str] = None):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(_init)


def build_mamba_vit(**kwargs) -> MambaViT:
    """Builder helper for the MambaViT branch."""
    return MambaViT(**kwargs)
