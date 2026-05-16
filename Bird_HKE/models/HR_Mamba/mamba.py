"""Pure-Mamba vision backbone (research-grade reference implementation).

Implements a true "pure Mamba" backbone:
 - Patch embedding (conv patching)
 - Flatten to sequence tokens
 - Stacked Mamba blocks operating on token sequences
 - Multi-directional scanning (LR, RL, TB, BT) fused per-layer
 - Project back to spatial feature map (B, C, Hp, Wp)

Provides `get_pure_mamba` builder for easy integration. Requires a real
`Mamba` implementation from `mamba_ssm`.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Try to import a Mamba block implementation from commonly-used locations.
MambaBlockImpl = None
for candidate in (
    'mamba_ssm.modules.Mamba',
    'mamba_ssm.models.Mamba',
    'mamba_ssm.Mamba',
):
    try:
        module_path, class_name = candidate.rsplit('.', 1)
        mod = __import__(module_path, fromlist=[class_name])
        MambaBlockImpl = getattr(mod, class_name)
        logger.info(f'Using Mamba block implementation from {candidate}')
        print(f'Using Mamba block implementation from {candidate}')
        break
    except Exception:
        MambaBlockImpl = None


class PatchEmbed(nn.Module):
    """Patch embedding via non-overlapping conv (kernel=stride=patch_size).

    Returns token sequence (B, N, C) and spatial size (Hp, Wp).
    """
    def __init__(self, in_chans: int = 3, embed_dim: int = 768, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, C', Hp, Wp)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, C') where N=Hp*Wp
        return x, (Hp, Wp)


class PureMambaBlock(nn.Module):
    """Single residual Mamba block wrapping a `Mamba` operator or fallback.

    Applies LayerNorm -> Mamba (or stub) -> residual add.
    """
    def __init__(self, dim: int, mamba_kwargs: Optional[dict] = None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba_kwargs = mamba_kwargs or {}

        if MambaBlockImpl is None:
            logger.error('Mamba implementation not found. Please install `mamba_ssm`.')
            raise ImportError('Mamba implementation not found. Please install `mamba_ssm`.')
        try:
            self.mamba = MambaBlockImpl(d_model=dim, **self.mamba_kwargs)
        except Exception as exc:
            logger.exception('Failed to instantiate Mamba implementation.')
            raise RuntimeError('Failed to instantiate Mamba implementation.') from exc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        return x + self.mamba(self.norm(x))


class MultiDirectionalMamba(nn.Module):
    """Fuse multi-directional Mamba scans (LR, RL, TB, BT).

    The module takes token sequence `x` (B, N, C) and spatial size (Hp, Wp).
    It produces a fused output with the same shape. Directions can be
    enabled via the `directions` argument which accepts an iterable of
    letters: 'R' (left->right), 'L' (right->left), 'D' (up->down),
    'U' (down->up). Order is ignored; enabled scans run in parallel.
    """
    def __init__(self, dim: int, mamba_kwargs: Optional[dict] = None, fuse='linear', directions: Optional[Iterable[str]] = None):
        super().__init__()
        dirs = {d.upper() for d in (directions or 'UDLR')}

        self.use_lr = 'R' in dirs
        self.use_rl = 'L' in dirs
        self.use_tb = 'D' in dirs
        self.use_bt = 'U' in dirs

        self.lr = PureMambaBlock(dim, mamba_kwargs=mamba_kwargs) if self.use_lr else None
        self.rl = PureMambaBlock(dim, mamba_kwargs=mamba_kwargs) if self.use_rl else None
        self.tb = PureMambaBlock(dim, mamba_kwargs=mamba_kwargs) if self.use_tb else None
        self.bt = PureMambaBlock(dim, mamba_kwargs=mamba_kwargs) if self.use_bt else None

        self.n_dirs = int(self.use_lr) + int(self.use_rl) + int(self.use_tb) + int(self.use_bt)
        self.fuse_mode = fuse

        if self.fuse_mode == 'linear' and self.n_dirs > 1:
            self.fuse = nn.Linear(dim * self.n_dirs, dim)
            self.gate_proj = None
            self.dir_norms = None
        elif self.fuse_mode == 'concat_gate' and self.n_dirs > 1:
            self.fuse = nn.Linear(dim * self.n_dirs, dim)
            self.gate_proj = nn.Linear(dim * self.n_dirs, self.n_dirs)
            self.dir_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.n_dirs)])
        else:
            self.fuse = None
            self.gate_proj = None
            self.dir_norms = None

    def forward(self, x: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
        B, N, C = x.shape
        assert N == Hp * Wp, 'Hp*Wp must equal token sequence length'

        outs = []

        if self.use_lr:
            outs.append(self.lr(x))

        if self.use_rl:
            rl_in = torch.flip(x, dims=[1])
            rl_out = self.rl(rl_in)
            rl_out = torch.flip(rl_out, dims=[1])
            outs.append(rl_out)

        if self.use_tb or self.use_bt:
            x2d = x.view(B, Hp, Wp, C)
            tb_in = x2d.permute(0, 2, 1, 3).contiguous().view(B, N, C)

            if self.use_tb:
                tb_out = self.tb(tb_in)
                tb_out = tb_out.view(B, Wp, Hp, C).permute(0, 2, 1, 3).contiguous().view(B, N, C)
                outs.append(tb_out)

            if self.use_bt:
                bt_in = torch.flip(tb_in, dims=[1])
                bt_out = self.bt(bt_in)
                bt_out = torch.flip(bt_out, dims=[1])
                bt_out = bt_out.view(B, Wp, Hp, C).permute(0, 2, 1, 3).contiguous().view(B, N, C)
                outs.append(bt_out)

        if len(outs) == 0:
            return x

        if len(outs) == 1:
            return outs[0]

        if self.fuse_mode == 'concat_gate':
            assert self.gate_proj is not None, 'concat_gate requires gate_proj to be initialized'
            if self.dir_norms is not None:
                outs = [self.dir_norms[i](o) for i, o in enumerate(outs)]
            cat = torch.cat(outs, dim=-1)
            out_mix = self.fuse(cat)
            logits = self.gate_proj(cat)
            alpha = torch.softmax(logits, dim=-1)
            stack = torch.stack(outs, dim=-2)
            out_gate = (alpha.unsqueeze(-1) * stack).sum(dim=-2)
            out = out_mix + out_gate
        else:
            out = torch.cat(outs, dim=-1)
            if self.fuse is not None:
                out = self.fuse(out)
            else:
                chunk_size = out.shape[-1] // len(outs)
                acc = out[:, :, :chunk_size]
                for i in range(1, len(outs)):
                    acc = acc + out[:, :, i * chunk_size:(i + 1) * chunk_size]
                out = acc

        return out


class PureMambaVision(nn.Module):
    """Stacked pure-Mamba backbone with multi-directional fusion.

    Returns a spatial feature map (B, embed_dim, Hp, Wp).
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 768, depth: int = 8, mamba_kwargs: Optional[dict] = None,
                 fuse: str = 'linear', directions: Optional[Iterable[str]] = None):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)
        self.depth = depth
        self.blocks = nn.ModuleList([
            MultiDirectionalMamba(embed_dim, mamba_kwargs=mamba_kwargs, fuse=fuse, directions=directions)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens, (Hp, Wp) = self.patch_embed(x)
        for blk in self.blocks:
            tokens = blk(tokens, Hp, Wp)

        tokens = self.norm(tokens)
        B, N, C = tokens.shape
        feat = tokens.transpose(1, 2).contiguous().view(B, C, Hp, Wp)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    def init_weights(self, pretrained: Optional[bool] = None):
        """Initialize weights for PureMambaVision modules."""
        def _init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                try:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                except Exception:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_init_weights)


def get_pure_mamba(cfg=None, img_size: int = 224, patch_size: int = 16,
                   in_chans: int = 3, embed_dim: int = 768, depth: int = 8, **kwargs) -> nn.Module:
    """Builder helper to create the PureMambaVision backbone."""
    mamba_kwargs = kwargs.pop('mamba_kwargs', None)
    fuse = kwargs.pop('fuse', 'linear')
    directions = kwargs.pop('directions', None)

    if directions is None and cfg is not None:
        try:
            if hasattr(cfg, 'get'):
                mcfg = cfg.get('MAMBA') or cfg.get('mamba') or {}
            else:
                mcfg = {}
            directions = None
            if isinstance(mcfg, dict):
                directions = mcfg.get('VARIANT') or mcfg.get('variant')
        except Exception:
            directions = None

    return PureMambaVision(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                           embed_dim=embed_dim, depth=depth, mamba_kwargs=mamba_kwargs, fuse=fuse, directions=directions)


__all__ = ['PatchEmbed', 'PureMambaBlock', 'MultiDirectionalMamba', 'PureMambaVision', 'get_pure_mamba']
