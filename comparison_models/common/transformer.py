"""Transformer primitives shared by the ViTPose-B and PoseBH-B adapters.

Both architectures descend from the same Vision-Transformer block used by the
pinned upstream references (ViTPose ``mmpose/models/backbones/vit.py`` at
c050ed29112da7704797cc1a65af0234b525010d and PoseBH's ViTMoE at
8f03438f913b559edb0a2beed54e77d1560edba3). Their attention and stochastic
depth operators are byte-for-byte equivalent, so they live here once.

This module is deliberately model-agnostic and must stay importable without
MMPose, MMCV or CUDA. Neither ViTPose-B nor PoseBH-B may import the other
model's implementation; both import these primitives instead. Architecture
specific behaviour (PoseBH's MoE FFN, ViTPose's simple head, the RTMCC/SimCC
decoder) stays inside each model package.
"""
from __future__ import annotations

import torch
from torch import nn


class DropPath(nn.Module):
    """Stochastic depth: drop residual paths per sample during training."""

    def __init__(self, probability: float):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x):
        if not self.training or self.probability <= 0:
            return x
        keep = 1 - self.probability
        mask = (torch.rand((x.shape[0],) + (1,) * (x.ndim - 1),
                           device=x.device, dtype=x.dtype) < keep)
        return x * mask / keep


class Attention(nn.Module):
    """Standard scaled dot-product QKV attention used by ViT-style backbones."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("Embedding must be divisible by the head count")
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(
            b, n, 3, self.num_heads, c // self.num_heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        y = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(y))


def init_vision_transformer(model: nn.Module, pos_embed: torch.Tensor,
                            extra_init=None) -> None:
    """Shared ViT weight initialisation used by both comparison backbones.

    Truncated-normal positional embeddings and projections, and unit
    LayerNorms. ``extra_init`` is called for every remaining module inside the
    *same* traversal so that a model adding its own rules (PoseBH initialises
    its transposed-convolution deconvolutions) consumes random numbers in
    exactly the same order as before the primitives were shared.
    """
    nn.init.trunc_normal_(pos_embed, std=0.02)
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif extra_init is not None:
            extra_init(layer)


def validate_vit_geometry(image_size: int, dim: int, heads: int,
                          keypoints: int) -> None:
    """Guard the shared BirdGaze input contract for both ViT backbones."""
    if image_size != 256:
        raise ValueError("BirdGaze comparison models require 256x256 crops")
    if keypoints != 4:
        raise ValueError("BirdGaze has exactly four landmarks")
    if dim % heads:
        raise ValueError("Embedding dimension must be divisible by heads")
