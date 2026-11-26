"""Standard ViTPose-B (ViT-B + simple heatmap head), from-scratch BirdGaze adapter.

Architectural reference (pinned): ViTAE-Transformer/ViTPose at
c050ed29112da7704797cc1a65af0234b525010d
mmpose/models/backbones/vit.py and topdown_heatmap_simple_head.py.
That project draws on OpenMMLab and the ViTPose authors; preserve its licenses
in upstream source. This is an independent PyTorch adaptation, NOT ViTPose++.
No pretrained checkpoint is loaded, imported, downloaded or required.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from comparison_models.common.losses import masked_heatmap_mse
from comparison_models.common.transformer import (
    Attention, DropPath, init_vision_transformer, validate_vit_geometry,
)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim),
            nn.Dropout(0.),
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))



def bilinear_weights(source_size: int, scale: int = 4) -> torch.Tensor:
    """Separable interpolation weights identical to align_corners=False.

    CUDA native upsample_bilinear2d_backward may be nondeterministic in some
    supported PyTorch versions. The two GEMMs below preserve bilinear values
    but use deterministic matrix-multiplication backward with cuBLAS workspace.
    """
    destination = source_size * scale
    source_coordinates = (
        (torch.arange(destination, dtype=torch.float32) + 0.5) / scale - 0.5
    )
    low = source_coordinates.floor().to(torch.long)
    high = low + 1
    alpha = source_coordinates - low
    low = low.clamp(0, source_size - 1)
    high = high.clamp(0, source_size - 1)
    weights = torch.zeros(destination, source_size, dtype=torch.float32)
    rows = torch.arange(destination)
    weights.index_put_((rows, low), 1.0 - alpha, accumulate=True)
    weights.index_put_((rows, high), alpha, accumulate=True)
    return weights


def deterministic_bilinear_4x(x, weights=None):
    """Same 4x bilinear upsampling as F.interpolate; deterministic CUDA grads."""
    if x.ndim != 4 or x.shape[-2:] != (16, 16):
        raise ValueError("ViTPose-B bilinear head expects 16x16 patch features")
    w = (bilinear_weights(16) if weights is None else weights).to(
        device=x.device, dtype=x.dtype)
    return torch.matmul(torch.matmul(w, x), w.transpose(0, 1))


class ViTPoseB(nn.Module):
    """Matches official single-task ViTPose-B ViT plus zero-deconv simple head.

    Official ViT PatchEmbed uses padding=2 with patch_size=stride=16;
    it retains a positional class-token slot, but no class token is passed to
    transformer blocks. The first positional slot is added to all patch tokens.
    The head uses ReLU + 4x bilinear upsample (align_corners=False) then
    3x3 convolution to four heatmaps, without sigmoid.
    """

    def __init__(self, image_size: int = 256, keypoints: int = 4,
                 dim: int = 768, depth: int = 12, heads: int = 12,
                 drop_path_rate: float = 0.3):
        super().__init__()
        validate_vit_geometry(image_size, dim, heads, keypoints)
        self.image_size = image_size
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, kernel_size=16,
                                          stride=16, padding=2)
        patches = (image_size // 16) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, heads, drop_path_rate * i / max(depth - 1, 1))
            for i in range(depth)
        ])
        self.last_norm = nn.LayerNorm(dim, eps=1e-6)
        self.final_layer = nn.Conv2d(dim, keypoints, 3, padding=1)
        self.register_buffer("_upsample4_weights", bilinear_weights(16),
                             persistent=False)
        self.reset_parameters()

    def reset_parameters(self):
        init_vision_transformer(self, self.pos_embed)
        nn.init.normal_(self.final_layer.weight, std=0.001)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, image):
        if image.ndim != 4 or tuple(image.shape[-2:]) != (256, 256):
            raise ValueError("Expected input [batch,3,256,256]")
        b = image.size(0)
        x = self.patch_embed.proj(image)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for block in self.blocks:
            x = block(x)
        x = self.last_norm(x).transpose(1, 2).reshape(b, -1, h, w)
        x = deterministic_bilinear_4x(F.relu(x), self._upsample4_weights)
        return self.final_layer(x)
