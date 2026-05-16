r"""MambaVision adapter to be a drop-in replacement for the ViT backbone.

This module wraps a MambaVision model (if installed) and projects its final
feature map to match the ViT output shape: (B, embed_dim, Hp, Wp). If the
`mambavision` package is not available at import/instantiation time, a small
convolutional stub is used so the rest of the codebase can import the
backbone without crashing; attempting to run will raise a clearer error later
if the real package is required.

The adapter aims to preserve the same external interface as `ViT` in
`models/VHR_BirdPose/vit.py` so it can be swapped in without changes elsewhere.
"""
# Adapted for the bird head pose estimation work in this repository.
# This wrapper preserves upstream MambaVision provenance while integrating it into the HR-MambaVision branch.

import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# allow importing the bundled MambaVision implementation from the workspace
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
mv_root = os.path.join(repo_root, 'MambaVision')
if mv_root not in sys.path:
    sys.path.insert(0, mv_root)
# also ensure the local `mamba` package (contains `mamba_ssm`) is importable
mamba_root = os.path.join(repo_root, 'mamba')
if os.path.isdir(mamba_root) and mamba_root not in sys.path:
    sys.path.insert(0, mamba_root)

from ..common.base_backbone import BaseBackbone

logger = logging.getLogger(__name__)

try:
    from mambavision.models.mamba_vision import MambaVision as _MVClass
    from mambavision.models.mamba_vision import mamba_vision_T as _build_mamba_T
    from mambavision.models.mamba_vision import mamba_vision_T2 as _build_mamba_T2
    from mambavision.models.mamba_vision import mamba_vision_S as _build_mamba_S
    from mambavision.models.mamba_vision import mamba_vision_B as _build_mamba_B
    from mambavision.models.mamba_vision import mamba_vision_B_21k as _build_mamba_B_21k
    from mambavision.models.mamba_vision import mamba_vision_L as _build_mamba_L
    from mambavision.models.mamba_vision import mamba_vision_L_21k as _build_mamba_L_21k
    from mambavision.models.mamba_vision import mamba_vision_L2 as _build_mamba_L2
    from mambavision.models.mamba_vision import mamba_vision_L2_512_21k as _build_mamba_L2_512_21k
    from mambavision.models.mamba_vision import mamba_vision_L3_256_21k as _build_mamba_L3_256_21k
    from mambavision.models.mamba_vision import mamba_vision_L3_512_21k as _build_mamba_L3_512_21k
    from mambavision.models.mamba_vision import mamba_vision_CUSTOM as _build_mamba_CUSTOM
except Exception as exc:
    logger.error('Could not import MambaVision package.', exc_info=True)
    _MVClass = None
    _build_mamba_T = None
    _build_mamba_T2 = None
    _build_mamba_S = None
    _build_mamba_B = None
    _build_mamba_B_21k = None
    _build_mamba_L = None
    _build_mamba_L_21k = None
    _build_mamba_L2 = None
    _build_mamba_L2_512_21k = None
    _build_mamba_L3_256_21k = None
    _build_mamba_L3_512_21k = None
    _build_mamba_CUSTOM = None


class MambaVision(BaseBackbone):
    """Wrapper around the bundled MambaVision implementation that exposes a
    ViT-like `forward_features` returning a spatial feature map tensor
    (B, embed_dim, Hp, Wp) so it can be dropped in for `models/VHR_BirdPose/vit.ViT`.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, mamba_variant='T', pretrained=False, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_features = embed_dim

        self._has_mv = False
        self.mv = None

        if _MVClass is not None and _build_mamba_T is not None:
            try:
                available_builders = {
                    'T': _build_mamba_T,
                    'T2': _build_mamba_T2,
                    'S': _build_mamba_S,
                    'B': _build_mamba_B,
                    'B_21K': _build_mamba_B_21k,
                    'L': _build_mamba_L,
                    'L_21K': _build_mamba_L_21k,
                    'L2': _build_mamba_L2,
                    'L2_512_21K': _build_mamba_L2_512_21k,
                    'L3_256_21K': _build_mamba_L3_256_21k,
                    'L3_512_21K': _build_mamba_L3_512_21k,
                }

                variant_key = str(mamba_variant).upper()
                if variant_key == 'CUSTOM':
                    bkwargs = dict(kwargs)
                    bkwargs.setdefault('in_chans', in_chans)
                    bkwargs.setdefault('resolution', img_size)
                    custom_cfg = bkwargs.pop('custom_cfg', {})
                    self.mv = _build_mamba_CUSTOM(pretrained=pretrained, custom_cfg=custom_cfg, **bkwargs)
                elif variant_key in available_builders:
                    bkwargs = dict(kwargs)
                    bkwargs.setdefault('in_chans', in_chans)
                    bkwargs.setdefault('resolution', img_size)
                    self.mv = available_builders[variant_key](pretrained=pretrained, **bkwargs)
                else:
                    valid = sorted(list(available_builders.keys()) + ['CUSTOM'])
                    raise ValueError(f'Unknown MambaVision variant "{mamba_variant}". Supported variants: {valid}')

                self._has_mv = True
            except Exception as exc:
                logger.exception('Failed to instantiate MambaVision backbone.')
                raise RuntimeError('Failed to instantiate MambaVision backbone.') from exc
        else:
            logger.error('MambaVision package is not available. Please install `mambavision`.')
            raise ImportError('MambaVision package is not available. Please install `mambavision`.')

        if self._has_mv and self.mv is not None:
            try:
                in_channels = list(self.mv.norm.parameters())[0].shape[0]
            except Exception:
                in_channels = embed_dim
        else:
            in_channels = embed_dim

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def init_weights(self, pretrained=None):
        if self._has_mv and hasattr(self.mv, '_load_state_dict') and pretrained is not None:
            try:
                self.mv._load_state_dict(pretrained, strict=False)
            except Exception:
                pass

        def _init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)

        self.proj.apply(_init_weights)

    def forward_features(self, x):
        B, C, H, W = x.shape

        if self._has_mv and self.mv is not None:
            x = self.mv.patch_embed(x)
            for lvl in self.mv.levels:
                x = lvl(x)
            x = self.mv.norm(x)
            feat = x
        else:
            raise RuntimeError('MambaVision backbone is not available.')

        feat = self.proj(feat)

        target_Hp = H // self.patch_size
        target_Wp = W // self.patch_size
        if feat.shape[2] != target_Hp or feat.shape[3] != target_Wp:
            feat = F.interpolate(feat, size=(target_Hp, target_Wp), mode='bicubic', align_corners=False)

        return feat

    def forward(self, x):
        return self.forward_features(x)
