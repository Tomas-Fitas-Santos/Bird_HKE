"""HR-MambaViT: HRNet + MambaViT (parallel SSM + Attention branch).

This is the dedicated pose model for the HR-MambaViT architecture.
The MambaViT branch runs in parallel with HRNet and is fused at stage4.

Config keys read from YAML:
    MODEL.EXTRA.MAMBA_VIT.PATCH_SIZE       (int, default 4)
    MODEL.EXTRA.MAMBA_VIT.EMBED_DIM        (int, default 768)
    MODEL.EXTRA.MAMBA_VIT.DEPTH            (int, default 4)
    MODEL.EXTRA.MAMBA_VIT.NUM_HEADS        (int, default 8)
    MODEL.EXTRA.MAMBA_VIT.MLP_RATIO        (float, default 4.0)
    MODEL.EXTRA.MAMBA_VIT.DROP_RATE        (float, default 0.0)
    MODEL.EXTRA.MAMBA_VIT.ATTN_DROP_RATE   (float, default 0.0)
    MODEL.EXTRA.MAMBA_VIT.DROP_PATH_RATE   (float, default 0.2)
    MODEL.EXTRA.MAMBA_VIT.DEPROJ_OUT_CH    (int, default 64)
    MODEL.EXTRA.FUSE_STREGY               ('add' | 'concat' | 'cross_att')
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.pose_hrnet import PoseHighResolutionNet
from .mamba_vit import MambaViT
from ..common.feature_fuse import AddFeatureCombiner, LinearFeatureCombiner, CrossAttentionFeatureCombiner

logger = logging.getLogger(__name__)


class HRMambaViTPose(PoseHighResolutionNet):
    """HR-MambaViT = HRNet + MambaViT branch."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        extra = cfg['MODEL']['EXTRA']
        mcfg = extra.get('MAMBA_VIT', {})

        patch_size   = mcfg.get('PATCH_SIZE', 4)
        embed_dim    = mcfg.get('EMBED_DIM', 768)
        depth        = mcfg.get('DEPTH', 4)
        num_heads    = mcfg.get('NUM_HEADS', 8)
        mlp_ratio    = mcfg.get('MLP_RATIO', 4.0)
        drop_rate    = mcfg.get('DROP_RATE', 0.0)
        attn_drop    = mcfg.get('ATTN_DROP_RATE', 0.0)
        drop_path    = mcfg.get('DROP_PATH_RATE', 0.2)
        deproj_out   = mcfg.get('DEPROJ_OUT_CH', 64)
        pos_embed    = mcfg.get('POS_EMBED_TYPE', 'sincos')

        # MambaViT branch
        self.vit = MambaViT(
            img_size=128,
            patch_size=patch_size,
            in_chans=64,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop,
            drop_path_rate=drop_path,
            pos_embed_type=pos_embed,
        )

        # De-projection: tokens → spatial, match conv1 channels
        self.vit_deproj = nn.ConvTranspose2d(
            embed_dim, deproj_out,
            kernel_size=patch_size, stride=patch_size,
        )

        # Channel/spatial adapter before fusion with HRNet stage4
        self.att_fit_out = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.Conv2d(64, 32, kernel_size=1),
        )

        # Fusion strategy
        fuse = extra.get('FUSE_STREGY', 'add')
        if fuse == 'add':
            self.feature_fuse = AddFeatureCombiner()
        elif fuse == 'concat':
            self.feature_fuse = LinearFeatureCombiner(64, 32)
        elif fuse == 'cross_att':
            self.feature_fuse = CrossAttentionFeatureCombiner(64, 32)
        else:
            raise NotImplementedError(f'Unknown FUSE_STREGY: {fuse}')

    def forward(self, x):
        # ── conv1 → branch split ──
        x = self.conv1(x)
        x_att = self.vit(x)
        x_att = self.vit_deproj(x_att)

        # Residual add with conv1 output (if shapes match)
        if x_att.shape == x.shape:
            x_att = x + x_att
        else:
            x_att = F.interpolate(x_att, size=x.shape[2:], mode='bilinear', align_corners=False)
            if x_att.shape[1] != x.shape[1]:
                x_att = nn.Conv2d(x_att.shape[1], x.shape[1], 1, device=x_att.device)(x_att)
            x_att = x + x_att

        # ── HRNet backbone ──
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # ── Fuse MambaViT features with HRNet stage4 ──
        x_att = self.att_fit_out(x_att)
        y_list[0] = self.feature_fuse(y_list[0], x_att)

        return self.final_layer(y_list[0])

    def init_weights(self, pretrained=''):
        super().init_weights(pretrained)
        if hasattr(self.vit, 'init_weights'):
            self.vit.init_weights()


def get_pose_net(cfg, is_train, **kwargs):
    model = HRMambaViTPose(cfg, **kwargs)
    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])
    return model
