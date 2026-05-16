import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.pose_hrnet import PoseHighResolutionNet
from .mamba import get_pure_mamba
from ..common.feature_fuse import AddFeatureCombiner, LinearFeatureCombiner, CrossAttentionFeatureCombiner


logger = logging.getLogger(__name__)


class HRMambaPose(PoseHighResolutionNet):
    """HRNet + Pure Mamba SSM backbone."""
    def __init__(self, cfg, **kwargs):
        super(HRMambaPose, self).__init__(cfg, **kwargs)
        extra = cfg['MODEL']['EXTRA']
        mamba_cfg = extra.get('MAMBA', {})

        pm_patch = mamba_cfg.get('PATCH_SIZE', 4)
        pm_embed = mamba_cfg.get('EMBED_DIM', 768)
        pm_depth = mamba_cfg.get('DEPTH', 8)
        pm_fuse = mamba_cfg.get('FUSE', 'linear')
        pm_dirs = mamba_cfg.get('VARIANT', 'UDLR')

        self.vit = get_pure_mamba(
            img_size=128,
            patch_size=pm_patch,
            in_chans=64,
            embed_dim=pm_embed,
            depth=pm_depth,
            fuse=pm_fuse,
            directions=pm_dirs
        )

        deproj_out = mamba_cfg.get('DEPROJ_OUT_CHANNELS', 64)
        self.vit_deproj = nn.ConvTranspose2d(pm_embed, deproj_out, (pm_patch, pm_patch), stride=pm_patch, padding=0)

        self.att_fit_out = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), stride=2, padding=1),
            nn.Conv2d(64, 32, kernel_size=1)
        )

        fuse_stregy = extra.get('FUSE_STREGY', 'add')
        if fuse_stregy == 'add':
            self.feature_fuse = AddFeatureCombiner()
        elif fuse_stregy == 'concat':
            self.feature_fuse = LinearFeatureCombiner(64, 32)
        elif fuse_stregy == 'cross_att':
            self.feature_fuse = CrossAttentionFeatureCombiner(64, 32)
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.conv1(x)
        x_att = self.vit(x)
        x_att = self.vit_deproj(x_att)

        if x_att.shape == x.shape:
            x_att = x + x_att
        else:
            try:
                x_att = F.interpolate(x_att, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
                if x_att.shape[1] != x.shape[1]:
                    proj = nn.Conv2d(x_att.shape[1], x.shape[1], kernel_size=1).to(x_att.device)
                    x_att = proj(x_att)
                x_att = x + x_att
            except Exception:
                pass

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

        x_att = self.att_fit_out(x_att)
        y_list[0] = self.feature_fuse(y_list[0], x_att)

        x = self.final_layer(y_list[0])
        return x

    def init_weights(self, pretrained=''):
        super(HRMambaPose, self).init_weights(pretrained)
        if hasattr(self.vit, 'init_weights'):
            self.vit.init_weights()


def get_pose_net(cfg, is_train, **kwargs):
    model = HRMambaPose(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
