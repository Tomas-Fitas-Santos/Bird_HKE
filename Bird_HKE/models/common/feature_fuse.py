"""Feature fusion utilities used by the adapted VHR-BirdPose and MambaVision branches.

The implementations in this module are part of the repository-specific
integration work for the bird head pose estimation pipeline.
"""

import torch
import torch.nn as nn

from .cross_attn import CrossAttentionBlock


class AddFeatureCombiner(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(AddFeatureCombiner, self).__init__()

    def forward(self, feat1, feat2):
        return feat1 + feat2


class LinearFeatureCombiner(nn.Module):
    def __init__(self, in_channel, out_channel, *args, **kwargs):
        super(LinearFeatureCombiner, self).__init__()
        self.linear1 = nn.Linear(in_channel, out_channel)
        self.linear2 = nn.Linear(in_channel, out_channel)

    def forward(self, feat1, feat2):
        feat1 = self.linear1(feat1)
        feat2 = self.linear2(feat2)
        return torch.concat((feat1, feat2), -1)


class CrossAttentionFeatureCombiner(nn.Module):
    def __init__(self, in_channels, out_channels, headers=8, *args, **kwargs):
        super(CrossAttentionFeatureCombiner, self).__init__()
        self.cross_att = CrossAttentionBlock(in_channels, headers)

    def forward(self, feat1, feat2):
        assert feat1.shape == feat2.shape

        B, C, H, W = feat1.shape
        feat1 = feat1.reshape(1, B * C * H, W).permute(1, 0, 2)
        feat2 = feat2.reshape(1, B * C * H, W).permute(1, 0, 2)
        feat = torch.cat((feat1, feat2), dim=1)
        feat = self.cross_att(feat)
        feat = feat.permute(1, 0, 2)
        feat = feat.reshape(B, C, H, W)
        return feat
