"""Heatmap supervision shared by the two single-task ViT comparison models.

This is the plain binary-masked joint MSE used by the official ViTPose
``JointsMSELoss``. ViTPose-B uses it directly; PoseBH-B reuses it for its
supervised and prototype-heatmap terms while keeping its own PPC/PPD/CSS
prototype losses inside ``comparison_models.posebh.model``.

It is **not** the focal-weighted MSE used by the native Bird_HKE models.
Architecture-specific supervision is intentionally not homogenised.
"""
from __future__ import annotations

import torch


def masked_heatmap_mse(prediction, target, valid):
    """Official ViTPose JointsMSELoss with binary BirdGaze coordinate weights.

    Args:
        prediction: ``[N, 4, 64, 64]`` predicted heatmaps.
        target: ``[N, 4, 64, 64]`` Gaussian targets.
        valid: ``[N, 4]`` per-keypoint supervision mask. Absent and
            unannotated joints contribute nothing.
    """
    if prediction.shape != target.shape or prediction.shape[1] != 4:
        raise ValueError("Expected matching [N,4,64,64] heatmaps")
    if valid.shape != prediction.shape[:2]:
        raise ValueError("Expected [N,4] coordinate supervision mask")
    # Same per-joint mean squared heatmap error as official ViTPose JointsMSELoss.
    sq = (prediction - target).square().mean(dim=(-1, -2))
    return (sq * valid.to(sq.dtype)).sum() / (prediction.shape[0] * 4)
