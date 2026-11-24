"""Immutable single-task ViTPose-B BirdGaze training recipe (not an MMPose config).

This is referenced and hashed by the isolated launcher. The architecture is
official ViTPose-B, adapted from the pinned 256x192 COCO reference to BirdGaze
256x256 crops and four keypoints. No pretrained weights, ever.
"""
from __future__ import annotations

RECIPE = {
    "architecture": "standard_vitpose_b_single_task",
    "backbone": {
        "name": "ViT",
        "patch_size": 16,
        "embedding_dimension": 768,
        "depth": 12,
        "attention_heads": 12,
        "mlp_ratio": 4,
        "drop_path_rate": 0.3,
        "qkv_bias": True,
        "pretrained": None,
    },
    "head": {
        "name": "TopdownHeatmapSimpleHead",
        "deconv_layers": 0,
        "upsample": 4,
        "kernel": 3,
        "out_channels": 4,
        "loss": "visibility_or_coordinate_masked_mse",
    },
    "image_size": [256, 256],
    "heatmap_size": [64, 64],
    "keypoints": ["crown", "left_eye", "right_eye", "beak_tip"],
    "seed": 2026,
    "epochs": 100,
    "microbatch": 4,
    "effective_batch": 64,
    "optimizer": "AdamW",
    "lr": 0.0005,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "min_lr": 0.00001,
    "gradient_clip_norm": 1.0,
    "mixed_precision": "fp16_grad_scaler",
    "validation_checkpoint_selection": "source_scale_normalized_PCK_at_0.05",
    "scientific_test_set": "held_out_annotated_videos_not_used_for_selection",
}
