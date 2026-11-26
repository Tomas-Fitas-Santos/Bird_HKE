"""Single-dataset BirdGaze PoseBH-B experiment recipe.

This is *not* the published six-dataset, pretrained ViTPose++ initialization,
six-expert, clustered-prototype/CSS model. This preserves PoseBH's ViTMoE,
prototype similarity head and EMA prototype learning with only one BirdGaze
expert, four keypoints and a one-dataset consistency adaptation.
See README for exact deviations. No pretrained weights are permitted.
"""
from __future__ import annotations

RECIPE = {
    "architecture": "posebh_b_single_birdgaze_adaptation",
    "methodology": "single_dataset_prototypes_and_unlabeled_consistency",
    "upstream_commit": "8f03438f913b559edb0a2beed54e77d1560edba3",
    "input_size": [256, 256],
    "heatmap_size": [64, 64],
    "keypoints": ["crown", "left_eye", "right_eye", "beak_tip"],
    "backbone": {
        "type": "ViTMoE", "layers": 12, "hidden": 768, "heads": 12,
        "patch_size": 16, "mlp_ratio": 4,
        "active_dataset_experts": 1, "expert_features": 192,
        "shared_features": 576, "drop_path_rate": 0.3,
    },
    "heads": {
        "auxiliary_regression": "two_transposed_convs_256_with_1x1_4ch",
        "prototype": "two_transposed_convs_256_res_neck_64d",
        "prototypes_per_keypoint": 3,
        "prototype_ema": 0.999,
        "proto_mse_weight": 3.33e-6,
        "ppc_weight": 0.01,
        "ppd_weight": 0.001,
        "proto_loss_weight": 1.25e-5,
        "self_supervision_weight": 0.001,
        "self_supervision_from_epoch": 50,
        "agreement_max_heatmap_px": 2.1,
        "pseudo_label_confidence": 0.25,
        "original_cross_dataset_css": False,
        "original_cluster_prototype_supervision": False,
    },
    "weights": "random_from_scratch_no_pretrained",
    "seed": 2026,
    "epochs": 100,
    "microbatch": 2,
    "effective_batch": 64,
    "optimizer": "AdamW",
    "lr": 0.001,
    "weight_decay": 0.1,
    "betas": [0.9, 0.999],
    # Paper-faithful optimizer/schedule (configs/.../mdt/vitb_posebh.py):
    # LayerDecayOptimizerConstructor with 12 layers and rate 0.75, linear
    # warmup for 500 iterations at ratio 1e-3, then step decay at epochs 50
    # and 90 of 100. Our budget is also 100 epochs, so the milestones are
    # used unchanged.
    "schedule": {
        "type": "vit_layer_decay",
        "num_layers": 12,
        "layer_decay_rate": 0.75,
    },
    "lr_schedule": {
        "type": "vit_step",
        "warmup_iters": 500,
        "warmup_ratio": 0.001,
        "milestones_epochs": [50, 90],
        "gamma": 0.1,
    },
    "warmup_epochs": 0,
    "min_lr": 0.00001,
    "gradient_clip_norm": 1.0,
    "amp": "fp16_scaler",
    "grad_scaler_init_scale": 16384.0,
    "checkpoint_selection": "prototype_heatmap_source_normalized_PCK_at_0.05",
    "scientific_test": "independently_annotated_held_out_videos",
}
