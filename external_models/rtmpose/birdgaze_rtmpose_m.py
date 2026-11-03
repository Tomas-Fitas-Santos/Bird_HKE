"""RTMPose-M BirdGaze from-scratch training, preserving the author's RTMCC head.

Run from the Bird_HKE repository root using the pinned upstream MMPose checkout.
Set BIRDGAZE_SCENARIO=FD, OS or CS before launching. Separate environment needed.
"""
import os
from pathlib import Path

_base_ = [
    "../../.external_pose_sources/rtmpose/projects/rtmpose/rtmpose/"
    "animal_2d_keypoint/rtmpose-m_8xb64-210e_ap10k-256x256.py"
]
custom_imports = dict(
    imports=["external_models.rtmpose.dataset",
             "external_models.rtmpose.model"],
    allow_failed_imports=False,
)

scenario = os.environ["BIRDGAZE_SCENARIO"].upper()
if scenario not in ("FD", "OS", "CS"):
    raise ValueError(f"BIRDGAZE_SCENARIO must be FD, OS or CS; got {scenario!r}")
repo_root = Path(os.environ.get("BIRD_HKE_REPO_ROOT", ".")).resolve()
dataset_name = {
    "FD": "BirdGaze_Full_Dataset",
    "OS": "birdgaze_original_subset",
    "CS": "birdgaze_corrected_subset",
}[scenario]
dataset_root = Path(os.environ.get("BIRDGAZE_DATASET_ROOT", str(repo_root / "BirdGaze_v2" / dataset_name))).resolve()
converted_root = (repo_root / "external_models" / "generated" / scenario).resolve()
assert (dataset_root / "annot" / "train.json").is_file(), f"Missing BirdGaze dataset at {dataset_root}"
assert (converted_root / "train_coco.json").is_file(), "Run convert_birdgaze_to_coco.py first"
assert (converted_root / "val_coco.json").is_file(), "Run convert_birdgaze_to_coco.py first"

# Same keypoint order as existing Bird_HKE: crown, left eye, right eye, beak.
metainfo = dict(
    dataset_name="birdgaze",
    keypoint_info={
        0: dict(name="crown", id=0, color=[255, 255, 0], type="upper", swap=""),
        1: dict(name="left_eye", id=1, color=[0, 255, 0], type="upper", swap="right_eye"),
        2: dict(name="right_eye", id=2, color=[0, 255, 0], type="upper", swap="left_eye"),
        3: dict(name="beak_tip", id=3, color=[255, 0, 0], type="upper", swap=""),
    },
    skeleton_info={},
    joint_weights=[1.0, 1.0, 1.0, 1.0],
    sigmas=[0.025, 0.025, 0.025, 0.025],
)

# Retain the original RTMPose-M architecture and SimCC decoder/loss, but set
# four landmarks and REMOVE pretrained backbone initialization.
model = dict(
    type="BirdGazeRTMPoseEstimator",
    backbone=dict(init_cfg=None, norm_cfg=dict(type="BN")),
    head=dict(out_channels=4),
    test_cfg=dict(flip_test=True),
)
load_from = None
resume = False

max_epochs = 100
train_cfg = dict(max_epochs=100, val_interval=1)
randomness = dict(seed=2026, deterministic=True)
train_batch_size = 8
val_batch_size = 8
optim_wrapper = dict(
    accumulative_counts=8,
    optimizer=dict(type="AdamW", lr=5e-4, weight_decay=0.01),
    paramwise_cfg=None,
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(type="CosineAnnealingLR", by_epoch=True, begin=5, end=100, T_max=95, eta_min=1e-5),
]
auto_scale_lr = dict(enable=False, base_batch_size=64)

# No second-stage pipeline switch or EMA, matching the single-stage Bird_HKE
# protocol. Disable half-body and color augmentation used in RTMPose defaults.
custom_hooks = []

# Preserve orientation metadata for correct validation-time decoding.
# Validation uses images only to select checkpoints; held-out annotated
# videos will be the definitive model comparison.
pack_meta_keys = (
    "id", "img_id", "img_path", "category_id", "crowd_index",
    "ori_shape", "img_shape", "input_size", "input_center", "input_scale",
    "flip", "flip_direction", "flip_indices", "raw_ann_info", "dataset_name",
    "birdgaze_negative_scale", "birdgaze_zero_scale",
    "birdgaze_crop_offset",
)
train_pipeline = [
    dict(type="LoadImage", to_float32=False),
    dict(type="BirdGazeApplySourceCrop"),
    dict(type="GetBBoxCenterScale", padding=1.25),
    dict(type="RandomFlip", direction="horizontal", prob=0.5),
    dict(type="BirdGazeRandomBBoxTransform"),
    dict(type="BirdGazeRestoreSignedScale"),
    dict(type="BirdGazeTopdownAffine", input_size=(256, 256)),
    dict(type="GenerateTarget", encoder=dict(
        type="SimCCLabel", input_size=(256, 256), sigma=(5.66, 5.66),
        simcc_split_ratio=2.0, normalize=False, use_dark=False)),
    dict(type="PackPoseInputs", meta_keys=pack_meta_keys),
]
val_pipeline = [
    dict(type="LoadImage", to_float32=False),
    dict(type="BirdGazeApplySourceCrop"),
    dict(type="GetBBoxCenterScale", padding=1.25),
    dict(type="BirdGazeRestoreSignedScale"),
    dict(type="BirdGazeTopdownAffine", input_size=(256, 256)),
    dict(type="PackPoseInputs", meta_keys=pack_meta_keys),
]
train_dataloader = dict(
    batch_size=8, num_workers=4, persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="BirdGazeCocoDataset", metainfo=metainfo, data_root=None,
        ann_file=str(converted_root / "train_coco.json"),
        data_prefix=dict(img=str(dataset_root / "images") + "/"),
        data_mode="topdown", pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=8, num_workers=4, persistent_workers=True, drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type="BirdGazeCocoDataset", metainfo=metainfo, data_root=None,
        ann_file=str(converted_root / "val_coco.json"),
        data_prefix=dict(img=str(dataset_root / "images") + "/"),
        data_mode="topdown", test_mode=True, pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(type="PCKAccuracy", norm_item="bbox", thr=0.05, prefix="birdgaze")
test_evaluator = val_evaluator
default_hooks = dict(checkpoint=dict(
    save_best="birdgaze/PCK", rule="greater", interval=1, max_keep_ckpts=2))
work_dir = str(repo_root / "Bird_HKE" / "trained_models" /
               "External_Baselines" / "RTMPose_M" / scenario / "seed_2026")
