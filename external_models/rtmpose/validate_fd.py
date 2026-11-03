#!/usr/bin/env python3
"""Read-only RTMPose-M FD data audit and disposable GPU training preflight.

The GPU mode does a fresh model initialization, real-data forward/backward,
optimizer step, and validation inference IN MEMORY. It never calls the
training runner, creates a checkpoint, or modifies the definitive work dir.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def build_cfg(root):
    os.environ["BIRDGAZE_SCENARIO"] = "FD"
    os.environ["BIRD_HKE_REPO_ROOT"] = str(root)
    from mmpose.utils import register_all_modules as register_mmpose
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings
    register_mmpose(init_default_scope=True)
    cfg = Config.fromfile(str(root / "external_models/rtmpose/birdgaze_rtmpose_m.py"))
    import_modules_from_strings(**cfg.custom_imports)
    if cfg.model.type != "BirdGazeRTMPoseEstimator":
        raise RuntimeError("Wrong pose model class")
    if cfg.model.backbone.init_cfg is not None or cfg.get("load_from"):
        raise RuntimeError("Pretrained weights are not allowed in this comparison")
    if cfg.model.head.out_channels != 4:
        raise RuntimeError("Expected four keypoints")
    if cfg.optim_wrapper.accumulative_counts != 8:
        raise RuntimeError("Expected effective batch 64 on one GPU")
    if cfg.train_dataloader.batch_size != 8:
        raise RuntimeError("Unexpected microbatch size")
    if cfg.train_cfg.max_epochs != 100:
        raise RuntimeError("Expected a 100-epoch run")
    # The actual Runner constructs val_evaluator before epoch 1. Guard that
    # train-only setup here so inherited AP-10K ann_file keys cannot survive.
    from mmpose.registry import METRICS
    for name in ("val_evaluator", "test_evaluator"):
        metric_cfg = cfg[name]
        unexpected = set(metric_cfg) - {"type", "norm_item", "thr", "prefix", "_scope_"}
        if unexpected:
            raise RuntimeError(
                f"{name} has unexpected inherited fields: {sorted(unexpected)}")
        metric = METRICS.build(metric_cfg)
        if metric.__class__.__name__ != "PCKAccuracy":
            raise RuntimeError(f"Unexpected {name}: {metric.__class__.__name__}")
    if cfg.default_hooks.checkpoint.save_best != "birdgaze/PCK":
        raise RuntimeError("Checkpoint selector does not match val metric")
    print("Validation/test PCKAccuracy evaluators: PASS", flush=True)
    return cfg


def find_sample_with_flag(dataset, flag: str):
    """Use MMEngine's public dataset API; data_list is empty when serialized.

    BaseDataset defaults to serialize_data=True, calls _serialize_data()
    during full_init(), and clears data_list. get_data_info() restores the
    selected entry from data_bytes and also supports unserialized datasets.
    """
    for index in range(len(dataset)):
        if bool(dataset.get_data_info(index).get(flag, False)):
            return index
    return None


def check_data(cfg, root):
    import torch
    from mmpose.registry import DATASETS
    collected = {}
    for split in ("train", "val"):
        dcfg = cfg.train_dataloader.dataset if split == "train" else cfg.val_dataloader.dataset
        ds = DATASETS.build(dcfg)
        path = root / "external_models/generated/FD" / f"{split}_coco.json"
        with path.open(encoding="utf-8") as f:
            expected = json.load(f)
        if len(ds) != len(expected["images"]):
            raise RuntimeError(
                f"{split}: MMPose loaded {len(ds)} of {len(expected['images'])} images; "
                "records were lost during COCO filtering"
            )
        indices = [0]
        # FD train contains negative and zero scales; FD validation contains
        # negative scales too. Exercise each unusual affine case directly.
        # Check both the converted COCO flags and their survival through
        # serialized MMPose dataset loading (without turning serialization off).
        flags = ("birdgaze_negative_scale", "birdgaze_zero_scale") if split == "train" else ("birdgaze_negative_scale",)
        expected_counts = {}
        for flag in flags:
            expected_count = sum(bool(a.get(flag, False)) for a in expected["annotations"])
            if expected_count == 0:
                raise RuntimeError(
                    f"FD {split} export has no {flag}; check converter/source data")
            match = find_sample_with_flag(ds, flag)
            if match is None:
                raise RuntimeError(
                    f"FD {split} COCO export contains {expected_count} {flag} "
                    f"annotations, but MMPose serialized dataset exposes none")
            record = ds.get_data_info(match)
            if not record.get("raw_ann_info", {}).get(flag, False):
                raise RuntimeError(
                    f"FD {split} index {match} lost {flag} in raw annotation")
            expected_counts[flag] = expected_count
            indices.append(match)
        samples = []
        for i in indices:
            packed = ds[i]
            raw = ds.get_data_info(i)
            for flag in flags:
                if bool(packed["data_samples"].metainfo.get(flag, False)) != bool(raw.get(flag, False)):
                    raise RuntimeError(
                        f"FD {split} index {i}: {flag} lost during pipeline packing")
            if tuple(packed["inputs"].shape) != (3, 256, 256):
                raise RuntimeError(f"Incorrect image dimensions at {split} index {i}")
            annotation = packed["data_samples"].gt_instances.keypoints
            if tuple(annotation.shape) != (1, 4, 2):
                raise RuntimeError(f"Expected exactly four keypoints: {annotation.shape}")
            if split == "train":
                labels = packed["data_samples"].gt_instance_labels
                if labels.keypoint_weights.numel() != 4:
                    raise RuntimeError("Incorrect number of SimCC target weights")
                if labels.keypoint_x_labels.shape[-1] != 512:
                    raise RuntimeError("Expected 512-bin SimCC X labels")
            samples.append(packed)
        collected[split] = (ds, samples)
        print(
            f"{split}: loaded={len(ds):,}, expected={len(expected['images']):,}, "
            f"sample indices={indices}, export flags={expected_counts}", flush=True
        )
    return collected


def check_gpu(cfg, collected):
    import torch
    from mmengine.dataset import pseudo_collate
    from mmpose.registry import MODELS

    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    if torch.cuda.get_device_capability()[0] >= 10:
        raise RuntimeError("This RTMPose-M single RTX 3090 configuration is not tested on Blackwell")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    model = MODELS.build(cfg.model)
    model.init_weights()
    model = model.cuda()
    model.train()

    examples = collected["train"][1]  # ordinary, negative-scale, zero-scale
    # Three real FD samples rather than synthetic tensors.
    batch = pseudo_collate(examples)
    batch = model.data_preprocessor(batch, training=True)
    losses = model.forward(
        batch["inputs"], batch["data_samples"], mode="loss")
    poseloss = losses.get("loss_kpt")
    if poseloss is None or not torch.isfinite(poseloss).all():
        raise RuntimeError(f"RTMPose SimCC loss missing/nonfinite: {losses}")
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    opt.zero_grad(set_to_none=True)
    poseloss.backward()
    grads = [
        p.grad for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError("Missing/nonfinite RTMPose gradients")
    opt.step()
    print("Real FD train forward/backward/optimizer step: PASS",
          "loss=", poseloss.item(), flush=True)

    model.eval()
    val_items = collected["val"][1]
    val_batch = pseudo_collate(val_items)
    val_batch = model.data_preprocessor(val_batch, training=False)
    with torch.no_grad():
        outputs = model.predict(val_batch["inputs"], val_batch["data_samples"])
    if len(outputs) != len(val_items):
        raise RuntimeError("Validation inference lost samples")
    for i, output in enumerate(outputs):
        pred = output.pred_instances.keypoints
        if tuple(pred.shape) != (1, 4, 2):
            raise RuntimeError(f"Bad RTMPose validation predictions at {i}: {pred.shape}")
        if not torch.as_tensor(pred).isfinite().all():
            raise RuntimeError(f"Nonfinite RTMPose predictions at {i}")
    if not outputs[1].metainfo.get("birdgaze_negative_scale", False):
        raise RuntimeError("Negative-scale validation orientation metadata was lost")
    print("Image validation inference and signed-scale decoding: PASS", flush=True)

    # Exercise the SAME metric object and PCK key used by Runner's
    # CheckpointHook, with genuine BirdGaze images and network predictions.
    import math
    from mmpose.registry import METRICS
    evaluator = METRICS.build(cfg.val_evaluator)
    evaluator.process([], [sample.to_dict() for sample in outputs])
    scores = evaluator.evaluate(len(outputs))
    value = scores.get("birdgaze/PCK")
    if value is None or not math.isfinite(float(value)):
        raise RuntimeError(f"Missing or nonfinite checkpoint PCK: {scores}")
    print(f"Real FD validation PCK evaluator: PASS birdgaze/PCK={value}",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("data", "gpu"), default="data")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = build_cfg(root)
    samples = check_data(cfg, root)
    if args.mode == "gpu":
        check_gpu(cfg, samples)
    print(f"RTMPose-M FD {args.mode} preflight: PASS", flush=True)
    print("No training run, checkpoints or work directories were modified.", flush=True)


if __name__ == "__main__":
    main()
