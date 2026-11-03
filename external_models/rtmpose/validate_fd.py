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
    import mmdet
    from mmdet.utils import register_all_modules as register_mmdet
    from mmpose.utils import register_all_modules as register_mmpose
    from mmengine.config import Config
    from mmengine.utils import import_modules_from_strings
    register_mmdet(init_default_scope=False)
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
    return cfg


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
        if split == "train":
            for flag in ("birdgaze_negative_scale", "birdgaze_zero_scale"):
                match = next((i for i, row in enumerate(ds.data_list) if row.get(flag)), None)
                if match is None:
                    raise RuntimeError(f"No FD training record with {flag}")
                indices.append(match)
        samples = []
        for i in indices:
            packed = ds[i]
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
            f"sample indices={indices}", flush=True
        )
    return collected


def check_gpu(cfg, collected):
    import torch
    from mmengine.dataset import pseudo_collate
    from mmpose.registry import MODELS

    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    if torch.cuda.get_device_capability()[0] >= 10:
        raise RuntimeError("This Torch 2.1 CUDA 12.1 environment cannot run on Blackwell")
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
    pred = outputs[0].pred_instances.keypoints
    if tuple(pred.shape) != (1, 4, 2):
        raise RuntimeError(f"Bad RTMPose validation predictions: {pred.shape}")
    if not torch.as_tensor(pred).isfinite().all():
        raise RuntimeError("Nonfinite RTMPose predictions")
    print("Image validation inference and four-keypoint decoding: PASS", flush=True)


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
