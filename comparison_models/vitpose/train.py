#!/usr/bin/env python3
"""Isolated single-GPU BirdGaze ViTPose-B training, preflight and strict resume.

Use the dedicated ViTPose virtualenv. The native six-model Bird_HKE trainer,
PoseBH, RTMPose and their environments/checkpoints are never imported/written.
Run from the repository root:
  python -m comparison_models.vitpose.train --scenario FD --smoke-test
  python -m comparison_models.vitpose.train --scenario FD --train
  python -m comparison_models.vitpose.train --scenario FD --resume

Smoke test and recipe checks write nothing. Training writes ONLY inside its
new, dedicated ViTPose_B/<scenario>/seed_2026 output directory.

Accumulation, AMP, checkpointing, strict resume and PCK validation are shared
with PoseBH-B through ``comparison_models.common.training_utils``. Only the
architecture, its loss and the real-data smoke test live here.
"""
from __future__ import annotations

# Must be set BEFORE the first CUDA context or CUDA BLAS operation.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from comparison_models.common.birdgaze.dataset import (
    BirdGazeCropDataset, decode_image_coordinates,
)
from comparison_models.common.losses import masked_heatmap_mse
from comparison_models.common.preflight import (
    ROOT, SCENARIOS, prepare, reserve, sha256,
)
from comparison_models.common.training_utils import (
    SingleTaskTrainer, require_idle_gpu, runtime_fingerprint,
)
from comparison_models.vitpose.model import ViTPoseB
from comparison_models.vitpose.recipe import RECIPE

#: Files whose bytes define this experiment; hashed into every run manifest.
IMPLEMENTATION_FILES = (
    "comparison_models/vitpose/model.py",
    "comparison_models/vitpose/train.py",
    "comparison_models/vitpose/recipe.py",
    "comparison_models/common/preflight.py",
    "comparison_models/common/training_utils.py",
    "comparison_models/common/losses.py",
    "comparison_models/common/transformer.py",
    "comparison_models/common/birdgaze/conversion.py",
    "comparison_models/common/birdgaze/dataset.py",
    "comparison_models/common/birdgaze/geometry.py",
)


class ViTPoseBTrainer(SingleTaskTrainer):
    """Standard single-task ViTPose-B: ViT-B plus the zero-deconv head."""

    def __init__(self):
        super().__init__(
            RECIPE, name="ViTPose-B", environment_marker="vitpose",
            documentation="comparison_models/vitpose/README.md")

    def ensure_recipe_is_from_scratch(self):
        if RECIPE["backbone"]["pretrained"] is not None:
            raise RuntimeError("Pretrained weights must stay disabled")

    def new_model(self):
        return ViTPoseB()

    def loss(self, model, images, targets, weights, epoch):
        prediction = model(images)
        return masked_heatmap_mse(prediction.float(), targets, weights), {}

    def validation_heatmaps(self, model, images):
        return model(images)


def manifest_for(scenario):
    run, document = prepare("vitpose-b", scenario, check_adapter=True)
    document["training_recipe"] = RECIPE
    document["runtime"] = runtime_fingerprint()
    document["implementation_sha256"] = {
        rel: sha256(ROOT / rel) for rel in IMPLEMENTATION_FILES}
    return run, document


def get_trainer():
    """Entry point used by the shared disposable integration probe."""
    return ViTPoseBTrainer()


def training_epoch(model, optimizer, scaler, loader, dataset_size, epoch,
                   scheduler=None):
    """Module-level entry point used by the disposable integration probe."""
    return ViTPoseBTrainer().training_epoch(
        model, optimizer, scaler, loader, dataset_size, epoch,
        scheduler=scheduler)


@torch.no_grad()
def validation_epoch(model, dataset, epoch):
    return ViTPoseBTrainer().validation_epoch(model, dataset, epoch)


def gpu_smoke(dataset, validation_dataset):
    """Disposable forward/backward/optimizer step with actual BirdGaze data."""
    trainer = ViTPoseBTrainer()
    print("ViTPose-B real-data GPU preflight (no writes)...", flush=True)
    trainer.set_determinism()
    torch.cuda.reset_peak_memory_stats()
    smoke_start = time.perf_counter()
    model = ViTPoseB().cuda()
    optim = trainer.make_optimizer(model)
    scaler = trainer.make_grad_scaler()
    sample = next(iter(trainer.make_epoch_loader(dataset, epoch=0, train=True)))
    model.train()
    if float(sample["weight"].sum()) <= 0:
        raise RuntimeError("Smoke batch has no labeled landmarks; select a supervised batch")
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(sample["image"].cuda(non_blocking=True))
        loss = masked_heatmap_mse(
            output.float(), sample["target"].cuda(non_blocking=True),
            sample["weight"].cuda(non_blocking=True))
    assert tuple(output.shape[1:]) == (4, 64, 64)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite ViTPose-B real-data smoke-test loss")
    scaler.scale(loss).backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all()
               and bool((p.grad.abs().sum() > 0).item())
               for p in model.parameters()):
        raise RuntimeError("ViTPose-B smoke test produced no nonzero finite gradients")
    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optim)
    scaler.update()
    model.eval()
    with torch.no_grad():
        validation = next(iter(trainer.make_epoch_loader(
            validation_dataset, epoch=0, train=False)))
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            test_output = model(validation["image"].cuda(non_blocking=True))
        test_xy = decode_image_coordinates(
            test_output.float().cpu().numpy(),
            validation["center"].numpy(), validation["scale"].numpy(),
            validation["offset"].numpy(),
        )
        if not np.isfinite(test_xy).all():
            raise RuntimeError("ViTPose-B validation decoding produced invalid coordinates")
    print(f"GPU preflight PASS; model={sum(p.numel() for p in model.parameters()):,} "
          f"parameters, first loss={loss.item():.7f}", flush=True)
    del model, optim, scaler, sample, output, loss, validation, test_output, test_xy
    peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    elapsed = time.perf_counter() - smoke_start
    print(f"GPU smoke resource profile: peak_allocated_mib={peak_mib:.1f} "
          f"elapsed_s={elapsed:.2f}", flush=True)
    torch.cuda.empty_cache()


def run(scenario, mode):
    trainer = ViTPoseBTrainer()
    trainer.ensure_isolated_runtime()
    # Check before any CUDA model allocation. Prevent accidentally competing
    # with already-active native training, RTMPose, or other GPU jobs.
    require_idle_gpu()
    trainer.set_determinism()
    run_dir, manifest = manifest_for(scenario)
    train_data = BirdGazeCropDataset(Path(manifest["dataset_root"]), "train")
    valid_data = BirdGazeCropDataset(Path(manifest["dataset_root"]), "val")
    for split, ds in (("train", train_data), ("val", valid_data)):
        if len(ds) != manifest["annotations"][split]["count"]:
            raise RuntimeError(f"{split} dataset record count differs from export")
    print(f"ViTPose-B {scenario}: train={len(train_data)} val={len(valid_data)}; "
          f"seed={RECIPE['seed']}; run={run_dir}", flush=True)
    print("Recipe hash:", trainer.manifest_digest(manifest), flush=True)
    gpu_smoke(train_data, valid_data)
    if mode == "smoke":
        print("All checks passed. No directory or checkpoint was created.", flush=True)
        return

    if mode == "train":
        reserve(run_dir, manifest)
    elif mode == "resume":
        manifest_path = run_dir / "training_recipe.json"
        if not manifest_path.is_file():
            raise RuntimeError("Cannot resume without the original run recipe")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError(
                "Refusing resume: model, source, data, environment, or config changed")
        if not (run_dir / "checkpoint_last.pth").is_file():
            raise RuntimeError("Cannot resume without a saved epoch checkpoint")
    else:
        raise ValueError(mode)

    trainer.train(run_dir=run_dir, manifest=manifest, train_data=train_data,
                  valid_data=valid_data, mode=mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke-test", action="store_true",
                       help="Real-data CUDA forward/backward, NO output writes")
    modes.add_argument("--train", action="store_true",
                       help="Start a fresh isolated 100-epoch run")
    modes.add_argument("--resume", action="store_true",
                       help="Resume strict matching epoch checkpoint")
    args = parser.parse_args()
    mode = "smoke" if args.smoke_test else "train" if args.train else "resume"
    try:
        run(args.scenario, mode)
    except (RuntimeError, ValueError, FileNotFoundError,
            subprocess.CalledProcessError, FileExistsError) as exc:
        parser.exit(1, f"ViTPose-B refused to launch: {exc}\n")


if __name__ == "__main__":
    main()
