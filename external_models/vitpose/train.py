#!/usr/bin/env python3
"""Isolated single-GPU BirdGaze ViTPose-B training, preflight and strict resume.

Use the dedicated ViTPose virtualenv. The native six-model Bird_HKE trainer,
PoseBH, RTMPose and their environments/checkpoints are never imported/written.
Run from the repository root:
  python -m external_models.vitpose.train --scenario FD --smoke-test
  python -m external_models.vitpose.train --scenario FD --train
  python -m external_models.vitpose.train --scenario FD --resume

Smoke test and recipe checks write nothing. Training writes ONLY inside its
new, dedicated ViTPose_B/<scenario>/seed_2026 output directory.
"""
from __future__ import annotations

# Must be set BEFORE the first CUDA context or CUDA BLAS operation.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from external_models.remaining_baselines.prepare import (
    ROOT, SCENARIOS, prepare, reserve, sha256,
)
from external_models.vitpose.birdgaze_vitpose_b import RECIPE
from external_models.vitpose.dataset import (
    BirdGazeViTPoseDataset, decode_image_coordinates,
)
from external_models.vitpose.model import ViTPoseB, masked_heatmap_mse


def ensure_isolated_runtime():
    # Do not run in an existing native Bird_HKE or RTMPose environment.
    executable = str(Path(sys.executable).resolve()).lower()
    prefix = str(Path(sys.prefix).resolve()).lower()
    if "vitpose" not in executable and "vitpose" not in prefix:
        raise RuntimeError(
            "ViTPose requires its own dedicated Python environment. "
            "See external_models/vitpose/README.md; native Bird_HKE env is forbidden.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Require exactly one visible CUDA GPU (set CUDA_VISIBLE_DEVICES)")
    if RECIPE["backbone"]["pretrained"] is not None:
        raise RuntimeError("Pretrained weights must stay disabled")


def require_idle_gpu():
    """Fail closed before allocating model CUDA memory if any compute PID exists.

    This is a startup check, not a scheduler. It cannot stop another job from
    starting subsequently. Coordinate Titan scheduling across researchers.
    """
    try:
        response = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader,nounits"],
            capture_output=True, check=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Cannot verify GPU is idle with nvidia-smi; refusing") from exc
    busy = [value.strip() for value in response.stdout.splitlines()
            if value.strip().isdigit()]
    if busy:
        raise RuntimeError(
            "Another GPU compute process is running (PIDs: "
            + ", ".join(busy) + "); refusing to interfere with it")


def set_determinism():
    random.seed(RECIPE["seed"])
    np.random.seed(RECIPE["seed"])
    torch.manual_seed(RECIPE["seed"])
    torch.cuda.manual_seed_all(RECIPE["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def runtime_fingerprint():
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
    }


def manifest_for(scenario):
    run, document = prepare("vitpose-b", scenario, check_adapter=True)
    document["training_recipe"] = RECIPE
    document["runtime"] = runtime_fingerprint()
    document["implementation_sha256"] = {
        rel: sha256(ROOT / rel) for rel in (
            "external_models/vitpose/model.py",
            "external_models/vitpose/dataset.py",
            "external_models/vitpose/train.py",
            "external_models/vitpose/birdgaze_vitpose_b.py",
            "external_models/remaining_baselines/prepare.py",
            "external_models/convert_birdgaze_to_coco.py",
            "external_models/rtmpose/geometry.py",
        )
    }
    return run, document


def make_epoch_loader(dataset, epoch, *, train):
    # Epoch-specific random sampler and worker seeds: exactly reproducible
    # after an epoch-boundary resume, independent of previous workers.
    seed = int(RECIPE["seed"]) + epoch * 1009 + (0 if train else 800000)
    sampler_gen = torch.Generator().manual_seed(seed)
    worker_gen = torch.Generator().manual_seed(seed + 29)
    if train:
        sampler = RandomSampler(dataset, generator=sampler_gen)
    else:
        sampler = None
    return DataLoader(
        dataset, batch_size=RECIPE["microbatch"],
        sampler=sampler, shuffle=False, drop_last=False,
        num_workers=2, pin_memory=True,
        generator=worker_gen, persistent_workers=False,
    )


def make_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(), lr=RECIPE["lr"],
        weight_decay=RECIPE["weight_decay"],
        betas=(0.9, 0.999),
    )


def lr_multiplier(epoch):
    warmup = RECIPE["warmup_epochs"]
    min_factor = RECIPE["min_lr"] / RECIPE["lr"]
    if epoch < warmup:
        return 0.1 + 0.9 * epoch / max(1, warmup)
    progress = min(1., (epoch - warmup) /
                   max(1, RECIPE["epochs"] - warmup))
    return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))


def rng_snapshot():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(snapshot):
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(snapshot, dict) or set(snapshot) != required:
        raise RuntimeError("Checkpoint is missing complete RNG state")
    if len(snapshot["torch_cuda"]) != torch.cuda.device_count():
        raise RuntimeError("Checkpoint CUDA device count mismatch")
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"])
    torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def manifest_digest(document):
    return hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write_json_once(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def save_atomic(payload, path):
    # Both temp and final file are under the exclusively locked run root.
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def exclusive_run_lock(folder):
    lock = folder / ".training_active.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Training lock already present: {lock}; inspect running jobs "
            "and recover the lock manually only after confirming it is stale") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
        yield
    finally:
        lock.unlink(missing_ok=True)


def training_epoch(model, optimizer, scaler, loader, dataset_size, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    count, running_loss, group_seen = 0, 0., 0
    group_size = min(RECIPE["effective_batch"], dataset_size)
    tick = time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].cuda(non_blocking=True)
        targets = batch["target"].cuda(non_blocking=True)
        weights = batch["weight"].cuda(non_blocking=True)
        micro_count = images.size(0)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(images)
            loss = masked_heatmap_mse(prediction.float(), targets, weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite train loss epoch={epoch} step={step}")
        # Normalize by actual number of images, including final partial group.
        scaler.scale(loss * (micro_count / group_size)).backward()
        running_loss += float(loss.item()) * micro_count
        count += micro_count
        group_seen += micro_count
        if group_seen == group_size:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           RECIPE["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            group_seen = 0
            group_size = min(RECIPE["effective_batch"], dataset_size - count)
        if step % 100 == 0:
            print(f"Epoch {epoch + 1}: {count}/{dataset_size} images, "
                  f"loss={running_loss / count:.6f}", flush=True)
    if count != dataset_size or group_seen != 0:
        raise RuntimeError(f"Epoch did not consume the entire dataset: {count}/{dataset_size}")
    return {"train_loss": running_loss / max(1, count),
            "train_seconds": round(time.perf_counter() - tick, 2)}


@torch.no_grad()
def validation_epoch(model, dataset, epoch):
    model.eval()
    count = 0
    good = np.zeros(4, dtype=np.int64)
    total = np.zeros(4, dtype=np.int64)
    loader = make_epoch_loader(dataset, epoch, train=False)
    tick = time.perf_counter()
    for batch in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predicted = model(batch["image"].cuda(non_blocking=True))
        heatmaps = predicted.float().cpu().numpy()
        xy = decode_image_coordinates(
            heatmaps, batch["center"].numpy(), batch["scale"].numpy(),
            batch["offset"].numpy(),
        )
        truth = batch["gt"].numpy()
        valid = batch["eval_valid"].numpy().astype(bool)
        scale = np.abs(batch["raw_scale"].numpy())
        threshold = 0.05 * scale[:, None] * 200.
        error = np.linalg.norm(xy - truth, axis=-1)
        # Zero original scale has no valid PCK normalizer. Keep these rows in
        # the dataset/training but exclude them from the validation metric.
        mask = valid & (scale[:, None] > 0) & np.isfinite(error)
        good += ((error <= threshold) & mask).sum(axis=0)
        total += mask.sum(axis=0)
        count += len(truth)
    scores = np.divide(good, total, out=np.full(4, np.nan), where=total > 0)
    value = float(good.sum() / total.sum()) if total.sum() else float("nan")
    if not np.isfinite(value):
        raise RuntimeError("No labeled validation landmarks available for checkpoint selection")
    return {
        "val_pck_005": value,
        "pck_per_keypoint": [None if np.isnan(x) else float(x) for x in scores],
        "val_landmarks": total.tolist(),
        "val_images": count,
        "val_seconds": round(time.perf_counter() - tick, 2),
    }


def gpu_smoke(dataset, validation_dataset):
    """Disposable forward/backward/optimizer step with actual BirdGaze data."""
    print("ViTPose-B real-data GPU preflight (no writes)...", flush=True)
    set_determinism()
    model = ViTPoseB().cuda()
    optim = make_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    sample = next(iter(make_epoch_loader(dataset, epoch=0, train=True)))
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
        validation = next(iter(make_epoch_loader(
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
    torch.cuda.empty_cache()


def run(scenario, mode):
    ensure_isolated_runtime()
    # Check before any CUDA model allocation. Prevent accidentally competing
    # with already-active native training, RTMPose, or other GPU jobs.
    require_idle_gpu()
    set_determinism()
    run_dir, manifest = manifest_for(scenario)
    train_data = BirdGazeViTPoseDataset(Path(manifest["dataset_root"]), "train")
    valid_data = BirdGazeViTPoseDataset(Path(manifest["dataset_root"]), "val")
    for split, ds in (("train", train_data), ("val", valid_data)):
        if len(ds) != manifest["annotations"][split]["count"]:
            raise RuntimeError(f"{split} dataset record count differs from export")
    print(f"ViTPose-B {scenario}: train={len(train_data)} val={len(valid_data)}; "
          f"seed={RECIPE['seed']}; run={run_dir}", flush=True)
    print("Recipe hash:", manifest_digest(manifest), flush=True)
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

    with exclusive_run_lock(run_dir):
        set_determinism()
        model = ViTPoseB().cuda()
        optimizer = make_optimizer(model)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_multiplier)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        begin_epoch = 0
        best = -float("inf")
        if mode == "resume":
            # This pickle file is generated only by our own training process,
            # inside the locked dedicated run directory. Never load arbitrary
            # externally supplied checkpoints with weights_only=False.
            state = torch.load(run_dir / "checkpoint_last.pth",
                               map_location="cpu", weights_only=False)
            if state.get("recipe_hash") != manifest_digest(manifest):
                raise RuntimeError("Checkpoint recipe fingerprint mismatch")
            begin_epoch = int(state["epoch"])
            if begin_epoch < 1 or begin_epoch > RECIPE["epochs"]:
                raise RuntimeError(f"Invalid resume epoch: {begin_epoch}")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["grad_scaler"])
            best = float(state["best_pck"])
            restore_rng(state["rng"])
            print(f"Restored full training state at epoch {begin_epoch}", flush=True)

        if begin_epoch >= RECIPE["epochs"]:
            print("All epochs are already complete. No changes.", flush=True)
            return
        metrics_file = run_dir / "training_history.jsonl"
        for epoch in range(begin_epoch, RECIPE["epochs"]):
            loader = make_epoch_loader(train_data, epoch, train=True)
            stats = training_epoch(model, optimizer, scaler, loader, len(train_data), epoch)
            validation = validation_epoch(model, valid_data, epoch)
            best_updated = validation["val_pck_005"] > best
            best = max(best, validation["val_pck_005"])
            metrics = {
                "epoch": epoch + 1,
                "lr": optimizer.param_groups[0]["lr"],
                **stats, **validation,
                "best_val_pck_005": best,
                "peak_cuda_memory_mib": round(
                    torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
            }
            # Scheduler advance happens before checkpoint, matching restart.
            scheduler.step()
            checkpoint = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "grad_scaler": scaler.state_dict(),
                "rng": rng_snapshot(),
                "recipe_hash": manifest_digest(manifest),
                "best_pck": best,
            }
            save_atomic(checkpoint, run_dir / "checkpoint_last.pth")
            if best_updated:
                save_atomic(model.state_dict(), run_dir / "model_best.pth")
            with metrics_file.open("a", encoding="utf-8") as logs:
                logs.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(json.dumps(metrics, sort_keys=True), flush=True)
        save_atomic(model.state_dict(), run_dir / "final_model.pth")
        print(f"ViTPose-B {scenario} training COMPLETE: {run_dir}", flush=True)


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
