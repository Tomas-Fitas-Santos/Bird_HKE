#!/usr/bin/env python3
"""Throwaway *real training* integration probe for isolated ViTPose-B/PoseBH-B.

Unlike --smoke-test, exercises the ACTUAL trainer's train_epoch, validation,
AMP/AdamW/scheduler, atomic checkpoint writing, complete optimizer/RNG restore,
best-slot and history recovery, and a second epoch after resuming. Only 12
training and eight validation BirdGaze images are used by default.

Never touches the reserved definitive 100-epoch run folder. Temporary outputs
are created exclusively under the ignored _Training_Probes directory and
automatically deleted after success/failure (unless killed by SIGKILL).
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import gc
import importlib
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from comparison_models.common.preflight import ROOT, SCENARIOS, sha256
from comparison_models.common.birdgaze.dataset import coordinate_valid


def select_real_examples(dataset, count, *, validation=False):
    """Pick real samples with labelled, finite coordinates for a useful probe.

    This selection is ONLY used for the disposable integration test; normal
    100-epoch training includes every source row with its original mask.
    """
    if count < 1:
        raise ValueError("Probe sample count must be positive")
    indexes = []
    for index, row in enumerate(dataset.records):
        try:
            joints = np.asarray(row["joints"], dtype=np.float32)
            if joints.shape[0] != 4 or joints.shape[1] < 2:
                continue
            visible = coordinate_valid(joints, row)
            if validation and row.get("joints_vis") is not None:
                visible &= (
                    np.asarray(row["joints_vis"], dtype=np.float32).reshape(4) > 0
                )
            if not bool(np.isfinite(joints[visible, :2]).all()):
                continue
            if validation and (
                not math.isfinite(float(row["scale"])) or
                float(row["scale"]) == 0
            ):
                continue
            if visible.any():
                indexes.append(index)
                if len(indexes) == count:
                    break
        except (TypeError, ValueError, KeyError):
            continue
    if len(indexes) != count:
        raise RuntimeError(
            f"Only {len(indexes)} eligible BirdGaze examples found, needed {count}. "
            "Do not train before inspecting the annotation data.")
    return Subset(dataset, indexes)


def instantiate(model_name, module):
    if model_name == "vitpose-b":
        return module.ViTPoseB()
    if model_name == "posebh-b":
        return module.PoseBHBirdGazeB()
    raise ValueError(f"Unsupported baseline: {model_name}")


def checkpoint_epoch(train, model, optimizer, scheduler, scaler, root, manifest,
                     epoch, best, best_slot, best_digest, stats, validation):
    """Persist a probe checkpoint through production's exact atomic machinery."""
    score = float(validation["val_pck_005"])
    if not math.isfinite(score):
        raise RuntimeError("Nonfinite validation PCK in integration test")
    updated = score > best
    best = max(best, score)
    record = {
        "epoch": epoch + 1,
        "lr": float(optimizer.param_groups[0]["lr"]),
        **stats,
        **validation,
        "best_val_pck_005": best,
        "peak_cuda_memory_mib": round(
            torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
    }
    if updated:
        best_slot = ("best_slot_b.pth" if best_slot == "best_slot_a.pth"
                     else "best_slot_a.pth")
        train.save_atomic(model.state_dict(), root / best_slot)
        best_digest = sha256(root / best_slot)
    if not getattr(scheduler, "steps_per_iteration", False):
        scheduler.step()
    snapshot = {
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "grad_scaler": scaler.state_dict(),
        "rng": train.rng_snapshot(),
        "recipe_hash": train.manifest_digest(manifest),
        "best_pck": best,
        "best_slot": best_slot,
        "best_sha256": best_digest,
        "metrics": record,
    }
    train.save_atomic(snapshot, root / "checkpoint_last.pth")
    # Intentionally simulate interruption BEFORE writing the metrics JSONL.
    # A faithful resume must recover that record from checkpoint_last.
    train.refresh_best_alias(root, best_slot, best_digest)
    train.recover_history(root, epoch + 1, record)
    return best, best_slot, best_digest


def run_probe(model_name, scenario, train_examples=12, val_examples=8):
    module_name = "vitpose" if model_name == "vitpose-b" else "posebh"
    train = importlib.import_module(f"comparison_models.{module_name}.train")
    model_module = importlib.import_module(f"comparison_models.{module_name}.model")
    data_module = importlib.import_module(
        "comparison_models.common.birdgaze.dataset" if module_name == "vitpose"
        else "comparison_models.posebh.adapter_dataset")
    dataset_type = (
        data_module.BirdGazeCropDataset if module_name == "vitpose"
        else data_module.BirdGazePoseBHDataset
    )
    # The probe exercises the production trainer object itself, so every
    # protocol helper below is the exact one used by a definitive run.
    trainer = train.get_trainer()
    trainer.ensure_isolated_runtime()
    trainer.require_gpu_capacity()
    trainer.set_determinism()
    _, manifest = train.manifest_for(scenario)
    data_root = Path(manifest["dataset_root"])
    train_ds = dataset_type(data_root, "train")
    val_ds = dataset_type(data_root, "val")
    train_set = select_real_examples(train_ds, train_examples)
    val_set = select_real_examples(val_ds, val_examples, validation=True)
    # Probe should use the SAME actual model class, losses and trainer, not
    # a lightweight toy model or a second optimizer implementation.
    scratch_root = ROOT / "Bird_HKE/trained_models/External_Baselines/_Training_Probes"
    scratch_root.mkdir(parents=True, exist_ok=True)
    # Two model/optimizer checkpoints + best slot may consume >2 GiB.
    if shutil.disk_usage(scratch_root).free < 5 * 1024 ** 3:
        raise RuntimeError(f"Insufficient scratch disk (<5 GiB): {scratch_root}")
    print(f"TRAINING INTEGRATION PROBE {model_name} / {scenario}", flush=True)
    print(f"Real samples: {len(train_set)} train, {len(val_set)} val", flush=True)
    print("Definitive run path is NOT used:", manifest["run_dir"], flush=True)

    with tempfile.TemporaryDirectory(
        prefix=f"probe_{module_name}_{scenario}_", dir=scratch_root,
    ) as temporary:
        temp_path = Path(temporary)
        model = instantiate(model_name, model_module).cuda()
        optimizer = trainer.make_optimizer(model)
        scheduler = trainer.make_scheduler(optimizer, len(train_set))
        scaler = trainer.make_grad_scaler()
        trainer.set_determinism()
        peak = []
        # Epoch 1: actually train, validate, persist real optimizer state.
        train_loader = trainer.make_epoch_loader(train_set, epoch=0, train=True)
        stats1 = trainer.training_epoch(
            model, optimizer, scaler, train_loader, len(train_set), 0,
            scheduler=scheduler)
        if not optimizer.state:
            raise RuntimeError("No AdamW optimizer update/state was produced")
        val1 = trainer.validation_epoch(model, val_set, 0)
        best, slot, digest = checkpoint_epoch(
            trainer, model, optimizer, scheduler, scaler, temp_path,
            manifest, 0, -float("inf"), None, None, stats1, val1)
        peak.append(round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1))
        print("Ep1 training and val PASS:", json.dumps(
            {"train_loss": stats1["train_loss"],
             "val_pck_005": val1["val_pck_005"]}), flush=True)

        # Dispose of ALL live model/optimizer tensors. Recover into brand-new
        # instances using the exact production strict resume state contract.
        del model, optimizer, scheduler, scaler, train_loader
        gc.collect()
        torch.cuda.empty_cache()
        state = torch.load(
            temp_path / "checkpoint_last.pth", map_location="cpu",
            weights_only=False,
        )
        if state["recipe_hash"] != trainer.manifest_digest(manifest):
            raise RuntimeError("Resumed checkpoint has incorrect recipe hash")
        if state["epoch"] != 1:
            raise RuntimeError("Resumed checkpoint has incorrect epoch")
        model = instantiate(model_name, model_module).cuda()
        optimizer = trainer.make_optimizer(model)
        scheduler = trainer.make_scheduler(optimizer, len(train_set))
        scaler = trainer.make_grad_scaler()
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["grad_scaler"])
        trainer.refresh_best_alias(
            temp_path, state["best_slot"], state["best_sha256"])
        trainer.recover_history(temp_path, 1, state["metrics"])
        trainer.restore_rng(state["rng"])
        for key, value in model.state_dict().items():
            if not torch.equal(value.detach().cpu(), state["model"][key]):
                raise RuntimeError(f"Resume changed model weight {key}")
        print("Full checkpoint and optimizer/scheduler/AMP/RNG restore PASS",
              flush=True)
        del state

        # Epoch 2: actually take more optimizer steps after the checkpoint.
        # Timed, because this is the steady-state epoch a definitive run
        # repeats, so it is the number the training plan is sized from.
        train_loader = trainer.make_epoch_loader(train_set, epoch=1, train=True)
        epoch_started = time.perf_counter()
        stats2 = trainer.training_epoch(
            model, optimizer, scaler, train_loader, len(train_set), 1,
            scheduler=scheduler)
        train_seconds = time.perf_counter() - epoch_started
        val_started = time.perf_counter()
        val2 = trainer.validation_epoch(model, val_set, 1)
        val_seconds = time.perf_counter() - val_started
        best, slot, digest = checkpoint_epoch(
            trainer, model, optimizer, scheduler, scaler, temp_path,
            manifest, 1, best, slot, digest, stats2, val2)
        peak.append(round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1))
        reloaded = torch.load(
            temp_path / "checkpoint_last.pth", map_location="cpu",
            weights_only=False)
        if reloaded["epoch"] != 2:
            raise RuntimeError("Second epoch checkpoint not committed")
        if not slot or not digest:
            raise RuntimeError("No integrity-verified best model saved")
        if sha256(temp_path / "model_best.pth") != digest:
            raise RuntimeError("Final probe best checkpoint hash mismatch")
        history = [
            json.loads(line)
            for line in (temp_path / "training_history.jsonl")
            .read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if [record["epoch"] for record in history] != [1, 2]:
            raise RuntimeError("Probe history did not reconcile both epochs")
        print("Ep2 training, validation and checkpoint PASS:", json.dumps(
            {"train_loss": stats2["train_loss"],
             "val_pck_005": val2["val_pck_005"]}), flush=True)
        print("INTEGRATION TRAINING PASS:", json.dumps({
            "model": model_name,
            "scenario": scenario,
            "optimizer_steps": "completed in both epochs",
            "epochs": 2,
            "best_val_pck_005": best,
            "peak_allocated_cuda_mib": max(peak),
            "device": torch.cuda.get_device_name(0),
            "throughput": {
                "train_images_per_s": round(len(train_set) / train_seconds, 2),
                "train_seconds_per_epoch": round(train_seconds, 2),
                "val_images_per_s": round(len(val_set) / val_seconds, 2),
                "val_seconds_per_epoch": round(val_seconds, 2),
            },
            "checkpoint": "written, reloaded, and hash-verified",
            "resume": "full state and second epoch PASS",
            "definitive_run_folder_written": False,
            "temporary_probe_outputs": "automatically removed",
        }, sort_keys=True), flush=True)
        # TemporaryDirectory removes test checkpoints after exiting.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("vitpose-b", "posebh-b"),
                        required=True)
    parser.add_argument("--scenario", choices=tuple(SCENARIOS),
                        default="FD")
    parser.add_argument("--train-examples", type=int, default=12)
    parser.add_argument("--val-examples", type=int, default=8)
    args = parser.parse_args()
    run_probe(args.model, args.scenario,
              train_examples=args.train_examples,
              val_examples=args.val_examples)


if __name__ == "__main__":
    main()
