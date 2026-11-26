#!/usr/bin/env python3
"""Read-only throughput diagnostic: which stage actually caps a training epoch?

Raising the dataloader worker count sped validation up 1.6x but left training
unchanged (25.06 -> 24.44 img/s on an H100 PCIe), which rules out image
decoding as the training bottleneck. Validation uses the same dataset, the same
decode path and the same workers, and reaches ~93 img/s, so the decoder is
clearly capable of feeding training. Guessing twice would be careless, so this
measures the remaining candidate ceilings directly:

  gpu             real fwd/bwd/optimizer steps on one fixed real batch
  gpu_relaxed     same, with determinism relaxed (TF32 on, cudnn.benchmark on)
  epoch_sync      a real epoch through a copy of the production loop that
                  converts each loss scalar with ``Tensor.item()`` per microbatch
  epoch_nosync    identical, but the running totals stay on the GPU and are
                  converted once per epoch

``epoch_sync`` vs ``epoch_nosync`` isolates the per-microbatch host
synchronisations. The production loop calls ``.item()`` once per reported loss
part per microbatch; PoseBH reports four parts and ViTPose none, which is the
only structural difference between the two trainers that matches their 2.7x
speed gap. Accumulating on the GPU cannot change gradients -- the scalar only
feeds the reported metric -- so a large gap here is a free speedup.

``gpu`` vs ``gpu_relaxed`` isolates deterministic kernels and the disabled TF32
paths, which are a reproducibility decision rather than a free speedup.

This module writes nothing and touches no run directory. It is deliberately
absent from ``IMPLEMENTATION_FILES``, so adding it cannot change a live run's
resume fingerprint.

  CUDA_VISIBLE_DEVICES=1 $POSEBH_ENV_PREFIX/bin/python \
      -m comparison_models.common.bottleneck_probe --model posebh-b --scenario FD
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import importlib
import json
import time
from pathlib import Path

import torch


def _relax_determinism():
    """Undo exactly what ``set_determinism`` forces, for measurement only."""
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def measure_gpu(trainer, model, batch, optimizer, scaler, seconds, *, relaxed):
    """Ceiling of the GPU path: a real batch, no dataloader, no decode."""
    if relaxed:
        _relax_determinism()
    recipe = trainer.recipe
    micro = int(recipe["microbatch"])
    group = min(int(recipe["effective_batch"]), len(batch["image"]))
    images = batch["image"].cuda(non_blocking=True)
    targets = batch["target"].cuda(non_blocking=True)
    weights = batch["weight"].cuda(non_blocking=True)
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()

    def body():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(max(1, group // micro)):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, _parts = trainer.loss(model, images, targets, weights, 0)
            scaler.scale(loss * (micro / group)).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       recipe["gradient_clip_norm"])
        scaler.step(optimizer)
        scaler.update()

    for _ in range(2):
        body()
    torch.cuda.synchronize()
    started = time.perf_counter()
    iterations = 0
    while time.perf_counter() - started < seconds:
        body()
        iterations += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not iterations:
        raise RuntimeError("Budget too small to complete a single iteration")
    # The relaxed pass must not leak into the next measurement.
    torch.random.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state)
    return {"images_per_s": round(iterations * group / elapsed, 2),
            "optimizer_steps": iterations, "microbatch": micro,
            "effective_batch": group}


def epoch_variant(trainer, model, optimizer, scaler, loader, dataset_size,
                  scheduler, *, sync):
    """A faithful copy of ``training_epoch`` with the sync behaviour switched.

    ``sync=True`` reproduces production exactly: the per-microbatch
    ``torch.isfinite(loss)`` guard, ``float(loss.item())`` and one
    ``float(parts[key].item())`` per reported loss part.

    ``sync=False`` is the candidate fix, not an idealisation. Gradients, the
    loss, the optimizer and the schedule are untouched; only the *reporting*
    path changes. The finiteness guard is kept but folded into a GPU-side
    accumulator checked once per optimizer step, and the metric totals stay on
    the GPU until the epoch ends. The reported ``train_loss`` is bit-identical
    because the same values are summed, just converted later.
    """
    recipe = trainer.recipe
    model.train()
    optimizer.zero_grad(set_to_none=True)
    count = group_seen = 0
    group_size = min(int(recipe["effective_batch"]), dataset_size)
    running = 0.0 if sync else torch.zeros((), device="cuda")
    parts_total = {key: (0.0 if sync else torch.zeros((), device="cuda"))
                   for key in trainer.loss_parts}
    nonfinite = torch.zeros((), device="cuda")
    started = time.perf_counter()
    for batch in loader:
        images = batch["image"].cuda(non_blocking=True)
        targets = batch["target"].cuda(non_blocking=True)
        weights = batch["weight"].cuda(non_blocking=True)
        micro_count = images.size(0)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, parts = trainer.loss(model, images, targets, weights, 0)
        scaler.scale(loss * (micro_count / group_size)).backward()
        if sync:
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite train loss")
            running += float(loss.detach().item()) * micro_count
            for key in parts_total:
                parts_total[key] += float(parts[key].detach().item()) * micro_count
        else:
            nonfinite += (~torch.isfinite(loss)).to(torch.float32)
            running = running + loss.detach() * micro_count
            for key in parts_total:
                parts_total[key] = parts_total[key] + parts[key].detach() * micro_count
        count += micro_count
        group_seen += micro_count
        if group_seen == group_size:
            if not sync and float(nonfinite) > 0:
                raise FloatingPointError("Nonfinite train loss")
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           recipe["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            group_seen = 0
            group_size = min(int(recipe["effective_batch"]), dataset_size - count)
            if getattr(scheduler, "steps_per_iteration", False):
                scheduler.step()
    total = running if sync else float(running)
    return {"images_per_s": round(count / (time.perf_counter() - started), 2),
            "images": count, "train_loss": total / max(1, count),
            "item_calls_per_microbatch": 1 + len(trainer.loss_parts)
            if sync else 0}


def run(model_name, scenario, examples, seconds):
    module_name = "vitpose" if model_name == "vitpose-b" else "posebh"
    train = importlib.import_module(f"comparison_models.{module_name}.train")
    model_module = importlib.import_module(
        f"comparison_models.{module_name}.model")
    data_module = importlib.import_module(
        "comparison_models.common.birdgaze.dataset" if module_name == "vitpose"
        else "comparison_models.posebh.adapter_dataset")
    from comparison_models.common.training_probe import (
        instantiate, select_real_examples)
    from comparison_models.common.training_utils import dataloader_workers

    trainer = train.get_trainer()
    trainer.ensure_isolated_runtime()
    trainer.require_gpu_capacity()
    trainer.set_determinism()
    _, manifest = train.manifest_for(scenario)
    data_root = Path(manifest["dataset_root"])
    dataset = select_real_examples(
        data_module.__dict__["BirdGazeCropDataset" if module_name == "vitpose"
                              else "BirdGazePoseBHDataset"](data_root, "train"),
        examples)

    report = {"model": model_name, "scenario": scenario,
              "device": torch.cuda.get_device_name(0),
              "examples": len(dataset), "seconds_per_stage": seconds,
              "loss_parts": list(trainer.loss_parts),
              "dataloader_workers": dataloader_workers()}
    print(f"BOTTLENECK PROBE {model_name} / {scenario}  "
          f"device={report['device']}  samples={len(dataset)}  "
          f"workers={report['dataloader_workers']}  "
          f"loss_parts={report['loss_parts']}", flush=True)

    model = instantiate(model_name, model_module).cuda()
    optimizer = trainer.make_optimizer(model)
    scaler = trainer.make_grad_scaler()
    scheduler = trainer.make_scheduler(optimizer, len(dataset))
    fixed_batch = next(iter(trainer.make_epoch_loader(dataset, epoch=0,
                                                     train=True)))

    def fresh():
        trainer.set_determinism()
        torch.manual_seed(int(trainer.recipe["seed"]))

    fresh()
    report["gpu"] = measure_gpu(trainer, model, fixed_batch, optimizer, scaler,
                                seconds, relaxed=False)
    print("gpu             :", json.dumps(report["gpu"]), flush=True)

    fresh()
    report["gpu_relaxed"] = measure_gpu(
        trainer, model, fixed_batch, optimizer, scaler, seconds, relaxed=True)
    print("gpu_relaxed     :", json.dumps(report["gpu_relaxed"]), flush=True)

    fresh()
    report["epoch_nosync"] = epoch_variant(
        trainer, model, optimizer, scaler,
        trainer.make_epoch_loader(dataset, epoch=0, train=True), len(dataset),
        scheduler, sync=False)
    print("epoch_nosync    :", json.dumps(report["epoch_nosync"]), flush=True)

    fresh()
    report["epoch_sync"] = epoch_variant(
        trainer, model, optimizer, scaler,
        trainer.make_epoch_loader(dataset, epoch=0, train=True), len(dataset),
        scheduler, sync=True)
    print("epoch_sync      :", json.dumps(report["epoch_sync"]), flush=True)

    actual = report["epoch_sync"]["images_per_s"]
    report["sync_speedup_available"] = round(
        report["epoch_nosync"]["images_per_s"] / max(actual, 1e-9), 2)
    report["determinism_speedup_available"] = round(
        report["gpu_relaxed"]["images_per_s"] /
        max(report["gpu"]["images_per_s"], 1e-9), 2)
    print("VERDICT:", json.dumps({
        "actual_images_per_s": actual,
        "sync_speedup_available": report["sync_speedup_available"],
        "determinism_speedup_available":
            report["determinism_speedup_available"],
        "reading": "sync_speedup_available > 1.2 means the per-microbatch "
                   "Tensor.item() calls dominate and hoisting them is a "
                   "numerics-preserving speedup. determinism_speedup_available "
                   "> 1.2 means deterministic kernels or disabled TF32 dominate, "
                   "which changes the reproducibility contract and is a "
                   "scientific decision, not a free speedup.",
    }, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("vitpose-b", "posebh-b"), required=True)
    parser.add_argument("--scenario", choices=("FD", "OS", "CS"), default="FD")
    parser.add_argument("--examples", type=int, default=2048)
    parser.add_argument("--seconds", type=int, default=30,
                        help="measurement budget per stage")
    args = parser.parse_args()
    run(args.model, args.scenario, args.examples, args.seconds)


if __name__ == "__main__":
    main()