#!/usr/bin/env python3
"""Guarded single-GPU RTMPose-M FD training using pinned official MMPose.

This launcher does not change Git branches or touch native Bird_HKE runs.
It checks the exact BirdGaze training recipe before creating a work directory,
runs a disposable GPU preflight, and resumes only from an existing compatible
epoch checkpoint. Use the isolated RTMPose overlay interpreter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from importlib import metadata

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_models.convert_birdgaze_to_coco import SCENARIOS, sha256

PINNED_MMPOSE_REVISION = "759b39c13fea6ba094afc1fa932f51dc1b11cbf9"


def require_pinned_upstream(root: Path) -> Path:
    upstream = root / ".external_pose_sources" / "rtmpose"
    script = upstream / "tools" / "train.py"
    if not script.is_file():
        raise RuntimeError(
            "Pinned MMPose checkout absent; run bash setup_external_pose_sources.sh")
    actual = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual != PINNED_MMPOSE_REVISION:
        raise RuntimeError(
            f"Wrong MMPose source revision: {actual}, expected {PINNED_MMPOSE_REVISION}")
    return upstream


def runtime_versions() -> dict:
    required = ("torch", "torchvision", "numpy", "mmcv", "mmengine",
                "mmpose")
    try:
        return {name: metadata.version(name) for name in required}
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "RTMPose dependencies are missing; activate the isolated "
            "environment created with bash setup_rtmpose_torch27.sh --build") from exc


def recipe(scenario: str, root: Path, config: Path, dataset: Path) -> dict:
    converted = root / "external_models" / "generated" / scenario
    paths = {
        "config": config,
        "source_train": dataset / "annot" / "train.json",
        "source_val": dataset / "annot" / "val.json",
        "coco_train": converted / "train_coco.json",
        "coco_val": converted / "val_coco.json",
        "converter": root / "external_models" / "convert_birdgaze_to_coco.py",
        "dataset_adapter": root / "external_models" / "rtmpose" / "dataset.py",
        "crop_geometry": root / "external_models" / "rtmpose" / "geometry.py",
        "model_adapter": root / "external_models" / "rtmpose" / "model.py",
        "accumulation_plan": root / "external_models" / "rtmpose" / "accumulation.py",
        "optimizer_protocol": root / "external_models" / "rtmpose" / "protocol.py",
        "upstream_rtmpose_m": (
            root / ".external_pose_sources" / "rtmpose" / "projects" / "rtmpose"
            / "rtmpose" / "animal_2d_keypoint"
            / "rtmpose-m_8xb64-210e_ap10k-256x256.py"
        ),
    }
    return {
        "model": "RTMPose-M",
        "scenario": scenario,
        "seed": 2026,
        "mmpose_source_revision": PINNED_MMPOSE_REVISION,
        "runtime_packages": runtime_versions(),
        "files_sha256": {key: sha256(path) for key, path in paths.items()},
    }


def check_export(scenario: str, root: Path, dataset: Path) -> None:
    for split in ("train", "val"):
        with (dataset / "annot" / f"{split}.json").open(encoding="utf-8") as src:
            source = json.load(src)
        path = root / "external_models" / "generated" / scenario / f"{split}_coco.json"
        with path.open(encoding="utf-8") as f:
            converted = json.load(f)
        n = len(source)
        if len(converted["images"]) != n or len(converted["annotations"]) != n:
            raise RuntimeError(
                f"{split}: COCO conversion lost records; rerun the converter.")
        if not all("birdgaze_source_bbox" in a and "birdgaze_zero_scale" in a
                   and "birdgaze_negative_scale" in a
                   for a in converted["annotations"]):
            raise RuntimeError(
                f"{split}: stale conversion missing BirdGaze crop metadata; "
                "rerun the converter.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="FD", choices=SCENARIOS.keys())
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", action="store_true",
                       help="Resume the latest matching epoch checkpoint")
    modes.add_argument("--retry-start", action="store_true",
                       help="Restart a previously failed run with NO checkpoint")
    modes.add_argument("--check-only", action="store_true",
                       help="Run real FD data+GPU preflight, create no work dir")
    args = parser.parse_args()

    if args.scenario != "FD":
        parser.error("Only RTMPose-M FD is validated in this phase")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Only single-process single-GPU training is supported")
    config = ROOT / "external_models/rtmpose/birdgaze_rtmpose_m.py"
    dataset = Path(os.environ.get(
        "BIRDGAZE_DATASET_ROOT",
        str(ROOT / "BirdGaze_v2" / SCENARIOS[args.scenario])
    )).resolve()
    work = (ROOT / "Bird_HKE/trained_models/External_Baselines/RTMPose_M"
            / args.scenario / "seed_2026")
    manifest_path = work / "birdgaze_training_recipe.json"

    try:
        upstream = require_pinned_upstream(ROOT)
        check_export(args.scenario, ROOT, dataset)
        current = recipe(args.scenario, ROOT, config, dataset)
    except (RuntimeError, FileNotFoundError) as exc:
        parser.error(str(exc))

    env = dict(os.environ)
    env["BIRDGAZE_SCENARIO"] = args.scenario
    env["BIRD_HKE_REPO_ROOT"] = str(ROOT)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT), str(upstream), env.get("PYTHONPATH", "")))
    print("RTMPose-M FD recipe SHA256:",
          hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest(),
          flush=True)

    # Perform the real-data/GPU check before any definitive run directory is
    # created. The checks are isolated to their own Python process.
    subprocess.run(
        [sys.executable, "-m", "external_models.rtmpose.validate_fd",
         "--mode", "gpu"],
        cwd=ROOT, env=env, check=True,
    )
    if args.check_only:
        print("Preflight completed; no training files created.", flush=True)
        return

    if args.resume or args.retry_start:
        if not manifest_path.is_file():
            parser.error("Cannot resume/retry: original recipe manifest is absent")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved != current:
            parser.error("Original config/source/annotations/runtime have changed; "
                         "cannot resume this training run")
        have_checkpoint = (
            (work / "last_checkpoint").is_file()
            or any(work.glob("*.pth"))
        )
        if args.resume and not (work / "last_checkpoint").is_file():
            parser.error("Cannot resume: last_checkpoint pointer is missing")
        if args.retry_start and have_checkpoint:
            parser.error("A checkpoint already exists: use --resume, not --retry-start")
    else:
        if work.exists() and any(work.iterdir()):
            parser.error("Training directory already used: resume or explicitly retry")
        work.mkdir(parents=True, exist_ok=True)
        temp = manifest_path.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(manifest_path)

    command = [
        sys.executable, str(upstream / "tools" / "train.py"), str(config),
        "--work-dir", str(work),
    ]
    if args.resume:
        command.append("--resume")
    print(f"Starting {'resume' if args.resume else 'new'} RTMPose-M FD run.",
          "Checkpoint directory:", work, flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
