#!/usr/bin/env python3
"""Guarded RTMPose training launcher for BirdGaze (one scenario per work dir).

Runs the pinned upstream MMPose training script. Rejects resume when the exact
training configuration or source/converted annotations changed. Does not gate
on Git branch/commit/repository cleanliness.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from external_models.convert_birdgaze_to_coco import SCENARIOS, sha256


def recipe(scenario: str, root: Path, config: Path, dataset: Path) -> dict:
    converted = root / "external_models" / "generated" / scenario
    paths = {
        "config": config,
        "train_annotations": dataset / "annot" / "train.json",
        "val_annotations": dataset / "annot" / "val.json",
        "train_coco": converted / "train_coco.json",
        "val_coco": converted / "val_coco.json",
        "dataset_adapter": root / "external_models" / "rtmpose" / "dataset.py",
        "crop_geometry": root / "external_models" / "rtmpose" / "geometry.py",
        "annotation_converter": root / "external_models" / "convert_birdgaze_to_coco.py",
    }
    return {
        "scenario": scenario,
        "seed": 2026,
        "files_sha256": {key: sha256(path) for key, path in paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=SCENARIOS.keys())
    parser.add_argument("--resume", action="store_true",
                        help="Only resume in an existing work directory with matching recipe")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config = repo_root / "external_models" / "rtmpose" / "birdgaze_rtmpose_m.py"
    dataset = Path(os.environ.get(
        "BIRDGAZE_DATASET_ROOT",
        str(repo_root / "BirdGaze_v2" / SCENARIOS[args.scenario]))).resolve()
    work = repo_root / "Bird_HKE" / "trained_models" / "External_Baselines" / "RTMPose_M" / args.scenario / "seed_2026"
    manifest_path = work / "birdgaze_training_recipe.json"
    current = recipe(args.scenario, repo_root, config, dataset)
    if args.resume:
        if not manifest_path.is_file():
            parser.error(f"Cannot resume without original recipe manifest: {manifest_path}")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved != current:
            parser.error("Training configuration/annotations changed since initial run: refusing resume.")
        if not list(work.glob("*.pth")) and not (work / "last_checkpoint").exists():
            parser.error("Cannot resume: no checkpoint found in work directory.")
    else:
        if manifest_path.exists() or (work.exists() and any(work.iterdir())):
            parser.error("Work directory already used. Use --resume or move the old run elsewhere.")
        work.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    upstream = repo_root / ".external_pose_sources" / "rtmpose"
    train_script = upstream / "tools" / "train.py"
    if not train_script.is_file():
        parser.error("Pinned upstream RTMPose/MMPose source not found. Run setup_external_pose_sources.sh.")
    env = dict(os.environ)
    env["BIRDGAZE_SCENARIO"] = args.scenario
    env["BIRD_HKE_REPO_ROOT"] = str(repo_root)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(repo_root), str(upstream), env.get("PYTHONPATH", ""))
    )
    command = [
        sys.executable, str(train_script), str(config),
        "--work-dir", str(work),
    ]
    if args.resume:
        command.append("--resume")
    print("Launching RTMPose-M:", args.scenario, "resume" if args.resume else "new run", flush=True)
    print("Recipe SHA256:", hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest(), flush=True)
    subprocess.run(command, cwd=repo_root, env=env, check=True)


if __name__ == "__main__":
    main()
