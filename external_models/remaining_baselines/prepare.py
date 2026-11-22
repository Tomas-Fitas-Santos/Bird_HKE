#!/usr/bin/env python3
"""Read-only preflight for ViTPose++-B and PoseBH-B BirdGaze training.

This is intentionally a preparation step, NOT a training launcher. It creates
no directories unless --reserve-run is explicitly passed, and never modifies
Bird_HKE models, upstream checkouts, source datasets or RTMPose artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
SEED = 2026
SCENARIOS = {
    "FD": "BirdGaze_Full_Dataset",
    "OS": "birdgaze_original_subset",
    "CS": "birdgaze_corrected_subset",
}


class Baseline(NamedTuple):
    upstream_folder: str
    revision: str
    run_folder: str
    adapter_config: str


BASELINES = {
    "vitposepp-b": Baseline(
        "vitposepp", "c050ed29112da7704797cc1a65af0234b525010d",
        "ViTPosePP_B", "external_models/vitposepp/birdgaze_vitposepp_b.py"),
    "posebh-b": Baseline(
        "posebh", "8f03438f913b559edb0a2beed54e77d1560edba3",
        "PoseBH_B", "external_models/posebh/birdgaze_posebh_b.py"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_under(child: Path, parent: Path) -> Path:
    """Reject traversal/symlink escape even if target does not yet exist."""
    resolved_child = child.resolve()
    resolved_parent = parent.resolve()
    if resolved_child == resolved_parent or not resolved_child.is_relative_to(resolved_parent):
        raise ValueError(f"Refusing path outside dedicated run root: {child}")
    return resolved_child


def check_git_pin(repo: Path, expected: str) -> str:
    if not (repo / ".git").exists():
        raise ValueError(f"Missing pinned upstream checkout: {repo}")
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
        stderr=subprocess.PIPE,
    ).strip()
    if revision != expected:
        raise ValueError(f"Pinned upstream revision mismatch: {revision}, expected {expected}")
    # Upstream checkouts must stay clean; overlays live in our own tree.
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True,
        stderr=subprocess.PIPE,
    ).strip()
    if dirty:
        raise ValueError(f"Upstream checkout has local modifications: {repo}")
    return revision


def validate_exports(dataset: Path, exports: Path) -> dict:
    """Confirm converted records preserve the existing BirdGaze data contract."""
    evidence = {}
    for split in ("train", "val"):
        source = dataset / "annot" / f"{split}.json"
        converted = exports / f"{split}_coco.json"
        if not source.is_file() or not converted.is_file():
            raise ValueError(f"Missing source/COCO annotations for {split}; run converter first")
        with source.open(encoding="utf-8") as f:
            original = json.load(f)
        with converted.open(encoding="utf-8") as f:
            coco = json.load(f)
        if not isinstance(original, list) or not isinstance(coco, dict):
            raise ValueError(f"Unexpected {split} annotation structure")
        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        if not original or len(original) != len(images) or len(original) != len(annotations):
            raise ValueError(f"{split}: missing or filtered BirdGaze records")
        keys = {"birdgaze_source_bbox", "birdgaze_zero_scale", "birdgaze_negative_scale"}
        for index, item in enumerate(annotations):
            if not isinstance(item, dict) or not keys.issubset(item):
                raise ValueError(f"{split}: stale crop metadata at item {index}")
            if len(item.get("keypoints", [])) != 12:
                raise ValueError(f"{split}: expected exactly four (x, y, visibility) keypoints")
        evidence[split] = {
            "count": len(original),
            "source_sha256": sha256(source),
            "converted_sha256": sha256(converted),
        }
    return evidence


def validate_os_cs_distinction(root: Path) -> None:
    """Block the known case where OS/CS differ only in a dropped crop bbox."""
    os_export = root / "external_models/generated/OS"
    cs_export = root / "external_models/generated/CS"
    os_data = root / "BirdGaze_v2" / SCENARIOS["OS"] / "annot"
    cs_data = root / "BirdGaze_v2" / SCENARIOS["CS"] / "annot"
    for split in ("train", "val"):
        a, b = os_export / f"{split}_coco.json", cs_export / f"{split}_coco.json"
        x, y = os_data / f"{split}.json", cs_data / f"{split}.json"
        if all(path.is_file() for path in (a, b, x, y)):
            if sha256(x) != sha256(y) and sha256(a) == sha256(b):
                raise ValueError(
                    f"{split}: OS/CS source annotations differ but COCO exports match; "
                    "regenerate with the corrected crop-aware exporter")


def prepare(model: str, scenario: str, root: Path = ROOT, dataset_root: Path | None = None,
            *, check_upstream: bool = True, check_adapter: bool = False) -> tuple[Path, dict]:
    spec = BASELINES[model]
    root = root.resolve()
    dataset = (dataset_root or root / "BirdGaze_v2" / SCENARIOS[scenario]).resolve()
    work = ensure_under(root / "Bird_HKE/trained_models/External_Baselines"
                        / spec.run_folder / scenario / f"seed_{SEED}",
                        root / "Bird_HKE/trained_models/External_Baselines")
    upstream = root / ".external_pose_sources" / spec.upstream_folder
    if check_upstream:
        check_git_pin(upstream, spec.revision)
    evidence = validate_exports(dataset, root / "external_models/generated" / scenario)
    if scenario in ("OS", "CS") and dataset_root is None:
        validate_os_cs_distinction(root)
    adapter = root / spec.adapter_config
    if check_adapter and not adapter.is_file():
        raise ValueError(
            f"Training adapter not yet implemented: {adapter}. "
            "Do not launch an unadapted multi-dataset upstream trainer.")
    manifest = {
        "schema_version": 1,
        "model": model,
        "scenario": scenario,
        "seed": SEED,
        "keypoints": ["crown", "left_eye", "right_eye", "beak_tip"],
        "input_size": [256, 256],
        "init": "random_no_pretrained_weights",
        "upstream": {"folder": spec.upstream_folder, "revision": spec.revision},
        "dataset_root": str(dataset),
        "annotations": evidence,
        "adapter_config": spec.adapter_config,
        "adapter_sha256": sha256(adapter) if adapter.is_file() else None,
        "run_dir": str(work),
        "training_ready": adapter.is_file(),
    }
    return work, manifest


def reserve(work: Path, manifest: dict, root: Path = ROOT) -> Path:
    """Create a new model-specific run exactly once; existing work is immutable."""
    dedicated_root = root / "Bird_HKE/trained_models/External_Baselines"
    ensure_under(work, dedicated_root)
    if not manifest["training_ready"]:
        raise ValueError("Cannot reserve: the model-specific training adapter is not implemented")
    if work.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {work}")
    work.mkdir(parents=True, exist_ok=False)
    path = work / "training_recipe.json"
    try:
        with path.open("x", encoding="utf-8") as output:
            json.dump(manifest, output, sort_keys=True, indent=2)
            output.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        work.rmdir()
        raise
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(BASELINES))
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--reserve-run", action="store_true",
                        help="Reserve a fresh run ONLY after model adapter exists")
    args = parser.parse_args()
    try:
        run, manifest = prepare(
            args.model, args.scenario, dataset_root=args.dataset_root,
            check_adapter=args.reserve_run,
        )
        if args.reserve_run:
            path = reserve(run, manifest)
            print(f"Reserved run: {path}")
        else:
            print("Read-only preflight completed (no training or writes).")
        print(json.dumps(manifest, indent=2, sort_keys=True))
    except (ValueError, FileExistsError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
