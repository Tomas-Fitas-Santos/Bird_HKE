#!/usr/bin/env python3
"""Verify that BirdGaze CS contains corrected *keypoint coordinates*, not just boxes.

This reads the original OS/CS train/val JSON files and never modifies them.
Pair samples by image filename instead of record position, and report exactly
which landmark coordinates change. Fails if the subsets are not paired or if
no coordinate corrections are present in either split.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

OS_DIR = "birdgaze_original_subset"
CS_DIR = "birdgaze_corrected_subset"
KEYPOINTS = ("crown", "left_eye", "right_eye", "beak_tip")


def load_annotations(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {path}")
    result = {}
    for record in records:
        name = str(record["image"]).replace("\\", "/")
        if name in result:
            raise ValueError(f"Duplicate image annotation {name!r} in {path}")
        joints = record["joints"]
        if len(joints) != len(KEYPOINTS):
            raise ValueError(f"{path}: {name} has {len(joints)} rather than 4 joints")
        result[name] = record
    return result


def compare_annotation_sets(original: dict, corrected: dict) -> dict:
    if set(original) != set(corrected):
        os_only = sorted(set(original) - set(corrected))
        cs_only = sorted(set(corrected) - set(original))
        raise ValueError(
            f"OS/CS refer to different images: OS-only={len(os_only)} "
            f"(examples={os_only[:3]}), CS-only={len(cs_only)} "
            f"(examples={cs_only[:3]})"
        )
    per_keypoint = collections.Counter()
    changed_images = 0
    bbox_changed = 0
    examples = []
    for name in original:
        old, new = original[name], corrected[name]
        if old.get("bbox") != new.get("bbox"):
            bbox_changed += 1
        changes = []
        for i, landmark in enumerate(KEYPOINTS):
            a, b = old["joints"][i][:2], new["joints"][i][:2]
            try:
                a = tuple(float(x) for x in a)
                b = tuple(float(x) for x in b)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid coordinate for {name} {landmark}") from exc
            if not all(math.isfinite(x) for x in a + b):
                raise ValueError(f"Nonfinite coordinate for {name} {landmark}")
            if a != b:
                per_keypoint[landmark] += 1
                changes.append({"landmark": landmark, "original": a, "corrected": b})
        if changes:
            changed_images += 1
            if len(examples) < 5:
                examples.append({"image": name, "changes": changes})
    return {
        "images": len(original),
        "images_with_changed_keypoint_coordinates": changed_images,
        "changed_coordinates_by_landmark": {key: per_keypoint[key] for key in KEYPOINTS},
        "images_with_changed_bounding_boxes": bbox_changed,
        "examples": examples,
    }


def verify_scenarios(root: Path) -> dict:
    output = {}
    for split in ("train", "val"):
        os_path = root / OS_DIR / "annot" / f"{split}.json"
        cs_path = root / CS_DIR / "annot" / f"{split}.json"
        result = compare_annotation_sets(
            load_annotations(os_path),
            load_annotations(cs_path),
        )
        output[split] = result
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("BirdGaze_v2"))
    args = parser.parse_args()
    results = verify_scenarios(args.dataset_root.resolve())
    print(json.dumps(results, indent=2))
    missing = [
        split for split, result in results.items()
        if result["images_with_changed_keypoint_coordinates"] == 0
    ]
    if missing:
        parser.exit(
            2,
            "FAILED: CS and OS have identical keypoint coordinates for "
            + ", ".join(missing)
            + ". Different bounding boxes do NOT establish corrected keypoint "
              "labels. Find the genuine corrected joint annotations before "
              "training or generating definitive CS comparison data.\n"
        )
    print("PASS: Corrected keypoint coordinates are present in both splits.")


if __name__ == "__main__":
    main()
