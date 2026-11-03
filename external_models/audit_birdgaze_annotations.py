#!/usr/bin/env python3
"""Read-only audit of BirdGaze FD annotation geometry and OS/CS differences."""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

DATASETS = {
    "FD": "BirdGaze_Full_Dataset",
    "OS": "birdgaze_original_subset",
    "CS": "birdgaze_corrected_subset",
}
TRAINING_FIELDS = frozenset(
    {"image", "joints", "joints_vis", "joints_valid", "center", "scale", "bbox"}
)


def read_records(root: Path, scenario: str, split: str) -> list[dict]:
    path = root / DATASETS[scenario] / "annot" / f"{split}.json"
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Not a list: {path}")
    return data


def audit_geometry(records: list[dict], label: str) -> None:
    cases: collections.Counter = collections.Counter()
    examples: dict[str, list[dict]] = {}
    for i, record in enumerate(records):
        raw = record.get("scale")
        try:
            scale = float(raw)
        except (ValueError, TypeError):
            scale = float("nan")
        center = record.get("center")
        flags = []
        if not math.isfinite(scale):
            flags.append("nonfinite_scale")
        elif scale == 0:
            flags.append("zero_scale")
        elif scale < 0:
            flags.append("negative_scale")
        if not isinstance(center, list) or len(center) != 2:
            flags.append("invalid_center_shape")
        else:
            try:
                x, y = map(float, center)
                if not math.isfinite(x) or not math.isfinite(y):
                    flags.append("nonfinite_center")
                if x == -1:
                    flags.append("center_minus_one")
                elif x < 0 or y < 0:
                    flags.append("negative_center")
            except (TypeError, ValueError):
                flags.append("nonnumeric_center")
        if "bbox" in record and record["bbox"] is not None:
            cases["original_bbox_present"] += 1
        for flag in flags:
            cases[flag] += 1
            examples.setdefault(flag, [])
            if len(examples[flag]) < 4:
                examples[flag].append({
                    "index": i,
                    "image": record.get("image"),
                    "center": center,
                    "scale": raw,
                    "bbox": record.get("bbox"),
                    "joints": record.get("joints"),
                })
    print(f"\n{label}: {len(records)} records")
    for kind, count in sorted(cases.items()):
        print(f"  {kind}: {count}")
        for example in examples.get(kind, []):
            print(f"    example: {json.dumps(example, ensure_ascii=False)}")


def compare_scenarios(root: Path, split: str) -> None:
    os_records = read_records(root, "OS", split)
    cs_records = read_records(root, "CS", split)
    print(f"\nOS vs CS / {split}: {len(os_records)} versus {len(cs_records)} records")
    if len(os_records) != len(cs_records):
        print("  WARNING: scenario record counts differ")
    differences: collections.Counter = collections.Counter()
    examples: dict[str, list[dict]] = {}
    for i in range(min(len(os_records), len(cs_records))):
        old, corrected = os_records[i], cs_records[i]
        for key in sorted(set(old) | set(corrected)):
            if old.get(key) != corrected.get(key):
                differences[key] += 1
                if len(examples.setdefault(key, [])) < 2:
                    examples[key].append({
                        "index": i, "image_OS": old.get("image"),
                        "image_CS": corrected.get("image"),
                        "OS": str(old.get(key))[:300],
                        "CS": str(corrected.get(key))[:300],
                    })
    if not differences:
        print("  No field-level differences detected.")
    for key, count in differences.most_common():
        impact = "TRAINING FIELD" if key in TRAINING_FIELDS else "other/metadata field"
        print(f"  {key}: {count} differing records [{impact}]")
        for example in examples[key]:
            print(f"    example: {json.dumps(example, ensure_ascii=False)}")
    if differences and not any(key in TRAINING_FIELDS for key in differences):
        print("  WARNING: Only fields outside the current training-data contract differ.")
    if differences.get("bbox", 0):
        print("  WARNING: Original Bird_HKE applies per-record bbox cropping; "
              "current external converter does NOT reproduce that behavior.")
    if not differences.get("joints", 0) and not differences.get("joints_valid", 0) and not differences.get("joints_vis", 0):
        print("  WARNING: No keypoint coordinate/validity/visibility differences detected.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("BirdGaze_v2"))
    args = parser.parse_args()
    for split in ("train", "val"):
        for scenario in DATASETS:
            audit_geometry(read_records(args.root, scenario, split), f"{scenario} / {split}")
        compare_scenarios(args.root, split)


if __name__ == "__main__":
    main()
