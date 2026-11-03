#!/usr/bin/env python3
"""Read-only, filename-matched comparison of BirdGaze OS and CS annotations.

Does not create or modify annotations, block training, or assume record order.
Optionally samples identical-named source image files to compare image bytes.
Run from repository root with:
  python -m external_models.inspect_os_cs_keypoints --check-images 30
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

OS_DIR = "birdgaze_original_subset"
CS_DIR = "birdgaze_corrected_subset"
LANDMARKS = ("crown", "left_eye", "right_eye", "beak_tip")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records_by_image(path: Path) -> dict[str, list[dict]]:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of annotations: {path}")
    indexed: dict[str, list[dict]] = defaultdict(list)
    for idx, record in enumerate(records):
        if not isinstance(record, dict) or "image" not in record:
            raise ValueError(f"Missing image identifier in {path}, record {idx}")
        name = str(record["image"]).replace("\\", "/")
        indexed[name].append(record)
    return dict(indexed)


def sample_keys(keys: list[str], count: int) -> list[str]:
    if not keys or count <= 0:
        return []
    n = min(len(keys), count)
    return [keys[i * len(keys) // n] for i in range(n)]


def compare_split(os_file: Path, cs_file: Path, check_images: int) -> dict:
    original = records_by_image(os_file)
    corrected = records_by_image(cs_file)
    os_names, cs_names = set(original), set(corrected)
    shared = sorted(os_names & cs_names)
    counters = Counter()
    changed_by_landmark = Counter()
    examples = []
    for name in shared:
        old_records, new_records = original[name], corrected[name]
        if len(old_records) != len(new_records):
            counters["images_with_different_annotation_multiplicity"] += 1
        for old, new in zip(old_records, new_records):
            counters["paired_records"] += 1
            if old.get("joints") != new.get("joints"):
                counters["records_with_different_full_joints"] += 1
            differences = []
            for idx, label in enumerate(LANDMARKS):
                try:
                    old_xy, new_xy = old["joints"][idx][:2], new["joints"][idx][:2]
                except (IndexError, KeyError, TypeError) as exc:
                    raise ValueError(f"Invalid joint data in image {name}") from exc
                if old_xy != new_xy:
                    changed_by_landmark[label] += 1
                    differences.append({
                        "landmark": label, "OS": old_xy, "CS": new_xy
                    })
            if differences:
                counters["records_with_different_keypoint_xy"] += 1
                if len(examples) < 5:
                    examples.append({"image": name, "changed_keypoints": differences})
            for field in ("joints_vis", "joints_valid", "bbox", "center", "scale"):
                if old.get(field) != new.get(field):
                    counters[f"records_with_different_{field}"] += 1

    duplicate_os = sum(1 for records in original.values() if len(records) > 1)
    duplicate_cs = sum(1 for records in corrected.values() if len(records) > 1)
    total_os = sum(map(len, original.values()))
    total_cs = sum(map(len, corrected.values()))

    image_comparison = {"sampled_pairs": 0, "identical_bytes": 0,
                        "different_bytes": 0, "missing_image_pairs": 0}
    if check_images > 0:
        from external_models.convert_birdgaze_to_coco import locate_image
        os_images = os_file.parent.parent / "images"
        cs_images = cs_file.parent.parent / "images"
        for name in sample_keys(shared, check_images):
            try:
                _, os_path = locate_image(os_images, name)
                _, cs_path = locate_image(cs_images, name)
            except FileNotFoundError:
                image_comparison["missing_image_pairs"] += 1
                continue
            image_comparison["sampled_pairs"] += 1
            field = "identical_bytes" if file_hash(os_path) == file_hash(cs_path) else "different_bytes"
            image_comparison[field] += 1

    return {
        "os_annotation_file": str(os_file),
        "cs_annotation_file": str(cs_file),
        "os_annotation_sha256": file_hash(os_file),
        "cs_annotation_sha256": file_hash(cs_file),
        "os_records": total_os,
        "cs_records": total_cs,
        "shared_image_names": len(shared),
        "os_only_image_names": len(os_names - cs_names),
        "cs_only_image_names": len(cs_names - os_names),
        "duplicate_image_names_os": duplicate_os,
        "duplicate_image_names_cs": duplicate_cs,
        "pairs_compared": counters["paired_records"],
        "records_with_different_keypoint_xy": counters["records_with_different_keypoint_xy"],
        "records_with_different_full_joints": counters["records_with_different_full_joints"],
        "changed_keypoints_by_landmark": {k: changed_by_landmark[k] for k in LANDMARKS},
        "records_with_different_visibility": counters["records_with_different_joints_vis"],
        "records_with_different_validity": counters["records_with_different_joints_valid"],
        "records_with_different_bbox": counters["records_with_different_bbox"],
        "records_with_different_center": counters["records_with_different_center"],
        "records_with_different_scale": counters["records_with_different_scale"],
        "images_with_different_annotation_multiplicity": counters["images_with_different_annotation_multiplicity"],
        "first_coordinate_differences": examples,
        "optional_source_image_byte_comparison": image_comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("BirdGaze_v2"))
    parser.add_argument("--check-images", type=int, default=30,
                        help="Number of distributed source image pairs to hash per split (0 to skip)")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report destination")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    os_dir = root / OS_DIR / "annot"
    cs_dir = root / CS_DIR / "annot"
    common = sorted(set(p.name for p in os_dir.glob("*.json")) &
                    set(p.name for p in cs_dir.glob("*.json")))
    reports = {}
    for filename in common:
        # Only annotation lists are relevant; metadata JSON is not a keypoint split.
        with (os_dir / filename).open(encoding="utf-8") as source:
            preview = json.load(source)
        if not isinstance(preview, list) or not preview or not isinstance(preview[0], dict):
            continue
        if "image" not in preview[0] or "joints" not in preview[0]:
            continue
        reports[filename] = compare_split(
            os_dir / filename, cs_dir / filename, args.check_images
        )
    if not reports:
        print("No common annotation JSON splits found.")
    for filename, report in reports.items():
        print(f"\n{filename}: {report['pairs_compared']:,} image records compared by filename")
        print(f"  Different keypoint XY records: {report['records_with_different_keypoint_xy']:,}")
        print(f"  Different complete joints arrays: {report['records_with_different_full_joints']:,}")
        print(f"  Differences per landmark: {report['changed_keypoints_by_landmark']}")
        print(f"  Different visibility / validity: {report['records_with_different_visibility']:,} / {report['records_with_different_validity']:,}")
        print(f"  Different bboxes / center / scale: {report['records_with_different_bbox']:,} / {report['records_with_different_center']:,} / {report['records_with_different_scale']:,}")
        print(f"  Image names OS-only / CS-only: {report['os_only_image_names']:,} / {report['cs_only_image_names']:,}")
        print(f"  Duplicate image names OS / CS: {report['duplicate_image_names_os']:,} / {report['duplicate_image_names_cs']:,}")
        print(f"  Sample source image bytes: {report['optional_source_image_byte_comparison']}")
        for example in report["first_coordinate_differences"][:3]:
            print(f"  Coordinate difference example: {json.dumps(example)}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
        print(f"\nReport saved: {args.output}")
    print("\nREAD-ONLY COMPARISON COMPLETED (no training gate).")


if __name__ == "__main__":
    main()
