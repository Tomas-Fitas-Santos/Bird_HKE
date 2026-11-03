#!/usr/bin/env python3
"""Convert BirdGaze's existing per-image annotations to COCO-style pose JSON.

This never modifies BirdGaze source annotations or images. It retains the
original Bird_HKE crop's center/scale semantics for the MMPose adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

SCENARIOS = {
    "FD": "BirdGaze_Full_Dataset",
    "OS": "birdgaze_original_subset",
    "CS": "birdgaze_corrected_subset",
}
KEYPOINT_NAMES = ("crown", "left_eye", "right_eye", "beak_tip")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def locate_image(images: Path, filename: str) -> tuple[str, Path]:
    name = filename.replace("\\", "/")
    if name.startswith("/") or ".." in Path(name).parts or name.startswith("images/"):
        raise ValueError(f"Unsafe/noncanonical image path: {name}")
    candidates = [name]
    if name.startswith(("Animal_Kingdom/", "Animal Kingdom/")):
        for candidate in (name, name.replace("Animal_Kingdom/", "Animal Kingdom/"),
                          name.replace("Animal Kingdom/", "Animal_Kingdom/")):
            candidates.extend((candidate, re.sub(r"_f(\d+\.[A-Za-z0-9]+)$", r"_t\1", candidate)))
    for candidate in dict.fromkeys(candidates):
        path = images / candidate
        if path.is_file():
            return candidate, path
    raise FileNotFoundError(f"Image not found under {images}: {name}")


def available(x: float, y: float) -> bool:
    return math.isfinite(x) and math.isfinite(y) and x >= 0 and y >= 0 and (x != 0 or y != 0)


def convert_record(record: dict, image_id: int, images: Path, image_size: tuple[int, int]) -> tuple[dict, dict]:
    name = str(record["image"])
    path_name, _ = locate_image(images, name)
    w_img, h_img = image_size
    center = record["center"]
    scale = float(record["scale"])
    if len(center) != 2 or not all(math.isfinite(float(q)) for q in center) or float(center[0]) < 0:
        raise ValueError(f"{name}: invalid center")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"{name}: invalid scale {scale!r}")
    joints = record["joints"]
    if len(joints) != 4:
        raise ValueError(f"{name}: expected 4 keypoints, got {len(joints)}")
    explicit = record.get("joints_valid")
    visible = record.get("joints_vis")
    if explicit is not None and len(explicit) != 4:
        raise ValueError(f"{name}: invalid joints_valid")
    if visible is not None and len(visible) != 4:
        raise ValueError(f"{name}: invalid joints_vis")
    keypoints = []
    count = 0
    for i, joint in enumerate(joints):
        x, y = float(joint[0]), float(joint[1])
        valid = bool(float(explicit[i]) > 0) if explicit is not None else (
            available(x, y) or (visible is not None and float(visible[i]) > 0))
        if valid and not available(x, y):
            raise ValueError(f"{name}: keypoint {i} marked valid with invalid coordinate")
        if valid:
            flag = 2 if visible is None or float(visible[i]) > 0 else 1
            keypoints.extend((x, y, flag))
            count += 1
        else:
            keypoints.extend((0.0, 0.0, 0))
    # Keep samples with no available landmarks, exactly as Bird_HKE does.
    # All four target weights will be zero: these images provide no pose-loss
    # signal, but are part of the original dataset and must not be discarded.
    # Existing Bird_HKE stores crop scale in units of 200 px and later applies
    # a 1.25 padding. The MMPose GetBBoxCenterScale step supplies this padding.
    # Do not clip this box at image borders; the custom dataset preserves it.
    side = 200.0 * scale
    x0 = float(center[0]) - side / 2.0
    y0 = float(center[1]) - side / 2.0
    img = {"id": image_id, "file_name": path_name, "width": w_img, "height": h_img}
    ann = {
        "id": image_id, "image_id": image_id, "category_id": 1,
        "bbox": [x0, y0, side, side], "area": side * side,
        "iscrowd": 0, "num_keypoints": count, "keypoints": keypoints,
    }
    return img, ann


def convert_split(source: Path, images_root: Path, output: Path) -> dict:
    from PIL import Image

    with source.open(encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise TypeError(f"{source} must contain a list of BirdGaze records")
    images, annotations = [], []
    sizes = {}
    unsupervised = 0
    for idx, record in enumerate(records, 1):
        resolved, file_path = locate_image(images_root, str(record["image"]))
        if resolved not in sizes:
            with Image.open(file_path) as frame:
                sizes[resolved] = tuple(frame.size)
        image, annotation = convert_record(record, idx, images_root, sizes[resolved])
        images.append(image)
        annotations.append(annotation)
        if annotation["num_keypoints"] == 0:
            unsupervised += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "info": {"description": "BirdGaze 4-keypoint evaluation-compatible export"},
        "licenses": [], "images": images,
        "categories": [{"id": 1, "name": "bird", "supercategory": "animal",
                        "keypoints": list(KEYPOINT_NAMES), "skeleton": [[1, 2], [1, 3], [1, 4]]}],
        "annotations": annotations,
    }
    temporary = output.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)
    temporary.replace(output)
    return {"source": str(source), "source_sha256": sha256(source),
            "output": str(output), "output_sha256": sha256(output),
            "images": len(images), "annotations": len(annotations),
            "samples_without_valid_keypoints": unsupervised}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS.keys(), required=True)
    parser.add_argument("--dataset-root", type=Path, help="Override BirdGaze_v2/<scenario> root")
    parser.add_argument("--output-root", type=Path, default=Path("external_models/generated"))
    args = parser.parse_args()
    root = args.dataset_root or Path("BirdGaze_v2") / SCENARIOS[args.scenario]
    root = root.resolve()
    annotations = root / "annot"
    images_root = root / "images"
    dest = args.output_root.resolve() / args.scenario
    for split in ("train", "val"):
        report = convert_split(annotations / f"{split}.json", images_root,
                               dest / f"{split}_coco.json")
        print(json.dumps({"scenario": args.scenario, "split": split, **report}, indent=2))
    print("Original annotations/images were not modified.")


if __name__ == "__main__":
    main()
