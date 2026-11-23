#!/usr/bin/env python3
"""Convert Annotatiathon Supervisely per-image JSONs into BirdGaze annotations.

The sources are read-only.  All writes are confined to --output.  Run --audit-only
first to catch name collisions, missing images, and unexpected annotation geometry.
Requires Python 3.10+ and Pillow: python -m pip install Pillow
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from PIL import Image, UnidentifiedImageError

DEFAULT_EBIRD_ANN = r"E:\Annotatiathon 2026\386759_Nightjar · Official Annotatiathon Dataset · Official Event [NJ_ff71f54f-9f77-45c8-8504-c9c4028b0249_425800a7-4c09-4ebf-b6ef-2e78d7e62541]\Complete video library · all species\ann"
DEFAULT_NABIRDS_ANN = r"E:\Annotatiathon 2026\386817_Nightjar · NABirds · Official Event [NJ_ff71f54f-9f77-45c8-8504-c9c4028b0249_b7ef26cf-2084-4a52-9293-19a46aceac87]\Complete video library · all species\ann"
DEFAULT_EBIRD_IMAGES = r"E:\Official-Annotatiathon-Dataset\Official Annotatiathon Dataset"
DEFAULT_NABIRDS_IMAGES = r"E:\BirdGaze_v2\BirdGaze_Full_Dataset\images\NABirds"
DEFAULT_OUTPUT = r"E:\Annotatiathon 2026\BirdGaze_Annotatiathon_2026"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
KEYPOINTS = ("crown", "left_eye", "right_eye", "beak")
KEYPOINT_ALIAS = {
    "crown": "crown", "head_mid_top": "crown", "head_top": "crown",
    "left_eye": "left_eye", "eye_left": "left_eye",
    "right_eye": "right_eye", "eye_right": "right_eye",
    "beak": "beak", "beak_tip": "beak", "mouth_front_top": "beak",
}
CSV_FIELDS = ("source", "source_file", "annotation_file", "image", "split",
              "width", "height", "visible_count", "occluded_count", "group")


def norm(s: str) -> str:
    return s.replace("\\", "/").strip("/").casefold()


def label_info(title: str) -> tuple[str | None, str | None]:
    k = title.strip().casefold().replace(" ", "_").replace("-", "_")
    if k in {"bird", "bird_bbox", "bird_box"}:
        return ("bird", "bbox")
    if k.endswith("_occluded"):
        k = k[:-9]
        state = "occluded"
    elif k.endswith("_occ"):
        k = k[:-4]
        state = "occluded"
    else:
        state = "visible"
    return (KEYPOINT_ALIAS.get(k), state)


def point_of(obj: dict[str, Any]) -> list[float]:
    pts = obj.get("points", {}).get("exterior", [])
    if len(pts) != 1 or len(pts[0]) != 2:
        raise ValueError("point must have exactly one [x,y] exterior coordinate")
    x, y = map(float, pts[0])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("nonfinite point coordinate")
    return [x, y]


def bbox_of(obj: dict[str, Any]) -> list[float]:
    pts = obj.get("points", {}).get("exterior", [])
    if len(pts) != 2 or any(len(pt) != 2 for pt in pts):
        raise ValueError("rectangle needs exactly two opposite corners")
    x1, x2 = sorted((float(pts[0][0]), float(pts[1][0])))
    y1, y2 = sorted((float(pts[0][1]), float(pts[1][1])))
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError("degenerate or nonfinite rectangle")
    return [x1, y1, x2, y2]


def discover_media(root: Path) -> dict[str, dict[str, list[Path]]]:
    """Index exact relative paths and basenames, keeping ambiguity visible."""
    by_rel: dict[str, list[Path]] = collections.defaultdict(list)
    by_name: dict[str, list[Path]] = collections.defaultdict(list)
    by_stem: dict[str, list[Path]] = collections.defaultdict(list)
    for d, _, files in os.walk(root):
        for name in files:
            p = Path(d) / name
            if p.suffix.casefold() not in IMAGE_EXTS:
                continue
            by_rel[norm(p.relative_to(root).as_posix())].append(p)
            by_name[name.casefold()].append(p)
            by_stem[p.stem.casefold()].append(p)
    return {"rel": by_rel, "name": by_name, "stem": by_stem}


def find_media(ann: Path, ann_root: Path, idx: dict, root: Path) -> Path:
    # Supervisely exports the annotation as original.jpg.json, optionally nested.
    rel = ann.relative_to(ann_root).as_posix()[:-5]
    rel_norm = norm(rel)
    name = Path(rel).name
    for k in (rel_norm,):
        candidates = idx["rel"].get(k, [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(f"ambiguous relative match {rel}: {len(candidates)} candidates")
    candidates = idx["name"].get(name.casefold(), [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Match annotation's trailing directory hierarchy, never choose arbitrarily.
        suffix = "/" + rel_norm
        narrowed = [p for p in candidates if ("/" + norm(p.relative_to(root).as_posix())).endswith(suffix)]
        if len(narrowed) == 1:
            return narrowed[0]
        raise ValueError(f"ambiguous image basename {name}: {len(candidates)} candidates; sample: " +
                         "; ".join(str(p.relative_to(root)) for p in candidates[:4]))
    if Path(name).suffix.lower() not in IMAGE_EXTS:
        candidates = idx["stem"].get(name.casefold(), [])
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(f"ambiguous extensionless basename {name}: {len(candidates)} candidates")
    raise ValueError(f"no source image matches {rel}")


def convert(ann: dict, image: str, physical_image: Path) -> tuple[dict, dict, int, int]:
    if not isinstance(ann, dict) or not isinstance(ann.get("objects"), list):
        raise ValueError("not a Supervisely single-image annotation with objects[]")
    size = ann.get("size") or {}
    w0, h0 = size.get("width"), size.get("height")
    try:
        with Image.open(physical_image) as im:
            w, h = im.size
    except (OSError, UnidentifiedImageError) as ex:
        raise ValueError(f"cannot read source image header: {ex}") from ex
    if w0 is not None and h0 is not None and (int(w0), int(h0)) != (w, h):
        raise ValueError(f"Supervisely image dimensions {w0}x{h0} do not match file {w}x{h}")
    bboxes, points = [], collections.defaultdict(list)
    unknown_labels = collections.Counter()
    for obj in ann["objects"]:
        if not isinstance(obj, dict):
            raise ValueError("non-object item in objects[]")
        label, state = label_info(str(obj.get("classTitle", "")))
        if label is None:
            unknown_labels[str(obj.get("classTitle", ""))] += 1
            continue
        geom = str(obj.get("geometryType", "")).casefold()
        if label == "bird":
            if geom != "rectangle":
                raise ValueError("bird label is not rectangle")
            bboxes.append(bbox_of(obj))
        else:
            if geom != "point":
                raise ValueError(f"{label} label is not point")
            points[label].append((state, point_of(obj)))
    if len(bboxes) != 1:
        raise ValueError(f"expected exactly one bird bounding box, got {len(bboxes)}")
    box = bboxes[0]
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise ValueError(f"bird rectangle outside image ({w}x{h}): {box}")
    center = [(x1 + x2) / 2, (y1 + y2) / 2]
    # BirdGaze loader adds its own 1.25x padding on the stored scalar scale.
    scale = max(x2 - x1, y2 - y1) / 200.0
    joints, joints_vis, supplemental = [], [], {}
    n_visible = n_occluded = 0
    for key in KEYPOINTS:
        entries = points.get(key, [])
        if len(entries) > 1:
            raise ValueError(f"duplicate or conflicting annotations for {key}: {len(entries)}")
        if entries:
            state, pt = entries[0]
            if not (0 <= pt[0] < w and 0 <= pt[1] < h):
                raise ValueError(f"{key} coordinate outside image: {pt}")
            supplemental[key] = {"state": state, "xy": pt}
            if state == "visible":
                joints.append(pt)
                joints_vis.append(1)
                n_visible += 1
            else:
                joints.append([-1, -1])
                joints_vis.append(0)
                n_occluded += 1
        else:
            joints.append([-1, -1])
            joints_vis.append(0)
            supplemental[key] = {"state": "not_annotated", "xy": None}
    if n_visible + n_occluded == 0:
        raise ValueError("bird rectangle has no annotated head keypoints")
    birdgaze = {"image": image, "center": center, "scale": scale,
                "bbox": box, "joints": joints, "joints_vis": joints_vis}
    extra = {"image": image, "keypoints": supplemental, "unknown_labels": dict(unknown_labels)}
    return birdgaze, extra, n_visible, n_occluded


def choose_split(group: str, seed: str, train: float, val: float) -> str:
    h = hashlib.sha256((seed + "\0" + group).encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "big") / (2**64)
    return "train" if u < train else "val" if u < train + val else "test"


def group_for(source: str, relative: Path) -> str:
    # Video groups include *all* frames of a video, preventing sequence leakage.
    if source == "eBird":
        if len(relative.parts) < 3:
            raise ValueError("eBird frame must be nested species/video/frame (at least 3 parts)")
        return f"eBird/{relative.parts[0]}/{relative.parts[1]}"
    return "NABirds/" + relative.as_posix()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ebird-ann", type=Path, default=DEFAULT_EBIRD_ANN)
    p.add_argument("--nabirds-ann", type=Path, default=DEFAULT_NABIRDS_ANN)
    p.add_argument("--ebird-images", type=Path, default=DEFAULT_EBIRD_IMAGES)
    p.add_argument("--nabirds-images", type=Path, default=DEFAULT_NABIRDS_IMAGES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--audit-only", action="store_true", help="validate everything and create reports without copying media")
    p.add_argument("--allow-partial", action="store_true", help="build valid subset despite rejected annotations")
    p.add_argument("--materialize", choices=("copy", "hardlink"), default="copy",
                   help="hardlinks are fast but share bytes with the originals")
    p.add_argument("--seed", default="nightjar-2026-v1")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--skip-sizes-if-existing", action="store_true", help="reuse existing output images with matching byte sizes")
    return p.parse_args()


def ensure_independent_roots(source_roots: list[Path], output: Path) -> None:
    out = output.resolve()
    for source in source_roots:
        root = source.resolve()
        if out == root or root in out.parents or out in root.parents:
            raise ValueError(f"output and input directory must not contain one another: {out} / {root}")


def json_write(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def materialize_one(source: Path, dest: Path, mode: str, allow_existing: bool) -> None:
    if dest.exists():
        if allow_existing and source.stat().st_size == dest.stat().st_size:
            return
        raise FileExistsError(f"output image exists: {dest}. Use --skip-sizes-if-existing to resume")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    try:
        if mode == "hardlink":
            os.link(source, tmp)
        else:
            shutil.copy2(source, tmp)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def scan_and_build(args: argparse.Namespace) -> int:
    for src in (args.ebird_ann, args.nabirds_ann, args.ebird_images, args.nabirds_images):
        if not src.is_dir():
            raise FileNotFoundError(f"input directory does not exist: {src}")
    if not 0 <= args.train_ratio <= 1 or not 0 <= args.val_ratio <= 1 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("invalid train and validation ratios")
    ensure_independent_roots([args.ebird_ann, args.nabirds_ann, args.ebird_images, args.nabirds_images], args.output)
    report_dir = args.output / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    accepted, rejected, unique_images = [], [], {}
    stats = collections.Counter()
    source_specs = (("eBird", args.ebird_ann, args.ebird_images),
                    ("NABirds", args.nabirds_ann, args.nabirds_images))
    for source, ann_root, image_root in source_specs:
        print(f"Indexing {source} original images: {image_root}", flush=True)
        idx = discover_media(image_root)
        files = sorted(ann_root.rglob("*.json"), key=lambda p: str(p).casefold())
        stats[f"{source}_annotation_files"] = len(files)
        stats[f"{source}_media_files_available"] = sum(len(v) for v in idx["rel"].values())
        print(f"Scanning {source}: {len(files):,} annotation JSON files", flush=True)
        for i, ap in enumerate(files, 1):
            try:
                original = find_media(ap, ann_root, idx, image_root)
                rel = original.relative_to(image_root)
                dest_rel = (Path(source) / rel).as_posix()
                key = norm(dest_rel)
                if key in unique_images:
                    raise ValueError(f"image already annotated by {unique_images[key]}")
                group = group_for(source, rel)
                with ap.open("r", encoding="utf-8-sig") as f:
                    raw = json.load(f)
                record, extra, n_vis, n_occ = convert(raw, dest_rel, original)
                split = choose_split(group, args.seed, args.train_ratio, args.val_ratio)
                if n_vis == 0:
                    split = "occluded_only"  # keep data; avoid undefined validation metrics
                extra.update({"source": source, "original_image": str(original),
                              "annotation_json": str(ap), "split": split, "group": group})
                accepted.append({"source": source, "from": original, "annotation": ap,
                                 "image": dest_rel, "split": split, "group": group,
                                 "record": record, "extra": extra,
                                 "visible": n_vis, "occluded": n_occ,
                                 "width": int(raw.get("size", {}).get("width") or 0),
                                 "height": int(raw.get("size", {}).get("height") or 0)})
                unique_images[key] = ap
                stats[f"{source}_accepted"] += 1
                stats[f"{source}_{split}"] += 1
                stats[f"{source}_visible_points"] += n_vis
                stats[f"{source}_occluded_points"] += n_occ
            except (ValueError, OSError, TypeError, KeyError, json.JSONDecodeError) as ex:
                rejected.append({"source": source, "annotation_file": str(ap), "reason": str(ex)})
                stats[f"{source}_rejected"] += 1
            if i % 10000 == 0:
                print(f"  {source}: {i:,}/{len(files):,} inspected", flush=True)
    report = {"dataset": "BirdGaze_Annotatiathon_2026", "status": "AUDITED",
              "audit_only": bool(args.audit_only), "valid_annotations": len(accepted),
              "rejected_annotations": len(rejected), "statistics": dict(stats),
              "train_ratio": args.train_ratio, "val_ratio": args.val_ratio,
              "seed": args.seed, "materialize": args.materialize,
              "note": "Occluded points are kept in reports/supplemental_keypoints.jsonl and masked in BirdGaze joints."}
    json_write(report_dir / "validation_report.json", report)
    with (report_dir / "rejected_annotations.csv").open("w", newline="", encoding="utf-8-sig") as f:
        cw = csv.DictWriter(f, fieldnames=("source", "annotation_file", "reason"))
        cw.writeheader()
        cw.writerows(rejected)
    with (report_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        cw = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        cw.writeheader()
        for a in accepted:
            cw.writerow({"source": a["source"], "source_file": str(a["from"]),
                         "annotation_file": str(a["annotation"]), "image": a["image"],
                         "split": a["split"], "width": a["width"], "height": a["height"],
                         "visible_count": a["visible"], "occluded_count": a["occluded"],
                         "group": a["group"]})
    print(f"Audit: {len(accepted):,} accepted, {len(rejected):,} rejected. Reports: {report_dir}", flush=True)
    if args.audit_only:
        return 0 if not rejected else 2
    if rejected and not args.allow_partial:
        print("Build blocked: inspect rejected_annotations.csv; or explicitly pass --allow-partial", file=sys.stderr)
        return 2
    if not accepted:
        raise ValueError("No valid annotations to build")
    outputs = {k: [] for k in ("train", "val", "test", "ebird", "nabirds", "occluded_only")}
    seen_media: set[str] = set()
    for i, a in enumerate(accepted, 1):
        key = norm(a["image"])
        if key not in seen_media:
            materialize_one(a["from"], args.output / "images" / Path(a["image"]),
                            args.materialize, args.skip_sizes_if_existing)
            seen_media.add(key)
        outputs[a["source"].lower()].append(a["record"])
        outputs[a["split"]].append(a["record"])
        if i % 10000 == 0:
            print(f"Materialized {i:,}/{len(accepted):,}", flush=True)
    annot_dir = args.output / "annot"
    annot_dir.mkdir(parents=True, exist_ok=True)
    for k, records in outputs.items():
        json_write(annot_dir / (k + ".json"), records)
    with (report_dir / "supplemental_keypoints.jsonl").open("w", encoding="utf-8") as f:
        for a in accepted:
            f.write(json.dumps(a["extra"], ensure_ascii=False) + "\n")
    report.update({"status": "BUILT", "images_copied_or_linked": len(seen_media),
                   "splits": {k: len(v) for k, v in outputs.items()}})
    json_write(report_dir / "validation_report.json", report)
    print(f"Built: {len(seen_media):,} images, {len(accepted):,} annotations into {args.output}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    try:
        return scan_and_build(args)
    except (OSError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
