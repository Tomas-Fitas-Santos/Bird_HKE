#!/usr/bin/env python3
"""Create BirdGaze eBird and NABirds datasets from the 2026 Annotatiathon.

The source media remain untouched. New full Supervisely exports with ann/img/
img_info are matched directly. The exported file hashes recover original paths
when possible; the Nightjar media map remains an optional legacy mode.

Requirements: Python 3.10+, Pillow. See ANNOTATIATHON_BIRDGAZE.md.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from typing import Any

from PIL import Image, UnidentifiedImageError

DEFAULT_EBIRD_ANN = r"E:\Annotatiathon 2026\386759_Nightjar · Official Annotatiathon Dataset · Official Event [NJ_ff71f54f-9f77-45c8-8504-c9c4028b0249_425800a7-4c09-4ebf-b6ef-2e78d7e62541]\Complete video library · all species\ann"
DEFAULT_NABIRDS_ANN = r"E:\Annotatiathon 2026\386817_Nightjar · NABirds · Official Event [NJ_ff71f54f-9f77-45c8-8504-c9c4028b0249_b7ef26cf-2084-4a52-9293-19a46aceac87]\Complete video library · all species\ann"
DEFAULT_EBIRD_IMAGES = r"E:\Official-Annotatiathon-Dataset\Official Annotatiathon Dataset"
DEFAULT_NABIRDS_IMAGES = r"E:\BirdGaze_v2\BirdGaze_Full_Dataset\images\NABirds"
DEFAULT_OUTPUT = r"E:\Annotatiathon 2026\BirdGaze_Annotatiathon_2026"
DEFAULT_MEDIA_MAP = r"E:\Annotatiathon 2026\Nightjar_media_asset_map.csv"
DEFAULT_EBIRD_EXPORT = r"E:\Annotatiathon 2026\Nightjar_Annotatiathon_eBird_videos"
DEFAULT_NABIRDS_EXPORT = r"E:\Annotatiathon 2026\Nightjar_Annotatiathon_NABirds"

MEDIA_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ANNOTATION_NAME = re.compile(
    r"(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r"(?P<ext>\.(?:jpg|jpeg|png|webp|bmp|tif|tiff))\.json$", re.IGNORECASE
)
LABEL_ALIASES = {
    "crown": "crown", "head_mid_top": "crown", "head_top": "crown",
    "left_eye": "left_eye", "eye_left": "left_eye",
    "right_eye": "right_eye", "eye_right": "right_eye",
    "beak": "beak", "beak_tip": "beak", "mouth_front_top": "beak",
}
KEYPOINTS = ("crown", "left_eye", "right_eye", "beak")
MANIFEST_FIELDS = (
    "source", "uuid", "original_media_path", "annotation_file", "image",
    "match_method", "video_group", "frame_index", "width", "height",
    "visible_count", "occluded_count",
)


def class_info(class_title: str) -> tuple[str | None, str]:
    name = class_title.strip().casefold().replace(" ", "_").replace("-", "_")
    if name in {"bird", "bird_bbox", "bird_box"}:
        return "bird", "bbox"
    if name.endswith("_occluded"):
        name, state = name[:-9], "occluded"
    elif name.endswith("_occ"):
        name, state = name[:-4], "occluded"
    else:
        state = "visible"
    return LABEL_ALIASES.get(name), state


def annotated_point(obj: dict) -> list[float]:
    xy = (obj.get("points") or {}).get("exterior", [])
    if len(xy) != 1 or not isinstance(xy[0], list) or len(xy[0]) != 2:
        raise ValueError("invalid point geometry")
    try:
        x, y = (float(v) for v in xy[0])
    except (TypeError, ValueError) as e:
        raise ValueError("invalid point numeric value") from e
    if not all(math.isfinite(v) for v in (x, y)):
        raise ValueError("nonfinite keypoint")
    return [x, y]


def annotated_bbox(obj: dict) -> list[float]:
    points = (obj.get("points") or {}).get("exterior", [])
    if len(points) != 2 or any(not isinstance(pt, list) or len(pt) != 2 for pt in points):
        raise ValueError("bird bbox must be a two-corner rectangle")
    try:
        x1, x2 = sorted((float(pt[0]) for pt in points))
        y1, y2 = sorted((float(pt[1]) for pt in points))
    except (ValueError, TypeError) as e:
        raise ValueError("invalid bbox numeric value") from e
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError("degenerate or nonfinite bounding box")
    return [x1, y1, x2, y2]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def convert_supervisely(ann: dict, image_name: str, media: Path, mapped: dict) -> tuple[dict, int, int, int, int]:
    if not isinstance(ann, dict) or not isinstance(ann.get("objects"), list):
        raise ValueError("not a Supervisely single-image objects[] JSON")
    try:
        with Image.open(media) as img:
            w, h = img.size
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("unable to read source media") from e
    ann_size = ann.get("size") or {}
    if ann_size.get("width") is not None and int(ann_size["width"]) != w:
        raise ValueError("annotation width does not match original image")
    if ann_size.get("height") is not None and int(ann_size["height"]) != h:
        raise ValueError("annotation height does not match original image")
    if mapped.get("frame_width") and int(mapped["frame_width"]) != w:
        raise ValueError("database image width does not match original image")
    if mapped.get("frame_height") and int(mapped["frame_height"]) != h:
        raise ValueError("database image height does not match original image")
    if mapped.get("size_bytes") and int(mapped["size_bytes"]) != media.stat().st_size:
        raise ValueError("original image size changed since Nightjar import")

    bboxes: list[list[float]] = []
    points = defaultdict(list)
    for obj in ann["objects"]:
        if not isinstance(obj, dict):
            raise ValueError("invalid entry in objects[]")
        cls, state = class_info(str(obj.get("classTitle", "")))
        if cls is None:
            continue
        if cls == "bird":
            if str(obj.get("geometryType", "")).lower() != "rectangle":
                raise ValueError("bird is not a rectangle")
            bboxes.append(annotated_bbox(obj))
        else:
            if str(obj.get("geometryType", "")).lower() != "point":
                raise ValueError("keypoint is not a point")
            points[cls].append((state, annotated_point(obj)))
    if len(bboxes) != 1:
        raise ValueError(f"expected 1 bird bounding box, got {len(bboxes)}")
    x1, y1, x2, y2 = bboxes[0]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        raise ValueError("bird rectangle outside original image")
    joints, vis = [], []
    visible = occluded = 0
    for cls in KEYPOINTS:
        items = points.get(cls, [])
        if len(items) > 1:
            raise ValueError(f"duplicate keypoint: {cls}")
        if not items:
            joints.append([-1, -1])
            vis.append(0)
            continue
        state, xy = items[0]
        if not (0 <= xy[0] < w and 0 <= xy[1] < h):
            raise ValueError(f"keypoint outside image: {cls}")
        # Occluded coordinates remain in the BirdGaze record; joints_vis=0 masks them.
        joints.append(xy)
        vis.append(int(state == "visible"))
        visible += int(state == "visible")
        occluded += int(state == "occluded")
    if visible + occluded == 0:
        raise ValueError("no head keypoints annotated")
    bbox = bboxes[0]
    return {
        "image": image_name,
        "center": [(x1 + x2) / 2, (y1 + y2) / 2],
        "scale": max(x2 - x1, y2 - y1) / 200.0,
        "bbox": bbox,
        "joints": joints,
        "joints_vis": vis,
    }, visible, occluded, w, h


def load_media_map(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Nightjar media mapping CSV missing: {path}. "
            "Export the read-only media_assets table first; see ANNOTATIATHON_BIRDGAZE.md."
        )
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"asset_id", "native_project_id", "relative_path", "kind", "size_bytes"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"media map lacks required columns: {sorted(missing)}")
        for n, row in enumerate(reader, 2):
            uid = str(row["asset_id"]).strip().lower()
            if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uid):
                raise ValueError(f"invalid asset_id on CSV row {n}")
            project = str(row["native_project_id"]).strip()
            if not project.isdigit():
                raise ValueError(f"invalid native_project_id on CSV row {n}")
            key = (project, uid)
            if key in mapping:
                raise ValueError(f"duplicate Nightjar asset mapping: {key}")
            mapping[key] = row
    if not mapping:
        raise ValueError("Nightjar media map has no rows for the requested projects")
    return mapping


def safe_source_relative(record: dict, source: str) -> PurePosixPath:
    raw = str(record["relative_path"]).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if (candidate.is_absolute() or not candidate.parts
            or any(p in {"..", "."} or ":" in p for p in candidate.parts)):
        raise ValueError(f"unsafe relative media path: {raw}")
    if candidate.parts[0].lower() in {"videos", "images"}:
        candidate = PurePosixPath(*candidate.parts[1:])
    if not candidate.parts or candidate.suffix.lower() not in MEDIA_EXTS:
        raise ValueError(f"invalid media path from Nightjar: {raw}")
    if source == "eBird" and len(candidate.parts) < 3:
        raise ValueError(f"eBird video path is missing species/video/frame: {raw}")
    return candidate



class ExactContentIndex:
    """Lazy size-bucket index: find the original file using Nightjar's SHA256."""

    def __init__(self, root: Path):
        self.root = root
        self.by_size = defaultdict(list)
        self.hash_cache = {}
        print(f"Indexing original media byte sizes for safe fallback: {root}", flush=True)
        for directory, _, names in os.walk(root):
            for name in names:
                p = Path(directory) / name
                if p.suffix.casefold() not in MEDIA_EXTS:
                    continue
                try:
                    self.by_size[p.stat().st_size].append(p)
                except OSError:
                    continue

    def lookup(self, entry: dict) -> tuple[Path, PurePosixPath]:
        raw_hash = str(entry.get("sha256") or "").strip().lower()
        raw_size = str(entry.get("size_bytes") or "").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", raw_hash) or not raw_size.isdigit():
            raise ValueError("Nightjar SHA-256 or media size missing for content lookup")
        candidates = self.by_size.get(int(raw_size), [])
        matches = []
        for p in candidates:
            if p not in self.hash_cache:
                self.hash_cache[p] = sha256_file(p)
            if self.hash_cache[p] == raw_hash:
                matches.append(p)
        if len(matches) != 1:
            raise ValueError(
                f"cannot restore unique original by database SHA-256: "
                f"{len(matches)} matching files out of {len(candidates)} size candidates"
            )
        path = matches[0]
        return path, PurePosixPath(path.relative_to(self.root).as_posix())


def resolve_original(record: dict, root: Path, source: str) -> tuple[Path, PurePosixPath]:
    relative = safe_source_relative(record, source)
    direct = root.joinpath(*relative.parts)
    if direct.is_file():
        return direct, relative
    # Allow a redundant NABirds directory prefix; never guess by image dimensions.
    if source == "NABirds" and relative.parts[0].casefold() == "nabirds":
        shorter = PurePosixPath(*relative.parts[1:])
        target = root.joinpath(*shorter.parts)
        if target.is_file():
            return target, shorter
    raise ValueError(f"original media path missing: {direct}")


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def put_media(original: Path, destination: Path, mode: str, resume: bool) -> None:
    if destination.exists():
        if resume and original.stat().st_size == destination.stat().st_size:
            return
        raise FileExistsError(f"output already exists: {destination}. Use --skip-sizes-if-existing for interrupted copies")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        partial.unlink()
    try:
        if mode == "hardlink":
            os.link(original, partial)
        else:
            shutil.copy2(original, partial)
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ebird-ann", type=Path, default=DEFAULT_EBIRD_ANN)
    parser.add_argument("--nabirds-ann", type=Path, default=DEFAULT_NABIRDS_ANN)
    parser.add_argument("--ebird-images", type=Path, default=DEFAULT_EBIRD_IMAGES)
    parser.add_argument("--nabirds-images", type=Path, default=DEFAULT_NABIRDS_IMAGES)
    parser.add_argument("--media-map", type=Path, default=DEFAULT_MEDIA_MAP)
    parser.add_argument("--ebird-export", type=Path, default=DEFAULT_EBIRD_EXPORT,
                        help="new eBird export containing img/, img_info/ and optionally ann/")
    parser.add_argument("--nabirds-export", type=Path, default=DEFAULT_NABIRDS_EXPORT,
                        help="new NABirds export containing img/, img_info/ and optionally ann/")
    parser.add_argument("--source-mode", choices=("auto", "exports", "nightjar-map"), default="auto",
                        help="auto uses new Supervisely image exports if both are present")
    parser.add_argument("--no-original-lookup", action="store_true",
                        help="convert directly from exported UUID images; skip hash recovery of old paths")
    parser.add_argument("--require-original-paths", action="store_true",
                        help="reject any image whose original species/video/frame path cannot be restored")
    parser.add_argument("--ignore-nightjar-map", action="store_true",
                        help="do not use an existing Nightjar mapping CSV, matching originals by SHA256 only")
    parser.add_argument("--ebird-project-id", default="386759")
    parser.add_argument("--nabirds-project-id", default="386817")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-sha256", action="store_true",
                        help="verify original media hashes from Nightjar database (can read 100+ GB)")
    parser.add_argument("--retain-annotation-issues", action="store_true",
                        help="keep matched images with fixable annotation problems in images/, "
                             "save raw labels under pending_corrections/, and exclude them "
                             "from the valid BirdGaze training JSONs")
    parser.add_argument("--allow-partial", action="store_true",
                        help="allow building valid subset despite rejected annotations")
    parser.add_argument("--materialize", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--skip-sizes-if-existing", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    roots = (args.ebird_ann, args.nabirds_ann, args.ebird_images, args.nabirds_images)
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"missing source directory: {root}")
    output = args.output.resolve()
    if any(root.resolve() == output or root.resolve() in output.parents or output in root.resolve().parents
           for root in (*roots, args.media_map)):
        raise ValueError("output directory must not contain or be contained by source directories")
    mapping = load_media_map(args.media_map)
    report_dir = output / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    accepted = []
    rejected = []
    stats = Counter()
    seen = set()
    content_indexes: dict[str, ExactContentIndex] = {}
    sources = (
        ("eBird", args.ebird_project_id, args.ebird_ann, args.ebird_images),
        ("NABirds", args.nabirds_project_id, args.nabirds_ann, args.nabirds_images),
    )
    for source, project_id, annotations, images in sources:
        jsons = sorted(annotations.rglob("*.json"), key=lambda p: str(p).casefold())
        stats[f"{source}_json_files"] = len(jsons)
        print(f"Scanning {source}: {len(jsons):,} UUID annotations; Nightjar mapping loaded", flush=True)
        for n, annpath in enumerate(jsons, 1):
            try:
                match = ANNOTATION_NAME.fullmatch(annpath.name)
                if not match:
                    raise ValueError(f"unrecognized Supervisely annotation filename: {annpath.name}")
                uid = match.group("uuid").lower()
                item = mapping.get((str(project_id), uid))
                if item is None:
                    raise ValueError(f"Nightjar media_assets.id not found in project {project_id}: {uid}")
                # The dataset/project is authoritative for source identity.
                # NABirds photo folders can have been indexed as 'video_frame'
                # by older app imports, so do NOT reject based on the old kind.
                kind = str(item["kind"]).lower()
                try:
                    original, relative = resolve_original(item, images, source)
                    match_method = "nightjar_asset_relative_path"
                except ValueError as missing_path:
                    # No guessed basename or dimension matches: require the exact
                    # original SHA-256 recorded at Nightjar dataset import time.
                    if source not in content_indexes:
                        content_indexes[source] = ExactContentIndex(images)
                    try:
                        original, relative = content_indexes[source].lookup(item)
                    except ValueError as lookup_error:
                        raise ValueError(f"{missing_path}; {lookup_error}") from lookup_error
                    if source == "eBird" and len(relative.parts) < 3:
                        raise ValueError("SHA-matched eBird file lacks species/video/frame structure")
                    match_method = "nightjar_source_sha256"
                if original.suffix.lower() != match.group("ext").lower():
                    raise ValueError(f"Supervisely filename extension differs from original: {annpath.name}")
                destrel = (PurePosixPath(source) / relative).as_posix()
                key = destrel.casefold()
                if key in seen:
                    raise ValueError(f"duplicate original media destination: {destrel}")
                with annpath.open("r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                rec, nv, no, w, h = convert_supervisely(data, destrel, original, item)
                if args.verify_sha256:
                    expected = str(item.get("sha256") or "").strip().lower()
                    if not re.fullmatch(r"[0-9a-f]{64}", expected):
                        raise ValueError(f"source SHA-256 absent or invalid for asset {uid}")
                    if sha256_file(original) != expected:
                        raise ValueError(f"SHA-256 mismatch for original {original}")
                accepted.append({
                    "source": source, "uuid": uid, "original": original,
                    "annotation_file": annpath, "image": destrel, "record": rec,
                    "match_method": match_method,
                    "video_group": str(item.get("video_group") or ""),
                    "frame_index": str(item.get("frame_index") or ""),
                    "width": w, "height": h, "visible_count": nv, "occluded_count": no,
                })
                seen.add(key)
                stats[f"{source}_matched"] += 1
                stats[f"{source}_indexed_kind_{kind}"] += 1
                stats[f"{source}_match_{match_method}"] += 1
                stats[f"{source}_visible_points"] += nv
                stats[f"{source}_occluded_points"] += no
            except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                rejected.append({"source": source, "annotation_file": str(annpath), "reason": str(exc)})
                stats[f"{source}_rejected"] += 1
            if n % 10000 == 0:
                print(f"  {source}: {n:,}/{len(jsons):,}", flush=True)
    report = {
        "schema": "birdgaze-annotatiathon-2026-v2", "state": "AUDITED",
        "accepted": len(accepted), "rejected": len(rejected),
        "counts": dict(stats), "mapping_csv": str(args.media_map),
        "note": "Exact Nightjar UUID mapping, original relative paths with SHA-256 fallback; no filename-order or dimension-based guesses.",
    }
    write_json(report_dir / "validation_report.json", report)
    with (report_dir / "rejected_annotations.csv").open("w", newline="", encoding="utf-8-sig") as out:
        wr = csv.DictWriter(out, fieldnames=("source", "annotation_file", "reason"))
        wr.writeheader()
        wr.writerows(rejected)
    with (report_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as out:
        wr = csv.DictWriter(out, fieldnames=MANIFEST_FIELDS)
        wr.writeheader()
        for item in accepted:
            wr.writerow({k: str(item["original"]) if k == "original_media_path" else
                         str(item["annotation_file"]) if k == "annotation_file" else
                         item[k] for k in MANIFEST_FIELDS})
    print(f"Audit: {len(accepted):,} matched; {len(rejected):,} rejected. Reports: {report_dir}", flush=True)
    if args.audit_only:
        return 0 if not rejected else 2
    if rejected and not args.allow_partial:
        print("Build blocked; inspect reports/rejected_annotations.csv first.", file=sys.stderr)
        return 2
    if not accepted:
        raise ValueError("no valid annotations")
    annot = {"eBird": [], "NABirds": []}
    for n, item in enumerate(accepted, 1):
        put_media(item["original"], output / "images" / item["image"],
                  args.materialize, args.skip_sizes_if_existing)
        annot[item["source"]].append(item["record"])
        if n % 10000 == 0:
            print(f"  Materialized {n:,}/{len(accepted):,}", flush=True)
    ann_dir = output / "annot"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for name, entries in (("ebird.json", annot["eBird"]), ("nabirds.json", annot["NABirds"])):
        write_json(ann_dir / name, entries)
    report["state"] = "BUILT"
    report["images_written"] = len(accepted)
    report["annotations"] = {"ebird.json": len(annot["eBird"]), "nabirds.json": len(annot["NABirds"])}
    write_json(report_dir / "validation_report.json", report)
    print(f"Built {len(accepted):,} matched images; exactly two annotation JSONs in {ann_dir}", flush=True)
    return 0


def main() -> int:
    try:
        args = cli()
        export_available = args.ebird_export.is_dir() and args.nabirds_export.is_dir()
        if args.source_mode == "exports" or (args.source_mode == "auto" and export_available):
            from build_annotatiathon_exports import build_from_exports
            return build_from_exports(args)
        if args.source_mode == "auto" and not args.media_map.is_file():
            raise ValueError(
                "Supervisely export directories were not both found, and the Nightjar mapping CSV "
                "is absent. Supply --ebird-export and --nabirds-export with actual folders."
            )
        return run(args)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
