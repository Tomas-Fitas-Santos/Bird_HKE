"""Assemble BirdGaze from full Supervisely exports (ann, img, img_info).

The two source export directories are authoritative for image/annotation pairs.
Restore original eBird/NABirds hierarchy when a unique exact SHA256 match exists.
No perceptual or size-only matching; never infer video order from UUIDs.
"""
from __future__ import annotations

import base64
import collections
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

from build_annotatiathon_birdgaze import (
    convert_supervisely, load_media_map, resolve_original, put_media, write_json,
    MEDIA_EXTS, ANNOTATION_NAME, sha256_file,
)


OUTPUT_COLUMNS = (
    "source", "annotation_json", "export_image", "image_info", "image",
    "match_method", "sha256", "original_path", "visible", "occluded",
)
UUID_RX = re.compile(
    r"(?i)^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def directories_named(root: Path, names: set[str]) -> list[Path]:
    result = []
    for folder, child_dirs, _ in os.walk(root):
        for child in child_dirs:
            if child.casefold() in names:
                result.append(Path(folder) / child)
    return sorted(result, key=str)


def annotation_files(export_root: Path, older_ann: Path) -> list[Path]:
    # Prefer the NEW export's annotation files; otherwise reuse the older
    # Supervisely JSON folder and join it to the new exported images by UUID.
    ann_dirs = directories_named(export_root, {"ann"})
    if ann_dirs:
        return sorted((p for d in ann_dirs for p in d.rglob("*.json")), key=str)
    if older_ann.is_dir():
        return sorted(older_ann.rglob("*.json"), key=str)
    raise FileNotFoundError(
        f"Neither an ann/ directory in {export_root} nor older annotations {older_ann} exists"
    )


class ExportInventory:
    def __init__(self, root: Path):
        self.root = root
        self.by_relative: dict[tuple[Path, str], Path] = {}
        self.by_basename: dict[str, list[Path]] = collections.defaultdict(list)
        self.info_by_basename: dict[str, list[Path]] = collections.defaultdict(list)
        img_dirs = directories_named(root, {"img", "images"})
        if not img_dirs:
            # Support a flat export containing JPEGs in the source root.
            img_dirs = [root]
        for img_dir in img_dirs:
            for folder, _, filenames in os.walk(img_dir):
                for filename in filenames:
                    candidate = Path(folder) / filename
                    if candidate.suffix.casefold() not in MEDIA_EXTS:
                        continue
                    relative_name = candidate.relative_to(img_dir).as_posix().casefold()
                    self.by_relative[(img_dir, relative_name)] = candidate
                    self.by_basename[candidate.name.casefold()].append(candidate)
        for info_dir in directories_named(root, {"img_info"}):
            for info_file in info_dir.rglob("*.json"):
                self.info_by_basename[info_file.name.casefold()].append(info_file)
        self.count = sum(len(files) for files in self.by_basename.values())
        if not self.count:
            raise FileNotFoundError(f"no Supervisely-exported images in {root}")

    def locate_image(self, annotation: Path) -> Path:
        image_name = annotation.name[:-5] if annotation.name.endswith(".json") else annotation.name
        if annotation.parent.name.casefold() == "ann":
            image_root = annotation.parent.parent / "img"
            local = image_root / image_name
            if local.is_file():
                return local
            # Some exports use 'images' rather than 'img'.
            local = annotation.parent.parent / "images" / image_name
            if local.is_file():
                return local
        options = self.by_basename.get(image_name.casefold(), [])
        if len(options) == 1:
            return options[0]
        if not options:
            raise ValueError(f"no exported image corresponds to {annotation.name}")
        raise ValueError(f"ambiguous exported filename {image_name}: {len(options)} candidates")

    def locate_info(self, annotation: Path) -> Path | None:
        if annotation.parent.name.casefold() == "ann":
            adjacent = annotation.parent.parent / "img_info" / annotation.name
            if adjacent.is_file():
                return adjacent
        matches = self.info_by_basename.get(annotation.name.casefold(), [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"multiple image-info JSON files for {annotation.name}")
        return None


def parse_image_info_hash(info: dict) -> str | None:
    """Supervisely ImageInfo.hash is base64(sha256(file_content))."""
    value = info.get("hash")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid img_info.hash")
    s = value.strip()
    if re.fullmatch(r"(?i)[0-9a-f]{64}", s):
        return s.lower()
    try:
        decoded = base64.b64decode(s + "=" * (-len(s) % 4), validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("unsupported img_info.hash encoding") from exc
    if len(decoded) != 32:
        raise ValueError(f"img_info.hash has {len(decoded)} decoded bytes, not SHA256")
    return decoded.hex()


class OriginalHashIndex:
    """Build once per source, reading only original files with relevant sizes."""

    def __init__(self, original_root: Path, wanted: dict[int, set[str]]):
        self.root = original_root
        self.by_hash: dict[str, list[Path]] = collections.defaultdict(list)
        candidates = seen = 0
        if not original_root.is_dir() or not wanted:
            return
        print(f"Indexing original {original_root} by exact SHA256 for path restoration", flush=True)
        for folder, _, filenames in os.walk(original_root):
            for filename in filenames:
                f = Path(folder) / filename
                if f.suffix.casefold() not in MEDIA_EXTS:
                    continue
                try:
                    size = f.stat().st_size
                    if size not in wanted:
                        continue
                    candidates += 1
                    if candidates % 2000 == 0:
                        print(f"  Hashed {candidates:,} candidate original images; {seen:,} exact matches found", flush=True)
                    digest = sha256_file(f)
                    if digest in wanted[size]:
                        self.by_hash[digest].append(f)
                        seen += 1
                except OSError:
                    continue
        print(f"  {candidates:,} original candidates hashed, {seen:,} exact matches", flush=True)

    def get_unique(self, digest: str) -> tuple[Path | None, str]:
        results = self.by_hash.get(digest, [])
        if len(results) == 1:
            return results[0], "sha256_original"
        if len(results) > 1:
            return None, "duplicate_original_sha256"
        return None, "no_exact_original_sha256"


def build_from_exports(args) -> int:
    output = args.output.resolve()
    sources = [
        ("eBird", args.ebird_export, args.ebird_ann, args.ebird_images, args.ebird_project_id),
        ("NABirds", args.nabirds_export, args.nabirds_ann, args.nabirds_images, args.nabirds_project_id),
    ]
    for label, export_root, old_ann_root, original_root, _project in sources:
        if not export_root.is_dir():
            raise FileNotFoundError(f"missing new {label} export folder: {export_root}")
        for source_root in (export_root, original_root, old_ann_root):
            if source_root.is_dir():
                src = source_root.resolve()
                if src == output or src in output.parents or output in src.parents:
                    raise ValueError("the output may not be inside or contain any input folder")
    if args.require_original_paths and args.no_original_lookup:
        raise ValueError("--require-original-paths cannot be combined with --no-original-lookup")

    mapping = None
    if not args.ignore_nightjar_map and args.media_map.is_file():
        mapping = load_media_map(args.media_map)
        print(f"Using available Nightjar mapping to restore video/frame paths: {args.media_map}", flush=True)

    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    accepted = []
    rejected = []
    stats = collections.Counter()

    for label, export_root, old_ann_root, original_root, project in sources:
        inventory = ExportInventory(export_root)
        annotations = annotation_files(export_root, old_ann_root)
        stats[f"{label}_annotation_files"] = len(annotations)
        stats[f"{label}_exported_images"] = inventory.count
        print(f"Scanning {label}: {len(annotations):,} annotation JSONs, "
              f"{inventory.count:,} exported images", flush=True)

        # Stage 1: pair annotation JSON to exported image by exact filename
        # and verify its image-info SHA-256 (when available).
        candidates = []
        wanted: dict[int, set[str]] = collections.defaultdict(set)
        for n, annotation_path in enumerate(annotations, 1):
            try:
                image_path = inventory.locate_image(annotation_path)
                info_path = inventory.locate_info(annotation_path)
                img_info = {}
                if info_path is not None:
                    with info_path.open("r", encoding="utf-8-sig") as f:
                        img_info = json.load(f)
                    if not isinstance(img_info, dict):
                        raise ValueError("img_info JSON must be an object")
                    stats[f"{label}_image_info"] += 1
                # Reading the actual image makes this pairing independently auditable.
                image_sha = sha256_file(image_path)
                image_size = image_path.stat().st_size
                expected_sha = parse_image_info_hash(img_info)
                if expected_sha is not None and image_sha != expected_sha:
                    raise ValueError(f"exported image SHA256 differs from img_info.hash: {image_path}")
                if expected_sha is not None:
                    stats[f"{label}_verified_image_info_sha256"] += 1
                uuid_match = ANNOTATION_NAME.fullmatch(annotation_path.name)
                uid = uuid_match.group("uuid").lower() if uuid_match else ""
                candidates.append({
                    "ann": annotation_path, "image": image_path, "info": info_path,
                    "sha": image_sha, "size": image_size, "uuid": uid,
                })
                if not args.no_original_lookup and original_root.is_dir():
                    wanted[image_size].add(image_sha)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                rejected.append({"source": label, "annotation_file": str(annotation_path),
                                 "reason": str(exc)})
                stats[f"{label}_rejected"] += 1
            if n % 2000 == 0:
                print(f"  {label}: {n:,}/{len(annotations):,} export image/info pairs verified", flush=True)

        # Stage 2: exact source-path restoration from Nightjar UUID (if available)
        # or a single-pass SHA-256 index of the original files.
        original_index = (
            OriginalHashIndex(original_root, wanted)
            if not args.no_original_lookup else None
        )
        reserved_paths: set[str] = set()
        for n, candidate in enumerate(candidates, 1):
            annotation_path = candidate["ann"]
            try:
                uid = candidate["uuid"]
                exported = candidate["image"]
                digest = candidate["sha"]
                relative: PurePosixPath | None = None
                physical = exported
                method = "supervisely_export"

                if not args.no_original_lookup and original_root.is_dir():
                    mapped = mapping.get((str(project), uid)) if mapping and uid else None
                    if mapped is not None:
                        try:
                            mapped_file, mapped_relative = resolve_original(mapped, original_root, label)
                            if (mapped_file.stat().st_size == candidate["size"]
                                    and sha256_file(mapped_file) == digest):
                                physical, relative, method = (
                                    mapped_file, PurePosixPath(mapped_relative.as_posix()),
                                    "nightjar_uuid_sha256",
                                )
                            else:
                                stats[f"{label}_nightjar_hash_mismatch"] += 1
                        except (ValueError, OSError):
                            stats[f"{label}_nightjar_path_missing"] += 1
                    if relative is None and original_index is not None:
                        original, reason = original_index.get_unique(digest)
                        if original is not None:
                            physical = original
                            relative = PurePosixPath(original.relative_to(original_root).as_posix())
                            method = reason
                        else:
                            stats[f"{label}_{reason}"] += 1

                if relative is None:
                    if args.require_original_paths:
                        raise ValueError(f"no unique original file identified by SHA256: {digest}")
                    relative = PurePosixPath("exported", exported.name)
                    # If different datasets contain the same UUID, refuse rather
                    # than silently overwriting the output file.
                if label == "eBird" and method != "supervisely_export" and len(relative.parts) < 3:
                    raise ValueError("resolved eBird frame lacks species/video/frame hierarchy")
                dest_name = (PurePosixPath(label) / relative).as_posix()
                if dest_name.casefold() in reserved_paths:
                    raise ValueError(f"duplicate output path: {dest_name}")
                with annotation_path.open("r", encoding="utf-8-sig") as f:
                    raw = json.load(f)
                # Exported annotation must match the SAME exported pixel matrix.
                rec, nv, no, w, h = convert_supervisely(raw, dest_name, exported, {})
                # For original byte-identical matches, dimensions necessarily agree.
                accepted.append({
                    "source": label, "annotation_json": str(annotation_path),
                    "export_image": str(exported), "image_info": str(candidate["info"] or ""),
                    "original_path": str(physical) if method != "supervisely_export" else "",
                    "image": dest_name, "record": rec, "source_file": physical,
                    "match_method": method, "sha256": digest,
                    "visible": nv, "occluded": no, "width": w, "height": h,
                })
                reserved_paths.add(dest_name.casefold())
                stats[f"{label}_accepted"] += 1
                stats[f"{label}_match_{method}"] += 1
                stats[f"{label}_visible_joints"] += nv
                stats[f"{label}_occluded_joints"] += no
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                rejected.append({"source": label, "annotation_file": str(annotation_path),
                                 "reason": str(exc)})
                stats[f"{label}_rejected"] += 1
            if n % 10000 == 0:
                print(f"  {label}: {n:,}/{len(candidates):,} converted", flush=True)

    summary = {
        "schema": "birdgaze-export-hash-2026-v1",
        "state": "AUDITED",
        "annotation_files": {
            "ebird": stats["eBird_annotation_files"],
            "nabirds": stats["NABirds_annotation_files"],
        },
        "accepted": len(accepted),
        "rejected": len(rejected),
        "statistics": dict(stats),
        "source_mode": "supervisely_exports",
        "image_info_hash_scheme": "base64(sha256(file_content))",
        "note": "UUID-based export images are valid BirdGaze sources; without exact original mapping, their video temporal identity remains unknown.",
    }
    write_json(reports / "validation_report.json", summary)
    with (reports / "rejected_annotations.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=("source", "annotation_file", "reason"))
        wr.writeheader()
        wr.writerows(rejected)
    with (reports / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        wr.writeheader()
        for entry in accepted:
            wr.writerow({k: entry[k] for k in OUTPUT_COLUMNS})

    print(f"Audit: {len(accepted):,} matched, {len(rejected):,} rejected "
          f"(reports in {reports})", flush=True)
    for source_label in ("eBird", "NABirds"):
        for k in ("accepted", "rejected", "match_nightjar_uuid_sha256",
                  "match_sha256_original", "match_supervisely_export"):
            print(f"  {source_label} {k}: {stats[source_label + '_' + k]:,}", flush=True)

    if args.audit_only:
        return 0 if not rejected else 2
    if rejected and not args.allow_partial:
        print("Build blocked by rejected records; inspect rejected_annotations.csv", file=sys.stderr)
        return 2
    if not accepted:
        raise ValueError("no valid annotation/image pairs")
    outputs: dict[str, list[dict]] = {"eBird": [], "NABirds": []}
    for n, item in enumerate(accepted, 1):
        put_media(item["source_file"], output / "images" / item["image"],
                  args.materialize, args.skip_sizes_if_existing)
        outputs[item["source"]].append(item["record"])
        if n % 10000 == 0:
            print(f"  Copied or linked {n:,}/{len(accepted):,}", flush=True)
    annot = output / "annot"
    annot.mkdir(parents=True, exist_ok=True)
    for label, dest in (("eBird", "ebird.json"), ("NABirds", "nabirds.json")):
        write_json(annot / dest, outputs[label])
    summary.update({
        "state": "BUILT",
        "images_materialized": len(accepted),
        "annotation_counts": {"ebird.json": len(outputs["eBird"]),
                              "nabirds.json": len(outputs["NABirds"])},
    })
    write_json(reports / "validation_report.json", summary)
    print(f"Built {len(accepted):,} images and exactly ebird.json, nabirds.json", flush=True)
    return 0
