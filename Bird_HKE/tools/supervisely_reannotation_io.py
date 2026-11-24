#!/usr/bin/env python3
"""Preflight, optionally upload, download and verify the BirdGaze double-blind sample.

This script never edits the original BirdGaze data. Remote operations require
explicit --execute. Secrets are read from environment variables, not arguments.
Requires Supervisely Python SDK only for upload and download, not preflight or
verification. Intended for project owner, not for individual annotators.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import sys

DEFAULT_SAMPLE = Path(r"E:\Annotatiathon 2026\BirdGaze_Reannotation_2026")
REQUIRED_CLASSES = {
    "bird": "rectangle",
    "crown": "point", "crown_occ": "point",
    "left_eye": "point", "left_eye_occ": "point",
    "right_eye": "point", "right_eye_occ": "point",
    "beak": "point", "beak_occ": "point",
}
KEYPOINT_BASES = ("crown", "left_eye", "right_eye", "beak")

def load_json(p: Path):
    with p.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)

def read_assignments(sample_dir: Path):
    p = sample_dir / "reports" / "assignments.csv"
    if not p.is_file():
        raise FileNotFoundError(f"Missing sample assignment map: {p}")
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = {"sample_id", "annotator", "dataset_name", "image_filename"}
    if not rows or not fields.issubset(rows[0]):
        raise ValueError(f"Assignment manifest missing expected rows/columns: {p}")
    unique = set()
    by_sample = defaultdict(list)
    for row in rows:
        entry = (row["dataset_name"], row["image_filename"])
        if entry in unique:
            raise ValueError(f"Repeated assignment filename within dataset: {entry}")
        unique.add(entry)
        by_sample[row["sample_id"]].append(row)
    for sample_id, assigned in by_sample.items():
        if len(assigned) != 2:
            raise ValueError(f"{sample_id} does not have exactly two assignments")
        if len({r["annotator"] for r in assigned}) != 2:
            raise ValueError(f"{sample_id} is assigned twice to the same person")
        if len({r["image_filename"] for r in assigned}) != 1:
            raise ValueError(f"{sample_id} renamed differently across annotators")
    return rows, by_sample

def validate_classes(project_dir: Path):
    meta_path = project_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Project-level meta.json missing: {meta_path}")
    meta = load_json(meta_path)
    definitions = {
        c.get("title"): c.get("shape") for c in meta.get("classes", [])
        if isinstance(c, dict)
    }
    if any(definitions.get(name) != shape for name, shape in REQUIRED_CLASSES.items()):
        raise ValueError("Project meta.json missing a required class or uses wrong geometry")
    return definitions

def inspect_annotation(path: Path, *, bbox_only: bool):
    ann = load_json(path)
    if not isinstance(ann, dict) or not isinstance(ann.get("objects"), list):
        raise ValueError(f"Invalid Supervisely annotation JSON: {path}")
    objects = ann["objects"]
    boxes = [o for o in objects if o.get("classTitle") == "bird"]
    if len(boxes) != 1 or boxes[0].get("geometryType") != "rectangle":
        raise ValueError(f"Expected exactly one bird rectangle: {path}")
    if bbox_only:
        if len(objects) != 1:
            raise ValueError(f"Upload annotation leaks or includes pre-existing keypoints: {path}")
        return ann, {}
    landmarks = {}
    for obj in objects:
        title = obj.get("classTitle")
        if title == "bird":
            continue
        if title not in REQUIRED_CLASSES or REQUIRED_CLASSES[title] != "point":
            raise ValueError(f"Unknown or unexpected annotation class {title}: {path}")
        if obj.get("geometryType") != "point":
            raise ValueError(f"Keypoint class {title} is not a point: {path}")
        base = title.removesuffix("_occ")
        if base in landmarks:
            raise ValueError(f"Multiple visible/occluded points for {base}: {path}")
        pts = (obj.get("points") or {}).get("exterior")
        if not (isinstance(pts, list) and len(pts) == 1 and
                isinstance(pts[0], list) and len(pts[0]) == 2 and
                all(isinstance(n, (int, float)) for n in pts[0])):
            raise ValueError(f"Invalid keypoint geometry for {title}: {path}")
        landmarks[base] = {"class": title, "x": float(pts[0][0]),
                           "y": float(pts[0][1]),
                           "occluded": title.endswith("_occ")}
    return ann, landmarks

def preflight(sample_dir: Path):
    upload = sample_dir / "supervisely_upload"
    validate_classes(upload)
    rows, by_sample = read_assignments(sample_dir)
    expected_datasets = {r["dataset_name"] for r in rows}
    actual_datasets = {p.name for p in upload.iterdir() if p.is_dir()}
    if actual_datasets != expected_datasets:
        raise ValueError(f"Unexpected or missing dataset folders: "
                         f"expected={sorted(expected_datasets)}, actual={sorted(actual_datasets)}")
    for row in rows:
        dataset = upload / row["dataset_name"]
        image = dataset / "img" / row["image_filename"]
        annotation = dataset / "ann" / (row["image_filename"] + ".json")
        if not image.is_file() or not annotation.is_file():
            raise FileNotFoundError(f"Missing image or box annotation: {image}, {annotation}")
        inspect_annotation(annotation, bbox_only=True)
    counts = Counter(r["annotator"] for r in rows)
    print(f"UPLOAD PREFLIGHT PASS: {len(by_sample):,} independent sample images, "
          f"{len(rows):,} assignments across {len(counts)} annotators")
    for person, count in sorted(counts.items()):
        print(f"  {person}: {count}")
    print(f"Validated original bounding box only, project meta and all image-JSON pairs.")
    return len(by_sample), len(rows)

def read_api():
    server = os.environ.get("SERVER_ADDRESS") or os.environ.get("SUPERVISELY_SERVER_ADDRESS")
    token = os.environ.get("API_TOKEN") or os.environ.get("SUPERVISELY_API_TOKEN")
    if not server or not token:
        raise RuntimeError(
            "Set SERVER_ADDRESS and API_TOKEN (or SUPERVISELY_SERVER_ADDRESS "
            "and SUPERVISELY_API_TOKEN) in this shell; never paste the token in chat."
        )
    try:
        import supervisely as sly
    except ImportError as exc:
        raise RuntimeError("Install the SDK: python -m pip install supervisely") from exc
    return sly, sly.Api(server_address=server, token=token)

def remote_upload(args):
    preflight(args.sample)
    if args.workspace_id is None or not args.project_name:
        raise ValueError("Upload needs --workspace-id and --project-name")
    print(f"Target: workspace {args.workspace_id}, new project {args.project_name!r}")
    if not args.execute:
        print("DRY RUN: no remote action taken; add --execute to upload.")
        return
    sly, api = read_api()
    if any(p.name == args.project_name for p in api.project.get_list(args.workspace_id)):
        raise ValueError("A project with this exact name already exists; refusing duplicate upload")
    project_id, project_name = sly.Project.upload(
        str(args.sample / "supervisely_upload"),
        api, workspace_id=args.workspace_id,
        project_name=args.project_name, log_progress=True,
    )
    print(f"UPLOAD COMPLETE: project_id={project_id}, project_name={project_name!r}")
    print("Now create a separate labeling job for EACH person's dataset in Supervisely.")

def remote_download(args):
    if args.project_id is None:
        raise ValueError("Download needs --project-id")
    dest = args.destination or (args.sample / "completed_exports" / f"project_{args.project_id}")
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty export directory: {dest}")
    print(f"Target: download Supervisely project ID {args.project_id} to {dest}")
    if not args.execute:
        print("DRY RUN: no remote action taken; add --execute to download.")
        return
    sly, api = read_api()
    dest.parent.mkdir(parents=True, exist_ok=True)
    sly.Project.download(api, args.project_id, str(dest), log_progress=True)
    print(f"DOWNLOAD COMPLETE: {dest}")
    print("Next: run the verify-export command with --export-dir.")

def verify_export(sample_dir: Path, export_dir: Path):
    rows, by_sample = read_assignments(sample_dir)
    validate_classes(export_dir)
    results = []
    found = set()
    problems = []
    for row in rows:
        dataset, filename = row["dataset_name"], row["image_filename"]
        ann = export_dir / dataset / "ann" / (filename + ".json")
        image = export_dir / dataset / "img" / filename
        if not ann.is_file() or not image.is_file():
            problems.append({"sample_id": row["sample_id"], "annotator": row["annotator"],
                             "reason": "missing exported annotation or image"})
            continue
        try:
            raw, landmarks = inspect_annotation(ann, bbox_only=False)
            status = "complete" if len(landmarks) == 4 else "missing_points"
            found.add((dataset, filename))
            results.append({
                "sample_id": row["sample_id"],
                "annotator": row["annotator"],
                "dataset_name": dataset,
                "image_filename": filename,
                "source_image": str(image),
                "annotation_file": str(ann),
                "status": status,
                "landmarks": landmarks,
                "annotation": raw,
            })
            if status != "complete":
                problems.append({"sample_id": row["sample_id"], "annotator": row["annotator"],
                                 "reason": f"only {len(landmarks)}/4 head points annotated"})
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            problems.append({"sample_id": row["sample_id"], "annotator": row["annotator"],
                             "reason": str(exc)})
    out = sample_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "reannotation_export_issues.csv").open(
            "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=("sample_id", "annotator", "reason"))
        writer.writeheader()
        writer.writerows(problems)
    output = out / "reannotation_review_records.json"
    temp = output.with_suffix(".json.tmp")
    with temp.open("w", encoding="utf-8") as fh:
        json.dump({
            "schema": "birdgaze-independent-reannotations-v1",
            "expected_assignments": len(rows),
            "found_annotations": len(results),
            "fully_annotated": sum(x["status"] == "complete" for x in results),
            "issues": len(problems),
            "records": results,
        }, fh, ensure_ascii=False, indent=2)
    temp.replace(output)
    print(f"EXPORT CHECK: {len(results):,}/{len(rows):,} matched annotations; "
          f"{sum(x['status'] == 'complete' for x in results):,} fully annotated; "
          f"{len(problems):,} issues")
    print(f"Review output: {output}")
    if problems:
        print(f"Review issues: {out / 'reannotation_export_issues.csv'}")
        return 2
    print("EXPORT CHECK PASS: all assignments present and four points per image.")
    return 0

def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    cmd = p.add_subparsers(dest="command", required=True)
    cmd.add_parser("preflight", help="Check prepared upload data without any network calls")
    up = cmd.add_parser("upload", help="Upload NEW project via Supervisely SDK")
    up.add_argument("--workspace-id", type=int)
    up.add_argument("--project-name", default=None)
    up.add_argument("--execute", action="store_true")
    down = cmd.add_parser("download", help="Download completed project including images and labels")
    down.add_argument("--project-id", type=int)
    down.add_argument("--destination", type=Path)
    down.add_argument("--execute", action="store_true")
    ver = cmd.add_parser("verify-export", help="Validate downloaded independent annotations")
    ver.add_argument("--export-dir", type=Path, required=True)
    return p.parse_args()

def main():
    args = parse()
    if args.command == "preflight":
        preflight(args.sample)
    elif args.command == "upload":
        remote_upload(args)
    elif args.command == "download":
        remote_download(args)
    else:
        return verify_export(args.sample, args.export_dir)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
