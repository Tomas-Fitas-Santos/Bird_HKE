#!/usr/bin/env python3
"""Build a blind, double-annotated eBird/NABirds Supervisely reannotation sample.

Inputs: BirdGaze_Annotatiathon_2026/images and annot/{ebird,nabirds}.json.
Outputs: Supervisely import project with per-person datasets containing ONLY
the bird bounding box, plus private reference annotations and assignment CSVs.

All selected images must have a validated event annotation. No sources change.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import hashlib
import json
import math
import random
import re
import shutil
import sys
from pathlib import Path, PurePosixPath
import zipfile

from PIL import Image

DEFAULT_ROOT = Path(r"E:\Annotatiathon 2026\BirdGaze_Annotatiathon_2026")
DEFAULT_OUTPUT = Path(r"E:\Annotatiathon 2026\BirdGaze_Reannotation_2026")

ORIGINAL_QUOTAS = {
    "Paulo Lobato Correia": 100,
    "Ricardo Araújo": 200,
    "Luís Almeida": 100,
    "Tomás Santos": 300,
    "Riccardo Rocchi": 500,
    "Filippo Bertozzo": 400,
    "Romain David": 200,
    "Miguel Marx": 200,
    "Victor Carvalho": 200,
}

# Preserve Nightjar's nine annotation classes, but initialize labels with
# bird rectangles only. Annotators can add head landmarks independently.
CLASSES = (
    ("bird", "rectangle", "#FFDF00"),
    ("left_eye", "point", "#FF21A8"),
    ("left_eye_occ", "point", "#CC0078"),
    ("right_eye", "point", "#BC46FF"),
    ("right_eye_occ", "point", "#8424C8"),
    ("beak", "point", "#FF6D00"),
    ("beak_occ", "point", "#C54A00"),
    ("crown", "point", "#FF3030"),
    ("crown_occ", "point", "#C91D35"),
)

@dataclass(frozen=True)
class Frame:
    source: str
    image: str
    species: str
    video: str
    filename: str
    record: dict
    pose: tuple[tuple[float, float, int, bool], ...]
    bbox: tuple[float, float, float, float]

def pose_features(record: dict, bbox: tuple[float, float, float, float]):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    joints, vis = record["joints"], record["joints_vis"]
    if len(joints) != 4 or len(vis) != 4:
        raise ValueError("expected 4 head keypoints in BirdGaze record")
    result = []
    for point, v in zip(joints, vis):
        has_xy = (isinstance(point, (list, tuple)) and len(point) == 2
                  and all(isinstance(c, (int, float)) and math.isfinite(c) for c in point)
                  and point[0] >= 0 and point[1] >= 0)
        if has_xy:
            result.append(((float(point[0]) - x1) / width,
                           (float(point[1]) - y1) / height,
                           int(bool(v)), True))
        else:
            result.append((0.0, 0.0, 0, False))
    return tuple(result)

def pose_distance(a: Frame, b: Frame) -> float:
    """Geometry within the bird box, weighted down for occluded landmarks."""
    squared = 0.0
    weight_sum = 0.0
    visibility_disagreements = 0
    for (ax, ay, av, ap), (bx, by, bv, bp) in zip(a.pose, b.pose):
        if ap and bp:
            weight = 1.0 if av and bv else 0.5
            squared += weight * ((ax - bx) ** 2 + (ay - by) ** 2)
            weight_sum += weight
            visibility_disagreements += int(bool(av) != bool(bv))
        elif ap != bp:
            squared += 0.3
            weight_sum += 1.0
    return math.sqrt(squared / max(1.0, weight_sum)) + 0.05 * visibility_disagreements

def frame_sort_key(frame: Frame):
    digits = re.findall(r"\d+", Path(frame.filename).stem)
    return (int(digits[-1]) if digits else -1, frame.filename.casefold())

def load_frames(root: Path, include_nabirds: bool, minimum_points: int):
    grouped = defaultdict(list)
    coverage = Counter()
    names = (("eBird", "ebird.json"), ("NABirds", "nabirds.json"))
    for source, filename in names:
        if source == "NABirds" and not include_nabirds:
            continue
        path = root / "annot" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing validated BirdGaze annotations: {path}")
        with path.open(encoding="utf-8-sig") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"Expected a list in {path}")
        seen_images = set()
        for rec in records:
            coverage[f"{source}_read"] += 1
            if not isinstance(rec, dict) or not isinstance(rec.get("image"), str):
                coverage[f"{source}_malformed"] += 1
                continue
            rel = PurePosixPath(rec["image"].replace("\\", "/"))
            if (rel.is_absolute() or ".." in rel.parts
                    or len(rel.parts) < 3 or rel.parts[0] != source
                    or ":" in rec["image"]):
                coverage[f"{source}_invalid_path"] += 1
                continue
            if source == "eBird" and (len(rel.parts) < 4 or rel.parts[1] == "exported"):
                # Original video identity is essential for sampling frames.
                coverage["eBird_no_verified_video"] += 1
                continue
            if str(rel) in seen_images:
                raise ValueError(f"Duplicate image in {filename}: {rel}")
            seen_images.add(str(rel))
            try:
                bbox = tuple(float(v) for v in rec["bbox"])
                if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
                    raise ValueError("bad bbox")
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    raise ValueError("degenerate bbox")
                pose = pose_features(rec, bbox)
                count = sum(1 for p in pose if p[3])
                if count < minimum_points:
                    coverage[f"{source}_too_few_points"] += 1
                    continue
            except (ValueError, TypeError, KeyError):
                coverage[f"{source}_invalid_annotation"] += 1
                continue
            # eBird: video ID is complete hierarchy until the frame file.
            # NABirds: group by species; independent photos need no temporal gap.
            species = rel.parts[1]
            video = "/".join(rel.parts[1:-1]) if source == "eBird" else ""
            frame = Frame(
                source=source, image=str(rel), species=species,
                video=video, filename=rel.name, record=rec,
                pose=pose, bbox=bbox,
            )
            grouped[(source, species, video)].append(frame)
            coverage[f"{source}_eligible"] += 1
    return grouped, coverage

def diversified_video_frames(frames: list[Frame], limit: int, frame_gap: int,
                             minimum_pose_distance: float, rng: random.Random):
    """Greedy farthest-pose-point selection, with temporal separation."""
    ordered = sorted(frames, key=frame_sort_key)
    picked = []
    used_indices = []
    # Random first frame, using fixed seed for full reproducibility.
    if not ordered:
        return []
    for _ in range(min(limit, len(ordered))):
        scores = []
        for i, frame in enumerate(ordered):
            if i in used_indices or any(abs(i - used) < frame_gap for used in used_indices):
                continue
            if picked:
                min_dist = min(pose_distance(frame, old) for old in picked)
                if min_dist < minimum_pose_distance:
                    continue
                score = min_dist
            else:
                # First frame should not always be the first frame in each video.
                score = rng.random()
            scores.append((score, rng.random(), i))
        if not scores:
            break
        _, _, index = max(scores)
        picked.append(ordered[index])
        used_indices.append(index)
    return picked

def diversify_photos(frames: list[Frame], rng: random.Random):
    # Keep one item per independent NABirds photo. Source images are already
    # labeled, and balancing by species happens at the global selector level.
    result = list(frames)
    rng.shuffle(result)
    return result

def sample_diverse(groups, n_ebird: int, n_nabirds: int, *,
                   max_per_video: int, min_frame_gap: int,
                   min_pose_distance: float, seed: int):
    rng = random.Random(seed)
    by_source_species = defaultdict(list)
    capacities = Counter()
    for (source, species, video), frames in sorted(groups.items()):
        picks = (
            diversified_video_frames(frames, max_per_video, min_frame_gap,
                                     min_pose_distance, rng)
            if source == "eBird" else diversify_photos(frames, rng)
        )
        if picks:
            by_source_species[(source, species)].append({
                "video": video, "frames": picks, "index": 0, "used": 0,
            })
            capacities[source] += len(picks)

    def draw(source: str, target: int):
        selected = []
        counters = Counter()
        pool = {species: groups for (src, species), groups in by_source_species.items()
                if src == source}
        for group in pool.values():
            rng.shuffle(group)
        while len(selected) < target:
            # Each species gets a turn before increasing another species'
            # allocation, unless that species is exhausted.
            eligible = []
            for species, videos in pool.items():
                available = [v for v in videos if v["index"] < len(v["frames"])]
                if available:
                    eligible.append((counters[species], rng.random(), species, available))
            if not eligible:
                raise ValueError(
                    f"Cannot find {target} diverse {source} items under current thresholds; "
                    f"only {len(selected)} selected. Capacity={capacities[source]}. "
                    "Relax --min-pose-distance or --min-frame-gap, or raise --max-per-video."
                )
            _, _, species, options = min(eligible, key=lambda item: (item[0], item[1]))
            video = min(options, key=lambda item: (item["used"], rng.random()))
            selected.append(video["frames"][video["index"]])
            video["index"] += 1
            video["used"] += 1
            counters[species] += 1
        return selected

    selected = draw("eBird", n_ebird) + draw("NABirds", n_nabirds)
    assert len(selected) == n_ebird + n_nabirds
    return selected, capacities

def quotas_from_args(args) -> dict[str, int]:
    if args.quotas_json:
        with args.quotas_json.open(encoding="utf-8-sig") as handle:
            quotas = json.load(handle)
        if not isinstance(quotas, dict):
            raise ValueError("--quotas-json must contain an object of names to integer counts")
    else:
        quotas = {name: round(q * args.quota_scale)
                  for name, q in ORIGINAL_QUOTAS.items()}
        if args.tomas_daily_cap is not None:
            quotas["Tomás Santos"] = min(quotas["Tomás Santos"],
                                        int(args.tomas_daily_cap * args.days))
    if len(quotas) < 2 or any(
            not isinstance(k, str) or not k.strip()
            or type(v) is not int or v < 0 for k, v in quotas.items()):
        raise ValueError("Quotas must be a mapping of people to nonnegative integer totals")
    quotas = {k: v for k, v in quotas.items() if v}
    total = sum(quotas.values())
    if total <= 0 or total % 2:
        raise ValueError(f"Total assignments must be a positive even number, got {total}")
    if max(quotas.values()) > total // 2:
        raise ValueError("One annotator has more assignments than there are unique frames")
    return quotas

def pairing(quotas: dict[str, int], rng: random.Random):
    """Guaranteed exact degree quotas and no self-pairs; random edge swaps."""
    count = sum(quotas.values()) // 2
    slots = []
    people = list(quotas)
    rng.shuffle(people)
    for person in people:
        slots.extend([person] * quotas[person])
    pairs = [[slots[i], slots[i + count]] for i in range(count)]
    if any(a == b for a, b in pairs):
        raise ValueError("Internal pairing construction error")
    # Randomize endpoints across edges, preserving exact degrees and two
    # different annotators per item.
    for _ in range(max(1, 15 * count)):
        i, j = rng.randrange(count), rng.randrange(count)
        if i == j:
            continue
        side_i, side_j = rng.randrange(2), rng.randrange(2)
        ai, aj = pairs[i][1-side_i], pairs[j][1-side_j]
        bi, bj = pairs[i][side_i], pairs[j][side_j]
        if ai != bj and aj != bi:
            pairs[i][side_i], pairs[j][side_j] = bj, bi
    rng.shuffle(pairs)
    return [tuple(p) for p in pairs]

def project_meta():
    return {
        "projectType": "images",
        "classes": [
            {"title": name, "shape": shape, "color": color, "geometry_config": {}}
            for name, shape, color in CLASSES
        ],
        "tags": [],
    }

def image_annotation(size: tuple[int, int], bbox: tuple[float, float, float, float]):
    width, height = size
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Bounding box outside image {size}: {bbox}")
    # Convert to Supervisely rectangle geometry, with NO original keypoints.
    return {
        "description": "",
        "tags": [],
        "size": {"width": height if False else width, "height": height},
        "objects": [{
            "classTitle": "bird",
            "description": "",
            "tags": [],
            "geometryType": "rectangle",
            "points": {"exterior": [[x1, y1], [x2, y2]], "interior": []},
        }],
    }

def slug_name(s: str):
    replacements = str.maketrans("áàãâäéèêëíìîïóòôõöúùûüçñÁÉÍÓÚ",
                                 "aaaaaeeeeiiiiooooouuuucnAEIOU")
    clean = s.translate(replacements)
    return re.sub(r"[^A-Za-z0-9]+", "_", clean).strip("_")

def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temp.replace(path)

def save_csv(path: Path, rows: list[dict], fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def copy_image(src: Path, dest: Path, mode: str):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        dest.hardlink_to(src)
    else:
        shutil.copy2(src, dest)

def check_frame_sources(root: Path, selected: list[Frame]):
    for i, frame in enumerate(selected, 1):
        source = root / "images" / Path(frame.image)
        if not source.is_file():
            raise FileNotFoundError(f"Sample source image does not exist: {source}")
        if i % 500 == 0:
            print(f"  Verified {i:,}/{len(selected):,} selected input image paths", flush=True)

def export(args, selected, pairs, quotas, capacities, coverage):
    rng = random.Random(args.seed ^ 0xBAD5EED)
    rng.shuffle(selected)
    check_frame_sources(args.dataset, selected)
    output = args.output.resolve()
    src_root = args.dataset.resolve()
    if output == src_root or output in src_root.parents or src_root in output.parents:
        raise ValueError("Output must be separate from original BirdGaze dataset")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output already exists and is not empty: {output}")
    if args.dry_run:
        print(f"DRY RUN: would export {len(selected):,} unique frames and "
              f"{sum(quotas.values()):,} independent assignments to {output}", flush=True)
        return
    out_import = output / "supervisely_upload"
    out_private = output / "private_reference"
    out_report = output / "reports"
    out_import.mkdir(parents=True, exist_ok=False)
    save_json(out_import / "meta.json", project_meta())
    sample_rows, assignment_rows, private_records = [], [], {}
    quota_done = Counter()
    pair_counts = Counter()
    selected_videos = Counter()
    selected_species = Counter()
    for n, (frame, pair) in enumerate(zip(selected, pairs), 1):
        rel = Path(frame.image)
        src = args.dataset / "images" / rel
        sample_id = f"F{n:05d}_{hashlib.sha256(frame.image.encode('utf-8')).hexdigest()[:12]}"
        filename = sample_id + rel.suffix.lower()
        with Image.open(src) as img:
            wh = img.size
        ann = image_annotation(wh, frame.bbox)
        private_records[sample_id] = {
            "source": frame.source, "image": frame.image,
            "species": frame.species, "video": frame.video,
            "event_reference": frame.record,
            "annotator_a": pair[0], "annotator_b": pair[1],
        }
        sample_rows.append({
            "sample_id": sample_id, "source": frame.source, "original_image": frame.image,
            "species": frame.species, "video": frame.video,
            "filename": filename, "annotator_a": pair[0], "annotator_b": pair[1],
        })
        selected_species[f"{frame.source}/{frame.species}"] += 1
        if frame.video:
            selected_videos[frame.video] += 1
        pair_counts[tuple(sorted(pair))] += 1
        for person in pair:
            dataset_slug = slug_name(person)
            person_dir = out_import / dataset_slug
            dest = person_dir / "img" / filename
            copy_image(src, dest, args.materialize)
            save_json(person_dir / "ann" / (filename + ".json"), ann)
            quota_done[person] += 1
            assignment_rows.append({
                "sample_id": sample_id, "annotator": person,
                "dataset_name": dataset_slug, "image_filename": filename,
                "status": "assigned",
            })
        if n % 250 == 0:
            print(f"  Exported {n:,}/{len(selected):,} doubly assigned images", flush=True)
    if quota_done != Counter(quotas):
        raise AssertionError(f"Assignment quotas mismatch: {quota_done} != {quotas}")
    save_csv(out_private / "selected_frames.csv", sample_rows, (
        "sample_id", "source", "original_image", "species", "video", "filename",
        "annotator_a", "annotator_b",
    ))
    save_json(out_private / "event_reference_annotations.json", private_records)
    save_csv(out_report / "assignments.csv", assignment_rows, (
        "sample_id", "annotator", "dataset_name", "image_filename", "status",
    ))
    save_csv(out_report / "workloads.csv", [
        {"annotator": person, "assigned": quota_done[person],
         "per_week": quota_done[person] / 2,
         "per_day_14_days": round(quota_done[person] / args.days, 2)}
        for person in quotas
    ], ("annotator", "assigned", "per_week", "per_day_14_days"))
    save_csv(out_report / "pair_counts.csv", [
        {"annotator_a": pair[0], "annotator_b": pair[1], "count": count}
        for pair, count in sorted(pair_counts.items())
    ], ("annotator_a", "annotator_b", "count"))
    report = {
        "schema": "birdgaze-double-blind-reannotation-v1",
        "unique_images": len(selected),
        "independent_assignments": sum(quotas.values()),
        "number_of_annotators_per_image": 2,
        "quotas": quotas,
        "unique_species": len(selected_species),
        "unique_videos": len(selected_videos),
        "species_counts": dict(sorted(selected_species.items())),
        "video_counts": dict(sorted(selected_videos.items())),
        "coverage": dict(coverage),
        "pose_diversity": {
            "minimum_normalized_keypoint_distance_within_video": args.min_pose_distance,
            "minimum_sorted_frame_index_gap_within_video": args.min_frame_gap,
            "maximum_frames_per_video": args.max_per_video,
            "method": "greedy farthest-point normalized 2D head keypoint geometry",
        },
        "source_eligible_diversity_capacity": dict(capacities),
        "random_seed": args.seed,
        "import_folder": str(out_import),
        "reference_is_private_and_excluded_from_import": True,
        "only_bird_bbox_present_in_uploaded_annotations": True,
        "zero_existing_keypoints_disclosed_to_annotators": True,
        "warning": "Existing event annotations are not independent gold ground truth; "
                   "compute agreement between the two assigned reviewers and compare "
                   "each independently to the event reference. Assigner cannot know "
                   "the original annotator from BirdGaze records alone.",
    }
    save_json(out_report / "selection_report.json", report)
    if args.make_zip:
        target = output / "supervisely_upload.zip"
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1, allowZip64=True) as archive:
            for f in sorted(out_import.rglob("*")):
                if f.is_file():
                    archive.write(f, arcname=f.relative_to(out_import).as_posix())
        print(f"Supervisely archive: {target}", flush=True)
    print(f"DONE: {len(selected):,} unique images, {sum(quotas.values()):,} assignments; "
          f"{len(selected_species)} species, {len(selected_videos)} eBird videos", flush=True)
    print(f"UPLOAD ONLY: {out_import}", flush=True)
    print(f"PRIVATE (DO NOT UPLOAD): {out_private}", flush=True)

def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--quota-scale", type=float, default=2.0)
    p.add_argument("--quotas-json", type=Path, default=None)
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--tomas-daily-cap", type=float, default=None,
                   help="limit Tomás's assignments to this many images per day")
    p.add_argument("--nabirds-share", type=float, default=0.0,
                   help="fraction of unique sample from NABirds photos; 0 selects only eBird video frames")
    p.add_argument("--max-per-video", type=int, default=12)
    p.add_argument("--min-frame-gap", type=int, default=6,
                   help="minimum separation by sorted original frames within each video")
    p.add_argument("--min-pose-distance", type=float, default=0.10,
                   help="minimum normalized 2D head pose distance within each video")
    p.add_argument("--min-keypoints", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261124)
    p.add_argument("--materialize", choices=("copy", "hardlink"), default="copy")
    p.add_argument("--make-zip", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()

def main():
    args = parse()
    if args.days < 1 or args.quota_scale <= 0 or not 0 <= args.nabirds_share <= 1:
        raise ValueError("Invalid days, quota scale or NABirds share")
    if args.max_per_video < 1 or args.min_frame_gap < 1 or args.min_pose_distance < 0:
        raise ValueError("Invalid video diversity constraints")
    if args.min_keypoints < 1 or args.min_keypoints > 4:
        raise ValueError("--min-keypoints must be from 1 to 4")
    quotas = quotas_from_args(args)
    total = sum(quotas.values()) // 2
    n_nabirds = round(total * args.nabirds_share)
    n_ebird = total - n_nabirds
    print(f"Target: {total:,} unique images / {2 * total:,} independent annotations "
          f"({n_ebird:,} eBird frames, {n_nabirds:,} NABirds photos)", flush=True)
    for name, quota in quotas.items():
        print(f"  {name}: {quota:,} ({quota / args.days:.1f} per day)", flush=True)
    grouped, coverage = load_frames(args.dataset, bool(n_nabirds), args.min_keypoints)
    print(f"Eligible validated records: {dict(coverage)}", flush=True)
    selected, capacities = sample_diverse(
        grouped, n_ebird, n_nabirds, max_per_video=args.max_per_video,
        min_frame_gap=args.min_frame_gap,
        min_pose_distance=args.min_pose_distance, seed=args.seed,
    )
    pairs = pairing(quotas, random.Random(args.seed + 123))
    export(args, selected, pairs, quotas, capacities, coverage)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
