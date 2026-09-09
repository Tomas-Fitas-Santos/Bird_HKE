"""Create deterministic folder-stratified BirdGaze annotation splits.

The input dataset is expected to contain::

    <dataset-root>/
      annot/train.json
      annot/test.json
      annot/val.json
      images/<folder>/<image>

The three existing annotation files are pooled before splitting. Image paths
inside the JSON records must be relative to ``images`` and must not contain the
``images/`` prefix. By default this script only prints the proposed split. Pass
``--write`` to create ``annot_repro_v1`` beside the original ``annot`` folder.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Sequence, Tuple


SOURCE_SPLITS = ('train', 'test', 'val')
OUTPUT_SPLITS = ('train', 'val', 'calibration')
DEFAULT_SEED = 2026
DEFAULT_RATIOS = (0.8, 0.1, 0.1)


class SplitError(ValueError):
    """Raised when the dataset cannot be split safely."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_image_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SplitError(f'Invalid image path: {value!r}')
    normalized = value.strip().replace('\\', '/')
    while normalized.startswith('./'):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if path.is_absolute() or '..' in path.parts:
        raise SplitError(f'Image path must be relative and stay inside images/: {value!r}')
    if path.parts and path.parts[0].lower() == 'images':
        raise SplitError(
            f'Image path must be relative to images/ and omit that prefix: {value!r}'
        )
    return path.as_posix()


def folder_key(image_path: str) -> str:
    parent = PurePosixPath(image_path).parent.as_posix()
    return parent if parent != '.' else '__images_root__'


def _load_json_list(path: Path) -> List[dict]:
    if not path.is_file():
        raise SplitError(f'Missing annotation file: {path}')
    with path.open('r', encoding='utf-8') as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise SplitError(f'Annotation file must contain a JSON list: {path}')
    return records


def load_source_annotations(
    dataset_root: Path,
    check_images: bool = True,
) -> Tuple[List[dict], Dict[str, dict]]:
    annotation_dir = dataset_root / 'annot'
    images_dir = dataset_root / 'images'
    if check_images and not images_dir.is_dir():
        raise SplitError(f'Missing images directory: {images_dir}')

    pooled: List[dict] = []
    source_info: Dict[str, dict] = {}
    seen: Dict[str, str] = {}
    missing_images: List[str] = []

    for split_name in SOURCE_SPLITS:
        source_path = annotation_dir / f'{split_name}.json'
        records = _load_json_list(source_path)
        source_info[split_name] = {
            'file': f'annot/{split_name}.json',
            'records': len(records),
            'sha256': file_sha256(source_path),
        }
        for index, source_record in enumerate(records):
            if not isinstance(source_record, dict):
                raise SplitError(
                    f'{source_path} record {index} is not a JSON object.'
                )
            if 'image' not in source_record:
                raise SplitError(f'{source_path} record {index} has no image field.')
            record = copy.deepcopy(source_record)
            image_path = normalize_image_path(record['image'])
            if image_path in seen:
                raise SplitError(
                    f'Duplicate image {image_path!r} appears in both '
                    f'{seen[image_path]} and {split_name}.json.'
                )
            seen[image_path] = f'{split_name}.json'
            record['image'] = image_path
            pooled.append(record)
            if check_images and not (images_dir / Path(image_path)).is_file():
                missing_images.append(image_path)

    if missing_images:
        preview = '\n'.join(f'  - {path}' for path in missing_images[:20])
        remainder = len(missing_images) - min(20, len(missing_images))
        suffix = f'\n  ... and {remainder} more' if remainder else ''
        raise SplitError(
            f'{len(missing_images)} annotated image(s) were not found below '
            f'{images_dir}:\n{preview}{suffix}'
        )
    if not pooled:
        raise SplitError('The input annotation files contain no records.')
    return pooled, source_info


def _derived_seed(seed: int, group: str) -> int:
    payload = f'{seed}:{group}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


def _hamilton_counts(size: int, ratios: Sequence[float]) -> List[int]:
    quotas = [size * ratio for ratio in ratios]
    counts = [math.floor(quota) for quota in quotas]
    remaining = size - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (-(quotas[index] - counts[index]), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def allocation_for_large_folder(
    size: int,
    ratios: Sequence[float],
) -> Tuple[int, int, int]:
    counts = _hamilton_counts(size, ratios)
    # A folder is considered large only when both held-out splits can receive
    # one image without reducing the default training allocation below 80%.
    for held_out_index in (1, 2):
        if counts[held_out_index] == 0:
            donor = max(range(3), key=lambda index: counts[index])
            counts[donor] -= 1
            counts[held_out_index] += 1
    return tuple(counts)


def split_annotations(
    records: Iterable[dict],
    seed: int = DEFAULT_SEED,
    ratios: Sequence[float] = DEFAULT_RATIOS,
) -> Tuple[Dict[str, List[dict]], Dict[str, dict]]:
    ratios = tuple(float(value) for value in ratios)
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise SplitError('Exactly three positive train/val/calibration ratios are required.')
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise SplitError(f'Split ratios must sum to 1.0; found {sum(ratios):.12g}.')

    groups: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        groups[folder_key(record['image'])].append(record)

    allocations: Dict[str, Tuple[int, int, int]] = {}
    totals = [0, 0, 0]
    small_groups: List[str] = []
    for group in sorted(groups):
        size = len(groups[group])
        if size <= 4:
            allocation = (size, 0, 0)
        elif size < 10:
            allocation = None
            small_groups.append(group)
            continue
        else:
            allocation = allocation_for_large_folder(size, ratios)
        allocations[group] = allocation
        totals = [totals[index] + allocation[index] for index in range(3)]

    desired = [ratio * sum(len(items) for items in groups.values()) for ratio in ratios]
    for group in small_groups:
        size = len(groups[group])
        val_deficit = desired[1] - totals[1]
        calibration_deficit = desired[2] - totals[2]
        if math.isclose(val_deficit, calibration_deficit, abs_tol=1e-12):
            destination = 1 + (_derived_seed(seed, group) % 2)
        else:
            destination = 1 if val_deficit > calibration_deficit else 2
        allocation_list = [size - 1, 0, 0]
        allocation_list[destination] = 1
        allocation = tuple(allocation_list)
        allocations[group] = allocation
        totals = [totals[index] + allocation[index] for index in range(3)]

    outputs = {name: [] for name in OUTPUT_SPLITS}
    folder_manifest = {}
    for group in sorted(groups):
        shuffled = sorted(groups[group], key=lambda record: record['image'])
        random.Random(_derived_seed(seed, group)).shuffle(shuffled)
        train_count, val_count, calibration_count = allocations[group]
        train_end = train_count
        val_end = train_end + val_count
        outputs['train'].extend(shuffled[:train_end])
        outputs['val'].extend(shuffled[train_end:val_end])
        outputs['calibration'].extend(shuffled[val_end:])
        folder_manifest[group] = {
            'total': len(shuffled),
            'train': train_count,
            'val': val_count,
            'calibration': calibration_count,
            'policy': (
                'training_only' if len(shuffled) <= 4
                else 'one_held_out' if len(shuffled) < 10
                else 'ratio_with_minimum_one_per_held_out_split'
            ),
        }

    for name in OUTPUT_SPLITS:
        outputs[name].sort(key=lambda record: record['image'])

    manifest = {
        'schema_version': 1,
        'protocol': 'bird_hke_folder_stratified_v1',
        'seed': int(seed),
        'ratios': dict(zip(OUTPUT_SPLITS, ratios)),
        'grouping': 'complete parent directory of image path relative to images/',
        'small_folder_policy': {
            '1_to_4_images': 'all training',
            '5_to_9_images': (
                'all but one training; one assigned to the held-out split '
                'with the larger global deficit'
            ),
            '10_or_more_images': (
                'Hamilton apportionment with at least one validation and one '
                'calibration image'
            ),
        },
        'counts': {name: len(outputs[name]) for name in OUTPUT_SPLITS},
        'folders': folder_manifest,
    }
    return outputs, manifest


def write_outputs(
    output_dir: Path,
    outputs: Dict[str, List[dict]],
    manifest: dict,
    source_info: Dict[str, dict],
    force: bool = False,
) -> None:
    output_paths = [output_dir / f'{name}.json' for name in OUTPUT_SPLITS]
    output_paths.append(output_dir / 'split_manifest.json')
    existing = [path for path in output_paths if path.exists()]
    if existing and not force:
        raise SplitError(
            'Refusing to overwrite existing output files without --force:\n'
            + '\n'.join(f'  - {path}' for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_SPLITS:
        path = output_dir / f'{name}.json'
        with path.open('w', encoding='utf-8') as handle:
            json.dump(outputs[name], handle, indent=2, ensure_ascii=False)
            handle.write('\n')

    completed_manifest = copy.deepcopy(manifest)
    completed_manifest['created_utc'] = datetime.now(timezone.utc).isoformat()
    completed_manifest['sources'] = source_info
    completed_manifest['outputs'] = {
        name: {
            'file': f'{name}.json',
            'records': len(outputs[name]),
            'sha256': file_sha256(output_dir / f'{name}.json'),
        }
        for name in OUTPUT_SPLITS
    }
    manifest_path = output_dir / 'split_manifest.json'
    with manifest_path.open('w', encoding='utf-8') as handle:
        json.dump(completed_manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write('\n')


def _print_summary(manifest: dict, output_dir: Path, will_write: bool) -> None:
    counts = manifest['counts']
    total = sum(counts.values())
    print(f'Total unique annotated images: {total}')
    for name in OUTPUT_SPLITS:
        fraction = counts[name] / total if total else 0.0
        print(f'  {name:11s}: {counts[name]:6d} ({fraction:7.2%})')
    folder_values = list(manifest['folders'].values())
    print(f'Folders: {len(folder_values)}')
    print(f'  1-4 images (training only): {sum(item["total"] <= 4 for item in folder_values)}')
    print(f'  5-9 images (one held out):  {sum(5 <= item["total"] < 10 for item in folder_values)}')
    print(f'  10+ images (three-way):     {sum(item["total"] >= 10 for item in folder_values)}')
    if will_write:
        print(f'Wrote reproducible annotations to: {output_dir}')
    else:
        print(f'Dry run only. Add --write to create: {output_dir}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument(
        '--output-annot-dir',
        type=Path,
        help=(
            'Output directory; relative values are resolved below dataset-root '
            '(default: <dataset-root>/annot_repro_v1).'
        ),
    )
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--train-ratio', type=float, default=DEFAULT_RATIOS[0])
    parser.add_argument('--val-ratio', type=float, default=DEFAULT_RATIOS[1])
    parser.add_argument(
        '--calibration-ratio', type=float, default=DEFAULT_RATIOS[2]
    )
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument(
        '--skip-image-check',
        action='store_true',
        help='Do not verify that every annotation resolves below images/.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if args.output_annot_dir:
        requested_output = args.output_annot_dir.expanduser()
        output_dir = (
            requested_output.resolve()
            if requested_output.is_absolute()
            else (dataset_root / requested_output).resolve()
        )
    else:
        output_dir = dataset_root / 'annot_repro_v1'
    source_dir = (dataset_root / 'annot').resolve()
    if output_dir == source_dir:
        raise SplitError(
            'The output directory cannot be the source annot directory. '
            'Write to a separate directory so the original split is preserved.'
        )

    records, source_info = load_source_annotations(
        dataset_root, check_images=not args.skip_image_check
    )
    outputs, manifest = split_annotations(
        records,
        seed=args.seed,
        ratios=(args.train_ratio, args.val_ratio, args.calibration_ratio),
    )
    if args.write:
        write_outputs(output_dir, outputs, manifest, source_info, force=args.force)
    _print_summary(manifest, output_dir, args.write)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (SplitError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)
