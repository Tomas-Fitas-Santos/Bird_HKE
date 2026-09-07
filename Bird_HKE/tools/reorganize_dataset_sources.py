"""Copy a flat BirdGaze dataset into a source-organized dataset.

Expected input layout::

    <dataset-root>/
      annot/train.json
      annot/test.json
      annot/val.json
      images/<original-folder>/<image>

Annotation ``image`` values must be relative to ``images/``. The output keeps
the original folder names but inserts one of four source directories::

    images/Animal_Kingdom/<UPPERCASE-folder>/...
    images/eBird/FINETUNE/...
    images/NABirds/<numeric-folder>/...
    images/Birdsnap/<all-other-folders>/...

The source dataset is never modified. By default the command performs a dry
run. Pass ``--write`` to create a sibling dataset named ``<name>_by_source``,
or select another location with ``--output-root``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple, TypeVar


ANNOTATION_FILES = ('train.json', 'test.json', 'val.json')
SOURCE_NAMES = ('Animal_Kingdom', 'eBird', 'NABirds', 'Birdsnap')
IMAGE_EXTENSIONS = frozenset(
    {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.gif', '.ppm', '.pgm'}
)
T = TypeVar('T')


class DatasetFormatError(ValueError):
    """Raised when a source-organized copy cannot be created safely."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_image_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetFormatError(f'Invalid image path: {value!r}')
    normalized = value.strip().replace('\\', '/')
    while normalized.startswith('./'):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if path.is_absolute() or '..' in path.parts:
        raise DatasetFormatError(
            f'Image path must be relative and stay inside images/: {value!r}'
        )
    if path.parts and path.parts[0].lower() == 'images':
        raise DatasetFormatError(
            f'Image path must omit the images/ prefix: {value!r}'
        )
    if len(path.parts) < 2:
        raise DatasetFormatError(
            f'Image must be inside a top-level source folder: {value!r}'
        )
    return path.as_posix()


def classify_folder(folder_name: str) -> str:
    """Classify one original top-level folder using the agreed precedence."""
    if folder_name == 'FINETUNE':
        return 'eBird'
    if re.fullmatch(r'[A-Z]+', folder_name):
        return 'Animal_Kingdom'
    if re.fullmatch(r'[0-9]+', folder_name):
        return 'NABirds'
    return 'Birdsnap'


def _progress(
    iterable: Iterable[T], total: int, description: str, enabled: bool = True
) -> Iterator[T]:
    """Use tqdm when installed and a small dependency-free fallback otherwise."""
    if not enabled:
        yield from iterable
        return
    try:
        from tqdm import tqdm
    except ImportError:
        interval = max(1, total // 20)
        print(f'{description}: 0/{total}')
        for index, value in enumerate(iterable, start=1):
            yield value
            if index == total or index % interval == 0:
                print(f'{description}: {index}/{total}')
        return
    yield from tqdm(iterable, total=total, desc=description, unit='image')


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def inventory_images(dataset_root: Path) -> Tuple[List[dict], dict]:
    images_dir = dataset_root / 'images'
    if not images_dir.is_dir():
        raise DatasetFormatError(f'Missing images directory: {images_dir}')

    top_level_files = sorted(path for path in images_dir.iterdir() if path.is_file())
    if top_level_files:
        preview = '\n'.join(f'  - {path.name}' for path in top_level_files[:20])
        raise DatasetFormatError(
            'Every image must be below an original top-level folder. Found '
            f'{len(top_level_files)} file(s) directly in images/:\n{preview}'
        )

    folders = sorted(path for path in images_dir.iterdir() if path.is_dir())
    if not folders:
        raise DatasetFormatError(f'No source folders found in: {images_dir}')

    already_formatted = [path.name for path in folders if path.name in SOURCE_NAMES]
    if already_formatted:
        raise DatasetFormatError(
            'The source appears to be already organized; refusing to add a second '
            f'source level. Found: {", ".join(already_formatted)}'
        )

    entries: List[dict] = []
    folder_counts: Counter = Counter()
    image_counts: Counter = Counter()
    byte_counts: Counter = Counter()
    ignored_non_images = 0

    for folder in folders:
        source_name = classify_folder(folder.name)
        folder_counts[source_name] += 1
        for path in sorted(folder.rglob('*')):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                ignored_non_images += 1
                continue
            old_relative = path.relative_to(images_dir).as_posix()
            new_relative = (PurePosixPath(source_name) / old_relative).as_posix()
            size = path.stat().st_size
            entries.append(
                {
                    'source_path': path,
                    'old_relative': old_relative,
                    'new_relative': new_relative,
                    'source_name': source_name,
                    'size': size,
                }
            )
            image_counts[source_name] += 1
            byte_counts[source_name] += size

    if not entries:
        raise DatasetFormatError(
            f'No recognized image files found below {images_dir}. Supported '
            f'extensions: {", ".join(sorted(IMAGE_EXTENSIONS))}'
        )
    entries.sort(key=lambda item: item['old_relative'])
    summary = {
        'top_level_folders': dict(folder_counts),
        'images': dict(image_counts),
        'bytes': dict(byte_counts),
        'ignored_non_image_files': ignored_non_images,
    }
    return entries, summary


def load_and_rewrite_annotations(
    dataset_root: Path, entries: Sequence[dict]
) -> Tuple[Dict[str, List[dict]], dict]:
    annotation_dir = dataset_root / 'annot'
    path_map = {entry['old_relative']: entry for entry in entries}
    rewritten: Dict[str, List[dict]] = {}
    file_info: Dict[str, dict] = {}
    seen: Dict[str, str] = {}
    missing: List[str] = []
    annotated_by_source: Counter = Counter()

    for filename in ANNOTATION_FILES:
        path = annotation_dir / filename
        if not path.is_file():
            raise DatasetFormatError(f'Missing annotation file: {path}')
        with path.open('r', encoding='utf-8') as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise DatasetFormatError(f'Annotation file must contain a JSON list: {path}')

        output_records: List[dict] = []
        for index, source_record in enumerate(records):
            if not isinstance(source_record, dict):
                raise DatasetFormatError(f'{path} record {index} is not a JSON object.')
            if 'image' not in source_record:
                raise DatasetFormatError(f'{path} record {index} has no image field.')
            old_relative = normalize_image_path(source_record['image'])
            if old_relative in seen:
                raise DatasetFormatError(
                    f'Duplicate image {old_relative!r} appears in both '
                    f'{seen[old_relative]} and {filename}.'
                )
            seen[old_relative] = filename
            entry = path_map.get(old_relative)
            if entry is None:
                missing.append(old_relative)
                continue
            record = copy.deepcopy(source_record)
            record['image'] = entry['new_relative']
            output_records.append(record)
            annotated_by_source[entry['source_name']] += 1

        rewritten[filename] = output_records
        file_info[filename] = {
            'file': f'annot/{filename}',
            'records': len(records),
            'sha256': file_sha256(path),
        }

    if missing:
        preview = '\n'.join(f'  - {value}' for value in missing[:20])
        remainder = len(missing) - min(20, len(missing))
        suffix = f'\n  ... and {remainder} more' if remainder else ''
        raise DatasetFormatError(
            f'{len(missing)} annotated image(s) were not found among recognized '
            f'image files:\n{preview}{suffix}'
        )
    if not seen:
        raise DatasetFormatError('The three annotation files contain no records.')

    summary = {
        'files': file_info,
        'records': sum(item['records'] for item in file_info.values()),
        'unique_images': len(seen),
        'by_source': dict(annotated_by_source),
        'missing_images': 0,
        'duplicate_references': 0,
    }
    return rewritten, summary


def build_plan(dataset_root: Path) -> Tuple[List[dict], Dict[str, List[dict]], dict]:
    entries, inventory = inventory_images(dataset_root)
    rewritten, annotations = load_and_rewrite_annotations(dataset_root, entries)
    total_images = len(entries)
    total_bytes = sum(entry['size'] for entry in entries)
    manifest = {
        'schema_version': 1,
        'protocol': 'bird_hke_source_organization_v1',
        'source_dataset': str(dataset_root),
        'source_classification': {
            'precedence': ['eBird', 'Animal_Kingdom', 'NABirds', 'Birdsnap'],
            'eBird': 'top-level folder is exactly FINETUNE',
            'Animal_Kingdom': 'top-level folder matches ^[A-Z]+$',
            'NABirds': 'top-level folder matches ^[0-9]+$',
            'Birdsnap': 'all remaining top-level folders',
        },
        'counts': {
            'original_images': total_images,
            'original_bytes': total_bytes,
            'annotated_unique_images': annotations['unique_images'],
            'unannotated_images': total_images - annotations['unique_images'],
            'annotation_records': annotations['records'],
            'images_by_source': inventory['images'],
            'folders_by_source': inventory['top_level_folders'],
            'bytes_by_source': inventory['bytes'],
            'annotated_images_by_source': annotations['by_source'],
            'ignored_non_image_files': inventory['ignored_non_image_files'],
            'missing_annotated_images': annotations['missing_images'],
            'duplicate_annotation_references': annotations['duplicate_references'],
        },
        'source_annotations': annotations['files'],
    }
    return entries, rewritten, manifest


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write('\n')
    temporary.replace(path)


def create_dataset(
    dataset_root: Path,
    output_root: Path,
    entries: Sequence[dict],
    rewritten: Dict[str, List[dict]],
    manifest: dict,
    resume: bool = False,
    show_progress: bool = True,
) -> dict:
    if output_root.exists():
        raise DatasetFormatError(f'Refusing to overwrite existing output: {output_root}')
    staging_root = output_root.with_name(output_root.name + '.partial')
    if staging_root.exists() and not resume:
        raise DatasetFormatError(
            f'An incomplete output already exists: {staging_root}\n'
            'Use --resume to continue it, or move/delete it after inspection.'
        )
    staging_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    resumed = 0
    for entry in _progress(entries, len(entries), 'Copying images', show_progress):
        destination = staging_root / 'images' / Path(entry['new_relative'])
        if destination.exists() and resume:
            if destination.is_file() and destination.stat().st_size == entry['size']:
                resumed += 1
                continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry['source_path'], destination)
        copied += 1

    for filename, records in rewritten.items():
        _write_json(staging_root / 'annot' / filename, records)

    expected_outputs = {entry['new_relative'] for entry in entries}
    actual_outputs = {
        path.relative_to(staging_root / 'images').as_posix()
        for path in (staging_root / 'images').rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    missing_outputs = sorted(expected_outputs - actual_outputs)
    unexpected_outputs = sorted(actual_outputs - expected_outputs)
    if missing_outputs:
        preview = '\n'.join(f'  - {value}' for value in missing_outputs[:20])
        raise DatasetFormatError(
            f'Output verification failed; {len(missing_outputs)} image(s) are missing:\n'
            f'{preview}'
        )
    if unexpected_outputs:
        preview = '\n'.join(f'  - {value}' for value in unexpected_outputs[:20])
        raise DatasetFormatError(
            f'Output verification failed; the partial output contains '
            f'{len(unexpected_outputs)} unexpected image(s):\n{preview}'
        )

    completed = copy.deepcopy(manifest)
    completed['created_utc'] = datetime.now(timezone.utc).isoformat()
    completed['output_dataset'] = str(output_root)
    completed['counts']['copied_this_run'] = copied
    completed['counts']['resumed_existing_images'] = resumed
    completed['counts']['output_images'] = len(entries)
    completed['counts']['output_bytes'] = sum(entry['size'] for entry in entries)
    completed['output_annotations'] = {
        filename: {
            'file': f'annot/{filename}',
            'records': len(records),
            'sha256': file_sha256(staging_root / 'annot' / filename),
        }
        for filename, records in rewritten.items()
    }
    _write_json(staging_root / 'source_format_manifest.json', completed)

    staging_root.rename(output_root)
    return completed


def print_summary(manifest: dict, output_root: Path, wrote: bool) -> None:
    counts = manifest['counts']
    print('\nDataset source organization summary')
    print(f'  Source dataset:           {manifest["source_dataset"]}')
    print(f'  Destination dataset:      {output_root}')
    print(f'  Original images:          {counts["original_images"]}')
    print(f'  Annotation records:       {counts["annotation_records"]}')
    print(f'  Unique annotated images:  {counts["annotated_unique_images"]}')
    print(f'  Unannotated images:        {counts["unannotated_images"]}')
    print(f'  Ignored non-image files:  {counts["ignored_non_image_files"]}')
    print('  Classification:')
    for source_name in SOURCE_NAMES:
        folders = counts['folders_by_source'].get(source_name, 0)
        images = counts['images_by_source'].get(source_name, 0)
        annotated = counts['annotated_images_by_source'].get(source_name, 0)
        print(
            f'    {source_name:15s} {folders:5d} folders, '
            f'{images:7d} images, {annotated:7d} annotated'
        )
    if wrote:
        print(f'  Images in output:         {counts["output_images"]}')
        print(f'  Copied this run:          {counts["copied_this_run"]}')
        print(f'  Resumed existing images:  {counts["resumed_existing_images"]}')
        print('  Integrity checks:         PASS')
        print(f'Created dataset: {output_root}')
    else:
        print('  Integrity checks:         PASS (source and annotations)')
        print(f'Dry run only. Add --write to create: {output_root}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument(
        '--output-root',
        type=Path,
        help='Output dataset root (default: sibling <dataset-name>_by_source).',
    )
    parser.add_argument('--write', action='store_true', help='Create the new dataset.')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume a previously interrupted <output-root>.partial copy.',
    )
    parser.add_argument(
        '--no-progress', action='store_true', help='Disable the copy progress bar.'
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise DatasetFormatError(f'Dataset root does not exist: {dataset_root}')
    if args.output_root:
        output_root = args.output_root.expanduser().resolve()
    else:
        output_root = dataset_root.with_name(dataset_root.name + '_by_source')
    if output_root == dataset_root or _is_inside(output_root, dataset_root):
        raise DatasetFormatError(
            'Output must be a separate dataset outside the source dataset. '
            f'Received: {output_root}'
        )
    if args.resume and not args.write:
        raise DatasetFormatError('--resume requires --write.')

    entries, rewritten, manifest = build_plan(dataset_root)
    if args.write:
        manifest = create_dataset(
            dataset_root,
            output_root,
            entries,
            rewritten,
            manifest,
            resume=args.resume,
            show_progress=not args.no_progress,
        )
    print_summary(manifest, output_root, args.write)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (DatasetFormatError, json.JSONDecodeError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)
