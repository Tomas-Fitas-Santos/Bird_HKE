"""Audit every experiment YAML against the controlled training protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.utilities.batching import resolve_batch_plan  # noqa: E402


EXPECTED = {
    'WORKERS': 3,
    'CUDNN.BENCHMARK': False,
    'CUDNN.DETERMINISTIC': True,
    'CUDNN.ENABLED': True,
    'REPRODUCIBILITY.PROTOCOL': 'bird_hke_repro_v1',
    'REPRODUCIBILITY.SEED': 2026,
    'REPRODUCIBILITY.STRICT': True,
    'REPRODUCIBILITY.USE_DETERMINISTIC_ALGORITHMS': True,
    'REPRODUCIBILITY.WARN_ONLY': False,
    'REPRODUCIBILITY.ALLOW_TF32': False,
    'REPRODUCIBILITY.MATMUL_PRECISION': 'highest',
    'DATASET.COLOR_RGB': True,
    'DATASET.ANNOT_DIR': 'annot_repro_v1',
    'DATASET.FLIP': True,
    'DATASET.SCALE_FACTOR': 0.25,
    'DATASET.ROT_FACTOR': 30,
    'DATASET.TRAIN_SET': 'train',
    'DATASET.TEST_SET': 'val',
    'MODEL.NUM_JOINTS': 4,
    'MODEL.INIT_WEIGHTS': True,
    'MODEL.PRETRAINED': "r''",
    'MODEL.IMAGE_SIZE': [256, 256],
    'MODEL.HEATMAP_SIZE': [64, 64],
    'MODEL.SIGMA': 2,
    'MODEL.TARGET_TYPE': 'gaussian',
    'LOSS.USE_TARGET_WEIGHT': True,
    'TRAIN.BATCH_SIZE_PER_GPU': 8,
    'TRAIN.EFFECTIVE_BATCH_SIZE': 64,
    'TRAIN.GRAD_ACCUM_STEPS': 0,
    'TRAIN.SHUFFLE': True,
    'TRAIN.DROP_LAST': False,
    'TRAIN.BEGIN_EPOCH': 0,
    'TRAIN.END_EPOCH': 100,
    'TRAIN.OPTIMIZER': 'adamw',
    'TRAIN.LR': 0.0005,
    'TRAIN.WD': 0.01,
    'TRAIN.LR_SCHEDULE': 'cosine',
    'TRAIN.WARMUP_EPOCHS': 5,
    'TRAIN.MIN_LR': 0.00001,
    'TRAIN.CLIP_GRAD_NORM': 1.0,
    'TEST.FLIP_TEST': True,
    'TEST.POST_PROCESS': True,
    'TEST.SHIFT_HEATMAP': True,
}


def _get(config: Any, dotted_key: str) -> Any:
    value = config
    for key in dotted_key.split('.'):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(dotted_key)
        value = value[key]
    return list(value) if isinstance(value, tuple) else value


def experiment_files(experiments_dir: Path) -> Iterable[Path]:
    return sorted(experiments_dir.glob('*/*.yaml'))


def _load(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError('configuration root must be a mapping')
    return config


def _resolve_batch_plan(train_cfg: dict, devices: int) -> Tuple[int, int, int]:
    plan = resolve_batch_plan(train_cfg, devices)
    return (
        plan.batch_size_per_device,
        plan.accumulation_steps,
        plan.effective_batch_size,
    )


def audit_file(path: Path) -> Tuple[dict, Any, List[str]]:
    config = _load(path)
    problems = []

    for key, expected in EXPECTED.items():
        try:
            actual = _get(config, key)
        except KeyError:
            problems.append(f'{key}: missing (expected {expected!r})')
            continue
        if actual != expected:
            problems.append(f'{key}: expected {expected!r}, found {actual!r}')

    try:
        plan = _resolve_batch_plan(config['TRAIN'], max(1, len(config['GPUS'])))
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f'batch plan: {exc}')
        plan = None

    root_value = config.get('DATASET', {}).get('ROOT', '')
    root = Path(root_value)
    if not root.parts or root.parts[0] != 'BirdGaze_v2':
        problems.append(
            'DATASET.ROOT must be relative to BirdGaze_v2 for a portable config; '
            f'found {root_value!r}'
        )
    return config, plan, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--experiments-dir',
        type=Path,
        default=PROJECT_ROOT / 'experiments',
    )
    args = parser.parse_args()

    paths = list(experiment_files(args.experiments_dir))
    if not paths:
        print(f'No experiment YAML files found under {args.experiments_dir}', file=sys.stderr)
        return 2

    failures = 0
    print('configuration\tmodel\tdataset\tmicro/GPU\taccum\teffective\tstatus')
    for path in paths:
        config, plan, problems = audit_file(path)
        status = 'PASS' if not problems else 'FAIL'
        if problems:
            failures += 1
        relative = path.relative_to(REPOSITORY_ROOT)
        print(
            f'{relative}\t{config.get("MODEL", {}).get("NAME", "-")}\t'
            f'{config.get("DATASET", {}).get("TRAIN_SET", "-")}\t'
            f'{plan[0] if plan else "-"}\t'
            f'{plan[1] if plan else "-"}\t'
            f'{plan[2] if plan else "-"}\t{status}'
        )
        for problem in problems:
            print(f'  - {problem}')

    if failures:
        print(f'\nProtocol audit failed for {failures}/{len(paths)} configuration(s).')
        return 1
    print(f'\nProtocol audit passed for all {len(paths)} configurations.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
