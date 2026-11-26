"""Audit every experiment YAML against the controlled training protocol."""

from __future__ import annotations

import argparse
import re
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
    'REPRODUCIBILITY.SEED': 2026,
    'REPRODUCIBILITY.STRICT': True,
    'REPRODUCIBILITY.USE_DETERMINISTIC_ALGORITHMS': True,
    'REPRODUCIBILITY.WARN_ONLY': False,
    'REPRODUCIBILITY.ALLOW_TF32': False,
    'REPRODUCIBILITY.MATMUL_PRECISION': 'highest',
    'DATASET.COLOR_RGB': True,
    'DATASET.ANNOT_DIR': 'annot',
    'DATASET.FLIP': True,
    'DATASET.SCALE_FACTOR': 0.25,
    'DATASET.ROT_FACTOR': 30,
    'DATASET.TRAIN_SET': 'train',
    'DATASET.TEST_SET': 'val',
    'MODEL.NUM_JOINTS': 4,
    'MODEL.INIT_WEIGHTS': True,
    'MODEL.PRETRAINED': '',
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
    'TRAIN.RESUME_FROM_CKPT': True,
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


SCENARIO_UNCERTAINTY = {
    'FD': True,
    'CS': False,
    'OS': False,
}

SCENARIO_PROTOCOL = {
    'FD': 'bird_hke_repro_v3',
    'CS': 'bird_hke_repro_v2',
    'OS': 'bird_hke_repro_v2',
}

PASSIVE_UNCERTAINTY_EXPECTED = {
    'UNCERTAINTY.RELIABILITY_GRADIENT_TO_BACKBONE': False,
    'UNCERTAINTY.HEAD_DROPOUT': 0.0,
    'UNCERTAINTY.QUALITY_PCK_THRESHOLD': 0.5,
}

RUN_DIRECTORY_KEYS = (
    'TRAIN.CKPT_DIR',
    'TRAIN.LOG_DIR',
    'TEST.POSE_MODEL_FILE',
    'TEST.OUTPUT_DIR',
)


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


def scenario_from_path(path: Path) -> str:
    """Return the FD, CS, or OS scenario encoded in an experiment filename."""
    match = re.search(r'(?:^|_)(FD|CS|OS)(?:_|$)', path.stem)
    if match is None:
        raise ValueError(
            f'{path.name}: filename must contain an FD, CS, or OS scenario token'
        )
    return match.group(1)


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
        scenario = scenario_from_path(path)
        uncertainty_expected = SCENARIO_UNCERTAINTY[scenario]
        protocol_expected = SCENARIO_PROTOCOL[scenario]
        try:
            protocol_actual = _get(config, 'REPRODUCIBILITY.PROTOCOL')
        except KeyError:
            protocol_actual = None
            problems.append('REPRODUCIBILITY.PROTOCOL: missing')
        if protocol_actual != protocol_expected:
            problems.append(
                'REPRODUCIBILITY.PROTOCOL: '
                f'{scenario} requires {protocol_expected!r}, '
                f'found {protocol_actual!r}'
            )
        try:
            uncertainty_actual = _get(config, 'UNCERTAINTY.ENABLED')
        except KeyError:
            uncertainty_actual = None
            problems.append('UNCERTAINTY.ENABLED: missing')
        if uncertainty_actual is not uncertainty_expected:
            problems.append(
                'UNCERTAINTY.ENABLED: '
                f'{scenario} requires {uncertainty_expected!r}, '
                f'found {uncertainty_actual!r}'
            )

        if uncertainty_expected:
            for key, expected in PASSIVE_UNCERTAINTY_EXPECTED.items():
                try:
                    actual = _get(config, key)
                except KeyError:
                    problems.append(f'{key}: missing (expected {expected!r})')
                    continue
                if actual != expected:
                    problems.append(
                        f'{key}: expected {expected!r}, found {actual!r}'
                    )

        namespace = 'uncertainty' if uncertainty_expected else 'baseline'
        version = 'repro_v3' if scenario == 'FD' else 'repro_v2'
        required_fragment = f'/{version}/{namespace}/'
        for key in RUN_DIRECTORY_KEYS:
            try:
                value = str(_get(config, key)).replace('\\', '/')
            except KeyError:
                problems.append(f'{key}: missing')
                continue
            if required_fragment not in value:
                problems.append(
                    f'{key}: {scenario} paths must contain '
                    f'{required_fragment!r}; found {value!r}'
                )
        try:
            checkpoint_dir = str(_get(config, 'TRAIN.CKPT_DIR')).rstrip('/\\')
            log_dir = str(_get(config, 'TRAIN.LOG_DIR')).rstrip('/\\')
            if checkpoint_dir != log_dir:
                problems.append(
                    'TRAIN.CKPT_DIR and TRAIN.LOG_DIR must identify the same '
                    'run directory so checkpoints, stop requests, and reports '
                    'cannot be mixed across runs'
                )
        except KeyError:
            pass
    except ValueError as exc:
        problems.append(str(exc))

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
    print('configuration\tmodel\tscenario\tUQ\tmicro/GPU\taccum\teffective\tstatus')
    for path in paths:
        config, plan, problems = audit_file(path)
        status = 'PASS' if not problems else 'FAIL'
        if problems:
            failures += 1
        relative = path.relative_to(REPOSITORY_ROOT)
        try:
            scenario = scenario_from_path(path)
        except ValueError:
            scenario = '-'
        uncertainty = config.get('UNCERTAINTY', {}).get('ENABLED', '-')
        print(
            f'{relative}\t{config.get("MODEL", {}).get("NAME", "-")}\t'
            f'{scenario}\t{uncertainty}\t'
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
