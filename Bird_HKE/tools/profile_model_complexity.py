#!/usr/bin/env python3
"""Profile parameters, GMACs and GFLOPs for Bird_HKE experiment configs.

With no ``--cfg`` arguments, the command profiles all six Full Dataset (FD)
models.  Repeat ``--cfg`` to profile a custom subset.

Examples:
    python Bird_HKE/tools/profile_model_complexity.py
    python Bird_HKE/tools/profile_model_complexity.py --output complexity_fd.json
    python Bird_HKE/tools/profile_model_complexity.py \
        --cfg Bird_HKE/experiments/HRNet/hrnet_w32_birdgaze_FD.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message='Importing from timm.models.registry is deprecated',
)
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message='Importing from timm.models.layers is deprecated',
)

import torch  # noqa: E402

from lib.config.default import _C  # noqa: E402
from lib.config.default import update_config  # noqa: E402
from lib.utilities.model_complexity import COUNTING_CONVENTION  # noqa: E402
from lib.utilities.model_complexity import profile_model_complexity  # noqa: E402
from models import get_pose_net  # noqa: E402


DEFAULT_FD_CONFIGS = (
    PROJECT_ROOT / 'experiments/HRNet/hrnet_w32_birdgaze_FD.yaml',
    PROJECT_ROOT / 'experiments/VHR_BirdPose/vhr_b_FD.yaml',
    PROJECT_ROOT / 'experiments/HR_Mamba/hr_mamba_FD_sum.yaml',
    PROJECT_ROOT / 'experiments/HR_Mamba/hr_mamba_FD_concat_gate.yaml',
    PROJECT_ROOT / 'experiments/HR_MambaVision/hr_mamba_vision_FD.yaml',
    PROJECT_ROOT / 'experiments/HR_MambaViT/hr_mamba_vit_FD.yaml',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile Bird_HKE model parameters and inference compute.'
    )
    parser.add_argument(
        '--cfg',
        dest='configs',
        action='append',
        type=Path,
        help='Experiment YAML; repeat for multiple models. Defaults to all FD configs.',
    )
    parser.add_argument(
        '--device',
        choices=('auto', 'cpu', 'cuda'),
        default='auto',
        help='Profiling device. Mamba CUDA extensions may require cuda.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Optional aggregate JSON output path.',
    )
    parser.add_argument(
        '--opts',
        nargs=argparse.REMAINDER,
        default=[],
        help='YACS KEY VALUE overrides applied to every selected config.',
    )
    return parser.parse_args()


def _resolve_config_path(path: Path) -> Path:
    if path.is_file():
        return path.resolve()
    candidate = REPOSITORY_ROOT / path
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f'configuration does not exist: {path}')


def _load_config(path: Path, opts):
    config = _C.clone()
    update_config(
        config,
        SimpleNamespace(
            cfg=str(path),
            opts=list(opts),
            modelDir='',
            logDir='',
        ),
    )
    return config


def _select_device(name: str) -> torch.device:
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda requested but CUDA is unavailable')
    return torch.device(name)


def _atomic_write_json(path: Path, payload) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _git_revision():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=REPOSITORY_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    device = _select_device(args.device)
    paths = args.configs or list(DEFAULT_FD_CONFIGS)
    paths = [_resolve_config_path(path) for path in paths]

    records = []
    print(f'Profiling {len(paths)} model(s) on {device} with batch size 1')
    print(COUNTING_CONVENTION)
    print()
    header = (
        f'{"Config":38} {"Trainable M":>12} {"Total M":>10} '
        f'{"GMACs":>12} {"GFLOPs":>12}'
    )
    print(header)
    print('-' * len(header))

    for path in paths:
        config = _load_config(path, args.opts)
        model = get_pose_net(config, is_train=False).to(device)
        image_width, image_height = (int(value) for value in config.MODEL.IMAGE_SIZE)
        dummy = torch.zeros(
            1, 3, image_height, image_width, device=device
        )
        complexity = profile_model_complexity(model, dummy)
        record = complexity.to_dict()
        try:
            display_path = path.relative_to(REPOSITORY_ROOT)
        except ValueError:
            display_path = path
        record['metadata'] = {
            'config_file': str(display_path),
            'model_name': str(config.MODEL.NAME),
            'uncertainty_enabled': bool(config.UNCERTAINTY.ENABLED),
        }
        records.append(record)
        print(
            f'{path.name:38} '
            f'{complexity.trainable_parameters / 1e6:12.6f} '
            f'{complexity.total_parameters / 1e6:10.6f} '
            f'{complexity.gmacs:12.6f} '
            f'{complexity.gflops:12.6f}'
        )

        del dummy, model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if args.output:
        payload = {
            'schema_version': 1,
            'device_type': device.type,
            'device_name': (
                torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'
            ),
            'git_revision': _git_revision(),
            'pytorch_version': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'batch_size': 1,
            'counting_convention': COUNTING_CONVENTION,
            'models': records,
        }
        _atomic_write_json(args.output, payload)
        print(f'\nSaved {args.output.expanduser().resolve()}')


if __name__ == '__main__':
    main()
