"""Utilities for controlled and resumable Bird_HKE training runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .batching import BatchPlan
from .batching import accumulation_group_size
from .batching import resolve_batch_plan


def seed_everything(
    seed: int,
    deterministic_algorithms: bool = True,
    warn_only: bool = True,
) -> None:
    """Seed all randomness used by the training and augmentation pipeline."""
    seed = int(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(
        bool(deterministic_algorithms), warn_only=bool(warn_only)
    )


def seed_worker(_worker_id: int) -> None:
    """Seed Python and NumPy in each DataLoader worker from PyTorch's seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state(generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
    """Capture RNG state needed to resume at the next epoch boundary."""
    state: Dict[str, Any] = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    if generator is not None:
        state['train_loader_generator'] = generator.get_state()
    return state


def restore_rng_state(
    state: Optional[Dict[str, Any]],
    generator: Optional[torch.Generator] = None,
) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])
    if generator is not None and 'train_loader_generator' in state:
        generator.set_state(state['train_loader_generator'])


def _plain(value: Any) -> Any:
    if hasattr(value, 'items'):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _file_sha256(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_annotation_hashes(cfg: Any) -> Dict[str, Optional[str]]:
    """Fingerprint the train and validation annotations without using paths."""
    annotation_dir = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.ANNOT_DIR)
    return {
        split: _file_sha256(os.path.join(annotation_dir, f'{name}.json'))
        for split, name in (
            ('train', cfg.DATASET.TRAIN_SET),
            ('validation', cfg.DATASET.TEST_SET),
        )
    }


def training_protocol(cfg: Any, plan: BatchPlan) -> Dict[str, Any]:
    """Return the training-relevant contract stored in every checkpoint."""
    return {
        'protocol': cfg.REPRODUCIBILITY.PROTOCOL,
        'seed': int(cfg.REPRODUCIBILITY.SEED),
        'workers': int(cfg.WORKERS),
        'cudnn': _plain(cfg.CUDNN),
        'model': {
            key: _plain(getattr(cfg.MODEL, key))
            for key in (
                'NAME', 'INIT_WEIGHTS', 'NUM_JOINTS', 'TAG_PER_JOINT', 'TARGET_TYPE',
                'IMAGE_SIZE', 'HEATMAP_SIZE', 'SIGMA', 'EXTRA',
            )
        },
        'pretrained_checkpoint_sha256': _file_sha256(
            os.path.expanduser(cfg.MODEL.PRETRAINED)
        ),
        'loss': _plain(cfg.LOSS),
        'uncertainty_training': {
            key: _plain(getattr(cfg.UNCERTAINTY, key))
            for key in (
                'ENABLED', 'DISTRIBUTION', 'TEMPERATURE',
                'HEAD_HIDDEN_CHANNELS', 'JOINT_EMBED_DIM', 'HEAD_DROPOUT',
                'RELIABILITY_GRADIENT_TO_BACKBONE',
                'BKS_SIGMA_FRACTION', 'LOCATION_WEIGHT', 'SMOOTHNESS_WEIGHT',
                'QUALITY_WEIGHT', 'VISIBILITY_WEIGHT', 'SYNTHETIC_OCCLUSION',
                'BALANCE_VISIBILITY_CLASSES', 'DECODER',
            )
        },
        # ROOT and model-file paths are deliberately excluded: changing a
        # mount point must not invalidate an otherwise identical resumed run.
        'dataset': {
            key: _plain(getattr(cfg.DATASET, key))
            for key in (
                'DATASET', 'ANNOT_DIR', 'TRAIN_SET', 'TEST_SET', 'DATA_FORMAT', 'FLIP',
                'SCALE_FACTOR', 'ROT_FACTOR', 'PROB_HALF_BODY',
                'NUM_JOINTS_HALF_BODY', 'COLOR_RGB',
            )
        },
        'dataset_annotation_sha256': dataset_annotation_hashes(cfg),
        'training': {
            key: _plain(getattr(cfg.TRAIN, key))
            for key in (
                'LR_FACTOR', 'LR_STEP', 'LR', 'OPTIMIZER', 'MOMENTUM', 'WD',
                'NESTEROV', 'LR_SCHEDULE', 'WARMUP_EPOCHS', 'MIN_LR',
                'CLIP_GRAD_NORM', 'BEGIN_EPOCH', 'END_EPOCH',
                'BATCH_SIZE_PER_GPU', 'EFFECTIVE_BATCH_SIZE', 'SHUFFLE',
                'DROP_LAST',
            )
        },
        # Validation settings affect epoch selection and therefore belong to
        # the reproducible training contract even though they do not affect
        # gradient computation directly.
        'validation': {
            key: _plain(getattr(cfg.TEST, key))
            for key in (
                'BATCH_SIZE_PER_GPU', 'FLIP_TEST', 'POST_PROCESS',
                'SHIFT_HEATMAP',
            )
        },
        # Include the complete plan so strict resume cannot silently switch
        # device count or accumulation boundaries mid-run.
        'resolved_batch': asdict(plan),
        'deterministic_algorithms': bool(
            cfg.REPRODUCIBILITY.USE_DETERMINISTIC_ALGORITHMS
        ),
        'deterministic_warn_only': bool(cfg.REPRODUCIBILITY.WARN_ONLY),
        'strict_resume_validation': bool(cfg.REPRODUCIBILITY.STRICT),
        'allow_tf32': bool(cfg.REPRODUCIBILITY.ALLOW_TF32),
        'matmul_precision': cfg.REPRODUCIBILITY.MATMUL_PRECISION,
    }


def protocol_hash(protocol: Dict[str, Any]) -> str:
    payload = json.dumps(protocol, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp suitable for run metadata."""
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str, payload: Any) -> None:
    """Atomically replace a JSON report, preserving the previous good copy."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{destination.name}.',
        suffix='.tmp',
        dir=str(destination.parent),
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(destination))
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_is_dirty() -> Optional[bool]:
    try:
        status = subprocess.check_output(
            ['git', 'status', '--porcelain', '--untracked-files=no'],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def resume_environment(report: Dict[str, Any]) -> Dict[str, Any]:
    """Return environment fields that must stay fixed across strict resume."""
    return {
        key: report.get(key)
        for key in (
            'git_revision', 'python', 'pytorch', 'cuda_runtime', 'cudnn',
            'configured_gpu_ids', 'gpu_names', 'package_versions',
        )
    }


def environment_report(cfg: Any, plan: BatchPlan) -> Dict[str, Any]:
    """Describe the software, hardware, and resolved protocol for a run."""
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    protocol = training_protocol(cfg, plan)
    calibration_path = os.path.join(
        cfg.DATASET.ROOT, cfg.DATASET.ANNOT_DIR, 'calibration.json'
    )
    packages = {}
    for package in (
        'torch', 'torchvision', 'numpy', 'opencv-python', 'scipy', 'timm',
        'mamba-ssm', 'yacs',
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        'recorded_at_utc': utc_timestamp(),
        'protocol_hash': protocol_hash(protocol),
        'protocol': protocol,
        'calibration_annotation_sha256': _file_sha256(calibration_path),
        'resolved_batch': asdict(plan),
        'git_revision': _git_revision(),
        'git_dirty': _git_is_dirty(),
        'platform': platform.platform(),
        'python': sys.version,
        'pytorch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'cuda_available': torch.cuda.is_available(),
        'configured_gpu_ids': list(cfg.GPUS),
        'gpu_names': gpu_names,
        'package_versions': packages,
    }
