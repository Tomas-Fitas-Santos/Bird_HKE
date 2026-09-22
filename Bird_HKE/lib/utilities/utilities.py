# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
import os
import logging
import time
import tempfile
from pathlib import Path

import torch
import torch.optim as optim

from .model_complexity import profile_model_complexity


_AUXILIARY_PARAMETER_MARKER = 'probabilistic_output.'


def split_pose_and_auxiliary_parameters(model):
    """Return stable, disjoint trainable parameter lists.

    Keeping the pose parameters in their own optimizer/clipping group ensures
    auxiliary uncertainty gradients cannot change pose gradient clipping or
    optimizer behavior.
    """
    pose_parameters = []
    auxiliary_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = (
            auxiliary_parameters
            if _AUXILIARY_PARAMETER_MARKER in name
            else pose_parameters
        )
        target.append(parameter)
    return pose_parameters, auxiliary_parameters


def clip_pose_and_auxiliary_gradients(model, max_norm):
    """Clip pose and passive uncertainty gradients independently."""
    pose_parameters, auxiliary_parameters = split_pose_and_auxiliary_parameters(
        model
    )
    norms = {}
    if pose_parameters:
        norms['pose'] = torch.nn.utils.clip_grad_norm_(
            pose_parameters, max_norm=max_norm
        )
    if auxiliary_parameters:
        norms['uncertainty'] = torch.nn.utils.clip_grad_norm_(
            auxiliary_parameters, max_norm=max_norm
        )
    return norms


def create_logger(cfg, cfg_name, phase='train', root_choice='log'):
    """Create logger and tensorboard dirs.
    root_choice: 'log' -> use LOG_DIR (never falls back to OUTPUT_DIR)
                 'output' -> use OUTPUT_DIR (never falls back to LOG_DIR)
                 'auto' -> legacy behavior (LOG_DIR > OUTPUT_DIR > ./output)
    """
    cwd = os.getcwd()
    if root_choice == 'log':
        if cfg.LOG_DIR:
            root_output_dir = Path(cfg.LOG_DIR)
        else:
            root_output_dir = Path(cwd) / 'log'
    elif root_choice == 'output':
        if cfg.OUTPUT_DIR:
            root_output_dir = Path(cfg.OUTPUT_DIR)
        else:
            root_output_dir = Path(cwd) / 'output'
    else:
        # legacy: prefer LOG_DIR, then OUTPUT_DIR
        root_output_dir = Path(cfg.LOG_DIR if cfg.LOG_DIR else (cfg.OUTPUT_DIR if cfg.OUTPUT_DIR else (Path(cwd) / 'output')))
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir(parents=True, exist_ok=True)

    # Use the root output directory directly (no per-dataset/model subfolders)
    final_output_dir = root_output_dir
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to console only; do not create a dated log file.
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # No tensorboard logs: return None for tb_log_dir to disable TB output
    return logger, str(final_output_dir), None


def get_optimizer(cfg, model):
    pose_parameters, auxiliary_parameters = split_pose_and_auxiliary_parameters(
        model
    )
    parameter_groups = [{'params': pose_parameters}]
    if auxiliary_parameters:
        parameter_groups.append({'params': auxiliary_parameters})

    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            parameter_groups,
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            parameter_groups,
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            parameter_groups,
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WD
        )

    return optimizer


def atomic_torch_save(value, destination):
    """Commit a PyTorch artifact without risking the previous good file.

    The temporary file is written and flushed in the destination directory,
    then atomically replaces the target.  If the process is interrupted while
    serializing, the last completed checkpoint remains available for resume.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{destination.name}.',
        suffix='.tmp',
        dir=str(destination.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(destination))
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth', best_filename='model_best.pth'):
    atomic_torch_save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        atomic_torch_save(
            states['best_state_dict'], os.path.join(output_dir, best_filename)
        )


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """Backward-compatible text wrapper around the shared profiler.

    ``item_length`` is retained for call-site compatibility.  The previous
    implementation silently undercounted grouped convolutions, token-wise
    linear layers, attention, Mamba scans, and uncertainty pooling.
    """
    del item_length
    return profile_model_complexity(model, *input_tensors).format(verbose)
