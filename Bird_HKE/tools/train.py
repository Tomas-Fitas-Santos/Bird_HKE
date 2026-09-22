# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

# This file is adapted from the original codebase of Simple Baselines for Human Pose Estimation and Tracking, which is licensed under the MIT License. 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import time
import os
import shlex
import tempfile

# Deterministic cuBLAS requires this before any imported CUDA extension can
# initialize a context.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
from lib.config.default import _C as cfg
# suppress TIMM deprecation noise when importing vendor code
warnings.filterwarnings('ignore', category=FutureWarning,
                        message='Importing from timm.models.registry is deprecated')
warnings.filterwarnings('ignore', category=FutureWarning,
                        message='Importing from timm.models.layers is deprecated')
from lib.config.default import update_config
from lib.core.loss import build_pose_criterion
from lib.core.function import train
from lib.core.function import validate
from lib.utilities.utilities import get_optimizer
from lib.utilities.utilities import save_checkpoint
from lib.utilities.utilities import atomic_torch_save
from lib.utilities.utilities import create_logger
from lib.utilities.model_complexity import profile_model_complexity
from lib.utilities.model_complexity import write_model_complexity_json
from lib.utilities.reproducibility import capture_rng_state
from lib.utilities.reproducibility import atomic_write_json
from lib.utilities.reproducibility import environment_report
from lib.utilities.reproducibility import protocol_hash
from lib.utilities.reproducibility import resolve_batch_plan
from lib.utilities.reproducibility import resume_environment
from lib.utilities.reproducibility import restore_rng_state
from lib.utilities.reproducibility import seed_everything
from lib.utilities.reproducibility import seed_worker
from lib.utilities.reproducibility import training_protocol
from lib.utilities.reproducibility import utc_timestamp


from models import get_pose_net

from dataset import birdgaze     # Bird dataset   (change also in default.py)

# `current_directory` is available via `os.getcwd()`; we'll select the base
# output directory from the config (cfg.OUTPUT_DIR) at runtime so paths are
# entirely driven by configuration.

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    args = parser.parse_args()

    return args


def _adapt_state_dict_for_model(model, state_dict):
    if not state_dict:
        return state_dict

    model_keys = list(model.state_dict().keys())
    state_keys = list(state_dict.keys())
    if not model_keys or not state_keys:
        return state_dict

    model_has_module = model_keys[0].startswith('module.')
    state_has_module = state_keys[0].startswith('module.')
    if model_has_module == state_has_module:
        return state_dict

    adapted = {}
    if model_has_module:
        for key, value in state_dict.items():
            adapted[f'module.{key}'] = value
    else:
        prefix = 'module.'
        for key, value in state_dict.items():
            adapted[key[len(prefix):] if key.startswith(prefix) else key] = value
    return adapted


def _load_state_dict_with_report(
    model, state_dict, logger, context='checkpoint', strict=False
):
    state_dict = _adapt_state_dict_for_model(model, state_dict)
    incompatible = model.load_state_dict(state_dict, strict=strict)

    missing = list(getattr(incompatible, 'missing_keys', []))
    unexpected = list(getattr(incompatible, 'unexpected_keys', []))
    total_keys = len(model.state_dict())
    loaded_keys = max(0, total_keys - len(missing))
    loaded_ratio = (100.0 * loaded_keys / total_keys) if total_keys else 0.0

    logger.info(
        "=> loaded %.1f%% of model keys from %s (%d/%d); missing=%d, unexpected=%d",
        loaded_ratio, context, loaded_keys, total_keys, len(missing), len(unexpected)
    )


def _atomic_write_text(path, text):
    destination = os.path.abspath(path)
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f'.{os.path.basename(destination)}.',
        suffix='.tmp',
        dir=directory,
        text=True,
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _epoch_number(line):
    try:
        return int(line.split('\t', 1)[0])
    except (TypeError, ValueError):
        return None


def _metric_map(values):
    if values is None:
        return {}
    if hasattr(values, 'get'):
        return values
    if isinstance(values, list):
        for item in values:
            if hasattr(item, 'get') and 'Mean' in item:
                return item
        for item in values:
            if hasattr(item, 'get'):
                return item
        try:
            return dict(values)
        except Exception:
            return {}
    return {}


def _rewrite_train_logs(
    logs_file, best_epoch_val, best_perf_val, best_values, epoch_line=None
):
    """Upsert an epoch row and atomically regenerate the run summary."""
    try:
        lines = []
        if os.path.exists(logs_file):
            with open(logs_file, 'r', encoding='utf-8') as handle:
                lines = [line.rstrip('\n') for line in handle]

        for index, line in enumerate(lines):
            if line.strip() == '# Summary':
                lines = lines[:index]
                break

        header = 'epoch\thead\teyes\tmouth\tmean\ttrain_time_s\tvalidation_time_s'
        rows = [line for line in lines if _epoch_number(line) is not None]
        if epoch_line:
            new_epoch = _epoch_number(epoch_line)
            rows = [line for line in rows if _epoch_number(line) != new_epoch]
            rows.append(epoch_line.rstrip('\n'))
        rows.sort(key=_epoch_number)

        train_times = []
        validation_times = []
        for row in rows:
            columns = row.split('\t')
            if len(columns) >= 7:
                try:
                    train_times.append(float(columns[5]))
                    validation_times.append(float(columns[6]))
                except ValueError:
                    pass

        avg_train_time = (
            sum(train_times) / len(train_times) if train_times else 0.0
        )
        avg_validation_time = (
            sum(validation_times) / len(validation_times)
            if validation_times else 0.0
        )
        output = [header, *rows, '', '# Summary']
        output.append(f'Avg epoch train time (s): {avg_train_time:.4f}')
        output.append(
            f'Avg epoch test time (s): {avg_validation_time:.4f}'
        )
        if best_epoch_val is not None:
            output.append(
                f'Best epoch: {best_epoch_val} (perf={float(best_perf_val):.6f})'
            )
            metrics = _metric_map(best_values)
            output.append(
                'Best epoch accuracies - Head: {0:.6f}, Eyes: {1:.6f}, '
                'Mouth: {2:.6f}, Mean: {3:.6f}'.format(
                    float(metrics.get('Head', 0.0)),
                    float(metrics.get('Eyes', 0.0)),
                    float(metrics.get('Mouth', 0.0)),
                    float(metrics.get('Mean', 0.0)),
                )
            )
        else:
            output.append('Best epoch: N/A')
        _atomic_write_text(logs_file, '\n'.join(output).rstrip() + '\n')
    except Exception as exc:
        logger.warning('Failed to update train_logs summary: %s', exc)


def _update_attempt(history_file, attempt_index, **updates):
    try:
        with open(history_file, 'r', encoding='utf-8') as handle:
            history = json.load(handle)
    except FileNotFoundError:
        history = []
    history[attempt_index].update(updates)
    atomic_write_json(history_file, history)

def main():

    args = parse_args()

    update_config(cfg, args)

    use_cuda = torch.cuda.is_available() and len(cfg.GPUS) > 0
    if use_cuda:
        available_devices = torch.cuda.device_count()
        invalid_devices = [gpu for gpu in cfg.GPUS if gpu < 0 or gpu >= available_devices]
        if invalid_devices:
            raise ValueError(
                f'Configured GPU ids {invalid_devices} are unavailable; '
                f'this host exposes {available_devices} CUDA device(s).'
            )
        torch.cuda.set_device(cfg.GPUS[0])
    active_devices = len(cfg.GPUS) if use_cuda else 1
    batch_plan = resolve_batch_plan(cfg.TRAIN, active_devices)
    seed_everything(
        cfg.REPRODUCIBILITY.SEED,
        deterministic_algorithms=cfg.REPRODUCIBILITY.USE_DETERMINISTIC_ALGORITHMS,
        warn_only=cfg.REPRODUCIBILITY.WARN_ONLY,
    )

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.backends.cuda.matmul.allow_tf32 = cfg.REPRODUCIBILITY.ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = cfg.REPRODUCIBILITY.ALLOW_TF32
    torch.set_float32_matmul_precision(cfg.REPRODUCIBILITY.MATMUL_PRECISION)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg = cfg, cfg_name = 'config', root_choice='log')
    # Determine base directories from the config so everything is configurable
    current_directory = os.getcwd()
    base_dir = cfg.LOG_DIR if cfg.LOG_DIR else os.path.join(current_directory, 'log')
    debug_images_directory = os.path.join(base_dir, 'debug_images')
    # Prefer TRAIN.CKPT_DIR, then top-level CKPT_DIR, else fall back to base_dir (log)
    if getattr(cfg, 'TRAIN', None) and getattr(cfg.TRAIN, 'CKPT_DIR', ''):
        ckpt_dir = cfg.TRAIN.CKPT_DIR
    else:
        ckpt_dir = base_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    run_environment = environment_report(cfg, batch_plan)
    current_protocol = training_protocol(cfg, batch_plan)
    current_protocol_hash = protocol_hash(current_protocol)
    current_resume_environment = resume_environment(run_environment)
    if cfg.REPRODUCIBILITY.STRICT and run_environment.get('git_dirty') is True:
        raise RuntimeError(
            'Strict reproducibility requires a clean Git working tree. '
            'Commit or restore tracked changes before starting training.'
        )
    os.makedirs(base_dir, exist_ok=True)
    _atomic_write_text(os.path.join(base_dir, 'resolved_config.yaml'), cfg.dump())
    atomic_write_json(os.path.join(base_dir, 'environment.json'), run_environment)

    logger.info(cfg)
    logger.info(
        'Resolved batch: %d device(s) x %d samples x %d accumulation = %d',
        batch_plan.devices,
        batch_plan.batch_size_per_device,
        batch_plan.accumulation_steps,
        batch_plan.effective_batch_size,
    )
    logger.info('Training protocol hash: %s', current_protocol_hash)

    
    model = get_pose_net(cfg, is_train=True)
    # Log and persist parameter/compute accounting before training.  Strict
    # runs abort if this report cannot be produced so every published run has
    # a matching complexity artifact.
    try:
        device = torch.device('cuda' if use_cuda else 'cpu')
        img_w, img_h = (
            cfg.MODEL.IMAGE_SIZE
            if hasattr(cfg.MODEL, 'IMAGE_SIZE')
            else (256, 256)
        )
        dummy = torch.zeros(1, 3, img_h, img_w, device=device)

        model.to(device)
        complexity = profile_model_complexity(model, dummy)
        summary = complexity.format(verbose=True)
        logger.info('%s', summary)
        try:
            os.makedirs(base_dir, exist_ok=True)
            _atomic_write_text(os.path.join(base_dir, 'model_summary.txt'), summary)
            write_model_complexity_json(
                os.path.join(base_dir, 'model_complexity.json'),
                complexity,
                metadata={
                    'config_file': os.path.normpath(args.cfg),
                    'model_name': str(cfg.MODEL.NAME),
                    'uncertainty_enabled': bool(cfg.UNCERTAINTY.ENABLED),
                    'git_revision': run_environment.get('git_revision'),
                    'training_protocol_hash': current_protocol_hash,
                },
            )
        except Exception:
            logger.exception('Failed to write model complexity artifacts')
            raise
        # Copy the YAML config used for this run into LOG_DIR
        try:
            with open(args.cfg, 'r', encoding='utf-8') as handle:
                source_config = handle.read()
            _atomic_write_text(
                os.path.join(base_dir, 'model_config.txt'), source_config
            )
        except Exception:
            logger.warning('Failed to write model_config.txt')

        logger.info(
            'Model complexity: %.6fM trainable / %.6fM total parameters; '
            '%.6f GMACs; %.6f GFLOPs',
            complexity.trainable_parameters / 1e6,
            complexity.total_parameters / 1e6,
            complexity.gmacs,
            complexity.gflops,
        )

        # cuda memory usage (MB)
        if use_cuda:
            try:
                max_alloc = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                max_reserved = getattr(torch.cuda, 'max_memory_reserved', None)
                if max_reserved is not None:
                    max_res = torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
                else:
                    max_res = torch.cuda.memory_reserved(device) / (1024.0 ** 2)
                logger.info('CUDA memory (MB) - max allocated: %.1f, max reserved: %.1f', max_alloc, max_res)
            except Exception:
                logger.info('CUDA memory query failed')
    except Exception as e:
        logger.exception('Model complexity profiling failed: %s', e)
        if bool(cfg.REPRODUCIBILITY.STRICT):
            raise RuntimeError(
                'Strict reproducibility requires a valid model-complexity '
                'report before training starts.'
            ) from e

    # wrap for multi-gpu and move to cuda if available
    if torch.cuda.is_available() and len(cfg.GPUS) > 0:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    else:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)

    criterion_device = torch.device('cuda' if use_cuda else 'cpu')
    criterion = build_pose_criterion(cfg).to(criterion_device)
    logger.info(
        'Uncertainty modelling: %s; criterion: %s',
        'enabled' if cfg.UNCERTAINTY.ENABLED else 'disabled',
        criterion.__class__.__name__,
    )

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )    

    train_dataset = eval(cfg.DATASET.NAME_+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )


    valid_dataset = eval(cfg.DATASET.NAME_+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )    

    train_generator = torch.Generator()
    train_generator.manual_seed(int(cfg.REPRODUCIBILITY.SEED))
    validation_generator = torch.Generator()
    validation_generator.manual_seed(int(cfg.REPRODUCIBILITY.SEED) + 1)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_plan.global_micro_batch,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=cfg.TRAIN.DROP_LAST,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )  

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*active_devices,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        worker_init_fn=seed_worker,
        generator=validation_generator,
    )      

    processed_samples = len(train_dataset)
    if cfg.TRAIN.DROP_LAST:
        processed_samples = (
            processed_samples // batch_plan.global_micro_batch
        ) * batch_plan.global_micro_batch
    final_optimizer_batch = processed_samples % batch_plan.effective_batch_size
    if final_optimizer_batch == 0 and processed_samples > 0:
        final_optimizer_batch = batch_plan.effective_batch_size
    run_environment['dataset_sizes'] = {
        'train': len(train_dataset),
        'validation': len(valid_dataset),
    }
    run_environment['micro_batches_per_epoch'] = len(train_loader)
    run_environment['training_samples_processed_per_epoch'] = processed_samples
    run_environment['optimizer_steps_per_epoch'] = (
        len(train_loader) + batch_plan.accumulation_steps - 1
    ) // batch_plan.accumulation_steps
    run_environment['final_optimizer_batch_size'] = final_optimizer_batch
    atomic_write_json(os.path.join(base_dir, 'environment.json'), run_environment)



    best_perf = 0.0
    best_model = False
    best_epoch = None
    best_name_values = None
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    # Determine the checkpoint file to use for resume. If `CKPT_FILE` is
    # provided in config we prefer it; if it's relative, resolve it inside
    # the chosen checkpoint directory. Otherwise fall back to the canonical
    # checkpoint name inside the checkpoint directory.
    if getattr(cfg, 'CKPT_FILE', ''):
        if os.path.isabs(cfg.CKPT_FILE):
            checkpoint_file = cfg.CKPT_FILE
        else:
            checkpoint_file = os.path.join(ckpt_dir, cfg.CKPT_FILE)
    else:
        checkpoint_file = os.path.join(ckpt_dir, 'checkpoint.pth')

    resume_from_ckpt = bool(getattr(cfg, 'RESUME_FROM_CKPT', False) and cfg.RESUME_FROM_CKPT and os.path.exists(checkpoint_file))
    if resume_from_ckpt:
        logger.info("=> loading checkpoint '%s'", checkpoint_file)
        try:
            # For PyTorch 2.7, explicitly request full load to maintain
            # compatibility with older checkpoint pickles.
            checkpoint = torch.load(checkpoint_file, weights_only=False)
        except TypeError:
            # Older/newer torch versions may not accept weights_only kwarg.
            checkpoint = torch.load(checkpoint_file)
        except Exception as e:
            logger.error("Failed to load checkpoint '%s': %s", checkpoint_file, e)
            raise
        saved_protocol_hash = checkpoint.get('training_protocol_hash')
        if cfg.REPRODUCIBILITY.STRICT:
            if saved_protocol_hash is None:
                raise RuntimeError(
                    'The checkpoint predates the reproducible-training protocol. '
                    'Start a fresh run directory, or set '
                    'REPRODUCIBILITY.STRICT false only for a legacy continuation.'
                )
            if saved_protocol_hash != current_protocol_hash:
                raise RuntimeError(
                    'Checkpoint training protocol does not match this run. '
                    f'checkpoint={saved_protocol_hash}, current={current_protocol_hash}. '
                    'Use a fresh run directory for a changed experiment.'
                )
            saved_resume_environment = checkpoint.get('resume_environment')
            if saved_resume_environment is None:
                raise RuntimeError(
                    'Checkpoint has no strict-resume environment record. '
                    'Start a fresh run directory for this protocol.'
                )
            if saved_resume_environment != current_resume_environment:
                raise RuntimeError(
                    'Checkpoint software, code revision, or GPU environment '
                    'does not match this run. Restore the recorded environment '
                    'or start a fresh run directory.'
                )
        begin_epoch = checkpoint.get('epoch', 0)
        best_perf = checkpoint.get('perf', 0.0)
        best_epoch = checkpoint.get('best_epoch')
        best_name_values = checkpoint.get('best_name_values')
        last_epoch = checkpoint.get('epoch', begin_epoch)
        # load state dict if present
        if 'state_dict' in checkpoint:
            _load_state_dict_with_report(
                model,
                checkpoint['state_dict'],
                logger,
                context=checkpoint_file,
                strict=cfg.REPRODUCIBILITY.STRICT,
            )
        elif cfg.REPRODUCIBILITY.STRICT:
            raise RuntimeError('Checkpoint is missing model state.')

        if 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError as e:
                if cfg.REPRODUCIBILITY.STRICT:
                    raise RuntimeError(
                        'Could not restore optimizer state reproducibly.'
                    ) from e
                logger.warning("Could not load optimizer state from '%s' (%s).",
                               checkpoint_file, e)
        elif cfg.REPRODUCIBILITY.STRICT:
            raise RuntimeError('Checkpoint is missing optimizer state.')
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    if getattr(cfg.TRAIN, 'LR_SCHEDULE', 'multistep') == 'cosine':
        warmup_epochs = max(int(getattr(cfg.TRAIN, 'WARMUP_EPOCHS', 0)), 0)
        total_epochs = int(cfg.TRAIN.END_EPOCH)
        min_lr = float(getattr(cfg.TRAIN, 'MIN_LR', 0.0))
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=min_lr
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_epochs), eta_min=min_lr
            )
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch
        )

    if resume_from_ckpt and 'lr_scheduler' in checkpoint:
        try:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info('=> restored lr_scheduler state from checkpoint')
        except Exception as e:
            if cfg.REPRODUCIBILITY.STRICT:
                raise RuntimeError(
                    'Could not restore learning-rate scheduler state reproducibly.'
                ) from e
            logger.warning(
                'Could not restore lr_scheduler state from checkpoint (%s). '
                'Scheduler will continue from last_epoch=%d.',
                e, last_epoch
            )
    elif resume_from_ckpt and cfg.REPRODUCIBILITY.STRICT:
        raise RuntimeError('Checkpoint is missing learning-rate scheduler state.')

    if resume_from_ckpt:
        if 'rng_state' in checkpoint:
            restore_rng_state(checkpoint['rng_state'], train_generator)
            validation_state = checkpoint.get('validation_loader_generator_state')
            if validation_state is not None:
                validation_generator.set_state(validation_state)
            elif cfg.REPRODUCIBILITY.STRICT:
                raise RuntimeError(
                    'Checkpoint has no validation DataLoader RNG state.'
                )
            logger.info('=> restored Python, NumPy, PyTorch, CUDA, and DataLoader RNG state')
        elif cfg.REPRODUCIBILITY.STRICT:
            raise RuntimeError(
                'Checkpoint has no RNG state and cannot be resumed reproducibly. '
                'Use a fresh run directory or disable strict mode for a legacy continuation.'
            )

    # Prepare train logs file directly under LOG_DIR (base_dir)
    train_logs_file = os.path.join(base_dir, 'train_logs.txt')
    _rewrite_train_logs(train_logs_file, best_epoch, best_perf, best_name_values)

    history_file = os.path.join(base_dir, 'run_history.json')
    try:
        with open(history_file, 'r', encoding='utf-8') as handle:
            run_history = json.load(handle)
        if not isinstance(run_history, list):
            raise ValueError('run_history.json must contain a JSON list')
    except FileNotFoundError:
        run_history = []

    attempt_index = len(run_history)
    run_history.append({
        'attempt': attempt_index + 1,
        'started_at_utc': utc_timestamp(),
        'status': 'running',
        'command': shlex.join([sys.executable, *sys.argv]),
        'config_file': os.path.abspath(args.cfg),
        'checkpoint_file': os.path.abspath(checkpoint_file),
        'resume_requested': bool(cfg.RESUME_FROM_CKPT),
        'resumed': resume_from_ckpt,
        'starting_epoch': int(begin_epoch),
        'target_epochs': int(cfg.TRAIN.END_EPOCH),
        'protocol_hash': current_protocol_hash,
        'environment': run_environment,
    })
    atomic_write_json(history_file, run_history)

    training_state_file = os.path.join(base_dir, 'training_state.json')
    stop_request_file = os.path.join(base_dir, 'stop_after_epoch.request')

    def _write_training_state(status, completed_epochs, validation_perf=None):
        state = {
            'status': status,
            'updated_at_utc': utc_timestamp(),
            'completed_epochs': int(completed_epochs),
            'target_epochs': int(cfg.TRAIN.END_EPOCH),
            'best_epoch_index': best_epoch,
            'best_epoch_number': best_epoch + 1 if best_epoch is not None else None,
            'best_validation_performance': float(best_perf),
            'last_validation_performance': (
                float(validation_perf) if validation_perf is not None else None
            ),
            'checkpoint_file': os.path.abspath(checkpoint_file),
            'protocol_hash': current_protocol_hash,
            'run_attempt': attempt_index + 1,
        }
        atomic_write_json(training_state_file, state)

    completed_epochs = int(begin_epoch)
    stopped_early = False
    _write_training_state('running', completed_epochs)

    try:
        for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

            # train for one epoch (measure time)
            t0 = time.time()
            train_acc = train(cfg, train_loader, model, criterion, optimizer, epoch,
                              debug_images_directory, None, None,
                              grad_accum_steps=batch_plan.accumulation_steps)
            train_time = time.time() - t0

            lr_scheduler.step()

            # evaluate on validation set (measure time)
            t1 = time.time()
            name_values, perf_indicator = validate(
                cfg, valid_loader, valid_dataset, model, criterion,
                debug_images_directory, None, None, epoch)
            test_time = time.time() - t1

            if perf_indicator >= best_perf:
                best_perf = perf_indicator
                best_model = True
                best_epoch = epoch
                best_name_values = name_values
            else:
                best_model = False

            # Extract per-keypoint accuracies (Head, Eyes, Mouth, Mean)
            try:
                head_acc = float(name_values.get('Head', 0.0))
                eyes_acc = float(name_values.get('Eyes', 0.0))
                mouth_acc = float(name_values.get('Mouth', 0.0))
                mean_acc = float(name_values.get('Mean', perf_indicator))
            except Exception:
                head_acc = eyes_acc = mouth_acc = mean_acc = float(perf_indicator)

            # Upsert before checkpointing. If interrupted between these two
            # operations, replaying the epoch replaces this row instead of
            # duplicating it.
            epoch_line = f"{epoch}\t{head_acc:.6f}\t{eyes_acc:.6f}\t{mouth_acc:.6f}\t{mean_acc:.6f}\t{train_time:.4f}\t{test_time:.4f}"
            _rewrite_train_logs(
                train_logs_file, best_epoch, best_perf, best_name_values,
                epoch_line=epoch_line,
            )

            logger.info('=> atomically saving checkpoint to {}'.format(ckpt_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': best_perf,
                'validation_perf': perf_indicator,
                'best_epoch': best_epoch,
                'best_name_values': best_name_values,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'rng_state': capture_rng_state(train_generator),
                'validation_loader_generator_state': validation_generator.get_state(),
                'training_protocol': current_protocol,
                'training_protocol_hash': current_protocol_hash,
                'resume_environment': current_resume_environment,
            }, best_model, ckpt_dir)
            completed_epochs = epoch + 1
            _write_training_state('running', completed_epochs, perf_indicator)
            _update_attempt(
                history_file,
                attempt_index,
                last_completed_epoch=completed_epochs,
                last_checkpoint_at_utc=utc_timestamp(),
            )

            if os.path.isfile(stop_request_file):
                try:
                    os.unlink(stop_request_file)
                except FileNotFoundError:
                    pass
                stopped_early = completed_epochs < cfg.TRAIN.END_EPOCH
                logger.info(
                    '=> received safe-stop request after %d completed epoch(s)',
                    completed_epochs,
                )
                break
    except BaseException:
        _write_training_state('interrupted', completed_epochs)
        _update_attempt(
            history_file,
            attempt_index,
            status='interrupted',
            ended_at_utc=utc_timestamp(),
            last_completed_epoch=completed_epochs,
        )
        raise

    if stopped_early:
        _write_training_state('paused', completed_epochs)
        _update_attempt(
            history_file,
            attempt_index,
            status='paused',
            ended_at_utc=utc_timestamp(),
            last_completed_epoch=completed_epochs,
        )
        logger.info(
            '=> run paused safely; rerun the same config to resume at epoch %d',
            completed_epochs + 1,
        )
        return

    final_model_state_file = os.path.join(ckpt_dir, 'final_model.pth')
    logger.info('=> atomically saving final model state to {}'.format(final_model_state_file))
    atomic_torch_save(model.module.state_dict(), final_model_state_file)
    _rewrite_train_logs(train_logs_file, best_epoch, best_perf, best_name_values)
    _write_training_state('completed', cfg.TRAIN.END_EPOCH)
    _update_attempt(
        history_file,
        attempt_index,
        status='completed',
        ended_at_utc=utc_timestamp(),
        last_completed_epoch=int(cfg.TRAIN.END_EPOCH),
    )
    
if __name__ == '__main__':
    main()
