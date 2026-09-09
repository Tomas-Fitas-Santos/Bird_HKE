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
import re

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
from lib.utilities.utilities import create_logger
from lib.utilities.utilities import get_model_summary
from lib.utilities.reproducibility import capture_rng_state
from lib.utilities.reproducibility import environment_report
from lib.utilities.reproducibility import protocol_hash
from lib.utilities.reproducibility import resolve_batch_plan
from lib.utilities.reproducibility import restore_rng_state
from lib.utilities.reproducibility import seed_everything
from lib.utilities.reproducibility import seed_worker
from lib.utilities.reproducibility import training_protocol


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
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, 'resolved_config.yaml'), 'w', encoding='utf-8') as handle:
        handle.write(cfg.dump())
    with open(os.path.join(base_dir, 'environment.json'), 'w', encoding='utf-8') as handle:
        json.dump(run_environment, handle, indent=2, sort_keys=True)

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
    # log model summary, params, flops and memory usage
    try:
        # prepare device and dummy input
        device = torch.device('cuda' if use_cuda else 'cpu')
        img_h, img_w = cfg.MODEL.IMAGE_SIZE if hasattr(cfg.MODEL, 'IMAGE_SIZE') else (256, 256)
        dummy = torch.randn(1, 3, img_h, img_w).to(device)

        # move model to device for accurate memory/summary
        model.to(device)
        model.eval()

        summary = get_model_summary(model, dummy, verbose=False)
        model.train()
        logger.info('%s', summary)
        # Save model summary and config to the root LOG_DIR (no subfolders)
        try:
            os.makedirs(base_dir, exist_ok=True)
            with open(os.path.join(base_dir, 'model_summary.txt'), 'w', encoding='utf-8') as f:
                f.write(summary)
        except Exception:
            logger.warning('Failed to write model_summary.txt')
        # Copy the YAML config used for this run into LOG_DIR
        try:
            import shutil as _sh
            _sh.copyfile(args.cfg, os.path.join(base_dir, 'model_config.txt'))
        except Exception:
            logger.warning('Failed to write model_config.txt')

        # total parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info('Total parameters: %.2fM', total_params / 1e6)

        # try to parse GFLOPs from summary
        m = re.search(r'Total Multiply Adds .*?:\s*([0-9,\.]+) GFLOPs', summary)
        if m:
            flops = float(m.group(1).replace(',', ''))
            logger.info('Approx GFLOPs (conv+linear): %s', flops)

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
        logger.warning('Model summary/logging failed: %s', e)

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
    with open(os.path.join(base_dir, 'environment.json'), 'w', encoding='utf-8') as handle:
        json.dump(run_environment, handle, indent=2, sort_keys=True)



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

    def _rewrite_train_logs(
        logs_file, avg_train_time, avg_test_time, best_epoch_val, best_perf_val, best_values, epoch_line=None
    ):
        def _to_metric_map(values):
            if values is None:
                return {}
            if hasattr(values, 'get'):
                return values
            if isinstance(values, list):
                # common shape from some evaluators: [OrderedDict(...), ...]
                for item in values:
                    if hasattr(item, 'get') and 'Mean' in item:
                        return item
                for item in values:
                    if hasattr(item, 'get'):
                        return item
                # fallback: list of tuples
                try:
                    return dict(values)
                except Exception:
                    return {}
            return {}

        try:
            lines = []
            if os.path.exists(logs_file):
                with open(logs_file, 'r', encoding='utf-8') as f:
                    lines = [ln.rstrip('\n') for ln in f.readlines()]

            summary_idx = None
            for i, line in enumerate(lines):
                if line.strip() == '# Summary':
                    summary_idx = i
                    break
            if summary_idx is not None:
                lines = lines[:summary_idx]

            header = 'epoch\thead\teyes\tmouth\tmean\ttrain_time_s\tvalidation_time_s'
            if not lines or not lines[0].startswith('epoch\t'):
                lines.insert(0, header)

            if epoch_line:
                lines.append(epoch_line.rstrip('\n'))

            lines.append('')
            lines.append('# Summary')
            lines.append(f'Avg epoch train time (s): {float(avg_train_time):.4f}')
            lines.append(f'Avg epoch test time (s): {float(avg_test_time):.4f}')

            if best_epoch_val is not None:
                lines.append(f'Best epoch: {best_epoch_val} (perf={float(best_perf_val):.6f})')
                metrics = _to_metric_map(best_values)
                lines.append(
                    'Best epoch accuracies - Head: {0:.6f}, Eyes: {1:.6f}, Mouth: {2:.6f}, Mean: {3:.6f}'.format(
                        float(metrics.get('Head', 0.0)),
                        float(metrics.get('Eyes', 0.0)),
                        float(metrics.get('Mouth', 0.0)),
                        float(metrics.get('Mean', 0.0)),
                    )
                )
            else:
                lines.append('Best epoch: N/A')

            with open(logs_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines).rstrip() + '\n')
        except Exception as exc:
            logger.warning('Failed to update train_logs summary: %s', exc)

    # Prepare train logs file directly under LOG_DIR (base_dir)
    train_logs_file = os.path.join(base_dir, 'train_logs.txt')
    _rewrite_train_logs(train_logs_file, 0.0, 0.0, best_epoch, best_perf, best_name_values)

    epoch_train_times = []
    epoch_test_times = []

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

        # train for one epoch (measure time)
        t0 = time.time()
        train_acc = train(cfg, train_loader, model, criterion, optimizer, epoch,
                          debug_images_directory, None, None,
                          grad_accum_steps=batch_plan.accumulation_steps)
        train_time = time.time() - t0
        epoch_train_times.append(train_time)

        lr_scheduler.step()

        # evaluate on validation set (measure time)
        t1 = time.time()
        name_values, perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            debug_images_directory, None, None, epoch)
        test_time = time.time() - t1
        epoch_test_times.append(test_time)

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

        # Update logs after each epoch (append row, then update summary)
        avg_train_time = sum(epoch_train_times) / len(epoch_train_times) if epoch_train_times else 0.0
        avg_test_time = sum(epoch_test_times) / len(epoch_test_times) if epoch_test_times else 0.0
        epoch_line = f"{epoch}\t{head_acc:.6f}\t{eyes_acc:.6f}\t{mouth_acc:.6f}\t{mean_acc:.6f}\t{train_time:.4f}\t{test_time:.4f}"
        _rewrite_train_logs(train_logs_file, avg_train_time, avg_test_time, best_epoch, best_perf, best_name_values, epoch_line=epoch_line)

        logger.info('=> saving checkpoint to {}'.format(ckpt_dir))
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
        }, best_model, ckpt_dir)

    final_model_state_file = os.path.join(ckpt_dir, 'final_model.pth')
    logger.info('=> saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    # Final summary update (ensure latest stats are written)
    avg_train_time = sum(epoch_train_times) / len(epoch_train_times) if epoch_train_times else 0.0
    avg_test_time = sum(epoch_test_times) / len(epoch_test_times) if epoch_test_times else 0.0
    _rewrite_train_logs(train_logs_file, avg_train_time, avg_test_time, best_epoch, best_perf, best_name_values)
    
if __name__ == '__main__':
    main()
