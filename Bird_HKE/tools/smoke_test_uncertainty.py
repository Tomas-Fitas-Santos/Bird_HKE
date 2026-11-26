"""Run one FD uncertainty-training update without writing checkpoints or logs."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Deterministic cuBLAS requires this before CUDA can initialize.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import birdgaze  # noqa: E402
from lib.config.default import _C as cfg  # noqa: E402
from lib.config.default import update_config  # noqa: E402
from lib.core.loss import JointsMSELoss  # noqa: E402
from lib.core.loss import build_pose_criterion  # noqa: E402
from lib.utilities.reproducibility import seed_everything  # noqa: E402
from lib.utilities.utilities import clip_pose_and_auxiliary_gradients  # noqa: E402
from lib.utilities.utilities import get_optimizer  # noqa: E402
from models import get_pose_net  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True, help='FD experiment YAML')
    # Keep the namespace compatible with update_config and train.py.
    parser.add_argument('--modelDir', default='')
    parser.add_argument('--logDir', default='')
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Optional YACS overrides, for example DATASET.ROOT /data/BirdGaze',
    )
    return parser.parse_args()


def _last_output(outputs):
    return outputs[-1] if isinstance(outputs, list) else outputs


def _validate_output(output, batch_size):
    required = {
        'location_logits',
        'probability_maps',
        'quality_logits',
        'visibility_logits',
    }
    if not isinstance(output, dict) or set(output) != required:
        keys = set(output) if isinstance(output, dict) else type(output).__name__
        raise RuntimeError(f'Unexpected uncertainty output keys/type: {keys}')

    expected_maps = (
        batch_size,
        int(cfg.MODEL.NUM_JOINTS),
        int(cfg.MODEL.HEATMAP_SIZE[1]),
        int(cfg.MODEL.HEATMAP_SIZE[0]),
    )
    if tuple(output['probability_maps'].shape) != expected_maps:
        raise RuntimeError(
            f'Expected probability-map shape {expected_maps}, found '
            f'{tuple(output["probability_maps"].shape)}'
        )
    expected_scores = (batch_size, int(cfg.MODEL.NUM_JOINTS))
    for key in ('quality_logits', 'visibility_logits'):
        if tuple(output[key].shape) != expected_scores:
            raise RuntimeError(
                f'Expected {key} shape {expected_scores}, found '
                f'{tuple(output[key].shape)}'
            )
    for key, value in output.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f'{key} contains a non-finite value')

    probability_sums = output['probability_maps'].sum(dim=(2, 3))
    normalization_error = float((probability_sums - 1.0).abs().max())
    if normalization_error > 1e-5:
        raise RuntimeError(
            f'Probability maps are not normalized; max error={normalization_error}'
        )
    return normalization_error


def _gradient_norms(model):
    squared = {'model': 0.0, 'uncertainty_head': 0.0, 'pose_network': 0.0}
    counts = {'model': 0, 'uncertainty_head': 0, 'pose_network': 0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f'Non-finite gradient in {name}')
        value = float(parameter.grad.detach().float().square().sum())
        group = 'uncertainty_head' if 'probabilistic_output.' in name else 'pose_network'
        squared['model'] += value
        squared[group] += value
        counts['model'] += 1
        counts[group] += 1
    norms = {key: math.sqrt(value) for key, value in squared.items()}
    if counts['model'] == 0 or norms['model'] == 0:
        raise RuntimeError('No non-zero model gradients were produced')
    if counts['uncertainty_head'] == 0 or norms['uncertainty_head'] == 0:
        raise RuntimeError('The quality/visibility head received no non-zero gradients')
    if counts['pose_network'] == 0 or norms['pose_network'] == 0:
        raise RuntimeError('The pose network received no non-zero gradients')
    return norms


def main():
    args = parse_args()
    update_config(cfg, args)

    config_name = Path(args.cfg).stem
    if '_FD' not in config_name:
        raise ValueError('The uncertainty smoke test accepts FD configurations only.')
    if not bool(cfg.UNCERTAINTY.ENABLED):
        raise ValueError('UNCERTAINTY.ENABLED must be true for an FD smoke test.')
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required because the production trainer uses CUDA.')
    if not cfg.GPUS:
        raise ValueError('Configure at least one GPU id in GPUS.')

    gpu_ids = [int(gpu) for gpu in cfg.GPUS]
    invalid_gpu_ids = [
        gpu for gpu in gpu_ids
        if gpu < 0 or gpu >= torch.cuda.device_count()
    ]
    if invalid_gpu_ids:
        raise ValueError(
            f'Configured GPUs {invalid_gpu_ids} are unavailable; this host exposes '
            f'{torch.cuda.device_count()} CUDA device(s).'
        )
    gpu_id = gpu_ids[0]
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    seed_everything(
        cfg.REPRODUCIBILITY.SEED,
        deterministic_algorithms=cfg.REPRODUCIBILITY.USE_DETERMINISTIC_ALGORITHMS,
        warn_only=cfg.REPRODUCIBILITY.WARN_ONLY,
    )
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED
    torch.backends.cuda.matmul.allow_tf32 = cfg.REPRODUCIBILITY.ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = cfg.REPRODUCIBILITY.ALLOW_TF32
    torch.set_float32_matmul_precision(cfg.REPRODUCIBILITY.MATMUL_PRECISION)

    annotation_file = (
        Path(cfg.DATASET.ROOT)
        / cfg.DATASET.ANNOT_DIR
        / f'{cfg.DATASET.TRAIN_SET}.json'
    )
    images_dir = Path(cfg.DATASET.ROOT) / 'images'
    if not annotation_file.is_file():
        raise FileNotFoundError(f'Training annotations not found: {annotation_file}')
    if not images_dir.is_dir():
        raise FileNotFoundError(f'Image directory not found: {images_dir}')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    dataset_class = getattr(birdgaze, cfg.DATASET.DATASET)
    dataset = dataset_class(
        cfg,
        cfg.DATASET.ROOT,
        cfg.DATASET.TRAIN_SET,
        True,
        transforms.Compose([transforms.ToTensor(), normalize]),
    )
    if len(dataset) == 0:
        raise RuntimeError('The configured training split is empty.')

    generator = torch.Generator()
    generator.manual_seed(int(cfg.REPRODUCIBILITY.SEED))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg.TRAIN.BATCH_SIZE_PER_GPU) * len(gpu_ids),
        shuffle=bool(cfg.TRAIN.SHUFFLE),
        num_workers=0,
        pin_memory=bool(cfg.PIN_MEMORY),
        drop_last=False,
        generator=generator,
    )
    input_tensor, target, target_weight, meta = next(iter(loader))
    input_tensor = input_tensor.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    target_weight = target_weight.to(device, non_blocking=True)

    model = get_pose_net(cfg, is_train=True).to(device)
    model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)
    criterion = build_pose_criterion(cfg).to(device)
    optimizer = get_optimizer(cfg, model)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    outputs = model(input_tensor)
    output_items = outputs if isinstance(outputs, list) else [outputs]
    normalization_error = max(
        _validate_output(output, input_tensor.shape[0]) for output in output_items
    )
    loss = sum(
        criterion(output, target, target_weight, meta) for output in output_items
    )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError(f'Loss is not finite: {float(loss.detach())}')
    baseline_criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).to(device)
    pose_only_loss = sum(
        baseline_criterion(
            output['location_logits'], target, target_weight, meta
        )
        for output in output_items
    )
    combined_pose_gradients = torch.autograd.grad(
        loss,
        [output['location_logits'] for output in output_items],
        retain_graph=True,
    )
    baseline_pose_gradients = torch.autograd.grad(
        pose_only_loss,
        [output['location_logits'] for output in output_items],
        retain_graph=True,
    )
    if not all(
        torch.equal(combined, baseline)
        for combined, baseline in zip(
            combined_pose_gradients, baseline_pose_gradients
        )
    ):
        raise RuntimeError(
            'Auxiliary uncertainty changed the pose-loss gradient'
        )
    loss.backward()
    gradient_norms = _gradient_norms(model)
    if cfg.TRAIN.CLIP_GRAD_NORM > 0:
        clip_pose_and_auxiliary_gradients(model, cfg.TRAIN.CLIP_GRAD_NORM)
    optimizer.step()
    torch.cuda.synchronize(device)

    components = getattr(criterion, 'last_components', {})
    print('Uncertainty training smoke test: PASS')
    print(f'  config:              {args.cfg}')
    device_names = ', '.join(torch.cuda.get_device_name(gpu) for gpu in gpu_ids)
    print(f'  devices:             {device_names}')
    print(f'  dataset samples:     {len(dataset)}')
    print(f'  batch per GPU:       {cfg.TRAIN.BATCH_SIZE_PER_GPU}')
    print(f'  global micro-batch:  {input_tensor.shape[0]}')
    print(f'  probability shape:   {tuple(_last_output(outputs)["probability_maps"].shape)}')
    print(f'  normalization error: {normalization_error:.3e}')
    print(f'  loss:                {float(loss.detach()):.6f}')
    print('  pose gradient match: exact')
    for key, value in components.items():
        print(f'  {key} loss:'.ljust(23) + f'{value:.6f}')
    for key, value in gradient_norms.items():
        print(f'  {key} grad norm:'.ljust(23) + f'{value:.6f}')
    print('No checkpoint or log files were written.')


if __name__ == '__main__':
    main()
