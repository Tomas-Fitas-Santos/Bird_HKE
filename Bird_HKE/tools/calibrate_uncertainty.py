"""Fit post-hoc calibration for a trained uncertainty-enabled pose model.

This command consumes ``annot/calibration.json``.  It never updates model
weights and never uses the validation split used for model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import birdgaze  # noqa: E402
from lib.config.default import _C as cfg  # noqa: E402
from lib.config.default import update_config  # noqa: E402
from lib.core.inference import spatial_probability_numpy  # noqa: E402
from lib.core.uncertainty import calibration_error  # noqa: E402
from lib.core.uncertainty import fit_binary_temperature  # noqa: E402
from lib.core.uncertainty import fit_hpd_mass_thresholds  # noqa: E402
from lib.core.uncertainty import fit_spatial_temperature  # noqa: E402
from lib.utilities.reproducibility import seed_everything  # noqa: E402
from lib.utilities.reproducibility import seed_worker  # noqa: E402
from models import get_pose_net  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True, help='uncertainty-enabled experiment YAML')
    parser.add_argument('--checkpoint', default='', help='trained model state; defaults to TEST.POSE_MODEL_FILE')
    parser.add_argument('--output', default='', help='output JSON; defaults beside the checkpoint')
    parser.add_argument('--split', default='', help='annotation stem; defaults to UNCERTAINTY.CALIBRATION_SET')
    parser.add_argument('--device', default='auto', help="'auto', 'cpu', or a CUDA device such as 'cuda:0'")
    parser.add_argument('--modelDir', default='')
    parser.add_argument('--logDir', default='')
    parser.add_argument('opts', nargs=argparse.REMAINDER)
    return parser.parse_args()


def _state_dict(checkpoint):
    loaded = torch.load(checkpoint, map_location='cpu')
    if isinstance(loaded, dict) and 'state_dict' in loaded:
        loaded = loaded['state_dict']
    if not isinstance(loaded, dict):
        raise TypeError('checkpoint does not contain a model state dictionary')
    return {
        (key[7:] if key.startswith('module.') else key): value
        for key, value in loaded.items()
    }


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _ground_truth_coordinates(target):
    _, _, _, width = target.shape
    index = target.flatten(start_dim=2).argmax(dim=-1)
    return torch.stack(
        (torch.remainder(index, width), torch.div(index, width, rounding_mode='floor')),
        dim=-1,
    )


def _quality_targets(probability_maps, coordinates, sigma):
    _, _, _, width = probability_maps.shape
    decoded = np.empty_like(probability_maps)
    for sample in range(probability_maps.shape[0]):
        for joint in range(probability_maps.shape[1]):
            decoded[sample, joint] = cv2.GaussianBlur(
                probability_maps[sample, joint],
                ksize=(0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_CONSTANT,
            )
    index = decoded.reshape(decoded.shape[0], decoded.shape[1], -1).argmax(axis=-1)
    predicted_x = np.remainder(index, width)
    predicted_y = np.floor_divide(index, width)
    distance_squared = (
        (predicted_x - coordinates[..., 0]) ** 2
        + (predicted_y - coordinates[..., 1]) ** 2
    )
    return np.exp(-distance_squared / (2 * sigma ** 2))


def _spatial_nll(logits, coordinates, valid, temperature):
    probability = spatial_probability_numpy(logits, 'softmax', temperature)
    losses = []
    for sample, joint in zip(*np.nonzero(valid)):
        x = int(np.clip(coordinates[sample, joint, 0], 0, probability.shape[3] - 1))
        y = int(np.clip(coordinates[sample, joint, 1], 0, probability.shape[2] - 1))
        losses.append(-np.log(max(float(probability[sample, joint, y, x]), 1e-12)))
    return float(np.mean(losses)) if losses else float('nan')


def main():
    args = parse_args()
    update_config(cfg, args)
    if not bool(cfg.UNCERTAINTY.ENABLED):
        raise ValueError(
            'The calibration config must set UNCERTAINTY.ENABLED: true and match the checkpoint.'
        )
    if str(cfg.UNCERTAINTY.DISTRIBUTION).lower() != 'softmax':
        raise ValueError(
            'Post-hoc temperature fitting currently requires '
            'UNCERTAINTY.DISTRIBUTION: softmax. Sparsemax is retained only as '
            'an explicitly uncalibrated ablation.'
        )

    checkpoint = args.checkpoint or cfg.TEST.POSE_MODEL_FILE
    if not checkpoint:
        raise ValueError('provide --checkpoint or TEST.POSE_MODEL_FILE')
    checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

    if args.device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    seed_everything(
        cfg.REPRODUCIBILITY.SEED,
        deterministic_algorithms=cfg.REPRODUCIBILITY.USE_DETERMINISTIC_ALGORITHMS,
        warn_only=cfg.REPRODUCIBILITY.WARN_ONLY,
    )

    model = get_pose_net(cfg, is_train=False)
    incompatible = model.load_state_dict(_state_dict(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f'checkpoint mismatch: missing={incompatible.missing_keys}, '
            f'unexpected={incompatible.unexpected_keys}'
        )
    model.to(device).eval()

    split = args.split or cfg.UNCERTAINTY.CALIBRATION_SET
    annotation_file = Path(cfg.DATASET.ROOT) / cfg.DATASET.ANNOT_DIR / f'{split}.json'
    if not annotation_file.is_file():
        raise FileNotFoundError(
            f'Calibration annotations were not found at {annotation_file}. '
            'Calibration must use an annotation split independent of train and val.'
        )
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    dataset = eval(cfg.DATASET.NAME_ + cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.ROOT,
        split,
        False,
        transforms.Compose([transforms.ToTensor(), normalize]),
    )
    generator = torch.Generator().manual_seed(int(cfg.REPRODUCIBILITY.SEED) + 2)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    collected = {
        'location_logits': [],
        'quality_logits': [],
        'visibility_logits': [],
        'coordinates': [],
        'coordinate_valid': [],
        'visibility_known': [],
        'visibility_target': [],
    }
    with torch.no_grad():
        for inputs, target, target_weight, meta in loader:
            output = model(inputs.to(device, non_blocking=True))
            if not isinstance(output, dict):
                raise TypeError('checkpoint/model did not return uncertainty outputs')
            collected['location_logits'].append(output['location_logits'].cpu().numpy())
            collected['quality_logits'].append(output['quality_logits'].cpu().numpy())
            collected['visibility_logits'].append(output['visibility_logits'].cpu().numpy())
            collected['coordinates'].append(_ground_truth_coordinates(target).numpy())
            collected['coordinate_valid'].append(target_weight[..., 0].numpy() > 0)
            collected['visibility_known'].append(meta['visibility_known'][..., 0].numpy() > 0)
            collected['visibility_target'].append(meta['visibility_target'][..., 0].numpy())

    arrays = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    sigma = max(
        float(cfg.UNCERTAINTY.BKS_SIGMA_FRACTION)
        * min(arrays['location_logits'].shape[2:]),
        1e-6,
    )
    location_temperature = fit_spatial_temperature(
        arrays['location_logits'], arrays['coordinates'], arrays['coordinate_valid']
    )
    calibrated_probability = spatial_probability_numpy(
        arrays['location_logits'],
        cfg.UNCERTAINTY.DISTRIBUTION,
        location_temperature,
    )
    quality_target = _quality_targets(
        calibrated_probability, arrays['coordinates'], sigma
    )
    quality_temperature = fit_binary_temperature(
        arrays['quality_logits'], quality_target, arrays['coordinate_valid']
    )
    visibility_temperature = fit_binary_temperature(
        arrays['visibility_logits'],
        arrays['visibility_target'],
        arrays['visibility_known'],
    )
    thresholds, conformal_counts = fit_hpd_mass_thresholds(
        calibrated_probability,
        arrays['coordinates'],
        arrays['coordinate_valid'],
        coverage=cfg.UNCERTAINTY.CONFORMAL_COVERAGE,
    )

    raw_quality = 1 / (1 + np.exp(-arrays['quality_logits']))
    calibrated_quality = 1 / (
        1 + np.exp(-arrays['quality_logits'] / quality_temperature)
    )
    raw_visibility = 1 / (1 + np.exp(-arrays['visibility_logits']))
    calibrated_visibility = 1 / (
        1 + np.exp(-arrays['visibility_logits'] / visibility_temperature)
    )
    result = {
        'format_version': 1,
        'checkpoint': checkpoint,
        'checkpoint_sha256': _sha256(checkpoint),
        'calibration_annotation': str(annotation_file.resolve()),
        'samples': int(arrays['location_logits'].shape[0]),
        'distribution': str(cfg.UNCERTAINTY.DISTRIBUTION),
        'location_temperature': location_temperature,
        'quality_temperature': quality_temperature,
        'visibility_temperature': visibility_temperature,
        'conformal_coverage': float(cfg.UNCERTAINTY.CONFORMAL_COVERAGE),
        'hpd_mass_per_joint': thresholds.tolist(),
        'conformal_samples_per_joint': conformal_counts.tolist(),
        'metrics': {
            'location_nll_before': _spatial_nll(
                arrays['location_logits'], arrays['coordinates'],
                arrays['coordinate_valid'], float(cfg.UNCERTAINTY.TEMPERATURE)
            ),
            'location_nll_after': _spatial_nll(
                arrays['location_logits'], arrays['coordinates'],
                arrays['coordinate_valid'], location_temperature
            ),
            'quality_before': calibration_error(
                raw_quality, quality_target, arrays['coordinate_valid']
            ),
            'quality_after': calibration_error(
                calibrated_quality, quality_target, arrays['coordinate_valid']
            ),
            'visibility_before': calibration_error(
                raw_visibility, arrays['visibility_target'], arrays['visibility_known']
            ),
            'visibility_after': calibration_error(
                calibrated_visibility, arrays['visibility_target'], arrays['visibility_known']
            ),
        },
    }
    output = args.output or str(Path(checkpoint).with_name('uncertainty_calibration.json'))
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(f'Calibrated {result["samples"]} images from {annotation_file}')
    print(f'Location temperature:   {location_temperature:.6f}')
    print(f'Quality temperature:    {quality_temperature:.6f}')
    print(f'Visibility temperature: {visibility_temperature:.6f}')
    print(f'Conformal HPD masses:   {thresholds.tolist()}')
    print(f'Wrote: {output_path}')


if __name__ == '__main__':
    main()
