import math
import json
import numpy as np
import os
import sys
from functools import lru_cache
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.uncertainty import hpd_region
from core.uncertainty import probability_map_moments
from utilities.transforms import get_affine_transform
from utilities.transforms import transform_preds


_SCORE_STATS = None


def reset_score_stats():
    """Reset running score statistics."""
    global _SCORE_STATS
    _SCORE_STATS = {
        'Confidence': _init_stats(),
    }


def print_score_stats(prefix='Score stats'):
    """Print running score statistics."""
    global _SCORE_STATS
    if not _SCORE_STATS:
        print(f"{prefix}: no data")
        return

    print(f"\n=== {prefix} ===")
    for name, stats in _SCORE_STATS.items():
        if stats['count'] == 0:
            print(f"{name}: no data")
            continue

        mean = stats['sum'] / stats['count']
        var = stats['sumsq'] / stats['count'] - mean ** 2
        std = math.sqrt(max(var, 0.0))
        print(
            f"{name}: min={stats['min']:.6f}, max={stats['max']:.6f}, "
            f"avg={mean:.6f}, std={std:.6f}, n={stats['count']}"
        )


def _init_stats():
    return {
        'count': 0,
        'sum': 0.0,
        'sumsq': 0.0,
        'min': float('inf'),
        'max': float('-inf'),
    }


def _update_stats(bucket, values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return

    bucket['count'] += int(arr.size)
    bucket['sum'] += float(np.sum(arr))
    bucket['sumsq'] += float(np.sum(arr ** 2))
    bucket['min'] = float(min(bucket['min'], float(np.min(arr))))
    bucket['max'] = float(max(bucket['max'], float(np.max(arr))))


def _update_score_stats(maxvals):
    global _SCORE_STATS
    if _SCORE_STATS is None:
        reset_score_stats()

    _update_stats(_SCORE_STATS['Confidence'], maxvals)


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def _compute_raw_scores(config, batch_heatmaps, maxvals):
    """Compute raw confidence scores from heatmaps."""
    try:
        mode = getattr(config.TEST, 'SCORE_MODE', 'max')
    except Exception:
        mode = 'max'
    mode = str(mode).lower()

    if mode != 'entropy':
        return maxvals

    # Entropy-based sharpness weighting
    try:
        beta = float(getattr(config.TEST, 'SCORE_ENTROPY_BETA', 1.0))
    except Exception:
        beta = 1.0
    try:
        eps = float(getattr(config.TEST, 'SCORE_EPS', 1e-6))
    except Exception:
        eps = 1e-6
    eps = max(1e-12, eps)

    b, j, h, w = batch_heatmaps.shape
    hm = batch_heatmaps.reshape((b, j, -1))
    hm_max = np.max(hm, axis=2, keepdims=True)
    exp = np.exp(hm - hm_max)
    prob = exp / (np.sum(exp, axis=2, keepdims=True) + eps)
    entropy = -np.sum(prob * np.log(prob + eps), axis=2, keepdims=True)
    entropy_norm = entropy / (np.log(h * w) + eps)

    return maxvals * np.exp(-beta * entropy_norm)


def _activate_scores(config, raw_scores):
    """Normalize raw scores to 0-1 if configured."""
    try:
        act = getattr(config.TEST, 'SCORE_ACTIVATION', 'none')
    except Exception:
        act = 'none'

    if act is None:
        act = 'none'

    act = str(act).lower()
    scores = raw_scores

    if act == 'sigmoid':
        try:
            temperature = float(getattr(config.TEST, 'SCORE_TEMPERATURE', 1.0))
        except Exception:
            temperature = 1.0
        temperature = max(1e-6, temperature)
        scores = 1.0 / (1.0 + np.exp(-scores / temperature))

    try:
        if getattr(config.TEST, 'SCORE_CLIP', True):
            scores = np.clip(scores, 0.0, 1.0)
    except Exception:
        scores = np.clip(scores, 0.0, 1.0)

    return scores


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)
    _update_score_stats(maxvals)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals


def _to_numpy(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _numpy_sparsemax(values):
    shifted = values - np.max(values, axis=-1, keepdims=True)
    ordered = np.sort(shifted, axis=-1)[..., ::-1]
    ranks = np.arange(1, values.shape[-1] + 1, dtype=values.dtype)
    cumulative = np.cumsum(ordered, axis=-1)
    support = 1 + ranks * ordered > cumulative
    support_size = np.maximum(support.sum(axis=-1, keepdims=True), 1)
    tau_sum = np.take_along_axis(cumulative, support_size - 1, axis=-1)
    tau = (tau_sum - 1) / support_size
    return np.maximum(shifted - tau, 0)


def spatial_probability_numpy(logits, distribution='softmax', temperature=1.0):
    logits = np.asarray(logits)
    shape = logits.shape
    flat = logits.reshape(shape[0], shape[1], -1) / max(float(temperature), 1e-6)
    if str(distribution).lower() == 'softmax':
        shifted = flat - np.max(flat, axis=-1, keepdims=True)
        probability = np.exp(shifted)
        probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-12)
    elif str(distribution).lower() == 'sparsemax':
        probability = _numpy_sparsemax(flat)
    else:
        raise ValueError(f'Unknown spatial distribution: {distribution!r}')
    return probability.reshape(shape)


@lru_cache(maxsize=8)
def _read_calibration(path, modified_time):
    del modified_time
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _load_calibration(config):
    try:
        path = str(config.UNCERTAINTY.CALIBRATION_FILE)
    except Exception:
        path = ''
    if not path:
        return {}
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Uncertainty calibration file not found: {path}')
    return _read_calibration(path, os.path.getmtime(path))


def _calibrated_probability_maps(config, output, calibration):
    logits = _to_numpy(output['location_logits'])
    probability_maps = _to_numpy(output['probability_maps'])
    if 'location_temperature' in calibration:
        probability_maps = spatial_probability_numpy(
            logits,
            distribution=config.UNCERTAINTY.DISTRIBUTION,
            temperature=float(calibration['location_temperature']),
        )
    return probability_maps


def _calibrated_reliability(output, calibration):
    quality_temperature = max(
        float(calibration.get('quality_temperature', 1.0)), 1e-6
    )
    visibility_temperature = max(
        float(calibration.get('visibility_temperature', 1.0)), 1e-6
    )
    quality = 1.0 / (
        1.0 + np.exp(-_to_numpy(output['quality_logits']) / quality_temperature)
    )
    visibility = 1.0 / (
        1.0 + np.exp(-_to_numpy(output['visibility_logits']) / visibility_temperature)
    )
    return quality, visibility


def get_probabilistic_preds(config, output, center, scale):
    """Decode unchanged pose logits and add calibrated keypoint confidence."""
    calibration = _load_calibration(config)
    predictions, _ = get_final_preds(
        config, _to_numpy(output['location_logits']), center, scale
    )

    quality, _ = _calibrated_reliability(output, calibration)
    scores = np.clip(quality[..., None], 0.0, 1.0).astype(np.float32)
    _update_score_stats(scores)
    return predictions, scores


def get_pose_output_uncertainty(config, output, center, scale):
    """Return interpretable per-keypoint uncertainty for annotated or raw video."""
    if not isinstance(output, dict):
        return None
    calibration = _load_calibration(config)
    probability_maps = _calibrated_probability_maps(config, output, calibration)
    quality, visibility = _calibrated_reliability(output, calibration)
    statistics = probability_map_moments(probability_maps)
    _, _, height, width = probability_maps.shape

    covariance_image = np.empty_like(statistics['covariance'])
    mean_image = np.empty_like(statistics['mean'])
    pixel_area_scale = np.empty(probability_maps.shape[:2], dtype=np.float64)
    image_to_heatmap = np.empty((probability_maps.shape[0], 2, 3), dtype=np.float64)
    for sample in range(probability_maps.shape[0]):
        image_to_heatmap[sample] = get_affine_transform(
            center[sample], scale[sample], 0, [width, height]
        )
        inverse = get_affine_transform(
            center[sample], scale[sample], 0, [width, height], inv=1
        )
        linear = inverse[:, :2]
        determinant = abs(float(np.linalg.det(linear)))
        for joint in range(probability_maps.shape[1]):
            covariance_image[sample, joint] = (
                linear @ statistics['covariance'][sample, joint] @ linear.T
            )
            point = np.append(statistics['mean'][sample, joint], 1.0)
            mean_image[sample, joint] = inverse @ point
            pixel_area_scale[sample, joint] = determinant

    eigenvalues = np.linalg.eigvalsh(covariance_image)
    standard_deviation = np.sqrt(np.maximum(eigenvalues, 0.0))[..., ::-1]
    normalized_entropy = statistics['entropy'] / max(math.log(height * width), 1e-12)

    conformal_mass = calibration.get('hpd_mass_per_joint')
    conformal_area = None
    conformal_regions = None
    if conformal_mass is not None:
        masses = np.asarray(conformal_mass, dtype=np.float64)
        conformal_area = np.zeros(probability_maps.shape[:2], dtype=np.float64)
        conformal_regions = []
        for sample in range(probability_maps.shape[0]):
            sample_regions = []
            for joint in range(probability_maps.shape[1]):
                if np.isfinite(masses[joint]):
                    region = hpd_region(
                        probability_maps[sample, joint], masses[joint]
                    )
                    cells = region.sum()
                    conformal_area[sample, joint] = (
                        float(cells) * pixel_area_scale[sample, joint]
                    )
                    flat = region.reshape(-1).astype(np.int8)
                    padded = np.pad(flat, (1, 1), constant_values=0)
                    changes = np.diff(padded)
                    starts = np.flatnonzero(changes == 1)
                    ends = np.flatnonzero(changes == -1)
                    sample_regions.append(
                        [[int(start), int(end - start)] for start, end in zip(starts, ends)]
                    )
                else:
                    sample_regions.append([])
            conformal_regions.append(sample_regions)

    return {
        'distribution_mean': mean_image,
        'covariance': covariance_image,
        'major_minor_std': standard_deviation,
        'normalized_entropy': normalized_entropy,
        'hpd90_area_heatmap_cells': statistics['hpd90_area'],
        'conformal_area_pixels': conformal_area,
        'conformal_region_rle_heatmap': conformal_regions,
        'image_to_heatmap_affine': image_to_heatmap,
        'heatmap_size': np.tile(
            np.asarray([width, height], dtype=np.int64),
            (probability_maps.shape[0], 1),
        ),
        'quality': quality,
        'visibility_probability': visibility,
    }


def get_pose_output_preds(config, output, center, scale):
    """Decode either a Phase-1 heatmap tensor or a Phase-2 output bundle."""
    if isinstance(output, dict):
        return get_probabilistic_preds(config, output, center, scale)
    heatmaps = _to_numpy(output)
    return get_final_preds(config, heatmaps, center, scale)
