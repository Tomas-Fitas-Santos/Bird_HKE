import math
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
