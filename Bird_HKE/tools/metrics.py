"""
Evaluation Metrics Module
Calculate PCK and other evaluation metrics against ground truth.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path


def evaluate_poses(poses_initial, poses_final, gt_annotations, results_dir, fps=None, has_final=True, pred_bboxes=None):
    """
    Evaluate pose estimations against ground truth.
    
    Args:
        poses_initial: Initial pose estimates
        poses_final: Final smoothed pose estimates
        gt_annotations: Ground truth annotations
        results_dir: Directory to save results
    
    Returns:
        Dictionary of metrics
    """
    if gt_annotations is None:
        print("No ground truth provided - skipping evaluation")
        return {}
    
    print("\n=== Evaluating Poses ===")

    initial_dir = Path(results_dir) / 'initial'
    initial_dir.mkdir(parents=True, exist_ok=True)

    final_dir = Path(results_dir) / 'final'
    if has_final and poses_final is not None:
        final_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'initial': _evaluate_split(
            poses_initial, gt_annotations, initial_dir, fps,
            label='Initial', pred_bboxes=pred_bboxes
        )
    }

    if has_final and poses_final is not None:
        metrics['final'] = _evaluate_split(
            poses_final, gt_annotations, final_dir, fps,
            label='Final', pred_bboxes=pred_bboxes
        )

    return metrics


def evaluate_jitter_only(poses_initial, poses_final, results_dir, fps=None, has_final=True):
    """Compute GT-independent temporal metrics (normalized jitter) and save them."""
    print("\n=== Evaluating Jitter (no GT) ===")

    dt = 1.0 / float(fps) if fps else 1.0
    pred_initial = np.asarray(poses_initial, dtype=float) if poses_initial is not None else None

    metrics = {}

    initial_dir = Path(results_dir) / 'initial'
    initial_dir.mkdir(parents=True, exist_ok=True)
    initial_temporal = _temporal_metrics(pred_initial, None, dt)
    metrics['initial'] = {'normalized_jitter': initial_temporal['normalized_jitter']}
    save_metrics(metrics['initial'], initial_dir, 'Initial')

    if has_final and poses_final is not None:
        final_dir = Path(results_dir) / 'final'
        final_dir.mkdir(parents=True, exist_ok=True)
        pred_final = np.asarray(poses_final, dtype=float)
        final_temporal = _temporal_metrics(pred_final, None, dt)
        metrics['final'] = {'normalized_jitter': final_temporal['normalized_jitter']}
        save_metrics(metrics['final'], final_dir, 'Final')

    return metrics


def compute_pck(pred, gt_joints, visibility, ref_size, threshold=0.05, apply_visibility=True):
    """
    Compute PCK (Percentage of Correct Keypoints) for a single frame.
    
    Args:
        pred: Predicted joints (N, 2)
        gt_joints: Ground truth joints (N, 2)
        visibility: Visibility flags (N,)
        headsize: Reference size for normalization
        threshold: Distance threshold (default 0.05 = 5% of headsize)
    
    Returns:
        Array of binary correctness per joint
    """
    if pred is None or ref_size is None or ref_size <= 0:
        return np.zeros(len(gt_joints))

    pred = np.array(pred)
    if pred.shape != gt_joints.shape:
        return np.zeros(len(gt_joints))
    
    # Handle NaN values in predictions
    if np.any(np.isnan(pred)):
        return np.zeros(len(gt_joints))
    
    # Calculate normalized error
    distances = np.linalg.norm(pred - gt_joints, axis=1)
    normalized_distances = distances / float(ref_size)
    
    # Check if within threshold
    correct = (normalized_distances <= threshold).astype(float)
    
    # Mask by visibility (optional)
    if apply_visibility:
        valid = visibility.astype(bool)
        correct[~valid] = 0.0
    
    return correct


def _normalize_visibility(vis):
    vis = np.array(vis)
    if vis.ndim == 2 and vis.shape[1] > 0:
        vis = vis[:, 0]
    return (vis > 0).astype(float)


def _bbox_reference_size(bbox):
    if bbox is None:
        return None
    try:
        if isinstance(bbox, (list, tuple)) and len(bbox) == 2 and len(bbox[0]) == 2:
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
        else:
            return None
        w = abs(float(x2) - float(x1))
        h = abs(float(y2) - float(y1))
        ref = max(w, h)
        return ref if ref > 0 else None
    except Exception:
        return None


def _pred_bbox_for_frame(pred_bboxes, frame_idx):
    if pred_bboxes is None or frame_idx >= len(pred_bboxes):
        return None

    frame_boxes = pred_bboxes[frame_idx]
    if not frame_boxes:
        return None

    # Common detector output shape: list of boxes per frame, use the first box.
    if isinstance(frame_boxes, (list, tuple)) and len(frame_boxes) > 0:
        first = frame_boxes[0]
        # If this is already [x1, y1, x2, y2], use it directly.
        if isinstance(first, (int, float, np.number)) and len(frame_boxes) == 4:
            return frame_boxes
        # If this is [[x1,y1,x2,y2], ...], return first box.
        if isinstance(first, (list, tuple)):
            return first

    return None


def _mean_masked(values, mask):
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.shape[0] == 0 or mask.sum() == 0:
        return 0.0
    return float(values[mask].mean())


def _auc_from_thresholds(pck_dict, thresholds):
    vals = []
    for thr in thresholds:
        if pck_dict[thr]:
            vals.append(np.mean(pck_dict[thr]))
        else:
            vals.append(0.0)
    if not vals:
        return 0.0
    return float(np.trapz(vals, x=thresholds))


def _collect_gt_sequences(gt_annotations, max_len):
    gt_seq = []
    gt_vis_seq = []
    for frame_idx in range(min(len(gt_annotations), max_len)):
        gt_item = gt_annotations[frame_idx]
        gt_joints = np.array(gt_item.get('joints'))
        gt_vis = _normalize_visibility(gt_item.get('joints_vis', np.ones(gt_joints.shape[0])))
        gt_seq.append(gt_joints)
        gt_vis_seq.append(gt_vis)
    return np.array(gt_seq), np.array(gt_vis_seq)


def _temporal_metrics(pred_seq, gt_seq, dt, vis_seq=None):
    if pred_seq is None:
        return {
            'normalized_jitter': 0.0,
            'velocity_error': 0.0,
            'acceleration_error': 0.0
        }

    pred_seq = np.asarray(pred_seq, dtype=float)
    metrics = {
        'normalized_jitter': 0.0,
        'velocity_error': 0.0,
        'acceleration_error': 0.0
    }

    if pred_seq.ndim != 3 or pred_seq.shape[0] < 2:
        return metrics

    valid_mask = ~np.any(np.isnan(pred_seq), axis=2)
    if vis_seq is not None and vis_seq.shape[:2] == valid_mask.shape:
        valid_mask = valid_mask & (vis_seq > 0)

    # Normalized Jitter (GT-independent):
    #   NJ_t = ||p_{t+1} - 2*p_t + p_{t-1}|| / (||p_{t+1} - p_{t-1}|| / 2 + eps)
    # Separates real fast motion (high accel + high speed -> moderate NJ)
    # from model instability (high accel + low speed -> high NJ).
    if pred_seq.shape[0] >= 3:
        eps = 1e-6
        accel = pred_seq[2:] - 2 * pred_seq[1:-1] + pred_seq[:-2]
        accel_norm = np.linalg.norm(accel, axis=2)  # (T-2, J)
        span = pred_seq[2:] - pred_seq[:-2]
        speed_proxy = np.linalg.norm(span, axis=2) / 2.0  # (T-2, J)
        nj = accel_norm / (speed_proxy + eps)
        valid_nj = valid_mask[2:] & valid_mask[1:-1] & valid_mask[:-2]
        metrics['normalized_jitter'] = _masked_mean(nj, valid_nj)

    if gt_seq is not None and gt_seq.shape[:2] == pred_seq.shape[:2] and gt_seq.shape[0] >= 2:
        gt_valid = ~np.any(np.isnan(gt_seq), axis=2)
        if vis_seq is not None and vis_seq.shape[:2] == gt_valid.shape:
            gt_valid = gt_valid & (vis_seq > 0)

        v_pred = (pred_seq[1:] - pred_seq[:-1]) / dt
        v_gt = (gt_seq[1:] - gt_seq[:-1]) / dt
        v_err = np.linalg.norm(v_pred - v_gt, axis=2)
        v_valid = valid_mask[1:] & valid_mask[:-1] & gt_valid[1:] & gt_valid[:-1]
        metrics['velocity_error'] = _masked_mean(v_err, v_valid)

        if gt_seq.shape[0] >= 3:
            a_pred = (v_pred[1:] - v_pred[:-1]) / dt
            a_gt = (v_gt[1:] - v_gt[:-1]) / dt
            a_err = np.linalg.norm(a_pred - a_gt, axis=2)
            a_valid = v_valid[1:] & v_valid[:-1]
            metrics['acceleration_error'] = _masked_mean(a_err, a_valid)

    return metrics


def _masked_mean(values, mask):
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.size == 0 or mask.sum() == 0:
        return 0.0
    return float(values[mask].mean())


def plot_pck_per_frame(pck_vals, results_dir, label):
    """Plot PCK over frames."""
    results_dir = Path(results_dir)
    
    plt.figure(figsize=(10, 6))
    
    frames = range(len(pck_vals))
    
    plt.plot(frames, pck_vals, label=label, color='green', linewidth=2)
    
    plt.xlabel('Frame', fontsize=21)
    plt.ylabel('PCK@0.05', fontsize=21)
    plt.title(f'Per-Frame PCK@0.05 ({label})', fontsize=23)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()
    
    save_path = results_dir / 'PCK_per_frame.png'
    plt.savefig(save_path)
    plt.close()
    print(f'Saved: {save_path}')
def plot_pck_threshold_curve(pck_dict, thresholds, metrics, results_dir, label):
    results_dir = Path(results_dir)

    def _mean_curve(pck_map):
        return [np.mean(pck_map[thr]) if pck_map[thr] else 0.0 for thr in thresholds]

    x = thresholds
    y_vals = _mean_curve(pck_dict)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_vals, label=f'{label} (AUC={metrics.get("PCK_AUC", 0.0):.4f})', color='green', linewidth=2)
    plt.fill_between(x, 0, y_vals, color='green', alpha=0.15)

    plt.xlabel('PCK Threshold', fontsize=21)
    plt.ylabel('PCK', fontsize=21)
    plt.title(f'PCK vs Threshold ({label})', fontsize=23)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=14)
    plt.tight_layout()

    save_path = results_dir / 'PCK_threshold_curve.png'
    plt.savefig(save_path)
    plt.close()
    print(f'Saved: {save_path}')


def _evaluate_split(poses, gt_annotations, results_dir, fps, label='Initial', pred_bboxes=None):
    thresholds = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]

    # Joint-level aggregation (all 4 keypoints independently)
    pck = {thr: [] for thr in thresholds}
    pck_vis = {thr: [] for thr in thresholds}
    pck_occ = {thr: [] for thr in thresholds}
    # Frame-level aggregation kept only for plotting per-frame PCK
    pck_frame = {thr: [] for thr in thresholds}
    ref_sizes = []

    for frame_idx in range(len(poses)):
        if frame_idx >= len(gt_annotations):
            break

        try:
            gt_item = gt_annotations[frame_idx]
            gt_joints = np.array(gt_item.get('joints'))
            gt_vis = _normalize_visibility(gt_item.get('joints_vis', np.ones(gt_joints.shape[0])))
            gt_scale = float(gt_item.get('scale', 1.0))
            gt_bbox = gt_item.get('bbox', None)

            headsize = gt_scale * 200.0 * math.sqrt(2)
            bbox_size = _bbox_reference_size(gt_bbox)
            if bbox_size is None:
                pred_bbox = _pred_bbox_for_frame(pred_bboxes, frame_idx)
                bbox_size = _bbox_reference_size(pred_bbox)
            ref_size = bbox_size if bbox_size is not None else headsize
            ref_sizes.append(ref_size)

            for thr in thresholds:
                pck_val = compute_pck(
                    poses[frame_idx], gt_joints, gt_vis, ref_size,
                    threshold=thr, apply_visibility=False
                )
                # Total mean over all keypoints (no pre-averaging between eyes)
                pck[thr].extend(np.asarray(pck_val, dtype=float).tolist())

                vis_mask = gt_vis.astype(bool)
                occ_mask = ~vis_mask
                pck_vis[thr].extend(np.asarray(pck_val, dtype=float)[vis_mask].tolist())
                pck_occ[thr].extend(np.asarray(pck_val, dtype=float)[occ_mask].tolist())

                # Keep frame-wise mean only for temporal plotting
                pck_frame[thr].append(float(np.mean(pck_val)))

        except Exception as e:
            print(f"Error evaluating frame {frame_idx}: {e}")
            continue

    metrics = {}
    for thr in thresholds:
        metrics[f'PCK@{thr:.2f}_mean'] = np.mean(pck[thr]) if pck[thr] else 0.0
        metrics[f'PCK@{thr:.2f}_visible_mean'] = np.mean(pck_vis[thr]) if pck_vis[thr] else 0.0
        metrics[f'PCK@{thr:.2f}_occluded_mean'] = np.mean(pck_occ[thr]) if pck_occ[thr] else 0.0

    avg_ref_size = float(np.mean(ref_sizes)) if ref_sizes else 0.0
    metrics['PCK@0.05_threshold_px'] = avg_ref_size * 0.05

    metrics['PCK_AUC'] = _auc_from_thresholds(pck, thresholds)
    metrics['PCK_AUC_visible'] = _auc_from_thresholds(pck_vis, thresholds)
    metrics['PCK_AUC_occluded'] = _auc_from_thresholds(pck_occ, thresholds)

    dt = 1.0 / float(fps) if fps else 1.0
    gt_seq, gt_vis_seq = _collect_gt_sequences(gt_annotations, len(poses))
    temporal = _temporal_metrics(poses, gt_seq, dt, gt_vis_seq)
    metrics.update(temporal)

    if pck_frame[thresholds[0]]:
        plot_pck_per_frame(pck_frame[thresholds[0]], results_dir, label)

    plot_pck_threshold_curve(pck, thresholds, metrics, results_dir, label)
    save_metrics(metrics, results_dir, label)

    return metrics


def save_metrics(metrics, results_dir, label):
    """Save metrics to text file."""
    results_dir = Path(results_dir)

    save_path = results_dir / 'evaluation_metrics.txt'

    with open(save_path, 'w') as f:
        f.write(f"=== Evaluation Metrics ({label}) ===\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    print(f'Saved metrics to: {save_path}')


def calculate_endpoint_error(pred, gt):
    """
    Calculate average endpoint error.
    
    Args:
        pred: Predicted joints
        gt: Ground truth joints
    
    Returns:
        Mean euclidean distance
    """
    pred = np.array(pred)
    gt = np.array(gt)
    
    if np.any(np.isnan(pred)):
        return np.nan
    
    distances = np.linalg.norm(pred - gt, axis=1)
    return np.mean(distances)