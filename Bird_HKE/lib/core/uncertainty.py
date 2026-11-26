"""Calibration and diagnostic utilities for keypoint probability maps."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def probability_map_moments(probability_maps):
    """Return mean, covariance, entropy, and 90% HPD area per keypoint."""
    probabilities = np.asarray(probability_maps, dtype=np.float64)
    if probabilities.ndim != 4:
        raise ValueError('probability maps must have shape [N, K, H, W]')
    normalizer = probabilities.sum(axis=(2, 3), keepdims=True)
    probabilities = probabilities / np.maximum(normalizer, 1e-12)
    _, _, height, width = probabilities.shape
    yy, xx = np.mgrid[:height, :width]

    mean_x = np.sum(probabilities * xx, axis=(2, 3))
    mean_y = np.sum(probabilities * yy, axis=(2, 3))
    dx = xx[None, None] - mean_x[..., None, None]
    dy = yy[None, None] - mean_y[..., None, None]
    covariance = np.empty(probabilities.shape[:2] + (2, 2), dtype=np.float64)
    covariance[..., 0, 0] = np.sum(probabilities * dx * dx, axis=(2, 3))
    covariance[..., 1, 1] = np.sum(probabilities * dy * dy, axis=(2, 3))
    covariance[..., 0, 1] = np.sum(probabilities * dx * dy, axis=(2, 3))
    covariance[..., 1, 0] = covariance[..., 0, 1]

    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=(2, 3)
    )
    hpd_area = np.zeros(probabilities.shape[:2], dtype=np.int64)
    flat = probabilities.reshape(probabilities.shape[0], probabilities.shape[1], -1)
    for sample in range(flat.shape[0]):
        for joint in range(flat.shape[1]):
            ordered = np.sort(flat[sample, joint])[::-1]
            hpd_area[sample, joint] = int(
                np.searchsorted(np.cumsum(ordered), 0.90, side='left') + 1
            )
    return {
        'mean': np.stack((mean_x, mean_y), axis=-1),
        'covariance': covariance,
        'entropy': entropy,
        'hpd90_area': hpd_area,
    }


def ensemble_uncertainty(member_probability_maps):
    """Decompose ensemble covariance into aleatoric and epistemic parts."""
    members = np.asarray(member_probability_maps, dtype=np.float64)
    if members.ndim != 5:
        raise ValueError('ensemble maps must have shape [M, N, K, H, W]')
    member_stats = [probability_map_moments(member) for member in members]
    member_means = np.stack([stats['mean'] for stats in member_stats], axis=0)
    member_covariances = np.stack(
        [stats['covariance'] for stats in member_stats], axis=0
    )
    mean_map = members.mean(axis=0)
    mean_map /= np.maximum(mean_map.sum(axis=(2, 3), keepdims=True), 1e-12)
    mean = member_means.mean(axis=0)
    centered = member_means - mean[None]
    epistemic = np.einsum('mnki,mnkj->nkij', centered, centered) / members.shape[0]
    aleatoric = member_covariances.mean(axis=0)
    return {
        'probability_maps': mean_map,
        'mean': mean,
        'aleatoric_covariance': aleatoric,
        'epistemic_covariance': epistemic,
        'total_covariance': aleatoric + epistemic,
    }


def required_hpd_mass(probability_map, coordinate):
    """Probability mass an HPD region needs in order to contain the target."""
    probability = np.asarray(probability_map, dtype=np.float64)
    probability = probability / max(float(probability.sum()), 1e-12)
    x = int(np.clip(round(float(coordinate[0])), 0, probability.shape[1] - 1))
    y = int(np.clip(round(float(coordinate[1])), 0, probability.shape[0] - 1))
    target_density = probability[y, x]
    return float(probability[probability >= target_density].sum())


def finite_sample_quantile(values, coverage):
    """Split-conformal quantile using the finite-sample ceiling correction."""
    scores = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if scores.size == 0:
        return float('nan')
    if not 0 < coverage < 1:
        raise ValueError('coverage must be between zero and one')
    rank = min(scores.size, math.ceil((scores.size + 1) * coverage))
    return float(scores[rank - 1])


def fit_hpd_mass_thresholds(probability_maps, coordinates, valid, coverage=0.90):
    """Fit one conformal HPD mass threshold for each landmark."""
    probability_maps = np.asarray(probability_maps)
    coordinates = np.asarray(coordinates)
    valid = np.asarray(valid, dtype=bool)
    thresholds = []
    counts = []
    for joint in range(probability_maps.shape[1]):
        scores = [
            required_hpd_mass(probability_maps[index, joint], coordinates[index, joint])
            for index in range(probability_maps.shape[0])
            if valid[index, joint]
        ]
        thresholds.append(finite_sample_quantile(scores, coverage))
        counts.append(len(scores))
    return np.asarray(thresholds), np.asarray(counts)


def hpd_region(probability_map, mass):
    """Return the smallest highest-density pixel set reaching ``mass``."""
    probability = np.asarray(probability_map, dtype=np.float64)
    probability = probability / max(float(probability.sum()), 1e-12)
    ordered = np.sort(probability.reshape(-1))[::-1]
    index = min(
        ordered.size - 1,
        int(np.searchsorted(np.cumsum(ordered), float(mass), side='left')),
    )
    return probability >= ordered[index]


def calibration_error(confidence, targets, valid, bins=10):
    """Expected calibration error and Brier score for soft or binary targets."""
    confidence = np.asarray(confidence, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    confidence = confidence[valid]
    targets = targets[valid]
    if confidence.size == 0:
        return {'ece': float('nan'), 'brier': float('nan'), 'count': 0}
    boundaries = np.linspace(0, 1, int(bins) + 1)
    ece = 0.0
    for index in range(len(boundaries) - 1):
        lower, upper = boundaries[index], boundaries[index + 1]
        member = (confidence >= lower) & (
            confidence <= upper if index == len(boundaries) - 2 else confidence < upper
        )
        if np.any(member):
            ece += member.mean() * abs(confidence[member].mean() - targets[member].mean())
    return {
        'ece': float(ece),
        'brier': float(np.mean((confidence - targets) ** 2)),
        'count': int(confidence.size),
    }


def fit_binary_temperature(logits, targets, valid, max_iter=50):
    """Fit one positive temperature by masked binary cross entropy."""
    logits = torch.as_tensor(logits, dtype=torch.float64)
    targets = torch.as_tensor(targets, dtype=torch.float64)
    valid = torch.as_tensor(valid, dtype=torch.bool)
    if not bool(valid.any()):
        return 1.0
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.25, max_iter=int(max_iter), line_search_fn='strong_wolfe'
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(
            logits[valid] / temperature, targets[valid]
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def fit_spatial_temperature(logits, coordinates, valid, max_iter=50):
    """Fit a scalar temperature from annotated heatmap-cell likelihoods."""
    logits = torch.as_tensor(logits, dtype=torch.float64)
    coordinates = torch.as_tensor(coordinates, dtype=torch.long)
    valid = torch.as_tensor(valid, dtype=torch.bool)
    if not bool(valid.any()):
        return 1.0
    _, _, height, width = logits.shape
    target = coordinates[..., 1].clamp(0, height - 1) * width + coordinates[..., 0].clamp(0, width - 1)
    flat_logits = logits.flatten(start_dim=2)[valid]
    flat_target = target[valid]
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.25, max_iter=int(max_iter), line_search_fn='strong_wolfe'
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(flat_logits / temperature, flat_target)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))
