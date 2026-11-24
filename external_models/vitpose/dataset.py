"""BirdGaze image dataset for the independent ViTPose-B training process.

Preserves original (not COCO-reexported) BirdGaze annotations and the
JointsDataset preprocessing contract. Training only reads images/annotations.
It intentionally does not import native Bird_HKE model/training modules.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from external_models.convert_birdgaze_to_coco import locate_image
from external_models.rtmpose.geometry import clip_original_bbox

IMAGE_SIZE = 256
HEATMAP_SIZE = 64
FLIP_PAIRS = ((1, 2),)
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def coordinate_valid(joints: np.ndarray, row: dict) -> np.ndarray:
    """Match BirdGaze coordinate_validity, including annotated occlusions."""
    xy = joints[:, :2]
    implicit = (np.isfinite(xy).all(axis=1)
                & (xy[:, 0] >= 0) & (xy[:, 1] >= 0)
                & np.any(xy != 0, axis=1))
    if row.get("joints_valid") is not None:
        return np.asarray(row["joints_valid"], dtype=np.float32).reshape(4) > 0
    if row.get("joints_vis") is None:
        return implicit
    return implicit | (np.asarray(row["joints_vis"], dtype=np.float32).reshape(4) > 0)


def get_dir(point, angle):
    sn, cs = np.sin(angle), np.cos(angle)
    return np.asarray([point[0] * cs - point[1] * sn,
                       point[0] * sn + point[1] * cs], dtype=np.float32)


def affine_matrix(center, scale, rotation, out_size, *, inverse=False):
    """Match Bird_HKE/lib/utilities/transforms.py get_affine_transform."""
    src_width = float(scale) * 200.
    dst = float(out_size)
    theta = np.deg2rad(rotation)
    c = np.asarray(center, dtype=np.float32)
    src_dir = get_dir([0, -src_width * 0.5], theta)
    dst_dir = np.asarray([0, -dst * 0.5], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    target = np.zeros((3, 2), dtype=np.float32)
    src[0] = c
    src[1] = c + src_dir
    src[2] = src[1] + np.asarray([-(src[0] - src[1])[1],
                                  (src[0] - src[1])[0]], dtype=np.float32)
    target[0] = [dst / 2, dst / 2]
    target[1] = target[0] + dst_dir
    target[2] = target[1] + np.asarray(
        [-(target[0] - target[1])[1], (target[0] - target[1])[0]],
        dtype=np.float32,
    )
    return cv2.getAffineTransform(target, src) if inverse else cv2.getAffineTransform(src, target)


def gaussian_targets(points, valid):
    maps = np.zeros((4, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    weights = np.asarray(valid, dtype=np.float32).copy()
    sigma = 2
    radius = 3 * sigma
    grid = np.arange(2 * radius + 1, dtype=np.float32) - radius
    gaussian = np.exp(-(grid[:, None] ** 2 + grid[None, :] ** 2) /
                      (2 * sigma ** 2)).astype(np.float32)
    for index in range(4):
        if not weights[index]:
            continue
        mu_x = int(points[index, 0] / 4. + 0.5)
        mu_y = int(points[index, 1] / 4. + 0.5)
        x0, x1 = mu_x - radius, mu_x + radius + 1
        y0, y1 = mu_y - radius, mu_y + radius + 1
        if x0 >= HEATMAP_SIZE or y0 >= HEATMAP_SIZE or x1 < 0 or y1 < 0:
            weights[index] = 0
            continue
        left, right = max(0, x0), min(HEATMAP_SIZE, x1)
        top, bottom = max(0, y0), min(HEATMAP_SIZE, y1)
        if right <= left or bottom <= top:
            weights[index] = 0
            continue
        maps[index, top:bottom, left:right] = gaussian[
            top - y0:bottom - y0, left - x0:right - x0]
    return maps, weights


def decode_image_coordinates(heatmaps: np.ndarray, center: np.ndarray,
                             scale: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Argmax + quarter-pixel correction + inverse heatmap affine.

    center/scale are from source-cropped input coordinates. Offset restores
    predictions to coordinates in the original full-resolution source image.
    """
    b, k, h, w = heatmaps.shape
    if (k, h, w) != (4, 64, 64):
        raise ValueError("Expected [N,4,64,64] heatmaps")
    xy = np.zeros((b, 4, 2), dtype=np.float32)
    flattened = heatmaps.reshape(b, 4, -1)
    flat = flattened.argmax(axis=-1)
    positive_peak = flattened.max(axis=-1) > 0
    xy[..., 0] = (flat % w) * positive_peak
    xy[..., 1] = (flat // w) * positive_peak
    # Same optional quarter-pixel refinement as Bird_HKE, enabled for ViTPose.
    for n in range(b):
        for j in range(k):
            px, py = [int(v) for v in xy[n, j]]
            if 1 < px < w - 1 and 1 < py < h - 1:
                hm = heatmaps[n, j]
                diff = np.array([hm[py, px + 1] - hm[py, px - 1],
                                 hm[py + 1, px] - hm[py - 1, px]], dtype=np.float32)
                xy[n, j] += np.sign(diff) * 0.25
        inv = affine_matrix(center[n], float(scale[n]), 0, 64, inverse=True)
        xy[n] = np.concatenate((xy[n], np.ones((k, 1), dtype=np.float32)), axis=1) @ inv.T
        xy[n] += offset[n]
    return xy


class BirdGazeViTPoseDataset(Dataset):
    def __init__(self, dataset_root: Path, split: str):
        if split not in ("train", "val"):
            raise ValueError("Only train/val annotations are allowed")
        self.root = Path(dataset_root).resolve()
        self.split = split
        with (self.root / "annot" / f"{split}.json").open(encoding="utf-8") as src:
            self.records = json.load(src)
        if not isinstance(self.records, list) or not self.records:
            raise ValueError(f"Empty/malformed {split} source annotations")
        self.images = self.root / "images"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        image_path = locate_image(self.images, str(row["image"]))[1]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise FileNotFoundError(f"Cannot decode BirdGaze image {image_path}")
        points = np.asarray(row["joints"], dtype=np.float32)[:, :2].copy()
        if points.shape != (4, 2):
            raise ValueError(f"Expected four BirdGaze keypoints: {image_path}")
        valid = coordinate_valid(points, row)
        if not np.isfinite(points[valid]).all():
            raise ValueError(f"Nonfinite labeled keypoint {image_path}")
        gt_points = points.copy()
        eval_valid = valid.copy()
        if row.get("joints_vis") is not None:
            eval_valid &= (np.asarray(row["joints_vis"], dtype=np.float32).reshape(4) > 0)
        raw_scale = float(row["scale"])
        center = np.asarray(row["center"], dtype=np.float32).copy()
        if not np.isfinite(center).all() or not math.isfinite(raw_scale):
            raise ValueError(f"Invalid crop metadata in {image_path}")
        scale = raw_scale * (1.25 if float(center[0]) != -1 else 1.0)

        offset = np.zeros(2, dtype=np.float32)
        source_crop = clip_original_bbox(row.get("bbox"), width=image.shape[1],
                                         height=image.shape[0])
        if source_crop:
            x0, y0, x1, y1 = source_crop
            image = image[y0:y1, x0:x1]
            offset = np.asarray([x0, y0], dtype=np.float32)
            points -= offset
            center -= offset

        rotation = 0.
        if self.split == "train":
            factor = float(np.clip(np.random.randn() * 0.25 + 1., 0.75, 1.25))
            scale *= factor
            if random.random() <= 0.6:
                rotation = float(np.clip(np.random.randn() * 30., -60., 60.))
            if random.random() <= 0.5:
                image = image[:, ::-1, :]
                points[:, 0] = image.shape[1] - points[:, 0] - 1
                points[[1, 2]] = points[[2, 1]]
                valid[[1, 2]] = valid[[2, 1]]
                center[0] = image.shape[1] - center[0] - 1

        affine = affine_matrix(center, scale, rotation, IMAGE_SIZE)
        image = cv2.warpAffine(image, affine, (IMAGE_SIZE, IMAGE_SIZE),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))
        xy_h = np.column_stack((points, np.ones((4,), dtype=np.float32)))
        transformed = (xy_h @ affine.T).astype(np.float32)
        target, weight = gaussian_targets(transformed, valid)
        chw = image.transpose(2, 0, 1).astype(np.float32) / 255.
        chw = (chw - MEAN) / STD
        # Pose crop parameters only used in validation. Train augmentation
        # updates these, but validation always uses original (nonaugmented) ones.
        return {
            "image": torch.from_numpy(chw.copy()),
            "target": torch.from_numpy(target),
            "weight": torch.from_numpy(weight),
            "gt": torch.from_numpy(gt_points),
            "eval_valid": torch.from_numpy(eval_valid),
            "center": torch.from_numpy(center.copy()),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "raw_scale": torch.tensor(raw_scale, dtype=torch.float32),
            "offset": torch.from_numpy(offset.copy()),
        }
