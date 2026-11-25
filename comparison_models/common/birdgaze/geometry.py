"""Framework-neutral BirdGaze geometry helpers.

Match JointsDataset.__getitem__: crop the RGB/BGR image using the *source*
annotation's [x1,y1,x2,y2] bbox, clipped to image dimensions using
floor/ceil; adjust center, crop bbox and keypoints by the same offset.
This is distinct from the MMPose *pose* bbox used by GetBBoxCenterScale.
"""
from __future__ import annotations

import numpy as np


def clip_original_bbox(original_bbox, width: int, height: int):
    """Return integer (x1,y1,x2,y2) or None if Bird_HKE would not crop."""
    if original_bbox is None:
        return None
    values = np.asarray(original_bbox, dtype=np.float32).reshape(-1)
    if values.size != 4:
        return None
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite source bbox")
    x1, y1, x2, y2 = values
    left = max(0, int(np.floor(x1)))
    top = max(0, int(np.floor(y1)))
    right = min(width, int(np.ceil(x2)))
    bottom = min(height, int(np.ceil(y2)))
    return (left, top, right, bottom) if right > left and bottom > top else None


def apply_source_crop(results: dict) -> dict:
    """Modify MMPose's loaded sample as the original Bird_HKE loader does."""
    bbox = results.get("birdgaze_source_bbox")
    if bbox is None:
        return results
    img = results["img"]
    if isinstance(img, list):
        raise ValueError("BirdGaze supports one image per sample")
    h, w = img.shape[:2]
    clip = clip_original_bbox(bbox, width=w, height=h)
    if clip is None:
        return results
    left, top, right, bottom = clip
    results["img"] = np.ascontiguousarray(img[top:bottom, left:right])
    results["img_shape"] = results["img"].shape[:2]
    offset = np.array([left, top], dtype=np.float32)
    if results.get("keypoints") is not None:
        results["keypoints"][..., :2] -= offset
    if results.get("bbox") is not None:
        results["bbox"][..., :2] -= offset
        results["bbox"][..., 2:4] -= offset
    # This offset is needed if final predictions are converted back into
    # full-image coordinates. It must not be lost in export/evaluation.
    results["birdgaze_crop_offset"] = offset
    return results
