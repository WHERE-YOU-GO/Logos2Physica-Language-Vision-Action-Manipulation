from __future__ import annotations

import numpy as np

from common.datatypes import BBox2D


def median_filter_depth(depth: np.ndarray, ksize: int = 5) -> np.ndarray:
    depth_np = np.asarray(depth, dtype=np.float32)
    if depth_np.ndim != 2:
        raise ValueError(f"depth must have shape (H, W), got {depth_np.shape}")
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("ksize must be a positive odd integer.")

    pad = ksize // 2
    padded = np.pad(depth_np, pad_width=pad, mode="edge")
    filtered = np.zeros_like(depth_np)
    for row in range(depth_np.shape[0]):
        for col in range(depth_np.shape[1]):
            patch = padded[row : row + ksize, col : col + ksize]
            valid = patch[np.isfinite(patch) & (patch > 0.0)]
            filtered[row, col] = float(np.median(valid)) if valid.size > 0 else 0.0
    return filtered


def remove_invalid_depth(depth: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    depth_np = np.asarray(depth, dtype=np.float32).copy()
    if depth_np.ndim != 2:
        raise ValueError(f"depth must have shape (H, W), got {depth_np.shape}")
    if min_m < 0.0 or max_m <= min_m:
        raise ValueError("Depth bounds must satisfy 0 <= min_m < max_m.")

    invalid_mask = ~np.isfinite(depth_np) | (depth_np < float(min_m)) | (depth_np > float(max_m))
    depth_np[invalid_mask] = 0.0
    return depth_np


def depth_patch_median(depth: np.ndarray, bbox: BBox2D, shrink_ratio: float = 0.3) -> float:
    depth_np = np.asarray(depth, dtype=np.float32)
    if depth_np.ndim != 2:
        raise ValueError(f"depth must have shape (H, W), got {depth_np.shape}")
    if not (0.0 <= shrink_ratio < 1.0):
        raise ValueError("shrink_ratio must be within [0.0, 1.0).")

    bbox_w = bbox.width()
    bbox_h = bbox.height()
    center_u, center_v = bbox.center_uv()
    crop_w = max(1, int(round(bbox_w * (1.0 - shrink_ratio))))
    crop_h = max(1, int(round(bbox_h * (1.0 - shrink_ratio))))

    x1 = max(0, center_u - crop_w // 2)
    y1 = max(0, center_v - crop_h // 2)
    x2 = min(depth_np.shape[1], x1 + crop_w)
    y2 = min(depth_np.shape[0], y1 + crop_h)
    patch = depth_np[y1:y2, x1:x2]
    if patch.size == 0:
        raise ValueError("Depth patch is empty.")

    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        raise ValueError("Depth patch contains no valid depth values.")
    return float(np.median(valid))
