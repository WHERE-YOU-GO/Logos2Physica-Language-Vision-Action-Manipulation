from __future__ import annotations

import colorsys
from dataclasses import replace
from typing import Any

import numpy as np

from common.datatypes import BBox2D, Detection2D


def _extract_rgb_patch(rgb: Any, bbox: BBox2D, shrink_ratio: float = 0.3) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError("rgb must have shape (H, W, 3) or (H, W, 4).")
    image = image[..., :3]

    height, width = image.shape[:2]
    crop_w = max(1, int(round(bbox.width() * (1.0 - shrink_ratio))))
    crop_h = max(1, int(round(bbox.height() * (1.0 - shrink_ratio))))
    center_u, center_v = bbox.center_uv()

    x1 = max(0, center_u - crop_w // 2)
    y1 = max(0, center_v - crop_h // 2)
    x2 = min(width, x1 + crop_w)
    y2 = min(height, y1 + crop_h)
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        raise ValueError("RGB patch is empty.")
    return patch.astype(np.float32)


def classify_color_hsv(rgb: Any, bbox: BBox2D) -> str | None:
    try:
        patch = _extract_rgb_patch(rgb, bbox, shrink_ratio=0.4)
    except ValueError:
        return None

    median_rgb = np.median(patch.reshape(-1, 3), axis=0) / 255.0
    hue, saturation, value = colorsys.rgb_to_hsv(
        float(median_rgb[0]),
        float(median_rgb[1]),
        float(median_rgb[2]),
    )
    hue_deg = hue * 360.0

    if value < 0.15:
        return "black"
    if saturation < 0.2 and value > 0.75:
        return "white"
    if saturation < 0.15:
        return "gray"
    if hue_deg >= 345.0 or hue_deg < 20.0:
        return "red"
    if 20.0 <= hue_deg < 50.0:
        return "yellow"
    if 50.0 <= hue_deg < 90.0:
        return "green"
    if 90.0 <= hue_deg < 170.0:
        return "green"
    if 170.0 <= hue_deg < 260.0:
        return "blue"
    if 260.0 <= hue_deg < 345.0:
        return "purple"
    return None


def classify_shape_simple(rgb: Any, bbox: BBox2D) -> str | None:
    _ = rgb
    aspect_ratio = bbox.width() / max(1.0, float(bbox.height()))
    if 0.8 <= aspect_ratio <= 1.25:
        return "cube"
    if 0.5 <= aspect_ratio <= 1.8:
        return "block"
    return None


def refine_detection_attributes(rgb: Any, det: Detection2D) -> Detection2D:
    color = classify_color_hsv(rgb, det.bbox)
    shape = classify_shape_simple(rgb, det.bbox)
    extras = dict(det.extras)
    if color is not None:
        extras["color"] = color
    if shape is not None:
        extras["shape"] = shape
    extras.setdefault("category", det.label)
    return replace(det, extras=extras)


def matches_object_query(
    det: Detection2D,
    color: str | None,
    shape: str | None,
    category: str | None,
) -> bool:
    extras = det.extras
    if color is not None and extras.get("color") != color:
        return False
    if shape is not None and extras.get("shape") != shape:
        return False
    if category is None:
        return True

    category_norm = category.strip().lower()
    label_norm = det.label.strip().lower()
    det_category = str(extras.get("category", det.label)).strip().lower()
    return category_norm in {label_norm, det_category}
