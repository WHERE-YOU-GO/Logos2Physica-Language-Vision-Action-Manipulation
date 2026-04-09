"""Object detection backends.

Plan A: zero-shot OWL-ViT (HuggingFace transformers) — accepts text prompts.
Plan B: HSV color thresholding fallback — no model dependency, works for known
        colored cubes when the GPU/network is unavailable.

Both backends return the same structure:
    {"bbox": (x1, y1, x2, y2),
     "score": float,
     "uv": (u, v) center,
     "xyz_cam": np.ndarray (3,) in meters,
     "xyz_base": np.ndarray (3,) in meters}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Detection:
    label: str
    score: float
    bbox: tuple[int, int, int, int]
    uv: tuple[int, int]
    xyz_cam: np.ndarray
    xyz_base: np.ndarray


def _bgra_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[-1] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    if image.ndim == 3 and image.shape[-1] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _patch_xyz_median(point_cloud: np.ndarray, u: int, v: int, half: int = 4) -> Optional[np.ndarray]:
    """Median XYZ in a (2*half+1)^2 patch around (u, v), ignoring NaN/inf."""
    h, w = point_cloud.shape[:2]
    u0, u1 = max(0, u - half), min(w, u + half + 1)
    v0, v1 = max(0, v - half), min(h, v + half + 1)
    patch = point_cloud[v0:v1, u0:u1, :3].reshape(-1, 3)
    finite = patch[np.all(np.isfinite(patch), axis=1)]
    if finite.size == 0:
        return None
    return np.median(finite, axis=0).astype(np.float64)


def _project_to_base(uv: tuple[int, int], point_cloud: np.ndarray, T_cam_robot: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    xyz_cam = _patch_xyz_median(point_cloud, uv[0], uv[1])
    if xyz_cam is None:
        return None
    # T_cam_robot maps base->cam, so we need its inverse to go cam->base.
    T_robot_cam = np.linalg.inv(T_cam_robot)
    xyz_base = (T_robot_cam @ np.append(xyz_cam, 1.0))[:3]
    return xyz_cam, xyz_base


# ----- Plan A: OWL-ViT zero-shot detector ------------------------------------
class OWLViTDetector:
    def __init__(self, model_id: str = "google/owlv2-base-patch16-ensemble", threshold: float = 0.1) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor  # lazy import
        import torch  # lazy import

        self._torch = torch
        self._processor = Owlv2Processor.from_pretrained(model_id)
        self._model = Owlv2ForObjectDetection.from_pretrained(model_id)
        self._model.eval()
        self.threshold = threshold

    def locate(
        self,
        image_bgra: np.ndarray,
        point_cloud: np.ndarray,
        prompt: str,
        T_cam_robot: np.ndarray,
    ) -> Optional[Detection]:
        rgb = _bgra_to_rgb(image_bgra)
        inputs = self._processor(text=[[prompt]], images=rgb, return_tensors="pt")
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = self._torch.tensor([rgb.shape[:2]])
        results = self._processor.post_process_object_detection(
            outputs=outputs, threshold=self.threshold, target_sizes=target_sizes
        )[0]

        if len(results["scores"]) == 0:
            return None

        idx = int(self._torch.argmax(results["scores"]).item())
        x1, y1, x2, y2 = (int(v) for v in results["boxes"][idx].tolist())
        score = float(results["scores"][idx].item())
        u, v = (x1 + x2) // 2, (y1 + y2) // 2

        projected = _project_to_base((u, v), point_cloud, T_cam_robot)
        if projected is None:
            return None
        xyz_cam, xyz_base = projected
        return Detection(prompt, score, (x1, y1, x2, y2), (u, v), xyz_cam, xyz_base)


# ----- Plan B: HSV color fallback --------------------------------------------
HSV_RANGES = {
    "red":    [(np.array([0, 120, 70]), np.array([10, 255, 255])),
               (np.array([170, 120, 70]), np.array([180, 255, 255]))],
    "green":  [(np.array([40, 80, 40]),  np.array([85, 255, 255]))],
    "blue":   [(np.array([95, 120, 60]), np.array([130, 255, 255]))],
    "yellow": [(np.array([20, 120, 100]),(np.array([35, 255, 255])))],
}


class HSVDetector:
    """Find a colored cube by HSV thresholding + connected components."""

    def __init__(self, min_area_px: int = 200) -> None:
        self.min_area_px = min_area_px

    @staticmethod
    def _color_from_prompt(prompt: str) -> Optional[str]:
        for color in HSV_RANGES:
            if color in prompt.lower():
                return color
        return None

    def locate(
        self,
        image_bgra: np.ndarray,
        point_cloud: np.ndarray,
        prompt: str,
        T_cam_robot: np.ndarray,
    ) -> Optional[Detection]:
        color = self._color_from_prompt(prompt)
        if color is None:
            return None

        bgr = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR) if image_bgra.shape[-1] == 4 else image_bgra
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in HSV_RANGES[color]:
            mask |= cv2.inRange(hsv, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels <= 1:
            return None

        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = int(np.argmax(areas)) + 1
        if stats[idx, cv2.CC_STAT_AREA] < self.min_area_px:
            return None

        x1 = int(stats[idx, cv2.CC_STAT_LEFT])
        y1 = int(stats[idx, cv2.CC_STAT_TOP])
        x2 = x1 + int(stats[idx, cv2.CC_STAT_WIDTH])
        y2 = y1 + int(stats[idx, cv2.CC_STAT_HEIGHT])
        u, v = (int(centroids[idx, 0]), int(centroids[idx, 1]))

        projected = _project_to_base((u, v), point_cloud, T_cam_robot)
        if projected is None:
            return None
        xyz_cam, xyz_base = projected
        score = float(stats[idx, cv2.CC_STAT_AREA]) / (mask.shape[0] * mask.shape[1])
        return Detection(prompt, score, (x1, y1, x2, y2), (u, v), xyz_cam, xyz_base)
