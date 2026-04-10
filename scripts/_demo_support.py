from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from common.datatypes import BBox2D, CameraIntrinsics, Detection2D, RGBDFrame
from common.logger import ProjectLogger


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


def default_demo_meta() -> dict[str, Any]:
    return {
        "prompt": "put the red cube on the blue block",
        "expected_source_color": "red",
        "expected_source_label": "cube",
        "expected_target_color": "blue",
        "expected_target_label": "block",
    }


def build_synthetic_frame() -> RGBDFrame:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[200:260, 280:340] = np.array([255, 0, 0], dtype=np.uint8)
    rgb[220:280, 360:450] = np.array([0, 0, 255], dtype=np.uint8)

    depth = np.ones((480, 640), dtype=np.float32) * 0.55
    intrinsics = CameraIntrinsics(
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        width=640,
        height=480,
    )
    T_base_cam = np.eye(4, dtype=np.float64)
    T_base_cam[:3, 3] = np.array([0.35, 0.0, -0.51], dtype=np.float64)
    return RGBDFrame(
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        T_base_cam=T_base_cam,
        timestamp=0.0,
    )


class SyntheticFrameProvider:
    def __init__(self, meta: dict[str, Any] | None = None) -> None:
        self._frame = build_synthetic_frame()
        self._meta = dict(meta or default_demo_meta())

    def get_frame(self) -> RGBDFrame:
        return self._frame

    def get_current_frame(self) -> RGBDFrame:
        return self.get_frame()

    def get_meta(self) -> dict[str, Any]:
        return dict(self._meta)


@dataclass(frozen=True)
class DemoObjectSpec:
    role: str
    color: str
    label: str


class ColorBlockDemoDetector:
    def __init__(self, scene_meta: dict[str, Any] | None = None, logger: ProjectLogger | None = None) -> None:
        self._scene_meta = dict(scene_meta or default_demo_meta())
        self._logger = logger

    def warmup(self) -> None:
        _log(self._logger, "info", "ColorBlockDemoDetector is ready.")

    def _scene_specs(self) -> list[DemoObjectSpec]:
        specs: list[DemoObjectSpec] = []
        source_color = str(self._scene_meta.get("expected_source_color", "red")).strip().lower()
        source_label = str(self._scene_meta.get("expected_source_label", "cube")).strip().lower()
        target_color = str(self._scene_meta.get("expected_target_color", "blue")).strip().lower()
        target_label = str(self._scene_meta.get("expected_target_label", "block")).strip().lower()
        specs.append(DemoObjectSpec(role="source", color=source_color, label=source_label))
        specs.append(DemoObjectSpec(role="target", color=target_color, label=target_label))
        return specs

    def _color_mask(self, rgb: np.ndarray, color: str) -> np.ndarray:
        image = np.asarray(rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"Expected an RGB image with shape (H, W, 3/4), got {image.shape}.")
        image = image[..., :3].astype(np.int16)
        red = image[..., 0]
        green = image[..., 1]
        blue = image[..., 2]

        color = color.lower()
        if color == "red":
            return (red > 120) & (red > green + 40) & (red > blue + 40)
        if color == "blue":
            return (blue > 120) & (blue > red + 40) & (blue > green + 20)
        if color == "green":
            return (green > 120) & (green > red + 30) & (green > blue + 30)
        if color == "yellow":
            return (red > 120) & (green > 120) & (blue < 100)
        if color == "orange":
            return (red > 150) & (green > 80) & (green < red) & (blue < 100)
        if color == "purple":
            return (red > 110) & (blue > 110) & (green < 100)
        if color == "white":
            return (red > 180) & (green > 180) & (blue > 180)
        if color == "black":
            return (red < 40) & (green < 40) & (blue < 40)
        if color == "gray":
            return (
                np.abs(red - green) < 20
            ) & (np.abs(red - blue) < 20) & (np.abs(green - blue) < 20) & (red > 60) & (red < 180)
        return np.zeros(image.shape[:2], dtype=bool)

    def _mask_to_bbox(self, mask: np.ndarray) -> BBox2D | None:
        ys, xs = np.where(mask)
        if xs.size < 25 or ys.size < 25:
            return None
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1
        if x2 <= x1 or y2 <= y1:
            return None
        return BBox2D(x1=x1, y1=y1, x2=x2, y2=y2)

    def _matches_phrase(self, spec: DemoObjectSpec, phrase: str) -> bool:
        normalized_phrase = phrase.strip().lower()
        if not normalized_phrase:
            return False
        return spec.color in normalized_phrase or spec.label in normalized_phrase

    def _matches_labels(self, spec: DemoObjectSpec, candidate_labels: list[str]) -> bool:
        normalized_labels = {label.strip().lower() for label in candidate_labels if label.strip()}
        if not normalized_labels:
            return True
        return spec.label in normalized_labels

    def _build_detection(self, rgb: Any, spec: DemoObjectSpec, phrase: str | None = None) -> Detection2D | None:
        mask = self._color_mask(np.asarray(rgb), spec.color)
        bbox = self._mask_to_bbox(mask)
        if bbox is None:
            _log(
                self._logger,
                "warn",
                f"Could not find a detectable {spec.color} region for demo object role '{spec.role}'.",
            )
            return None
        return Detection2D(
            label=spec.label,
            score=0.99,
            bbox=bbox,
            phrase=phrase,
            extras={
                "backend": "demo_color",
                "color": spec.color,
                "category": spec.label,
                "role": spec.role,
            },
        )

    def detect(self, rgb: Any, candidate_labels: list[str]) -> list[Detection2D]:
        detections: list[Detection2D] = []
        for spec in self._scene_specs():
            if not self._matches_labels(spec, candidate_labels):
                continue
            detection = self._build_detection(rgb, spec)
            if detection is not None:
                detections.append(detection)
        return detections

    def detect_phrase(self, rgb: Any, phrase: str) -> list[Detection2D]:
        normalized_phrase = phrase.strip()
        if not normalized_phrase:
            raise ValueError("phrase must not be empty.")

        detections: list[Detection2D] = []
        for spec in self._scene_specs():
            if not self._matches_phrase(spec, normalized_phrase):
                continue
            detection = self._build_detection(rgb, spec, phrase=normalized_phrase)
            if detection is not None:
                detections.append(detection)

        if detections:
            return detections
        return [
            detection
            for detection in (self._build_detection(rgb, spec, phrase=normalized_phrase) for spec in self._scene_specs())
            if detection is not None
        ]


__all__ = [
    "ColorBlockDemoDetector",
    "DemoObjectSpec",
    "SyntheticFrameProvider",
    "build_synthetic_frame",
    "default_demo_meta",
]
