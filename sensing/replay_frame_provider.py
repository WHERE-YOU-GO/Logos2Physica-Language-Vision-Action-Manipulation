from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from common.config_loader import load_camera_config
from common.datatypes import CameraIntrinsics, RGBDFrame
from common.exceptions import ProjectError
from common.logger import ProjectLogger
from common.path_manager import resolve_path


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


class ReplayFrameProvider:
    def __init__(
        self,
        scene_dir: str | Path,
        camera_config_path: str = "config/camera.yaml",
        logger: ProjectLogger | None = None,
    ) -> None:
        self._scene_dir = resolve_path(scene_dir)
        self._camera_config_path = str(resolve_path(camera_config_path))
        self._logger = logger

        self._rgb_path = self._scene_dir / "rgb.png"
        self._depth_path = self._scene_dir / "depth.npy"
        self._meta_path = self._scene_dir / "meta.json"

        self._validate_required_files()
        self._meta = self._load_meta()
        self._camera_config = self._load_camera_config()
        self._frame_cache: RGBDFrame | None = None

    def _validate_required_files(self) -> None:
        if not self._scene_dir.exists():
            raise FileNotFoundError(f"Replay scene directory does not exist: {self._scene_dir}")
        if not self._scene_dir.is_dir():
            raise FileNotFoundError(f"Replay scene path is not a directory: {self._scene_dir}")
        for path_obj in (self._rgb_path, self._depth_path, self._meta_path):
            if not path_obj.exists():
                raise FileNotFoundError(f"Replay scene is missing required file: {path_obj}")
            if not path_obj.is_file():
                raise FileNotFoundError(f"Replay scene entry is not a file: {path_obj}")

    def _load_camera_config(self) -> dict[str, Any]:
        try:
            return load_camera_config(self._camera_config_path)
        except ProjectError as exc:
            _log(self._logger, "warn", f"Falling back to replay metadata only for camera config: {exc}")
            return {}

    def _load_meta(self) -> dict[str, Any]:
        try:
            with self._meta_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except OSError as exc:
            raise RuntimeError(f"Failed to read replay scene metadata: {self._meta_path}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in replay scene metadata: {self._meta_path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Replay scene metadata must be a JSON object, got {type(payload).__name__} in {self._meta_path}."
            )
        return dict(payload)

    def _load_rgb(self) -> np.ndarray:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Pillow is required to load replay scene rgb.png files.") from exc

        with Image.open(self._rgb_path) as handle:
            rgb = np.asarray(handle.convert("RGB"))
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(f"Replay RGB image must have shape (H, W, 3), got {rgb.shape}.")
        return rgb

    def _load_depth(self) -> np.ndarray:
        try:
            depth = np.load(self._depth_path, allow_pickle=False)
        except OSError as exc:
            raise RuntimeError(f"Failed to load replay depth file: {self._depth_path}") from exc
        depth_array = np.asarray(depth, dtype=np.float32)
        if depth_array.ndim != 2:
            raise RuntimeError(f"Replay depth array must have shape (H, W), got {depth_array.shape}.")
        return depth_array

    def _resolve_intrinsics(self, rgb: np.ndarray) -> CameraIntrinsics:
        intrinsics_payload = self._meta.get("intrinsics", self._camera_config.get("intrinsics"))
        if not isinstance(intrinsics_payload, dict):
            raise RuntimeError(
                "Replay scene intrinsics are missing. Add an 'intrinsics' object to meta.json "
                "or provide config/camera.yaml with an intrinsics section."
            )
        try:
            intrinsics = CameraIntrinsics(
                fx=float(intrinsics_payload["fx"]),
                fy=float(intrinsics_payload["fy"]),
                cx=float(intrinsics_payload["cx"]),
                cy=float(intrinsics_payload["cy"]),
                width=int(intrinsics_payload["width"]),
                height=int(intrinsics_payload["height"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Replay scene intrinsics payload is incomplete or invalid.") from exc

        if (intrinsics.height, intrinsics.width) != tuple(rgb.shape[:2]):
            raise RuntimeError(
                f"Replay scene intrinsics size {(intrinsics.width, intrinsics.height)} does not match "
                f"rgb.png size {(rgb.shape[1], rgb.shape[0])}. "
                "Add matching intrinsics to meta.json or update config/camera.yaml."
            )
        return intrinsics

    def _resolve_T_base_cam(self) -> np.ndarray:
        transform_payload = self._meta.get("T_base_cam", self._camera_config.get("T_base_cam"))
        if transform_payload is None:
            raise RuntimeError(
                "Replay scene extrinsics are missing. Add 'T_base_cam' to meta.json or config/camera.yaml."
            )
        transform = np.asarray(transform_payload, dtype=np.float64)
        if transform.shape != (4, 4):
            raise RuntimeError(f"T_base_cam must have shape (4, 4), got {transform.shape}.")
        if not np.all(np.isfinite(transform)):
            raise RuntimeError("T_base_cam must contain only finite numeric values.")
        return transform

    def get_frame(self) -> RGBDFrame:
        if self._frame_cache is not None:
            return self._frame_cache

        rgb = self._load_rgb()
        depth = self._load_depth()
        if tuple(rgb.shape[:2]) != tuple(depth.shape[:2]):
            raise RuntimeError(
                f"Replay scene rgb/depth mismatch: rgb shape {rgb.shape[:2]}, depth shape {depth.shape[:2]}."
            )

        intrinsics = self._resolve_intrinsics(rgb)
        T_base_cam = self._resolve_T_base_cam()
        timestamp = float(self._meta.get("timestamp", 0.0))

        self._frame_cache = RGBDFrame(
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            T_base_cam=T_base_cam,
            timestamp=timestamp,
        )
        return self._frame_cache

    def get_current_frame(self) -> RGBDFrame:
        return self.get_frame()

    def get_meta(self) -> dict[str, Any]:
        return dict(self._meta)


__all__ = ["ReplayFrameProvider"]
