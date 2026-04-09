from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.config_loader import load_camera_config
from common.datatypes import CameraIntrinsics
from common.exceptions import ProjectError
from common.logger import ProjectLogger

try:
    import pyzed.sl as sl
except ImportError as exc:  # pragma: no cover
    sl = None
    _ZED_IMPORT_ERROR = exc
else:
    _ZED_IMPORT_ERROR = None


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


class ZEDCapture:
    def __init__(self, config_path: str, logger: ProjectLogger) -> None:
        self._config_path = str(Path(config_path))
        self._logger = logger
        self._camera: Any | None = None
        self._runtime_params: Any | None = None
        self._is_open = False
        self._config = self._load_config()
        self._intrinsics_cache: CameraIntrinsics | None = self._intrinsics_from_config()

    def _load_config(self) -> dict[str, Any]:
        try:
            return load_camera_config(self._config_path)
        except ProjectError as exc:
            _log(self._logger, "warn", f"Falling back to default camera config: {exc}")
            return {}

    def _intrinsics_from_config(self) -> CameraIntrinsics | None:
        intrinsics_cfg = self._config.get("intrinsics")
        if not isinstance(intrinsics_cfg, dict):
            return None
        try:
            return CameraIntrinsics(
                fx=float(intrinsics_cfg["fx"]),
                fy=float(intrinsics_cfg["fy"]),
                cx=float(intrinsics_cfg["cx"]),
                cy=float(intrinsics_cfg["cy"]),
                width=int(intrinsics_cfg["width"]),
                height=int(intrinsics_cfg["height"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def open(self) -> None:
        if self._is_open:
            return
        if sl is None:
            raise RuntimeError("ZED SDK is not installed. Install pyzed.sl to use ZEDCapture.") from _ZED_IMPORT_ERROR

        init_params = sl.InitParameters()
        resolution = str(self._config.get("resolution", "HD720")).upper()
        depth_mode = str(self._config.get("depth_mode", "QUALITY")).upper()
        fps = int(self._config.get("fps", 30))
        coordinate_units = str(self._config.get("coordinate_units", "METER")).upper()

        resolution_map = {
            "HD2K": sl.RESOLUTION.HD2K,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720,
            "VGA": sl.RESOLUTION.VGA,
        }
        depth_mode_map = {
            "NONE": sl.DEPTH_MODE.NONE,
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
            "QUALITY": sl.DEPTH_MODE.QUALITY,
            "ULTRA": sl.DEPTH_MODE.ULTRA,
            "NEURAL": sl.DEPTH_MODE.NEURAL,
        }
        units_map = {
            "METER": sl.UNIT.METER,
            "CENTIMETER": sl.UNIT.CENTIMETER,
            "MILLIMETER": sl.UNIT.MILLIMETER,
        }

        init_params.camera_resolution = resolution_map.get(resolution, sl.RESOLUTION.HD720)
        init_params.depth_mode = depth_mode_map.get(depth_mode, sl.DEPTH_MODE.QUALITY)
        init_params.camera_fps = fps
        init_params.coordinate_units = units_map.get(coordinate_units, sl.UNIT.METER)

        self._camera = sl.Camera()
        status = self._camera.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self._camera = None
            raise RuntimeError(f"Failed to open ZED camera: {status}")

        self._runtime_params = sl.RuntimeParameters()
        self._is_open = True
        self._intrinsics_cache = self._read_intrinsics_from_camera()
        _log(self._logger, "info", "ZED camera opened.")

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()
        self._camera = None
        self._runtime_params = None
        self._is_open = False
        _log(self._logger, "info", "ZED camera closed.")

    def _read_intrinsics_from_camera(self) -> CameraIntrinsics:
        if not self._is_open or self._camera is None:
            if self._intrinsics_cache is None:
                raise RuntimeError("Camera is not open and no intrinsics are cached.")
            return self._intrinsics_cache

        camera_info = self._camera.get_camera_information()
        calibration = camera_info.camera_configuration.calibration_parameters.left_cam
        resolution = camera_info.camera_configuration.resolution
        return CameraIntrinsics(
            fx=float(calibration.fx),
            fy=float(calibration.fy),
            cx=float(calibration.cx),
            cy=float(calibration.cy),
            width=int(resolution.width),
            height=int(resolution.height),
        )

    def get_intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics_cache is not None:
            return self._intrinsics_cache
        if not self._is_open:
            raise RuntimeError("Camera is not open and intrinsics are unavailable.")
        self._intrinsics_cache = self._read_intrinsics_from_camera()
        return self._intrinsics_cache

    def _ensure_open(self) -> None:
        if not self._is_open or self._camera is None or self._runtime_params is None:
            raise RuntimeError("ZED camera is not open.")

    def grab_rgb(self) -> Any:
        self._ensure_open()
        assert sl is not None

        if self._camera.grab(self._runtime_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to grab RGB frame from ZED camera.")

        rgb_mat = sl.Mat()
        self._camera.retrieve_image(rgb_mat, sl.VIEW.LEFT)
        rgb = rgb_mat.get_data()
        if rgb is None:
            raise RuntimeError("ZED camera returned an empty RGB frame.")
        rgb_np = np.asarray(rgb)
        if rgb_np.ndim == 3 and rgb_np.shape[-1] == 4:
            rgb_np = rgb_np[..., :3]
        return rgb_np

    def grab_depth(self) -> Any:
        self._ensure_open()
        assert sl is not None

        if self._camera.grab(self._runtime_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to grab depth frame from ZED camera.")

        depth_mat = sl.Mat()
        self._camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        depth = depth_mat.get_data()
        if depth is None:
            raise RuntimeError("ZED camera returned an empty depth frame.")
        return np.asarray(depth, dtype=np.float32)

    def grab_rgbd(self) -> tuple[Any, Any]:
        self._ensure_open()
        assert sl is not None

        if self._camera.grab(self._runtime_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to grab RGB-D frame from ZED camera.")

        rgb_mat = sl.Mat()
        depth_mat = sl.Mat()
        self._camera.retrieve_image(rgb_mat, sl.VIEW.LEFT)
        self._camera.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

        rgb = rgb_mat.get_data()
        depth = depth_mat.get_data()
        if rgb is None or depth is None:
            raise RuntimeError("ZED camera returned an incomplete RGB-D frame.")

        rgb_np = np.asarray(rgb)
        if rgb_np.ndim == 3 and rgb_np.shape[-1] == 4:
            rgb_np = rgb_np[..., :3]
        depth_np = np.asarray(depth, dtype=np.float32)
        return rgb_np, depth_np
