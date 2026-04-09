from __future__ import annotations

import time
from typing import Any

import numpy as np

from common.datatypes import RGBDFrame
from common.logger import ProjectLogger
from sensing.apriltag_calib import AprilTagCalibrator
from sensing.zed_capture import ZEDCapture


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


class FrameProvider:
    def __init__(
        self,
        capture: ZEDCapture,
        calibrator: AprilTagCalibrator,
        logger: ProjectLogger,
        use_cached_extrinsics: bool = True,
    ) -> None:
        self._capture = capture
        self._calibrator = calibrator
        self._logger = logger
        self._use_cached_extrinsics = use_cached_extrinsics
        self._T_base_cam: np.ndarray | None = None

    def _resolve_extrinsics(self, rgb: Any, intrinsics) -> np.ndarray:
        if self._T_base_cam is not None:
            return self._T_base_cam

        if self._calibrator is None:
            raise RuntimeError("FrameProvider requires a calibrator to provide T_base_cam.")

        if self._use_cached_extrinsics:
            try:
                T_base_cam = self._calibrator.load_cached_T_base_cam()
            except Exception as exc:
                _log(self._logger, "warn", f"Failed to load cached extrinsics, refreshing: {exc}")
            else:
                self._T_base_cam = np.asarray(T_base_cam, dtype=np.float64)
                return self._T_base_cam

        self._T_base_cam = np.asarray(
            self._calibrator.estimate_T_base_cam(rgb, intrinsics),
            dtype=np.float64,
        )
        return self._T_base_cam

    def get_current_frame(self) -> RGBDFrame:
        rgb, depth = self._capture.grab_rgbd()
        intrinsics = self._capture.get_intrinsics()
        T_base_cam = self._resolve_extrinsics(rgb, intrinsics)
        return RGBDFrame(
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            T_base_cam=T_base_cam,
            timestamp=time.time(),
        )

    def refresh_extrinsics(self) -> None:
        if self._calibrator is None:
            raise RuntimeError("Cannot refresh extrinsics without a calibrator.")

        rgb = self._capture.grab_rgb()
        intrinsics = self._capture.get_intrinsics()
        T_base_cam = np.asarray(
            self._calibrator.estimate_T_base_cam(rgb, intrinsics),
            dtype=np.float64,
        )
        self._calibrator.save_cached_T_base_cam(T_base_cam)
        self._T_base_cam = T_base_cam
        _log(self._logger, "info", "Camera extrinsics refreshed.")
