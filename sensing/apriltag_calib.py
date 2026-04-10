from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.config_loader import load_camera_config
from common.datatypes import CameraIntrinsics
from common.exceptions import ProjectError
from common.logger import ProjectLogger
from common.path_manager import resolve_path
from common.transforms import invert_transform, make_transform

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import apriltag
except ImportError:  # pragma: no cover
    apriltag = None  # type: ignore[assignment]

try:
    from pupil_apriltags import Detector as PupilAprilTagDetector
except ImportError:  # pragma: no cover
    PupilAprilTagDetector = None  # type: ignore[assignment]


class AprilTagCalibrator:
    def __init__(self, config_path: str, logger: ProjectLogger) -> None:
        self._config_path = str(resolve_path(config_path))
        self._logger = logger
        self._config = self._load_config()
        self._tag_size_m = float(self._config.get("tag_size_m", 0.04))
        self._tag_id = self._config.get("tag_id")
        default_cache = Path(self._config_path).with_name("T_base_cam.npy")
        self._cache_path = resolve_path(self._config.get("extrinsics_cache_path", default_cache))

    def _load_config(self) -> dict[str, Any]:
        try:
            return load_camera_config(self._config_path)
        except ProjectError as exc:
            self._logger.warn(f"Falling back to default calibration config: {exc}")
            return {}

    def _as_transform(self, value: Any, field_name: str) -> np.ndarray:
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"{field_name} must have shape (4, 4), got {matrix.shape}.")
        return matrix

    def _rgb_to_gray(self, rgb: Any) -> np.ndarray:
        image = np.asarray(rgb)
        if image.ndim == 2:
            return image.astype(np.uint8)
        if image.ndim == 3 and image.shape[-1] >= 3:
            return np.dot(image[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32)).astype(np.uint8)
        raise ValueError(f"Unsupported RGB image shape for AprilTag detection: {image.shape}")

    def _build_detector(self):
        if PupilAprilTagDetector is not None:
            return "pupil", PupilAprilTagDetector(families="tag36h11")
        if apriltag is not None:
            options = apriltag.DetectorOptions(families="tag36h11")
            return "apriltag", apriltag.Detector(options)
        raise RuntimeError("No AprilTag detector backend is installed.")

    def detect_tags(self, rgb, intrinsics: CameraIntrinsics) -> list[dict]:
        _ = intrinsics
        detector_type, detector = self._build_detector()
        gray = self._rgb_to_gray(rgb)
        tags: list[dict] = []

        if detector_type == "pupil":
            detections = detector.detect(gray, estimate_tag_pose=False)
            for det in detections:
                tags.append(
                    {
                        "tag_id": int(det.tag_id),
                        "center": np.asarray(det.center, dtype=np.float64),
                        "corners": np.asarray(det.corners, dtype=np.float64),
                        "raw": det,
                    }
                )
            return tags

        detections = detector.detect(gray)
        for det in detections:
            tags.append(
                {
                    "tag_id": int(det.tag_id),
                    "center": np.asarray(det.center, dtype=np.float64),
                    "corners": np.asarray(det.corners, dtype=np.float64),
                    "raw": det,
                }
            )
        return tags

    def _select_tag(self, tags: list[dict]) -> dict:
        if not tags:
            raise RuntimeError("No AprilTag detections are available for calibration.")
        if self._tag_id is None:
            return tags[0]
        for tag in tags:
            if int(tag["tag_id"]) == int(self._tag_id):
                return tag
        raise RuntimeError(f"Configured AprilTag id {self._tag_id!r} was not detected.")

    def estimate_T_cam_tag(self, rgb, intrinsics: CameraIntrinsics) -> np.ndarray:
        static_transform = self._config.get("T_cam_tag")
        if static_transform is not None:
            return self._as_transform(static_transform, "T_cam_tag")

        tags = self.detect_tags(rgb, intrinsics)
        tag = self._select_tag(tags)
        raw = tag["raw"]
        if hasattr(raw, "pose_R") and hasattr(raw, "pose_t"):
            rotation = np.asarray(raw.pose_R, dtype=np.float64).reshape(3, 3)
            translation = np.asarray(raw.pose_t, dtype=np.float64).reshape(3)
            return make_transform(rotation, translation)

        if cv2 is None:
            raise RuntimeError("OpenCV is required to estimate AprilTag pose from image corners.")

        half = self._tag_size_m / 2.0
        object_points = np.array(
            [
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
            ],
            dtype=np.float32,
        )
        image_points = np.asarray(tag["corners"], dtype=np.float32)
        camera_matrix = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.cx],
                [0.0, intrinsics.fy, intrinsics.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            raise RuntimeError("OpenCV failed to estimate AprilTag pose.")
        rotation, _ = cv2.Rodrigues(rvec)
        translation = np.asarray(tvec, dtype=np.float64).reshape(3)
        return make_transform(rotation, translation)

    def estimate_T_base_cam(self, rgb, intrinsics: CameraIntrinsics) -> np.ndarray:
        static_transform = self._config.get("T_base_cam")
        if static_transform is not None:
            return self._as_transform(static_transform, "T_base_cam")

        T_base_tag = self._config.get("T_base_tag")
        if T_base_tag is None:
            raise RuntimeError("Calibration config must provide T_base_tag or T_base_cam for AprilTag calibration.")

        T_cam_tag = self.estimate_T_cam_tag(rgb, intrinsics)
        T_base_tag_np = self._as_transform(T_base_tag, "T_base_tag")
        T_tag_cam = invert_transform(T_cam_tag)
        return T_base_tag_np @ T_tag_cam

    def load_cached_T_base_cam(self) -> np.ndarray:
        if not self._cache_path.exists():
            raise FileNotFoundError(f"Cached extrinsics file does not exist: {self._cache_path}")
        matrix = np.load(self._cache_path)
        if matrix.shape != (4, 4):
            raise ValueError(f"Cached extrinsics must have shape (4, 4), got {matrix.shape}")
        return matrix.astype(np.float64)

    def save_cached_T_base_cam(self, T_base_cam: np.ndarray) -> None:
        matrix = self._as_transform(T_base_cam, "T_base_cam")
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._cache_path, matrix)
