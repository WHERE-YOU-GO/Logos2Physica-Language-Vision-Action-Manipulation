"""ZED 2i RGB-D camera wrapper + AprilTag-based base-frame calibration.

The wrapper mirrors `robotics5551/utils/zed_camera.py` exactly so it runs on the
same hardware setup, with one improvement: we explicitly set coordinate units to
METER so that the point cloud is in the same units as the world frame defined by
the AprilTag PnP solver in `compute_T_cam_robot`.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    import pyzed.sl as sl
except ImportError:  # pragma: no cover
    sl = None  # type: ignore[assignment]

try:
    from pupil_apriltags import Detector
except ImportError:  # pragma: no cover
    Detector = None  # type: ignore[assignment]


# ----- AprilTag layout: matches robotics5551/checkpoint0.py exactly. ----------
TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.08
# top-left, top-right, bottom-left, bottom-right of the workspace in robot base frame
TAG_CENTER_COORDINATES = [
    [0.38, 0.4],
    [0.38, -0.4],
    [0.0, 0.4],
    [0.0, -0.4],
]


# ----- ZED camera wrapper -----------------------------------------------------
class ZedCamera:
    """Background-threaded ZED 2i wrapper that exposes BGRA image + XYZ cloud."""

    def __init__(self, fps: int = 15, exposure: int = 20) -> None:
        if sl is None:
            raise RuntimeError("pyzed.sl is not installed; install the ZED SDK Python bindings")

        self._zed = sl.Camera()
        init = sl.InitParameters()
        init.enable_image_validity_check = True
        init.camera_resolution = sl.RESOLUTION.HD2K
        init.camera_fps = fps
        init.coordinate_units = sl.UNIT.METER  # match world frame from PnP
        init.depth_mode = sl.DEPTH_MODE.QUALITY

        err = self._zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED camera open failed: {err}")

        self._runtime = sl.RuntimeParameters()
        self._zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
        self._zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)
        for _ in range(50):
            self._zed.grab(self._runtime)
        self._zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure)

        info = self._zed.get_camera_information()
        cam = info.camera_configuration.calibration_parameters.left_cam
        self._intrinsic = np.eye(3)
        self._intrinsic[0, 0] = cam.fx
        self._intrinsic[1, 1] = cam.fy
        self._intrinsic[0, 2] = cam.cx
        self._intrinsic[1, 2] = cam.cy

        self._image_mat = sl.Mat()
        self._xyz_mat = sl.Mat()
        self._image: Optional[np.ndarray] = None
        self._xyz: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        while self._image is None or self._xyz is None:
            time.sleep(0.05)

    def _loop(self) -> None:
        while self._running:
            if self._zed.grab(self._runtime) == sl.ERROR_CODE.SUCCESS:
                self._zed.retrieve_image(self._image_mat, sl.VIEW.LEFT)
                self._zed.retrieve_measure(self._xyz_mat, sl.MEASURE.XYZ)
                with self._lock:
                    self._image = self._image_mat.get_data().copy()
                    self._xyz = self._xyz_mat.get_data().copy()
            else:
                time.sleep(0.01)

    @property
    def intrinsic(self) -> np.ndarray:
        return self._intrinsic

    @property
    def image(self) -> np.ndarray:
        with self._lock:
            assert self._image is not None
            return self._image.copy()

    @property
    def point_cloud(self) -> np.ndarray:
        """HxWx{3 or 4} float32, XYZ in meters (W slot may hold packed RGBA)."""
        with self._lock:
            assert self._xyz is not None
            return self._xyz.copy()

    def close(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join()
        self._zed.close()


# ----- AprilTag base-frame calibration ---------------------------------------
def _pnp_pairs(tags) -> tuple[np.ndarray, np.ndarray]:
    """Build (world, image) point pairs from base AprilTags 0..3."""
    world_points = np.empty((0, 3))
    image_points = np.empty((0, 2))
    for tag in tags:
        if tag.tag_id > 3:
            continue
        cx, cy = TAG_CENTER_COORDINATES[tag.tag_id]
        half = TAG_SIZE_M / 2
        # bottom-left, bottom-right, top-right, top-left (matches checkpoint0)
        corners_world = np.array(
            [
                [cx - half, cy + half, 0.0],
                [cx - half, cy - half, 0.0],
                [cx + half, cy - half, 0.0],
                [cx + half, cy + half, 0.0],
            ]
        )
        world_points = np.vstack([world_points, corners_world])
        image_points = np.vstack([image_points, tag.corners])
    return world_points, image_points


def compute_T_cam_robot(image: np.ndarray, intrinsic: np.ndarray) -> Optional[np.ndarray]:
    """Detect base-frame AprilTags and solve PnP for the camera-from-robot transform.

    Returns a 4x4 transform from robot base frame to camera frame, or None on failure.
    """
    if Detector is None:
        raise RuntimeError("pupil_apriltags not installed")

    detector = Detector(families=TAG_FAMILY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY) if image.ndim == 3 else image
    tags = detector.detect(gray, estimate_tag_pose=False)
    print(f"[calibration] tags found: {len(tags)}")

    world, pixels = _pnp_pairs(tags)
    if world.shape[0] < 4:
        print("[calibration] insufficient AprilTag corners")
        return None

    success, rvec, tvec = cv2.solvePnP(world, pixels, intrinsic, None)
    if not success:
        print("[calibration] solvePnP failed")
        return None

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def draw_pose_axes(image: np.ndarray, intrinsic: np.ndarray, T: np.ndarray, size: float = 0.1) -> None:
    """Draw RGB axes for a pose (matches robotics5551/utils/vis_utils.py)."""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3]
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float).reshape(-1, 3) * size
    img_pts, _ = cv2.projectPoints(pts, rvec, tvec, intrinsic, None)
    img_pts = np.round(img_pts).astype(int)
    o, x, y, z = (tuple(p.ravel()) for p in img_pts)
    cv2.line(image, o, x, (0, 0, 255), 2)
    cv2.line(image, o, y, (0, 255, 0), 2)
    cv2.line(image, o, z, (255, 0, 0), 2)
