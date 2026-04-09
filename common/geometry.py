from __future__ import annotations

import math

import numpy as np

from .datatypes import BBox2D, CameraIntrinsics, Pose3D


def _as_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _as_transform(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    trace = float(np.trace(R))
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm <= 0.0:
        raise ValueError("Failed to convert rotation matrix to a valid quaternion.")
    return quaternion / norm


def bbox_center_uv(bbox: BBox2D) -> tuple[int, int]:
    """Return the integer center pixel of a bounding box."""

    if not isinstance(bbox, BBox2D):
        raise TypeError("bbox must be a BBox2D instance.")
    return bbox.center_uv()


def deproject_uv_to_cam(
    uv: tuple[int, int],
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Back-project an image pixel with depth into the camera frame."""

    if not isinstance(intrinsics, CameraIntrinsics):
        raise TypeError("intrinsics must be a CameraIntrinsics instance.")
    if len(uv) != 2:
        raise ValueError("uv must contain exactly two elements.")

    u = int(uv[0])
    v = int(uv[1])
    if not (0 <= u < intrinsics.width):
        raise ValueError(f"u must be within [0, {intrinsics.width - 1}], got {u}.")
    if not (0 <= v < intrinsics.height):
        raise ValueError(f"v must be within [0, {intrinsics.height - 1}], got {v}.")

    z = float(depth_m)
    if not np.isfinite(z) or z <= 0.0:
        raise ValueError("depth_m must be a finite value greater than 0.")

    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    return np.array([x, y, z], dtype=np.float64)


def transform_point(T_dst_src: np.ndarray, p_src: np.ndarray) -> np.ndarray:
    """Transform a single 3D point with a homogeneous transform."""

    matrix = _as_transform(T_dst_src, "T_dst_src")
    point = _as_vector(p_src, 3, "p_src")

    point_h = np.concatenate([point, np.array([1.0], dtype=np.float64)])
    transformed = matrix @ point_h
    w = transformed[3]
    if np.isclose(w, 0.0):
        raise ValueError("Transformed homogeneous point has w close to zero.")
    return transformed[:3] / w


def transform_points(T_dst_src: np.ndarray, pts_src: np.ndarray) -> np.ndarray:
    """Transform an array of 3D points with a homogeneous transform."""

    matrix = _as_transform(T_dst_src, "T_dst_src")
    points = np.asarray(pts_src, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"pts_src must have shape (N, 3), got {points.shape}.")
    if not np.all(np.isfinite(points)):
        raise ValueError("pts_src must contain only finite values.")

    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    points_h = np.hstack([points, ones])
    transformed_h = (matrix @ points_h.T).T
    w = transformed_h[:, 3:4]
    if np.any(np.isclose(w, 0.0)):
        raise ValueError("At least one transformed homogeneous point has w close to zero.")
    return transformed_h[:, :3] / w


def pose_distance(a: Pose3D, b: Pose3D) -> float:
    """Return Euclidean translational distance between two poses in the same frame."""

    if not isinstance(a, Pose3D):
        raise TypeError("a must be a Pose3D instance.")
    if not isinstance(b, Pose3D):
        raise TypeError("b must be a Pose3D instance.")
    if a.frame_id != b.frame_id:
        raise ValueError(
            f"Pose frames must match to compute distance, got {a.frame_id!r} and {b.frame_id!r}."
        )
    return float(np.linalg.norm(a.position - b.position))


def make_topdown_quaternion(yaw_rad: float = 0.0) -> np.ndarray:
    """Create a top-down quaternion with an additional in-plane yaw rotation."""

    yaw = float(yaw_rad)
    if not np.isfinite(yaw):
        raise ValueError("yaw_rad must be finite.")

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    Rz = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    Rx_pi = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    rotation = Rz @ Rx_pi
    return _rotation_matrix_to_quaternion(rotation)


def clamp_xyz_to_bounds(
    xyz: np.ndarray,
    xyz_min: np.ndarray,
    xyz_max: np.ndarray,
) -> np.ndarray:
    """Clamp a 3D point into axis-aligned bounds."""

    point = _as_vector(xyz, 3, "xyz")
    minimum = _as_vector(xyz_min, 3, "xyz_min")
    maximum = _as_vector(xyz_max, 3, "xyz_max")
    if np.any(minimum > maximum):
        raise ValueError("xyz_min must be <= xyz_max element-wise.")
    return np.clip(point, minimum, maximum)


__all__ = [
    "bbox_center_uv",
    "deproject_uv_to_cam",
    "transform_point",
    "transform_points",
    "pose_distance",
    "make_topdown_quaternion",
    "clamp_xyz_to_bounds",
]
