from __future__ import annotations

import numpy as np

from common.datatypes import BBox2D, CameraIntrinsics, Pose3D
from common.geometry import (
    bbox_center_uv,
    clamp_xyz_to_bounds,
    deproject_uv_to_cam,
    make_topdown_quaternion,
    pose_distance,
    transform_point,
)


def test_bbox_center_uv() -> None:
    bbox = BBox2D(10, 20, 30, 40)
    assert bbox_center_uv(bbox) == (20, 30)


def test_deproject_uv_to_cam() -> None:
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    point = deproject_uv_to_cam((320, 240), 1.2, intrinsics)
    assert np.allclose(point, np.array([0.0, 0.0, 1.2]))


def test_transform_point() -> None:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    transformed = transform_point(T, np.array([1.0, 2.0, 3.0], dtype=np.float64))
    assert np.allclose(transformed, np.array([1.1, 1.8, 3.3], dtype=np.float64))


def test_pose_distance() -> None:
    pose_a = Pose3D(position=[0.0, 0.0, 0.0], quaternion=[0.0, 0.0, 0.0, 1.0])
    pose_b = Pose3D(position=[0.0, 0.3, 0.4], quaternion=[0.0, 0.0, 0.0, 1.0])
    assert pose_distance(pose_a, pose_b) == 0.5


def test_make_topdown_quaternion_is_normalized() -> None:
    quat = make_topdown_quaternion(0.0)
    assert quat.shape == (4,)
    assert np.isclose(np.linalg.norm(quat), 1.0)


def test_clamp_xyz_to_bounds() -> None:
    xyz = np.array([2.0, -1.0, 0.5], dtype=np.float64)
    clamped = clamp_xyz_to_bounds(
        xyz,
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
    )
    assert np.allclose(clamped, np.array([1.0, 0.0, 0.5], dtype=np.float64))
