from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _ensure_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty.")
    return value


def _ensure_dict(value: dict[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dict[str, Any].")
    return dict(value)


def _ensure_float(value: Any, field_name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{field_name} must be finite.")
    return scalar


def _ensure_positive_float(value: Any, field_name: str) -> float:
    scalar = _ensure_float(value, field_name)
    if scalar <= 0.0:
        raise ValueError(f"{field_name} must be > 0.")
    return scalar


def _ensure_non_negative_float(value: Any, field_name: str) -> float:
    scalar = _ensure_float(value, field_name)
    if scalar < 0.0:
        raise ValueError(f"{field_name} must be >= 0.")
    return scalar


def _ensure_int(value: Any, field_name: str) -> int:
    integer = int(value)
    if integer != value and not (isinstance(value, float) and value.is_integer()):
        raise ValueError(f"{field_name} must be an integer-compatible value.")
    return integer


def _ensure_positive_int(value: Any, field_name: str) -> int:
    integer = _ensure_int(value, field_name)
    if integer <= 0:
        raise ValueError(f"{field_name} must be > 0.")
    return integer


def _ensure_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool.")
    return value


def _ensure_optional_bool(value: Any, field_name: str) -> bool | None:
    if value is None:
        return None
    return _ensure_bool(value, field_name)


def _ensure_numpy_vector(value: Any, field_name: str, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (size,):
        raise ValueError(f"{field_name} must have shape ({size},), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{field_name} must contain only finite values.")
    return vector.copy()


def _ensure_transform_matrix(value: Any, field_name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{field_name} must have shape (4, 4), got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field_name} must contain only finite values.")
    return matrix.copy()


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        self.fx = _ensure_positive_float(self.fx, "fx")
        self.fy = _ensure_positive_float(self.fy, "fy")
        self.cx = _ensure_float(self.cx, "cx")
        self.cy = _ensure_float(self.cy, "cy")
        self.width = _ensure_positive_int(self.width, "width")
        self.height = _ensure_positive_int(self.height, "height")


@dataclass
class BBox2D:
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        self.x1 = _ensure_int(self.x1, "x1")
        self.y1 = _ensure_int(self.y1, "y1")
        self.x2 = _ensure_int(self.x2, "x2")
        self.y2 = _ensure_int(self.y2, "y2")
        if self.x2 <= self.x1:
            raise ValueError("BBox2D requires x2 > x1.")
        if self.y2 <= self.y1:
            raise ValueError("BBox2D requires y2 > y1.")

    def center_uv(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def width(self) -> int:
        return self.x2 - self.x1

    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class Detection2D:
    label: str
    score: float
    bbox: BBox2D
    phrase: str | None = None
    mask: Any | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label = _ensure_non_empty_string(self.label, "label")
        self.score = _ensure_non_negative_float(self.score, "score")
        if not isinstance(self.bbox, BBox2D):
            raise TypeError("bbox must be a BBox2D instance.")
        if self.phrase is not None:
            self.phrase = _ensure_non_empty_string(self.phrase, "phrase")
        self.extras = _ensure_dict(self.extras, "extras")


@dataclass
class RGBDFrame:
    rgb: Any
    depth: Any
    intrinsics: CameraIntrinsics
    T_base_cam: Any
    timestamp: float

    def __post_init__(self) -> None:
        if not isinstance(self.intrinsics, CameraIntrinsics):
            raise TypeError("intrinsics must be a CameraIntrinsics instance.")
        self.T_base_cam = _ensure_transform_matrix(self.T_base_cam, "T_base_cam")
        self.timestamp = _ensure_float(self.timestamp, "timestamp")


@dataclass
class Pose3D:
    position: Any
    quaternion: Any
    frame_id: str = "base"

    def __post_init__(self) -> None:
        self.position = _ensure_numpy_vector(self.position, "position", 3)
        self.quaternion = _ensure_numpy_vector(self.quaternion, "quaternion", 4)
        norm = float(np.linalg.norm(self.quaternion))
        if norm <= 0.0:
            raise ValueError("quaternion must have non-zero norm.")
        self.quaternion = self.quaternion / norm
        self.frame_id = _ensure_non_empty_string(self.frame_id, "frame_id")


@dataclass
class SceneObject:
    object_id: str
    label: str
    bbox: BBox2D
    center_cam: Any
    center_base: Any
    confidence: float
    color: str | None = None
    shape: str | None = None
    is_graspable: bool = True
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.object_id = _ensure_non_empty_string(self.object_id, "object_id")
        self.label = _ensure_non_empty_string(self.label, "label")
        if not isinstance(self.bbox, BBox2D):
            raise TypeError("bbox must be a BBox2D instance.")
        self.center_cam = _ensure_numpy_vector(self.center_cam, "center_cam", 3)
        self.center_base = _ensure_numpy_vector(self.center_base, "center_base", 3)
        self.confidence = _ensure_non_negative_float(self.confidence, "confidence")
        if self.color is not None:
            self.color = _ensure_non_empty_string(self.color, "color")
        if self.shape is not None:
            self.shape = _ensure_non_empty_string(self.shape, "shape")
        self.is_graspable = _ensure_bool(self.is_graspable, "is_graspable")
        self.extras = _ensure_dict(self.extras, "extras")


@dataclass
class SceneState:
    frame_timestamp: float
    objects: list[SceneObject]
    table_height_m: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frame_timestamp = _ensure_float(self.frame_timestamp, "frame_timestamp")
        self.objects = list(self.objects)
        for index, obj in enumerate(self.objects):
            if not isinstance(obj, SceneObject):
                raise TypeError(f"objects[{index}] must be a SceneObject instance.")
        if self.table_height_m is not None:
            self.table_height_m = _ensure_float(self.table_height_m, "table_height_m")
        self.extras = _ensure_dict(self.extras, "extras")

    def get_object_by_id(self, object_id: str) -> SceneObject | None:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def graspable_objects(self) -> list[SceneObject]:
        return [obj for obj in self.objects if obj.is_graspable]


@dataclass
class GraspPose:
    pose: Pose3D
    approach_vector: Any = field(
        default_factory=lambda: np.array([0.0, 0.0, -1.0], dtype=np.float64)
    )
    pregrasp_offset_m: float = 0.05
    grasp_width_m: float | None = None
    score: float = 1.0
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.pose, Pose3D):
            raise TypeError("pose must be a Pose3D instance.")
        self.approach_vector = _ensure_numpy_vector(self.approach_vector, "approach_vector", 3)
        norm = float(np.linalg.norm(self.approach_vector))
        if norm <= 0.0:
            raise ValueError("approach_vector must have non-zero norm.")
        self.approach_vector = self.approach_vector / norm
        self.pregrasp_offset_m = _ensure_non_negative_float(
            self.pregrasp_offset_m,
            "pregrasp_offset_m",
        )
        if self.grasp_width_m is not None:
            self.grasp_width_m = _ensure_positive_float(self.grasp_width_m, "grasp_width_m")
        self.score = _ensure_non_negative_float(self.score, "score")
        self.extras = _ensure_dict(self.extras, "extras")


@dataclass
class Waypoint:
    pose: Pose3D
    name: str = ""
    speed_scale: float = 1.0
    tolerance_m: float = 0.002
    gripper_open: bool | None = None
    hold_time_s: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name:
            self.name = _ensure_non_empty_string(self.name, "name")
        if not isinstance(self.pose, Pose3D):
            raise TypeError("pose must be a Pose3D instance.")
        self.speed_scale = _ensure_positive_float(self.speed_scale, "speed_scale")
        self.tolerance_m = _ensure_non_negative_float(self.tolerance_m, "tolerance_m")
        self.gripper_open = _ensure_optional_bool(self.gripper_open, "gripper_open")
        self.hold_time_s = _ensure_non_negative_float(self.hold_time_s, "hold_time_s")
        self.extras = _ensure_dict(self.extras, "extras")


@dataclass
class MotionPlan:
    waypoints: list[Waypoint]
    frame_id: str = "base"
    max_linear_speed_mps: float | None = None
    max_angular_speed_rps: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.waypoints = list(self.waypoints)
        for index, waypoint in enumerate(self.waypoints):
            if not isinstance(waypoint, Waypoint):
                raise TypeError(f"waypoints[{index}] must be a Waypoint instance.")
        self.frame_id = _ensure_non_empty_string(self.frame_id, "frame_id")
        if self.max_linear_speed_mps is not None:
            self.max_linear_speed_mps = _ensure_positive_float(
                self.max_linear_speed_mps,
                "max_linear_speed_mps",
            )
        if self.max_angular_speed_rps is not None:
            self.max_angular_speed_rps = _ensure_positive_float(
                self.max_angular_speed_rps,
                "max_angular_speed_rps",
            )
        self.extras = _ensure_dict(self.extras, "extras")

    def is_empty(self) -> bool:
        return len(self.waypoints) == 0


@dataclass
class PickPlacePlan:
    pick_object_id: str
    grasp_pose: GraspPose
    place_pose: Pose3D
    approach_plan: MotionPlan | None = None
    transfer_plan: MotionPlan | None = None
    retreat_plan: MotionPlan | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pick_object_id = _ensure_non_empty_string(self.pick_object_id, "pick_object_id")
        if not isinstance(self.grasp_pose, GraspPose):
            raise TypeError("grasp_pose must be a GraspPose instance.")
        if not isinstance(self.place_pose, Pose3D):
            raise TypeError("place_pose must be a Pose3D instance.")
        for field_name in ("approach_plan", "transfer_plan", "retreat_plan"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, MotionPlan):
                raise TypeError(f"{field_name} must be a MotionPlan instance or None.")
        self.extras = _ensure_dict(self.extras, "extras")


@dataclass
class VerificationResult:
    success: bool
    message: str = ""
    score: float | None = None
    measured_pose: Pose3D | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.success = _ensure_bool(self.success, "success")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string.")
        if self.score is not None:
            self.score = _ensure_non_negative_float(self.score, "score")
        if self.measured_pose is not None and not isinstance(self.measured_pose, Pose3D):
            raise TypeError("measured_pose must be a Pose3D instance or None.")
        self.extras = _ensure_dict(self.extras, "extras")


__all__ = [
    "CameraIntrinsics",
    "BBox2D",
    "Detection2D",
    "RGBDFrame",
    "Pose3D",
    "SceneObject",
    "SceneState",
    "GraspPose",
    "Waypoint",
    "MotionPlan",
    "PickPlacePlan",
    "VerificationResult",
]
