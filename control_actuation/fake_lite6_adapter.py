from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from common.datatypes import Pose3D
from common.logger import ProjectLogger


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


def _default_pose() -> Pose3D:
    return Pose3D(
        position=np.array([0.30, 0.0, 0.22], dtype=np.float64),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        frame_id="base",
    )


class FakeLite6Adapter:
    def __init__(
        self,
        logger: ProjectLogger | None = None,
        initial_pose: Pose3D | None = None,
        initial_joints: Sequence[float] | None = None,
        simulate_grasp_success: bool = True,
    ) -> None:
        self._logger = logger
        self._current_pose = initial_pose if initial_pose is not None else _default_pose()
        self._current_joints = self._validate_joints(initial_joints if initial_joints is not None else [0.0] * 6)
        self._connected = False
        self._simulate_grasp_success = bool(simulate_grasp_success)
        self._gripper_state = "open"
        self._gripper_width_m = 0.085
        self._is_holding = False
        self._command_log: list[dict[str, Any]] = []
        self._last_grasp_pose: Pose3D | None = None
        self._last_release_pose: Pose3D | None = None

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("FakeLite6Adapter is not connected.")

    def _record(self, action: str, **payload: Any) -> None:
        entry = {"action": action, **payload}
        self._command_log.append(entry)
        _log(self._logger, "info", f"FakeLite6Adapter action={action} payload={payload}")

    def _validate_speed(self, speed: float | None) -> float | None:
        if speed is None:
            return None
        speed_value = float(speed)
        if not np.isfinite(speed_value) or speed_value <= 0.0:
            raise ValueError(f"speed must be a positive finite float when provided, got {speed}.")
        return speed_value

    def _validate_joints(self, joints: Sequence[float]) -> list[float]:
        joint_values = [float(value) for value in joints]
        if len(joint_values) != 6:
            raise ValueError(f"Lite6 expects 6 joint values, got {len(joint_values)}.")
        if not np.all(np.isfinite(np.asarray(joint_values, dtype=np.float64))):
            raise ValueError("joints must contain only finite numeric values.")
        return joint_values

    def connect(self) -> None:
        if self._connected:
            return
        self._connected = True
        self._record("connect")

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        self._record("disconnect")

    def get_current_pose(self) -> Pose3D:
        self._ensure_connected()
        return Pose3D(
            position=self._current_pose.position.copy(),
            quaternion=self._current_pose.quaternion.copy(),
            frame_id=self._current_pose.frame_id,
        )

    def get_current_joints(self) -> list[float]:
        self._ensure_connected()
        return list(self._current_joints)

    def move_linear(self, target: Pose3D, speed: float | None = None) -> None:
        self._ensure_connected()
        if not isinstance(target, Pose3D):
            raise TypeError("target must be a Pose3D instance.")
        if target.frame_id != "base":
            raise ValueError(f"FakeLite6Adapter only accepts base-frame poses, got {target.frame_id!r}.")
        command_speed = self._validate_speed(speed)
        self._current_pose = Pose3D(
            position=target.position.copy(),
            quaternion=target.quaternion.copy(),
            frame_id=target.frame_id,
        )
        self._record(
            "move_linear",
            target_position=self._current_pose.position.tolist(),
            target_quaternion=self._current_pose.quaternion.tolist(),
            speed=command_speed,
        )

    def move_joints(self, joints: list[float], speed: float | None = None) -> None:
        self._ensure_connected()
        command_speed = self._validate_speed(speed)
        self._current_joints = self._validate_joints(joints)
        self._record("move_joints", joints=list(self._current_joints), speed=command_speed)

    def open_gripper(self) -> None:
        self._ensure_connected()
        self._gripper_state = "open"
        self._gripper_width_m = 0.085
        self._is_holding = False
        self._last_release_pose = self.get_current_pose()
        self._record("open_gripper", width_m=self._gripper_width_m)

    def close_gripper(self) -> None:
        self._ensure_connected()
        self._gripper_state = "closed"
        self._is_holding = self._simulate_grasp_success
        self._gripper_width_m = 0.03 if self._is_holding else 0.0
        self._last_grasp_pose = self.get_current_pose()
        self._record("close_gripper", width_m=self._gripper_width_m, is_holding=self._is_holding)

    def set_gripper_width(self, width_m: float) -> None:
        self._ensure_connected()
        width_value = float(width_m)
        if not np.isfinite(width_value) or width_value < 0.0:
            raise ValueError(f"width_m must be a non-negative finite float, got {width_m}.")
        self._gripper_width_m = width_value
        self._gripper_state = "custom"
        self._is_holding = 0.002 < width_value < 0.08 and self._simulate_grasp_success
        self._record("set_gripper_width", width_m=width_value, is_holding=self._is_holding)

    def get_gripper_state(self) -> dict[str, Any]:
        self._ensure_connected()
        return {
            "state": self._gripper_state,
            "width_m": self._gripper_width_m,
            "is_holding": self._is_holding,
        }

    def get_command_log(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self._command_log]

    def get_last_grasp_pose(self) -> Pose3D | None:
        return self._last_grasp_pose

    def get_last_release_pose(self) -> Pose3D | None:
        return self._last_release_pose


__all__ = ["FakeLite6Adapter"]
