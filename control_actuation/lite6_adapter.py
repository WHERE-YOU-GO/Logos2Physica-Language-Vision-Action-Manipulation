from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from common.config_loader import load_robot_config
from common.datatypes import Pose3D
from common.exceptions import ExecutionError, ProjectError
from common.logger import ProjectLogger

try:
    from xarm.wrapper import XArmAPI
except ImportError as exc:  # pragma: no cover
    XArmAPI = None
    _XARM_IMPORT_ERROR = exc
else:
    _XARM_IMPORT_ERROR = None


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


def _rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def _quaternion_to_rpy(quaternion: np.ndarray) -> tuple[float, float, float]:
    x, y, z, w = [float(value) for value in quaternion]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class Lite6Adapter:
    def __init__(self, robot_config_path: str, logger: ProjectLogger) -> None:
        self._logger = logger
        self._robot_config = self._safe_load(robot_config_path)
        self._robot_ip = str(self._robot_config.get("robot_ip", "")).strip()
        self._angles_in_degrees = bool(self._robot_config.get("angles_in_degrees", True))
        self._default_linear_speed = float(self._robot_config.get("linear_speed", 50.0))
        self._default_joint_speed = float(self._robot_config.get("joint_speed", 30.0))
        self._arm: Any | None = None
        self._connected = False
        self._last_gripper_width_m: float | None = None

    def _safe_load(self, path: str) -> dict[str, Any]:
        try:
            return load_robot_config(str(Path(path)))
        except ProjectError as exc:
            _log(self._logger, "warn", f"Falling back to default robot config: {exc}")
            return {}

    def _ensure_sdk(self) -> None:
        if XArmAPI is None:
            raise RuntimeError("xArm SDK is not installed. Install xarm-python-sdk to use Lite6Adapter.") from _XARM_IMPORT_ERROR

    def _ensure_connected(self) -> None:
        self._ensure_sdk()
        if not self._connected or self._arm is None:
            raise RuntimeError("Lite6Adapter is not connected.")

    def connect(self) -> None:
        self._ensure_sdk()
        if self._connected:
            return
        if not self._robot_ip:
            raise RuntimeError("robot_ip is required to connect to Lite6.")

        self._arm = XArmAPI(self._robot_ip)
        try:
            if hasattr(self._arm, "connect"):
                self._arm.connect()
            if hasattr(self._arm, "motion_enable"):
                self._arm.motion_enable(True)
            if hasattr(self._arm, "set_mode"):
                self._arm.set_mode(0)
            if hasattr(self._arm, "set_state"):
                self._arm.set_state(0)
        except Exception as exc:  # pragma: no cover
            self._arm = None
            raise ExecutionError("Failed to connect to Lite6 robot.") from exc

        self._connected = True
        _log(self._logger, "info", "Lite6 robot connected.")

    def disconnect(self) -> None:
        if self._arm is not None and hasattr(self._arm, "disconnect"):
            self._arm.disconnect()
        self._arm = None
        self._connected = False
        _log(self._logger, "info", "Lite6 robot disconnected.")

    def emergency_stop(self) -> None:
        self._ensure_connected()
        if hasattr(self._arm, "emergency_stop"):
            self._arm.emergency_stop()
        elif hasattr(self._arm, "set_state"):
            self._arm.set_state(4)

    def get_current_joints(self):
        self._ensure_connected()
        if hasattr(self._arm, "get_servo_angle"):
            result = self._arm.get_servo_angle(is_radian=not self._angles_in_degrees)
            return result[1] if isinstance(result, tuple) and len(result) >= 2 else result
        raise RuntimeError("xArm SDK does not expose get_servo_angle on this adapter.")

    def get_current_pose(self):
        self._ensure_connected()
        if not hasattr(self._arm, "get_position"):
            raise RuntimeError("xArm SDK does not expose get_position on this adapter.")

        result = self._arm.get_position(is_radian=not self._angles_in_degrees)
        pose_values = result[1] if isinstance(result, tuple) and len(result) >= 2 else result
        if pose_values is None or len(pose_values) < 6:
            raise RuntimeError("Robot returned an invalid Cartesian pose.")

        x_mm, y_mm, z_mm, roll, pitch, yaw = [float(value) for value in pose_values[:6]]
        if self._angles_in_degrees:
            roll = math.radians(roll)
            pitch = math.radians(pitch)
            yaw = math.radians(yaw)
        quaternion = _rpy_to_quaternion(roll, pitch, yaw)
        return Pose3D(
            position=np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0,
            quaternion=quaternion,
            frame_id="base",
        )

    def move_joints(self, joints, speed: float) -> None:
        self._ensure_connected()
        if not hasattr(self._arm, "set_servo_angle"):
            raise RuntimeError("xArm SDK does not expose set_servo_angle on this adapter.")

        command_speed = float(speed) if speed > 0.0 else self._default_joint_speed
        try:
            self._arm.set_servo_angle(
                angle=list(joints),
                speed=command_speed,
                wait=True,
                is_radian=not self._angles_in_degrees,
            )
        except Exception as exc:  # pragma: no cover
            raise ExecutionError("Failed to move Lite6 joints.") from exc

    def move_linear(self, pose: Pose3D, speed: float) -> None:
        self._ensure_connected()
        if not hasattr(self._arm, "set_position"):
            raise RuntimeError("xArm SDK does not expose set_position on this adapter.")

        roll, pitch, yaw = _quaternion_to_rpy(pose.quaternion)
        if self._angles_in_degrees:
            roll = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw = math.degrees(yaw)

        x_mm, y_mm, z_mm = (pose.position * 1000.0).tolist()
        command_speed = float(speed) if speed > 0.0 else self._default_linear_speed
        try:
            self._arm.set_position(
                x=x_mm,
                y=y_mm,
                z=z_mm,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                speed=command_speed,
                wait=True,
            )
        except Exception as exc:  # pragma: no cover
            raise ExecutionError("Failed to execute linear motion on Lite6.") from exc

    def open_gripper(self) -> None:
        self._ensure_connected()
        if hasattr(self._arm, "open_lite6_gripper"):
            self._arm.open_lite6_gripper()
            return
        if hasattr(self._arm, "set_gripper_position"):
            self._arm.set_gripper_position(850, wait=True)
            self._last_gripper_width_m = 0.085
            return
        raise RuntimeError("xArm SDK gripper open command is unavailable.")

    def close_gripper(self) -> None:
        self._ensure_connected()
        if hasattr(self._arm, "close_lite6_gripper"):
            self._arm.close_lite6_gripper()
            self._last_gripper_width_m = 0.0
            return
        if hasattr(self._arm, "set_gripper_position"):
            self._arm.set_gripper_position(0, wait=True)
            self._last_gripper_width_m = 0.0
            return
        raise RuntimeError("xArm SDK gripper close command is unavailable.")

    def set_gripper_width(self, width_m: float) -> None:
        self._ensure_connected()
        width_m = float(width_m)
        if width_m < 0.0:
            raise ValueError("width_m must be non-negative.")

        if hasattr(self._arm, "set_gripper_position"):
            self._arm.set_gripper_position(int(round(width_m * 1000.0)), wait=True)
            self._last_gripper_width_m = width_m
            return
        raise RuntimeError("xArm SDK does not expose set_gripper_position on this adapter.")

    def get_gripper_state(self) -> dict:
        self._ensure_connected()
        state: dict[str, Any] = {"width_m": self._last_gripper_width_m}
        if hasattr(self._arm, "get_gripper_position"):
            result = self._arm.get_gripper_position()
            raw_position = result[1] if isinstance(result, tuple) and len(result) >= 2 else result
            if raw_position is not None:
                state["width_m"] = float(raw_position) / 1000.0
        if hasattr(self._arm, "get_gripper_status"):
            result = self._arm.get_gripper_status()
            status = result[1] if isinstance(result, tuple) and len(result) >= 2 else result
            if isinstance(status, dict):
                state.update(status)
        if "is_holding" not in state and state.get("width_m") is not None:
            width_m = float(state["width_m"])
            state["is_holding"] = 0.002 < width_m < 0.08
        return state
