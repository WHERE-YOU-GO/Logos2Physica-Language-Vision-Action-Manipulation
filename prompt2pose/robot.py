"""Lite6 robot wrapper with pick / place primitives.

Mirrors the conventions used in `robotics5551/checkpoint1.py`:
- TCP offset of 67 mm in Z (Lite6 gripper length)
- xArm SDK uses millimeters and degrees
- Top-down grasping orientation: roll=180, pitch=0, yaw configurable
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

try:
    from xarm.wrapper import XArmAPI
except ImportError:  # pragma: no cover
    XArmAPI = None  # type: ignore[assignment]


GRIPPER_LENGTH_MM = 67.0          # Lite6 gripper length, matches checkpoint1
APPROACH_HEIGHT_MM = 80.0         # hover height before grasp/place
LIFT_HEIGHT_MM = 120.0            # post-grasp lift
DEFAULT_LINEAR_SPEED_MM_S = 80.0
DEFAULT_APPROACH_SPEED_MM_S = 40.0


class Lite6:
    """High-level Lite6 wrapper. All public coordinates are in METERS."""

    def __init__(self, robot_ip: str) -> None:
        if XArmAPI is None:
            raise RuntimeError("xarm SDK not installed")
        if not robot_ip:
            raise ValueError("robot_ip is required")
        self._arm = XArmAPI(robot_ip)
        self._connected = False

    # ----- lifecycle ----------------------------------------------------------
    def connect(self) -> None:
        self._arm.connect()
        self._arm.motion_enable(enable=True)
        self._arm.set_tcp_offset([0, 0, GRIPPER_LENGTH_MM, 0, 0, 0])
        self._arm.set_mode(0)
        self._arm.set_state(0)
        self._connected = True

    def home(self) -> None:
        self._arm.move_gohome(wait=True)
        time.sleep(0.3)

    def disconnect(self) -> None:
        try:
            self.home()
        finally:
            self._arm.disconnect()
            self._connected = False

    # ----- gripper ------------------------------------------------------------
    def open_gripper(self) -> None:
        if hasattr(self._arm, "open_lite6_gripper"):
            self._arm.open_lite6_gripper()
        else:
            self._arm.set_gripper_position(850, wait=True)
        time.sleep(0.3)

    def close_gripper(self) -> None:
        if hasattr(self._arm, "close_lite6_gripper"):
            self._arm.close_lite6_gripper()
        else:
            self._arm.set_gripper_position(0, wait=True)
        time.sleep(0.3)

    # ----- low-level move -----------------------------------------------------
    def _move_xyz(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        yaw_deg: float = 0.0,
        speed_mm_s: float = DEFAULT_LINEAR_SPEED_MM_S,
    ) -> None:
        self._arm.set_position(
            x=x_m * 1000.0,
            y=y_m * 1000.0,
            z=z_m * 1000.0,
            roll=180.0,
            pitch=0.0,
            yaw=yaw_deg,
            speed=speed_mm_s,
            wait=True,
        )

    # ----- high-level skills --------------------------------------------------
    def grasp_at(self, position_m: np.ndarray, yaw_deg: float = 0.0) -> None:
        """Approach from above, close gripper at the target, then lift."""
        x, y, z = (float(v) for v in position_m)
        self.open_gripper()
        self._move_xyz(x, y, z + APPROACH_HEIGHT_MM / 1000.0, yaw_deg)
        self._move_xyz(x, y, z, yaw_deg, speed_mm_s=DEFAULT_APPROACH_SPEED_MM_S)
        self.close_gripper()
        self._move_xyz(x, y, z + LIFT_HEIGHT_MM / 1000.0, yaw_deg)

    def place_at(
        self,
        position_m: np.ndarray,
        yaw_deg: float = 0.0,
        stack_offset_m: float = 0.0,
    ) -> None:
        """Move above target, descend, release gripper, retreat."""
        x, y, z = (float(v) for v in position_m)
        z = z + stack_offset_m
        self._move_xyz(x, y, z + APPROACH_HEIGHT_MM / 1000.0, yaw_deg)
        self._move_xyz(x, y, z, yaw_deg, speed_mm_s=DEFAULT_APPROACH_SPEED_MM_S)
        self.open_gripper()
        self._move_xyz(x, y, z + APPROACH_HEIGHT_MM / 1000.0, yaw_deg)

    def goto_pose_m(
        self,
        position_m: np.ndarray,
        yaw_deg: float = 0.0,
        speed_mm_s: Optional[float] = None,
    ) -> None:
        x, y, z = (float(v) for v in position_m)
        self._move_xyz(x, y, z, yaw_deg, speed_mm_s=speed_mm_s or DEFAULT_LINEAR_SPEED_MM_S)
