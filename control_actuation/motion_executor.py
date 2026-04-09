from __future__ import annotations

import time
from typing import Any

from common.datatypes import MotionPlan, Waypoint
from common.exceptions import ExecutionError
from common.logger import ProjectLogger
from control_actuation.ik_solver import IKSolver
from control_actuation.lite6_adapter import Lite6Adapter


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


class MotionExecutor:
    def __init__(self, robot_adapter: Lite6Adapter, ik_solver: IKSolver, logger: ProjectLogger) -> None:
        self._robot_adapter = robot_adapter
        self._ik_solver = ik_solver
        self._logger = logger

    def _validate_waypoint(self, waypoint: Waypoint) -> None:
        if waypoint.pose.frame_id != "base":
            raise ExecutionError(
                f"Waypoint {(waypoint.name or waypoint.extras.get('name', '<unnamed>'))!r} must be expressed in the base frame."
            )

    def _execute_waypoint_side_effects(self, waypoint: Waypoint) -> None:
        if waypoint.gripper_open is True:
            self._robot_adapter.open_gripper()
        elif waypoint.gripper_open is False:
            self._robot_adapter.close_gripper()
        if waypoint.hold_time_s > 0.0:
            time.sleep(float(waypoint.hold_time_s))

    def execute_cartesian_plan(self, motion_plan: MotionPlan) -> None:
        for waypoint in motion_plan.waypoints:
            speed = float(waypoint.extras.get("speed", waypoint.speed_scale))
            try:
                self._validate_waypoint(waypoint)
                self._robot_adapter.move_linear(waypoint.pose, speed=speed)
                self._execute_waypoint_side_effects(waypoint)
            except Exception as exc:
                name = waypoint.name or waypoint.extras.get("name", "<unnamed>")
                raise ExecutionError(f"Failed to execute Cartesian waypoint {name!r}.") from exc

    def execute_joint_plan(self, motion_plan: MotionPlan) -> None:
        for waypoint in motion_plan.waypoints:
            self._validate_waypoint(waypoint)
            joint_positions = waypoint.extras.get("joint_positions")
            if joint_positions is None:
                try:
                    joint_positions = self._ik_solver.solve(waypoint.pose)
                except Exception as exc:
                    name = waypoint.name or waypoint.extras.get("name", "<unnamed>")
                    raise ExecutionError(f"Failed to solve IK for waypoint {name!r}.") from exc
            try:
                speed = float(waypoint.extras.get("speed", waypoint.speed_scale))
                self._robot_adapter.move_joints(joint_positions, speed=speed)
                self._execute_waypoint_side_effects(waypoint)
            except Exception as exc:
                name = waypoint.name or waypoint.extras.get("name", "<unnamed>")
                raise ExecutionError(f"Failed to execute joint waypoint {name!r}.") from exc
