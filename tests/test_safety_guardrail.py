from __future__ import annotations

import numpy as np
import pytest

from common.datatypes import MotionPlan, Pose3D, Waypoint
from common.exceptions import SafetyViolationError
from common.logger import ProjectLogger
from control_actuation.safety_guardrail import SafetyGuardrail


def _write_config(path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


def _make_pose(x: float, y: float, z: float) -> Pose3D:
    return Pose3D(position=np.array([x, y, z], dtype=np.float64), quaternion=np.array([0.0, 0.0, 0.0, 1.0]))


def test_validate_pose_accepts_legal_pose(tmp_path) -> None:
    workspace = _write_config(
        tmp_path / "workspace.yaml",
        "table_height_m: 0.0\nxyz_min_m: [0.1, -0.2, 0.02]\nxyz_max_m: [0.8, 0.2, 0.5]\n",
    )
    robot = _write_config(tmp_path / "robot.yaml", "table_clearance_m: 0.01\n")
    guardrail = SafetyGuardrail(workspace, robot, ProjectLogger("logs/test_safety_guardrail"))
    guardrail.validate_pose(_make_pose(0.3, 0.0, 0.1))


def test_validate_pose_rejects_out_of_bounds_pose(tmp_path) -> None:
    workspace = _write_config(
        tmp_path / "workspace.yaml",
        "table_height_m: 0.0\nxyz_min_m: [0.1, -0.2, 0.02]\nxyz_max_m: [0.8, 0.2, 0.5]\n",
    )
    robot = _write_config(tmp_path / "robot.yaml", "table_clearance_m: 0.01\n")
    guardrail = SafetyGuardrail(workspace, robot, ProjectLogger("logs/test_safety_guardrail"))
    with pytest.raises(SafetyViolationError):
        guardrail.validate_pose(_make_pose(0.05, 0.0, 0.1))


def test_validate_motion_plan_rejects_low_waypoint(tmp_path) -> None:
    workspace = _write_config(
        tmp_path / "workspace.yaml",
        "table_height_m: 0.0\nxyz_min_m: [0.1, -0.2, 0.02]\nxyz_max_m: [0.8, 0.2, 0.5]\n",
    )
    robot = _write_config(tmp_path / "robot.yaml", "table_clearance_m: 0.01\n")
    guardrail = SafetyGuardrail(workspace, robot, ProjectLogger("logs/test_safety_guardrail"))
    motion_plan = MotionPlan(
        waypoints=[
            Waypoint(pose=_make_pose(0.3, 0.0, 0.1)),
            Waypoint(pose=_make_pose(0.3, 0.0, 0.005)),
        ]
    )
    with pytest.raises(SafetyViolationError):
        guardrail.validate_motion_plan(motion_plan)
