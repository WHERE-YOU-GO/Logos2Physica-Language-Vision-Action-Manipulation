from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.config_loader import load_robot_config, load_workspace_config
from common.datatypes import MotionPlan, PickPlacePlan, Pose3D, SceneState
from common.exceptions import ProjectError, SafetyViolationError
from common.logger import ProjectLogger


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


class SafetyGuardrail:
    def __init__(self, workspace_config_path: str, robot_config_path: str, logger: ProjectLogger) -> None:
        self._logger = logger
        self._workspace_config = self._safe_load(workspace_config_path, load_workspace_config)
        self._robot_config = self._safe_load(robot_config_path, load_robot_config)

        self._xyz_min = np.asarray(
            self._workspace_config.get("xyz_min_m", [0.15, -0.35, 0.02]),
            dtype=np.float64,
        ).reshape(3)
        self._xyz_max = np.asarray(
            self._workspace_config.get("xyz_max_m", [0.75, 0.35, 0.45]),
            dtype=np.float64,
        ).reshape(3)
        self._table_height_m = float(self._workspace_config.get("table_height_m", 0.0))
        self._table_margin_m = float(self._robot_config.get("table_clearance_m", 0.01))

    def _safe_load(self, path: str, loader) -> dict[str, Any]:
        try:
            return loader(str(Path(path)))
        except ProjectError as exc:
            _log(self._logger, "warn", f"Falling back to default safety config: {exc}")
            return {}

    def validate_pose(self, pose: Pose3D) -> None:
        if pose.frame_id != "base":
            raise SafetyViolationError(f"Pose must be expressed in base frame, got {pose.frame_id!r}.")

        xyz = pose.position
        if np.any(xyz < self._xyz_min) or np.any(xyz > self._xyz_max):
            raise SafetyViolationError(
                f"Pose {xyz.tolist()} is outside workspace bounds {self._xyz_min.tolist()} - {self._xyz_max.tolist()}."
            )
        if float(xyz[2]) < self._table_height_m + self._table_margin_m:
            raise SafetyViolationError(
                f"Pose z={float(xyz[2]):.4f} is below table safety threshold "
                f"{self._table_height_m + self._table_margin_m:.4f}."
            )

    def validate_motion_plan(self, motion_plan: MotionPlan, scene_state: SceneState | None = None) -> None:
        table_floor = self._table_height_m if scene_state is None or scene_state.table_height_m is None else scene_state.table_height_m
        for waypoint in motion_plan.waypoints:
            self.validate_pose(waypoint.pose)
            if float(waypoint.pose.position[2]) < table_floor + self._table_margin_m:
                raise SafetyViolationError(
                    f"Waypoint {waypoint.extras.get('name', '<unnamed>')!r} goes below the table clearance."
                )

    def validate_pick_place_plan(self, plan: PickPlacePlan) -> None:
        self.validate_pose(plan.grasp_pose.pose)
        self.validate_pose(plan.place_pose)

        for subplan in [plan.approach_plan, plan.transfer_plan, plan.retreat_plan]:
            if subplan is not None:
                self.validate_motion_plan(subplan)

        for key in ("pick_motion", "place_motion"):
            subplan = plan.extras.get(key)
            if isinstance(subplan, MotionPlan):
                self.validate_motion_plan(subplan)
