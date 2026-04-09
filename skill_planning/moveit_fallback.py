from __future__ import annotations

from pathlib import Path
from typing import Any

from common.config_loader import load_robot_config
from common.datatypes import MotionPlan, Pose3D, SceneState
from common.exceptions import ProjectError
from common.logger import ProjectLogger

try:
    import moveit  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    moveit = None  # type: ignore[assignment]


class MoveItFallbackPlanner:
    def __init__(self, robot_config_path: str, logger: ProjectLogger) -> None:
        self._robot_config_path = str(Path(robot_config_path))
        self._logger = logger
        self._config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        try:
            return load_robot_config(self._robot_config_path)
        except ProjectError as exc:
            self._logger.warn(f"Falling back to default MoveIt config: {exc}")
            return {}

    def plan_to_pose(self, start_joints, goal_pose: Pose3D, scene_state: SceneState) -> MotionPlan:
        _ = start_joints
        _ = goal_pose
        _ = scene_state
        if not bool(self._config.get("moveit_enabled", False)):
            raise RuntimeError("MoveIt fallback is disabled in robot config.")
        if moveit is None:
            raise RuntimeError("MoveIt is not installed or not available in the current environment.")
        raise RuntimeError("MoveIt fallback planning is configured but not implemented in this MVP.")
