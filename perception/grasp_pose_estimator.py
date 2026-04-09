from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.config_loader import load_robot_config, load_workspace_config
from common.datatypes import GraspPose, Pose3D, SceneObject, SceneState
from common.exceptions import PlanningError, ProjectError
from common.geometry import make_topdown_quaternion
from common.logger import ProjectLogger


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


class TopDownGraspEstimator:
    def __init__(self, robot_config_path: str, workspace_config_path: str, logger: ProjectLogger) -> None:
        self._logger = logger
        self._robot_config = self._safe_load(robot_config_path, load_robot_config)
        self._workspace_config = self._safe_load(workspace_config_path, load_workspace_config)

        self._approach_height_m = float(self._robot_config.get("approach_height_m", 0.08))
        self._retreat_height_m = float(self._robot_config.get("retreat_height_m", 0.10))
        self._default_gripper_width_m = float(self._robot_config.get("default_gripper_width_m", 0.045))
        self._object_half_height_m = float(self._workspace_config.get("default_object_half_height_m", 0.02))
        self._table_height_m = float(self._workspace_config.get("table_height_m", 0.0))

    def _safe_load(self, path: str, loader) -> dict[str, Any]:
        try:
            return loader(str(Path(path)))
        except ProjectError as exc:
            _log(self._logger, "warn", f"Falling back to default grasp settings: {exc}")
            return {}

    def estimate(self, obj: SceneObject, scene: SceneState) -> GraspPose:
        if not obj.is_graspable:
            raise PlanningError(f"Object {obj.object_id!r} is not marked as graspable.")

        yaw_rad = 0.0
        quaternion = make_topdown_quaternion(yaw_rad)

        table_height = scene.table_height_m if scene.table_height_m is not None else self._table_height_m
        grasp_z = max(float(obj.center_base[2]), table_height + self._object_half_height_m)
        grasp_position = np.array(
            [float(obj.center_base[0]), float(obj.center_base[1]), grasp_z],
            dtype=np.float64,
        )

        pregrasp_position = grasp_position.copy()
        pregrasp_position[2] += self._approach_height_m

        retreat_position = grasp_position.copy()
        retreat_position[2] += self._retreat_height_m

        grasp_pose = Pose3D(position=grasp_position, quaternion=quaternion, frame_id="base")
        pregrasp_pose = Pose3D(position=pregrasp_position, quaternion=quaternion, frame_id="base")
        retreat_pose = Pose3D(position=retreat_position, quaternion=quaternion, frame_id="base")

        return GraspPose(
            pose=grasp_pose,
            approach_vector=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            pregrasp_offset_m=self._approach_height_m,
            grasp_width_m=self._default_gripper_width_m,
            score=float(obj.confidence),
            extras={
                "pregrasp_pose": pregrasp_pose,
                "grasp_pose": grasp_pose,
                "retreat_pose": retreat_pose,
                "recommended_gripper_width_m": self._default_gripper_width_m,
                "yaw_rad": yaw_rad,
                "source_object_id": obj.object_id,
            },
        )
