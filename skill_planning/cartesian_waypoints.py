from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.config_loader import load_robot_config
from common.datatypes import GraspPose, MotionPlan, Pose3D, Waypoint
from common.exceptions import PlanningError, ProjectError
from common.logger import ProjectLogger


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


class CartesianWaypointBuilder:
    def __init__(self, robot_config_path: str, logger: ProjectLogger) -> None:
        self._logger = logger
        self._robot_config = self._safe_load(robot_config_path)
        self._pick_speed = float(self._robot_config.get("pick_speed_scale", 0.2))
        self._place_speed = float(self._robot_config.get("place_speed_scale", 0.25))
        self._place_approach_height_m = float(self._robot_config.get("place_approach_height_m", 0.10))
        self._settle_time_s = float(self._robot_config.get("gripper_settle_time_s", 0.2))

    def _safe_load(self, path: str) -> dict[str, Any]:
        try:
            return load_robot_config(str(Path(path)))
        except ProjectError as exc:
            _log(self._logger, "warn", f"Falling back to default waypoint settings: {exc}")
            return {}

    def _offset_pose(self, pose: Pose3D, delta_xyz: np.ndarray) -> Pose3D:
        return Pose3D(
            position=pose.position + np.asarray(delta_xyz, dtype=np.float64),
            quaternion=pose.quaternion.copy(),
            frame_id=pose.frame_id,
        )

    def _make_waypoint(
        self,
        name: str,
        pose: Pose3D,
        speed: float,
        gripper_open: bool | None = None,
        hold_time_s: float = 0.0,
    ) -> Waypoint:
        return Waypoint(
            name=name,
            pose=pose,
            speed_scale=float(speed),
            gripper_open=gripper_open,
            hold_time_s=float(hold_time_s),
            extras={"name": name, "speed": float(speed)},
        )

    def build_pick_motion(self, grasp: GraspPose) -> MotionPlan:
        pregrasp_pose = grasp.extras.get("pregrasp_pose")
        grasp_pose = grasp.extras.get("grasp_pose", grasp.pose)
        retreat_pose = grasp.extras.get("retreat_pose")

        if not isinstance(grasp_pose, Pose3D):
            raise PlanningError("Grasp pose payload is invalid.")
        if pregrasp_pose is None:
            pregrasp_pose = self._offset_pose(
                grasp_pose,
                -grasp.approach_vector * grasp.pregrasp_offset_m,
            )
        if retreat_pose is None:
            retreat_pose = self._offset_pose(
                grasp_pose,
                -grasp.approach_vector * max(0.08, grasp.pregrasp_offset_m),
            )
        if not isinstance(pregrasp_pose, Pose3D) or not isinstance(retreat_pose, Pose3D):
            raise PlanningError("Pregrasp or retreat pose payload is invalid.")

        waypoints = [
            self._make_waypoint("pick_pregrasp", pregrasp_pose, self._pick_speed, gripper_open=True),
            self._make_waypoint(
                "pick_grasp",
                grasp_pose,
                self._pick_speed * 0.75,
                gripper_open=False,
                hold_time_s=self._settle_time_s,
            ),
            self._make_waypoint("pick_retreat", retreat_pose, self._pick_speed),
        ]
        return MotionPlan(
            waypoints=waypoints,
            frame_id=grasp_pose.frame_id,
            extras={"plan_type": "pick"},
        )

    def build_place_motion(self, current_retreat_pose: Pose3D, place_pose: Pose3D) -> MotionPlan:
        if current_retreat_pose.frame_id != place_pose.frame_id:
            raise PlanningError("Place motion requires poses in the same frame.")

        place_approach_pose = self._offset_pose(place_pose, np.array([0.0, 0.0, self._place_approach_height_m]))
        place_retreat_pose = self._offset_pose(place_pose, np.array([0.0, 0.0, self._place_approach_height_m]))

        waypoints = [
            self._make_waypoint("place_transfer_start", current_retreat_pose, self._place_speed),
            self._make_waypoint("place_approach", place_approach_pose, self._place_speed),
            self._make_waypoint(
                "place_release",
                place_pose,
                self._place_speed * 0.75,
                gripper_open=True,
                hold_time_s=self._settle_time_s,
            ),
            self._make_waypoint("place_retreat", place_retreat_pose, self._place_speed),
        ]
        return MotionPlan(
            waypoints=waypoints,
            frame_id=place_pose.frame_id,
            extras={"plan_type": "place"},
        )
