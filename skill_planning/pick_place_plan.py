from __future__ import annotations

from typing import Any

from common.datatypes import MotionPlan, PickPlacePlan, Pose3D, SceneState
from common.exceptions import PlanningError
from common.logger import ProjectLogger
from semantic_interface.command_schema import ResolvedCommand


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


class PickPlacePlanner:
    def __init__(
        self,
        grasp_estimator,
        place_pose_resolver,
        waypoint_builder,
        moveit_fallback,
        logger: ProjectLogger,
    ) -> None:
        self._grasp_estimator = grasp_estimator
        self._place_pose_resolver = place_pose_resolver
        self._waypoint_builder = waypoint_builder
        self._moveit_fallback = moveit_fallback
        self._logger = logger

    def build(self, resolved_cmd: ResolvedCommand, scene_state: SceneState) -> PickPlacePlan:
        source_obj = scene_state.get_object_by_id(resolved_cmd.source_id)
        if source_obj is None:
            raise PlanningError(f"Source object {resolved_cmd.source_id!r} is missing from scene state.")

        grasp = self._grasp_estimator.estimate(source_obj, scene_state)
        place_pose = self._place_pose_resolver.resolve(resolved_cmd, scene_state)
        try:
            pick_motion = self._waypoint_builder.build_pick_motion(grasp)
        except Exception as exc:
            if self._moveit_fallback is None:
                raise PlanningError("Failed to build pick motion and no MoveIt fallback is configured.") from exc
            _log(self._logger, "warn", f"Cartesian pick motion failed, trying MoveIt fallback: {exc}")
            pick_motion = self._moveit_fallback.plan_to_pose(
                start_joints=None,
                goal_pose=grasp.pose,
                scene_state=scene_state,
            )

        retreat_pose = grasp.extras.get("retreat_pose")
        if not isinstance(retreat_pose, Pose3D):
            if pick_motion.is_empty():
                raise PlanningError("Pick motion is empty, cannot build place motion.")
            retreat_pose = pick_motion.waypoints[-1].pose
        try:
            place_motion = self._waypoint_builder.build_place_motion(retreat_pose, place_pose)
        except Exception as exc:
            if self._moveit_fallback is None:
                raise PlanningError("Failed to build place motion and no MoveIt fallback is configured.") from exc
            _log(self._logger, "warn", f"Cartesian place motion failed, trying MoveIt fallback: {exc}")
            place_motion = self._moveit_fallback.plan_to_pose(
                start_joints=None,
                goal_pose=place_pose,
                scene_state=scene_state,
            )

        final_retreat = MotionPlan(
            waypoints=[place_motion.waypoints[-1]],
            frame_id=place_motion.frame_id,
            extras={"plan_type": "post_place_retreat"},
        )

        plan = PickPlacePlan(
            pick_object_id=resolved_cmd.source_id,
            grasp_pose=grasp,
            place_pose=place_pose,
            approach_plan=pick_motion,
            transfer_plan=place_motion,
            retreat_plan=final_retreat,
            extras={
                "pick_motion": pick_motion,
                "place_motion": place_motion,
                "pick_grasp": grasp,
                "resolved_target_id": resolved_cmd.target_id,
            },
        )
        _log(self._logger, "info", f"Built pick-place plan for {resolved_cmd.source_id!r}.")
        return plan
