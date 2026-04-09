from __future__ import annotations

import numpy as np

from common.datatypes import BBox2D, SceneObject, SceneState
from common.logger import ProjectLogger
from perception.grasp_pose_estimator import TopDownGraspEstimator
from semantic_interface.command_schema import (
    ActionType,
    ObjectQuery,
    ParsedCommand,
    ResolvedCommand,
    SpatialRelation,
)
from skill_planning.cartesian_waypoints import CartesianWaypointBuilder
from skill_planning.moveit_fallback import MoveItFallbackPlanner
from skill_planning.pick_place_plan import PickPlacePlanner
from skill_planning.place_pose_resolver import PlacePoseResolver


def main() -> None:
    logger = ProjectLogger("logs/pick_plan_demo")
    scene_state = SceneState(
        frame_timestamp=0.0,
        table_height_m=0.0,
        objects=[
            SceneObject(
                object_id="obj_1",
                label="block",
                bbox=BBox2D(10, 10, 40, 40),
                center_cam=np.array([0.3, 0.0, 0.55], dtype=np.float64),
                center_base=np.array([0.35, 0.0, 0.04], dtype=np.float64),
                confidence=0.95,
                color="red",
                shape="cube",
            ),
            SceneObject(
                object_id="obj_2",
                label="block",
                bbox=BBox2D(50, 10, 80, 40),
                center_cam=np.array([0.4, 0.1, 0.55], dtype=np.float64),
                center_base=np.array([0.45, 0.1, 0.04], dtype=np.float64),
                confidence=0.90,
                color="blue",
                shape="block",
            ),
        ],
    )
    parsed = ParsedCommand(
        action=ActionType.PICK_AND_PLACE,
        source=ObjectQuery(raw_text="red cube", category="cube", color="red", shape="cube"),
        target=ObjectQuery(raw_text="blue block", category="block", color="blue", shape="block"),
        relation=SpatialRelation.ON,
        grasp_mode="topdown",
        raw_prompt="put the red cube on the blue block",
    )
    resolved = ResolvedCommand(parsed=parsed, source_id="obj_1", target_id="obj_2")

    planner = PickPlacePlanner(
        grasp_estimator=TopDownGraspEstimator("config/robot.yaml", "config/workspace.yaml", logger),
        place_pose_resolver=PlacePoseResolver("config/workspace.yaml", logger),
        waypoint_builder=CartesianWaypointBuilder("config/robot.yaml", logger),
        moveit_fallback=MoveItFallbackPlanner("config/robot.yaml", logger),
        logger=logger,
    )
    plan = planner.build(resolved, scene_state)
    print(
        {
            "pick_object_id": plan.pick_object_id,
            "grasp_pose_m": plan.grasp_pose.pose.position.tolist(),
            "place_pose_m": plan.place_pose.position.tolist(),
            "pick_waypoints": [wp.extras.get("name") for wp in plan.extras["pick_motion"].waypoints],
            "place_waypoints": [wp.extras.get("name") for wp in plan.extras["place_motion"].waypoints],
        }
    )


if __name__ == "__main__":
    main()
