from __future__ import annotations

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import argparse
import json
from typing import Any

from common.logger import ProjectLogger
from control_actuation.safety_guardrail import SafetyGuardrail
from perception.grasp_pose_estimator import TopDownGraspEstimator
from perception.scene_state import SceneStateBuilder
from scripts._backend_factory import available_backend_names, build_backend
from scripts._demo_support import ColorBlockDemoDetector, SyntheticFrameProvider, default_demo_meta
from semantic_interface.regex_parser import RegexCommandParser
from semantic_interface.target_resolver import TargetResolver
from sensing.replay_frame_provider import ReplayFrameProvider
from skill_planning.cartesian_waypoints import CartesianWaypointBuilder
from skill_planning.moveit_fallback import MoveItFallbackPlanner
from skill_planning.pick_place_plan import PickPlacePlanner
from skill_planning.place_pose_resolver import PlacePoseResolver


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pick-place planning on one scene.")
    parser.add_argument("--scene_dir", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--backend", default="demo", choices=["demo", *available_backend_names()])
    parser.add_argument("--config", default="config/detector.yaml")
    return parser.parse_args()


def _candidate_labels(scene_meta: dict[str, Any]) -> list[str]:
    labels = [
        str(scene_meta.get("expected_source_label", "cube")).strip().lower(),
        str(scene_meta.get("expected_target_label", "block")).strip().lower(),
        "cube",
        "block",
    ]
    return [label for index, label in enumerate(labels) if label and label not in labels[:index]]


def _build_frame_provider(args: argparse.Namespace, logger: ProjectLogger):
    if args.scene_dir:
        return ReplayFrameProvider(args.scene_dir, logger=logger)
    return SyntheticFrameProvider(meta=default_demo_meta())


def _build_detector(args: argparse.Namespace, logger: ProjectLogger, scene_meta: dict[str, Any]):
    if args.backend == "demo":
        detector = ColorBlockDemoDetector(scene_meta=scene_meta, logger=logger)
        detector.warmup()
        return detector
    detector = build_backend(args.backend, args.config, logger)
    detector.warmup()
    return detector


def main() -> None:
    args = _parse_args()
    logger = ProjectLogger("logs/pick_plan_demo")
    frame_provider = _build_frame_provider(args, logger)
    scene_meta = frame_provider.get_meta() if hasattr(frame_provider, "get_meta") else default_demo_meta()
    prompt = args.prompt or str(scene_meta.get("prompt", "put the red cube on the blue block"))
    detector = _build_detector(args, logger, scene_meta)

    frame = frame_provider.get_current_frame()
    detections = detector.detect(frame.rgb, _candidate_labels(scene_meta))
    scene_state = SceneStateBuilder(logger).build(frame, detections)
    parsed = RegexCommandParser(logger).parse(prompt)
    resolved = TargetResolver(logger).resolve(parsed, scene_state)

    planner = PickPlacePlanner(
        grasp_estimator=TopDownGraspEstimator("config/robot.yaml", "config/workspace.yaml", logger),
        place_pose_resolver=PlacePoseResolver("config/workspace.yaml", logger),
        waypoint_builder=CartesianWaypointBuilder("config/robot.yaml", logger),
        moveit_fallback=MoveItFallbackPlanner("config/robot.yaml", logger),
        logger=logger,
    )
    plan = planner.build(resolved, scene_state)
    SafetyGuardrail("config/workspace.yaml", "config/robot.yaml", logger).validate_pick_place_plan(plan)

    pick_motion = plan.extras["pick_motion"]
    place_motion = plan.extras["place_motion"]
    payload = {
        "pick_object_id": plan.pick_object_id,
        "pregrasp_pose_m": plan.grasp_pose.extras["pregrasp_pose"].position.tolist(),
        "grasp_pose_m": plan.grasp_pose.pose.position.tolist(),
        "retreat_pose_m": plan.grasp_pose.extras["retreat_pose"].position.tolist(),
        "place_pose_m": plan.place_pose.position.tolist(),
        "motion_plan_waypoints": [waypoint.name or waypoint.extras.get("name", "") for waypoint in pick_motion.waypoints]
        + [waypoint.name or waypoint.extras.get("name", "") for waypoint in place_motion.waypoints],
        "safety_guardrail_passed": True,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
