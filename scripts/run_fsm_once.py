from __future__ import annotations

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import argparse
import json
from typing import Any

import numpy as np

from common.datatypes import SceneObject, SceneState
from common.logger import ProjectLogger
from control_actuation.fake_lite6_adapter import FakeLite6Adapter
from control_actuation.gripper_executor import GripperExecutor
from control_actuation.ik_solver import IKSolver
from control_actuation.lite6_adapter import Lite6Adapter
from control_actuation.motion_executor import MotionExecutor
from control_actuation.safety_guardrail import SafetyGuardrail
from fsm.main_fsm import Prompt2PoseFSM
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
from verification.grasp_verify import GraspVerifier
from verification.place_verify import PlaceVerifier


class _SimulatedSceneRechecker:
    def __init__(
        self,
        frame_provider: Any,
        detector: Any,
        scene_builder: SceneStateBuilder,
        logger: ProjectLogger,
        fake_robot: FakeLite6Adapter | None = None,
    ) -> None:
        self._frame_provider = frame_provider
        self._detector = detector
        self._scene_builder = scene_builder
        self._logger = logger
        self._fake_robot = fake_robot
        self._resolved_cmd = None
        self._plan = None
        self._scene_before: SceneState | None = None
        self._reacquire_count = 0

    def set_resolved_context(self, resolved_cmd, scene_before: SceneState) -> None:
        self._resolved_cmd = resolved_cmd
        self._scene_before = scene_before
        self._reacquire_count = 0

    def set_plan_context(self, resolved_cmd, plan, scene_before: SceneState) -> None:
        self._resolved_cmd = resolved_cmd
        self._plan = plan
        self._scene_before = scene_before

    def _real_reacquire(self, candidate_labels: list[str]) -> SceneState:
        frame = self._frame_provider.get_current_frame()
        detections = self._detector.detect(frame.rgb, candidate_labels)
        return self._scene_builder.build(frame, detections)

    def _clone_object_with_new_center(self, obj: SceneObject, center_base: np.ndarray) -> SceneObject:
        delta = np.asarray(center_base, dtype=np.float64) - obj.center_base
        return SceneObject(
            object_id=obj.object_id,
            label=obj.label,
            bbox=obj.bbox,
            center_cam=obj.center_cam + delta,
            center_base=np.asarray(center_base, dtype=np.float64),
            confidence=obj.confidence,
            color=obj.color,
            shape=obj.shape,
            is_graspable=obj.is_graspable,
            extras=dict(obj.extras),
        )

    def _simulate_post_pick_scene(self) -> SceneState:
        assert self._scene_before is not None
        assert self._resolved_cmd is not None
        gripper_state = self._fake_robot.get_gripper_state() if self._fake_robot is not None else {}
        source_id = self._resolved_cmd.source_id
        keep_source = not bool(gripper_state.get("is_holding"))
        objects = [
            obj
            for obj in self._scene_before.objects
            if keep_source or obj.object_id != source_id
        ]
        return SceneState(
            frame_timestamp=self._scene_before.frame_timestamp + 1.0,
            objects=objects,
            table_height_m=self._scene_before.table_height_m,
            extras={"simulated_stage": "post_pick"},
        )

    def _simulate_post_place_scene(self) -> SceneState:
        assert self._scene_before is not None
        assert self._resolved_cmd is not None
        assert self._plan is not None
        source_before = self._scene_before.get_object_by_id(self._resolved_cmd.source_id)
        if source_before is None:
            raise RuntimeError("Cannot simulate post-place scene because source object is missing.")

        release_pose = None
        if self._fake_robot is not None:
            release_pose = self._fake_robot.get_last_release_pose()
        if release_pose is None:
            release_pose = self._plan.place_pose

        objects = [obj for obj in self._scene_before.objects if obj.object_id != self._resolved_cmd.source_id]
        objects.append(self._clone_object_with_new_center(source_before, release_pose.position))
        return SceneState(
            frame_timestamp=self._scene_before.frame_timestamp + 2.0,
            objects=objects,
            table_height_m=self._scene_before.table_height_m,
            extras={"simulated_stage": "post_place"},
        )

    def reacquire_scene(self, candidate_labels: list[str]) -> SceneState:
        if self._fake_robot is None or self._resolved_cmd is None or self._scene_before is None or self._plan is None:
            return self._real_reacquire(candidate_labels)

        self._reacquire_count += 1
        if self._reacquire_count == 1:
            return self._simulate_post_pick_scene()
        return self._simulate_post_place_scene()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one prompt-to-pose FSM cycle.")
    parser.add_argument("prompt_arg", nargs="?", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--scene_dir", default=None)
    parser.add_argument("--use_fake_robot", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--backend",
        default="demo",
        choices=["demo", *available_backend_names()],
        help="Detector backend to use. 'demo' uses a lightweight color-based detector for dry runs.",
    )
    parser.add_argument("--config", default="config/detector.yaml")
    return parser.parse_args()


def _resolve_prompt(args: argparse.Namespace, scene_meta: dict[str, Any]) -> str:
    if isinstance(args.prompt, str) and args.prompt.strip():
        return args.prompt.strip()
    if isinstance(args.prompt_arg, str) and args.prompt_arg.strip():
        return args.prompt_arg.strip()
    meta_prompt = scene_meta.get("prompt")
    if isinstance(meta_prompt, str) and meta_prompt.strip():
        return meta_prompt.strip()
    return "put the red cube on the blue block"


def _build_frame_provider(args: argparse.Namespace, logger: ProjectLogger):
    if args.scene_dir:
        return ReplayFrameProvider(args.scene_dir, logger=logger)
    return SyntheticFrameProvider(meta=default_demo_meta())


def _build_detector(args: argparse.Namespace, logger: ProjectLogger, scene_meta: dict[str, Any]) -> Any:
    if args.backend == "demo" or args.synthetic:
        detector = ColorBlockDemoDetector(scene_meta=scene_meta, logger=logger)
        detector.warmup()
        return detector

    detector = build_backend(args.backend, args.config, logger)
    detector.warmup()
    return detector


def _build_robot_adapter(args: argparse.Namespace, logger: ProjectLogger) -> Any:
    if args.use_fake_robot:
        adapter = FakeLite6Adapter(logger=logger)
    else:
        adapter = Lite6Adapter("config/robot.yaml", logger)
    adapter.connect()
    return adapter


def main() -> None:
    args = _parse_args()
    logger = ProjectLogger("logs/fsm_once")

    frame_provider = _build_frame_provider(args, logger)
    scene_meta = frame_provider.get_meta() if hasattr(frame_provider, "get_meta") else default_demo_meta()
    prompt = _resolve_prompt(args, scene_meta)
    detector = _build_detector(args, logger, scene_meta)
    robot_adapter = _build_robot_adapter(args, logger)

    try:
        scene_builder = SceneStateBuilder(logger)
        scene_rechecker = _SimulatedSceneRechecker(
            frame_provider=frame_provider,
            detector=detector,
            scene_builder=scene_builder,
            logger=logger,
            fake_robot=robot_adapter if isinstance(robot_adapter, FakeLite6Adapter) else None,
        )

        fsm = Prompt2PoseFSM(
            regex_parser=RegexCommandParser(logger),
            frame_provider=frame_provider,
            detector=detector,
            scene_builder=scene_builder,
            target_resolver=TargetResolver(logger),
            pick_place_planner=PickPlacePlanner(
                grasp_estimator=TopDownGraspEstimator("config/robot.yaml", "config/workspace.yaml", logger),
                place_pose_resolver=PlacePoseResolver("config/workspace.yaml", logger),
                waypoint_builder=CartesianWaypointBuilder("config/robot.yaml", logger),
                moveit_fallback=MoveItFallbackPlanner("config/robot.yaml", logger),
                logger=logger,
            ),
            safety_guardrail=SafetyGuardrail("config/workspace.yaml", "config/robot.yaml", logger),
            motion_executor=MotionExecutor(robot_adapter, IKSolver("config/robot.yaml", logger), logger),
            gripper_executor=GripperExecutor(robot_adapter, logger),
            grasp_verifier=GraspVerifier(logger),
            place_verifier=PlaceVerifier(logger),
            scene_rechecker=scene_rechecker,
            logger=logger,
        )
        result = fsm.run_once(prompt)
        result["prompt"] = prompt
        result["backend"] = args.backend
        result["scene_dir"] = args.scene_dir
        result["use_fake_robot"] = bool(args.use_fake_robot)
        if isinstance(robot_adapter, FakeLite6Adapter):
            result["robot_command_log"] = robot_adapter.get_command_log()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        if hasattr(robot_adapter, "disconnect"):
            robot_adapter.disconnect()


if __name__ == "__main__":
    main()
