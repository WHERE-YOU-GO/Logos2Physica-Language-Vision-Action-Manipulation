from __future__ import annotations

import argparse
import numpy as np

from common.datatypes import BBox2D, CameraIntrinsics, Detection2D, RGBDFrame
from common.logger import ProjectLogger
from control_actuation.gripper_executor import GripperExecutor
from control_actuation.ik_solver import IKSolver
from control_actuation.motion_executor import MotionExecutor
from control_actuation.safety_guardrail import SafetyGuardrail
from fsm.main_fsm import Prompt2PoseFSM
from perception.grasp_pose_estimator import TopDownGraspEstimator
from perception.scene_state import SceneStateBuilder
from perception.yolo_world_backend import YOLOWorldBackend
from semantic_interface.regex_parser import RegexCommandParser
from semantic_interface.target_resolver import TargetResolver
from skill_planning.cartesian_waypoints import CartesianWaypointBuilder
from skill_planning.moveit_fallback import MoveItFallbackPlanner
from skill_planning.pick_place_plan import PickPlacePlanner
from skill_planning.place_pose_resolver import PlacePoseResolver
from verification.grasp_verify import GraspVerifier
from verification.place_verify import PlaceVerifier
from verification.scene_recheck import SceneRechecker


class _SyntheticFrameProvider:
    def get_current_frame(self) -> RGBDFrame:
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[200:260, 280:340] = np.array([255, 0, 0], dtype=np.uint8)
        rgb[220:280, 360:420] = np.array([0, 0, 255], dtype=np.uint8)
        depth = np.ones((480, 640), dtype=np.float32) * 0.55
        intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
        return RGBDFrame(
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            T_base_cam=np.eye(4, dtype=np.float64),
            timestamp=0.0,
        )


class _SyntheticDetector:
    def detect(self, rgb, candidate_labels: list[str]) -> list[Detection2D]:
        _ = rgb
        _ = candidate_labels
        return [
            Detection2D(label="block", score=0.95, bbox=BBox2D(280, 200, 340, 260), extras={"category": "cube"}),
            Detection2D(label="block", score=0.91, bbox=BBox2D(360, 220, 420, 280), extras={"category": "block"}),
        ]


class _NoopRobotAdapter:
    def move_linear(self, pose, speed: float) -> None:
        _ = pose
        _ = speed

    def move_joints(self, joints, speed: float) -> None:
        _ = joints
        _ = speed

    def open_gripper(self) -> None:
        return None

    def close_gripper(self) -> None:
        return None

    def set_gripper_width(self, width_m: float) -> None:
        _ = width_m

    def get_gripper_state(self) -> dict:
        return {"width_m": 0.03, "is_holding": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="put the red cube on the blue block")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    logger = ProjectLogger("logs/fsm_once")
    frame_provider = _SyntheticFrameProvider()
    detector = _SyntheticDetector()

    if not args.synthetic:
        try:
            detector = YOLOWorldBackend("config/detector.yaml", logger)
            detector.warmup()
        except Exception as exc:
            raise SystemExit(f"Real detector setup failed. Use --synthetic for dry-run mode. Error: {exc}")

    robot_adapter = _NoopRobotAdapter()
    fsm = Prompt2PoseFSM(
        regex_parser=RegexCommandParser(logger),
        frame_provider=frame_provider,
        detector=detector,
        scene_builder=SceneStateBuilder(logger),
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
        scene_rechecker=SceneRechecker(frame_provider, detector, SceneStateBuilder(logger), logger),
        logger=logger,
    )
    print(fsm.run_once(args.prompt))


if __name__ == "__main__":
    main()
