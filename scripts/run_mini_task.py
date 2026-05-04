from __future__ import annotations

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root_on_path

REPO_ROOT = ensure_repo_root_on_path()

import argparse
import itertools
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pupil_apriltags import Detector
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from common.config_loader import load_robot_config


DEFAULT_ROBOT_IP = "192.168.1.158"
GRIPPER_LENGTH_MM = 0.067 * 1000.0

CALIBRATION_TAG_FAMILY = "tag36h11"
CALIBRATION_TAG_SIZE_M = 0.08
TARGET_TAG_SIZE_M = 0.020
DUPLICATE_ASSIGNMENT_REPORT_PATH = "logs/duplicate_assignment_report.json"
DUPLICATE_ASSIGNMENT_PREVIEW_PATH = "logs/run_mini_task_duplicate_assignment_preview.png"
DUPLICATE_POSE_PLAN_DIR = "logs/duplicate_pose_plans"
PRESET_LAYOUT_ASSIGNMENT_REPORT_PATH = "logs/preset_layout_assignment_report.json"
PRESET_LAYOUT_PREVIEW_PATH = "logs/run_mini_task_preset_layout_preview.png"
PRESET_POSE_PLAN_DIR = "logs/preset_pose_plans"
MAX_DUPLICATE_GROUP_COUNT = 10


def _argv_option_value(flag_name: str) -> str | None:
    value: str | None = None
    prefix = f"{flag_name}="
    tokens = sys.argv[1:]
    for index, token in enumerate(tokens):
        if token == flag_name and index + 1 < len(tokens):
            value = tokens[index + 1]
        elif token.startswith(prefix):
            value = token[len(prefix) :]
    return value


# Tag centers in the robot/base table frame from the older working checkpoint0.
TAG_CENTER_COORDINATES_M = {
    0: np.array([0.38, 0.40], dtype=np.float64),
    1: np.array([0.38, -0.40], dtype=np.float64),
    2: np.array([0.00, 0.40], dtype=np.float64),
    3: np.array([0.00, -0.40], dtype=np.float64),
}

CUBE_SIZE_M = 0.025
CUBE_DIAGONAL_M = CUBE_SIZE_M * math.sqrt(3.0)
POINT_CLOUD_SCALE_TO_M = 1e-3
GRASP_Z_BIAS_M = 0.01
TARGET_TAG_Z_EXPECTED_BELOW_TABLE_M = 0.03
TARGET_TAG_Z_EXPECTED_ABOVE_TABLE_M = 0.05

MIN_POINTS_FIT = 120
MIN_POINTS_AFTER_OUTLIER = 60

TCP_SPEED_MM_S = 100.0
APPROACH_HEIGHT_M = 0.12
RETREAT_HEIGHT_M = 0.12
GRIPPER_SETTLE_S = 1.5

# Loose guardrail for this lab table. It is intentionally local to this script.
WORKSPACE_MIN_M = np.array([0.08, -0.45, 0.00], dtype=np.float64)
WORKSPACE_MAX_M = np.array([0.78, 0.45, 0.45], dtype=np.float64)

MINI_TASK_CONTACT_MIN_MM = np.array([120.0, -250.0, 15.0], dtype=np.float64)
MINI_TASK_CONTACT_MAX_MM = np.array([360.0, 250.0, 80.0], dtype=np.float64)
MINI_TASK_STAGING_MIN_MM = np.array([40.0, -350.0, 50.0], dtype=np.float64)
MINI_TASK_STAGING_MAX_MM = np.array([420.0, 500.0, 180.0], dtype=np.float64)
MINI_TASK_STAGING_Z_MM = 160.0
FIRST_MOVE_STAGING_DISTANCE_MM = 100.0
FORWARD_STEP_MM = 50.0
FORWARD_STEPS = 1
FORWARD_STAGE_SPEED_MM_S = 30.0


class ZedCamera:
    """Small copy of the older working ZED camera path with lazy pyzed import."""

    def __init__(self, resolution: str = "HD2K", fps: int = 15) -> None:
        import pyzed.sl as sl

        self._sl = sl
        self._zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.enable_image_validity_check = True
        init_params.camera_resolution = self._resolution_from_name(resolution)
        init_params.camera_fps = int(fps)
        init_params.coordinate_units = sl.UNIT.MILLIMETER

        self._runtime_parameters = sl.RuntimeParameters()
        err = self._zed.open(init_params)
        if err > sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err!r}")

        self._zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
        self._zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)
        for _ in range(50):
            self._zed.grab(sl.RuntimeParameters())

        camera_info = self._zed.get_camera_information()
        left_camera_param = camera_info.camera_configuration.calibration_parameters.left_cam
        self._camera_intrinsic = np.eye(3, dtype=np.float64)
        self._camera_intrinsic[0, 0] = left_camera_param.fx
        self._camera_intrinsic[1, 1] = left_camera_param.fy
        self._camera_intrinsic[0, 2] = left_camera_param.cx
        self._camera_intrinsic[1, 2] = left_camera_param.cy

        self._image_mat = sl.Mat()
        self._measure_xyz = sl.Mat()
        self._image: np.ndarray | None = None
        self._point_cloud: np.ndarray | None = None

        self._running = True
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

        deadline = time.monotonic() + 10.0
        while self._image is None or self._point_cloud is None:
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out waiting for ZED image and point cloud.")
            time.sleep(0.1)

    def _resolution_from_name(self, resolution: str):
        sl = self._sl
        lookup = {
            "HD2K": sl.RESOLUTION.HD2K,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720,
            "VGA": sl.RESOLUTION.VGA,
        }
        return lookup.get(str(resolution).upper(), sl.RESOLUTION.HD2K)

    def _update(self) -> None:
        sl = self._sl
        while self._running:
            if self._zed.grab(self._runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                self._zed.retrieve_image(self._image_mat, sl.VIEW.LEFT)
                self._zed.retrieve_measure(self._measure_xyz, sl.MEASURE.XYZ)
                with self._lock:
                    self._image = self._image_mat.get_data().copy()
                    self._point_cloud = self._measure_xyz.get_data().copy()
            else:
                time.sleep(0.01)

    def close(self) -> None:
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=2.0)
        self._zed.close()

    @property
    def image(self) -> np.ndarray | None:
        with self._lock:
            return self._image.copy() if self._image is not None else None

    @property
    def point_cloud(self) -> np.ndarray | None:
        with self._lock:
            return self._point_cloud.copy() if self._point_cloud is not None else None

    @property
    def camera_intrinsic(self) -> np.ndarray:
        return self._camera_intrinsic.copy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal live ZED + AprilTag + Lite6 cube-to-tag lab task.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Run instructions:\n"
            "Checkpoint8-style dry run, place back down:\n"
            "    python scripts/run_mini_task.py --execution_backend checkpoint8_style --cube_prompt \"red cube\" --dry_run --no_gui\n\n"
            "Checkpoint8-style real run, place back down:\n"
            "    python scripts/run_mini_task.py --execution_backend checkpoint8_style --cube_prompt \"red cube\" --no_gui\n\n"
            "Checkpoint8-style real run, place to tag 7:\n"
            "    python scripts/run_mini_task.py --execution_backend checkpoint8_style --cube_prompt \"red cube\" --place_to_tag --target_tag_id 7 --target_tag_size_m 0.020 --no_gui\n\n"
            "Checkpoint8-style multi-cube real run:\n"
            "    python scripts/run_mini_task.py --execution_backend checkpoint8_style --multi_place_to_tags --cube_tag_map \"red cube:7,green cube:8,blue cube:9\" --target_tag_size_m 0.020 --no_gui\n\n"
            "Warning: the legacy backend uses custom mini_task motion and is not recommended for real robot execution."
        ),
    )
    parser.add_argument(
        "--execution_backend",
        choices=["checkpoint8_style", "legacy"],
        default="checkpoint8_style",
        help="Execution backend. checkpoint8_style is the recommended real-robot path.",
    )
    parser.add_argument("--tag_id", type=int, default=0, help="AprilTag ID used as the placement target.")
    parser.add_argument("--target_tag_id", type=int, default=None, help="AprilTag ID for --place_to_tag. Defaults to --tag_id.")
    parser.add_argument("--robot_ip", default=None, help="Override Lite6 IP. Defaults to config/robot.yaml.")
    parser.add_argument("--robot_config", default="config/robot.yaml", help="Robot config path.")
    parser.add_argument("--cube_prompt", default="red cube", help="Source cube prompt for checkpoint8_style detection.")
    parser.add_argument("--place_to_tag", action="store_true", help="Place the cube on a target AprilTag in checkpoint8_style mode.")
    parser.add_argument("--multi_place_to_tags", action="store_true", help="Place multiple prompted cubes onto mapped AprilTags in checkpoint8_style mode.")
    parser.add_argument(
        "--duplicate_aware_multi_place",
        action="store_true",
        help="Detect duplicate cube/tag instances, assign them by distance, and execute pose-plan pairs.",
    )
    parser.add_argument(
        "--preset_layout_place",
        action="store_true",
        help="Detect duplicate cube instances and place them into preset robot-base-frame layout slots.",
    )
    parser.add_argument(
        "--preset_place_layout_json",
        default=None,
        help="Preset placement layout JSON file for --preset_layout_place.",
    )
    parser.add_argument(
        "--preset_cube_counts",
        default=None,
        help='Preset cube counts, for example "red cube:2,green cube:2,blue cube:2".',
    )
    parser.add_argument(
        "--preset_cube_slot_map",
        default=None,
        help='Preset slot map, for example "red cube:1,2;green cube:3,4;blue cube:5,6".',
    )
    parser.add_argument(
        "--preset_assignment_metric",
        choices=["nearest"],
        default="nearest",
        help="Preset layout assignment metric.",
    )
    parser.add_argument(
        "--preset_use_slot_yaw",
        action="store_true",
        help="Use slot yaw_deg for preset place pose instead of preserving source cube yaw.",
    )
    parser.add_argument(
        "--allow_preset_slots_outside_workspace",
        action="store_true",
        help="Allow preset layout slots outside the conservative contact workspace bounds.",
    )
    parser.add_argument(
        "--multi_subprocess",
        action="store_true",
        help="Run each --multi_place_to_tags pair in an isolated single-pair checkpoint8_style subprocess.",
    )
    parser.add_argument(
        "--continue_on_pair_failure",
        action="store_true",
        help="Continue multi subprocess execution after a failed pair instead of aborting immediately.",
    )
    parser.add_argument(
        "--auto_confirm",
        action="store_true",
        help="Skip interactive confirmation after successful detection/planning.",
    )
    parser.add_argument(
        "--cube_tag_map",
        default=None,
        help='Comma-separated multi-place mapping, for example "red cube:7,green cube:8,blue cube:9".',
    )
    parser.add_argument(
        "--duplicate_cube_tag_map",
        default=None,
        help='Duplicate-aware map, for example "red cube:6:2,green cube:8:2,blue cube:7:2".',
    )
    parser.add_argument(
        "--assignment_metric",
        choices=["nearest"],
        default="nearest",
        help="Duplicate-aware assignment metric.",
    )
    parser.add_argument(
        "--assignment_space",
        choices=["xy"],
        default="xy",
        help="Duplicate-aware assignment coordinate space.",
    )
    parser.add_argument(
        "--max_assignment_distance_m",
        type=float,
        default=0.30,
        help="Maximum allowed duplicate-aware cube/tag XY assignment distance in meters.",
    )
    parser.add_argument("--candidate_x_min_m", type=float, default=0.05)
    parser.add_argument("--candidate_x_max_m", type=float, default=0.80)
    parser.add_argument("--candidate_y_min_m", type=float, default=-0.45)
    parser.add_argument("--candidate_y_max_m", type=float, default=0.45)
    parser.add_argument("--candidate_z_min_m", type=float, default=0.00)
    parser.add_argument("--candidate_z_max_m", type=float, default=0.08)
    parser.add_argument("--candidate_min_area_px", type=int, default=500)
    parser.add_argument("--candidate_merge_distance_m", type=float, default=0.035)
    parser.add_argument("--candidate_min_extent_m", type=float, default=0.012)
    parser.add_argument("--candidate_max_extent_m", type=float, default=0.040)
    parser.add_argument("--candidate_merge_prompts", default="green cube")
    parser.add_argument("--green_candidate_min_area_px", type=int, default=1200)
    parser.add_argument("--red_candidate_min_area_px", type=int, default=500)
    parser.add_argument("--blue_candidate_min_area_px", type=int, default=500)
    parser.add_argument(
        "--debug_duplicate_candidates_only",
        action="store_true",
        help="Run duplicate-aware perception, filtering, merge, and assignment diagnostics without robot motion.",
    )
    parser.add_argument(
        "--save_assignment_report",
        action="store_true",
        default=True,
        help=f"Save duplicate-aware assignment JSON report to {DUPLICATE_ASSIGNMENT_REPORT_PATH}.",
    )
    parser.add_argument(
        "--execute_pose_plan_json",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--execute_pose_plan_refine_only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--refine_pose_plan_json",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--refined_pose_plan_output_json",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no_pose_plan_refine",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--pose_plan_refine_before_execute",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--pose_plan_refine_radius_m", type=float, default=0.060, help=argparse.SUPPRESS)
    parser.add_argument("--pose_plan_refine_tag_radius_m", type=float, default=0.080, help=argparse.SUPPRESS)
    parser.add_argument("--min_after_grasp_z_mm", type=float, default=100.0, help=argparse.SUPPRESS)
    parser.add_argument("--recover_robot", action="store_true", help="Clear Lite6 warn/error and re-enable motion without moving home.")
    parser.add_argument("--home_only", action="store_true", help="Run only the checkpoint8-style move_gohome diagnostic.")
    parser.add_argument("--cube_color", choices=["auto", "red", "green", "blue"], default="auto")
    parser.add_argument("--cube_size_m", type=float, default=CUBE_SIZE_M)
    parser.add_argument("--table_z_m", type=float, default=0.0)
    parser.add_argument("--target_tag_size_m", type=float, default=TARGET_TAG_SIZE_M)
    parser.add_argument("--calibration_tag_size_m", type=float, default=CALIBRATION_TAG_SIZE_M)
    parser.add_argument("--zed_resolution", default="HD2K", choices=["HD2K", "HD1080", "HD720", "VGA"])
    parser.add_argument("--zed_fps", type=int, default=15)
    parser.add_argument("--point_cloud_scale", type=float, default=POINT_CLOUD_SCALE_TO_M)
    parser.add_argument("--approach_height_m", type=float, default=APPROACH_HEIGHT_M)
    parser.add_argument("--retreat_height_m", type=float, default=RETREAT_HEIGHT_M)
    parser.add_argument("--speed_mm_s", type=float, default=TCP_SPEED_MM_S)
    parser.add_argument("--gripper_settle_s", type=float, default=GRIPPER_SETTLE_S)
    parser.add_argument("--place_x_offset_m", type=float, default=0.0)
    parser.add_argument("--place_y_offset_m", type=float, default=0.0)
    parser.add_argument(
        "--motion_profile",
        choices=["checkpoint", "experimental"],
        default="checkpoint",
        help="Motion profile for real execution. checkpoint skips all first-move staging.",
    )
    task_gate = parser.add_argument_group(
        "mini-task conservative execution gate",
        (
            "Enabled by default. Widening these bounds can increase collision/self-collision risk; "
            "tune with --dry_run first and attempt real execution only after the preview is confirmed safe."
        ),
    )
    task_gate.add_argument("--task_x_min_mm", type=float, default=float(MINI_TASK_CONTACT_MIN_MM[0]))
    task_gate.add_argument("--task_x_max_mm", type=float, default=float(MINI_TASK_CONTACT_MAX_MM[0]))
    task_gate.add_argument("--task_y_min_mm", type=float, default=float(MINI_TASK_CONTACT_MIN_MM[1]))
    task_gate.add_argument("--task_y_max_mm", type=float, default=float(MINI_TASK_CONTACT_MAX_MM[1]))
    task_gate.add_argument("--task_z_min_mm", type=float, default=float(MINI_TASK_CONTACT_MIN_MM[2]))
    task_gate.add_argument("--task_z_max_mm", type=float, default=float(MINI_TASK_CONTACT_MAX_MM[2]))
    stage_gate = parser.add_argument_group(
        "first-move staging gate",
        (
            "Transition-pose controls for the first move. These are separate from contact-pose bounds; "
            "widening them can increase collision/self-collision risk."
        ),
    )
    stage_gate.add_argument(
        "--first_move_strategy",
        choices=["none", "forward_stage", "cartesian_stage"],
        default="none",
        help="Experimental first move before the planned pregrasp. Ignored when --motion_profile checkpoint.",
    )
    stage_gate.add_argument("--forward_axis", choices=["x", "y"], default="x")
    stage_gate.add_argument("--forward_sign", choices=["pos", "neg"], default="pos")
    stage_gate.add_argument("--forward_step_mm", type=float, default=FORWARD_STEP_MM)
    stage_gate.add_argument("--forward_steps", type=int, default=FORWARD_STEPS)
    stage_gate.add_argument("--forward_stage_speed_mm_s", type=float, default=FORWARD_STAGE_SPEED_MM_S)
    stage_gate.add_argument("--stage_x_min_mm", type=float, default=float(MINI_TASK_STAGING_MIN_MM[0]))
    stage_gate.add_argument("--stage_x_max_mm", type=float, default=float(MINI_TASK_STAGING_MAX_MM[0]))
    stage_gate.add_argument("--stage_y_min_mm", type=float, default=float(MINI_TASK_STAGING_MIN_MM[1]))
    stage_gate.add_argument("--stage_y_max_mm", type=float, default=float(MINI_TASK_STAGING_MAX_MM[1]))
    stage_gate.add_argument("--stage_z_min_mm", type=float, default=float(MINI_TASK_STAGING_MIN_MM[2]))
    stage_gate.add_argument("--stage_z_max_mm", type=float, default=float(MINI_TASK_STAGING_MAX_MM[2]))
    stage_gate.add_argument("--stage_z_mm", type=float, default=float(MINI_TASK_STAGING_Z_MM))
    parser.add_argument(
        "--move_home_before_task",
        action="store_true",
        help="Opt in to xArm move_gohome before the task. Warning: may trigger self-collision in this lab setup.",
    )
    parser.add_argument(
        "--skip_home",
        action="store_true",
        help="Deprecated compatibility flag; home motion is skipped by default.",
    )
    parser.add_argument(
        "--no_final_home",
        action="store_true",
        help="Diagnostic legacy-only flag. checkpoint8_style requires final home.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Confirm perception without connecting to Lite6.")
    parser.add_argument(
        "--allow_outside_workspace",
        action="store_true",
        help="Allow execution even if the local mini_task workspace guardrail rejects a pose.",
    )
    parser.add_argument("--no_gui", action="store_true", help="Always use terminal confirmation and save preview image.")
    parser.add_argument("--preview_path", default="logs/run_mini_task_preview.png", help="Path for the saved confirmation preview.")
    args = parser.parse_args()
    args._target_tag_size_m_arg = _argv_option_value("--target_tag_size_m")
    return args


def _load_robot_ip(config_path: str, override: str | None) -> str:
    if override:
        return override
    try:
        config = load_robot_config(Path(config_path))
    except Exception as exc:
        print(f"Warning: failed to read {config_path}: {exc}")
        return DEFAULT_ROBOT_IP
    robot_ip = str(config.get("robot_ip", "")).strip()
    return robot_ip or DEFAULT_ROBOT_IP


@dataclass(frozen=True)
class CubeTagPair:
    cube_prompt: str
    target_tag_id: int


@dataclass(frozen=True)
class DuplicateCubeTagGroup:
    cube_prompt: str
    tag_id: int
    count: int


@dataclass(frozen=True)
class PresetCubeGroup:
    cube_prompt: str
    count: int


@dataclass(frozen=True)
class PresetSlot:
    slot_id: int
    x: float
    y: float
    z: float
    yaw_deg: float = 0.0

    @property
    def center_robot(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def to_json_data(self) -> dict[str, float | int]:
        return {
            "slot_id": int(self.slot_id),
            "x": float(self.x),
            "y": float(self.y),
            "z": float(self.z),
            "yaw_deg": float(self.yaw_deg),
        }


@dataclass(frozen=True)
class PresetLayout:
    name: str
    frame: str
    slots: dict[int, PresetSlot]


@dataclass(frozen=True)
class Checkpoint8MultiPlanEntry:
    index: int
    cube_prompt: str
    target_tag_id: int
    T_robot_cube: np.ndarray
    T_cam_cube: np.ndarray
    T_robot_tag: np.ndarray
    T_cam_tag: np.ndarray
    T_robot_place: np.ndarray
    T_cam_place: np.ndarray


@dataclass(frozen=True)
class DuplicateCubeCandidate:
    cube_prompt: str
    cube_color: str
    instance_index: int
    component_label: int
    area_px: int
    score: float
    bbox_diag_m: float
    max_extent_m: float
    yaw_rad: float
    T_robot_cube: np.ndarray
    T_cam_cube: np.ndarray
    center_robot: np.ndarray
    member_candidate_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class RejectedDuplicateCubeCandidate:
    candidate: DuplicateCubeCandidate
    rejection_reasons: list[str]


@dataclass(frozen=True)
class MergedDuplicateCubeCandidates:
    physical_candidate: DuplicateCubeCandidate
    merged_candidates: list[DuplicateCubeCandidate]
    merge_distance_m: float

    @property
    def kept_candidate(self) -> DuplicateCubeCandidate:
        return self.physical_candidate


@dataclass(frozen=True)
class DuplicateTagCandidate:
    tag_id: int
    instance_index: int
    detection_index: int
    decision_margin: float
    hamming: int
    T_robot_tag: np.ndarray
    T_cam_tag: np.ndarray
    center_robot: np.ndarray


@dataclass(frozen=True)
class DuplicateAssignedPair:
    execution_index: int
    group_index: int
    within_group_index: int
    cube_prompt: str
    tag_id: int
    cube: DuplicateCubeCandidate
    tag: DuplicateTagCandidate
    distance_m: float
    T_robot_place: np.ndarray
    T_cam_place: np.ndarray


@dataclass(frozen=True)
class PresetAssignedPair:
    execution_index: int
    group_index: int
    within_group_index: int
    cube_prompt: str
    cube: DuplicateCubeCandidate
    slot: PresetSlot
    tag: DuplicateTagCandidate
    distance_m: float
    T_robot_place: np.ndarray
    T_cam_place: np.ndarray
    preset_use_slot_yaw: bool

    @property
    def tag_id(self) -> int:
        return int(self.slot.slot_id)


@dataclass(frozen=True)
class DuplicatePosePlan:
    execution_index: int
    cube_prompt: str
    target_tag_id: int
    cube_instance_index: int
    tag_instance_index: int
    T_robot_cube: np.ndarray
    T_robot_place: np.ndarray
    target_source: str = "apriltag"
    slot_id: int | None = None
    preset_slot: dict[str, Any] | None = None
    preset_use_slot_yaw: bool = False


@dataclass(frozen=True)
class PosePlanRefinement:
    plan: DuplicatePosePlan
    refined_cube: DuplicateCubeCandidate
    refined_tag: DuplicateTagCandidate | None
    T_robot_cube: np.ndarray
    T_robot_place: np.ndarray
    cube_delta_m: float
    tag_delta_m: float


@dataclass(frozen=True)
class DuplicateAssignmentSelection:
    selected_cubes: list[DuplicateCubeCandidate]
    selected_tags: list[DuplicateTagCandidate]
    tag_permutation: list[int]
    distance_matrix: np.ndarray
    pair_distances: list[float]
    total_distance_m: float
    objective: float


@dataclass(frozen=True)
class PresetAssignmentSelection:
    selected_cubes: list[DuplicateCubeCandidate]
    selected_slots: list[PresetSlot]
    slot_permutation: list[int]
    distance_matrix: np.ndarray
    pair_distances: list[float]
    total_distance_m: float
    objective: float


class PosePlanSafetyAbort(RuntimeError):
    pass


def _normalize_cube_prompt_key(prompt: str) -> str:
    return " ".join(prompt.strip().casefold().split())


def parse_cube_tag_map(value: str | None) -> list[tuple[str, int]]:
    if value is None or not str(value).strip():
        raise ValueError("--cube_tag_map must be a nonempty string.")

    pairs: list[tuple[str, int]] = []
    seen_prompts: set[str] = set()
    seen_tag_ids: set[int] = set()

    for raw_index, raw_item in enumerate(str(value).split(","), start=1):
        item = raw_item.strip()
        if not item:
            raise ValueError(f"entry {raw_index} is empty.")
        if ":" not in item:
            raise ValueError(f"entry {raw_index} must use the format 'cube prompt:tag_id'.")

        prompt_text, tag_text = item.rsplit(":", 1)
        cube_prompt = prompt_text.strip()
        if not cube_prompt:
            raise ValueError(f"entry {raw_index} has an empty cube prompt.")

        tag_text = tag_text.strip()
        if not tag_text:
            raise ValueError(f"entry {raw_index} has an empty target tag ID.")
        try:
            tag_id = int(tag_text, 10)
        except ValueError as exc:
            raise ValueError(f"entry {raw_index} target tag ID is not an integer: {tag_text!r}.") from exc

        prompt_key = _normalize_cube_prompt_key(cube_prompt)
        if prompt_key in seen_prompts:
            raise ValueError(f"duplicate cube prompt: {cube_prompt!r}.")
        if tag_id in seen_tag_ids:
            raise ValueError(f"duplicate target tag ID: {tag_id}.")

        seen_prompts.add(prompt_key)
        seen_tag_ids.add(tag_id)
        pairs.append((cube_prompt, tag_id))

    if not pairs:
        raise ValueError("--cube_tag_map must contain at least one cube/tag pair.")
    return pairs


def parse_duplicate_cube_tag_map(value: str | None) -> list[dict[str, int | str]]:
    if value is None or not str(value).strip():
        raise ValueError("--duplicate_cube_tag_map must be a nonempty string.")

    groups: list[dict[str, int | str]] = []
    seen_groups: set[tuple[str, int]] = set()

    for raw_index, raw_item in enumerate(str(value).split(","), start=1):
        item = raw_item.strip()
        if not item:
            raise ValueError(f"entry {raw_index} is empty.")

        parts = item.rsplit(":", 2)
        if len(parts) != 3:
            raise ValueError(f"entry {raw_index} must use the format 'cube prompt:tag_id:count'.")

        prompt_text, tag_text, count_text = parts
        cube_prompt = prompt_text.strip()
        if not cube_prompt:
            raise ValueError(f"entry {raw_index} has an empty cube prompt.")

        tag_text = tag_text.strip()
        if not tag_text:
            raise ValueError(f"entry {raw_index} has an empty target tag ID.")
        try:
            tag_id = int(tag_text, 10)
        except ValueError as exc:
            raise ValueError(f"entry {raw_index} target tag ID is not an integer: {tag_text!r}.") from exc

        count_text = count_text.strip()
        if not count_text:
            raise ValueError(f"entry {raw_index} has an empty count.")
        try:
            count = int(count_text, 10)
        except ValueError as exc:
            raise ValueError(f"entry {raw_index} count is not an integer: {count_text!r}.") from exc
        if count <= 0:
            raise ValueError(f"entry {raw_index} count must be a positive integer.")
        if count > MAX_DUPLICATE_GROUP_COUNT:
            raise ValueError(
                f"entry {raw_index} count {count} is too large; maximum is {MAX_DUPLICATE_GROUP_COUNT}."
            )

        group_key = (_normalize_cube_prompt_key(cube_prompt), tag_id)
        if group_key in seen_groups:
            raise ValueError(f"duplicate cube prompt/tag ID group: {cube_prompt!r} -> tag {tag_id}.")

        seen_groups.add(group_key)
        groups.append({"cube_prompt": cube_prompt, "tag_id": tag_id, "count": count})

    if not groups:
        raise ValueError("--duplicate_cube_tag_map must contain at least one group.")
    return groups


def parse_preset_cube_counts(value: str | None) -> list[dict[str, int | str]]:
    if value is None or not str(value).strip():
        raise ValueError("--preset_cube_counts must be a nonempty string.")

    groups: list[dict[str, int | str]] = []
    seen_prompts: set[str] = set()
    for raw_index, raw_item in enumerate(str(value).split(","), start=1):
        item = raw_item.strip()
        if not item:
            raise ValueError(f"entry {raw_index} is empty.")
        if ":" not in item:
            raise ValueError(f"entry {raw_index} must use the format 'cube prompt:count'.")
        prompt_text, count_text = item.rsplit(":", 1)
        cube_prompt = prompt_text.strip()
        if not cube_prompt:
            raise ValueError(f"entry {raw_index} has an empty cube prompt.")
        try:
            count = int(count_text.strip(), 10)
        except ValueError as exc:
            raise ValueError(f"entry {raw_index} count is not an integer: {count_text!r}.") from exc
        if count <= 0:
            raise ValueError(f"entry {raw_index} count must be a positive integer.")
        if count > MAX_DUPLICATE_GROUP_COUNT:
            raise ValueError(
                f"entry {raw_index} count {count} is too large; maximum is {MAX_DUPLICATE_GROUP_COUNT}."
            )
        prompt_key = _normalize_cube_prompt_key(cube_prompt)
        if prompt_key in seen_prompts:
            raise ValueError(f"duplicate cube prompt: {cube_prompt!r}.")
        seen_prompts.add(prompt_key)
        groups.append({"cube_prompt": cube_prompt, "count": count})

    if not groups:
        raise ValueError("--preset_cube_counts must contain at least one group.")
    return groups


def parse_preset_cube_slot_map(value: str | None) -> list[dict[str, Any]]:
    if value is None or not str(value).strip():
        raise ValueError("--preset_cube_slot_map must be a nonempty string.")

    groups: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for raw_index, raw_item in enumerate(str(value).split(";"), start=1):
        item = raw_item.strip()
        if not item:
            raise ValueError(f"entry {raw_index} is empty.")
        if ":" not in item:
            raise ValueError(f"entry {raw_index} must use the format 'cube prompt:slot_id,slot_id'.")
        prompt_text, slots_text = item.rsplit(":", 1)
        cube_prompt = prompt_text.strip()
        if not cube_prompt:
            raise ValueError(f"entry {raw_index} has an empty cube prompt.")
        slot_ids: list[int] = []
        for slot_index, raw_slot in enumerate(slots_text.split(","), start=1):
            slot_text = raw_slot.strip()
            if not slot_text:
                raise ValueError(f"entry {raw_index} slot {slot_index} is empty.")
            try:
                slot_id = int(slot_text, 10)
            except ValueError as exc:
                raise ValueError(f"entry {raw_index} slot {slot_index} is not an integer: {slot_text!r}.") from exc
            slot_ids.append(slot_id)
        if not slot_ids:
            raise ValueError(f"entry {raw_index} must contain at least one slot ID.")
        prompt_key = _normalize_cube_prompt_key(cube_prompt)
        if prompt_key in seen_prompts:
            raise ValueError(f"duplicate cube prompt: {cube_prompt!r}.")
        seen_prompts.add(prompt_key)
        groups.append({"cube_prompt": cube_prompt, "slot_ids": slot_ids})

    if not groups:
        raise ValueError("--preset_cube_slot_map must contain at least one group.")
    return groups


def _cube_tag_pairs_from_args(args: argparse.Namespace) -> list[CubeTagPair]:
    raw_pairs = getattr(args, "_cube_tag_pairs", None)
    if raw_pairs is None:
        raw_pairs = parse_cube_tag_map(args.cube_tag_map)
        args._cube_tag_pairs = raw_pairs
    return [CubeTagPair(cube_prompt=prompt, target_tag_id=tag_id) for prompt, tag_id in raw_pairs]


def _duplicate_groups_from_args(args: argparse.Namespace) -> list[DuplicateCubeTagGroup]:
    raw_groups = getattr(args, "_duplicate_cube_tag_groups", None)
    if raw_groups is None:
        raw_groups = parse_duplicate_cube_tag_map(args.duplicate_cube_tag_map)
        args._duplicate_cube_tag_groups = raw_groups
    return [
        DuplicateCubeTagGroup(
            cube_prompt=str(group["cube_prompt"]),
            tag_id=int(group["tag_id"]),
            count=int(group["count"]),
        )
        for group in raw_groups
    ]


def _preset_groups_from_args(args: argparse.Namespace) -> list[PresetCubeGroup]:
    raw_groups = getattr(args, "_preset_cube_count_groups", None)
    if raw_groups is None:
        raw_groups = parse_preset_cube_counts(args.preset_cube_counts)
        args._preset_cube_count_groups = raw_groups
    return [
        PresetCubeGroup(cube_prompt=str(group["cube_prompt"]), count=int(group["count"]))
        for group in raw_groups
    ]


def _preset_slot_map_from_args(args: argparse.Namespace) -> dict[str, list[int]]:
    raw_map = getattr(args, "_preset_cube_slot_map_groups", None)
    if raw_map is None:
        raw_map = parse_preset_cube_slot_map(args.preset_cube_slot_map)
        args._preset_cube_slot_map_groups = raw_map
    return {
        _normalize_cube_prompt_key(str(group["cube_prompt"])): [int(slot_id) for slot_id in group["slot_ids"]]
        for group in raw_map
    }


def _cube_color_from_prompt(cube_prompt: str) -> str:
    normalized = _normalize_cube_prompt_key(cube_prompt).replace("-", " ").replace("_", " ")
    tokens = set(normalized.split())
    matches = [color for color in ("red", "green", "blue") if color in tokens]
    if len(matches) != 1:
        raise ValueError(
            f"cube prompt {cube_prompt!r} must contain exactly one supported color word: red, green, or blue."
        )
    return matches[0]


def parse_candidate_merge_prompts(value: str | None) -> list[str]:
    if value is None:
        return []
    prompts: list[str] = []
    seen: set[str] = set()
    for raw_item in str(value).split(","):
        prompt = raw_item.strip()
        if not prompt:
            continue
        key = _normalize_cube_prompt_key(prompt)
        if key in seen:
            continue
        seen.add(key)
        prompts.append(prompt)
    return prompts


def _candidate_merge_prompt_keys_from_args(args: argparse.Namespace) -> set[str]:
    raw_prompts = getattr(args, "_candidate_merge_prompts", None)
    if raw_prompts is None:
        raw_prompts = parse_candidate_merge_prompts(args.candidate_merge_prompts)
        args._candidate_merge_prompts = raw_prompts
    return {_normalize_cube_prompt_key(prompt) for prompt in raw_prompts}


def _candidate_merge_enabled_for_prompt(args: argparse.Namespace, cube_prompt: str) -> bool:
    return _normalize_cube_prompt_key(cube_prompt) in _candidate_merge_prompt_keys_from_args(args)


def _resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def _preset_slot_from_json_data(data: Any, index: int) -> PresetSlot:
    if not isinstance(data, dict):
        raise ValueError(f"slot {index} must be an object.")
    required = ["slot_id", "x", "y", "z"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"slot {index} is missing required keys: {', '.join(missing)}")
    try:
        slot_id = int(data["slot_id"])
        x = float(data["x"])
        y = float(data["y"])
        z = float(data["z"])
        yaw_deg = float(data.get("yaw_deg", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"slot {index} contains non-numeric values.") from exc
    values = np.array([x, y, z, yaw_deg], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"slot {index} contains non-finite values.")
    return PresetSlot(slot_id=slot_id, x=x, y=y, z=z, yaw_deg=yaw_deg)


def load_preset_place_layout_json(path: str | Path) -> PresetLayout:
    layout_path = _resolve_repo_path(path)
    with layout_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("preset layout JSON root must be an object.")
    name = str(data.get("name", "")).strip()
    if not name:
        raise ValueError("preset layout JSON name must be nonempty.")
    frame = str(data.get("frame", "")).strip()
    if frame != "robot_base":
        raise ValueError("preset layout JSON frame must be 'robot_base'.")
    raw_slots = data.get("slots")
    if not isinstance(raw_slots, list) or not raw_slots:
        raise ValueError("preset layout JSON slots must be a nonempty list.")

    slots: dict[int, PresetSlot] = {}
    for index, raw_slot in enumerate(raw_slots, start=1):
        slot = _preset_slot_from_json_data(raw_slot, index)
        if slot.slot_id in slots:
            raise ValueError(f"duplicate preset slot ID: {slot.slot_id}.")
        slots[slot.slot_id] = slot
    return PresetLayout(name=name, frame=frame, slots=slots)


def _preset_counts_by_prompt(groups: list[dict[str, int | str]] | list[PresetCubeGroup]) -> dict[str, tuple[str, int]]:
    counts: dict[str, tuple[str, int]] = {}
    for group in groups:
        if isinstance(group, PresetCubeGroup):
            cube_prompt = group.cube_prompt
            count = group.count
        else:
            cube_prompt = str(group["cube_prompt"])
            count = int(group["count"])
        counts[_normalize_cube_prompt_key(cube_prompt)] = (cube_prompt, int(count))
    return counts


def _preset_slot_map_by_prompt(groups: list[dict[str, Any]] | dict[str, list[int]]) -> dict[str, list[int]]:
    if isinstance(groups, dict):
        return {str(key): [int(slot_id) for slot_id in value] for key, value in groups.items()}
    return {
        _normalize_cube_prompt_key(str(group["cube_prompt"])): [int(slot_id) for slot_id in group["slot_ids"]]
        for group in groups
    }


def validate_preset_layout_request(
    layout: PresetLayout,
    cube_counts: list[dict[str, int | str]] | list[PresetCubeGroup],
    cube_slot_map: list[dict[str, Any]] | dict[str, list[int]],
    slot_minimum_mm: np.ndarray,
    slot_maximum_mm: np.ndarray,
    allow_outside_workspace: bool = False,
) -> None:
    counts = _preset_counts_by_prompt(cube_counts)
    slot_map = _preset_slot_map_by_prompt(cube_slot_map)
    count_keys = set(counts)
    map_keys = set(slot_map)
    missing_from_map = count_keys - map_keys
    unknown_in_map = map_keys - count_keys
    if missing_from_map:
        prompts = ", ".join(counts[key][0] for key in sorted(missing_from_map))
        raise ValueError(f"preset slot map is missing prompts from counts: {prompts}.")
    if unknown_in_map:
        prompts = ", ".join(sorted(unknown_in_map))
        raise ValueError(f"preset slot map contains prompts not present in counts: {prompts}.")

    seen_slot_ids: set[int] = set()
    for prompt_key, slot_ids in slot_map.items():
        cube_prompt, count = counts[prompt_key]
        if len(slot_ids) != count:
            raise ValueError(
                f"preset slot count mismatch for {cube_prompt!r}: "
                f"count is {count}, but slot map has {len(slot_ids)} slots."
            )
        for slot_id in slot_ids:
            if slot_id not in layout.slots:
                raise ValueError(f"preset slot ID {slot_id} for {cube_prompt!r} does not exist in layout.")
            if slot_id in seen_slot_ids:
                raise ValueError(f"duplicate preset slot ID in slot map: {slot_id}.")
            seen_slot_ids.add(slot_id)

    if allow_outside_workspace:
        return

    for slot in layout.slots.values():
        point_mm = slot.center_robot * 1000.0
        if not _point_in_bounds_mm(point_mm, slot_minimum_mm, slot_maximum_mm):
            raise ValueError(
                f"preset slot {slot.slot_id} is outside the conservative contact workspace: "
                f"point={np.array2string(point_mm, precision=1)}mm "
                f"bounds={_format_mm_bounds(slot_minimum_mm, slot_maximum_mm)}."
            )


def _validate_args(args: argparse.Namespace) -> None:
    if args.execute_pose_plan_refine_only and not args.execute_pose_plan_json:
        raise SystemExit("--execute_pose_plan_refine_only requires --execute_pose_plan_json.")
    if float(args.pose_plan_refine_radius_m) <= 0.0:
        raise SystemExit("--pose_plan_refine_radius_m must be positive.")
    if float(args.pose_plan_refine_tag_radius_m) <= 0.0:
        raise SystemExit("--pose_plan_refine_tag_radius_m must be positive.")
    if float(args.min_after_grasp_z_mm) <= 0.0:
        raise SystemExit("--min_after_grasp_z_mm must be positive.")

    refine_mode_requested = bool(args.refine_pose_plan_json or args.refined_pose_plan_output_json)
    if refine_mode_requested:
        if args.execution_backend != "checkpoint8_style":
            raise SystemExit("--refine_pose_plan_json is only supported with --execution_backend checkpoint8_style.")
        if not args.refine_pose_plan_json:
            raise SystemExit("--refined_pose_plan_output_json requires --refine_pose_plan_json.")
        if not args.refined_pose_plan_output_json:
            raise SystemExit("--refine_pose_plan_json requires --refined_pose_plan_output_json.")
        if args.execute_pose_plan_json:
            raise SystemExit("--refine_pose_plan_json cannot be combined with --execute_pose_plan_json.")
        if args.no_pose_plan_refine:
            raise SystemExit("--no_pose_plan_refine is only valid with --execute_pose_plan_json.")
        return

    if args.execute_pose_plan_json:
        if args.no_pose_plan_refine and args.execute_pose_plan_refine_only:
            raise SystemExit("--no_pose_plan_refine cannot be combined with --execute_pose_plan_refine_only.")
        if args.no_pose_plan_refine:
            args.pose_plan_refine_before_execute = False
        if args.pose_plan_refine_before_execute is None:
            args.pose_plan_refine_before_execute = True
        if args.execution_backend != "checkpoint8_style":
            raise SystemExit("--execute_pose_plan_json is only supported with --execution_backend checkpoint8_style.")
        if args.skip_home or args.no_final_home:
            raise SystemExit("--execute_pose_plan_json requires checkpoint8-style home behavior; do not use --skip_home or --no_final_home.")
        conflicting_flags = [
            "--place_to_tag" if args.place_to_tag else None,
            "--multi_place_to_tags" if args.multi_place_to_tags else None,
            "--duplicate_aware_multi_place" if args.duplicate_aware_multi_place else None,
            "--cube_tag_map" if args.cube_tag_map else None,
            "--duplicate_cube_tag_map" if args.duplicate_cube_tag_map else None,
        ]
        conflicts = [flag for flag in conflicting_flags if flag is not None]
        if conflicts:
            raise SystemExit("--execute_pose_plan_json cannot be combined with " + ", ".join(conflicts) + ".")
        return

    if args.pose_plan_refine_before_execute is None:
        args.pose_plan_refine_before_execute = False
    if args.no_pose_plan_refine:
        raise SystemExit("--no_pose_plan_refine is only valid with --execute_pose_plan_json.")

    if args.preset_layout_place:
        if args.execution_backend != "checkpoint8_style":
            raise SystemExit("--preset_layout_place is only supported with --execution_backend checkpoint8_style.")
        conflicting_flags = [
            "--place_to_tag" if args.place_to_tag else None,
            "--multi_place_to_tags" if args.multi_place_to_tags else None,
            "--duplicate_aware_multi_place" if args.duplicate_aware_multi_place else None,
            "--cube_tag_map" if args.cube_tag_map else None,
            "--duplicate_cube_tag_map" if args.duplicate_cube_tag_map else None,
        ]
        conflicts = [flag for flag in conflicting_flags if flag is not None]
        if conflicts:
            raise SystemExit("--preset_layout_place cannot be combined with " + ", ".join(conflicts) + ".")
        if not args.preset_place_layout_json:
            raise SystemExit("--preset_layout_place requires --preset_place_layout_json.")
        if not args.preset_cube_counts:
            raise SystemExit("--preset_layout_place requires --preset_cube_counts.")
        if not args.preset_cube_slot_map:
            raise SystemExit("--preset_layout_place requires --preset_cube_slot_map.")
        if args.skip_home or args.no_final_home:
            raise SystemExit(
                "--preset_layout_place requires checkpoint8-style home behavior; "
                "do not use --skip_home or --no_final_home."
            )
        try:
            args._preset_cube_count_groups = parse_preset_cube_counts(args.preset_cube_counts)
            args._preset_cube_slot_map_groups = parse_preset_cube_slot_map(args.preset_cube_slot_map)
            for group in _preset_groups_from_args(args):
                _cube_color_from_prompt(group.cube_prompt)
            args._preset_layout = load_preset_place_layout_json(args.preset_place_layout_json)
            preset_minimum_mm, preset_maximum_mm = _task_gate_bounds_from_args(args)
            validate_preset_layout_request(
                layout=args._preset_layout,
                cube_counts=args._preset_cube_count_groups,
                cube_slot_map=args._preset_cube_slot_map_groups,
                slot_minimum_mm=preset_minimum_mm,
                slot_maximum_mm=preset_maximum_mm,
                allow_outside_workspace=args.allow_preset_slots_outside_workspace,
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid preset layout placement configuration: {exc}") from exc
        return

    if args.preset_place_layout_json or args.preset_cube_counts or args.preset_cube_slot_map:
        raise SystemExit("--preset_place_layout_json/--preset_cube_counts/--preset_cube_slot_map require --preset_layout_place.")
    if args.preset_use_slot_yaw:
        raise SystemExit("--preset_use_slot_yaw is only valid with --preset_layout_place.")
    if args.allow_preset_slots_outside_workspace:
        raise SystemExit("--allow_preset_slots_outside_workspace is only valid with --preset_layout_place.")

    if args.duplicate_aware_multi_place:
        if args.execution_backend != "checkpoint8_style":
            raise SystemExit("--duplicate_aware_multi_place is only supported with --execution_backend checkpoint8_style.")
        if args.place_to_tag:
            raise SystemExit("--place_to_tag cannot be combined with --duplicate_aware_multi_place.")
        if args.multi_place_to_tags:
            raise SystemExit("--multi_place_to_tags cannot be combined with --duplicate_aware_multi_place.")
        if args.cube_tag_map:
            raise SystemExit("--duplicate_aware_multi_place cannot be combined with --cube_tag_map; use --duplicate_cube_tag_map.")
        if not args.duplicate_cube_tag_map:
            raise SystemExit(
                "--duplicate_aware_multi_place requires --duplicate_cube_tag_map, "
                'for example "red cube:6:2,green cube:8:2".'
            )
        if getattr(args, "_target_tag_size_m_arg", None) is None:
            raise SystemExit("--duplicate_aware_multi_place requires an explicit --target_tag_size_m value.")
        if args.skip_home or args.no_final_home:
            raise SystemExit(
                "--duplicate_aware_multi_place requires checkpoint8-style home behavior; "
                "do not use --skip_home or --no_final_home."
            )
        if float(args.max_assignment_distance_m) <= 0.0:
            raise SystemExit("--max_assignment_distance_m must be positive.")
        if int(args.candidate_min_area_px) < 0:
            raise SystemExit("--candidate_min_area_px must be nonnegative.")
        if float(args.candidate_merge_distance_m) <= 0.0:
            raise SystemExit("--candidate_merge_distance_m must be positive.")
        if float(args.candidate_min_extent_m) < 0.0:
            raise SystemExit("--candidate_min_extent_m must be nonnegative.")
        if float(args.candidate_max_extent_m) <= 0.0:
            raise SystemExit("--candidate_max_extent_m must be positive.")
        if float(args.candidate_min_extent_m) > float(args.candidate_max_extent_m):
            raise SystemExit("--candidate_min_extent_m must be less than or equal to --candidate_max_extent_m.")
        for flag_name in ("green_candidate_min_area_px", "red_candidate_min_area_px", "blue_candidate_min_area_px"):
            if int(getattr(args, flag_name)) < 0:
                raise SystemExit(f"--{flag_name} must be nonnegative.")
        args._candidate_merge_prompts = parse_candidate_merge_prompts(args.candidate_merge_prompts)
        candidate_bounds = [
            ("x", args.candidate_x_min_m, args.candidate_x_max_m),
            ("y", args.candidate_y_min_m, args.candidate_y_max_m),
            ("z", args.candidate_z_min_m, args.candidate_z_max_m),
        ]
        for axis_name, minimum, maximum in candidate_bounds:
            if float(minimum) > float(maximum):
                raise SystemExit(
                    f"Invalid duplicate-aware candidate {axis_name} bounds: "
                    f"minimum {float(minimum):.3f} m is greater than maximum {float(maximum):.3f} m."
                )
        try:
            args._duplicate_cube_tag_groups = parse_duplicate_cube_tag_map(args.duplicate_cube_tag_map)
            for group in _duplicate_groups_from_args(args):
                _cube_color_from_prompt(group.cube_prompt)
        except ValueError as exc:
            raise SystemExit(f"Invalid --duplicate_cube_tag_map: {exc}") from exc
        return

    if args.duplicate_cube_tag_map:
        raise SystemExit("--duplicate_cube_tag_map is only valid when --duplicate_aware_multi_place is used.")
    if args.debug_duplicate_candidates_only:
        raise SystemExit("--debug_duplicate_candidates_only is only valid when --duplicate_aware_multi_place is used.")

    if args.multi_place_to_tags:
        if args.execution_backend != "checkpoint8_style":
            raise SystemExit("--multi_place_to_tags is only supported with --execution_backend checkpoint8_style.")
        if args.place_to_tag:
            raise SystemExit("--place_to_tag cannot be combined with --multi_place_to_tags; use only the multi mapping.")
        if not args.cube_tag_map:
            raise SystemExit("--multi_place_to_tags requires --cube_tag_map, for example \"red cube:7,green cube:8\".")
        if args.skip_home or args.no_final_home:
            raise SystemExit("--multi_place_to_tags requires checkpoint8-style home behavior; do not use --skip_home or --no_final_home.")
        try:
            args._cube_tag_pairs = parse_cube_tag_map(args.cube_tag_map)
        except ValueError as exc:
            raise SystemExit(f"Invalid --cube_tag_map: {exc}") from exc
        return

    if args.cube_tag_map:
        raise SystemExit("--cube_tag_map is only valid when --multi_place_to_tags is used.")


def _target_tag_size_command_arg(args: argparse.Namespace) -> str:
    original_value = getattr(args, "_target_tag_size_m_arg", None)
    if original_value is not None and str(original_value).strip():
        return str(original_value)
    value = float(args.target_tag_size_m)
    if math.isclose(value, TARGET_TAG_SIZE_M, rel_tol=0.0, abs_tol=1e-12):
        return f"{TARGET_TAG_SIZE_M:.3f}"
    return f"{value:.12g}"


def _build_checkpoint8_multi_subprocess_child_command(
    args: argparse.Namespace,
    pair: CubeTagPair,
) -> list[str]:
    return [
        "python",
        "scripts/run_mini_task.py",
        "--execution_backend",
        "checkpoint8_style",
        "--cube_prompt",
        pair.cube_prompt,
        "--place_to_tag",
        "--target_tag_id",
        str(pair.target_tag_id),
        "--target_tag_size_m",
        _target_tag_size_command_arg(args),
        "--no_gui",
        "--auto_confirm",
    ]


def _format_command_for_log(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _print_multi_subprocess_child_commands(args: argparse.Namespace, cube_tag_pairs: list[CubeTagPair]) -> None:
    print("Subprocess child commands that would be used for each pair:")
    for index, pair in enumerate(cube_tag_pairs, start=1):
        command = _build_checkpoint8_multi_subprocess_child_command(args, pair)
        print(f"  Pair {index}/{len(cube_tag_pairs)}: {_format_command_for_log(command)}")


def _print_multi_subprocess_pair_map(cube_tag_pairs: list[CubeTagPair]) -> None:
    print("Multi subprocess cube/tag map:")
    for index, pair in enumerate(cube_tag_pairs, start=1):
        print(f"  Pair {index}/{len(cube_tag_pairs)}: {pair.cube_prompt} -> tag {pair.target_tag_id}")


def _child_returncode_looks_like_native_crash(returncode: int) -> bool:
    return int(returncode) < 0 or int(returncode) in {134, 135, 136, 137, 138, 139}


def _print_multi_subprocess_summary(
    total_pairs: int,
    succeeded_pairs: list[CubeTagPair],
    failed_pairs: list[CubeTagPair],
) -> None:
    print("\nMulti subprocess parent summary:")
    print(f"Total pairs: {total_pairs}")
    print(f"Succeeded pairs: {len(succeeded_pairs)}")
    print(f"Failed pairs: {len(failed_pairs)}")
    if failed_pairs:
        failed_mapping = ", ".join(f"{pair.cube_prompt} -> tag {pair.target_tag_id}" for pair in failed_pairs)
        print(f"Failed cube/tag mappings: {failed_mapping}")
    else:
        print("Failed cube/tag mappings: none")


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _bgr_from_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return np.ascontiguousarray(image)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _camera_params(camera_intrinsic: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(camera_intrinsic[0, 0]),
        float(camera_intrinsic[1, 1]),
        float(camera_intrinsic[0, 2]),
        float(camera_intrinsic[1, 2]),
    )


def _detect_tags(image: np.ndarray, estimate_pose: bool, camera_intrinsic: np.ndarray, tag_size_m: float):
    detector = Detector(families=CALIBRATION_TAG_FAMILY)
    gray = _to_gray(image)
    if estimate_pose:
        return detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=_camera_params(camera_intrinsic),
            tag_size=float(tag_size_m),
        )
    return detector.detect(gray, estimate_tag_pose=False)


def _is_calibration_tag_id(tag_id: int) -> bool:
    return int(tag_id) in TAG_CENTER_COORDINATES_M


def _pnp_pairs_from_calibration_tags(tags: list[Any], tag_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    world_points = np.empty((0, 3), dtype=np.float64)
    image_points = np.empty((0, 2), dtype=np.float64)
    half = float(tag_size_m) * 0.5

    for tag in tags:
        tag_id = int(tag.tag_id)
        if tag_id not in TAG_CENTER_COORDINATES_M:
            continue

        tag_center = TAG_CENTER_COORDINATES_M[tag_id]
        corner_world_points = np.array(
            [
                [tag_center[0] - half, tag_center[1] + half, 0.0],
                [tag_center[0] - half, tag_center[1] - half, 0.0],
                [tag_center[0] + half, tag_center[1] - half, 0.0],
                [tag_center[0] + half, tag_center[1] + half, 0.0],
            ],
            dtype=np.float64,
        )
        world_points = np.vstack([world_points, corner_world_points])
        image_points = np.vstack([image_points, np.asarray(tag.corners, dtype=np.float64)])

    return world_points, image_points


def estimate_T_cam_base(
    image: np.ndarray,
    camera_intrinsic: np.ndarray,
    calibration_tag_size_m: float,
) -> tuple[np.ndarray | None, list[Any]]:
    tags = _detect_tags(image, estimate_pose=False, camera_intrinsic=camera_intrinsic, tag_size_m=calibration_tag_size_m)
    print(f"AprilTags detected for calibration: {[int(tag.tag_id) for tag in tags]}")
    print(f"Calibration tag size assumption: {calibration_tag_size_m:.4f} m for tag IDs 0..3.")

    world_points, image_points = _pnp_pairs_from_calibration_tags(tags, calibration_tag_size_m)
    if world_points.shape[0] < 4:
        print("Insufficient calibration tag corners. Need at least one known tag ID from 0..3.")
        return None, list(tags)

    success, rotation_vec, translation = cv2.solvePnP(world_points, image_points, camera_intrinsic, None)
    if not success:
        print("Calibration solvePnP failed.")
        return None, list(tags)

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    T_cam_base = np.eye(4, dtype=np.float64)
    T_cam_base[:3, :3] = rotation_mat
    T_cam_base[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T_cam_base, list(tags)


def _known_base_tag_pose(tag_id: int) -> np.ndarray | None:
    if tag_id not in TAG_CENTER_COORDINATES_M:
        return None
    T_base_tag = np.eye(4, dtype=np.float64)
    T_base_tag[:2, 3] = TAG_CENTER_COORDINATES_M[tag_id]
    return T_base_tag


def _estimate_detected_target_tag_pose(
    image: np.ndarray,
    camera_intrinsic: np.ndarray,
    T_cam_base: np.ndarray,
    target_tag_id: int,
    target_tag_size_m: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    pose_tags = _detect_tags(
        image,
        estimate_pose=True,
        camera_intrinsic=camera_intrinsic,
        tag_size_m=target_tag_size_m,
    )
    detected_ids = [int(tag.tag_id) for tag in pose_tags]
    print(f"AprilTags detected with target pose solve: {detected_ids}")
    print(f"Target tag size assumption: {target_tag_size_m:.4f} m for tag ID {target_tag_id}.")

    matches = [tag for tag in pose_tags if int(tag.tag_id) == int(target_tag_id)]
    if len(matches) > 1:
        print(f"Ambiguous target tag: found {len(matches)} detections with id {target_tag_id}.")
        return None, None
    if len(matches) != 1:
        print(f"Target AprilTag id {target_tag_id} was not detected with pose.")
        return None, None

    target = matches[0]
    if target.pose_R is None or target.pose_t is None:
        print(f"Target AprilTag id {target_tag_id} was detected, but pose estimation was unavailable.")
        return None, None

    T_cam_tag = np.eye(4, dtype=np.float64)
    T_cam_tag[:3, :3] = np.asarray(target.pose_R, dtype=np.float64).reshape(3, 3)
    T_cam_tag[:3, 3] = np.asarray(target.pose_t, dtype=np.float64).reshape(3)
    T_base_tag = np.linalg.inv(T_cam_base) @ T_cam_tag
    if not np.isfinite(T_base_tag).all():
        print("Target AprilTag pose contains invalid values after conversion to base frame.")
        return None, None
    return T_base_tag, T_cam_tag


def estimate_target_tag_pose(
    image: np.ndarray,
    camera_intrinsic: np.ndarray,
    T_cam_base: np.ndarray,
    target_tag_id: int,
    target_tag_size_m: float,
    calibration_tags: list[Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    # The older checkpoint0 calibration defines tag IDs 0..3 as fixed points in
    # the robot/base table frame. For those IDs, do not estimate a separate
    # single-tag pose; use the known table coordinates that were also used for
    # camera-to-base PnP. This avoids mixing two different tag-size assumptions.
    detected_ids = [int(tag.tag_id) for tag in calibration_tags]
    if _is_calibration_tag_id(target_tag_id):
        if int(target_tag_id) not in detected_ids:
            print(f"Target calibration AprilTag id {target_tag_id} was not detected.")
            return None, None
        if abs(float(target_tag_size_m) - float(CALIBRATION_TAG_SIZE_M)) > 1e-6:
            print(
                "Note: target_tag_size_m is ignored for calibration tag IDs 0..3; "
                f"checkpoint0 calibration size is {CALIBRATION_TAG_SIZE_M:.4f} m."
            )
        T_base_tag = _known_base_tag_pose(int(target_tag_id))
        if T_base_tag is not None:
            print("Using checkpoint0 fixed base-frame coordinates for the target calibration tag.")
            return T_base_tag, T_cam_base @ T_base_tag

    # Non-calibration target tags are solved independently. The caller must pass
    # the physical printed tag size for this specific target, for example 0.025 m
    # for the small lab target tags. A wrong size scales the recovered single-tag
    # translation, especially depth, and can also bias XY. Later logic therefore
    # uses the detected tag mainly for XY and yaw, while reconstructing place Z
    # from the table height and cube geometry.
    return _estimate_detected_target_tag_pose(
        image=image,
        camera_intrinsic=camera_intrinsic,
        T_cam_base=T_cam_base,
        target_tag_id=target_tag_id,
        target_tag_size_m=target_tag_size_m,
    )


def _segment_mask_from_color(bgr_image: np.ndarray, cube_color: str) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    red_1 = cv2.inRange(hsv, (0, 70, 70), (12, 255, 255))
    red_2 = cv2.inRange(hsv, (168, 70, 70), (180, 255, 255))
    red = cv2.bitwise_or(red_1, red_2)
    green = cv2.inRange(hsv, (38, 50, 50), (92, 255, 255))
    blue = cv2.inRange(hsv, (98, 50, 50), (132, 255, 255))

    if cube_color == "red":
        mask = red
    elif cube_color == "green":
        mask = green
    elif cube_color == "blue":
        mask = blue
    else:
        mask = cv2.bitwise_or(cv2.bitwise_or(red, green), blue)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _points_from_mask(point_cloud: np.ndarray, mask: np.ndarray, point_cloud_scale: float) -> np.ndarray:
    xyz = np.asarray(point_cloud[..., :3], dtype=np.float64).reshape(-1, 3)
    selected = xyz[mask.reshape(-1).astype(bool)] * float(point_cloud_scale)
    valid = np.isfinite(selected).all(axis=1) & (selected[:, 2] > 0.01)
    return selected[valid]


def _statistical_outlier_trim(pts: np.ndarray, nb_neighbors: int = 30, std_ratio: float = 2.0) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    n_points = pts.shape[0]
    if n_points <= nb_neighbors + 1:
        return pts

    tree = cKDTree(pts)
    k = min(nb_neighbors + 1, n_points)
    dists, _ = tree.query(pts, k=k)
    if dists.ndim == 1:
        return pts

    mean_dist = dists[:, 1:].mean(axis=1)
    threshold = float(mean_dist.mean() + std_ratio * (mean_dist.std() + 1e-12))
    return pts[mean_dist < threshold]


def _robust_extent(pts: np.ndarray, lo: float = 10.0, hi: float = 90.0) -> np.ndarray:
    q_lo = np.percentile(pts, lo, axis=0)
    q_hi = np.percentile(pts, hi, axis=0)
    return q_hi - q_lo


def _select_cube_component(
    mask: np.ndarray,
    point_cloud: np.ndarray,
    point_cloud_scale: float,
    cube_size_m: float,
) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.empty((0, 3), dtype=np.float64)

    image_h, image_w = mask.shape[:2]
    cube_diag = float(cube_size_m) * math.sqrt(3.0)
    candidates: list[tuple[float, int, np.ndarray]] = []

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px < 40:
            continue

        component_mask = labels == label
        points_m = _points_from_mask(point_cloud, component_mask, point_cloud_scale)
        if points_m.shape[0] < MIN_POINTS_AFTER_OUTLIER:
            continue

        trimmed = _statistical_outlier_trim(points_m, nb_neighbors=12, std_ratio=2.0)
        if trimmed.shape[0] < MIN_POINTS_AFTER_OUTLIER:
            continue

        extent = _robust_extent(trimmed)
        bbox_diag = float(np.linalg.norm(extent))
        max_extent = float(extent.max())
        touches_border = (
            x <= 1
            or y <= 1
            or (x + w) >= (image_w - 1)
            or (y + h) >= (image_h - 1)
        )

        score = abs(bbox_diag - cube_diag)
        score += max(0.0, max_extent - (float(cube_size_m) * 1.8)) * 3.0
        if touches_border:
            score += 0.25

        print(
            "Cube candidate"
            f" label={label}"
            f" area_px={area_px}"
            f" points={trimmed.shape[0]}"
            f" bbox_diag_m={bbox_diag:.4f}"
            f" max_extent_m={max_extent:.4f}"
            f" score={score:.4f}"
        )
        candidates.append((score, label, trimmed))

    if not candidates:
        return np.empty((0, 3), dtype=np.float64)

    candidates.sort(key=lambda item: item[0])
    best_score, best_label, best_points = candidates[0]
    if len(candidates) > 1:
        second_score = candidates[1][0]
        print(f"Selected cube component {best_label}; second score delta={second_score - best_score:.4f}")
    else:
        print(f"Selected cube component {best_label}.")
    return best_points


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts_h = np.column_stack((points, np.ones(points.shape[0], dtype=np.float64)))
    transformed = (transform @ pts_h.T).T
    return transformed[:, :3]


def _normalize_angle_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _select_preferred_grasp_yaw(yaw: float) -> float:
    candidates = [_normalize_angle_pi(yaw + k * (math.pi * 0.5)) for k in range(4)]
    return min(candidates, key=lambda value: abs(value))


def _estimate_cube_yaw_on_table(points_base: np.ndarray) -> float:
    z_values = points_base[:, 2]
    z_cut = np.percentile(z_values, 70.0)
    top_points_xy = points_base[z_values >= z_cut, :2]
    if top_points_xy.shape[0] < 20:
        top_points_xy = points_base[:, :2]
    if top_points_xy.shape[0] < 4:
        return 0.0

    rect = cv2.minAreaRect(top_points_xy.astype(np.float32).reshape(-1, 1, 2))
    (_, _), (width, height), angle_deg = rect
    if width < 1e-6 or height < 1e-6:
        return 0.0

    yaw = math.radians(float(angle_deg))
    if width < height:
        yaw += math.pi * 0.5
    return _normalize_angle_pi(yaw)


def _top_down_rotation(yaw: float) -> np.ndarray:
    return Rotation.from_euler("xyz", [math.pi, 0.0, float(yaw)]).as_matrix()


def estimate_cube_pose(
    image: np.ndarray,
    point_cloud: np.ndarray,
    T_cam_base: np.ndarray,
    cube_color: str,
    cube_size_m: float,
    table_z_m: float,
    point_cloud_scale: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    if image is None or point_cloud is None:
        return None, None, np.zeros((1, 1), dtype=np.uint8)
    if point_cloud.ndim != 3 or point_cloud.shape[-1] < 3:
        print(f"Invalid point cloud shape: {point_cloud.shape}")
        return None, None, np.zeros(image.shape[:2], dtype=np.uint8)
    if image.shape[:2] != point_cloud.shape[:2]:
        print(f"Image/point cloud shape mismatch: image={image.shape}, point_cloud={point_cloud.shape}")
        return None, None, np.zeros(image.shape[:2], dtype=np.uint8)

    bgr = _bgr_from_image(image)
    mask = _segment_mask_from_color(bgr, cube_color)
    points_cam = _select_cube_component(mask, point_cloud, point_cloud_scale, cube_size_m)
    if points_cam.shape[0] < MIN_POINTS_FIT:
        print(f"Cube detection failed: only {points_cam.shape[0]} usable points.")
        return None, None, mask

    trimmed = _statistical_outlier_trim(points_cam, nb_neighbors=24, std_ratio=2.5)
    if trimmed.shape[0] < MIN_POINTS_AFTER_OUTLIER:
        print(f"Cube pose fit failed after outlier trim: {trimmed.shape[0]} points.")
        return None, None, mask

    T_base_cam = np.linalg.inv(T_cam_base)
    points_base = _transform_points(trimmed, T_base_cam)
    if not np.isfinite(points_base).all():
        print("Cube points transformed to base frame contain invalid values.")
        return None, None, mask

    xy_lo = np.percentile(points_base[:, :2], 10.0, axis=0)
    xy_hi = np.percentile(points_base[:, :2], 90.0, axis=0)
    center_base = np.array(
        [
            0.5 * (xy_lo[0] + xy_hi[0]),
            0.5 * (xy_lo[1] + xy_hi[1]),
            float(table_z_m) + float(cube_size_m) * 0.5 + GRASP_Z_BIAS_M,
        ],
        dtype=np.float64,
    )

    cube_yaw = _estimate_cube_yaw_on_table(points_base)
    grasp_yaw = _select_preferred_grasp_yaw(cube_yaw)

    T_base_cube = np.eye(4, dtype=np.float64)
    T_base_cube[:3, :3] = _top_down_rotation(grasp_yaw)
    T_base_cube[:3, 3] = center_base
    T_cam_cube = T_cam_base @ T_base_cube

    print(
        "Cube center base frame:"
        f" x={center_base[0]:.4f}"
        f" y={center_base[1]:.4f}"
        f" z={center_base[2]:.4f}"
    )
    print(
        "Cube yaw:"
        f" observed={math.degrees(cube_yaw):.1f} deg"
        f" grasp={math.degrees(grasp_yaw):.1f} deg"
    )
    return T_base_cube, T_cam_cube, mask


def _detect_duplicate_cube_candidates(
    image: np.ndarray,
    point_cloud: np.ndarray,
    T_cam_robot: np.ndarray,
    cube_prompt: str,
    cube_size_m: float,
    table_z_m: float,
    point_cloud_scale: float,
) -> list[DuplicateCubeCandidate]:
    if image is None or point_cloud is None:
        return []
    if point_cloud.ndim != 3 or point_cloud.shape[-1] < 3:
        raise RuntimeError(f"Invalid point cloud shape: {point_cloud.shape}")
    if image.shape[:2] != point_cloud.shape[:2]:
        raise RuntimeError(f"Image/point cloud shape mismatch: image={image.shape}, point_cloud={point_cloud.shape}")

    cube_color = _cube_color_from_prompt(cube_prompt)
    bgr = _bgr_from_image(image)
    mask = _segment_mask_from_color(bgr, cube_color)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return []

    image_h, image_w = mask.shape[:2]
    cube_diag = float(cube_size_m) * math.sqrt(3.0)
    T_robot_cam = np.linalg.inv(T_cam_robot)
    scored: list[tuple[float, int, DuplicateCubeCandidate]] = []

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px < 40:
            continue

        component_mask = labels == label
        points_cam = _points_from_mask(point_cloud, component_mask, point_cloud_scale)
        if points_cam.shape[0] < MIN_POINTS_AFTER_OUTLIER:
            continue

        trimmed_cam = _statistical_outlier_trim(points_cam, nb_neighbors=12, std_ratio=2.0)
        if trimmed_cam.shape[0] < MIN_POINTS_AFTER_OUTLIER:
            continue

        extent = _robust_extent(trimmed_cam)
        bbox_diag = float(np.linalg.norm(extent))
        max_extent = float(extent.max())
        touches_border = (
            x <= 1
            or y <= 1
            or (x + w) >= (image_w - 1)
            or (y + h) >= (image_h - 1)
        )

        score = abs(bbox_diag - cube_diag)
        score += max(0.0, max_extent - (float(cube_size_m) * 1.8)) * 3.0
        if touches_border:
            score += 0.25

        fit_points_cam = _statistical_outlier_trim(trimmed_cam, nb_neighbors=24, std_ratio=2.5)
        if fit_points_cam.shape[0] < MIN_POINTS_AFTER_OUTLIER:
            continue

        points_robot = _transform_points(fit_points_cam, T_robot_cam)
        if not np.isfinite(points_robot).all():
            continue

        xy_lo = np.percentile(points_robot[:, :2], 10.0, axis=0)
        xy_hi = np.percentile(points_robot[:, :2], 90.0, axis=0)
        center_robot = np.array(
            [
                0.5 * (xy_lo[0] + xy_hi[0]),
                0.5 * (xy_lo[1] + xy_hi[1]),
                float(table_z_m) + float(cube_size_m) * 0.5 + GRASP_Z_BIAS_M,
            ],
            dtype=np.float64,
        )

        cube_yaw = _estimate_cube_yaw_on_table(points_robot)
        grasp_yaw = _select_preferred_grasp_yaw(cube_yaw)
        T_robot_cube = np.eye(4, dtype=np.float64)
        T_robot_cube[:3, :3] = _top_down_rotation(grasp_yaw)
        T_robot_cube[:3, 3] = center_robot
        T_cam_cube = T_cam_robot @ T_robot_cube

        candidate = DuplicateCubeCandidate(
            cube_prompt=cube_prompt,
            cube_color=cube_color,
            instance_index=0,
            component_label=label,
            area_px=area_px,
            score=float(score),
            bbox_diag_m=float(bbox_diag),
            max_extent_m=float(max_extent),
            yaw_rad=float(grasp_yaw),
            T_robot_cube=T_robot_cube,
            T_cam_cube=T_cam_cube,
            center_robot=center_robot,
            member_candidate_indices=(0,),
        )
        scored.append((float(score), label, candidate))

    scored.sort(key=lambda item: (item[0], item[1]))
    candidates: list[DuplicateCubeCandidate] = []
    for instance_index, (_score, _label, candidate) in enumerate(scored, start=1):
        candidates.append(
            DuplicateCubeCandidate(
                cube_prompt=candidate.cube_prompt,
                cube_color=candidate.cube_color,
                instance_index=instance_index,
                component_label=candidate.component_label,
                area_px=candidate.area_px,
                score=candidate.score,
                bbox_diag_m=candidate.bbox_diag_m,
                max_extent_m=candidate.max_extent_m,
                yaw_rad=candidate.yaw_rad,
                T_robot_cube=candidate.T_robot_cube,
                T_cam_cube=candidate.T_cam_cube,
                center_robot=candidate.center_robot,
                member_candidate_indices=(instance_index,),
            )
        )
    return candidates


def _detect_duplicate_target_tag_candidates(
    image: np.ndarray,
    camera_intrinsic: np.ndarray,
    T_cam_robot: np.ndarray,
    target_tag_id: int,
    target_tag_size_m: float,
) -> list[DuplicateTagCandidate]:
    pose_tags = _detect_tags(
        image,
        estimate_pose=True,
        camera_intrinsic=camera_intrinsic,
        tag_size_m=target_tag_size_m,
    )
    detected_ids = [int(tag.tag_id) for tag in pose_tags]
    print(f"AprilTags detected with duplicate-aware target pose solve: {detected_ids}")
    print(f"Target tag size assumption: {float(target_tag_size_m):.4f} m for tag ID {target_tag_id}.")

    T_robot_cam = np.linalg.inv(T_cam_robot)
    scored: list[tuple[float, int, int, DuplicateTagCandidate]] = []
    for detection_index, tag in enumerate(pose_tags):
        if int(tag.tag_id) != int(target_tag_id):
            continue
        if tag.pose_R is None or tag.pose_t is None:
            print(f"Skipping tag {target_tag_id} detection {detection_index}: pose estimation unavailable.")
            continue

        T_cam_tag = np.eye(4, dtype=np.float64)
        T_cam_tag[:3, :3] = np.asarray(tag.pose_R, dtype=np.float64).reshape(3, 3)
        T_cam_tag[:3, 3] = np.asarray(tag.pose_t, dtype=np.float64).reshape(3)
        T_robot_tag = T_robot_cam @ T_cam_tag
        if not np.isfinite(T_robot_tag).all():
            print(f"Skipping tag {target_tag_id} detection {detection_index}: non-finite robot-frame pose.")
            continue

        decision_margin = float(getattr(tag, "decision_margin", 0.0) or 0.0)
        hamming = int(getattr(tag, "hamming", 0) or 0)
        candidate = DuplicateTagCandidate(
            tag_id=int(target_tag_id),
            instance_index=0,
            detection_index=int(detection_index),
            decision_margin=decision_margin,
            hamming=hamming,
            T_robot_tag=T_robot_tag,
            T_cam_tag=T_cam_tag,
            center_robot=np.asarray(T_robot_tag[:3, 3], dtype=np.float64).copy(),
        )
        scored.append((-decision_margin, hamming, detection_index, candidate))

    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    candidates: list[DuplicateTagCandidate] = []
    for instance_index, (_neg_margin, _hamming, _detection_index, candidate) in enumerate(scored, start=1):
        candidates.append(
            DuplicateTagCandidate(
                tag_id=candidate.tag_id,
                instance_index=instance_index,
                detection_index=candidate.detection_index,
                decision_margin=candidate.decision_margin,
                hamming=candidate.hamming,
                T_robot_tag=candidate.T_robot_tag,
                T_cam_tag=candidate.T_cam_tag,
                center_robot=candidate.center_robot,
            )
        )
    return candidates


def _yaw_from_pose_xy(T_base_tag: np.ndarray) -> float:
    x_axis = np.asarray(T_base_tag[:2, 0], dtype=np.float64)
    norm = float(np.linalg.norm(x_axis))
    if norm < 1e-6:
        return 0.0
    return _normalize_angle_pi(math.atan2(float(x_axis[1]), float(x_axis[0])))


def _target_tag_xy_within_workspace(T_base_tag: np.ndarray) -> bool:
    point = np.asarray(T_base_tag[:3, 3], dtype=np.float64)
    return bool(
        (WORKSPACE_MIN_M[0] <= point[0] <= WORKSPACE_MAX_M[0])
        and (WORKSPACE_MIN_M[1] <= point[1] <= WORKSPACE_MAX_M[1])
    )


def _expected_target_tag_z_range(table_z_m: float, target_tag_size_m: float) -> tuple[float, float]:
    lower_margin = max(float(target_tag_size_m), TARGET_TAG_Z_EXPECTED_BELOW_TABLE_M)
    upper_margin = max(float(target_tag_size_m) * 2.0, TARGET_TAG_Z_EXPECTED_ABOVE_TABLE_M)
    return float(table_z_m) - lower_margin, float(table_z_m) + upper_margin


def _constrain_target_tag_pose_for_placement(
    T_base_tag: np.ndarray,
    target_tag_id: int,
    table_z_m: float,
    target_tag_size_m: float,
) -> tuple[np.ndarray, bool]:
    if _is_calibration_tag_id(int(target_tag_id)):
        return T_base_tag, False

    raw_point = np.asarray(T_base_tag[:3, 3], dtype=np.float64)
    z_min, z_max = _expected_target_tag_z_range(table_z_m, target_tag_size_m)
    xy_ok = _target_tag_xy_within_workspace(T_base_tag)
    z_ok = bool(z_min <= raw_point[2] <= z_max)

    if xy_ok and not z_ok:
        constrained = T_base_tag.copy()
        constrained[2, 3] = float(table_z_m)
        print(
            "Target AprilTag raw z is inconsistent with the table plane; "
            f"keeping detected x/y/yaw and replacing z {raw_point[2]:.4f} -> {constrained[2, 3]:.4f} m "
            f"(expected near [{z_min:.4f}, {z_max:.4f}] m around table_z_m={float(table_z_m):.4f})."
        )
        print("Using table-constrained target pose for placement instead of rejecting the plan.")
        return constrained, True

    if xy_ok:
        print(
            "Target AprilTag z is within the expected table-adjacent range:"
            f" z={raw_point[2]:.4f} m expected=[{z_min:.4f}, {z_max:.4f}] m."
        )
    else:
        print(
            "Target AprilTag x/y is outside the local workspace; "
            "keeping the raw pose so the existing guardrail can reject it."
        )
    return T_base_tag, False


def construct_place_pose(
    T_base_tag: np.ndarray,
    cube_size_m: float,
    table_z_m: float,
    place_x_offset_m: float,
    place_y_offset_m: float,
) -> np.ndarray:
    target_yaw = _select_preferred_grasp_yaw(_yaw_from_pose_xy(T_base_tag))
    T_base_place = np.eye(4, dtype=np.float64)
    T_base_place[:3, :3] = _top_down_rotation(target_yaw)
    T_base_place[:3, 3] = np.array(
        [
            float(T_base_tag[0, 3]) + float(place_x_offset_m),
            float(T_base_tag[1, 3]) + float(place_y_offset_m),
            float(table_z_m) + float(cube_size_m) * 0.5 + GRASP_Z_BIAS_M,
        ],
        dtype=np.float64,
    )
    print(
        "Place pose base frame:"
        f" x={T_base_place[0, 3]:.4f}"
        f" y={T_base_place[1, 3]:.4f}"
        f" z={T_base_place[2, 3]:.4f}"
        f" yaw={math.degrees(target_yaw):.1f} deg"
    )
    return T_base_place


def draw_pose_axes(image: np.ndarray, camera_intrinsic: np.ndarray, pose: np.ndarray, size: float = 0.05) -> None:
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    tvec = pose[:3, 3]
    frame_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ) * float(size)
    image_points, _ = cv2.projectPoints(frame_points, rvec, tvec, camera_intrinsic, None)
    image_points = np.round(image_points).astype(int)
    origin = tuple(image_points[0].ravel())
    unit_x = tuple(image_points[1].ravel())
    unit_y = tuple(image_points[2].ravel())
    unit_z = tuple(image_points[3].ravel())
    cv2.line(image, origin, unit_x, (0, 0, 255), 2)
    cv2.line(image, origin, unit_y, (0, 255, 0), 2)
    cv2.line(image, origin, unit_z, (255, 0, 0), 2)


def _project_origin(camera_intrinsic: np.ndarray, pose: np.ndarray) -> tuple[int, int] | None:
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    tvec = pose[:3, 3]
    point, _ = cv2.projectPoints(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        rvec,
        tvec,
        camera_intrinsic,
        None,
    )
    px = np.round(point[0, 0]).astype(int)
    if not np.isfinite(px).all():
        return None
    return int(px[0]), int(px[1])


def _draw_label(image: np.ndarray, camera_intrinsic: np.ndarray, pose: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
    origin = _project_origin(camera_intrinsic, pose)
    if origin is None:
        return
    cv2.circle(image, origin, 5, color, -1)
    cv2.putText(
        image,
        label,
        (origin[0] + 8, origin[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def _gui_display_available() -> bool:
    if os.name != "posix":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _build_confirmation_overlay(
    image: np.ndarray,
    camera_intrinsic: np.ndarray,
    T_cam_cube: np.ndarray,
    T_cam_tag: np.ndarray,
    T_cam_place: np.ndarray,
    mask: np.ndarray,
    cube_size_m: float,
    target_tag_size_m: float,
) -> np.ndarray:
    display = _bgr_from_image(image)
    if mask.shape[:2] == display.shape[:2]:
        overlay = display.copy()
        overlay[mask > 0] = (0, 180, 255)
        display = cv2.addWeighted(overlay, 0.22, display, 0.78, 0.0)

    draw_pose_axes(display, camera_intrinsic, T_cam_cube, size=float(cube_size_m) * 1.5)
    draw_pose_axes(display, camera_intrinsic, T_cam_tag, size=float(target_tag_size_m) * 0.75)
    draw_pose_axes(display, camera_intrinsic, T_cam_place, size=float(cube_size_m) * 1.5)
    _draw_label(display, camera_intrinsic, T_cam_cube, "cube grasp", (0, 255, 255))
    _draw_label(display, camera_intrinsic, T_cam_tag, "target tag", (255, 255, 0))
    _draw_label(display, camera_intrinsic, T_cam_place, "place pose", (255, 0, 255))
    return display


def _save_preview_image(display: np.ndarray, preview_path: str) -> Path | None:
    output_path = Path(preview_path).expanduser()
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), display):
        print(f"Warning: failed to write preview image to {output_path}")
        return None
    return output_path


def show_confirmation(
    image: np.ndarray,
    camera_intrinsic: np.ndarray,
    T_cam_cube: np.ndarray,
    T_cam_tag: np.ndarray,
    T_cam_place: np.ndarray,
    mask: np.ndarray,
    cube_size_m: float,
    target_tag_size_m: float,
    no_gui: bool,
    preview_path: str,
    auto_confirm: bool = False,
) -> bool:
    display = _build_confirmation_overlay(
        image=image,
        camera_intrinsic=camera_intrinsic,
        T_cam_cube=T_cam_cube,
        T_cam_tag=T_cam_tag,
        T_cam_place=T_cam_place,
        mask=mask,
        cube_size_m=cube_size_m,
        target_tag_size_m=target_tag_size_m,
    )
    saved_path = _save_preview_image(display, preview_path)
    if saved_path is not None:
        print(f"Saved confirmation preview: {saved_path}")

    if auto_confirm:
        print("Auto-confirm enabled; skipping interactive confirmation.")
        return True

    use_gui = (not no_gui) and _gui_display_available()
    if not use_gui:
        if no_gui:
            print("GUI disabled by --no_gui; using terminal confirmation.")
        else:
            print("No GUI display detected; using terminal confirmation.")
        try:
            response = input("Type 'k' then Enter to execute, or anything else to cancel: ")
        except EOFError:
            print("No terminal input is available. Cancelling without robot motion.")
            return False
        return response.strip().lower() == "k"

    cv2.namedWindow("mini_task confirmation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("mini_task confirmation", 1280, 720)
    cv2.imshow("mini_task confirmation", display)
    print("Press 'k' in the OpenCV window to execute. Press 'q' or Esc to cancel.")
    key = cv2.waitKey(0) & 0xFF
    try:
        cv2.destroyWindow("mini_task confirmation")
    except cv2.error as exc:
        print(f"Warning: failed to destroy confirmation window cleanly: {exc}")
    return key == ord("k")


def _pose_to_mm_deg(pose: np.ndarray) -> tuple[float, float, float, float, float, float]:
    px, py, pz = pose[:3, 3]
    roll, pitch, yaw = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
    return px * 1000.0, py * 1000.0, pz * 1000.0, float(roll), float(pitch), float(yaw)


def _offset_pose_z(pose: np.ndarray, offset_m: float) -> np.ndarray:
    moved = pose.copy()
    moved[2, 3] += float(offset_m)
    return moved


def _check_arm_code(code: Any, action: str) -> None:
    if code is None:
        return
    if isinstance(code, tuple):
        code = code[0]
    if int(code) != 0:
        raise RuntimeError(f"Lite6 command failed during {action}: code={code}")


def _split_arm_response(response: Any) -> tuple[int | None, Any]:
    if isinstance(response, tuple):
        if len(response) >= 2 and isinstance(response[0], (int, np.integer)):
            return int(response[0]), response[1]
        if len(response) == 1:
            return None, response[0]
    return None, response


def _read_arm_value(arm: Any, method_name: str) -> tuple[Any, Any]:
    method = getattr(arm, method_name, None)
    if method is None:
        raise RuntimeError(f"Lite6 SDK object does not provide {method_name}().")
    raw = method()
    code, value = _split_arm_response(raw)
    if code is not None and code != 0:
        raise RuntimeError(f"Lite6 status read failed during {method_name}(): code={code}, response={raw!r}")
    return value, raw


def _as_int_status(value: Any, name: str) -> int:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise RuntimeError(f"Unable to parse Lite6 {name}: {value!r}")
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Unable to parse Lite6 {name}: {value!r}") from exc


def _parse_err_warn(value: Any) -> tuple[int, int]:
    if isinstance(value, dict):
        err = value.get("err", value.get("error", value.get("error_code", 0)))
        warn = value.get("warn", value.get("warning", value.get("warn_code", 0)))
        return int(err), int(warn)

    if isinstance(value, (int, np.integer)):
        return int(value), 0

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) < 2:
            raise RuntimeError(f"Unable to parse Lite6 err/warn code: {value!r}")
        return int(value[0]), int(value[1])

    raise RuntimeError(f"Unable to parse Lite6 err/warn code: {value!r}")


def _read_and_print_arm_status(arm: Any, label: str) -> tuple[int, int, int]:
    state_value, state_raw = _read_arm_value(arm, "get_state")
    err_warn_value, err_warn_raw = _read_arm_value(arm, "get_err_warn_code")
    _position_value, position_raw = _read_arm_value(arm, "get_position")
    _servo_angle_value, servo_angle_raw = _read_arm_value(arm, "get_servo_angle")

    state = _as_int_status(state_value, "state")
    err, warn = _parse_err_warn(err_warn_value)

    print(f"xArm status {label}:")
    print(f"  get_state(): {state_raw!r}")
    print(f"  get_err_warn_code(): {err_warn_raw!r}")
    print(f"  get_position(): {position_raw!r}")
    print(f"  get_servo_angle(): {servo_angle_raw!r}")
    return state, err, warn


def _parse_pose_response_mm_deg(value: Any, name: str) -> tuple[float, float, float, float, float, float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) < 6:
        raise RuntimeError(f"Unable to parse Lite6 {name}: {value!r}")
    return tuple(float(item) for item in value[:6])


def _read_current_tcp_pose_mm_deg(arm: Any) -> tuple[float, float, float, float, float, float]:
    position_value, position_raw = _read_arm_value(arm, "get_position")
    pose = _parse_pose_response_mm_deg(position_value, "TCP pose")
    print(f"Current TCP pose before first planned move: {position_raw!r}")
    return pose


def _ensure_arm_ready(arm: Any, label: str, recover_mode_changed: bool = False) -> None:
    state, err, warn = _read_and_print_arm_status(arm, label)

    if err != 0 or warn != 0:
        raise RuntimeError(
            f"Lite6 is not ready {label}: err={err}, warn={warn}. "
            "Refusing to send motion commands."
        )

    if state == 5 and recover_mode_changed:
        print(f"xArm state is 5 {label}; calling set_state(0) and re-checking.")
        _check_arm_code(arm.set_state(0), "recover state 5")
        time.sleep(0.5)
        state, err, warn = _read_and_print_arm_status(arm, f"{label} after set_state(0)")
        if err != 0 or warn != 0:
            raise RuntimeError(
                f"Lite6 is not ready {label} after set_state(0): err={err}, warn={warn}. "
                "Refusing to send motion commands."
            )

    if state == 2:
        print("Lite6 ready: state=2, err/warn=[0, 0]")
        return

    if state == 0:
        return

    if state == 3:
        raise RuntimeError(
            f"Lite6 is not ready {label}: state=3 PAUSE state. "
            "Refusing to send motion commands."
        )

    if state == 4:
        raise RuntimeError(
            f"Lite6 is not ready {label}: state=4 STOP state. "
            "Refusing to send motion commands."
        )

    raise RuntimeError(
        f"Lite6 is not ready {label}: state={state}. "
        "Refusing to send motion commands."
    )


def _move_pose_mm_deg(
    arm: Any,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    speed_mm_s: float,
    name: str,
) -> None:
    _ensure_arm_ready(arm, f"before {name}")
    print(
        f"Moving {name}:"
        f" x={x:.1f} y={y:.1f} z={z:.1f}"
        f" roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f}"
    )
    code = arm.set_position(
        x=x,
        y=y,
        z=z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        speed=float(speed_mm_s),
        wait=True,
    )
    _check_arm_code(code, name)


def _move_pose(arm: Any, pose: np.ndarray, speed_mm_s: float, name: str) -> None:
    x, y, z, roll, pitch, yaw = _pose_to_mm_deg(pose)
    _move_pose_mm_deg(arm, x, y, z, roll, pitch, yaw, speed_mm_s, name)


def _point_in_bounds_mm(point_mm: np.ndarray, minimum_mm: np.ndarray, maximum_mm: np.ndarray) -> bool:
    return bool(np.all(point_mm >= minimum_mm) and np.all(point_mm <= maximum_mm))


def _format_mm_bounds(minimum_mm: np.ndarray, maximum_mm: np.ndarray) -> str:
    return (
        f"x=[{minimum_mm[0]:.0f}, {maximum_mm[0]:.0f}]mm "
        f"y=[{minimum_mm[1]:.0f}, {maximum_mm[1]:.0f}]mm "
        f"z=[{minimum_mm[2]:.0f}, {maximum_mm[2]:.0f}]mm"
    )


def _forward_staging_targets(
    current_pose: tuple[float, float, float, float, float, float],
    forward_axis: str,
    forward_sign: str,
    forward_step_mm: float,
    forward_steps: int,
) -> list[tuple[float, float, float, float, float, float]]:
    direction = 1.0 if forward_sign == "pos" else -1.0
    targets = []
    for index in range(1, int(forward_steps) + 1):
        x, y, z, roll, pitch, yaw = current_pose
        offset_mm = direction * float(forward_step_mm) * float(index)
        if forward_axis == "x":
            x += offset_mm
        else:
            y += offset_mm
        targets.append((x, y, z, roll, pitch, yaw))
    return targets


def _run_forward_first_move_staging(
    arm: Any,
    pregrasp: np.ndarray,
    stage_minimum_mm: np.ndarray,
    stage_maximum_mm: np.ndarray,
    forward_axis: str,
    forward_sign: str,
    forward_step_mm: float,
    forward_steps: int,
    forward_stage_speed_mm_s: float,
) -> None:
    current_pose = _read_current_tcp_pose_mm_deg(arm)
    current_x, current_y, current_z, current_roll, current_pitch, current_yaw = current_pose
    pre_x, pre_y, pre_z, _pre_roll, _pre_pitch, _pre_yaw = _pose_to_mm_deg(pregrasp)
    distance_mm = float(np.linalg.norm(np.array([current_x - pre_x, current_y - pre_y, current_z - pre_z])))
    print(
        "Current TCP before forward staging:"
        f" x={current_x:.1f} y={current_y:.1f} z={current_z:.1f}"
        f" roll={current_roll:.1f} pitch={current_pitch:.1f} yaw={current_yaw:.1f}"
    )
    print(f"Current TCP to pregrasp distance: {distance_mm:.1f} mm")
    print(
        "First-move forward staging gate:"
        f" bounds={_format_mm_bounds(stage_minimum_mm, stage_maximum_mm)}"
        f" axis={forward_axis}"
        f" sign={forward_sign}"
        f" step_mm={float(forward_step_mm):.1f}"
        f" steps={int(forward_steps)}"
        f" speed_mm_s={float(forward_stage_speed_mm_s):.1f}"
    )

    targets = _forward_staging_targets(
        current_pose=current_pose,
        forward_axis=forward_axis,
        forward_sign=forward_sign,
        forward_step_mm=forward_step_mm,
        forward_steps=forward_steps,
    )
    for index, target in enumerate(targets, start=1):
        point = np.array(target[:3], dtype=np.float64)
        print(
            f"Forward staging target {index}/{len(targets)}:"
            f" x={target[0]:.1f} y={target[1]:.1f} z={target[2]:.1f}"
            f" roll={target[3]:.1f} pitch={target[4]:.1f} yaw={target[5]:.1f}"
        )
        if not _point_in_bounds_mm(point, stage_minimum_mm, stage_maximum_mm):
            raise RuntimeError(
                "Forward first-move staging target is outside active staging bounds. "
                f"target={np.array2string(point, precision=1)}mm "
                f"bounds={_format_mm_bounds(stage_minimum_mm, stage_maximum_mm)}. "
                "Refusing to send risky staging motion."
            )

    for index, target in enumerate(targets, start=1):
        print(
            f"Executing forward staging stage {index}/{len(targets)}:"
            f" axis={forward_axis}"
            f" sign={forward_sign}"
            f" step_mm={float(forward_step_mm):.1f}"
            f" speed_mm_s={float(forward_stage_speed_mm_s):.1f}"
        )
        try:
            _move_pose_mm_deg(
                arm,
                target[0],
                target[1],
                target[2],
                target[3],
                target[4],
                target[5],
                forward_stage_speed_mm_s,
                f"forward staging stage {index}/{len(targets)}",
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Forward staging stage {index}/{len(targets)} failed: {exc}") from exc
        _ensure_arm_ready(arm, f"after forward staging stage {index}/{len(targets)}", recover_mode_changed=True)


def _run_cartesian_first_move_staging(
    arm: Any,
    pregrasp: np.ndarray,
    speed_mm_s: float,
    stage_minimum_mm: np.ndarray,
    stage_maximum_mm: np.ndarray,
    stage_z_mm: float,
) -> None:
    current_x, current_y, current_z, current_roll, current_pitch, current_yaw = _read_current_tcp_pose_mm_deg(arm)
    pre_x, pre_y, pre_z, _pre_roll, _pre_pitch, _pre_yaw = _pose_to_mm_deg(pregrasp)
    distance_mm = float(np.linalg.norm(np.array([current_x - pre_x, current_y - pre_y, current_z - pre_z])))
    print(f"Current TCP to pregrasp distance: {distance_mm:.1f} mm")
    print(
        "First-move cartesian staging gate:"
        f" bounds={_format_mm_bounds(stage_minimum_mm, stage_maximum_mm)}"
        f" stage_z_mm={stage_z_mm:.1f}"
    )

    if distance_mm <= FIRST_MOVE_STAGING_DISTANCE_MM:
        print("Cartesian first-move staging skipped because the planned pregrasp is within the staging threshold.")
        return

    staging_z = max(current_z, pre_z, float(stage_z_mm))
    staging_point = np.array([current_x, current_y, staging_z], dtype=np.float64)
    if not _point_in_bounds_mm(staging_point, stage_minimum_mm, stage_maximum_mm):
        raise RuntimeError(
            "Cannot construct a conservative first-move intermediate pose. "
            f"candidate={np.array2string(staging_point, precision=1)}mm "
            f"bounds={_format_mm_bounds(stage_minimum_mm, stage_maximum_mm)}. "
            "Move the cube/tag closer to the workspace center, place the robot TCP in a safer start pose, "
            "and rerun dry_run."
        )

    print(
        "Using conservative cartesian first-move staging pose:"
        f" x={current_x:.1f} y={current_y:.1f} z={staging_z:.1f}"
        f" roll={current_roll:.1f} pitch={current_pitch:.1f} yaw={current_yaw:.1f}"
    )
    _move_pose_mm_deg(
        arm,
        current_x,
        current_y,
        staging_z,
        current_roll,
        current_pitch,
        current_yaw,
        speed_mm_s,
        "cartesian first-move staging pose",
    )


def _stage_first_move_if_needed(
    arm: Any,
    pregrasp: np.ndarray,
    speed_mm_s: float,
    first_move_strategy: str,
    stage_minimum_mm: np.ndarray,
    stage_maximum_mm: np.ndarray,
    stage_z_mm: float,
    forward_axis: str,
    forward_sign: str,
    forward_step_mm: float,
    forward_steps: int,
    forward_stage_speed_mm_s: float,
) -> None:
    if first_move_strategy == "none":
        print("First-move staging skipped because --first_move_strategy none was selected.")
        return
    if first_move_strategy == "forward_stage":
        _run_forward_first_move_staging(
            arm=arm,
            pregrasp=pregrasp,
            stage_minimum_mm=stage_minimum_mm,
            stage_maximum_mm=stage_maximum_mm,
            forward_axis=forward_axis,
            forward_sign=forward_sign,
            forward_step_mm=forward_step_mm,
            forward_steps=forward_steps,
            forward_stage_speed_mm_s=forward_stage_speed_mm_s,
        )
        return
    if first_move_strategy == "cartesian_stage":
        _run_cartesian_first_move_staging(
            arm=arm,
            pregrasp=pregrasp,
            speed_mm_s=speed_mm_s,
            stage_minimum_mm=stage_minimum_mm,
            stage_maximum_mm=stage_maximum_mm,
            stage_z_mm=stage_z_mm,
        )
        return
    raise RuntimeError(f"Unknown first_move_strategy: {first_move_strategy!r}")


def _stop_gripper_if_supported(arm: Any, action: str) -> None:
    if not hasattr(arm, "stop_lite6_gripper"):
        return
    _check_arm_code(arm.stop_lite6_gripper(), f"stop gripper after {action}")
    time.sleep(0.2)


def _open_gripper(arm: Any, settle_s: float) -> None:
    _ensure_arm_ready(arm, "before open gripper")
    _check_arm_code(arm.open_lite6_gripper(sync=True), "open gripper")
    time.sleep(float(settle_s))
    _stop_gripper_if_supported(arm, "open")
    _ensure_arm_ready(arm, "after open gripper", recover_mode_changed=True)


def _close_gripper(arm: Any, settle_s: float) -> None:
    _ensure_arm_ready(arm, "before close gripper")
    _check_arm_code(arm.close_lite6_gripper(sync=True), "close gripper")
    time.sleep(float(settle_s))
    _stop_gripper_if_supported(arm, "close")
    _ensure_arm_ready(arm, "after close gripper", recover_mode_changed=True)


def connect_lite6(robot_ip: str, move_home_before_task: bool) -> Any:
    from xarm.wrapper import XArmAPI

    print(f"Connecting to Lite6 at {robot_ip}...")
    arm = XArmAPI(robot_ip)
    try:
        _check_arm_code(arm.connect(), "connect")
        if getattr(arm, "connected", True) is False:
            raise RuntimeError("Lite6 SDK reports that the arm is not connected.")

        if hasattr(arm, "clean_warn"):
            _check_arm_code(arm.clean_warn(), "clean warnings")
        if hasattr(arm, "clean_error"):
            _check_arm_code(arm.clean_error(), "clean errors")

        _check_arm_code(arm.motion_enable(enable=True), "motion enable")
        _check_arm_code(arm.set_mode(0), "set mode")
        _check_arm_code(arm.set_state(0), "set state")
        time.sleep(0.5)
        _ensure_arm_ready(arm, "before TCP offset setup", recover_mode_changed=True)

        _check_arm_code(arm.set_tcp_offset([0, 0, GRIPPER_LENGTH_MM, 0, 0, 0]), "set TCP offset")
        time.sleep(0.2)
        _ensure_arm_ready(arm, "after TCP offset setup", recover_mode_changed=True)

        if move_home_before_task:
            print("Moving to xArm home before the planned approach.")
            _ensure_arm_ready(arm, "before move_gohome")
            _check_arm_code(arm.move_gohome(wait=True), "move home")
            time.sleep(0.5)
            _ensure_arm_ready(arm, "after move_gohome", recover_mode_changed=True)
        else:
            print("Skipping xArm move_gohome before task; using current robot state.")
            _ensure_arm_ready(arm, "before planned task motion", recover_mode_changed=True)
    except Exception:
        _safe_disconnect(arm)
        raise
    return arm


def execute_pick_place(
    arm: Any,
    T_base_cube: np.ndarray,
    T_base_place: np.ndarray,
    approach_height_m: float,
    retreat_height_m: float,
    speed_mm_s: float,
    gripper_settle_s: float,
    motion_profile: str,
    first_move_strategy: str,
    stage_minimum_mm: np.ndarray,
    stage_maximum_mm: np.ndarray,
    stage_z_mm: float,
    forward_axis: str,
    forward_sign: str,
    forward_step_mm: float,
    forward_steps: int,
    forward_stage_speed_mm_s: float,
) -> None:
    pregrasp = _offset_pose_z(T_base_cube, approach_height_m)
    retreat_after_grasp = _offset_pose_z(T_base_cube, retreat_height_m)
    preplace = _offset_pose_z(T_base_place, approach_height_m)
    retreat_after_place = _offset_pose_z(T_base_place, retreat_height_m)

    print("Checkpoint motion poses:")
    _print_pose("Cube pregrasp", pregrasp)
    _print_pose("Cube grasp", T_base_cube)
    _print_pose("Target preplace", preplace)
    _print_pose("Target place", T_base_place)

    if motion_profile == "checkpoint":
        print("Using checkpoint-compatible motion profile.")
        print("Skipping all first-move staging.")
    else:
        print("Using experimental motion profile.")

    _open_gripper(arm, gripper_settle_s)

    if motion_profile == "checkpoint":
        print("Moving directly to cube pregrasp.")
    else:
        _stage_first_move_if_needed(
            arm,
            pregrasp,
            speed_mm_s,
            first_move_strategy,
            stage_minimum_mm,
            stage_maximum_mm,
            stage_z_mm,
            forward_axis,
            forward_sign,
            forward_step_mm,
            forward_steps,
            forward_stage_speed_mm_s,
        )

    try:
        _move_pose(arm, pregrasp, speed_mm_s, "cube pregrasp")
    except RuntimeError as exc:
        raise RuntimeError(f"checkpoint motion failed at cube pregrasp: {exc}") from exc
    _move_pose(arm, T_base_cube, speed_mm_s, "cube grasp")
    _close_gripper(arm, gripper_settle_s)
    _move_pose(arm, retreat_after_grasp, speed_mm_s, "cube pregrasp after grasp")
    _move_pose(arm, preplace, speed_mm_s, "target preplace")
    _move_pose(arm, T_base_place, speed_mm_s, "target place")
    _open_gripper(arm, gripper_settle_s)
    _move_pose(arm, retreat_after_place, speed_mm_s, "target preplace after place")


def _print_pose(name: str, pose: np.ndarray) -> None:
    x, y, z, roll, pitch, yaw = _pose_to_mm_deg(pose)
    print(
        f"{name}:"
        f" x={x:.1f}mm"
        f" y={y:.1f}mm"
        f" z={z:.1f}mm"
        f" roll={roll:.1f}deg"
        f" pitch={pitch:.1f}deg"
        f" yaw={yaw:.1f}deg"
    )


def _mini_task_contact_point_mm(pose: np.ndarray) -> np.ndarray:
    x, y, z, _roll, _pitch, _yaw = _pose_to_mm_deg(pose)
    return np.array([x, y, z], dtype=np.float64)


def _task_gate_bounds_from_args(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    minimum_mm = np.array([args.task_x_min_mm, args.task_y_min_mm, args.task_z_min_mm], dtype=np.float64)
    maximum_mm = np.array([args.task_x_max_mm, args.task_y_max_mm, args.task_z_max_mm], dtype=np.float64)
    if np.any(minimum_mm > maximum_mm):
        raise ValueError(
            "Invalid mini-task conservative gate bounds: "
            f"minimum={np.array2string(minimum_mm, precision=1)}mm "
            f"maximum={np.array2string(maximum_mm, precision=1)}mm"
        )
    return minimum_mm, maximum_mm


def _stage_gate_from_args(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float]:
    minimum_mm = np.array([args.stage_x_min_mm, args.stage_y_min_mm, args.stage_z_min_mm], dtype=np.float64)
    maximum_mm = np.array([args.stage_x_max_mm, args.stage_y_max_mm, args.stage_z_max_mm], dtype=np.float64)
    stage_z_mm = float(args.stage_z_mm)
    if np.any(minimum_mm > maximum_mm):
        raise ValueError(
            "Invalid first-move staging gate bounds: "
            f"minimum={np.array2string(minimum_mm, precision=1)}mm "
            f"maximum={np.array2string(maximum_mm, precision=1)}mm"
        )
    if args.motion_profile == "checkpoint":
        return minimum_mm, maximum_mm, stage_z_mm
    if args.first_move_strategy == "cartesian_stage" and not (minimum_mm[2] <= stage_z_mm <= maximum_mm[2]):
        raise ValueError(
            "Invalid first-move staging z: "
            f"stage_z_mm={stage_z_mm:.1f} must be within z bounds "
            f"[{minimum_mm[2]:.1f}, {maximum_mm[2]:.1f}]mm"
        )
    if args.first_move_strategy == "forward_stage":
        if float(args.forward_step_mm) <= 0.0:
            raise ValueError("--forward_step_mm must be positive for forward_stage.")
        if int(args.forward_steps) < 1:
            raise ValueError("--forward_steps must be at least 1 for forward_stage.")
        if float(args.forward_stage_speed_mm_s) <= 0.0:
            raise ValueError("--forward_stage_speed_mm_s must be positive for forward_stage.")
    return minimum_mm, maximum_mm, stage_z_mm


def _print_first_move_configuration(
    args: argparse.Namespace,
    stage_minimum_mm: np.ndarray,
    stage_maximum_mm: np.ndarray,
    stage_z_mm: float,
    dry_run: bool,
) -> None:
    print("First-move staging configuration:")
    print(f"  motion_profile={args.motion_profile}")
    if args.motion_profile == "checkpoint":
        print("  checkpoint profile ignores --first_move_strategy and skips all first-move staging.")
        if dry_run:
            print(
                "Dry run first-move staging preview: checkpoint profile does not connect to the robot "
                "and will move directly to cube pregrasp during real execution."
            )
        return
    print(f"  first_move_strategy={args.first_move_strategy}")
    print(f"  staging bounds={_format_mm_bounds(stage_minimum_mm, stage_maximum_mm)}")
    if args.first_move_strategy == "forward_stage":
        print(
            "  forward staging:"
            f" axis={args.forward_axis}"
            f" sign={args.forward_sign}"
            f" step_mm={float(args.forward_step_mm):.1f}"
            f" steps={int(args.forward_steps)}"
            f" speed_mm_s={float(args.forward_stage_speed_mm_s):.1f}"
        )
    elif args.first_move_strategy == "cartesian_stage":
        print(f"  cartesian staging: stage_z_mm={stage_z_mm:.1f}")
    else:
        print("  staging will be skipped by strategy.")
    if dry_run:
        print(
            "Dry run first-move staging preview: current TCP cannot be read in dry_run unless robot is connected; "
            "real staging will be evaluated during real execution."
        )


def _mini_task_contact_pose_ok(
    name: str,
    pose: np.ndarray,
    minimum_mm: np.ndarray,
    maximum_mm: np.ndarray,
) -> bool:
    point_mm = _mini_task_contact_point_mm(pose)
    ok = _point_in_bounds_mm(point_mm, minimum_mm, maximum_mm)
    status = "PASS" if ok else "FAIL"
    print(
        f"  {name}: {status}"
        f" point={np.array2string(point_mm, precision=1)}mm"
        f" bounds={_format_mm_bounds(minimum_mm, maximum_mm)}"
    )
    return ok


def _mini_task_conservative_execution_ok(
    T_base_cube: np.ndarray,
    T_base_place: np.ndarray,
    minimum_mm: np.ndarray,
    maximum_mm: np.ndarray,
) -> bool:
    print("Mini-task conservative execution gate:")
    print(
        "  Contact-pose bounds: "
        f"{_format_mm_bounds(minimum_mm, maximum_mm)}"
    )
    cube_ok = _mini_task_contact_pose_ok("Cube grasp pose", T_base_cube, minimum_mm, maximum_mm)
    place_ok = _mini_task_contact_pose_ok("Target place pose", T_base_place, minimum_mm, maximum_mm)
    ok = cube_ok and place_ok
    if ok:
        print("Mini-task conservative execution gate PASS.")
    else:
        print(
            "Mini-task conservative execution gate FAILED. "
            "Move the cube/tag closer to the workspace center and rerun dry_run before real execution."
        )
    return ok


def _workspace_pose_ok(name: str, pose: np.ndarray, z_extra_m: float = 0.0) -> bool:
    points = [pose[:3, 3].copy()]
    lifted = pose[:3, 3].copy()
    lifted[2] += float(z_extra_m)
    points.append(lifted)

    ok = True
    for point in points:
        if np.any(point < WORKSPACE_MIN_M) or np.any(point > WORKSPACE_MAX_M):
            print(
                f"{name} is outside the local workspace guardrail:"
                f" point={np.array2string(point, precision=4)}"
                f" min={np.array2string(WORKSPACE_MIN_M, precision=4)}"
                f" max={np.array2string(WORKSPACE_MAX_M, precision=4)}"
            )
            ok = False
    return ok


def _target_tag_pose_ok(name: str, pose: np.ndarray) -> bool:
    point = pose[:3, 3].copy()
    minimum = WORKSPACE_MIN_M.copy()
    maximum = WORKSPACE_MAX_M.copy()
    minimum[2] = -0.05
    maximum[2] = 0.20
    if np.any(point < minimum) or np.any(point > maximum):
        print(
            f"{name} is outside the local target-tag guardrail:"
            f" point={np.array2string(point, precision=4)}"
            f" min={np.array2string(minimum, precision=4)}"
            f" max={np.array2string(maximum, precision=4)}"
        )
        print(
            "Check --tag_id and --target_tag_size_m. For non-calibration tags, "
            "a wrong printed tag size scales the recovered single-tag pose."
        )
        return False
    return True


def _arm_return_code(response: Any) -> int:
    if response is None:
        return 0
    if isinstance(response, tuple):
        if not response:
            return 0
        response = response[0]
    return int(response)


def _checkpoint8_call_optional(arm: Any, method_name: str, action: str) -> None:
    method = getattr(arm, method_name, None)
    if method is None:
        return
    _check_arm_code(method(), action)


def _checkpoint8_status_ready(state: int | None, err: int | None, warn: int | None) -> bool:
    return state in (0, 2) and err == 0 and warn == 0


def _checkpoint8_status_bad_message(state: int | None, err: int | None, warn: int | None) -> str:
    return f"state={state}, err={err}, warn={warn}"


def _checkpoint8_print_status(arm: Any, label: str) -> tuple[int | None, int | None, int | None]:
    try:
        return _read_and_print_arm_status(arm, label)
    except Exception as exc:
        print(f"Warning: failed to read complete xArm status {label}: {exc}")
        return None, None, None


def _checkpoint8_require_ready(arm: Any, label: str) -> None:
    state, err, warn = _checkpoint8_print_status(arm, label)
    if _checkpoint8_status_ready(state, err, warn):
        return
    if state == 4 or err == 22:
        print(
            "Robot is in STOP/C22 after checkpoint8-style motion. "
            "Run --recover_robot, then --home_only before retrying."
        )
    raise RuntimeError(
        f"Lite6 is not ready {label}: "
        f"{_checkpoint8_status_bad_message(state, err, warn)}"
    )


def _checkpoint8_recover_connected_arm(arm: Any) -> None:
    _checkpoint8_call_optional(arm, "clean_warn", "clean warnings")
    _checkpoint8_call_optional(arm, "clean_error", "clean errors")
    _check_arm_code(arm.motion_enable(enable=True), "motion enable")
    _check_arm_code(arm.set_mode(0), "set mode")
    _check_arm_code(arm.set_state(0), "set state")
    time.sleep(0.5)


def _checkpoint8_connect_arm(robot_ip: str) -> Any:
    from xarm.wrapper import XArmAPI

    print(f"Connecting to Lite6 at {robot_ip}...")
    arm = XArmAPI(robot_ip)
    _check_arm_code(arm.connect(), "connect")
    if getattr(arm, "connected", True) is False:
        raise RuntimeError("Lite6 SDK reports that the arm is not connected.")
    return arm


def _checkpoint8_initialize_and_home(arm: Any, gripper_length_mm: float) -> None:
    _checkpoint8_call_optional(arm, "clean_warn", "clean warnings")
    _checkpoint8_call_optional(arm, "clean_error", "clean errors")
    _check_arm_code(arm.motion_enable(enable=True), "motion enable")
    _check_arm_code(arm.set_tcp_offset([0, 0, float(gripper_length_mm), 0, 0, 0]), "set TCP offset")
    _check_arm_code(arm.set_mode(0), "set mode")
    _check_arm_code(arm.set_state(0), "set state")
    time.sleep(0.5)

    _checkpoint8_print_status(arm, "before checkpoint8 initial move_gohome")
    print("Moving to xArm home before checkpoint8-style task.")
    home_code = arm.move_gohome(wait=True)
    print(f"move_gohome return code: {home_code!r}")
    if _arm_return_code(home_code) != 0:
        _checkpoint8_print_status(arm, "after failed checkpoint8 initial move_gohome")
        print(
            "move_gohome failed. The robot is not in a valid state to start checkpoint-style tasks. "
            "Recover the robot in xArm Studio or power-cycle/re-enable, then retry."
        )
        raise RuntimeError("checkpoint8_style initial move_gohome failed")
    time.sleep(0.5)
    _checkpoint8_require_ready(arm, "after checkpoint8 initial move_gohome")


def _run_checkpoint8_recover_robot(robot_ip: str) -> None:
    arm: Any | None = None
    try:
        print("Running checkpoint8-style recovery helper.")
        print("This clears warn/error and re-enables motion without moving home.")
        arm = _checkpoint8_connect_arm(robot_ip)
        _checkpoint8_print_status(arm, "before recovery")
        _checkpoint8_recover_connected_arm(arm)
        _checkpoint8_print_status(arm, "after recovery")
    finally:
        if arm is not None:
            try:
                arm.disconnect()
            except Exception as exc:
                print(f"Warning: failed to disconnect cleanly: {exc}")


def _run_checkpoint8_home_only(robot_ip: str) -> None:
    arm: Any | None = None
    try:
        print("Running home-only diagnostic.")
        print("This tests the original checkpoint8 move_gohome precondition.")
        arm = _checkpoint8_connect_arm(robot_ip)
        _checkpoint8_initialize_and_home(arm, GRIPPER_LENGTH_MM)
        print("home_only diagnostic complete: move_gohome succeeded.")
    except Exception:
        if arm is not None:
            _checkpoint8_print_status(arm, "after home_only failure")
        print("move_gohome failed; checkpoint8-style blackbox cannot be safely used from current robot state.")
        raise
    finally:
        if arm is not None:
            try:
                arm.disconnect()
            except Exception as exc:
                print(f"Warning: failed to disconnect cleanly: {exc}")


def _checkpoint8_print_transform(name: str, transform: np.ndarray) -> None:
    print(f"{name} translation (m): {np.array2string(transform[:3, 3], precision=4)}")
    print(f"{name} rotation matrix:\n{np.array2string(transform[:3, :3], precision=4)}")
    _print_pose(name, transform)


def _checkpoint8_target_pose_from_tag(T_base_cube: np.ndarray, T_base_tag: np.ndarray) -> np.ndarray:
    T_base_target = np.eye(4, dtype=np.float64)
    T_base_target[:3, :3] = T_base_cube[:3, :3]
    T_base_target[0, 3] = float(T_base_tag[0, 3])
    T_base_target[1, 3] = float(T_base_tag[1, 3])
    T_base_target[2, 3] = float(T_base_cube[2, 3])
    print(
        "Constructed checkpoint8-style tag placement pose: "
        "using target tag x/y, source cube contact z, and source cube rotation."
    )
    return T_base_target


def _preset_place_pose_from_slot(
    T_base_cube: np.ndarray,
    slot: PresetSlot,
    preset_use_slot_yaw: bool,
) -> np.ndarray:
    T_base_target = np.eye(4, dtype=np.float64)
    if preset_use_slot_yaw:
        T_base_target[:3, :3] = _top_down_rotation(math.radians(float(slot.yaw_deg)))
    else:
        T_base_target[:3, :3] = T_base_cube[:3, :3]
    T_base_target[0, 3] = float(slot.x)
    T_base_target[1, 3] = float(slot.y)
    T_base_target[2, 3] = float(T_base_cube[2, 3])
    print(
        "Constructed preset-slot placement pose: "
        "using preset slot x/y, source cube contact z, and "
        + ("slot yaw." if preset_use_slot_yaw else "source cube rotation.")
    )
    return T_base_target


def _checkpoint8_save_or_confirm(
    display: np.ndarray,
    no_gui: bool,
    preview_path: str,
    dry_run: bool,
    auto_confirm: bool = False,
) -> bool:
    saved_path = _save_preview_image(display, preview_path)
    if saved_path is not None:
        print(f"Saved checkpoint8-style preview: {saved_path}")

    if dry_run:
        print("Dry run selected. Perception and pose generation completed; no robot motion executed.")
        return False

    if auto_confirm:
        print("Auto-confirm enabled; skipping interactive confirmation.")
        return True

    use_gui = (not no_gui) and _gui_display_available()
    if not use_gui:
        if no_gui:
            print("GUI disabled by --no_gui; using terminal confirmation.")
        else:
            print("No GUI display detected; using terminal confirmation.")
        try:
            response = input("Type 'k' then Enter to execute, or anything else to cancel: ")
        except EOFError:
            print("No terminal input is available. Cancelling without robot motion.")
            return False
        return response.strip().lower() == "k"

    cv2.namedWindow("checkpoint8_style confirmation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("checkpoint8_style confirmation", 1280, 720)
    cv2.imshow("checkpoint8_style confirmation", display)
    print("Press 'k' in the OpenCV window to execute. Press 'q' or Esc to cancel.")
    key = cv2.waitKey(0) & 0xFF
    try:
        cv2.destroyWindow("checkpoint8_style confirmation")
    except cv2.error as exc:
        print(f"Warning: failed to destroy confirmation window cleanly: {exc}")
    return key == ord("k")


def _checkpoint8_final_home_if_ready(arm: Any) -> None:
    state, err, warn = _checkpoint8_print_status(arm, "before checkpoint8 final move_gohome")
    if not _checkpoint8_status_ready(state, err, warn):
        print(
            "Skipping checkpoint8 final move_gohome because the robot is not ready. "
            "Run --recover_robot, then --home_only before retrying."
        )
        return
    print("Returning to xArm home after checkpoint8-style task.")
    home_code = arm.move_gohome(wait=True)
    print(f"final move_gohome return code: {home_code!r}")
    if _arm_return_code(home_code) != 0:
        _checkpoint8_print_status(arm, "after failed checkpoint8 final move_gohome")
        print(
            "Final move_gohome failed. Run --recover_robot, then --home_only before retrying."
        )
        return
    time.sleep(0.5)
    _checkpoint8_require_ready(arm, "after checkpoint8 final move_gohome")


def _checkpoint8_require_home_ready(arm: Any, label: str) -> None:
    state, err, warn = _checkpoint8_print_status(arm, label)
    if state == 2 and err == 0 and warn == 0:
        return
    if state == 4 or err == 22:
        print(
            "Robot is in STOP/C22 after checkpoint8-style motion. "
            "Run --recover_robot, then --home_only before retrying."
        )
    raise RuntimeError(
        f"Lite6 is not at required checkpoint8 home-ready state {label}: "
        f"state={state}, err={err}, warn={warn}; expected state=2, err=0, warn=0."
    )


def _checkpoint8_move_home_required(arm: Any, label: str) -> None:
    _checkpoint8_require_ready(arm, f"before {label}")
    print(f"Moving to xArm home {label}.")
    home_code = arm.move_gohome(wait=True)
    print(f"move_gohome return code {label}: {home_code!r}")
    if _arm_return_code(home_code) != 0:
        state, err, _warn = _checkpoint8_print_status(arm, f"after failed {label}")
        if state == 4 or err == 22:
            print(
                "Robot is in STOP/C22 after checkpoint8-style motion. "
                "Run --recover_robot, then --home_only before retrying."
            )
        raise RuntimeError(f"checkpoint8_style move_gohome failed {label}")
    time.sleep(0.5)
    _checkpoint8_require_home_ready(arm, f"after {label}")


def _checkpoint8_capture_frame(zed: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    image = zed.image
    point_cloud = zed.point_cloud
    if image is None or point_cloud is None:
        return None, None
    return image, point_cloud


def _checkpoint8_pose_xyz_mm(pose: np.ndarray) -> tuple[float, float, float]:
    point = np.asarray(pose[:3, 3], dtype=np.float64) * 1000.0
    return float(point[0]), float(point[1]), float(point[2])


def _checkpoint8_format_xyz_mm(pose: np.ndarray) -> str:
    x, y, z = _checkpoint8_pose_xyz_mm(pose)
    return f"{x:.1f}/{y:.1f}/{z:.1f}"


def _checkpoint8_print_multi_plan_table(entries: list[Checkpoint8MultiPlanEntry]) -> None:
    print("\nCheckpoint8-style multi-place plan:")
    print("index | cube prompt | source x/y/z mm | target tag | tag x/y/z mm | place x/y/z mm")
    print("----- | ----------- | --------------- | ---------- | ------------ | --------------")
    for entry in entries:
        print(
            f"{entry.index:5d} | "
            f"{entry.cube_prompt} | "
            f"{_checkpoint8_format_xyz_mm(entry.T_robot_cube)} | "
            f"{entry.target_tag_id:10d} | "
            f"{_checkpoint8_format_xyz_mm(entry.T_robot_tag)} | "
            f"{_checkpoint8_format_xyz_mm(entry.T_robot_place)}"
        )


def _center_xyz_mm(center_m: np.ndarray) -> str:
    point = np.asarray(center_m, dtype=np.float64) * 1000.0
    return f"{point[0]:.1f}/{point[1]:.1f}/{point[2]:.1f}"


def select_nearest_refined_candidate(
    candidates: list[Any],
    planned_xy_m: np.ndarray,
    max_distance_m: float,
    label: str,
) -> tuple[Any, float]:
    if float(max_distance_m) <= 0.0:
        raise ValueError("max_distance_m must be positive.")
    if not candidates:
        raise ValueError(f"No refined {label} candidates were detected.")

    planned_xy = np.asarray(planned_xy_m, dtype=np.float64).reshape(-1)
    if planned_xy.shape[0] < 2 or not np.isfinite(planned_xy[:2]).all():
        raise ValueError(f"planned {label} x/y must contain finite values.")

    ranked: list[tuple[float, int, Any]] = []
    for index, candidate in enumerate(candidates):
        center = np.asarray(candidate.center_robot, dtype=np.float64).reshape(-1)
        if center.shape[0] < 2 or not np.isfinite(center[:2]).all():
            continue
        distance_m = float(np.linalg.norm(center[:2] - planned_xy[:2]))
        ranked.append((distance_m, int(getattr(candidate, "instance_index", index + 1)), candidate))

    if not ranked:
        raise ValueError(f"No refined {label} candidates had finite x/y centers.")

    ranked.sort(key=lambda item: (item[0], item[1]))
    distance_m, _instance_index, candidate = ranked[0]
    if distance_m > float(max_distance_m):
        raise ValueError(
            f"Nearest refined {label} candidate is {distance_m * 1000.0:.1f} mm from the planned pose, "
            f"exceeding the allowed radius {float(max_distance_m) * 1000.0:.1f} mm."
        )
    return candidate, distance_m


def _candidate_filter_reasons(
    candidate: DuplicateCubeCandidate,
    x_min_m: float,
    x_max_m: float,
    y_min_m: float,
    y_max_m: float,
    z_min_m: float,
    z_max_m: float,
    min_area_px: int,
    red_min_area_px: int,
    green_min_area_px: int,
    blue_min_area_px: int,
    min_extent_m: float,
    max_extent_m: float,
) -> list[str]:
    center = np.asarray(candidate.center_robot, dtype=np.float64)
    bounds = [
        ("x", float(center[0]), float(x_min_m), float(x_max_m)),
        ("y", float(center[1]), float(y_min_m), float(y_max_m)),
        ("z", float(center[2]), float(z_min_m), float(z_max_m)),
    ]
    reasons: list[str] = []
    for axis_name, value_m, minimum_m, maximum_m in bounds:
        if value_m < minimum_m or value_m > maximum_m:
            reasons.append(
                f"{axis_name}={round(value_m * 1000.0):.0f}mm outside "
                f"[{round(minimum_m * 1000.0):.0f},{round(maximum_m * 1000.0):.0f}]mm"
            )
    color_area_minima = {
        "red": int(red_min_area_px),
        "green": int(green_min_area_px),
        "blue": int(blue_min_area_px),
    }
    effective_min_area_px = max(int(min_area_px), color_area_minima.get(candidate.cube_color, int(min_area_px)))
    if int(candidate.area_px) < effective_min_area_px:
        reasons.append(f"area {int(candidate.area_px)} < {effective_min_area_px}")

    bbox_diag_m = float(getattr(candidate, "bbox_diag_m", math.nan))
    max_candidate_extent_m = float(getattr(candidate, "max_extent_m", math.nan))
    if math.isfinite(max_candidate_extent_m):
        if max_candidate_extent_m < float(min_extent_m):
            reasons.append(f"max_extent={max_candidate_extent_m * 1000.0:.1f}mm < {float(min_extent_m) * 1000.0:.1f}mm")
        if max_candidate_extent_m > float(max_extent_m):
            reasons.append(f"max_extent={max_candidate_extent_m * 1000.0:.1f}mm > {float(max_extent_m) * 1000.0:.1f}mm")
    elif math.isfinite(bbox_diag_m):
        if bbox_diag_m < float(min_extent_m):
            reasons.append(f"bbox_diag={bbox_diag_m * 1000.0:.1f}mm < {float(min_extent_m) * 1000.0:.1f}mm")
        max_bbox_diag_m = float(max_extent_m) * math.sqrt(3.0)
        if bbox_diag_m > max_bbox_diag_m:
            reasons.append(f"bbox_diag={bbox_diag_m * 1000.0:.1f}mm > {max_bbox_diag_m * 1000.0:.1f}mm")
    return reasons


def filter_duplicate_cube_candidates(
    candidates: list[DuplicateCubeCandidate],
    x_min_m: float,
    x_max_m: float,
    y_min_m: float,
    y_max_m: float,
    z_min_m: float,
    z_max_m: float,
    min_area_px: int,
    red_min_area_px: int = 500,
    green_min_area_px: int = 1200,
    blue_min_area_px: int = 500,
    min_extent_m: float = 0.012,
    max_extent_m: float = 0.040,
) -> tuple[list[DuplicateCubeCandidate], list[RejectedDuplicateCubeCandidate]]:
    valid_candidates: list[DuplicateCubeCandidate] = []
    rejected_candidates: list[RejectedDuplicateCubeCandidate] = []
    for candidate in candidates:
        reasons = _candidate_filter_reasons(
            candidate=candidate,
            x_min_m=x_min_m,
            x_max_m=x_max_m,
            y_min_m=y_min_m,
            y_max_m=y_max_m,
            z_min_m=z_min_m,
            z_max_m=z_max_m,
            min_area_px=min_area_px,
            red_min_area_px=red_min_area_px,
            green_min_area_px=green_min_area_px,
            blue_min_area_px=blue_min_area_px,
            min_extent_m=min_extent_m,
            max_extent_m=max_extent_m,
        )
        if reasons:
            rejected_candidates.append(
                RejectedDuplicateCubeCandidate(candidate=candidate, rejection_reasons=reasons)
            )
        else:
            valid_candidates.append(candidate)
    return valid_candidates, rejected_candidates


def _print_rejected_duplicate_cube_candidates(rejected: list[RejectedDuplicateCubeCandidate]) -> None:
    for rejected_candidate in rejected:
        candidate = rejected_candidate.candidate
        for reason in rejected_candidate.rejection_reasons:
            print(f"rejected {candidate.cube_color} cube candidate #{candidate.instance_index}: {reason}")


def _xy_distance_m(left: DuplicateCubeCandidate, right: DuplicateCubeCandidate) -> float:
    return float(np.linalg.norm(left.center_robot[:2] - right.center_robot[:2]))


def _best_merge_yaw_source(candidates: list[DuplicateCubeCandidate]) -> DuplicateCubeCandidate:
    return min(candidates, key=lambda candidate: (-int(candidate.area_px), float(candidate.score), int(candidate.instance_index)))


def _build_physical_cube_candidate(
    cluster_index: int,
    component: list[DuplicateCubeCandidate],
    T_cam_robot: np.ndarray | None,
) -> DuplicateCubeCandidate:
    if not component:
        raise ValueError("physical cube cluster cannot be empty.")

    yaw_source = _best_merge_yaw_source(component)
    weights = np.array([max(1, int(candidate.area_px)) for candidate in component], dtype=np.float64)
    centers = np.vstack([candidate.center_robot for candidate in component])
    center_robot = np.average(centers, axis=0, weights=weights)
    yaw_rad = float(yaw_source.yaw_rad)

    T_robot_cube = np.eye(4, dtype=np.float64)
    T_robot_cube[:3, :3] = _top_down_rotation(yaw_rad)
    T_robot_cube[:3, 3] = center_robot
    if T_cam_robot is not None:
        T_cam_cube = T_cam_robot @ T_robot_cube
    else:
        T_cam_cube = yaw_source.T_cam_cube.copy()

    total_area = int(sum(int(candidate.area_px) for candidate in component))
    weighted_score = float(np.average([candidate.score for candidate in component], weights=weights))
    member_indices = tuple(int(candidate.instance_index) for candidate in component)
    return DuplicateCubeCandidate(
        cube_prompt=yaw_source.cube_prompt,
        cube_color=yaw_source.cube_color,
        instance_index=int(cluster_index),
        component_label=int(yaw_source.component_label),
        area_px=total_area,
        score=weighted_score,
        bbox_diag_m=float(max(candidate.bbox_diag_m for candidate in component)),
        max_extent_m=float(max(candidate.max_extent_m for candidate in component)),
        yaw_rad=yaw_rad,
        T_robot_cube=T_robot_cube,
        T_cam_cube=T_cam_cube,
        center_robot=center_robot,
        member_candidate_indices=member_indices,
    )


def merge_duplicate_cube_candidates(
    candidates: list[DuplicateCubeCandidate],
    merge_distance_m: float,
    T_cam_robot: np.ndarray | None = None,
) -> tuple[list[DuplicateCubeCandidate], list[MergedDuplicateCubeCandidates]]:
    if float(merge_distance_m) <= 0.0:
        raise ValueError("merge_distance_m must be positive.")

    remaining = set(range(len(candidates)))
    physical_candidates: list[DuplicateCubeCandidate] = []
    cluster_records: list[MergedDuplicateCubeCandidates] = []
    while remaining:
        seed_index = min(remaining)
        remaining.remove(seed_index)
        stack = [seed_index]
        component_indices: set[int] = set()

        while stack:
            current_index = stack.pop()
            component_indices.add(current_index)
            current = candidates[current_index]
            neighbors = [
                other_index
                for other_index in list(remaining)
                if current.cube_color == candidates[other_index].cube_color
                and _xy_distance_m(current, candidates[other_index]) < float(merge_distance_m)
            ]
            for other_index in neighbors:
                remaining.remove(other_index)
                stack.append(other_index)

        component = [candidates[index] for index in sorted(component_indices)]
        physical = _build_physical_cube_candidate(
            cluster_index=len(physical_candidates) + 1,
            component=component,
            T_cam_robot=T_cam_robot,
        )
        physical_candidates.append(physical)
        cluster_records.append(
            MergedDuplicateCubeCandidates(
                physical_candidate=physical,
                merged_candidates=component,
                merge_distance_m=float(merge_distance_m),
            )
        )

    physical_candidates.sort(key=lambda candidate: int(candidate.instance_index))
    return physical_candidates, cluster_records


def _print_merged_duplicate_cube_candidates(cluster_records: list[MergedDuplicateCubeCandidates]) -> None:
    for record in cluster_records:
        physical = record.physical_candidate
        color = physical.cube_color
        candidate_ids = "/".join(f"#{candidate.instance_index}" for candidate in record.merged_candidates)
        print(
            f"cluster #{physical.instance_index}: members {candidate_ids},"
            f" merged {color} center={_center_xyz_mm(physical.center_robot)} mm,"
            f" total area={physical.area_px}"
        )
        print(
            f"merged physical candidate #{physical.instance_index}:"
            f" yaw_source_member=#{_best_merge_yaw_source(record.merged_candidates).instance_index}"
            f" representative_yaw_rad={physical.yaw_rad:.4f}"
            f" weighted_score={physical.score:.4f}"
        )


def _assignment_without_threshold(
    cube_centers_m: np.ndarray,
    tag_centers_m: np.ndarray,
) -> tuple[list[int], np.ndarray, list[float], float]:
    cube_centers = np.asarray(cube_centers_m, dtype=np.float64)
    tag_centers = np.asarray(tag_centers_m, dtype=np.float64)
    if cube_centers.ndim != 2 or tag_centers.ndim != 2 or cube_centers.shape[1] < 2 or tag_centers.shape[1] < 2:
        raise ValueError("cube_centers_m and tag_centers_m must be Nx2 or Nx3 arrays.")
    if cube_centers.shape[0] != tag_centers.shape[0]:
        raise ValueError(
            f"assignment requires equal counts, got {cube_centers.shape[0]} cubes and {tag_centers.shape[0]} tags."
        )
    if cube_centers.shape[0] == 0:
        raise ValueError("assignment requires at least one cube and one tag.")

    diff = cube_centers[:, None, :2] - tag_centers[None, :, :2]
    distance_matrix = np.linalg.norm(diff, axis=2)

    best_perm: tuple[int, ...] | None = None
    best_total = math.inf
    for perm in itertools.permutations(range(tag_centers.shape[0])):
        total = float(sum(distance_matrix[cube_index, tag_index] for cube_index, tag_index in enumerate(perm)))
        if total < best_total:
            best_total = total
            best_perm = tuple(int(item) for item in perm)

    if best_perm is None:
        raise ValueError("assignment failed to produce a permutation.")
    pair_distances = [float(distance_matrix[cube_index, tag_index]) for cube_index, tag_index in enumerate(best_perm)]
    return list(best_perm), distance_matrix, pair_distances, best_total


def solve_nearest_xy_assignment(
    cube_centers_m: np.ndarray,
    tag_centers_m: np.ndarray,
    max_distance_m: float,
) -> tuple[list[int], np.ndarray, list[float]]:
    if float(max_distance_m) <= 0.0:
        raise ValueError("max_distance_m must be positive.")

    best_perm, distance_matrix, pair_distances, _best_total = _assignment_without_threshold(
        cube_centers_m=cube_centers_m,
        tag_centers_m=tag_centers_m,
    )
    too_far = [distance for distance in pair_distances if distance > float(max_distance_m)]
    if too_far:
        distances_mm = ", ".join(f"{distance * 1000.0:.1f}" for distance in pair_distances)
        print("Assignment distance matrix exceeded the threshold (mm):")
        print(np.array2string(distance_matrix * 1000.0, precision=1, suppress_small=False))
        print(f"Best assignment permutation: {best_perm}")
        raise ValueError(
            f"assignment pair distance exceeds --max_assignment_distance_m={float(max_distance_m):.3f} m; "
            f"pair distances mm=[{distances_mm}]."
        )

    return best_perm, distance_matrix, pair_distances


def _selection_tie_break_score(cubes: list[DuplicateCubeCandidate]) -> float:
    return float(sum(candidate.score for candidate in cubes))


def _selection_objective(total_distance_m: float, cubes: list[DuplicateCubeCandidate]) -> float:
    return float(total_distance_m) + 0.001 * _selection_tie_break_score(cubes)


def _selection_better(
    candidate: DuplicateAssignmentSelection,
    incumbent: DuplicateAssignmentSelection | None,
) -> bool:
    if incumbent is None:
        return True
    if candidate.objective < incumbent.objective - 1e-12:
        return True
    if math.isclose(candidate.objective, incumbent.objective, rel_tol=0.0, abs_tol=1e-12):
        candidate_area = sum(cube.area_px for cube in candidate.selected_cubes)
        incumbent_area = sum(cube.area_px for cube in incumbent.selected_cubes)
        if candidate_area > incumbent_area:
            return True
    return False


def select_duplicate_assignment_subset(
    cube_candidates: list[DuplicateCubeCandidate],
    tag_candidates: list[DuplicateTagCandidate],
    count: int,
    max_distance_m: float,
) -> DuplicateAssignmentSelection:
    if int(count) <= 0:
        raise ValueError("count must be positive.")
    if len(cube_candidates) < int(count):
        raise ValueError(f"need {count} valid cube candidates, found {len(cube_candidates)}.")
    if len(tag_candidates) < int(count):
        raise ValueError(f"need {count} tag candidates, found {len(tag_candidates)}.")
    if float(max_distance_m) <= 0.0:
        raise ValueError("max_distance_m must be positive.")

    best_valid: DuplicateAssignmentSelection | None = None
    best_failed: DuplicateAssignmentSelection | None = None

    for cube_subset in itertools.combinations(cube_candidates, int(count)):
        tag_subsets = itertools.combinations(tag_candidates, int(count))
        for tag_subset in tag_subsets:
            cube_centers = np.vstack([candidate.center_robot for candidate in cube_subset])
            tag_centers = np.vstack([candidate.center_robot for candidate in tag_subset])
            tag_permutation, distance_matrix, pair_distances, total_distance_m = _assignment_without_threshold(
                cube_centers_m=cube_centers,
                tag_centers_m=tag_centers,
            )
            selection = DuplicateAssignmentSelection(
                selected_cubes=list(cube_subset),
                selected_tags=list(tag_subset),
                tag_permutation=tag_permutation,
                distance_matrix=distance_matrix,
                pair_distances=pair_distances,
                total_distance_m=float(total_distance_m),
                objective=_selection_objective(total_distance_m, list(cube_subset)),
            )
            if any(distance > float(max_distance_m) for distance in pair_distances):
                if _selection_better(selection, best_failed):
                    best_failed = selection
                continue
            if _selection_better(selection, best_valid):
                best_valid = selection

    if best_valid is not None:
        return best_valid

    if best_failed is not None:
        cube_ids = ", ".join(f"#{candidate.instance_index}" for candidate in best_failed.selected_cubes)
        tag_ids = ", ".join(f"#{candidate.instance_index}" for candidate in best_failed.selected_tags)
        distances_mm = ", ".join(f"{distance * 1000.0:.1f}" for distance in best_failed.pair_distances)
        print("No duplicate-aware candidate set satisfies the max assignment distance.")
        print(f"Best failed cube candidates: {cube_ids}")
        print(f"Best failed tag candidates: {tag_ids}")
        print(f"Best failed pair distances mm: [{distances_mm}]")
        print("Best failed final distance matrix (mm):")
        print(np.array2string(best_failed.distance_matrix * 1000.0, precision=1, suppress_small=False))
    raise ValueError(
        f"no valid duplicate-aware assignment satisfies --max_assignment_distance_m={float(max_distance_m):.3f} m."
    )


def select_preset_assignment_subset(
    cube_candidates: list[DuplicateCubeCandidate],
    slots: list[PresetSlot],
    count: int,
) -> PresetAssignmentSelection:
    if int(count) <= 0:
        raise ValueError("count must be positive.")
    if len(cube_candidates) < int(count):
        raise ValueError(f"need {count} valid cube candidates, found {len(cube_candidates)}.")
    if len(slots) != int(count):
        raise ValueError(f"preset assignment requires exactly {count} slots, found {len(slots)}.")

    best_selection: PresetAssignmentSelection | None = None
    selected_slots = list(slots)
    slot_centers = np.vstack([slot.center_robot for slot in selected_slots])
    for cube_subset in itertools.combinations(cube_candidates, int(count)):
        cube_centers = np.vstack([candidate.center_robot for candidate in cube_subset])
        slot_permutation, distance_matrix, pair_distances, total_distance_m = _assignment_without_threshold(
            cube_centers_m=cube_centers,
            tag_centers_m=slot_centers,
        )
        selection = PresetAssignmentSelection(
            selected_cubes=list(cube_subset),
            selected_slots=selected_slots,
            slot_permutation=slot_permutation,
            distance_matrix=distance_matrix,
            pair_distances=pair_distances,
            total_distance_m=float(total_distance_m),
            objective=_selection_objective(total_distance_m, list(cube_subset)),
        )
        if _selection_better(
            DuplicateAssignmentSelection(
                selected_cubes=selection.selected_cubes,
                selected_tags=[],
                tag_permutation=selection.slot_permutation,
                distance_matrix=selection.distance_matrix,
                pair_distances=selection.pair_distances,
                total_distance_m=selection.total_distance_m,
                objective=selection.objective,
            ),
            DuplicateAssignmentSelection(
                selected_cubes=best_selection.selected_cubes,
                selected_tags=[],
                tag_permutation=best_selection.slot_permutation,
                distance_matrix=best_selection.distance_matrix,
                pair_distances=best_selection.pair_distances,
                total_distance_m=best_selection.total_distance_m,
                objective=best_selection.objective,
            )
            if best_selection is not None
            else None,
        ):
            best_selection = selection

    if best_selection is None:
        raise ValueError("preset assignment failed to produce a selection.")
    return best_selection


def _print_duplicate_cube_candidates(group: DuplicateCubeTagGroup, candidates: list[DuplicateCubeCandidate]) -> None:
    print(f"Detected cube candidates for {group.cube_prompt!r} (need {group.count}): {len(candidates)}")
    for candidate in candidates:
        print(
            f"  cube #{candidate.instance_index}:"
            f" label={candidate.component_label}"
            f" color={candidate.cube_color}"
            f" center={_center_xyz_mm(candidate.center_robot)} mm"
            f" area_px={candidate.area_px}"
            f" score={candidate.score:.4f}"
            f" bbox_diag_m={candidate.bbox_diag_m:.4f}"
            f" max_extent_m={candidate.max_extent_m:.4f}"
            f" yaw={math.degrees(candidate.yaw_rad):.1f} deg"
        )


def _print_duplicate_tag_candidates(group: DuplicateCubeTagGroup, candidates: list[DuplicateTagCandidate]) -> None:
    print(f"Detected tag candidates for tag {group.tag_id} (need {group.count}): {len(candidates)}")
    for candidate in candidates:
        print(
            f"  tag #{candidate.instance_index}:"
            f" detection_index={candidate.detection_index}"
            f" center={_center_xyz_mm(candidate.center_robot)} mm"
            f" decision_margin={candidate.decision_margin:.2f}"
            f" hamming={candidate.hamming}"
        )


def _print_duplicate_selected_candidates(
    group: DuplicateCubeTagGroup,
    cubes: list[DuplicateCubeCandidate],
    tags: list[DuplicateTagCandidate],
) -> None:
    cube_ids = ", ".join(f"#{candidate.instance_index}" for candidate in cubes)
    tag_ids = ", ".join(f"#{candidate.instance_index}" for candidate in tags)
    print(f"Selected cube candidates for {group.cube_prompt!r}: {cube_ids}")
    print(f"Selected tag candidates for tag {group.tag_id}: {tag_ids}")


def _print_duplicate_distance_matrix(
    group: DuplicateCubeTagGroup,
    cubes: list[DuplicateCubeCandidate],
    tags: list[DuplicateTagCandidate],
    distance_matrix: np.ndarray,
) -> None:
    print(f"Distance matrix for group {group.cube_prompt!r} -> tag {group.tag_id} (mm):")
    header = "cube/tag | " + " | ".join(f"tag #{tag.instance_index}" for tag in tags)
    print(header)
    print("-" * len(header))
    for cube_index, cube in enumerate(cubes):
        row = " | ".join(f"{distance_matrix[cube_index, tag_index] * 1000.0:.1f}" for tag_index in range(len(tags)))
        print(f"cube #{cube.instance_index} | {row}")


def _print_preset_distance_matrix(
    group: PresetCubeGroup,
    cubes: list[DuplicateCubeCandidate],
    slots: list[PresetSlot],
    distance_matrix: np.ndarray,
) -> None:
    print(f"Distance matrix for preset group {group.cube_prompt!r} (mm):")
    header = "cube/slot | " + " | ".join(f"slot {slot.slot_id}" for slot in slots)
    print(header)
    print("-" * len(header))
    for cube_index, cube in enumerate(cubes):
        row = " | ".join(f"{distance_matrix[cube_index, slot_index] * 1000.0:.1f}" for slot_index in range(len(slots)))
        print(f"cube #{cube.instance_index} | {row}")


def _print_duplicate_assignment_table(assignments: list[DuplicateAssignedPair]) -> None:
    print("\nDuplicate-aware assignment table:")
    print("group | cube instance | cube x/y/z mm | tag instance | tag id | tag x/y/z mm | distance mm")
    print("----- | ------------- | ------------- | ------------ | ------ | ------------ | -----------")
    for assignment in assignments:
        group_label = f"{assignment.cube_prompt} -> tag {assignment.tag_id}"
        print(
            f"{group_label} | "
            f"#{assignment.cube.instance_index} | "
            f"{_center_xyz_mm(assignment.cube.center_robot)} | "
            f"#{assignment.tag.instance_index} | "
            f"{assignment.tag_id} | "
            f"{_center_xyz_mm(assignment.tag.center_robot)} | "
            f"{assignment.distance_m * 1000.0:.1f}"
        )


def _print_duplicate_execution_order(assignments: list[DuplicateAssignedPair]) -> None:
    print("\nFinal duplicate-aware execution order:")
    for assignment in assignments:
        print(
            f"  Pair {assignment.execution_index}/{len(assignments)}:"
            f" {assignment.cube_prompt} cube #{assignment.cube.instance_index}"
            f" -> tag {assignment.tag_id} #{assignment.tag.instance_index}"
            f" distance={assignment.distance_m * 1000.0:.1f} mm"
        )


def _print_preset_assignment_table(assignments: list[PresetAssignedPair]) -> None:
    print("\nPreset layout assignment table:")
    print("group | cube instance | cube x/y/z mm | slot id | slot x/y/z mm | distance mm")
    print("----- | ------------- | ------------- | ------- | ------------ | -----------")
    for assignment in assignments:
        print(
            f"{assignment.cube_prompt} | "
            f"#{assignment.cube.instance_index} | "
            f"{_center_xyz_mm(assignment.cube.center_robot)} | "
            f"{assignment.slot.slot_id} | "
            f"{_center_xyz_mm(assignment.slot.center_robot)} | "
            f"{assignment.distance_m * 1000.0:.1f}"
        )


def _print_preset_execution_order(assignments: list[PresetAssignedPair]) -> None:
    print("\nFinal preset layout execution order:")
    for assignment in assignments:
        print(
            f"  Pair {assignment.execution_index}/{len(assignments)}:"
            f" {assignment.cube_prompt} cube #{assignment.cube.instance_index}"
            f" -> slot {assignment.slot.slot_id}"
            f" distance={assignment.distance_m * 1000.0:.1f} mm"
        )


def _matrix_to_json(matrix: np.ndarray) -> list[list[float]]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"expected a 4x4 transform matrix, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("transform matrix contains non-finite values.")
    return [[float(value) for value in row] for row in array]


def _matrix_from_json(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix, got shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def duplicate_pose_plan_to_json_data(plan: DuplicatePosePlan) -> dict[str, Any]:
    data: dict[str, Any] = {
        "execution_index": int(plan.execution_index),
        "cube_prompt": str(plan.cube_prompt),
        "target_tag_id": int(plan.target_tag_id),
        "cube_instance_index": int(plan.cube_instance_index),
        "tag_instance_index": int(plan.tag_instance_index),
        "t_robot_cube": _matrix_to_json(plan.T_robot_cube),
        "t_robot_place": _matrix_to_json(plan.T_robot_place),
    }
    if plan.target_source != "apriltag" or plan.slot_id is not None or plan.preset_slot is not None:
        data["target_source"] = str(plan.target_source)
    if plan.slot_id is not None:
        data["slot_id"] = int(plan.slot_id)
    if plan.preset_slot is not None:
        data["preset_slot"] = dict(plan.preset_slot)
    if plan.target_source == "preset_slot":
        data["preset_use_slot_yaw"] = bool(plan.preset_use_slot_yaw)
    return data


def duplicate_pose_plan_from_json_data(data: dict[str, Any]) -> DuplicatePosePlan:
    required = [
        "execution_index",
        "cube_prompt",
        "target_tag_id",
        "cube_instance_index",
        "tag_instance_index",
        "t_robot_cube",
        "t_robot_place",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("pose plan JSON is missing required keys: " + ", ".join(missing))
    cube_prompt = str(data["cube_prompt"]).strip()
    if not cube_prompt:
        raise ValueError("pose plan JSON cube_prompt must be nonempty.")
    target_source = str(data.get("target_source", "apriltag")).strip() or "apriltag"
    if target_source not in {"apriltag", "preset_slot"}:
        raise ValueError(f"pose plan JSON target_source is unsupported: {target_source!r}.")
    slot_id = data.get("slot_id")
    preset_slot = data.get("preset_slot")
    preset_use_slot_yaw = bool(data.get("preset_use_slot_yaw", False))
    if target_source == "preset_slot":
        if slot_id is None:
            raise ValueError("preset_slot pose plan JSON requires slot_id.")
        if not isinstance(preset_slot, dict):
            raise ValueError("preset_slot pose plan JSON requires preset_slot object.")
        _preset_slot_from_json_data(preset_slot, 1)
    return DuplicatePosePlan(
        execution_index=int(data["execution_index"]),
        cube_prompt=cube_prompt,
        target_tag_id=int(data["target_tag_id"]),
        cube_instance_index=int(data["cube_instance_index"]),
        tag_instance_index=int(data["tag_instance_index"]),
        T_robot_cube=_matrix_from_json(data["t_robot_cube"], "t_robot_cube"),
        T_robot_place=_matrix_from_json(data["t_robot_place"], "t_robot_place"),
        target_source=target_source,
        slot_id=int(slot_id) if slot_id is not None else None,
        preset_slot=dict(preset_slot) if isinstance(preset_slot, dict) else None,
        preset_use_slot_yaw=preset_use_slot_yaw,
    )


def _write_duplicate_pose_plan_json(plan: DuplicatePosePlan, path: Path) -> Path:
    output_path = path.expanduser()
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(duplicate_pose_plan_to_json_data(plan), handle, indent=2)
        handle.write("\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError as exc:
            print(f"Warning: failed to fsync pose-plan JSON {output_path}: {exc}")
    return output_path


def _load_duplicate_pose_plan_json(path: str | Path) -> DuplicatePosePlan:
    input_path = Path(path).expanduser()
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("pose plan JSON root must be an object.")
    return duplicate_pose_plan_from_json_data(data)


def _duplicate_pose_plan_from_assignment(assignment: DuplicateAssignedPair) -> DuplicatePosePlan:
    return DuplicatePosePlan(
        execution_index=assignment.execution_index,
        cube_prompt=assignment.cube_prompt,
        target_tag_id=assignment.tag_id,
        cube_instance_index=assignment.cube.instance_index,
        tag_instance_index=assignment.tag.instance_index,
        T_robot_cube=assignment.cube.T_robot_cube,
        T_robot_place=assignment.T_robot_place,
    )


def _duplicate_pose_plan_from_preset_assignment(assignment: PresetAssignedPair) -> DuplicatePosePlan:
    return DuplicatePosePlan(
        execution_index=assignment.execution_index,
        cube_prompt=assignment.cube_prompt,
        target_tag_id=int(assignment.slot.slot_id),
        cube_instance_index=assignment.cube.instance_index,
        tag_instance_index=int(assignment.slot.slot_id),
        T_robot_cube=assignment.cube.T_robot_cube,
        T_robot_place=assignment.T_robot_place,
        target_source="preset_slot",
        slot_id=int(assignment.slot.slot_id),
        preset_slot=assignment.slot.to_json_data(),
        preset_use_slot_yaw=bool(assignment.preset_use_slot_yaw),
    )


def _slugify_for_filename(value: str) -> str:
    cleaned = []
    for char in value.strip().lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in {" ", "-", "_"}:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "item"


def _write_duplicate_pose_plans(assignments: list[DuplicateAssignedPair]) -> dict[int, Path]:
    plan_paths: dict[int, Path] = {}
    base_dir = REPO_ROOT / DUPLICATE_POSE_PLAN_DIR
    for assignment in assignments:
        slug = _slugify_for_filename(assignment.cube_prompt)
        path = base_dir / f"pair_{assignment.execution_index:03d}_{slug}_to_tag_{assignment.tag_id}.json"
        plan_path = _write_duplicate_pose_plan_json(_duplicate_pose_plan_from_assignment(assignment), path)
        plan_paths[assignment.execution_index] = plan_path
        print(f"Pose-plan JSON for pair {assignment.execution_index}: {plan_path}")
    return plan_paths


def _write_preset_pose_plans(assignments: list[PresetAssignedPair]) -> dict[int, Path]:
    plan_paths: dict[int, Path] = {}
    base_dir = REPO_ROOT / PRESET_POSE_PLAN_DIR
    for assignment in assignments:
        slug = _slugify_for_filename(assignment.cube_prompt)
        path = base_dir / f"pair_{assignment.execution_index:03d}_{slug}_to_slot_{assignment.slot.slot_id}.json"
        plan_path = _write_duplicate_pose_plan_json(_duplicate_pose_plan_from_preset_assignment(assignment), path)
        plan_paths[assignment.execution_index] = plan_path
        print(f"Preset pose-plan JSON for pair {assignment.execution_index}: {plan_path}")
    return plan_paths


def _candidate_cube_report(candidate: DuplicateCubeCandidate) -> dict[str, Any]:
    member_indices = tuple(candidate.member_candidate_indices) or (int(candidate.instance_index),)
    return {
        "instance_index": int(candidate.instance_index),
        "component_label": int(candidate.component_label),
        "cube_color": candidate.cube_color,
        "center_robot_m": [float(value) for value in candidate.center_robot],
        "merged_center_robot_m": [float(value) for value in candidate.center_robot],
        "area_px": int(candidate.area_px),
        "merged_area_px": int(candidate.area_px),
        "score": float(candidate.score),
        "bbox_diag_m": float(candidate.bbox_diag_m),
        "max_extent_m": float(candidate.max_extent_m),
        "yaw_rad": float(candidate.yaw_rad),
        "representative_yaw_rad": float(candidate.yaw_rad),
        "member_candidate_indices": [int(index) for index in member_indices],
    }


def _rejected_candidate_cube_report(rejected: RejectedDuplicateCubeCandidate) -> dict[str, Any]:
    report = _candidate_cube_report(rejected.candidate)
    report["rejection_reason"] = list(rejected.rejection_reasons)
    report["rejection_reasons"] = list(rejected.rejection_reasons)
    return report


def _merged_candidate_cube_report(merged: MergedDuplicateCubeCandidates) -> dict[str, Any]:
    return {
        "physical_candidate": _candidate_cube_report(merged.physical_candidate),
        "merged_candidates": [_candidate_cube_report(candidate) for candidate in merged.merged_candidates],
        "merge_distance_m": float(merged.merge_distance_m),
        "merge_reason": "xy centers closer than candidate_merge_distance_m",
    }


def _candidate_tag_report(candidate: DuplicateTagCandidate) -> dict[str, Any]:
    return {
        "instance_index": int(candidate.instance_index),
        "detection_index": int(candidate.detection_index),
        "tag_id": int(candidate.tag_id),
        "center_robot_m": [float(value) for value in candidate.center_robot],
        "decision_margin": float(candidate.decision_margin),
        "hamming": int(candidate.hamming),
    }


def _assignment_report_row(assignment: DuplicateAssignedPair, plan_paths: dict[int, Path] | None) -> dict[str, Any]:
    plan_path = None
    if plan_paths is not None and assignment.execution_index in plan_paths:
        plan_path = str(plan_paths[assignment.execution_index])
    return {
        "execution_index": int(assignment.execution_index),
        "group_index": int(assignment.group_index),
        "within_group_index": int(assignment.within_group_index),
        "cube_prompt": assignment.cube_prompt,
        "tag_id": int(assignment.tag_id),
        "cube_instance_index": int(assignment.cube.instance_index),
        "tag_instance_index": int(assignment.tag.instance_index),
        "cube_center_robot_m": [float(value) for value in assignment.cube.center_robot],
        "tag_center_robot_m": [float(value) for value in assignment.tag.center_robot],
        "distance_m": float(assignment.distance_m),
        "pose_plan_path": plan_path,
    }


def _preset_assignment_report_row(assignment: PresetAssignedPair, plan_paths: dict[int, Path] | None) -> dict[str, Any]:
    plan_path = None
    if plan_paths is not None and assignment.execution_index in plan_paths:
        plan_path = str(plan_paths[assignment.execution_index])
    return {
        "execution_index": int(assignment.execution_index),
        "group_index": int(assignment.group_index),
        "within_group_index": int(assignment.within_group_index),
        "cube_prompt": assignment.cube_prompt,
        "cube_instance_index": int(assignment.cube.instance_index),
        "slot_id": int(assignment.slot.slot_id),
        "cube_center_robot_m": [float(value) for value in assignment.cube.center_robot],
        "slot_center_robot_m": [float(value) for value in assignment.slot.center_robot],
        "distance_m": float(assignment.distance_m),
        "pose_plan_path": plan_path,
        "preset_use_slot_yaw": bool(assignment.preset_use_slot_yaw),
    }


def _save_duplicate_assignment_report(
    groups: list[dict[str, Any]],
    assignments: list[DuplicateAssignedPair],
    preview_path: Path | None,
    plan_paths: dict[int, Path] | None = None,
    execution_confirmed: bool = False,
) -> Path:
    report_path = REPO_ROOT / DUPLICATE_ASSIGNMENT_REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_unix_s": time.time(),
        "assignment_metric": "nearest",
        "assignment_space": "xy",
        "preview_path": str(preview_path) if preview_path is not None else None,
        "execution_confirmed": bool(execution_confirmed),
        "groups": groups,
        "execution_order": [_assignment_report_row(assignment, plan_paths) for assignment in assignments],
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Saved duplicate-aware assignment report: {report_path}")
    return report_path


def _save_preset_assignment_report(
    layout: PresetLayout,
    groups: list[dict[str, Any]],
    assignments: list[PresetAssignedPair],
    preview_path: Path | None,
    plan_paths: dict[int, Path] | None = None,
    execution_confirmed: bool = False,
) -> Path:
    report_path = REPO_ROOT / PRESET_LAYOUT_ASSIGNMENT_REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_unix_s": time.time(),
        "assignment_metric": "nearest",
        "assignment_space": "xy",
        "layout": {
            "name": layout.name,
            "frame": layout.frame,
            "slots": [slot.to_json_data() for slot in layout.slots.values()],
        },
        "preview_path": str(preview_path) if preview_path is not None else None,
        "execution_confirmed": bool(execution_confirmed),
        "groups": groups,
        "execution_order": [_preset_assignment_report_row(assignment, plan_paths) for assignment in assignments],
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Saved preset layout assignment report: {report_path}")
    return report_path


def _draw_duplicate_assignment_preview(
    image: np.ndarray,
    camera_intrinsic: np.ndarray,
    assignments: list[DuplicateAssignedPair],
    cube_size_m: float,
    target_tag_size_m: float,
    draw_pose_axes_fn: Any,
) -> np.ndarray:
    display = _bgr_from_image(image)
    colors = [
        (0, 255, 255),
        (255, 180, 0),
        (0, 180, 255),
        (180, 255, 0),
        (255, 0, 255),
        (0, 255, 120),
    ]
    for assignment in assignments:
        color = colors[(assignment.execution_index - 1) % len(colors)]
        draw_pose_axes_fn(display, camera_intrinsic, assignment.cube.T_cam_cube, size=float(cube_size_m) * 1.5)
        draw_pose_axes_fn(display, camera_intrinsic, assignment.tag.T_cam_tag, size=float(target_tag_size_m) * 0.75)
        draw_pose_axes_fn(display, camera_intrinsic, assignment.T_cam_place, size=float(cube_size_m) * 1.5)
        cube_label = f"{assignment.execution_index}: {assignment.cube_prompt} #{assignment.cube.instance_index}"
        tag_label = f"{assignment.execution_index}: tag {assignment.tag_id} #{assignment.tag.instance_index}"
        _draw_label(display, camera_intrinsic, assignment.cube.T_cam_cube, cube_label, color)
        _draw_label(display, camera_intrinsic, assignment.tag.T_cam_tag, tag_label, (255, 255, 0))
        _draw_label(display, camera_intrinsic, assignment.T_cam_place, f"place {assignment.execution_index}", (255, 0, 255))

        cube_origin = _project_origin(camera_intrinsic, assignment.cube.T_cam_cube)
        tag_origin = _project_origin(camera_intrinsic, assignment.tag.T_cam_tag)
        if cube_origin is not None and tag_origin is not None:
            cv2.line(display, cube_origin, tag_origin, color, 2, cv2.LINE_AA)
    return display


def _draw_preset_assignment_preview(
    image: np.ndarray,
    camera_intrinsic: np.ndarray,
    assignments: list[PresetAssignedPair],
    cube_size_m: float,
    draw_pose_axes_fn: Any,
) -> np.ndarray:
    display = _bgr_from_image(image)
    colors = [
        (0, 255, 255),
        (255, 180, 0),
        (0, 180, 255),
        (180, 255, 0),
        (255, 0, 255),
        (0, 255, 120),
    ]
    for assignment in assignments:
        color = colors[(assignment.execution_index - 1) % len(colors)]
        draw_pose_axes_fn(display, camera_intrinsic, assignment.cube.T_cam_cube, size=float(cube_size_m) * 1.5)
        draw_pose_axes_fn(display, camera_intrinsic, assignment.T_cam_place, size=float(cube_size_m) * 1.5)
        _draw_label(display, camera_intrinsic, assignment.cube.T_cam_cube, f"{assignment.execution_index}: {assignment.cube_prompt}", color)
        _draw_label(display, camera_intrinsic, assignment.T_cam_place, f"slot {assignment.slot.slot_id}", (255, 0, 255))
        cube_origin = _project_origin(camera_intrinsic, assignment.cube.T_cam_cube)
        place_origin = _project_origin(camera_intrinsic, assignment.T_cam_place)
        if cube_origin is not None and place_origin is not None:
            cv2.line(display, cube_origin, place_origin, color, 2, cv2.LINE_AA)
    return display


def _checkpoint8_validate_multi_plan_distinct(entries: list[Checkpoint8MultiPlanEntry], min_distance_m: float = 0.005) -> None:
    for left_index, left in enumerate(entries):
        for right in entries[left_index + 1 :]:
            tag_distance = float(np.linalg.norm(left.T_robot_tag[:2, 3] - right.T_robot_tag[:2, 3]))
            if tag_distance < min_distance_m:
                raise RuntimeError(
                    "Target tag poses are not distinct enough for multi-place execution: "
                    f"tag {left.target_tag_id} and tag {right.target_tag_id} are {tag_distance * 1000.0:.1f} mm apart."
                )
            cube_distance = float(np.linalg.norm(left.T_robot_cube[:2, 3] - right.T_robot_cube[:2, 3]))
            if cube_distance < min_distance_m:
                raise RuntimeError(
                    "Source cube poses are not distinct enough for multi-place execution: "
                    f"{left.cube_prompt!r} and {right.cube_prompt!r} are {cube_distance * 1000.0:.1f} mm apart."
                )


def _checkpoint8_detect_cube_for_prompt(
    image: np.ndarray,
    point_cloud: np.ndarray,
    camera_intrinsic: np.ndarray,
    T_cam_robot: np.ndarray,
    cube_prompt: str,
    get_transform_cube: Any,
) -> tuple[np.ndarray, np.ndarray]:
    cube_result = get_transform_cube(
        [image, point_cloud],
        camera_intrinsic,
        T_cam_robot,
        cube_prompt=cube_prompt,
    )
    if cube_result is None:
        raise RuntimeError(f"No cube matched prompt: {cube_prompt}")
    T_robot_cube, T_cam_cube = cube_result
    if T_robot_cube is None or T_cam_cube is None:
        raise RuntimeError(f"No cube matched prompt: {cube_prompt}")
    return T_robot_cube, T_cam_cube


def _checkpoint8_detect_target_tag_for_id(
    image: np.ndarray,
    camera_intrinsic: np.ndarray,
    T_cam_robot: np.ndarray,
    target_tag_id: int,
    target_tag_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    T_robot_tag, T_cam_tag = _estimate_detected_target_tag_pose(
        image=image,
        camera_intrinsic=camera_intrinsic,
        T_cam_base=T_cam_robot,
        target_tag_id=int(target_tag_id),
        target_tag_size_m=float(target_tag_size_m),
    )
    if T_robot_tag is None or T_cam_tag is None:
        raise RuntimeError(f"Target AprilTag id {target_tag_id} was not detected with a valid pose.")
    return T_robot_tag, T_cam_tag


def _checkpoint8_build_multi_plan_entries(
    image: np.ndarray,
    point_cloud: np.ndarray,
    camera_intrinsic: np.ndarray,
    T_cam_robot: np.ndarray,
    cube_tag_pairs: list[CubeTagPair],
    target_tag_size_m: float,
    get_transform_cube: Any,
) -> list[Checkpoint8MultiPlanEntry]:
    entries: list[Checkpoint8MultiPlanEntry] = []
    for index, pair in enumerate(cube_tag_pairs, start=1):
        print(f"\nDetecting multi-place pair {index}: {pair.cube_prompt!r} -> tag {pair.target_tag_id}")
        T_robot_cube, T_cam_cube = _checkpoint8_detect_cube_for_prompt(
            image=image,
            point_cloud=point_cloud,
            camera_intrinsic=camera_intrinsic,
            T_cam_robot=T_cam_robot,
            cube_prompt=pair.cube_prompt,
            get_transform_cube=get_transform_cube,
        )
        T_robot_tag, T_cam_tag = _checkpoint8_detect_target_tag_for_id(
            image=image,
            camera_intrinsic=camera_intrinsic,
            T_cam_robot=T_cam_robot,
            target_tag_id=pair.target_tag_id,
            target_tag_size_m=target_tag_size_m,
        )
        T_robot_place = _checkpoint8_target_pose_from_tag(T_robot_cube, T_robot_tag)
        T_cam_place = T_cam_robot @ T_robot_place
        entries.append(
            Checkpoint8MultiPlanEntry(
                index=index,
                cube_prompt=pair.cube_prompt,
                target_tag_id=pair.target_tag_id,
                T_robot_cube=T_robot_cube,
                T_cam_cube=T_cam_cube,
                T_robot_tag=T_robot_tag,
                T_cam_tag=T_cam_tag,
                T_robot_place=T_robot_place,
                T_cam_place=T_cam_place,
            )
        )

    _checkpoint8_validate_multi_plan_distinct(entries)
    return entries


def _checkpoint8_draw_multi_preview(
    image: np.ndarray,
    camera_intrinsic: np.ndarray,
    entries: list[Checkpoint8MultiPlanEntry],
    cube_size_m: float,
    target_tag_size_m: float,
    draw_pose_axes_fn: Any,
) -> np.ndarray:
    display = _bgr_from_image(image)
    for entry in entries:
        draw_pose_axes_fn(display, camera_intrinsic, entry.T_cam_cube, size=float(cube_size_m) * 1.5)
        draw_pose_axes_fn(display, camera_intrinsic, entry.T_cam_tag, size=float(target_tag_size_m) * 0.75)
        draw_pose_axes_fn(display, camera_intrinsic, entry.T_cam_place, size=float(cube_size_m) * 1.5)
        _draw_label(display, camera_intrinsic, entry.T_cam_cube, f"{entry.index}: {entry.cube_prompt}", (0, 255, 255))
        _draw_label(display, camera_intrinsic, entry.T_cam_tag, f"tag {entry.target_tag_id}", (255, 255, 0))
        _draw_label(display, camera_intrinsic, entry.T_cam_place, f"place {entry.index}", (255, 0, 255))
    return display


def _checkpoint8_multi_preflight(
    zed: Any,
    cube_tag_pairs: list[CubeTagPair],
    target_tag_size_m: float,
    get_transform_camera_robot: Any,
    get_transform_cube: Any,
    draw_pose_axes_fn: Any,
    cube_size_m: float,
    no_gui: bool,
    preview_path: str,
    dry_run: bool,
    auto_confirm: bool = False,
    skip_confirmation: bool = False,
) -> tuple[bool, list[Checkpoint8MultiPlanEntry]]:
    image, point_cloud = _checkpoint8_capture_frame(zed)
    if image is None or point_cloud is None:
        raise RuntimeError("Camera data is not ready.")

    camera_intrinsic = zed.camera_intrinsic
    T_cam_robot = get_transform_camera_robot(image, camera_intrinsic)
    if T_cam_robot is None:
        raise RuntimeError("Camera-to-robot calibration failed.")

    entries = _checkpoint8_build_multi_plan_entries(
        image=image,
        point_cloud=point_cloud,
        camera_intrinsic=camera_intrinsic,
        T_cam_robot=T_cam_robot,
        cube_tag_pairs=cube_tag_pairs,
        target_tag_size_m=target_tag_size_m,
        get_transform_cube=get_transform_cube,
    )
    _checkpoint8_print_multi_plan_table(entries)
    display = _checkpoint8_draw_multi_preview(
        image=image,
        camera_intrinsic=camera_intrinsic,
        entries=entries,
        cube_size_m=cube_size_m,
        target_tag_size_m=target_tag_size_m,
        draw_pose_axes_fn=draw_pose_axes_fn,
    )
    if skip_confirmation:
        saved_path = _save_preview_image(display, preview_path)
        if saved_path is not None:
            print(f"Saved checkpoint8-style preview: {saved_path}")
        print("Parent subprocess preflight completed; interactive confirmation will happen once before launching children.")
        return False, entries

    confirmed = _checkpoint8_save_or_confirm(
        display=display,
        no_gui=no_gui,
        preview_path=preview_path,
        dry_run=dry_run,
        auto_confirm=auto_confirm,
    )
    return confirmed, entries


def _build_duplicate_assignments_for_group(
    group_index: int,
    group: DuplicateCubeTagGroup,
    cube_candidates: list[DuplicateCubeCandidate],
    tag_candidates: list[DuplicateTagCandidate],
    max_assignment_distance_m: float,
    candidate_x_min_m: float,
    candidate_x_max_m: float,
    candidate_y_min_m: float,
    candidate_y_max_m: float,
    candidate_z_min_m: float,
    candidate_z_max_m: float,
    candidate_min_area_px: int,
    red_candidate_min_area_px: int,
    green_candidate_min_area_px: int,
    blue_candidate_min_area_px: int,
    candidate_min_extent_m: float,
    candidate_max_extent_m: float,
    candidate_merge_distance_m: float,
    candidate_merge_prompts: list[str],
    next_execution_index: int,
    T_cam_robot: np.ndarray,
) -> tuple[list[DuplicateAssignedPair], dict[str, Any]]:
    _print_duplicate_cube_candidates(group, cube_candidates)
    _print_duplicate_tag_candidates(group, tag_candidates)
    valid_cube_candidates, rejected_cube_candidates = filter_duplicate_cube_candidates(
        candidates=cube_candidates,
        x_min_m=candidate_x_min_m,
        x_max_m=candidate_x_max_m,
        y_min_m=candidate_y_min_m,
        y_max_m=candidate_y_max_m,
        z_min_m=candidate_z_min_m,
        z_max_m=candidate_z_max_m,
        min_area_px=candidate_min_area_px,
        red_min_area_px=red_candidate_min_area_px,
        green_min_area_px=green_candidate_min_area_px,
        blue_min_area_px=blue_candidate_min_area_px,
        min_extent_m=candidate_min_extent_m,
        max_extent_m=candidate_max_extent_m,
    )
    _print_rejected_duplicate_cube_candidates(rejected_cube_candidates)
    physical_cube_candidates, physical_cube_clusters = merge_duplicate_cube_candidates(
        candidates=valid_cube_candidates,
        merge_distance_m=candidate_merge_distance_m,
        T_cam_robot=T_cam_robot,
    )
    _print_merged_duplicate_cube_candidates(physical_cube_clusters)
    merged_cube_candidates = [
        cluster for cluster in physical_cube_clusters if len(cluster.merged_candidates) > 1
    ]
    print(
        f"Duplicate cube candidate filter for {group.cube_prompt!r}:"
        f" raw={len(cube_candidates)}"
        f" rejected={len(rejected_cube_candidates)}"
        f" valid={len(valid_cube_candidates)}"
        f" clusters={len(physical_cube_clusters)}"
        f" merged_clusters={len(merged_cube_candidates)}"
        f" physical={len(physical_cube_candidates)}"
    )
    print(
        "Candidate filter bounds:"
        f" x=[{candidate_x_min_m * 1000.0:.0f},{candidate_x_max_m * 1000.0:.0f}]mm"
        f" y=[{candidate_y_min_m * 1000.0:.0f},{candidate_y_max_m * 1000.0:.0f}]mm"
        f" z=[{candidate_z_min_m * 1000.0:.0f},{candidate_z_max_m * 1000.0:.0f}]mm"
        f" min_area_px={int(candidate_min_area_px)}"
        f" red_min_area_px={int(red_candidate_min_area_px)}"
        f" green_min_area_px={int(green_candidate_min_area_px)}"
        f" blue_min_area_px={int(blue_candidate_min_area_px)}"
        f" extent=[{candidate_min_extent_m * 1000.0:.1f},{candidate_max_extent_m * 1000.0:.1f}]mm"
        f" merge_distance={candidate_merge_distance_m * 1000.0:.1f}mm"
    )
    print(
        f"Physical cube candidates for {group.cube_prompt!r}: "
        + (", ".join(f"#{candidate.instance_index}" for candidate in physical_cube_candidates) or "none")
    )
    if len(physical_cube_candidates) < group.count:
        raise RuntimeError(
            f"Insufficient physical cube candidates for {group.cube_prompt!r}: "
            f"need {group.count}, found {len(physical_cube_candidates)} "
            f"(raw={len(cube_candidates)}, rejected={len(rejected_cube_candidates)}, "
            f"valid={len(valid_cube_candidates)}, clusters={len(physical_cube_clusters)})."
        )
    if len(tag_candidates) < group.count:
        raise RuntimeError(
            f"Insufficient tag candidates for tag {group.tag_id}: "
            f"need {group.count}, found {len(tag_candidates)}."
        )

    selection = select_duplicate_assignment_subset(
        cube_candidates=physical_cube_candidates,
        tag_candidates=tag_candidates,
        count=group.count,
        max_distance_m=max_assignment_distance_m,
    )
    selected_cubes = selection.selected_cubes
    selected_tags = selection.selected_tags
    _print_duplicate_selected_candidates(group, selected_cubes, selected_tags)
    _print_duplicate_distance_matrix(group, selected_cubes, selected_tags, selection.distance_matrix)
    print(
        "Final pair distances mm:"
        f" {[round(distance * 1000.0, 1) for distance in selection.pair_distances]}"
        f" total={selection.total_distance_m * 1000.0:.1f}"
        f" objective={selection.objective:.6f}"
    )

    assignments: list[DuplicateAssignedPair] = []
    assignment_rows: list[dict[str, Any]] = []
    for cube_order_index, tag_order_index in enumerate(selection.tag_permutation):
        cube = selected_cubes[cube_order_index]
        tag = selected_tags[tag_order_index]
        T_robot_place = _checkpoint8_target_pose_from_tag(cube.T_robot_cube, tag.T_robot_tag)
        T_cam_place = T_cam_robot @ T_robot_place
        execution_index = next_execution_index + len(assignments)
        assignment = DuplicateAssignedPair(
            execution_index=execution_index,
            group_index=group_index,
            within_group_index=cube_order_index + 1,
            cube_prompt=group.cube_prompt,
            tag_id=group.tag_id,
            cube=cube,
            tag=tag,
            distance_m=float(selection.pair_distances[cube_order_index]),
            T_robot_place=T_robot_place,
            T_cam_place=T_cam_place,
        )
        assignments.append(assignment)
        assignment_rows.append(
            {
                "cube_instance_index": int(cube.instance_index),
                "tag_instance_index": int(tag.instance_index),
                "distance_m": float(selection.pair_distances[cube_order_index]),
            }
        )

    group_report = {
        "group_index": int(group_index),
        "cube_prompt": group.cube_prompt,
        "tag_id": int(group.tag_id),
        "count": int(group.count),
        "filter_parameters": {
            "candidate_x_min_m": float(candidate_x_min_m),
            "candidate_x_max_m": float(candidate_x_max_m),
            "candidate_y_min_m": float(candidate_y_min_m),
            "candidate_y_max_m": float(candidate_y_max_m),
            "candidate_z_min_m": float(candidate_z_min_m),
            "candidate_z_max_m": float(candidate_z_max_m),
            "candidate_min_area_px": int(candidate_min_area_px),
            "red_candidate_min_area_px": int(red_candidate_min_area_px),
            "green_candidate_min_area_px": int(green_candidate_min_area_px),
            "blue_candidate_min_area_px": int(blue_candidate_min_area_px),
            "candidate_min_extent_m": float(candidate_min_extent_m),
            "candidate_max_extent_m": float(candidate_max_extent_m),
            "candidate_merge_distance_m": float(candidate_merge_distance_m),
            "candidate_merge_prompts": list(candidate_merge_prompts),
            "candidate_merge_prompts_ignored": True,
        },
        "merge_enabled": True,
        "raw_cube_candidates": [_candidate_cube_report(candidate) for candidate in cube_candidates],
        "rejected_cube_candidates": [
            _rejected_candidate_cube_report(rejected) for rejected in rejected_cube_candidates
        ],
        "valid_cube_candidates": [_candidate_cube_report(candidate) for candidate in valid_cube_candidates],
        "physical_cube_clusters": [
            _merged_candidate_cube_report(cluster) for cluster in physical_cube_clusters
        ],
        "merged_cube_candidates": [
            _merged_candidate_cube_report(merged) for merged in merged_cube_candidates
        ],
        "physical_cube_candidates": [_candidate_cube_report(candidate) for candidate in physical_cube_candidates],
        "detected_cube_candidates": [_candidate_cube_report(candidate) for candidate in cube_candidates],
        "detected_tag_candidates": [_candidate_tag_report(candidate) for candidate in tag_candidates],
        "selected_cube_instance_indices": [int(candidate.instance_index) for candidate in selected_cubes],
        "selected_physical_candidate_indices": [int(candidate.instance_index) for candidate in selected_cubes],
        "selected_tag_instance_indices": [int(candidate.instance_index) for candidate in selected_tags],
        "distance_matrix_m": selection.distance_matrix.tolist(),
        "pair_distances_m": [float(distance) for distance in selection.pair_distances],
        "total_assignment_distance_m": float(selection.total_distance_m),
        "assignment_objective": float(selection.objective),
        "assignment": assignment_rows,
    }
    return assignments, group_report


def _slot_as_tag_candidate(slot: PresetSlot) -> DuplicateTagCandidate:
    T_robot_slot = np.eye(4, dtype=np.float64)
    T_robot_slot[:3, :3] = _top_down_rotation(math.radians(float(slot.yaw_deg)))
    T_robot_slot[:3, 3] = slot.center_robot
    return DuplicateTagCandidate(
        tag_id=int(slot.slot_id),
        instance_index=int(slot.slot_id),
        detection_index=int(slot.slot_id),
        decision_margin=0.0,
        hamming=0,
        T_robot_tag=T_robot_slot,
        T_cam_tag=T_robot_slot.copy(),
        center_robot=slot.center_robot,
    )


def _build_preset_assignments_for_group(
    group_index: int,
    group: PresetCubeGroup,
    cube_candidates: list[DuplicateCubeCandidate],
    slots: list[PresetSlot],
    candidate_x_min_m: float,
    candidate_x_max_m: float,
    candidate_y_min_m: float,
    candidate_y_max_m: float,
    candidate_z_min_m: float,
    candidate_z_max_m: float,
    candidate_min_area_px: int,
    red_candidate_min_area_px: int,
    green_candidate_min_area_px: int,
    blue_candidate_min_area_px: int,
    candidate_min_extent_m: float,
    candidate_max_extent_m: float,
    candidate_merge_distance_m: float,
    next_execution_index: int,
    T_cam_robot: np.ndarray,
    preset_use_slot_yaw: bool,
) -> tuple[list[PresetAssignedPair], dict[str, Any]]:
    _print_duplicate_cube_candidates(
        DuplicateCubeTagGroup(cube_prompt=group.cube_prompt, tag_id=0, count=group.count),
        cube_candidates,
    )
    print(f"Preset slots for {group.cube_prompt!r}: " + ", ".join(str(slot.slot_id) for slot in slots))
    valid_cube_candidates, rejected_cube_candidates = filter_duplicate_cube_candidates(
        candidates=cube_candidates,
        x_min_m=candidate_x_min_m,
        x_max_m=candidate_x_max_m,
        y_min_m=candidate_y_min_m,
        y_max_m=candidate_y_max_m,
        z_min_m=candidate_z_min_m,
        z_max_m=candidate_z_max_m,
        min_area_px=candidate_min_area_px,
        red_min_area_px=red_candidate_min_area_px,
        green_min_area_px=green_candidate_min_area_px,
        blue_min_area_px=blue_candidate_min_area_px,
        min_extent_m=candidate_min_extent_m,
        max_extent_m=candidate_max_extent_m,
    )
    _print_rejected_duplicate_cube_candidates(rejected_cube_candidates)
    physical_cube_candidates, physical_cube_clusters = merge_duplicate_cube_candidates(
        candidates=valid_cube_candidates,
        merge_distance_m=candidate_merge_distance_m,
        T_cam_robot=T_cam_robot,
    )
    _print_merged_duplicate_cube_candidates(physical_cube_clusters)
    print(
        f"Preset cube candidate filter for {group.cube_prompt!r}:"
        f" raw={len(cube_candidates)}"
        f" rejected={len(rejected_cube_candidates)}"
        f" valid={len(valid_cube_candidates)}"
        f" clusters={len(physical_cube_clusters)}"
        f" physical={len(physical_cube_candidates)}"
    )
    if len(physical_cube_candidates) < group.count:
        raise RuntimeError(
            f"Insufficient physical cube candidates for preset group {group.cube_prompt!r}: "
            f"need {group.count}, found {len(physical_cube_candidates)}."
        )

    selection = select_preset_assignment_subset(
        cube_candidates=physical_cube_candidates,
        slots=slots,
        count=group.count,
    )
    selected_cubes = selection.selected_cubes
    selected_slots = selection.selected_slots
    print(f"Selected cube candidates for preset {group.cube_prompt!r}: " + ", ".join(f"#{candidate.instance_index}" for candidate in selected_cubes))
    print(f"Selected preset slots for {group.cube_prompt!r}: " + ", ".join(f"{slot.slot_id}" for slot in selected_slots))
    _print_preset_distance_matrix(group, selected_cubes, selected_slots, selection.distance_matrix)
    print(
        "Final preset pair distances mm:"
        f" {[round(distance * 1000.0, 1) for distance in selection.pair_distances]}"
        f" total={selection.total_distance_m * 1000.0:.1f}"
        f" objective={selection.objective:.6f}"
    )

    assignments: list[PresetAssignedPair] = []
    assignment_rows: list[dict[str, Any]] = []
    for cube_order_index, slot_order_index in enumerate(selection.slot_permutation):
        cube = selected_cubes[cube_order_index]
        slot = selected_slots[slot_order_index]
        T_robot_place = _preset_place_pose_from_slot(
            T_base_cube=cube.T_robot_cube,
            slot=slot,
            preset_use_slot_yaw=preset_use_slot_yaw,
        )
        T_cam_place = T_cam_robot @ T_robot_place
        execution_index = next_execution_index + len(assignments)
        assignment = PresetAssignedPair(
            execution_index=execution_index,
            group_index=group_index,
            within_group_index=cube_order_index + 1,
            cube_prompt=group.cube_prompt,
            cube=cube,
            slot=slot,
            tag=_slot_as_tag_candidate(slot),
            distance_m=float(selection.pair_distances[cube_order_index]),
            T_robot_place=T_robot_place,
            T_cam_place=T_cam_place,
            preset_use_slot_yaw=bool(preset_use_slot_yaw),
        )
        assignments.append(assignment)
        assignment_rows.append(
            {
                "cube_instance_index": int(cube.instance_index),
                "slot_id": int(slot.slot_id),
                "distance_m": float(selection.pair_distances[cube_order_index]),
            }
        )

    group_report = {
        "group_index": int(group_index),
        "cube_prompt": group.cube_prompt,
        "count": int(group.count),
        "slot_ids": [int(slot.slot_id) for slot in slots],
        "raw_cube_candidates": [_candidate_cube_report(candidate) for candidate in cube_candidates],
        "rejected_cube_candidates": [
            _rejected_candidate_cube_report(rejected) for rejected in rejected_cube_candidates
        ],
        "valid_cube_candidates": [_candidate_cube_report(candidate) for candidate in valid_cube_candidates],
        "physical_cube_clusters": [
            _merged_candidate_cube_report(cluster) for cluster in physical_cube_clusters
        ],
        "physical_cube_candidates": [_candidate_cube_report(candidate) for candidate in physical_cube_candidates],
        "selected_cube_instance_indices": [int(candidate.instance_index) for candidate in selected_cubes],
        "selected_slot_ids": [int(slot.slot_id) for slot in selected_slots],
        "distance_matrix_m": selection.distance_matrix.tolist(),
        "pair_distances_m": [float(distance) for distance in selection.pair_distances],
        "total_assignment_distance_m": float(selection.total_distance_m),
        "assignment_objective": float(selection.objective),
        "assignment": assignment_rows,
    }
    return assignments, group_report


def _duplicate_aware_preflight(
    args: argparse.Namespace,
    zed: Any,
    get_transform_camera_robot: Any,
    draw_pose_axes_fn: Any,
    cube_size_m: float,
) -> tuple[list[DuplicateAssignedPair], list[dict[str, Any]], Path | None]:
    image, point_cloud = _checkpoint8_capture_frame(zed)
    if image is None or point_cloud is None:
        raise RuntimeError("Camera data is not ready.")

    camera_intrinsic = zed.camera_intrinsic
    T_cam_robot = get_transform_camera_robot(image, camera_intrinsic)
    if T_cam_robot is None:
        raise RuntimeError("Camera-to-robot calibration failed.")

    groups = _duplicate_groups_from_args(args)
    assignments: list[DuplicateAssignedPair] = []
    group_reports: list[dict[str, Any]] = []
    print("\nRunning duplicate-aware multi-place preflight.")
    print(f"Assignment metric: {args.assignment_metric}")
    print(f"Assignment space: {args.assignment_space}")
    print(f"Max assignment distance: {float(args.max_assignment_distance_m) * 1000.0:.1f} mm")
    candidate_merge_prompts = parse_candidate_merge_prompts(args.candidate_merge_prompts)
    print(
        "duplicate-aware mode clusters all colors into physical cube instances; "
        "--candidate_merge_prompts is ignored."
    )
    print(f"Ignored candidate_merge_prompts value: {candidate_merge_prompts}")

    for group_index, group in enumerate(groups, start=1):
        print(f"\nDuplicate-aware group {group_index}/{len(groups)}: {group.cube_prompt!r} -> tag {group.tag_id} x {group.count}")
        cube_candidates = _detect_duplicate_cube_candidates(
            image=image,
            point_cloud=point_cloud,
            T_cam_robot=T_cam_robot,
            cube_prompt=group.cube_prompt,
            cube_size_m=cube_size_m,
            table_z_m=args.table_z_m,
            point_cloud_scale=args.point_cloud_scale,
        )
        tag_candidates = _detect_duplicate_target_tag_candidates(
            image=image,
            camera_intrinsic=camera_intrinsic,
            T_cam_robot=T_cam_robot,
            target_tag_id=group.tag_id,
            target_tag_size_m=args.target_tag_size_m,
        )
        group_assignments, group_report = _build_duplicate_assignments_for_group(
            group_index=group_index,
            group=group,
            cube_candidates=cube_candidates,
            tag_candidates=tag_candidates,
            max_assignment_distance_m=args.max_assignment_distance_m,
            candidate_x_min_m=args.candidate_x_min_m,
            candidate_x_max_m=args.candidate_x_max_m,
            candidate_y_min_m=args.candidate_y_min_m,
            candidate_y_max_m=args.candidate_y_max_m,
            candidate_z_min_m=args.candidate_z_min_m,
            candidate_z_max_m=args.candidate_z_max_m,
            candidate_min_area_px=args.candidate_min_area_px,
            red_candidate_min_area_px=args.red_candidate_min_area_px,
            green_candidate_min_area_px=args.green_candidate_min_area_px,
            blue_candidate_min_area_px=args.blue_candidate_min_area_px,
            candidate_min_extent_m=args.candidate_min_extent_m,
            candidate_max_extent_m=args.candidate_max_extent_m,
            candidate_merge_distance_m=args.candidate_merge_distance_m,
            candidate_merge_prompts=candidate_merge_prompts,
            next_execution_index=len(assignments) + 1,
            T_cam_robot=T_cam_robot,
        )
        assignments.extend(group_assignments)
        group_reports.append(group_report)

    _print_duplicate_assignment_table(assignments)
    _print_duplicate_execution_order(assignments)

    display = _draw_duplicate_assignment_preview(
        image=image,
        camera_intrinsic=camera_intrinsic,
        assignments=assignments,
        cube_size_m=cube_size_m,
        target_tag_size_m=args.target_tag_size_m,
        draw_pose_axes_fn=draw_pose_axes_fn,
    )
    preview_path = _save_preview_image(display, DUPLICATE_ASSIGNMENT_PREVIEW_PATH)
    if preview_path is not None:
        print(f"Saved duplicate-aware assignment preview: {preview_path}")

    if args.save_assignment_report:
        _save_duplicate_assignment_report(
            groups=group_reports,
            assignments=assignments,
            preview_path=preview_path,
            plan_paths=None,
            execution_confirmed=False,
        )
    return assignments, group_reports, preview_path


def _preset_layout_preflight(
    args: argparse.Namespace,
    zed: Any,
    get_transform_camera_robot: Any,
    draw_pose_axes_fn: Any,
    cube_size_m: float,
) -> tuple[list[PresetAssignedPair], list[dict[str, Any]], Path | None]:
    image, point_cloud = _checkpoint8_capture_frame(zed)
    if image is None or point_cloud is None:
        raise RuntimeError("Camera data is not ready.")

    camera_intrinsic = zed.camera_intrinsic
    T_cam_robot = get_transform_camera_robot(image, camera_intrinsic)
    if T_cam_robot is None:
        raise RuntimeError("Camera-to-robot calibration failed.")

    layout = getattr(args, "_preset_layout", None)
    if layout is None:
        layout = load_preset_place_layout_json(args.preset_place_layout_json)
        args._preset_layout = layout
    groups = _preset_groups_from_args(args)
    slot_map = _preset_slot_map_from_args(args)

    print("\nRunning preset layout place preflight.")
    print(f"Layout file path: {args.preset_place_layout_json}")
    print(f"Loaded preset layout: name={layout.name} frame={layout.frame}")
    for slot in layout.slots.values():
        print(
            f"  slot {slot.slot_id}:"
            f" x={slot.x * 1000.0:.1f}mm"
            f" y={slot.y * 1000.0:.1f}mm"
            f" z={slot.z * 1000.0:.1f}mm"
            f" yaw={slot.yaw_deg:.1f}deg"
        )
    print("Preset cube counts: " + ", ".join(f"{group.cube_prompt}:{group.count}" for group in groups))
    print(
        "Preset cube-slot map: "
        + "; ".join(
            f"{group.cube_prompt}:{','.join(str(slot_id) for slot_id in slot_map[_normalize_cube_prompt_key(group.cube_prompt)])}"
            for group in groups
        )
    )
    print(f"Preset assignment metric: {args.preset_assignment_metric}")
    print(f"Preset use slot yaw: {bool(args.preset_use_slot_yaw)}")

    assignments: list[PresetAssignedPair] = []
    group_reports: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups, start=1):
        prompt_key = _normalize_cube_prompt_key(group.cube_prompt)
        slots = [layout.slots[int(slot_id)] for slot_id in slot_map[prompt_key]]
        print(f"\nPreset group {group_index}/{len(groups)}: {group.cube_prompt!r} x {group.count}")
        cube_candidates = _detect_duplicate_cube_candidates(
            image=image,
            point_cloud=point_cloud,
            T_cam_robot=T_cam_robot,
            cube_prompt=group.cube_prompt,
            cube_size_m=cube_size_m,
            table_z_m=args.table_z_m,
            point_cloud_scale=args.point_cloud_scale,
        )
        group_assignments, group_report = _build_preset_assignments_for_group(
            group_index=group_index,
            group=group,
            cube_candidates=cube_candidates,
            slots=slots,
            candidate_x_min_m=args.candidate_x_min_m,
            candidate_x_max_m=args.candidate_x_max_m,
            candidate_y_min_m=args.candidate_y_min_m,
            candidate_y_max_m=args.candidate_y_max_m,
            candidate_z_min_m=args.candidate_z_min_m,
            candidate_z_max_m=args.candidate_z_max_m,
            candidate_min_area_px=args.candidate_min_area_px,
            red_candidate_min_area_px=args.red_candidate_min_area_px,
            green_candidate_min_area_px=args.green_candidate_min_area_px,
            blue_candidate_min_area_px=args.blue_candidate_min_area_px,
            candidate_min_extent_m=args.candidate_min_extent_m,
            candidate_max_extent_m=args.candidate_max_extent_m,
            candidate_merge_distance_m=args.candidate_merge_distance_m,
            next_execution_index=len(assignments) + 1,
            T_cam_robot=T_cam_robot,
            preset_use_slot_yaw=args.preset_use_slot_yaw,
        )
        assignments.extend(group_assignments)
        group_reports.append(group_report)

    _print_preset_assignment_table(assignments)
    _print_preset_execution_order(assignments)

    display = _draw_preset_assignment_preview(
        image=image,
        camera_intrinsic=camera_intrinsic,
        assignments=assignments,
        cube_size_m=cube_size_m,
        draw_pose_axes_fn=draw_pose_axes_fn,
    )
    preview_path = _save_preview_image(display, PRESET_LAYOUT_PREVIEW_PATH)
    if preview_path is not None:
        print(f"Saved preset layout assignment preview: {preview_path}")

    _save_preset_assignment_report(
        layout=layout,
        groups=group_reports,
        assignments=assignments,
        preview_path=preview_path,
        plan_paths=None,
        execution_confirmed=False,
    )
    return assignments, group_reports, preview_path


def _run_checkpoint8_style_multi_subprocess_parent_preflight(
    args: argparse.Namespace,
    cube_tag_pairs: list[CubeTagPair],
) -> bool:
    from checkpoint0 import get_transform_camera_robot
    from checkpoint6 import CUBE_SIZE, get_transform_cube
    from utils.vis_utils import draw_pose_axes as checkpoint_draw_pose_axes
    from utils.zed_camera import ZedCamera as CheckpointZedCamera

    zed: Any | None = None
    try:
        print("Running parent subprocess preflight: one frame, all requested cubes/tags, no Lite6 connection.")
        zed = CheckpointZedCamera()
        _confirmed, _entries = _checkpoint8_multi_preflight(
            zed=zed,
            cube_tag_pairs=cube_tag_pairs,
            target_tag_size_m=args.target_tag_size_m,
            get_transform_camera_robot=get_transform_camera_robot,
            get_transform_cube=get_transform_cube,
            draw_pose_axes_fn=checkpoint_draw_pose_axes,
            cube_size_m=CUBE_SIZE,
            no_gui=args.no_gui,
            preview_path=args.preview_path,
            dry_run=False,
            skip_confirmation=True,
        )
        return True
    except Exception as exc:
        print(f"Parent multi subprocess preflight failed before starting child processes: {exc}")
        return False
    finally:
        if zed is not None:
            zed.close()
        if (not args.no_gui) and _gui_display_available():
            try:
                cv2.destroyAllWindows()
            except cv2.error as exc:
                print(f"Warning: failed to destroy OpenCV windows cleanly: {exc}")


def _confirm_checkpoint8_multi_subprocess_parent(args: argparse.Namespace) -> bool:
    if args.auto_confirm:
        print("Auto-confirm enabled: executing all subprocess pairs without interactive confirmation.")
        return True
    try:
        response = input("Type 'k' then Enter to execute all pairs, or anything else to cancel: ")
    except EOFError:
        print("No terminal input is available. Cancelling before starting child processes.")
        return False
    return response.strip().lower() == "k"


def _run_checkpoint8_style_multi_subprocess(args: argparse.Namespace, robot_ip: str) -> None:
    del robot_ip
    cube_tag_pairs = _cube_tag_pairs_from_args(args)
    total_pairs = len(cube_tag_pairs)
    succeeded_pairs: list[CubeTagPair] = []
    failed_pairs: list[CubeTagPair] = []

    print("Using checkpoint8_style multi_place_to_tags subprocess parent runner.")
    print("Parent process will run one checkpoint8_style child process per cube/tag pair.")
    _print_multi_subprocess_pair_map(cube_tag_pairs)
    _print_multi_subprocess_child_commands(args, cube_tag_pairs)

    if not _run_checkpoint8_style_multi_subprocess_parent_preflight(args, cube_tag_pairs):
        _print_multi_subprocess_summary(
            total_pairs=total_pairs,
            succeeded_pairs=succeeded_pairs,
            failed_pairs=cube_tag_pairs,
        )
        raise SystemExit(1)

    if not _confirm_checkpoint8_multi_subprocess_parent(args):
        print("Operator cancelled. No child subprocesses started.")
        _print_multi_subprocess_summary(
            total_pairs=total_pairs,
            succeeded_pairs=succeeded_pairs,
            failed_pairs=failed_pairs,
        )
        return

    for index, pair in enumerate(cube_tag_pairs, start=1):
        command = _build_checkpoint8_multi_subprocess_child_command(args, pair)
        print(f"\nStarting subprocess pair {index}/{total_pairs}: {pair.cube_prompt} -> tag {pair.target_tag_id}")
        print(f"Child command: {_format_command_for_log(command)}")

        completed = subprocess.run(command, cwd=str(REPO_ROOT))
        returncode = int(completed.returncode)
        print(f"Child return code: {returncode}")

        if returncode == 0:
            succeeded_pairs.append(pair)
            print(f"Pair {index}/{total_pairs} succeeded.")
            continue

        failed_pairs.append(pair)
        print(f"Pair {index}/{total_pairs} failed: {pair.cube_prompt} -> tag {pair.target_tag_id}")
        if _child_returncode_looks_like_native_crash(returncode):
            print("Child process crashed, likely native perception stack failure.")
        if not args.continue_on_pair_failure:
            print("Aborting multi subprocess run after first failed pair.")
            break
        print("Continuing because --continue_on_pair_failure was provided.")

    _print_multi_subprocess_summary(
        total_pairs=total_pairs,
        succeeded_pairs=succeeded_pairs,
        failed_pairs=failed_pairs,
    )
    if failed_pairs:
        raise SystemExit(1)


def _append_pose_plan_refinement_args(command: list[str], args: argparse.Namespace) -> None:
    command.extend(
        [
            "--pose_plan_refine_radius_m",
            f"{float(args.pose_plan_refine_radius_m):.12g}",
            "--pose_plan_refine_tag_radius_m",
            f"{float(args.pose_plan_refine_tag_radius_m):.12g}",
            "--candidate_x_min_m",
            f"{float(args.candidate_x_min_m):.12g}",
            "--candidate_x_max_m",
            f"{float(args.candidate_x_max_m):.12g}",
            "--candidate_y_min_m",
            f"{float(args.candidate_y_min_m):.12g}",
            "--candidate_y_max_m",
            f"{float(args.candidate_y_max_m):.12g}",
            "--candidate_z_min_m",
            f"{float(args.candidate_z_min_m):.12g}",
            "--candidate_z_max_m",
            f"{float(args.candidate_z_max_m):.12g}",
            "--candidate_min_area_px",
            str(int(args.candidate_min_area_px)),
            "--red_candidate_min_area_px",
            str(int(args.red_candidate_min_area_px)),
            "--green_candidate_min_area_px",
            str(int(args.green_candidate_min_area_px)),
            "--blue_candidate_min_area_px",
            str(int(args.blue_candidate_min_area_px)),
            "--candidate_min_extent_m",
            f"{float(args.candidate_min_extent_m):.12g}",
            "--candidate_max_extent_m",
            f"{float(args.candidate_max_extent_m):.12g}",
            "--candidate_merge_distance_m",
            f"{float(args.candidate_merge_distance_m):.12g}",
            "--table_z_m",
            f"{float(args.table_z_m):.12g}",
            "--point_cloud_scale",
            f"{float(args.point_cloud_scale):.12g}",
        ]
    )


def _build_duplicate_pose_plan_refinement_child_command(
    args: argparse.Namespace,
    input_plan_path: Path,
    output_plan_path: Path,
) -> list[str]:
    command = [
        "python",
        "scripts/run_mini_task.py",
        "--execution_backend",
        "checkpoint8_style",
        "--refine_pose_plan_json",
        str(input_plan_path),
        "--refined_pose_plan_output_json",
        str(output_plan_path),
        "--target_tag_size_m",
        _target_tag_size_command_arg(args),
        "--no_gui",
    ]
    _append_pose_plan_refinement_args(command, args)
    return command


def _build_duplicate_pose_plan_robot_child_command(args: argparse.Namespace, plan_path: Path) -> list[str]:
    command = [
        "python",
        "scripts/run_mini_task.py",
        "--execution_backend",
        "checkpoint8_style",
        "--execute_pose_plan_json",
        str(plan_path),
        "--no_pose_plan_refine",
        "--target_tag_size_m",
        _target_tag_size_command_arg(args),
        "--min_after_grasp_z_mm",
        f"{float(args.min_after_grasp_z_mm):.12g}",
        "--no_gui",
    ]
    if args.robot_ip:
        command.extend(["--robot_ip", str(args.robot_ip)])
    if args.robot_config and str(args.robot_config) != "config/robot.yaml":
        command.extend(["--robot_config", str(args.robot_config)])
    return command


def _build_duplicate_pose_plan_child_command(args: argparse.Namespace, plan_path: Path) -> list[str]:
    return _build_duplicate_pose_plan_robot_child_command(args, plan_path)


def _refined_duplicate_pose_plan_path(raw_plan_path: Path) -> Path:
    return raw_plan_path.with_name(f"{raw_plan_path.stem}_refined{raw_plan_path.suffix}")


def _checkpoint8_duplicate_home_precheck(robot_ip: str, gripper_length_mm: float) -> None:
    arm: Any | None = None
    try:
        print("Running checkpoint8-style home precheck before pose-plan children.")
        arm = _checkpoint8_connect_arm(robot_ip)
        _checkpoint8_initialize_and_home(arm, gripper_length_mm)
        print("Home precheck succeeded; no grasp/place motion has been sent by the parent.")
    finally:
        if arm is not None:
            try:
                arm.disconnect()
            except Exception as exc:
                print(f"Warning: failed to disconnect cleanly after home precheck: {exc}")


def _confirm_duplicate_aware_parent(args: argparse.Namespace) -> bool:
    if args.auto_confirm:
        print("Auto-confirm enabled: executing all assigned placements without interactive confirmation.")
        return True
    try:
        response = input("Type 'k' then Enter to execute all assigned placements, or anything else to cancel: ")
    except EOFError:
        print("No terminal input is available. Cancelling before starting child processes.")
        return False
    return response.strip().lower() == "k"


def _pose_xy_yaw_rad(pose: np.ndarray) -> tuple[float, float, float]:
    transform = np.asarray(pose, dtype=np.float64)
    yaw_rad = float(Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=False)[2])
    return float(transform[0, 3]), float(transform[1, 3]), yaw_rad


def _print_pose_plan_refinement_summary(refinement: PosePlanRefinement) -> None:
    plan = refinement.plan
    planned_cube_x, planned_cube_y, planned_cube_yaw = _pose_xy_yaw_rad(plan.T_robot_cube)
    refined_cube_x, refined_cube_y, refined_cube_yaw = _pose_xy_yaw_rad(refinement.T_robot_cube)
    planned_target_xy = np.asarray(plan.T_robot_place[:2, 3], dtype=np.float64)
    refined_target_xy = np.asarray(refinement.T_robot_place[:2, 3], dtype=np.float64)

    print("\nPose-plan local fresh refinement:")
    print(
        "  planned cube x/y/yaw:"
        f" x={planned_cube_x * 1000.0:.1f}mm"
        f" y={planned_cube_y * 1000.0:.1f}mm"
        f" yaw={math.degrees(planned_cube_yaw):.1f}deg"
    )
    print(
        "  refined cube x/y/yaw:"
        f" x={refined_cube_x * 1000.0:.1f}mm"
        f" y={refined_cube_y * 1000.0:.1f}mm"
        f" yaw={math.degrees(refined_cube_yaw):.1f}deg"
    )
    print(f"  cube refinement delta in mm: {refinement.cube_delta_m * 1000.0:.1f}")
    print(
        "  planned target x/y:"
        f" x={planned_target_xy[0] * 1000.0:.1f}mm"
        f" y={planned_target_xy[1] * 1000.0:.1f}mm"
    )
    print(
        "  refined target x/y:"
        f" x={refined_target_xy[0] * 1000.0:.1f}mm"
        f" y={refined_target_xy[1] * 1000.0:.1f}mm"
    )
    print(f"  target refinement delta in mm: {refinement.tag_delta_m * 1000.0:.1f}")


def _preset_slot_from_pose_plan(plan: DuplicatePosePlan) -> PresetSlot:
    if plan.preset_slot is None:
        raise ValueError("preset-slot pose plan is missing preset_slot data.")
    slot = _preset_slot_from_json_data(plan.preset_slot, 1)
    if plan.slot_id is not None and int(plan.slot_id) != int(slot.slot_id):
        raise ValueError(
            f"preset-slot pose plan slot_id mismatch: slot_id={plan.slot_id}, preset_slot.slot_id={slot.slot_id}."
        )
    return slot


def _refine_duplicate_pose_plan_from_frame(
    args: argparse.Namespace,
    plan: DuplicatePosePlan,
    image: np.ndarray,
    point_cloud: np.ndarray,
    camera_intrinsic: np.ndarray,
    T_cam_robot: np.ndarray,
    cube_size_m: float,
) -> PosePlanRefinement:
    print("Re-detecting duplicate-aware physical cube candidates for pose-plan refinement.")
    cube_candidates = _detect_duplicate_cube_candidates(
        image=image,
        point_cloud=point_cloud,
        T_cam_robot=T_cam_robot,
        cube_prompt=plan.cube_prompt,
        cube_size_m=cube_size_m,
        table_z_m=args.table_z_m,
        point_cloud_scale=args.point_cloud_scale,
    )
    valid_cube_candidates, rejected_cube_candidates = filter_duplicate_cube_candidates(
        candidates=cube_candidates,
        x_min_m=args.candidate_x_min_m,
        x_max_m=args.candidate_x_max_m,
        y_min_m=args.candidate_y_min_m,
        y_max_m=args.candidate_y_max_m,
        z_min_m=args.candidate_z_min_m,
        z_max_m=args.candidate_z_max_m,
        min_area_px=args.candidate_min_area_px,
        red_min_area_px=args.red_candidate_min_area_px,
        green_min_area_px=args.green_candidate_min_area_px,
        blue_min_area_px=args.blue_candidate_min_area_px,
        min_extent_m=args.candidate_min_extent_m,
        max_extent_m=args.candidate_max_extent_m,
    )
    _print_rejected_duplicate_cube_candidates(rejected_cube_candidates)
    physical_cube_candidates, physical_cube_clusters = merge_duplicate_cube_candidates(
        candidates=valid_cube_candidates,
        merge_distance_m=args.candidate_merge_distance_m,
        T_cam_robot=T_cam_robot,
    )
    _print_merged_duplicate_cube_candidates(physical_cube_clusters)
    print(
        "Pose-plan refinement cube candidates:"
        f" raw={len(cube_candidates)}"
        f" rejected={len(rejected_cube_candidates)}"
        f" valid={len(valid_cube_candidates)}"
        f" physical={len(physical_cube_candidates)}"
    )
    refined_cube, cube_delta_m = select_nearest_refined_candidate(
        candidates=physical_cube_candidates,
        planned_xy_m=plan.T_robot_cube[:2, 3],
        max_distance_m=args.pose_plan_refine_radius_m,
        label="cube",
    )

    T_robot_cube = refined_cube.T_robot_cube.copy()
    if plan.target_source == "preset_slot":
        print("Preset slot target: skipping target AprilTag refinement.")
        slot = _preset_slot_from_pose_plan(plan)
        T_robot_place = _preset_place_pose_from_slot(
            T_base_cube=T_robot_cube,
            slot=slot,
            preset_use_slot_yaw=plan.preset_use_slot_yaw,
        )
        refinement = PosePlanRefinement(
            plan=plan,
            refined_cube=refined_cube,
            refined_tag=None,
            T_robot_cube=T_robot_cube,
            T_robot_place=T_robot_place,
            cube_delta_m=float(cube_delta_m),
            tag_delta_m=0.0,
        )
        _print_pose_plan_refinement_summary(refinement)
        return refinement

    print("Re-detecting target tag candidates for pose-plan refinement.")
    tag_candidates = _detect_duplicate_target_tag_candidates(
        image=image,
        camera_intrinsic=camera_intrinsic,
        T_cam_robot=T_cam_robot,
        target_tag_id=plan.target_tag_id,
        target_tag_size_m=args.target_tag_size_m,
    )
    _print_duplicate_tag_candidates(
        DuplicateCubeTagGroup(cube_prompt=plan.cube_prompt, tag_id=plan.target_tag_id, count=1),
        tag_candidates,
    )
    refined_tag, tag_delta_m = select_nearest_refined_candidate(
        candidates=tag_candidates,
        planned_xy_m=plan.T_robot_place[:2, 3],
        max_distance_m=args.pose_plan_refine_tag_radius_m,
        label="target tag",
    )

    T_robot_place = _checkpoint8_target_pose_from_tag(T_robot_cube, refined_tag.T_robot_tag)
    refinement = PosePlanRefinement(
        plan=plan,
        refined_cube=refined_cube,
        refined_tag=refined_tag,
        T_robot_cube=T_robot_cube,
        T_robot_place=T_robot_place,
        cube_delta_m=float(cube_delta_m),
        tag_delta_m=float(tag_delta_m),
    )
    _print_pose_plan_refinement_summary(refinement)
    return refinement


def _run_checkpoint8_pose_plan_refinement(args: argparse.Namespace, plan: DuplicatePosePlan) -> PosePlanRefinement:
    from checkpoint0 import get_transform_camera_robot
    from checkpoint6 import CUBE_SIZE
    from utils.zed_camera import ZedCamera as CheckpointZedCamera

    zed: Any | None = None
    try:
        print("Opening live ZED camera for pose-plan local fresh refinement...")
        zed = CheckpointZedCamera()
        image, point_cloud = _checkpoint8_capture_frame(zed)
        if image is None or point_cloud is None:
            raise RuntimeError("Camera data is not ready for pose-plan refinement.")

        camera_intrinsic = zed.camera_intrinsic
        print("Computing camera-to-robot transform with checkpoint0 for pose-plan refinement.")
        T_cam_robot = get_transform_camera_robot(image, camera_intrinsic)
        if T_cam_robot is None:
            raise RuntimeError("Camera-to-robot calibration failed during pose-plan refinement.")

        return _refine_duplicate_pose_plan_from_frame(
            args=args,
            plan=plan,
            image=image,
            point_cloud=point_cloud,
            camera_intrinsic=camera_intrinsic,
            T_cam_robot=T_cam_robot,
            cube_size_m=CUBE_SIZE,
        )
    finally:
        if zed is not None:
            zed.close()
        if (not args.no_gui) and _gui_display_available():
            try:
                cv2.destroyAllWindows()
            except cv2.error as exc:
                print(f"Warning: failed to destroy OpenCV windows cleanly: {exc}")


def _duplicate_pose_plan_from_refinement(refinement: PosePlanRefinement) -> DuplicatePosePlan:
    plan = refinement.plan
    return DuplicatePosePlan(
        execution_index=plan.execution_index,
        cube_prompt=plan.cube_prompt,
        target_tag_id=plan.target_tag_id,
        cube_instance_index=plan.cube_instance_index,
        tag_instance_index=plan.tag_instance_index,
        T_robot_cube=refinement.T_robot_cube,
        T_robot_place=refinement.T_robot_place,
        target_source=plan.target_source,
        slot_id=plan.slot_id,
        preset_slot=dict(plan.preset_slot) if plan.preset_slot is not None else None,
        preset_use_slot_yaw=plan.preset_use_slot_yaw,
    )


def _exit_refinement_subprocess(returncode: int) -> None:
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(int(returncode))


def _run_checkpoint8_refine_pose_plan_json(args: argparse.Namespace) -> None:
    try:
        plan = _load_duplicate_pose_plan_json(args.refine_pose_plan_json)
        print("Using checkpoint8_style pose-plan refinement child.")
        print(f"Input pose-plan JSON: {args.refine_pose_plan_json}")
        print(f"Output refined pose-plan JSON: {args.refined_pose_plan_output_json}")
        print(
            f"Refining plan pair {plan.execution_index}: {plan.cube_prompt}"
            f" cube #{plan.cube_instance_index} -> tag {plan.target_tag_id} #{plan.tag_instance_index}"
        )
        refinement = _run_checkpoint8_pose_plan_refinement(args, plan)
        refined_plan = _duplicate_pose_plan_from_refinement(refinement)
        output_path = _write_duplicate_pose_plan_json(refined_plan, Path(args.refined_pose_plan_output_json))
        print(f"Wrote refined pose-plan JSON: {output_path}")
        print("Pose-plan refinement child complete. No Lite6 connection or motion executed.")
        print("Refined pose-plan JSON written successfully; exiting refinement subprocess with code 0.")
    except BaseException as exc:
        print(f"Pose-plan refinement child failed before writing refined JSON: {exc}", file=sys.stderr)
        _exit_refinement_subprocess(1)
    _exit_refinement_subprocess(0)


def _checkpoint8_tcp_z_meets_minimum(
    tcp_pose_mm_deg: tuple[float, float, float, float, float, float],
    min_after_grasp_z_mm: float,
) -> bool:
    return float(tcp_pose_mm_deg[2]) >= float(min_after_grasp_z_mm)


def _checkpoint8_read_status_and_tcp_pose(
    arm: Any,
    label: str,
) -> tuple[int, int, int, tuple[float, float, float, float, float, float]]:
    state_value, state_raw = _read_arm_value(arm, "get_state")
    err_warn_value, err_warn_raw = _read_arm_value(arm, "get_err_warn_code")
    position_value, position_raw = _read_arm_value(arm, "get_position")

    state = _as_int_status(state_value, "state")
    err, warn = _parse_err_warn(err_warn_value)
    tcp_pose = _parse_pose_response_mm_deg(position_value, "TCP pose")

    print(f"xArm status {label}:")
    print(f"  get_state(): {state_raw!r}")
    print(f"  get_err_warn_code(): {err_warn_raw!r}")
    print(f"  get_position(): {position_raw!r}")
    return state, err, warn, tcp_pose


def _checkpoint8_stop_gripper_if_motion_safe(arm: Any, state: int | None, err: int | None, warn: int | None) -> None:
    if not _checkpoint8_status_ready(state, err, warn):
        print("Skipping gripper stop because xArm status is not ready.")
        return
    try:
        _stop_gripper_if_supported(arm, "pose-plan safety abort")
    except Exception as exc:
        print(f"Warning: failed to stop gripper during pose-plan safety abort: {exc}")


def _raise_pose_plan_recovery_abort(message: str) -> None:
    print(message)
    print("Run --recover_robot, then --home_only before retrying.")
    raise PosePlanSafetyAbort(message)


def _checkpoint8_validate_tcp_ready_height(
    arm: Any,
    label: str,
    min_after_grasp_z_mm: float,
    abnormal_height_message: str,
) -> None:
    state, err, warn, tcp_pose = _checkpoint8_read_status_and_tcp_pose(arm, label)
    if not _checkpoint8_tcp_z_meets_minimum(tcp_pose, min_after_grasp_z_mm):
        print(
            f"TCP z safety check failed {label}:"
            f" z={tcp_pose[2]:.1f}mm"
            f" min={float(min_after_grasp_z_mm):.1f}mm"
        )
        _checkpoint8_stop_gripper_if_motion_safe(arm, state, err, warn)
        _raise_pose_plan_recovery_abort(abnormal_height_message)
    if _checkpoint8_status_ready(state, err, warn):
        return
    if state == 4 or err in (22, 31):
        _raise_pose_plan_recovery_abort(
            "Robot is in STOP/C22/C31 after checkpoint8-style motion."
        )
    _raise_pose_plan_recovery_abort(
        f"Lite6 is not ready {label}: {_checkpoint8_status_bad_message(state, err, warn)}"
    )


def _execute_checkpoint8_pose_plan_grasp_place(
    arm: Any,
    execution_index: int,
    T_robot_cube: np.ndarray,
    T_robot_place: np.ndarray,
    grasp_cube_fn: Any,
    place_cube_fn: Any,
    min_after_grasp_z_mm: float,
) -> None:
    print(f"Calling checkpoint1.grasp_cube for pose-plan pair {execution_index}.")
    grasp_cube_fn(arm, T_robot_cube)
    print(f"Robot state after grasp_cube for pose-plan pair {execution_index}:")
    abnormal_height_message = "Abnormal grasp/retreat height after grasp_cube; not calling place_cube."
    _checkpoint8_validate_tcp_ready_height(
        arm=arm,
        label=f"after checkpoint1.grasp_cube pose-plan pair {execution_index}",
        min_after_grasp_z_mm=min_after_grasp_z_mm,
        abnormal_height_message=abnormal_height_message,
    )

    print(f"Pre-place safety check for pose-plan pair {execution_index}:")
    _checkpoint8_validate_tcp_ready_height(
        arm=arm,
        label=f"before checkpoint1.place_cube pose-plan pair {execution_index}",
        min_after_grasp_z_mm=min_after_grasp_z_mm,
        abnormal_height_message=abnormal_height_message,
    )

    print(f"Calling checkpoint1.place_cube for pose-plan pair {execution_index}.")
    place_cube_fn(arm, T_robot_place)
    print(f"Robot state after place_cube for pose-plan pair {execution_index}:")
    _checkpoint8_require_ready(arm, f"after checkpoint1.place_cube pose-plan pair {execution_index}")


def _run_checkpoint8_pose_plan_child(args: argparse.Namespace, robot_ip: str) -> None:
    if args.skip_home or args.no_final_home:
        raise SystemExit(
            "checkpoint8_style pose-plan execution requires start and final home. "
            "Remove --skip_home/--no_final_home."
        )

    plan = _load_duplicate_pose_plan_json(args.execute_pose_plan_json)
    print("Using checkpoint8_style pose-plan JSON child execution.")
    print(f"Pose-plan JSON: {args.execute_pose_plan_json}")
    if plan.target_source == "preset_slot":
        print(
            f"Plan pair {plan.execution_index}: {plan.cube_prompt}"
            f" cube #{plan.cube_instance_index} -> preset slot {plan.slot_id}"
        )
    else:
        print(
            f"Plan pair {plan.execution_index}: {plan.cube_prompt}"
            f" cube #{plan.cube_instance_index} -> tag {plan.target_tag_id} #{plan.tag_instance_index}"
        )
    _checkpoint8_print_transform("Pose-plan source cube pose", plan.T_robot_cube)
    _checkpoint8_print_transform("Pose-plan target place pose", plan.T_robot_place)

    T_robot_cube = plan.T_robot_cube
    T_robot_place = plan.T_robot_place
    if args.no_pose_plan_refine:
        print("Robot-only pose-plan child: --no_pose_plan_refine set; skipping ZED/perception refinement.")
    if args.execute_pose_plan_refine_only or args.pose_plan_refine_before_execute:
        refinement = _run_checkpoint8_pose_plan_refinement(args, plan)
        T_robot_cube = refinement.T_robot_cube
        T_robot_place = refinement.T_robot_place
        _checkpoint8_print_transform("Refined pose-plan source cube pose", T_robot_cube)
        _checkpoint8_print_transform("Refined pose-plan target place pose", T_robot_place)

    if args.execute_pose_plan_refine_only:
        print("Pose-plan refine-only diagnostic complete. No Lite6 connection or motion executed.")
        return

    if args.dry_run:
        print("Dry run selected for pose-plan child. No Lite6 connection or motion executed.")
        return

    from checkpoint1 import GRIPPER_LENGTH, grasp_cube, place_cube

    arm: Any | None = None
    initial_home_ok = False
    task_started = False
    task_success = False
    pose_plan_safety_abort = False
    try:
        arm = _checkpoint8_connect_arm(robot_ip)
        _checkpoint8_initialize_and_home(arm, GRIPPER_LENGTH)
        _checkpoint8_require_home_ready(arm, "after pose-plan initial move_gohome")
        initial_home_ok = True

        task_started = True
        try:
            _execute_checkpoint8_pose_plan_grasp_place(
                arm=arm,
                execution_index=plan.execution_index,
                T_robot_cube=T_robot_cube,
                T_robot_place=T_robot_place,
                grasp_cube_fn=grasp_cube,
                place_cube_fn=place_cube,
                min_after_grasp_z_mm=args.min_after_grasp_z_mm,
            )
        except PosePlanSafetyAbort:
            pose_plan_safety_abort = True
            raise SystemExit(1)

        _checkpoint8_move_home_required(arm, f"after pose-plan pair {plan.execution_index}")
        task_success = True
        print(f"Pose-plan pair {plan.execution_index} complete and robot returned home.")
    finally:
        if arm is not None:
            if not pose_plan_safety_abort:
                try:
                    state, err, warn = _checkpoint8_print_status(arm, "before pose-plan cleanup gripper stop")
                    _checkpoint8_stop_gripper_if_motion_safe(arm, state, err, warn)
                except Exception as exc:
                    print(f"Warning: failed to stop gripper cleanly: {exc}")
            if initial_home_ok and task_started and not task_success:
                print(
                    "Pose-plan task did not complete successfully; not sending extra home motion. "
                    "Run --recover_robot, then --home_only before retrying."
                )
            try:
                arm.disconnect()
            except Exception as exc:
                print(f"Warning: failed to disconnect cleanly: {exc}")


def _run_duplicate_pose_plan_child_subprocesses(
    args: argparse.Namespace,
    assignments: list[DuplicateAssignedPair],
    plan_paths: dict[int, Path],
    subprocess_run_fn: Any | None = None,
) -> tuple[list[DuplicateAssignedPair], list[DuplicateAssignedPair]]:
    if subprocess_run_fn is None:
        subprocess_run_fn = subprocess.run

    succeeded: list[DuplicateAssignedPair] = []
    failed: list[DuplicateAssignedPair] = []

    for assignment in assignments:
        raw_plan_path = plan_paths[assignment.execution_index]
        refined_plan_path = _refined_duplicate_pose_plan_path(raw_plan_path)
        refinement_command = _build_duplicate_pose_plan_refinement_child_command(
            args=args,
            input_plan_path=raw_plan_path,
            output_plan_path=refined_plan_path,
        )
        robot_command = _build_duplicate_pose_plan_robot_child_command(args, refined_plan_path)
        target_label = (
            f"slot {assignment.slot.slot_id}"
            if hasattr(assignment, "slot")
            else f"tag {assignment.tag_id} #{assignment.tag.instance_index}"
        )

        print(
            f"\nStarting pose-plan pair {assignment.execution_index}/{len(assignments)}:"
            f" {assignment.cube_prompt} cube #{assignment.cube.instance_index}"
            f" -> {target_label}"
        )
        print(f"Raw pose-plan path: {raw_plan_path}")
        print(f"Refined pose-plan path: {refined_plan_path}")
        print(f"Refinement child command: {_format_command_for_log(refinement_command)}")

        refinement_completed = subprocess_run_fn(refinement_command, cwd=str(REPO_ROOT))
        refinement_returncode = int(refinement_completed.returncode)
        print(f"Refinement return code: {refinement_returncode}")

        if refinement_returncode != 0:
            failed.append(assignment)
            print(
                f"Pose-plan refinement failed for pair {assignment.execution_index}/{len(assignments)}; "
                "aborting before robot motion."
            )
            if _child_returncode_looks_like_native_crash(refinement_returncode):
                print("Refinement child crashed, likely native perception stack cleanup.")
            break

        print(f"Robot child command: {_format_command_for_log(robot_command)}")
        robot_completed = subprocess_run_fn(robot_command, cwd=str(REPO_ROOT))
        robot_returncode = int(robot_completed.returncode)
        print(f"Robot child return code: {robot_returncode}")

        if robot_returncode == 0:
            succeeded.append(assignment)
            print(f"Pose-plan pair {assignment.execution_index}/{len(assignments)} succeeded.")
            continue

        failed.append(assignment)
        print(f"Pose-plan pair {assignment.execution_index}/{len(assignments)} failed during robot execution.")
        if _child_returncode_looks_like_native_crash(robot_returncode):
            print("Robot child crashed.")
        if not args.continue_on_pair_failure:
            print("Aborting duplicate-aware run after first failed robot child.")
            break
        print("Continuing because --continue_on_pair_failure was provided.")

    return succeeded, failed


def _confirm_preset_layout_parent(args: argparse.Namespace) -> bool:
    if args.auto_confirm:
        print("Auto-confirm enabled: executing all preset layout placements without interactive confirmation.")
        return True
    try:
        response = input("Type 'k' then Enter to execute all preset layout placements, or anything else to cancel: ")
    except EOFError:
        print("No terminal input is available. Cancelling before starting child processes.")
        return False
    return response.strip().lower() == "k"


def _run_checkpoint8_style_preset_layout_place(args: argparse.Namespace, robot_ip: str) -> None:
    del robot_ip
    from checkpoint0 import get_transform_camera_robot
    from checkpoint6 import CUBE_SIZE
    from utils.vis_utils import draw_pose_axes as checkpoint_draw_pose_axes
    from utils.zed_camera import ZedCamera as CheckpointZedCamera

    layout = getattr(args, "_preset_layout", None)
    if layout is None:
        layout = load_preset_place_layout_json(args.preset_place_layout_json)
        args._preset_layout = layout
    groups = _preset_groups_from_args(args)
    total_requested = sum(group.count for group in groups)
    print("Using checkpoint8_style preset layout place mode.")
    print("Parent process will detect duplicate cube instances and execute concrete preset-slot pose-plan children.")
    print(f"Requested preset placements: {total_requested}")
    print(f"Preset layout file path: {args.preset_place_layout_json}")
    print(f"Preset layout name: {layout.name}")
    print(f"Preset use slot yaw: {bool(args.preset_use_slot_yaw)}")

    zed: Any | None = None
    assignments: list[PresetAssignedPair] = []
    group_reports: list[dict[str, Any]] = []
    preview_path: Path | None = None

    try:
        print("Opening live ZED camera with checkpoint8-style pipeline for preset layout preflight...")
        zed = CheckpointZedCamera()
        assignments, group_reports, preview_path = _preset_layout_preflight(
            args=args,
            zed=zed,
            get_transform_camera_robot=get_transform_camera_robot,
            draw_pose_axes_fn=checkpoint_draw_pose_axes,
            cube_size_m=CUBE_SIZE,
        )
    finally:
        if zed is not None:
            zed.close()
        if (not args.no_gui) and _gui_display_available():
            try:
                cv2.destroyAllWindows()
            except cv2.error as exc:
                print(f"Warning: failed to destroy OpenCV windows cleanly: {exc}")

    if args.dry_run:
        print("Preset layout dry run selected. Assignment completed; no robot motion or child subprocesses executed.")
        return

    if not _confirm_preset_layout_parent(args):
        print("Operator cancelled. No child subprocesses started.")
        return

    plan_paths = _write_preset_pose_plans(assignments)
    _save_preset_assignment_report(
        layout=layout,
        groups=group_reports,
        assignments=assignments,
        preview_path=preview_path,
        plan_paths=plan_paths,
        execution_confirmed=True,
    )

    print(
        "Skipping parent robot home precheck; each robot-only preset pose-plan child performs "
        "the checkpoint8 home-start and final-home sequence."
    )
    succeeded, failed = _run_duplicate_pose_plan_child_subprocesses(
        args=args,
        assignments=assignments,
        plan_paths=plan_paths,
    )

    print("\nPreset layout subprocess summary:")
    print(f"Total assigned pairs: {len(assignments)}")
    print(f"Succeeded pairs: {len(succeeded)}")
    print(f"Failed pairs: {len(failed)}")
    if failed:
        failed_text = ", ".join(
            f"{item.cube_prompt} cube #{item.cube.instance_index} -> slot {item.slot.slot_id}"
            for item in failed
        )
        print(f"Failed preset assignments: {failed_text}")
        raise SystemExit(1)
    print("Failed preset assignments: none")
    print("Preset layout task complete: every pose-plan child completed successfully.")


def _run_checkpoint8_style_duplicate_aware_multi_place(args: argparse.Namespace, robot_ip: str) -> None:
    del robot_ip
    from checkpoint0 import get_transform_camera_robot
    from checkpoint6 import CUBE_SIZE
    from utils.vis_utils import draw_pose_axes as checkpoint_draw_pose_axes
    from utils.zed_camera import ZedCamera as CheckpointZedCamera

    groups = _duplicate_groups_from_args(args)
    total_requested = sum(group.count for group in groups)
    print("Using checkpoint8_style duplicate-aware multi-place mode.")
    print("Parent process will detect duplicate instances once and execute concrete pose-plan children.")
    print(f"Requested duplicate-aware placements: {total_requested}")
    print(f"Target tag size assumption for duplicate target tags: {float(args.target_tag_size_m):.4f} m.")
    if args.multi_subprocess:
        print("--multi_subprocess accepted; duplicate-aware real execution always uses pose-plan child subprocesses.")

    zed: Any | None = None
    assignments: list[DuplicateAssignedPair] = []
    group_reports: list[dict[str, Any]] = []
    preview_path: Path | None = None

    try:
        print("Opening live ZED camera with checkpoint8-style pipeline for duplicate-aware preflight...")
        zed = CheckpointZedCamera()
        assignments, group_reports, preview_path = _duplicate_aware_preflight(
            args=args,
            zed=zed,
            get_transform_camera_robot=get_transform_camera_robot,
            draw_pose_axes_fn=checkpoint_draw_pose_axes,
            cube_size_m=CUBE_SIZE,
        )
    finally:
        if zed is not None:
            zed.close()
        if (not args.no_gui) and _gui_display_available():
            try:
                cv2.destroyAllWindows()
            except cv2.error as exc:
                print(f"Warning: failed to destroy OpenCV windows cleanly: {exc}")

    if args.dry_run or args.debug_duplicate_candidates_only:
        if args.debug_duplicate_candidates_only:
            print(
                "Duplicate candidate debug mode selected. "
                "Perception, filtering, merging, assignment, preview, and report completed; no robot motion or child subprocesses executed."
            )
        else:
            print("Duplicate-aware dry run selected. Assignment completed; no robot motion or child subprocesses executed.")
        return

    if not _confirm_duplicate_aware_parent(args):
        print("Operator cancelled. No child subprocesses started.")
        return

    plan_paths = _write_duplicate_pose_plans(assignments)
    if args.save_assignment_report:
        _save_duplicate_assignment_report(
            groups=group_reports,
            assignments=assignments,
            preview_path=preview_path,
            plan_paths=plan_paths,
            execution_confirmed=True,
        )

    print(
        "Skipping parent robot home precheck; each robot-only pose-plan child performs "
        "the checkpoint8 home-start and final-home sequence."
    )
    succeeded, failed = _run_duplicate_pose_plan_child_subprocesses(
        args=args,
        assignments=assignments,
        plan_paths=plan_paths,
    )

    print("\nDuplicate-aware subprocess summary:")
    print(f"Total assigned pairs: {len(assignments)}")
    print(f"Succeeded pairs: {len(succeeded)}")
    print(f"Failed pairs: {len(failed)}")
    if failed:
        failed_text = ", ".join(
            f"{item.cube_prompt} cube #{item.cube.instance_index} -> tag {item.tag_id} #{item.tag.instance_index}"
            for item in failed
        )
        print(f"Failed assignments: {failed_text}")
        raise SystemExit(1)
    print("Failed assignments: none")
    print("Duplicate-aware multi-place task complete: every pose-plan child completed successfully.")


def _run_checkpoint8_style_multi_place_to_tags(args: argparse.Namespace, robot_ip: str) -> None:
    if args.skip_home or args.no_final_home:
        raise SystemExit(
            "checkpoint8_style multi_place_to_tags requires start, per-pair, and final home. "
            "Remove --skip_home/--no_final_home."
        )

    from checkpoint0 import get_transform_camera_robot
    from checkpoint1 import GRIPPER_LENGTH, grasp_cube, place_cube
    from checkpoint6 import CUBE_SIZE, get_transform_cube
    from utils.vis_utils import draw_pose_axes as checkpoint_draw_pose_axes
    from utils.zed_camera import ZedCamera as CheckpointZedCamera

    cube_tag_pairs = _cube_tag_pairs_from_args(args)
    print("Using checkpoint8_style multi_place_to_tags execution backend.")
    print(f"Target tag size assumption for non-calibration placement tags: {float(args.target_tag_size_m):.4f} m.")
    if args.multi_subprocess and args.dry_run:
        print("Dry run requested with --multi_subprocess; no child subprocesses will be started.")
        _print_multi_subprocess_pair_map(cube_tag_pairs)
        _print_multi_subprocess_child_commands(args, cube_tag_pairs)
    elif not args.dry_run:
        print(
            "Warning: in-process multi mode may trigger native perception crashes after repeated "
            "ZED/AprilTag calls. For real robot multi-cube runs, prefer --multi_subprocess."
        )

    zed: Any | None = None
    arm: Any | None = None
    initial_home_ok = False
    task_started = False
    task_success = False

    try:
        print("Opening live ZED camera with checkpoint8-style pipeline...")
        zed = CheckpointZedCamera()

        if args.dry_run:
            print("Running multi-place dry run: one frame, all requested cubes/tags, no Lite6 connection.")
            try:
                _confirmed, _entries = _checkpoint8_multi_preflight(
                    zed=zed,
                    cube_tag_pairs=cube_tag_pairs,
                    target_tag_size_m=args.target_tag_size_m,
                    get_transform_camera_robot=get_transform_camera_robot,
                    get_transform_cube=get_transform_cube,
                    draw_pose_axes_fn=checkpoint_draw_pose_axes,
                    cube_size_m=CUBE_SIZE,
                    no_gui=args.no_gui,
                    preview_path=args.preview_path,
                    dry_run=True,
                    auto_confirm=args.auto_confirm,
                )
            except RuntimeError as exc:
                print(f"Multi-place dry run failed: {exc}")
            return

        arm = _checkpoint8_connect_arm(robot_ip)
        _checkpoint8_initialize_and_home(arm, GRIPPER_LENGTH)
        _checkpoint8_require_home_ready(arm, "after checkpoint8 multi initial move_gohome")
        initial_home_ok = True

        print("Running multi-place preflight from home before robot grasp/place motion.")
        try:
            confirmed, _entries = _checkpoint8_multi_preflight(
                zed=zed,
                cube_tag_pairs=cube_tag_pairs,
                target_tag_size_m=args.target_tag_size_m,
                get_transform_camera_robot=get_transform_camera_robot,
                get_transform_cube=get_transform_cube,
                draw_pose_axes_fn=checkpoint_draw_pose_axes,
                cube_size_m=CUBE_SIZE,
                no_gui=args.no_gui,
                preview_path=args.preview_path,
                dry_run=False,
                auto_confirm=args.auto_confirm,
            )
        except RuntimeError as exc:
            print(f"Multi-place preflight failed before robot grasp/place motion: {exc}")
            return

        if not confirmed:
            print("Operator cancelled. No grasp/place motion executed.")
            return

        camera_intrinsic = zed.camera_intrinsic
        for index, pair in enumerate(cube_tag_pairs, start=1):
            _checkpoint8_require_home_ready(arm, f"before pair {index} fresh perception")
            print(f"\nExecuting multi-place pair {index}/{len(cube_tag_pairs)}: {pair.cube_prompt!r} -> tag {pair.target_tag_id}")

            image, point_cloud = _checkpoint8_capture_frame(zed)
            if image is None or point_cloud is None:
                raise RuntimeError(f"Camera data is not ready before pair {index}.")

            T_cam_robot = get_transform_camera_robot(image, camera_intrinsic)
            if T_cam_robot is None:
                raise RuntimeError(f"Camera-to-robot calibration failed before pair {index}.")

            entries = _checkpoint8_build_multi_plan_entries(
                image=image,
                point_cloud=point_cloud,
                camera_intrinsic=camera_intrinsic,
                T_cam_robot=T_cam_robot,
                cube_tag_pairs=[pair],
                target_tag_size_m=args.target_tag_size_m,
                get_transform_cube=get_transform_cube,
            )
            entry = entries[0]
            _checkpoint8_print_multi_plan_table(entries)
            _checkpoint8_print_transform(f"Pair {index} source cube pose", entry.T_robot_cube)
            _checkpoint8_print_transform(f"Pair {index} target place pose", entry.T_robot_place)

            task_started = True
            print(f"Calling checkpoint1.grasp_cube for pair {index}.")
            grasp_cube(arm, entry.T_robot_cube)
            print(f"Robot state after grasp_cube for pair {index}:")
            _checkpoint8_require_ready(arm, f"after checkpoint1.grasp_cube pair {index}")

            print(f"Calling checkpoint1.place_cube for pair {index}.")
            place_cube(arm, entry.T_robot_place)
            print(f"Robot state after place_cube for pair {index}:")
            _checkpoint8_require_ready(arm, f"after checkpoint1.place_cube pair {index}")

            _checkpoint8_move_home_required(arm, f"after pair {index}")
            print(f"Pair {index}/{len(cube_tag_pairs)} complete and robot returned home.")

        task_success = True
        print("checkpoint8_style multi-place task complete: every cube-tag pair completed successfully and the robot is home.")

    finally:
        if arm is not None:
            try:
                if hasattr(arm, "stop_lite6_gripper"):
                    arm.stop_lite6_gripper()
                    time.sleep(0.2)
            except Exception as exc:
                print(f"Warning: failed to stop gripper cleanly: {exc}")
            if initial_home_ok and task_started and not task_success:
                print(
                    "Multi-place task did not complete successfully; not sending extra home motion. "
                    "Run --recover_robot, then --home_only before retrying."
                )
            try:
                arm.disconnect()
            except Exception as exc:
                print(f"Warning: failed to disconnect cleanly: {exc}")
        if zed is not None:
            zed.close()
        if (not args.no_gui) and _gui_display_available():
            try:
                cv2.destroyAllWindows()
            except cv2.error as exc:
                print(f"Warning: failed to destroy OpenCV windows cleanly: {exc}")


def _run_checkpoint8_style_backend(args: argparse.Namespace, robot_ip: str) -> None:
    if args.skip_home or args.no_final_home:
        raise SystemExit(
            "checkpoint8_style requires start and final home, matching the original checkpoint. "
            "Remove --skip_home/--no_final_home or select --execution_backend legacy for diagnostics."
        )

    from checkpoint0 import get_transform_camera_robot
    from checkpoint1 import GRIPPER_LENGTH, grasp_cube, place_cube
    from checkpoint6 import CUBE_SIZE, get_transform_cube
    from utils.vis_utils import draw_pose_axes as checkpoint_draw_pose_axes
    from utils.zed_camera import ZedCamera as CheckpointZedCamera

    print("Using checkpoint8_style execution backend.")
    print("Real robot motion will start from home and use checkpoint1.grasp_cube/place_cube.")
    if args.move_home_before_task:
        print("--move_home_before_task is ignored because checkpoint8_style always homes before execution.")

    zed: Any | None = None
    arm: Any | None = None
    initial_home_ok = False
    task_started = False
    task_success = False

    try:
        print("Opening live ZED camera with checkpoint8-style pipeline...")
        zed = CheckpointZedCamera()
        camera_intrinsic = zed.camera_intrinsic

        if not args.dry_run:
            arm = _checkpoint8_connect_arm(robot_ip)
            _checkpoint8_initialize_and_home(arm, GRIPPER_LENGTH)
            initial_home_ok = True

        image = zed.image
        point_cloud = zed.point_cloud
        if image is None or point_cloud is None:
            print("Camera data is not ready.")
            return

        T_cam_robot = get_transform_camera_robot(image, camera_intrinsic)
        if T_cam_robot is None:
            print("Camera-to-robot calibration failed.")
            return

        cube_result = get_transform_cube(
            [image, point_cloud],
            camera_intrinsic,
            T_cam_robot,
            cube_prompt=args.cube_prompt,
        )
        if cube_result is None:
            print(f"No cube matched prompt: {args.cube_prompt}")
            return
        T_robot_cube, T_cam_cube = cube_result
        if T_robot_cube is None or T_cam_cube is None:
            print(f"No cube matched prompt: {args.cube_prompt}")
            return

        T_robot_place = T_robot_cube
        T_cam_tag: np.ndarray | None = None
        T_cam_place: np.ndarray | None = None
        if args.place_to_tag:
            target_tag_id = args.target_tag_id if args.target_tag_id is not None else args.tag_id
            T_robot_tag, T_cam_tag = _estimate_detected_target_tag_pose(
                image=image,
                camera_intrinsic=camera_intrinsic,
                T_cam_base=T_cam_robot,
                target_tag_id=int(target_tag_id),
                target_tag_size_m=args.target_tag_size_m,
            )
            if T_robot_tag is None or T_cam_tag is None:
                return
            T_robot_place = _checkpoint8_target_pose_from_tag(T_robot_cube, T_robot_tag)
            T_cam_place = T_cam_robot @ T_robot_place

        print("\nCheckpoint8-style detected/planned poses:")
        _checkpoint8_print_transform("Source cube pose", T_robot_cube)
        if args.place_to_tag:
            _checkpoint8_print_transform("Target place pose", T_robot_place)
        else:
            print("No --place_to_tag provided; the cube will be placed back down at its source pose.")

        display = _bgr_from_image(image)
        checkpoint_draw_pose_axes(display, camera_intrinsic, T_cam_cube, size=float(CUBE_SIZE) * 1.5)
        _draw_label(display, camera_intrinsic, T_cam_cube, "source cube", (0, 255, 255))
        if args.place_to_tag and T_cam_tag is not None and T_cam_place is not None:
            checkpoint_draw_pose_axes(display, camera_intrinsic, T_cam_tag, size=float(args.target_tag_size_m) * 0.75)
            checkpoint_draw_pose_axes(display, camera_intrinsic, T_cam_place, size=float(CUBE_SIZE) * 1.5)
            _draw_label(display, camera_intrinsic, T_cam_tag, "target tag", (255, 255, 0))
            _draw_label(display, camera_intrinsic, T_cam_place, "target place", (255, 0, 255))

        confirmed = _checkpoint8_save_or_confirm(
            display=display,
            no_gui=args.no_gui,
            preview_path=args.preview_path,
            dry_run=args.dry_run,
            auto_confirm=args.auto_confirm,
        )
        if args.dry_run:
            return
        if not confirmed:
            print("Operator cancelled. No grasp/place motion executed.")
            return

        if arm is None:
            raise RuntimeError("Internal error: checkpoint8_style real execution has no Lite6 connection.")

        task_started = True
        print("Calling checkpoint1.grasp_cube.")
        grasp_cube(arm, T_robot_cube)
        print("Robot state after grasp_cube:")
        _checkpoint8_require_ready(arm, "after checkpoint1.grasp_cube")

        print("Calling checkpoint1.place_cube.")
        place_cube(arm, T_robot_place)
        print("Robot state after place_cube:")
        _checkpoint8_require_ready(arm, "after checkpoint1.place_cube")

        task_success = True
        print("checkpoint8_style mini task complete: cube grasped and placed successfully.")

    finally:
        if arm is not None:
            try:
                if hasattr(arm, "stop_lite6_gripper"):
                    arm.stop_lite6_gripper()
                    time.sleep(0.2)
            except Exception as exc:
                print(f"Warning: failed to stop gripper cleanly: {exc}")
            if initial_home_ok and task_success:
                _checkpoint8_final_home_if_ready(arm)
            elif initial_home_ok and task_started and not task_success:
                print(
                    "Task did not complete successfully; not sending final home motion. "
                    "Run --recover_robot, then --home_only before retrying."
                )
            try:
                arm.disconnect()
            except Exception as exc:
                print(f"Warning: failed to disconnect cleanly: {exc}")
        if zed is not None:
            zed.close()
        if (not args.no_gui) and _gui_display_available():
            try:
                cv2.destroyAllWindows()
            except cv2.error as exc:
                print(f"Warning: failed to destroy OpenCV windows cleanly: {exc}")


def _safe_disconnect(arm: Any | None) -> None:
    if arm is None:
        return
    try:
        if hasattr(arm, "stop_lite6_gripper"):
            arm.stop_lite6_gripper()
            time.sleep(0.2)
    except Exception as exc:
        print(f"Warning: failed to stop gripper cleanly: {exc}")
    try:
        arm.disconnect()
    except Exception as exc:
        print(f"Warning: failed to disconnect cleanly: {exc}")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    robot_ip = _load_robot_ip(args.robot_config, args.robot_ip)

    if args.recover_robot:
        _run_checkpoint8_recover_robot(robot_ip)
        return

    if args.home_only:
        _run_checkpoint8_home_only(robot_ip)
        return

    if args.execution_backend == "checkpoint8_style":
        if args.refine_pose_plan_json:
            _run_checkpoint8_refine_pose_plan_json(args)
            return
        if args.execute_pose_plan_json:
            _run_checkpoint8_pose_plan_child(args, robot_ip)
            return
        if args.preset_layout_place:
            _run_checkpoint8_style_preset_layout_place(args, robot_ip)
            return
        if args.duplicate_aware_multi_place:
            _run_checkpoint8_style_duplicate_aware_multi_place(args, robot_ip)
            return
        if args.multi_place_to_tags:
            if args.multi_subprocess and not args.dry_run:
                _run_checkpoint8_style_multi_subprocess(args, robot_ip)
                return
            _run_checkpoint8_style_multi_place_to_tags(args, robot_ip)
            return
        _run_checkpoint8_style_backend(args, robot_ip)
        return

    print("Warning: legacy backend uses custom mini_task motion and is not recommended for real robot execution.")

    zed: ZedCamera | None = None
    arm: Any | None = None

    try:
        print("Opening live ZED camera...")
        zed = ZedCamera(resolution=args.zed_resolution, fps=args.zed_fps)
        camera_intrinsic = zed.camera_intrinsic

        image = zed.image
        point_cloud = zed.point_cloud
        if image is None or point_cloud is None:
            print("Camera data is not ready.")
            return

        T_cam_base, calibration_tags = estimate_T_cam_base(
            image,
            camera_intrinsic,
            calibration_tag_size_m=args.calibration_tag_size_m,
        )
        if T_cam_base is None:
            return

        T_base_cube, T_cam_cube, cube_mask = estimate_cube_pose(
            image=image,
            point_cloud=point_cloud,
            T_cam_base=T_cam_base,
            cube_color=args.cube_color,
            cube_size_m=args.cube_size_m,
            table_z_m=args.table_z_m,
            point_cloud_scale=args.point_cloud_scale,
        )
        if T_base_cube is None or T_cam_cube is None:
            return

        T_base_tag, T_cam_tag = estimate_target_tag_pose(
            image=image,
            camera_intrinsic=camera_intrinsic,
            T_cam_base=T_cam_base,
            target_tag_id=args.tag_id,
            target_tag_size_m=args.target_tag_size_m,
            calibration_tags=calibration_tags,
        )
        if T_base_tag is None or T_cam_tag is None:
            return

        T_base_tag_for_place, target_tag_z_replaced = _constrain_target_tag_pose_for_placement(
            T_base_tag=T_base_tag,
            target_tag_id=args.tag_id,
            table_z_m=args.table_z_m,
            target_tag_size_m=args.target_tag_size_m,
        )
        T_cam_tag_for_preview = T_cam_base @ T_base_tag_for_place if target_tag_z_replaced else T_cam_tag

        T_base_place = construct_place_pose(
            T_base_tag=T_base_tag_for_place,
            cube_size_m=args.cube_size_m,
            table_z_m=args.table_z_m,
            place_x_offset_m=args.place_x_offset_m,
            place_y_offset_m=args.place_y_offset_m,
        )
        T_cam_place = T_cam_base @ T_base_place

        print("\nDetected/planned poses:")
        _print_pose("Cube grasp pose", T_base_cube)
        if target_tag_z_replaced:
            _print_pose("Target AprilTag raw pose", T_base_tag)
            _print_pose("Target AprilTag pose used for placement", T_base_tag_for_place)
        else:
            _print_pose("Target AprilTag pose", T_base_tag_for_place)
        _print_pose("Target place pose", T_base_place)

        task_gate_min_mm, task_gate_max_mm = _task_gate_bounds_from_args(args)
        conservative_execution_ok = _mini_task_conservative_execution_ok(
            T_base_cube,
            T_base_place,
            task_gate_min_mm,
            task_gate_max_mm,
        )
        if args.dry_run:
            status = "PASS" if conservative_execution_ok else "FAIL"
            print(f"Dry run conservative execution gate: {status}.")

        stage_min_mm, stage_max_mm, stage_z_mm = _stage_gate_from_args(args)
        _print_first_move_configuration(
            args=args,
            stage_minimum_mm=stage_min_mm,
            stage_maximum_mm=stage_max_mm,
            stage_z_mm=stage_z_mm,
            dry_run=args.dry_run,
        )

        if (not args.dry_run) and not conservative_execution_ok:
            print("Aborting before Lite6 connection or robot motion.")
            return

        workspace_ok = _workspace_pose_ok("Cube grasp pose", T_base_cube, args.approach_height_m)
        workspace_ok = _target_tag_pose_ok("Target AprilTag pose", T_base_tag_for_place) and workspace_ok
        workspace_ok = _workspace_pose_ok("Target place pose", T_base_place, args.approach_height_m) and workspace_ok
        if not workspace_ok and not args.allow_outside_workspace:
            print("Workspace guardrail rejected the plan. Re-run with --allow_outside_workspace only if the lab setup is known safe.")
            return

        confirmed = show_confirmation(
            image=image,
            camera_intrinsic=camera_intrinsic,
            T_cam_cube=T_cam_cube,
            T_cam_tag=T_cam_tag_for_preview,
            T_cam_place=T_cam_place,
            mask=cube_mask,
            cube_size_m=args.cube_size_m,
            target_tag_size_m=args.target_tag_size_m,
            no_gui=args.no_gui,
            preview_path=args.preview_path,
            auto_confirm=args.auto_confirm,
        )
        if not confirmed:
            print("Operator cancelled. No robot motion executed.")
            return

        if args.dry_run:
            print("Dry run selected. Perception confirmed; no robot motion executed.")
            return

        move_home_before_task = bool(args.move_home_before_task and not args.skip_home)
        if args.move_home_before_task and args.skip_home:
            print("Both --move_home_before_task and --skip_home were provided; skipping xArm home motion.")

        arm = connect_lite6(robot_ip, move_home_before_task=move_home_before_task)
        execute_pick_place(
            arm=arm,
            T_base_cube=T_base_cube,
            T_base_place=T_base_place,
            approach_height_m=args.approach_height_m,
            retreat_height_m=args.retreat_height_m,
            speed_mm_s=args.speed_mm_s,
            gripper_settle_s=args.gripper_settle_s,
            motion_profile=args.motion_profile,
            first_move_strategy=args.first_move_strategy,
            stage_minimum_mm=stage_min_mm,
            stage_maximum_mm=stage_max_mm,
            stage_z_mm=stage_z_mm,
            forward_axis=args.forward_axis,
            forward_sign=args.forward_sign,
            forward_step_mm=args.forward_step_mm,
            forward_steps=args.forward_steps,
            forward_stage_speed_mm_s=args.forward_stage_speed_mm_s,
        )
        print("mini_task complete: cube placed and arm retreated after place.")

    finally:
        _safe_disconnect(arm)
        if zed is not None:
            zed.close()
        if (not args.no_gui) and _gui_display_available():
            try:
                cv2.destroyAllWindows()
            except cv2.error as exc:
                print(f"Warning: failed to destroy OpenCV windows cleanly: {exc}")


if __name__ == "__main__":
    main()
