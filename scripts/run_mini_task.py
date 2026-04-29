from __future__ import annotations

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root_on_path

REPO_ROOT = ensure_repo_root_on_path()

import argparse
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


def _cube_tag_pairs_from_args(args: argparse.Namespace) -> list[CubeTagPair]:
    raw_pairs = getattr(args, "_cube_tag_pairs", None)
    if raw_pairs is None:
        raw_pairs = parse_cube_tag_map(args.cube_tag_map)
        args._cube_tag_pairs = raw_pairs
    return [CubeTagPair(cube_prompt=prompt, target_tag_id=tag_id) for prompt, tag_id in raw_pairs]


def _validate_args(args: argparse.Namespace) -> None:
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
