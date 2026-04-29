from __future__ import annotations

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root_on_path

REPO_ROOT = ensure_repo_root_on_path()

import argparse
import os
import time
from numbers import Integral
from pathlib import Path
from typing import Any

import cv2
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import GRIPPER_LENGTH, grasp_cube, place_cube
from checkpoint6 import CUBE_SIZE, get_transform_cube

try:
    from checkpoint1 import _pose_to_mm_deg as checkpoint_pose_to_mm_deg
except ImportError:  # pragma: no cover
    from scipy.spatial.transform import Rotation

    def checkpoint_pose_to_mm_deg(pose):
        px, py, pz = pose[:3, 3]
        roll, pitch, yaw = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
        return px * 1000.0, py * 1000.0, pz * 1000.0, roll, pitch, yaw

try:
    from checkpoint6 import robot_ip as CK6_ROBOT_IP
except ImportError:  # pragma: no cover
    CK6_ROBOT_IP = "192.168.1.158"


DEFAULT_CUBE_PROMPT = "blue cube"
DEFAULT_PREVIEW_PATH = "logs/checkpoint8_style_preview.png"


class CubePoseDetector:
    """
    Pure-vision cube pose detector for prompted colors.

    This mirrors checkpoint8's wrapper around checkpoint0 and checkpoint6.
    """

    def __init__(self, camera_intrinsic):
        self.camera_intrinsic = camera_intrinsic

    def get_transforms(self, observation, cube_prompt):
        image, point_cloud = observation
        if image is None or point_cloud is None:
            return None

        t_cam_robot = get_transform_camera_robot(image, self.camera_intrinsic)
        if t_cam_robot is None:
            return None

        t_robot_cube, t_cam_cube = get_transform_cube(
            [image, point_cloud],
            self.camera_intrinsic,
            t_cam_robot,
            cube_prompt=cube_prompt,
        )
        if t_robot_cube is None or t_cam_cube is None:
            return None

        return t_robot_cube, t_cam_cube


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal checkpoint8-style live ZED + Lite6 cube grasp/place script."
    )
    parser.add_argument("--cube_prompt", default=DEFAULT_CUBE_PROMPT)
    parser.add_argument("--robot_ip", default=CK6_ROBOT_IP)
    parser.add_argument("--no_gui", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--strict_checkpoint8",
        action="store_true",
        help="Run in the original checkpoint8 order: initial home is required and final home is required.",
    )
    parser.add_argument(
        "--recover_robot",
        action="store_true",
        help="Clear warning/error state and re-enable motion without moving home or opening ZED.",
    )
    parser.add_argument(
        "--home_only",
        action="store_true",
        help="Only test the checkpoint8 move_gohome initialization precondition; do not open ZED or move the gripper.",
    )
    parser.add_argument(
        "--skip_home",
        action="store_true",
        help="Compatibility flag; home motion is skipped unless --move_home is passed.",
    )
    parser.add_argument(
        "--move_home",
        action="store_true",
        help="Explicitly call move_gohome during initialization. Warning: this previously triggered C22 in this lab setup.",
    )
    parser.add_argument(
        "--no_final_home",
        action="store_true",
        help="Compatibility flag; final home is skipped unless --final_home is passed.",
    )
    parser.add_argument(
        "--final_home",
        action="store_true",
        help="Explicitly call move_gohome during cleanup. Warning: this may trigger C22 in this lab setup.",
    )
    parser.add_argument(
        "--return_to_start_joints",
        action="store_true",
        help="After a successful task, return to the recorded starting 6-axis joint pose instead of move_gohome.",
    )
    parser.add_argument("--return_joint_speed", type=float, default=10.0)
    parser.add_argument(
        "--motion_impl",
        choices=["checkpoint_blackbox", "checked_pregrasp"],
        default="checkpoint_blackbox",
        help="Robot motion implementation. checkpoint_blackbox is the known working default.",
    )
    parser.add_argument("--pregrasp_clearance_m", type=float, default=0.12)
    parser.add_argument("--tcp_speed_mm_s", type=float, default=100.0)
    parser.add_argument("--grasp_contact_z_offset_mm", type=float, default=0.0)
    parser.add_argument(
        "--pregrasp_only",
        action="store_true",
        help="Only move to cube pregrasp, then stop. Useful for reachability diagnosis.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        default=True,
        help="Require operator confirmation before robot motion. This is the default.",
    )
    parser.add_argument("--preview_path", default=DEFAULT_PREVIEW_PATH)
    return parser.parse_args()


def _gui_display_available() -> bool:
    if os.name != "posix":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _preview_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_preview(image, preview_path: str) -> Path | None:
    path = _preview_path(preview_path)
    if not cv2.imwrite(str(path), image):
        print(f"Warning: failed to write preview image to {path}")
        return None
    print(f"Saved checkpoint8-style preview: {path}")
    return path


def _confirm_or_preview(image, no_gui: bool, dry_run: bool, preview_path: str) -> bool:
    if dry_run:
        _save_preview(image, preview_path)
        if no_gui or not _gui_display_available():
            print("Dry run selected. Preview saved; no robot confirmation required.")
            return False
        cv2.namedWindow("checkpoint8-style cube pose preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("checkpoint8-style cube pose preview", 1280, 720)
        cv2.imshow("checkpoint8-style cube pose preview", image)
        print("Dry run selected. Press any key in the OpenCV window to close the preview.")
        cv2.waitKey(0)
        cv2.destroyWindow("checkpoint8-style cube pose preview")
        return False

    if no_gui or not _gui_display_available():
        _save_preview(image, preview_path)
        try:
            response = input("Type 'k' then Enter to execute, or anything else to cancel: ")
        except EOFError:
            print("No terminal input is available. Cancelling without robot motion.")
            return False
        return response.strip().lower() == "k"

    cv2.namedWindow("Verifying Cube Pose", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Verifying Cube Pose", 1280, 720)
    cv2.imshow("Verifying Cube Pose", image)
    print("Press 'k' in the OpenCV window to execute. Press any other key to cancel.")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow("Verifying Cube Pose")
    return key == ord("k")


def _check_code(code: Any, action: str) -> None:
    if code is None:
        return
    if isinstance(code, tuple):
        code = code[0]
    if int(code) != 0:
        raise RuntimeError(f"Lite6 command failed during {action}: code={code}")


def _split_response(response: Any) -> tuple[int | None, Any]:
    if isinstance(response, tuple):
        if len(response) >= 2 and isinstance(response[0], Integral):
            return int(response[0]), response[1]
        if len(response) == 1:
            return None, response[0]
    return None, response


def _read_arm_value(arm: Any, method_name: str) -> tuple[Any, Any]:
    raw = getattr(arm, method_name)()
    code, value = _split_response(raw)
    if code is not None and code != 0:
        raise RuntimeError(f"Lite6 status read failed during {method_name}(): code={code}, response={raw!r}")
    return value, raw


def _parse_state(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise RuntimeError(f"Unable to parse Lite6 state: {value!r}")
        value = value[0]
    return int(value)


def _parse_err_warn(value: Any) -> tuple[int, int]:
    if isinstance(value, dict):
        err = value.get("err", value.get("error", value.get("error_code", 0)))
        warn = value.get("warn", value.get("warning", value.get("warn_code", 0)))
        return int(err), int(warn)
    if isinstance(value, Integral):
        return int(value), 0
    if isinstance(value, (list, tuple)):
        if len(value) < 2:
            raise RuntimeError(f"Unable to parse Lite6 err/warn: {value!r}")
        return int(value[0]), int(value[1])
    raise RuntimeError(f"Unable to parse Lite6 err/warn: {value!r}")


def _print_and_check_ready(arm: Any, label: str) -> None:
    state_value, state_raw = _read_arm_value(arm, "get_state")
    err_warn_value, err_warn_raw = _read_arm_value(arm, "get_err_warn_code")
    state = _parse_state(state_value)
    err, warn = _parse_err_warn(err_warn_value)

    print(f"Lite6 status {label}:")
    print(f"  get_state(): {state_raw!r}")
    print(f"  get_err_warn_code(): {err_warn_raw!r}")

    if err != 0 or warn != 0:
        raise RuntimeError(f"Lite6 is not ready {label}: err={err}, warn={warn}.")
    if state not in (0, 2):
        state_note = " PAUSE" if state == 3 else " STOP" if state == 4 else ""
        raise RuntimeError(f"Lite6 is not ready {label}: state={state}{state_note}.")
    print(f"Lite6 ready {label}: state={state}, err/warn=[0, 0]")


def _parse_six_joint_angles(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 6:
        raise RuntimeError(f"Unable to parse Lite6 servo angles: {value!r}")
    return [float(angle) for angle in value[:6]]


def _record_start_pose(arm: Any) -> list[float]:
    servo_value, servo_raw = _read_arm_value(arm, "get_servo_angle")
    position_value, position_raw = _read_arm_value(arm, "get_position")
    start_angles_6d = _parse_six_joint_angles(servo_value)

    print(f"Recorded start joint pose: {start_angles_6d}")
    print(f"Recorded start TCP pose: {position_raw!r}")
    if position_value is not None:
        print(f"Parsed start TCP pose: {position_value!r}")
    return start_angles_6d


def _print_full_robot_status(arm: Any, label: str, require_ready: bool = True) -> bool:
    state_value, state_raw = _read_arm_value(arm, "get_state")
    err_warn_value, err_warn_raw = _read_arm_value(arm, "get_err_warn_code")
    state = _parse_state(state_value)
    err, warn = _parse_err_warn(err_warn_value)

    print(f"Lite6 status {label}:")
    print(f"  get_state(): {state_raw!r}")
    print(f"  get_err_warn_code(): {err_warn_raw!r}")
    try:
        position_value, position_raw = _read_arm_value(arm, "get_position")
        print(f"  get_position(): {position_raw!r}")
        if position_value is not None:
            print(f"  parsed TCP pose: {position_value!r}")
    except Exception as exc:
        print(f"  get_position(): failed: {exc}")
    try:
        servo_value, servo_raw = _read_arm_value(arm, "get_servo_angle")
        print(f"  get_servo_angle(): {servo_raw!r}")
        if servo_value is not None:
            print(f"  parsed servo angles: {servo_value!r}")
    except Exception as exc:
        print(f"  get_servo_angle(): failed: {exc}")

    if err != 0 or warn != 0:
        if require_ready:
            raise RuntimeError(f"Lite6 is not ready {label}: err={err}, warn={warn}.")
        return False
    if state not in (0, 2):
        state_note = " PAUSE" if state == 3 else " STOP" if state == 4 else ""
        if require_ready:
            raise RuntimeError(f"Lite6 is not ready {label}: state={state}{state_note}.")
        return False
    return True


def _robot_ready_for_return(arm: Any, label: str) -> bool:
    try:
        return _print_full_robot_status(arm, label, require_ready=False)
    except Exception as exc:
        print(f"Warning: failed to read robot status {label}: {exc}")
        return False


def _print_failure_status(arm: Any, label: str) -> None:
    try:
        _print_full_robot_status(arm, label, require_ready=False)
    except Exception as exc:
        print(f"Warning: failed to read robot status {label}: {exc}")


def _check_command_and_state(arm: Any, code: Any, action: str) -> None:
    try:
        _check_code(code, action)
    except Exception:
        _print_failure_status(arm, f"after failed {action}")
        raise
    try:
        _print_full_robot_status(arm, f"after {action}", require_ready=True)
    except Exception:
        _print_failure_status(arm, f"after failed {action}")
        raise


def _print_motion_target(
    arm: Any,
    name: str,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    speed: float,
) -> None:
    _print_full_robot_status(arm, f"before {name}", require_ready=True)
    print(
        f"Moving {name}:"
        f" x={x:.1f} y={y:.1f} z={z:.1f}"
        f" roll={roll:.4f} pitch={pitch:.4f} yaw={yaw:.4f}"
        f" speed={float(speed):.1f}"
    )


def _set_position_checked(
    arm: Any,
    name: str,
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    speed: float,
) -> None:
    _print_motion_target(arm, name, x, y, z, roll, pitch, yaw, speed)
    code = arm.set_position(
        x=x,
        y=y,
        z=z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        speed=float(speed),
        wait=True,
    )
    _check_command_and_state(arm, code, name)


def _open_gripper_checked(arm: Any) -> None:
    _print_full_robot_status(arm, "before open gripper", require_ready=True)
    _check_command_and_state(arm, arm.open_lite6_gripper(sync=True), "open gripper")


def _close_gripper_checked(arm: Any) -> None:
    _print_full_robot_status(arm, "before close gripper", require_ready=True)
    _check_command_and_state(arm, arm.close_lite6_gripper(sync=True), "close gripper")


def _return_to_start_joints(arm: Any, start_angles_6d: list[float], speed: float) -> None:
    print("Returning to recorded start joint pose.")
    _check_command_and_state(
        arm,
        arm.set_servo_angle(angle=start_angles_6d, speed=float(speed), wait=True),
        "return to recorded start joint pose",
    )
    print("Return-to-start complete.")


def _cube_pose_to_motion_args(t_robot_cube, grasp_contact_z_offset_mm: float) -> tuple[float, float, float, float, float, float]:
    x, y, z, roll, pitch, yaw = checkpoint_pose_to_mm_deg(t_robot_cube)
    return (
        float(x),
        float(y),
        float(z) + float(grasp_contact_z_offset_mm),
        float(roll),
        float(pitch),
        float(yaw),
    )


def _execute_checked_pregrasp_motion(
    arm: Any,
    t_robot_cube,
    pregrasp_clearance_m: float,
    tcp_speed_mm_s: float,
    grasp_contact_z_offset_mm: float,
    pregrasp_only: bool,
) -> None:
    x, y, z_contact, roll, pitch, yaw = _cube_pose_to_motion_args(t_robot_cube, grasp_contact_z_offset_mm)
    z_pre = z_contact + float(pregrasp_clearance_m) * 1000.0

    print("Executing checked_pregrasp motion implementation.")
    print(
        "Converted cube pose:"
        f" x={x:.1f} y={y:.1f} z_contact={z_contact:.1f}"
        f" z_pre={z_pre:.1f}"
        f" roll={roll:.4f} pitch={pitch:.4f} yaw={yaw:.4f}"
    )

    _open_gripper_checked(arm)
    _set_position_checked(arm, "cube pregrasp", x, y, z_pre, roll, pitch, yaw, tcp_speed_mm_s)
    if pregrasp_only:
        print("pregrasp_only selected. Stopping after successful cube pregrasp; no descent, grasp, or place executed.")
        return

    _set_position_checked(arm, "cube grasp/contact", x, y, z_contact, roll, pitch, yaw, tcp_speed_mm_s)
    _close_gripper_checked(arm)
    _set_position_checked(arm, "cube pregrasp after grasp", x, y, z_pre, roll, pitch, yaw, tcp_speed_mm_s)

    _set_position_checked(arm, "place prepose", x, y, z_pre, roll, pitch, yaw, tcp_speed_mm_s)
    _set_position_checked(arm, "place contact", x, y, z_contact, roll, pitch, yaw, tcp_speed_mm_s)
    _open_gripper_checked(arm)
    _set_position_checked(arm, "place prepose after release", x, y, z_pre, roll, pitch, yaw, tcp_speed_mm_s)


def _execute_checkpoint_blackbox_motion(arm: Any, t_robot_cube) -> None:
    print("Using checkpoint_blackbox motion implementation.")
    print("Calling checkpoint1.grasp_cube.")
    grasp_cube(arm, t_robot_cube)
    print("Robot state after grasp_cube:")
    _print_full_robot_status(arm, "after checkpoint1.grasp_cube", require_ready=True)
    print("Calling checkpoint1.place_cube.")
    place_cube(arm, t_robot_cube)
    print("Robot state after place_cube:")
    _print_full_robot_status(arm, "after checkpoint1.place_cube", require_ready=True)


def _connect_lite6(robot_ip: str, move_home: bool) -> Any:
    print(f"Connecting to Lite6 at {robot_ip}...")
    arm = XArmAPI(robot_ip)
    try:
        _check_code(arm.connect(), "connect")
        _check_code(arm.motion_enable(enable=True), "motion enable")
        _check_code(arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0]), "set TCP offset")
        _check_code(arm.set_mode(0), "set mode")
        _check_code(arm.set_state(0), "set state")
        time.sleep(0.5)

        if move_home:
            print("Warning: --move_home was requested. move_gohome previously triggered C22 in this lab setup.")
            _check_code(arm.move_gohome(wait=True), "move home")
            time.sleep(0.5)
        else:
            print("Skipping initial move_gohome; using current robot state.")
    except Exception:
        _safe_disconnect(arm, final_home=False)
        raise

    return arm


def _safe_disconnect(arm: Any | None, final_home: bool) -> None:
    if arm is None:
        return
    try:
        if hasattr(arm, "stop_lite6_gripper"):
            arm.stop_lite6_gripper()
            time.sleep(0.2)
    except Exception as exc:
        print(f"Warning: failed to stop gripper cleanly: {exc}")
    if final_home:
        try:
            print("Warning: --final_home was requested. move_gohome may trigger C22 in this lab setup.")
            _check_code(arm.move_gohome(wait=True), "final move home")
            time.sleep(0.5)
        except Exception as exc:
            print(f"Warning: failed to move home during cleanup: {exc}")
    else:
        print("Skipping final move_gohome.")
    try:
        arm.disconnect()
    except Exception as exc:
        print(f"Warning: failed to disconnect cleanly: {exc}")


def _print_home_diagnostic_status(arm: Any, label: str) -> None:
    print(f"Lite6 home-only diagnostic status {label}:")
    for method_name in ("get_state", "get_err_warn_code", "get_position", "get_servo_angle"):
        try:
            raw = getattr(arm, method_name)()
            print(f"  {method_name}(): {raw!r}")
        except Exception as exc:
            print(f"  {method_name}(): failed: {exc}")
    if _robot_status_needs_recovery(arm):
        print("Robot is in STOP/C22. Run --recover_robot before doing anything else.")


def _robot_status_needs_recovery(arm: Any) -> bool:
    try:
        _state_code, state_value = _split_response(arm.get_state())
        _err_code, err_warn_value = _split_response(arm.get_err_warn_code())
        state = _parse_state(state_value)
        err, _warn = _parse_err_warn(err_warn_value)
        return state == 4 or err == 22
    except Exception:
        return False


def _recover_connected_arm(arm: Any) -> None:
    if hasattr(arm, "clean_warn"):
        _check_code(arm.clean_warn(), "clean warnings")
    if hasattr(arm, "clean_error"):
        _check_code(arm.clean_error(), "clean errors")
    _check_code(arm.motion_enable(enable=True), "motion enable")
    _check_code(arm.set_mode(0), "set mode")
    _check_code(arm.set_state(0), "set state")
    time.sleep(0.5)


def _run_recover_robot(robot_ip: str) -> None:
    print("Running robot recovery helper.")
    print("This clears C22/error state and re-enables motion without calling move_gohome.")
    print(f"Connecting to Lite6 at {robot_ip}...")
    arm = XArmAPI(robot_ip)
    try:
        _check_code(arm.connect(), "connect")
        _print_home_diagnostic_status(arm, "before recovery")
        _recover_connected_arm(arm)
        _print_home_diagnostic_status(arm, "after recovery")
    finally:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"Warning: failed to disconnect cleanly: {exc}")


def _run_home_only_diagnostic(robot_ip: str) -> None:
    print("Running home-only diagnostic.")
    print("This tests the original checkpoint8 move_gohome precondition.")
    print(f"Connecting to Lite6 at {robot_ip}...")
    arm = XArmAPI(robot_ip)
    try:
        _check_code(arm.connect(), "connect")
        _print_home_diagnostic_status(arm, "before recovery")
        _recover_connected_arm(arm)
        _print_home_diagnostic_status(arm, "before move_gohome")
        home_code = arm.move_gohome(wait=True)
        print(f"move_gohome(wait=True) return code: {home_code!r}")
        if isinstance(home_code, tuple):
            failed = int(home_code[0]) != 0
        else:
            failed = int(home_code) != 0
        if failed:
            print("move_gohome failed; the robot cannot currently satisfy checkpoint home precondition.")
        time.sleep(0.5)
        _print_home_diagnostic_status(arm, "after move_gohome")
    finally:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"Warning: failed to disconnect cleanly: {exc}")


def _strict_checkpoint8_args_ok(args: argparse.Namespace) -> bool:
    if args.skip_home or args.no_final_home or args.return_to_start_joints:
        print("strict_checkpoint8 requires start and final home, matching the original checkpoint.")
        raise SystemExit(2)
    return True


def _strict_checkpoint8_initialize_arm(robot_ip: str) -> Any:
    print(f"Connecting to Lite6 at {robot_ip}...")
    arm = XArmAPI(robot_ip)
    try:
        _check_code(arm.connect(), "connect")
        if hasattr(arm, "clean_warn"):
            _check_code(arm.clean_warn(), "clean warnings")
        if hasattr(arm, "clean_error"):
            _check_code(arm.clean_error(), "clean errors")
        _check_code(arm.motion_enable(enable=True), "motion enable")
        _check_code(arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0]), "set TCP offset")
        _check_code(arm.set_mode(0), "set mode")
        _check_code(arm.set_state(0), "set state")
        time.sleep(0.5)

        _print_home_diagnostic_status(arm, "before initial move_gohome")
        home_code = arm.move_gohome(wait=True)
        print(f"initial move_gohome(wait=True) return code: {home_code!r}")
        if isinstance(home_code, tuple):
            failed = int(home_code[0]) != 0
        else:
            failed = int(home_code) != 0
        time.sleep(0.5)
        _print_home_diagnostic_status(arm, "after initial move_gohome")
        if failed:
            print(
                "move_gohome failed. The robot is not in a valid state to start checkpoint-style tasks. "
                "Recover the robot in xArm Studio or power-cycle/re-enable, then retry."
            )
            raise RuntimeError("strict checkpoint8 initial move_gohome failed")
    except Exception:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"Warning: failed to disconnect cleanly after strict init failure: {exc}")
        raise
    return arm


def _strict_checkpoint8_cleanup(arm: Any | None) -> None:
    if arm is None:
        return
    try:
        if hasattr(arm, "stop_lite6_gripper"):
            arm.stop_lite6_gripper()
            time.sleep(0.2)
    except Exception as exc:
        print(f"Warning: failed to stop gripper cleanly: {exc}")
    try:
        _print_home_diagnostic_status(arm, "before final move_gohome")
        home_code = arm.move_gohome(wait=True)
        print(f"final move_gohome(wait=True) return code: {home_code!r}")
        time.sleep(0.5)
        _print_home_diagnostic_status(arm, "after final move_gohome")
    except Exception as exc:
        print(f"Warning: final move_gohome failed: {exc}")
        if _robot_status_needs_recovery(arm):
            print("Robot is in STOP/C22. Run --recover_robot before doing anything else.")
    try:
        arm.disconnect()
    except Exception as exc:
        print(f"Warning: failed to disconnect cleanly: {exc}")


def _run_strict_checkpoint8(args: argparse.Namespace) -> None:
    if not _strict_checkpoint8_args_ok(args):
        return

    print("Running strict checkpoint8 mode.")
    print("This mode requires initial move_gohome and final move_gohome, matching the original checkpoint.")
    zed = None
    arm = None
    try:
        print("Opening live ZED camera...")
        zed = ZedCamera()
        camera_intrinsic = zed.camera_intrinsic
        cube_pose_detector = CubePoseDetector(camera_intrinsic)

        arm = _strict_checkpoint8_initialize_arm(args.robot_ip)

        cv_image = zed.image
        point_cloud = zed.point_cloud
        if cv_image is None or point_cloud is None:
            print("Camera data is not ready.")
            return

        pose_pair = cube_pose_detector.get_transforms([cv_image, point_cloud], args.cube_prompt)
        if pose_pair is None:
            print(f"No cube matched prompt: {args.cube_prompt}")
            return

        t_robot_cube, t_cam_cube = pose_pair
        _print_cube_pose(t_robot_cube)

        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube, size=CUBE_SIZE)
        confirmed = _confirm_or_preview(
            image=cv_image,
            no_gui=args.no_gui,
            dry_run=args.dry_run,
            preview_path=args.preview_path,
        )
        if args.dry_run:
            print("Strict checkpoint8 dry run selected. Home precondition was tested; no grasp/place executed.")
            return
        if not confirmed:
            print("Operator cancelled. No robot motion executed.")
            return

        print("Calling checkpoint1.grasp_cube.")
        grasp_cube(arm, t_robot_cube)
        print("Robot state after grasp_cube:")
        _print_full_robot_status(arm, "after checkpoint1.grasp_cube", require_ready=True)
        print("Calling checkpoint1.place_cube.")
        place_cube(arm, t_robot_cube)
        print("Robot state after place_cube:")
        _print_full_robot_status(arm, "after checkpoint1.place_cube", require_ready=True)
        print("strict checkpoint8 task complete: cube grasped and placed back down.")
    finally:
        _strict_checkpoint8_cleanup(arm)
        if zed is not None:
            zed.close()
        if (not args.no_gui) and _gui_display_available():
            try:
                cv2.destroyAllWindows()
            except cv2.error as exc:
                print(f"Warning: failed to destroy OpenCV windows cleanly: {exc}")


def _print_cube_pose(t_robot_cube) -> None:
    print("Detected cube pose in robot/base frame:")
    print(
        "  translation:"
        f" x={float(t_robot_cube[0, 3]):.4f}m"
        f" y={float(t_robot_cube[1, 3]):.4f}m"
        f" z={float(t_robot_cube[2, 3]):.4f}m"
    )
    print("  rotation matrix:")
    print(t_robot_cube[:3, :3])


def main() -> None:
    args = _parse_args()
    if args.recover_robot:
        _run_recover_robot(args.robot_ip)
        return
    if args.home_only:
        _run_home_only_diagnostic(args.robot_ip)
        return
    if args.strict_checkpoint8:
        _run_strict_checkpoint8(args)
        return

    print("Non-home mode is diagnostic only and does not match course checkpoints.")

    zed = None
    arm = None

    try:
        print("Opening live ZED camera...")
        zed = ZedCamera()
        camera_intrinsic = zed.camera_intrinsic
        cube_pose_detector = CubePoseDetector(camera_intrinsic)

        cv_image = zed.image
        point_cloud = zed.point_cloud
        if cv_image is None or point_cloud is None:
            print("Camera data is not ready.")
            return

        pose_pair = cube_pose_detector.get_transforms([cv_image, point_cloud], args.cube_prompt)
        if pose_pair is None:
            print(f"No cube matched prompt: {args.cube_prompt}")
            return

        t_robot_cube, t_cam_cube = pose_pair
        _print_cube_pose(t_robot_cube)

        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube, size=CUBE_SIZE)
        confirmed = _confirm_or_preview(
            image=cv_image,
            no_gui=args.no_gui,
            dry_run=args.dry_run,
            preview_path=args.preview_path,
        )

        if args.dry_run:
            print(f"motion_impl={args.motion_impl}")
            print("return_to_start_joints only applies to real robot execution.")
            print("Dry run selected. Detection completed; no robot motion executed.")
            return
        if not confirmed:
            print("Operator cancelled. No robot motion executed.")
            return

        if args.move_home and args.skip_home:
            print("Both --move_home and --skip_home were provided; skipping initial move_gohome.")
        arm = _connect_lite6(robot_ip=args.robot_ip, move_home=(args.move_home and not args.skip_home))
        _print_and_check_ready(arm, "before checkpoint8-style grasp/place")
        start_angles_6d = _record_start_pose(arm)

        motion_succeeded = False
        try:
            if args.motion_impl == "checkpoint_blackbox":
                if args.pregrasp_only:
                    print("--pregrasp_only is ignored with --motion_impl checkpoint_blackbox.")
                _execute_checkpoint_blackbox_motion(arm, t_robot_cube)
            else:
                print("Warning: checked_pregrasp is diagnostic and has triggered C22 in this setup.")
                _execute_checked_pregrasp_motion(
                    arm=arm,
                    t_robot_cube=t_robot_cube,
                    pregrasp_clearance_m=args.pregrasp_clearance_m,
                    tcp_speed_mm_s=args.tcp_speed_mm_s,
                    grasp_contact_z_offset_mm=args.grasp_contact_z_offset_mm,
                    pregrasp_only=args.pregrasp_only,
                )
            motion_succeeded = True
        except Exception as exc:
            print(f"Robot motion failed: {exc}")
            if args.return_to_start_joints:
                if _robot_ready_for_return(arm, "after task failure"):
                    print("Skipping return_to_start_joints because task motion failed before completion.")
                else:
                    print("Skipping return_to_start_joints because robot is not ready after task failure.")
            raise

        if args.pregrasp_only and args.motion_impl == "checked_pregrasp":
            print("pregrasp_only motion complete. No grasp/place sequence was executed.")
        else:
            print("checkpoint8-style mini task complete: cube grasped and placed back down.")

        if args.return_to_start_joints and motion_succeeded:
            if _robot_ready_for_return(arm, "before return_to_start_joints"):
                try:
                    _return_to_start_joints(arm, start_angles_6d, args.return_joint_speed)
                except Exception as exc:
                    print(f"Error: return to recorded start joint pose failed: {exc}")
                    raise
            else:
                print("Skipping return_to_start_joints because robot is not ready after task completion.")

    finally:
        if args.return_to_start_joints and args.final_home:
            print("Both --return_to_start_joints and --final_home were provided; skipping final_home.")
        _safe_disconnect(
            arm,
            final_home=(args.final_home and not args.no_final_home and not args.return_to_start_joints),
        )
        if zed is not None:
            zed.close()
        if (not args.no_gui) and _gui_display_available():
            try:
                cv2.destroyAllWindows()
            except cv2.error as exc:
                print(f"Warning: failed to destroy OpenCV windows cleanly: {exc}")


if __name__ == "__main__":
    main()
