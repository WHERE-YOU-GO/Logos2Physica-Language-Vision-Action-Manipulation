from __future__ import annotations

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root_on_path

bootstrap_root = ensure_repo_root_on_path()

from pathlib import Path

from common.path_manager import get_config_dir, get_data_dir, get_logs_dir, get_project_root, resolve_path


def _check_logs_writable(logs_dir: Path) -> tuple[bool, str]:
    probe_path = logs_dir / ".path_probe"
    try:
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "ok"


def main() -> None:
    project_root = get_project_root()
    config_dir = get_config_dir()
    data_dir = get_data_dir()
    logs_dir = get_logs_dir()
    demo_scene_dir = resolve_path("data/scenes/scene_01")
    demo_rgb = demo_scene_dir / "rgb.png"
    demo_depth = demo_scene_dir / "depth.npy"
    demo_meta = demo_scene_dir / "meta.json"

    logs_writable, logs_message = _check_logs_writable(logs_dir)
    critical_scripts = [
        resolve_path("scripts/check_platform.py"),
        resolve_path("scripts/check_project_paths.py"),
        resolve_path("scripts/run_scene_state_demo.py"),
        resolve_path("scripts/run_pick_plan_demo.py"),
        resolve_path("scripts/run_fsm_once.py"),
    ]

    print("== Project Path Check ==")
    print(f"project_root: {project_root}")
    print(f"bootstrap_root: {bootstrap_root}")
    print(f"project_root_matches_bootstrap: {project_root == bootstrap_root}")
    print(f"config_dir: {config_dir}")
    print(f"config_dir_exists: {config_dir.exists() and config_dir.is_dir()}")
    print(f"data_dir: {data_dir}")
    print(f"data_dir_exists: {data_dir.exists() and data_dir.is_dir()}")
    print(f"logs_dir: {logs_dir}")
    print(f"logs_dir_exists: {logs_dir.exists() and logs_dir.is_dir()}")
    print(f"logs_dir_writable: {logs_writable}")
    print(f"logs_dir_write_check: {logs_message}")
    print(f"demo_scene_dir: {demo_scene_dir}")
    print(f"demo_scene_dir_exists: {demo_scene_dir.exists() and demo_scene_dir.is_dir()}")
    print(f"demo_rgb_exists: {demo_rgb.exists() and demo_rgb.is_file()}")
    print(f"demo_depth_exists: {demo_depth.exists() and demo_depth.is_file()}")
    print(f"demo_meta_exists: {demo_meta.exists() and demo_meta.is_file()}")

    print()
    print("== Critical Scripts ==")
    for script_path in critical_scripts:
        print(f"{script_path.name}: {script_path.exists()}")


if __name__ == "__main__":
    main()
