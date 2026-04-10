from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_TO_MODULE = {
    "BBox2D": "common.datatypes",
    "CameraIntrinsics": "common.datatypes",
    "Detection2D": "common.datatypes",
    "GraspPose": "common.datatypes",
    "MotionPlan": "common.datatypes",
    "PickPlacePlan": "common.datatypes",
    "Pose3D": "common.datatypes",
    "RGBDFrame": "common.datatypes",
    "SceneObject": "common.datatypes",
    "SceneState": "common.datatypes",
    "VerificationResult": "common.datatypes",
    "Waypoint": "common.datatypes",
    "DetectionNotFoundError": "common.exceptions",
    "ExecutionError": "common.exceptions",
    "PerceptionError": "common.exceptions",
    "PlanningError": "common.exceptions",
    "ProjectError": "common.exceptions",
    "SafetyViolationError": "common.exceptions",
    "TargetResolutionError": "common.exceptions",
    "VerificationError": "common.exceptions",
    "ProjectLogger": "common.logger",
    "get_config_dir": "common.path_manager",
    "get_data_dir": "common.path_manager",
    "get_logs_dir": "common.path_manager",
    "get_project_root": "common.path_manager",
    "resolve_path": "common.path_manager",
    "get_platform_name": "common.platform_utils",
    "is_linux": "common.platform_utils",
    "is_windows": "common.platform_utils",
    "is_wsl": "common.platform_utils",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module 'common' has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORT_TO_MODULE))


__all__ = sorted(_EXPORT_TO_MODULE)
