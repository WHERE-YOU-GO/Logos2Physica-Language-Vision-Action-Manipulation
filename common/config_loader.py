from __future__ import annotations

from pathlib import Path
from typing import Any

from .exceptions import ProjectError
from .path_manager import resolve_path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


def _ensure_yaml_available() -> None:
    if yaml is None:
        raise ProjectError("PyYAML is required to load configuration files.") from _YAML_IMPORT_ERROR


def _read_mapping(path: str | Path) -> dict[str, Any]:
    _ensure_yaml_available()

    path_obj = resolve_path(path)
    if not path_obj.exists():
        raise ProjectError(f"Configuration file does not exist: {path_obj}")
    if not path_obj.is_file():
        raise ProjectError(f"Configuration path is not a file: {path_obj}")

    try:
        with path_obj.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ProjectError(f"Invalid YAML in configuration file: {path_obj}") from exc
    except OSError as exc:
        raise ProjectError(f"Failed to read configuration file: {path_obj}") from exc

    if not isinstance(payload, dict):
        payload_type = type(payload).__name__
        raise ProjectError(
            f"Configuration root must be a mapping in file {path_obj}, got {payload_type}."
        )

    return dict(payload)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its top-level mapping."""

    return _read_mapping(path)


def load_camera_config(path: str | Path) -> dict[str, Any]:
    """Load camera configuration."""

    return _read_mapping(path)


def load_robot_config(path: str | Path) -> dict[str, Any]:
    """Load robot configuration."""

    return _read_mapping(path)


def load_workspace_config(path: str | Path) -> dict[str, Any]:
    """Load workspace configuration."""

    return _read_mapping(path)


def load_detector_config(path: str | Path) -> dict[str, Any]:
    """Load detector configuration."""

    return _read_mapping(path)


__all__ = [
    "load_yaml",
    "load_camera_config",
    "load_robot_config",
    "load_workspace_config",
    "load_detector_config",
]
