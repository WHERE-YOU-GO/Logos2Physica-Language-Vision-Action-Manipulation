from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_path(path_like: str | Path) -> Path:
    path_obj = Path(path_like).expanduser()
    if path_obj.is_absolute():
        return path_obj.resolve(strict=False)
    return (get_project_root() / path_obj).resolve(strict=False)


def get_data_dir() -> Path:
    return resolve_path("data")


def get_logs_dir() -> Path:
    logs_dir = resolve_path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_config_dir() -> Path:
    return resolve_path("config")


__all__ = [
    "get_project_root",
    "get_data_dir",
    "get_logs_dir",
    "get_config_dir",
    "resolve_path",
]
