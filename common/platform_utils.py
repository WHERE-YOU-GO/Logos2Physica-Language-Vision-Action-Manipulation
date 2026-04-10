from __future__ import annotations

import os
import platform
from pathlib import Path


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_wsl() -> bool:
    if not is_linux():
        return False
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        version_text = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        version_text = ""
    return "microsoft" in platform.release().lower() or "microsoft" in version_text


def get_platform_name() -> str:
    if is_windows():
        return "windows"
    if is_wsl():
        return "wsl"
    if is_linux():
        return "linux"
    return platform.system().lower()


def get_mounted_windows_prefix(path_like: str | Path) -> str | None:
    path_obj = Path(path_like)
    posix_path = path_obj.as_posix().lower()
    for drive_letter in ("c", "d", "e", "f"):
        prefix = f"/mnt/{drive_letter}"
        if posix_path == prefix or posix_path.startswith(f"{prefix}/"):
            return prefix
    return None


def is_mounted_windows_path(path_like: str | Path) -> bool:
    return get_mounted_windows_prefix(path_like) is not None


def is_linux_native_filesystem(path_like: str | Path) -> bool:
    return is_linux() and not is_mounted_windows_path(path_like)


__all__ = [
    "get_mounted_windows_prefix",
    "get_platform_name",
    "is_linux",
    "is_linux_native_filesystem",
    "is_mounted_windows_path",
    "is_windows",
    "is_wsl",
]
