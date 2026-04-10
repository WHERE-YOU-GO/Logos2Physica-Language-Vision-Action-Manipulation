from __future__ import annotations

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import platform
import sys
from pathlib import Path

from common.path_manager import get_project_root
from common.platform_utils import (
    get_mounted_windows_prefix,
    get_platform_name,
    is_linux_native_filesystem,
    is_mounted_windows_path,
    is_wsl,
)


def main() -> None:
    cwd = Path.cwd()
    system_name = platform.system()
    platform_name = get_platform_name()
    project_root = get_project_root()
    running_in_wsl = is_wsl()
    mounted_drive = get_mounted_windows_prefix(cwd)

    print("== Platform Report ==")
    print(f"platform_name: {platform_name}")
    print(f"platform.system(): {system_name}")
    print(f"platform.release(): {platform.release()}")
    print(f"platform.platform(): {platform.platform()}")
    print(f"sys.executable: {sys.executable}")
    print(f"cwd: {cwd}")
    print(f"project_root: {project_root}")
    print(f"is_wsl: {running_in_wsl}")
    print(f"is_linux_native_filesystem: {is_linux_native_filesystem(cwd)}")
    print(f"is_mounted_windows_path: {is_mounted_windows_path(cwd)}")
    if mounted_drive is not None:
        print(f"mounted_windows_prefix: {mounted_drive}")

    if running_in_wsl and mounted_drive is not None:
        print()
        print("WARNING: The project is running inside WSL from a mounted Windows path.")
        print("Move the repository to /home/<user>/... for better filesystem performance and tool reliability.")


if __name__ == "__main__":
    main()
