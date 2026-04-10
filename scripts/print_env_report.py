from __future__ import annotations

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from scripts.check_runtime_env import main as runtime_env_main


def main() -> None:
    runtime_env_main()


if __name__ == "__main__":
    main()
