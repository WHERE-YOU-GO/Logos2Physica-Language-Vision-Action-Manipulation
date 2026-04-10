from __future__ import annotations

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import importlib
import sys


MODULES_TO_CHECK = (
    "common",
    "sensing",
    "perception",
    "semantic_interface",
    "skill_planning",
    "control_actuation",
    "verification",
    "fsm",
    "eval",
)


def main() -> None:
    failures: list[tuple[str, str]] = []
    print("== Python Import Verification ==")
    for module_name in MODULES_TO_CHECK:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append((module_name, f"{type(exc).__name__}: {exc}"))
            print(f"{module_name}: FAILED")
            print(f"  reason: {type(exc).__name__}: {exc}")
        else:
            print(f"{module_name}: OK")

    if failures:
        print()
        print("Import verification failed.")
        sys.exit(1)

    print()
    print("all imports ok")


if __name__ == "__main__":
    main()
