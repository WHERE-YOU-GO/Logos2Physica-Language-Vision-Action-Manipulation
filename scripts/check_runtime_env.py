from __future__ import annotations

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import importlib
import importlib.util
import io
import os
import site
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from common.path_manager import get_project_root, resolve_path

MODULE_NAME_ALIASES = {
    "Pillow": "PIL",
}

BACKEND_REGISTRY: dict[str, dict[str, Any]] = {
    "yolo_world": {
        "dependency_hints": ("ultralytics", "torch", "numpy"),
    },
    "florence2": {
        "dependency_hints": ("torch", "transformers", "Pillow"),
    },
    "owlv2": {
        "dependency_hints": ("torch", "transformers", "Pillow"),
    },
    "groundingdino": {
        "dependency_hints": ("groundingdino", "torch", "numpy"),
    },
}


def _normalize_path(path: str | os.PathLike[str] | None) -> Path | None:
    if path is None:
        return None
    try:
        return Path(path).expanduser().resolve()
    except OSError:
        return Path(path).expanduser()


def _is_within(path: str | os.PathLike[str] | None, root: str | os.PathLike[str] | None) -> bool:
    normalized_path = _normalize_path(path)
    normalized_root = _normalize_path(root)
    if normalized_path is None or normalized_root is None:
        return False
    try:
        normalized_path.relative_to(normalized_root)
        return True
    except ValueError:
        return False


def _short_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _compact_stream_output(text: str) -> str:
    compact = text.strip()
    if not compact:
        return ""
    traceback_marker = "Traceback (most recent call last):"
    if traceback_marker in compact:
        compact = compact.split(traceback_marker, 1)[0].strip()
    lines = [line.rstrip() for line in compact.splitlines() if line.strip()]
    if len(lines) > 8:
        lines = lines[:8] + ["..."]
    return "\n".join(lines)


def _try_import(module_name: str) -> dict[str, Any]:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "name": module_name,
            "ok": False,
            "error": _short_error(exc),
            "import_stdout": _compact_stream_output(stdout_buffer.getvalue()),
            "import_stderr": _compact_stream_output(stderr_buffer.getvalue()),
        }

    return {
        "name": module_name,
        "ok": True,
        "version": getattr(module, "__version__", None),
        "file": getattr(module, "__file__", None),
        "warnings": [str(item.message) for item in caught_warnings],
        "import_stdout": _compact_stream_output(stdout_buffer.getvalue()),
        "import_stderr": _compact_stream_output(stderr_buffer.getvalue()),
    }


def _find_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _safe_usersite() -> str | None:
    try:
        return str(site.getusersitepackages())
    except Exception:
        return None


def _configured_backend_name() -> tuple[str | None, str | None]:
    config_path = resolve_path("config/detector.yaml")
    if not config_path.exists():
        return None, f"detector config not found: {config_path}"
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, _short_error(exc)
    backend_name = None
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("backend:"):
            backend_name = stripped.split(":", 1)[1].strip().strip("'\"").lower() or None
            break
    return backend_name, None


def _print_section(title: str) -> None:
    print()
    print(f"== {title} ==")


def main() -> None:
    usersite = _safe_usersite()
    executable = _normalize_path(sys.executable)
    environment_root = _normalize_path(sys.prefix)
    project_root = get_project_root()
    configured_backend, configured_backend_error = _configured_backend_name()

    package_statuses = {
        "numpy": _try_import("numpy"),
        "torch": _try_import("torch"),
        "transformers": _try_import("transformers"),
        "ultralytics": _try_import("ultralytics"),
        "PIL": _try_import("PIL"),
    }

    risks: list[str] = []

    numpy_status = package_statuses["numpy"]
    torch_status = package_statuses["torch"]

    numpy_major = None
    if numpy_status["ok"] and isinstance(numpy_status.get("version"), str):
        version_text = str(numpy_status["version"])
        major_text = version_text.split(".", 1)[0]
        if major_text.isdigit():
            numpy_major = int(major_text)

    if torch_status["ok"] and usersite and _is_within(torch_status.get("file"), usersite):
        risks.append(
            "torch appears to be imported from the user site-packages directory instead of the active environment."
        )
    if torch_status["ok"] and environment_root is not None and not _is_within(torch_status.get("file"), environment_root):
        risks.append(
            "torch does not appear to live under the active Python environment prefix. Mixed environments can cause import and ABI issues."
        )

    if numpy_major is not None and numpy_major >= 2 and not torch_status["ok"]:
        torch_error = str(torch_status.get("error", "")).lower()
        numpy_abi_markers = (
            "numpy",
            "compiled using numpy 1",
            "_array_api",
            "dtype size changed",
            "module that was compiled",
        )
        if any(marker in torch_error for marker in numpy_abi_markers):
            risks.append(
                "numpy 2.x is installed and torch failed with a NumPy ABI-style error. A compiled extension may have been built against NumPy 1.x."
            )
    if numpy_major is not None and numpy_major >= 2 and torch_status["ok"]:
        torch_warnings = " ".join(torch_status.get("warnings", []))
        lowered_warnings = torch_warnings.lower()
        if any(marker in lowered_warnings for marker in ("numpy", "_array_api", "compiled using numpy 1")):
            risks.append(
                "numpy 2.x is installed and torch emitted NumPy compatibility warnings during import. ABI mismatches are still likely even though import completed."
            )

    _print_section("Interpreter")
    print(f"sys.executable: {sys.executable}")
    print(f"sys.version: {sys.version.replace(os.linesep, ' ')}")
    print(f"cwd: {Path.cwd()}")
    print(f"project_root: {project_root}")
    print(f"user site-packages: {usersite}")
    if configured_backend is not None:
        print(f"configured detector backend: {configured_backend}")
    elif configured_backend_error is not None:
        print(f"configured detector backend: unavailable ({configured_backend_error})")

    _print_section("sys.path")
    max_items = min(8, len(sys.path))
    for index in range(max_items):
        print(f"[{index}] {sys.path[index]}")
    if len(sys.path) > max_items:
        print(f"... ({len(sys.path) - max_items} more entries)")

    _print_section("Package Imports")
    for display_name in ("numpy", "torch", "transformers", "ultralytics", "PIL"):
        status = package_statuses[display_name]
        if status["ok"]:
            version = status.get("version")
            version_text = f" | version={version}" if version else ""
            print(f"{display_name}: OK{version_text}")
            print(f"  file: {status.get('file')}")
            captured_warnings = status.get("warnings", [])
            if captured_warnings:
                print(f"  warnings: {captured_warnings}")
            if status.get("import_stdout"):
                print(f"  import_stdout: {status.get('import_stdout')}")
            if status.get("import_stderr"):
                print(f"  import_stderr: {status.get('import_stderr')}")
        else:
            print(f"{display_name}: IMPORT FAILED")
            print(f"  reason: {status.get('error')}")
            if status.get("import_stdout"):
                print(f"  import_stdout: {status.get('import_stdout')}")
            if status.get("import_stderr"):
                print(f"  import_stderr: {status.get('import_stderr')}")

    _print_section("Backend Dependency Check")
    for backend_name in sorted(BACKEND_REGISTRY):
        spec = BACKEND_REGISTRY[backend_name]
        dependency_reports: list[str] = []
        missing_modules: list[str] = []
        for dependency_name in spec["dependency_hints"]:
            module_name = MODULE_NAME_ALIASES.get(dependency_name, dependency_name)
            cached_status = package_statuses.get(module_name)
            if cached_status is not None:
                is_available = bool(cached_status["ok"])
            else:
                is_available = _find_module(module_name)
            dependency_reports.append(f"{dependency_name}={'OK' if is_available else 'MISSING'}")
            if not is_available:
                missing_modules.append(dependency_name)

        suffix = ""
        if configured_backend == backend_name and missing_modules:
            suffix = " | HIGH RISK: configured backend has missing dependencies"
        elif configured_backend == backend_name:
            suffix = " | configured backend"
        print(f"{backend_name}: {', '.join(dependency_reports)}{suffix}")

        if configured_backend == backend_name and missing_modules:
            risks.append(
                f"Configured backend '{backend_name}' is missing required packages: {', '.join(missing_modules)}."
            )

    _print_section("Risk Signals")
    if risks:
        for risk in risks:
            print(f"- {risk}")
    else:
        print("No high-risk signals detected from this lightweight diagnostic.")


if __name__ == "__main__":
    main()
