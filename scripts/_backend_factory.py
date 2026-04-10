from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from common.logger import ProjectLogger


@dataclass(frozen=True)
class BackendSpec:
    module_name: str
    class_name: str
    dependency_hints: tuple[str, ...]


BACKEND_REGISTRY: dict[str, BackendSpec] = {
    "yolo_world": BackendSpec(
        module_name="perception.yolo_world_backend",
        class_name="YOLOWorldBackend",
        dependency_hints=("ultralytics", "torch", "numpy"),
    ),
    "florence2": BackendSpec(
        module_name="perception.florence2_backend",
        class_name="Florence2Backend",
        dependency_hints=("torch", "transformers", "Pillow"),
    ),
    "owlv2": BackendSpec(
        module_name="perception.owlv2_backend",
        class_name="OWLv2Backend",
        dependency_hints=("torch", "transformers", "Pillow"),
    ),
    "groundingdino": BackendSpec(
        module_name="perception.groundingdino_backend",
        class_name="GroundingDINOBackend",
        dependency_hints=("groundingdino", "torch", "numpy"),
    ),
}


def available_backend_names() -> list[str]:
    return sorted(BACKEND_REGISTRY)


def get_backend_spec(backend_name: str) -> BackendSpec:
    normalized = backend_name.strip().lower()
    if normalized not in BACKEND_REGISTRY:
        supported = ", ".join(available_backend_names())
        raise ValueError(f"Unsupported backend {backend_name!r}. Supported backends: {supported}.")
    return BACKEND_REGISTRY[normalized]


def load_backend_class(backend_name: str) -> type[Any]:
    normalized = backend_name.strip().lower()
    spec = get_backend_spec(normalized)
    dependency_text = ", ".join(spec.dependency_hints)
    try:
        module = importlib.import_module(spec.module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import backend '{normalized}'. "
            f"This is a dependency or compatibility problem for the selected backend only, "
            f"not proof that the entire project is broken. "
            f"Expected runtime packages include: {dependency_text}. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        backend_class = getattr(module, spec.class_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Backend '{normalized}' was imported from '{spec.module_name}', "
            f"but class '{spec.class_name}' is missing. "
            f"This only affects the selected backend implementation."
        ) from exc

    if not isinstance(backend_class, type):
        raise RuntimeError(
            f"Backend '{normalized}' resolved '{spec.class_name}' from '{spec.module_name}', "
            "but it is not a class."
        )
    return backend_class


def build_backend(backend_name: str, config_path: str, logger: ProjectLogger) -> Any:
    normalized = backend_name.strip().lower()
    backend_class = load_backend_class(normalized)
    try:
        return backend_class(config_path=config_path, logger=logger)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize backend '{normalized}'. "
            f"This only affects the selected backend, not the entire project. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


__all__ = [
    "BACKEND_REGISTRY",
    "BackendSpec",
    "available_backend_names",
    "build_backend",
    "get_backend_spec",
    "load_backend_class",
]
