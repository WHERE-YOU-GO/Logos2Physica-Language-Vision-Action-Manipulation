from __future__ import annotations

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ImportError:  # pragma: no cover
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common.path_manager import get_logs_dir, resolve_path

_BACKEND_CHOICES = ("yolo_world", "florence2", "owlv2", "groundingdino")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a detector backend on one image and print structured detection results.",
    )
    parser.add_argument("--backend", default="yolo_world", choices=_BACKEND_CHOICES)
    parser.add_argument("--config", default="config/detector.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--phrase", default=None)
    parser.add_argument("--labels", nargs="*", default=["block", "cube"])
    return parser.parse_args()


def _resolve_image_path(image_path: str) -> Path:
    path_obj = resolve_path(image_path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"Input image does not exist: {path_obj}. "
            "Fix the --image path before loading any detector backend."
        )
    if not path_obj.is_file():
        raise FileNotFoundError(
            f"Input image path is not a file: {path_obj}. "
            "Fix the --image path before loading any detector backend."
        )
    return path_obj


def _load_image(path_obj: Path) -> np.ndarray:
    if path_obj.suffix.lower() == ".npy":
        return np.load(path_obj)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            f"Failed to load image '{path_obj}' because Pillow is not installed. "
            "This is an image-loading dependency issue, not a detector backend failure."
        ) from exc

    with Image.open(path_obj) as handle:
        return np.asarray(handle.convert("RGB"))


def _detect(backend: Any, rgb: np.ndarray, phrase: str | None, labels: list[str]) -> list[Any]:
    if phrase is not None:
        normalized_phrase = phrase.strip()
        if not normalized_phrase:
            raise ValueError("--phrase must not be empty when provided.")
        return backend.detect_phrase(rgb, normalized_phrase)

    normalized_labels = [label.strip() for label in labels if isinstance(label, str) and label.strip()]
    if not normalized_labels:
        raise ValueError("Provide at least one non-empty label when --phrase is not used.")
    return backend.detect(rgb, normalized_labels)


def _detection_to_dict(detection: Any) -> dict[str, Any]:
    return {
        "bbox": {
            "x1": detection.bbox.x1,
            "y1": detection.bbox.y1,
            "x2": detection.bbox.x2,
            "y2": detection.bbox.y2,
        },
        "score": float(detection.score),
        "label": detection.label,
        "phrase": detection.phrase,
    }


def main() -> None:
    args = _parse_args()
    image_path = _resolve_image_path(args.image)
    rgb = _load_image(image_path)

    from common.logger import ProjectLogger
    from scripts._backend_factory import build_backend

    logger = ProjectLogger(get_logs_dir() / "detector_demo")
    backend = build_backend(args.backend, str(resolve_path(args.config)), logger)
    backend.warmup()
    detections = _detect(backend, rgb, args.phrase, args.labels)

    payload = {
        "backend": args.backend,
        "image": str(image_path),
        "image_shape": list(np.asarray(rgb).shape),
        "detection_count": len(detections),
        "detections": [_detection_to_dict(detection) for detection in detections],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
