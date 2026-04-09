from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.logger import ProjectLogger
from perception.florence2_backend import Florence2Backend
from perception.groundingdino_backend import GroundingDINOBackend
from perception.owlv2_backend import OWLv2Backend
from perception.yolo_world_backend import YOLOWorldBackend


def _load_image(path: str):
    path_obj = Path(path)
    if path_obj.suffix.lower() == ".npy":
        return np.load(path_obj)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required to load non-NPY images.") from exc
    return np.asarray(Image.open(path_obj).convert("RGB"))


def _build_backend(name: str, config_path: str, logger: ProjectLogger):
    backends = {
        "yolo_world": YOLOWorldBackend,
        "florence2": Florence2Backend,
        "owlv2": OWLv2Backend,
        "groundingdino": GroundingDINOBackend,
    }
    if name not in backends:
        raise ValueError(f"Unsupported backend: {name}")
    return backends[name](config_path=config_path, logger=logger)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="yolo_world")
    parser.add_argument("--config", default="config/detector.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--phrase", default=None)
    parser.add_argument("--labels", nargs="*", default=["block", "cube"])
    args = parser.parse_args()

    logger = ProjectLogger("logs/detector_demo")
    rgb = _load_image(args.image)
    backend = _build_backend(args.backend, args.config, logger)

    try:
        backend.warmup()
        if args.phrase:
            detections = backend.detect_phrase(rgb, args.phrase)
        else:
            detections = backend.detect(rgb, args.labels)
    except Exception as exc:
        raise SystemExit(f"Detector demo failed: {exc}")

    for det in detections:
        print(det)


if __name__ == "__main__":
    main()
