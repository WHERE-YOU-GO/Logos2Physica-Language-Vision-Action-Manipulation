from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.config_loader import load_detector_config
from common.datatypes import BBox2D, Detection2D
from common.exceptions import PerceptionError, ProjectError
from common.logger import ProjectLogger
from perception.bbox_postprocess import classwise_nms, clip_boxes_to_image, filter_by_score, keep_topk_per_label
from perception.detector_base import DetectorBase

try:
    from ultralytics import YOLOWorld
except ImportError as exc:  # pragma: no cover
    YOLOWorld = None
    _YOLO_IMPORT_ERROR = exc
else:
    _YOLO_IMPORT_ERROR = None


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


class YOLOWorldBackend(DetectorBase):
    def __init__(self, config_path: str, logger: ProjectLogger) -> None:
        self._config_path = str(Path(config_path))
        self._logger = logger
        self._config = self._load_config()
        self._model: Any | None = None
        self._class_names: list[str] = []

    def _load_config(self) -> dict[str, Any]:
        try:
            return load_detector_config(self._config_path)
        except ProjectError as exc:
            _log(self._logger, "warn", f"Falling back to default detector config: {exc}")
            return {}

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if YOLOWorld is None:
            raise RuntimeError(
                "YOLO-World dependencies are not installed. Install ultralytics to use YOLOWorldBackend."
            ) from _YOLO_IMPORT_ERROR

        model_path = str(self._config.get("model_path", "yolov8s-world.pt"))
        try:
            self._model = YOLOWorld(model_path)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to initialize YOLO-World model from {model_path!r}.") from exc

    def warmup(self) -> None:
        self._ensure_model()
        _log(self._logger, "info", "YOLO-World backend is ready.")

    def set_classes(self, candidate_labels: list[str]) -> None:
        sanitized = [label.strip() for label in candidate_labels if isinstance(label, str) and label.strip()]
        if not sanitized:
            raise ValueError("candidate_labels must contain at least one non-empty label.")

        self._ensure_model()
        assert self._model is not None
        self._class_names = list(dict.fromkeys(sanitized))
        try:
            self._model.set_classes(self._class_names)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Failed to set classes for YOLO-World.") from exc

    def _run_prediction(self, rgb: Any) -> list[Any]:
        self._ensure_model()
        assert self._model is not None

        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise PerceptionError("YOLOWorldBackend expects an RGB image with shape (H, W, 3/4).")
        if image.shape[-1] == 4:
            image = image[..., :3]

        conf = float(self._config.get("score_thresh", 0.2))
        iou = float(self._config.get("iou_thresh", 0.5))
        try:
            results = self._model.predict(source=image, conf=conf, iou=iou, verbose=False)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("YOLO-World inference failed.") from exc
        return list(results)

    def _results_to_detections(self, results: list[Any]) -> list[Detection2D]:
        detections: list[Detection2D] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            names = getattr(result, "names", {})
            if boxes is None:
                continue
            xyxy = getattr(boxes, "xyxy", None)
            conf = getattr(boxes, "conf", None)
            cls = getattr(boxes, "cls", None)
            if xyxy is None or conf is None or cls is None:
                continue

            xyxy_np = np.asarray(xyxy, dtype=np.float32)
            conf_np = np.asarray(conf, dtype=np.float32).reshape(-1)
            cls_np = np.asarray(cls, dtype=np.int32).reshape(-1)
            for idx in range(min(len(xyxy_np), len(conf_np), len(cls_np))):
                x1, y1, x2, y2 = xyxy_np[idx].tolist()
                if x2 <= x1 or y2 <= y1:
                    continue
                class_id = int(cls_np[idx])
                label = str(names.get(class_id, class_id))
                detections.append(
                    Detection2D(
                        label=label,
                        score=float(conf_np[idx]),
                        bbox=BBox2D(
                            x1=int(round(x1)),
                            y1=int(round(y1)),
                            x2=int(round(x2)),
                            y2=int(round(y2)),
                        ),
                        extras={"backend": "yolo_world", "class_id": class_id},
                    )
                )
        return detections

    def detect(self, rgb: Any, candidate_labels: list[str]) -> list[Detection2D]:
        self.set_classes(candidate_labels)
        results = self._run_prediction(rgb)
        detections = self._results_to_detections(results)
        image = np.asarray(rgb)
        if image.ndim < 2:
            raise PerceptionError(f"YOLOWorldBackend expects an image-like array, got shape {image.shape}.")
        detections = clip_boxes_to_image(detections, image_shape=(image.shape[0], image.shape[1]))
        detections = filter_by_score(detections, score_thresh=float(self._config.get("score_thresh", 0.2)))
        detections = classwise_nms(detections, iou_thresh=float(self._config.get("iou_thresh", 0.5)))
        return keep_topk_per_label(detections, k=int(self._config.get("topk_per_label", 3)))

    def detect_phrase(self, rgb: Any, phrase: str) -> list[Detection2D]:
        phrase = phrase.strip()
        if not phrase:
            raise ValueError("phrase must not be empty.")
        return self.detect(rgb, [phrase])
