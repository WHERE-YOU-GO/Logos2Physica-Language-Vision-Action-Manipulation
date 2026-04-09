from __future__ import annotations

from pathlib import Path
from typing import Any

from common.config_loader import load_detector_config
from common.datatypes import Detection2D
from common.exceptions import ProjectError
from common.logger import ProjectLogger
from perception.detector_base import DetectorBase

try:
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
except ImportError:  # pragma: no cover
    Owlv2ForObjectDetection = None  # type: ignore[assignment]
    Owlv2Processor = None  # type: ignore[assignment]


class OWLv2Backend(DetectorBase):
    def __init__(self, config_path: str, logger: ProjectLogger) -> None:
        self._config_path = str(Path(config_path))
        self._logger = logger
        self._config = self._load_config()
        self._processor = None
        self._model = None

    def _load_config(self) -> dict[str, Any]:
        try:
            return load_detector_config(self._config_path)
        except ProjectError as exc:
            self._logger.warn(f"Falling back to default OWLv2 config: {exc}")
            return {}

    def _ensure_runtime(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        if Owlv2ForObjectDetection is None or Owlv2Processor is None:
            raise RuntimeError("OWLv2 runtime dependencies are not installed.")
        model_id = str(self._config.get("owlv2", {}).get("model_id", "google/owlv2-base-patch16-ensemble")).strip()
        self._processor = Owlv2Processor.from_pretrained(model_id)
        self._model = Owlv2ForObjectDetection.from_pretrained(model_id)
        self._model.eval()

    def warmup(self) -> None:
        self._ensure_runtime()

    def detect(self, rgb, candidate_labels: list[str]) -> list[Detection2D]:
        _ = rgb
        _ = candidate_labels
        self._ensure_runtime()
        raise RuntimeError("OWLv2 detection is not implemented in this MVP placeholder.")

    def detect_phrase(self, rgb, phrase: str) -> list[Detection2D]:
        _ = rgb
        _ = phrase
        self._ensure_runtime()
        raise RuntimeError("OWLv2 phrase grounding is not implemented in this MVP placeholder.")
