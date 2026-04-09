from __future__ import annotations

from pathlib import Path
from typing import Any

from common.config_loader import load_detector_config
from common.datatypes import Detection2D
from common.exceptions import ProjectError
from common.logger import ProjectLogger
from perception.detector_base import DetectorBase

try:
    import groundingdino  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    groundingdino = None  # type: ignore[assignment]


class GroundingDINOBackend(DetectorBase):
    def __init__(self, config_path: str, logger: ProjectLogger) -> None:
        self._config_path = str(Path(config_path))
        self._logger = logger
        self._config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        try:
            return load_detector_config(self._config_path)
        except ProjectError as exc:
            self._logger.warn(f"Falling back to default GroundingDINO config: {exc}")
            return {}

    def _ensure_runtime(self) -> None:
        if groundingdino is None:
            raise RuntimeError("GroundingDINO runtime dependencies are not installed.")

    def warmup(self) -> None:
        self._ensure_runtime()

    def detect(self, rgb, candidate_labels: list[str]) -> list[Detection2D]:
        _ = rgb
        _ = candidate_labels
        self._ensure_runtime()
        raise RuntimeError("GroundingDINO detection is not implemented in this MVP placeholder.")

    def detect_phrase(self, rgb, phrase: str) -> list[Detection2D]:
        _ = rgb
        _ = phrase
        self._ensure_runtime()
        raise RuntimeError("GroundingDINO phrase grounding is not implemented in this MVP placeholder.")
