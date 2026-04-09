from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from common.config_loader import load_detector_config
from common.datatypes import BBox2D, Detection2D
from common.exceptions import PerceptionError, ProjectError
from common.logger import ProjectLogger
from perception.detector_base import DetectorBase

try:
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]


class Florence2Backend(DetectorBase):
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
            self._logger.warn(f"Falling back to default Florence-2 config: {exc}")
            return {}

    def _ensure_runtime(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        if AutoProcessor is None or AutoModelForCausalLM is None or Image is None or torch is None:
            raise RuntimeError("Florence-2 runtime dependencies are not installed.")

        model_id = str(self._config.get("florence2", {}).get("model_id", "microsoft/Florence-2-base")).strip()
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        self._model.eval()

    def warmup(self) -> None:
        self._ensure_runtime()

    def _run_generation(self, rgb: Any, task_prompt: str) -> str:
        self._ensure_runtime()
        assert self._processor is not None and self._model is not None and Image is not None

        image_np = np.asarray(rgb)
        if image_np.ndim != 3 or image_np.shape[-1] < 3:
            raise PerceptionError("Florence2Backend expects an RGB image with shape (H, W, 3/4).")
        pil_image = Image.fromarray(image_np[..., :3].astype(np.uint8))

        inputs = self._processor(text=task_prompt, images=pil_image, return_tensors="pt")
        generated_ids = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=int(self._config.get("florence2", {}).get("max_new_tokens", 128)),
        )
        return self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def detect(self, rgb, candidate_labels: list[str]) -> list[Detection2D]:
        labels = [label.strip() for label in candidate_labels if label.strip()]
        if not labels:
            raise ValueError("candidate_labels must contain at least one non-empty label.")
        raw_text = self._run_generation(rgb, f"<OD>{', '.join(labels)}")
        return self._parse_generated_output(raw_text)

    def detect_phrase(self, rgb, phrase: str) -> list[Detection2D]:
        phrase = phrase.strip()
        if not phrase:
            raise ValueError("phrase must not be empty.")
        raw_text = self._run_generation(rgb, f"<CAPTION_TO_PHRASE_GROUNDING>{phrase}")
        detections = self._parse_generated_output(raw_text)
        return [
            Detection2D(
                label=phrase,
                score=det.score,
                bbox=det.bbox,
                phrase=phrase,
                extras=dict(det.extras),
            )
            for det in detections
        ]

    def _parse_generated_output(self, raw_text: str) -> list[Detection2D]:
        text = raw_text.strip()
        if not text:
            return []

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None

        detections: list[Detection2D] = []
        if isinstance(payload, dict):
            items = payload.get("detections", [])
        elif isinstance(payload, list):
            items = payload
        else:
            pattern = re.compile(
                r"(?P<label>[a-zA-Z0-9_ -]+)\s*[:=]\s*\[\s*(?P<x1>\d+)\s*,\s*(?P<y1>\d+)\s*,\s*(?P<x2>\d+)\s*,\s*(?P<y2>\d+)\s*\]"
            )
            items = []
            for match in pattern.finditer(text):
                items.append(
                    {
                        "label": match.group("label").strip(),
                        "bbox": [
                            int(match.group("x1")),
                            int(match.group("y1")),
                            int(match.group("x2")),
                            int(match.group("y2")),
                        ],
                        "score": 1.0,
                    }
                )

        for item in items:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection2D(
                    label=str(item.get("label", "object")).strip() or "object",
                    score=float(item.get("score", 1.0)),
                    bbox=BBox2D(x1=x1, y1=y1, x2=x2, y2=y2),
                    phrase=item.get("phrase"),
                    extras={"backend": "florence2"},
                )
            )
        return detections
