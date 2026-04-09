from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


class ProjectLogger:
    def __init__(self, log_dir: str) -> None:
        self._log_dir = Path(log_dir).expanduser()
        self._log_dir.mkdir(parents=True, exist_ok=True)

        logger_name = f"Logos2Physica.{self._log_dir.resolve()}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        if not self._logger.handlers:
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self._logger.addHandler(stream_handler)

            file_handler = logging.FileHandler(self._log_dir / "project.log", encoding="utf-8")
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warn(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    def log_json(self, filename: str, payload: dict) -> None:
        path = self._log_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)

    def save_image(self, filename: str, image: Any) -> None:
        path = self._log_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        array = np.asarray(image)
        if array.ndim not in (2, 3):
            raise ValueError(f"Unsupported image shape: {array.shape}")

        suffix = path.suffix.lower()
        try:
            from PIL import Image
        except ImportError:
            Image = None  # type: ignore[assignment]

        if Image is not None and suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
            if array.ndim == 3 and array.shape[-1] == 4:
                pil_image = Image.fromarray(array.astype(np.uint8), mode="RGBA")
            elif array.ndim == 3 and array.shape[-1] == 3:
                pil_image = Image.fromarray(array.astype(np.uint8), mode="RGB")
            elif array.ndim == 2:
                pil_image = Image.fromarray(array.astype(np.uint8), mode="L")
            else:
                raise ValueError(f"Unsupported image channel layout: {array.shape}")
            pil_image.save(path)
            return

        np.save(path.with_suffix(".npy"), array)
