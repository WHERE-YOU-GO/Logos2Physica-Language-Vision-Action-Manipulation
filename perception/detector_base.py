from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from common.datatypes import Detection2D


class DetectorBase(ABC):
    @abstractmethod
    def warmup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def detect(self, rgb: Any, candidate_labels: list[str]) -> list[Detection2D]:
        raise NotImplementedError

    @abstractmethod
    def detect_phrase(self, rgb: Any, phrase: str) -> list[Detection2D]:
        raise NotImplementedError
