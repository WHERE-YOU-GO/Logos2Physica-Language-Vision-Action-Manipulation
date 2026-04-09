from __future__ import annotations

import re
from typing import Any

from common.exceptions import TargetResolutionError
from common.logger import ProjectLogger
from semantic_interface.command_schema import (
    ActionType,
    ObjectQuery,
    ParsedCommand,
    SpatialRelation,
)


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


class RegexCommandParser:
    _COLORS = {"red", "blue", "green", "yellow", "orange", "purple", "black", "white", "gray"}
    _SHAPES = {"cube", "block", "box"}
    _CATEGORIES = {"cube", "block", "area", "zone", "region", "container", "bin", "box"}

    def __init__(self, logger: ProjectLogger) -> None:
        self._logger = logger

    def _normalize_text(self, text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"[,.!?]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _extract_object_query(self, text: str) -> ObjectQuery:
        normalized = self._normalize_text(text)
        normalized = re.sub(r"^(the|a|an)\s+", "", normalized)
        tokens = normalized.split()

        color = next((token for token in tokens if token in self._COLORS), None)
        shape = next((token for token in tokens if token in self._SHAPES), None)
        category = next((token for token in tokens if token in self._CATEGORIES), None)
        if category is None and shape is not None:
            category = shape

        return ObjectQuery(
            raw_text=normalized,
            category=category,
            color=color,
            shape=shape,
        )

    def parse(self, prompt: str) -> ParsedCommand:
        normalized = self._normalize_text(prompt)
        if not normalized:
            raise TargetResolutionError("Prompt is empty.")

        patterns = [
            (
                re.compile(
                    r"^(?:pick up|pick|grab|place|put|move)\s+(?P<src>.+?)\s+(?:onto|on)\s+(?P<tgt>.+)$"
                ),
                SpatialRelation.ON,
            ),
            (
                re.compile(
                    r"^(?:place|put|move)\s+(?P<src>.+?)\s+(?:to)\s+(?P<tgt>.+)$"
                ),
                SpatialRelation.TO,
            ),
            (
                re.compile(
                    r"^(?:place|put|move)\s+(?P<src>.+?)\s+(?:into|in)\s+(?P<tgt>.+)$"
                ),
                SpatialRelation.IN,
            ),
            (
                re.compile(r"^(?:pick up|pick|grab)\s+(?P<src>.+)$"),
                None,
            ),
        ]

        for pattern, relation in patterns:
            match = pattern.match(normalized)
            if match is None:
                continue

            source = self._extract_object_query(match.group("src"))
            target = self._extract_object_query(match.group("tgt")) if "tgt" in match.groupdict() else None
            parsed = ParsedCommand(
                action=ActionType.PICK_AND_PLACE,
                source=source,
                target=target,
                relation=relation,
                grasp_mode="topdown",
                raw_prompt=prompt,
            )
            _log(self._logger, "info", f"Parsed prompt into command: {parsed}")
            return parsed

        raise TargetResolutionError(f"Unable to parse prompt: {prompt!r}")
