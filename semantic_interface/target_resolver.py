from __future__ import annotations

from typing import Any

from common.datatypes import SceneObject, SceneState
from common.exceptions import DetectionNotFoundError, TargetResolutionError
from common.logger import ProjectLogger
from semantic_interface.command_schema import ObjectQuery, ParsedCommand, ResolvedCommand, SpatialRelation
from semantic_interface.llm_parser import LLMCommandParser


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


def _norm(value: str | None) -> str | None:
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


class TargetResolver:
    def __init__(self, logger: ProjectLogger, llm_parser: LLMCommandParser | None = None) -> None:
        self._logger = logger
        self._llm_parser = llm_parser

    def _matches_query(self, obj: SceneObject, query: ObjectQuery) -> bool:
        if query.color is not None and _norm(obj.color) != _norm(query.color):
            return False
        if query.shape is not None and _norm(obj.shape) != _norm(query.shape):
            return False
        if query.category is None:
            return True

        category = _norm(query.category)
        label = _norm(obj.label)
        extras_category = _norm(str(obj.extras.get("category", obj.label)))
        shape = _norm(obj.shape)
        return category in {label, extras_category, shape}

    def _rank_object(self, obj: SceneObject, query: ObjectQuery) -> tuple[float, float, float]:
        exact_color = 1.0 if _norm(obj.color) == _norm(query.color) and query.color is not None else 0.0
        exact_shape = 1.0 if _norm(obj.shape) == _norm(query.shape) and query.shape is not None else 0.0
        return (exact_color + exact_shape, float(obj.confidence), -float(obj.center_base[2]))

    def _resolve_object(
        self,
        query: ObjectQuery,
        scene_state: SceneState,
        exclude_object_id: str | None = None,
    ) -> SceneObject:
        matches = [
            obj
            for obj in scene_state.objects
            if obj.object_id != exclude_object_id and self._matches_query(obj, query)
        ]
        if not matches:
            raise DetectionNotFoundError(f"No object matches query: {query.raw_text!r}")
        matches.sort(key=lambda obj: self._rank_object(obj, query), reverse=True)
        return matches[0]

    def resolve(self, cmd: ParsedCommand, scene_state: SceneState) -> ResolvedCommand:
        source_obj = self._resolve_object(cmd.source, scene_state)

        target_id: str | None = None
        if cmd.target is not None:
            symbolic_target_categories = {"area", "zone", "region"}
            try:
                target_obj = self._resolve_object(cmd.target, scene_state, exclude_object_id=source_obj.object_id)
            except DetectionNotFoundError:
                if cmd.relation == SpatialRelation.TO and _norm(cmd.target.category) in symbolic_target_categories:
                    _log(
                        self._logger,
                        "info",
                        f"Using symbolic workspace target for query {cmd.target.raw_text!r}.",
                    )
                    target_obj = None
                else:
                    raise
            target_id = None if target_obj is None else target_obj.object_id

        resolved = ResolvedCommand(parsed=cmd, source_id=source_obj.object_id, target_id=target_id)
        if cmd.target is not None and cmd.relation == SpatialRelation.ON and target_id is None:
            raise TargetResolutionError("ON relation requires a concrete target object.")
        return resolved
