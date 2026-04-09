from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.config_loader import load_workspace_config
from common.datatypes import Pose3D, SceneState
from common.exceptions import PlanningError, ProjectError
from common.geometry import make_topdown_quaternion
from common.logger import ProjectLogger
from semantic_interface.command_schema import ResolvedCommand, SpatialRelation


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


class PlacePoseResolver:
    def __init__(self, workspace_config_path: str, logger: ProjectLogger) -> None:
        self._logger = logger
        self._workspace_config = self._safe_load(workspace_config_path)
        self._table_height_m = float(self._workspace_config.get("table_height_m", 0.0))
        self._object_half_height_m = float(self._workspace_config.get("default_object_half_height_m", 0.02))
        self._place_margin_m = float(self._workspace_config.get("place_margin_m", 0.01))
        self._default_place_xy = np.asarray(
            self._workspace_config.get("default_place_xy_m", [0.35, 0.0]),
            dtype=np.float64,
        ).reshape(2)
        self._named_areas = dict(self._workspace_config.get("named_areas", {}))

    def _safe_load(self, path: str) -> dict[str, Any]:
        try:
            return load_workspace_config(str(Path(path)))
        except ProjectError as exc:
            _log(self._logger, "warn", f"Falling back to default place settings: {exc}")
            return {}

    def _area_xy_from_query(self, query) -> np.ndarray:
        if query is None:
            return self._default_place_xy.copy()
        for key in filter(None, [query.raw_text, query.color, query.category]):
            candidate = self._named_areas.get(str(key))
            if candidate is not None:
                return np.asarray(candidate, dtype=np.float64).reshape(2)
        return self._default_place_xy.copy()

    def resolve(self, resolved_cmd: ResolvedCommand, scene_state: SceneState) -> Pose3D:
        source_obj = scene_state.get_object_by_id(resolved_cmd.source_id)
        if source_obj is None:
            raise PlanningError(f"Source object {resolved_cmd.source_id!r} not found in scene state.")

        quaternion = make_topdown_quaternion(0.0)
        relation = resolved_cmd.parsed.relation

        if relation == SpatialRelation.ON:
            if resolved_cmd.target_id is None:
                raise PlanningError("ON relation requires a target object.")
            target_obj = scene_state.get_object_by_id(resolved_cmd.target_id)
            if target_obj is None:
                raise PlanningError(f"Target object {resolved_cmd.target_id!r} not found in scene state.")
            position = np.array(
                [
                    float(target_obj.center_base[0]),
                    float(target_obj.center_base[1]),
                    float(target_obj.center_base[2])
                    + self._object_half_height_m
                    + self._object_half_height_m
                    + self._place_margin_m,
                ],
                dtype=np.float64,
            )
            return Pose3D(position=position, quaternion=quaternion, frame_id="base")

        if relation == SpatialRelation.TO and resolved_cmd.target_id is not None:
            target_obj = scene_state.get_object_by_id(resolved_cmd.target_id)
            if target_obj is None:
                raise PlanningError(f"Target object {resolved_cmd.target_id!r} not found in scene state.")
            xy = np.array([float(target_obj.center_base[0]), float(target_obj.center_base[1])], dtype=np.float64)
        else:
            xy = self._area_xy_from_query(resolved_cmd.parsed.target)

        position = np.array(
            [
                float(xy[0]),
                float(xy[1]),
                self._table_height_m + self._object_half_height_m + self._place_margin_m,
            ],
            dtype=np.float64,
        )
        return Pose3D(position=position, quaternion=quaternion, frame_id="base")
