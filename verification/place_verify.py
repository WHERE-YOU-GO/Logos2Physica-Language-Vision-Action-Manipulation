from __future__ import annotations

from typing import Any

import numpy as np

from common.datatypes import SceneObject, SceneState
from common.logger import ProjectLogger
from semantic_interface.command_schema import ObjectQuery, ResolvedCommand, SpatialRelation


def _matches_query(obj: SceneObject, query: ObjectQuery) -> bool:
    if query.color is not None and obj.color != query.color:
        return False
    if query.shape is not None and obj.shape != query.shape:
        return False
    if query.category is None:
        return True
    return query.category in {
        obj.label,
        obj.shape,
        str(obj.extras.get("category", obj.label)),
    }


class PlaceVerifier:
    def __init__(self, logger: ProjectLogger) -> None:
        self._logger = logger

    def _find_best_match(self, scene_after: SceneState, query: ObjectQuery) -> SceneObject | None:
        matches = [obj for obj in scene_after.objects if _matches_query(obj, query)]
        if not matches:
            return None
        matches.sort(key=lambda obj: float(obj.confidence), reverse=True)
        return matches[0]

    def verify(self, resolved_cmd: ResolvedCommand, scene_after: SceneState) -> tuple[bool, str]:
        source_obj = self._find_best_match(scene_after, resolved_cmd.parsed.source)
        if source_obj is None:
            return False, "Placed source object is not visible in the post-place scene."

        relation = resolved_cmd.parsed.relation
        target_query = resolved_cmd.parsed.target
        target_obj = None if target_query is None else self._find_best_match(scene_after, target_query)

        if relation == SpatialRelation.ON:
            if target_obj is None:
                return False, "Target object is not visible after placement."
            xy_distance = float(np.linalg.norm(source_obj.center_base[:2] - target_obj.center_base[:2]))
            z_gap = float(source_obj.center_base[2] - target_obj.center_base[2])
            if xy_distance < 0.06 and z_gap > 0.02:
                return True, "Source object is close to the target top surface."
            return False, "Source object is not sufficiently aligned with the target top surface."

        if relation == SpatialRelation.TO:
            if target_obj is not None:
                xy_distance = float(np.linalg.norm(source_obj.center_base[:2] - target_obj.center_base[:2]))
                if xy_distance < 0.08:
                    return True, "Source object is close to the requested target area."
                return False, "Source object is too far from the requested target area."

            if scene_after.table_height_m is not None:
                table_delta = abs(float(source_obj.center_base[2]) - float(scene_after.table_height_m))
                if table_delta < 0.08:
                    return True, "Source object appears to rest on the table in the default area."
            return True, "Source object is visible after placement, but target area could not be grounded visually."

        if relation == SpatialRelation.IN:
            if target_obj is None:
                return False, "Container target is not visible after placement."
            xy_distance = float(np.linalg.norm(source_obj.center_base[:2] - target_obj.center_base[:2]))
            if xy_distance < 0.06:
                return True, "Source object appears to be inside the target container footprint."
            return False, "Source object is not aligned with the target container footprint."

        return True, "No spatial relation was requested; source object remains visible after placement."
