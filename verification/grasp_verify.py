from __future__ import annotations

from typing import Any

import numpy as np

from common.datatypes import SceneObject, SceneState
from common.logger import ProjectLogger


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


class GraspVerifier:
    def __init__(self, logger: ProjectLogger) -> None:
        self._logger = logger

    def _gripper_holding(self, gripper_state: dict) -> bool:
        if "is_holding" in gripper_state:
            return bool(gripper_state["is_holding"])
        width_m = gripper_state.get("width_m")
        if width_m is None:
            return False
        width_m = float(width_m)
        return 0.002 < width_m < 0.08

    def _find_matching_after_object(self, source_before: SceneObject, scene_after: SceneState) -> SceneObject | None:
        candidates = []
        for obj in scene_after.objects:
            if obj.label != source_before.label:
                continue
            if source_before.color is not None and obj.color != source_before.color:
                continue
            if source_before.shape is not None and obj.shape != source_before.shape:
                continue
            distance = float(np.linalg.norm(obj.center_base - source_before.center_base))
            candidates.append((distance, obj))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def verify(
        self,
        scene_before: SceneState,
        scene_after: SceneState,
        source_id: str,
        gripper_state: dict,
    ) -> tuple[bool, str]:
        source_before = scene_before.get_object_by_id(source_id)
        if source_before is None:
            return False, f"Source object {source_id!r} was not present in the pre-grasp scene."

        gripper_holding = self._gripper_holding(gripper_state)
        matched_after = self._find_matching_after_object(source_before, scene_after)

        if matched_after is None and gripper_holding:
            return True, "Source object disappeared from the table and gripper appears to hold an object."
        if matched_after is None and not gripper_holding:
            return False, "Source object disappeared but gripper state does not indicate a stable grasp."

        assert matched_after is not None
        translation = float(np.linalg.norm(matched_after.center_base - source_before.center_base))
        lifted = float(matched_after.center_base[2] - source_before.center_base[2]) > 0.03

        if gripper_holding and (lifted or translation > 0.05):
            return True, "Source object moved substantially and gripper indicates a grasp."
        if not gripper_holding and translation < 0.03:
            return False, "Gripper does not indicate a grasp and the source object remains near its original pose."
        if gripper_holding:
            return True, "Gripper indicates a grasp even though visual evidence is ambiguous."
        return False, "Grasp verification failed due to inconsistent gripper and scene observations."
