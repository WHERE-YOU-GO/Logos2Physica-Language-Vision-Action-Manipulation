from __future__ import annotations

from typing import Any

from common.datatypes import Detection2D, RGBDFrame, SceneState
from common.exceptions import PerceptionError
from common.logger import ProjectLogger
from perception.color_shape_refiner import refine_detection_attributes
from perception.depth_project import detection_to_scene_object


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


class SceneStateBuilder:
    def __init__(self, logger: ProjectLogger) -> None:
        self._logger = logger

    def build(self, frame: RGBDFrame, detections: list[Detection2D]) -> SceneState:
        objects = []
        object_index = 1
        ordered_detections = sorted(
            detections,
            key=lambda det: (det.bbox.center_uv()[1], det.bbox.center_uv()[0], det.label),
        )
        for detection in ordered_detections:
            refined = refine_detection_attributes(frame.rgb, detection)
            object_id = f"obj_{object_index}"
            try:
                obj = detection_to_scene_object(refined, frame, object_id)
            except PerceptionError as exc:
                _log(
                    self._logger,
                    "warn",
                    f"Skipping detection {detection.label!r} because 3D projection failed: {exc}",
                )
                continue

            obj.color = refined.extras.get("color")
            obj.shape = refined.extras.get("shape")
            obj.extras["raw_label"] = detection.label
            objects.append(obj)
            object_index += 1

        return SceneState(
            frame_timestamp=frame.timestamp,
            objects=objects,
            extras={"num_detections": len(detections)},
        )

    def summarize_for_llm(self, scene_state: SceneState) -> dict[str, Any]:
        return {
            "frame_timestamp": scene_state.frame_timestamp,
            "object_count": len(scene_state.objects),
            "objects": [
                {
                    "object_id": obj.object_id,
                    "label": obj.label,
                    "color": obj.color,
                    "shape": obj.shape,
                    "is_graspable": obj.is_graspable,
                    "center_base_m": [round(float(value), 3) for value in obj.center_base.tolist()],
                }
                for obj in scene_state.objects
            ],
        }
