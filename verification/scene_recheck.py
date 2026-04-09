from __future__ import annotations

from common.datatypes import SceneState
from common.logger import ProjectLogger


class SceneRechecker:
    def __init__(self, frame_provider, detector, scene_builder, logger: ProjectLogger) -> None:
        self._frame_provider = frame_provider
        self._detector = detector
        self._scene_builder = scene_builder
        self._logger = logger

    def reacquire_scene(self, candidate_labels: list[str]) -> SceneState:
        frame = self._frame_provider.get_current_frame()
        detections = self._detector.detect(frame.rgb, candidate_labels)
        return self._scene_builder.build(frame, detections)

    def summarize(self, scene_state: SceneState) -> dict:
        return self._scene_builder.summarize_for_llm(scene_state)
