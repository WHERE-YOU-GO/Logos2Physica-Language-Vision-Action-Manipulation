from __future__ import annotations

import numpy as np
import pytest

from common.datatypes import BBox2D, SceneObject, SceneState
from common.exceptions import DetectionNotFoundError
from common.logger import ProjectLogger
from semantic_interface.command_schema import ActionType, ObjectQuery, ParsedCommand
from semantic_interface.target_resolver import TargetResolver


def _make_scene() -> SceneState:
    return SceneState(
        frame_timestamp=0.0,
        objects=[
            SceneObject(
                object_id="obj_1",
                label="block",
                bbox=BBox2D(10, 10, 40, 40),
                center_cam=np.array([0.1, 0.0, 0.5]),
                center_base=np.array([0.2, 0.0, 0.04]),
                confidence=0.8,
                color="red",
                shape="cube",
                extras={"category": "cube"},
            ),
            SceneObject(
                object_id="obj_2",
                label="block",
                bbox=BBox2D(50, 10, 80, 40),
                center_cam=np.array([0.2, 0.0, 0.5]),
                center_base=np.array([0.4, 0.0, 0.04]),
                confidence=0.95,
                color="red",
                shape="cube",
                extras={"category": "cube"},
            ),
            SceneObject(
                object_id="obj_3",
                label="block",
                bbox=BBox2D(90, 10, 120, 40),
                center_cam=np.array([0.3, 0.0, 0.5]),
                center_base=np.array([0.5, 0.1, 0.04]),
                confidence=0.9,
                color="blue",
                shape="block",
                extras={"category": "block"},
            ),
        ],
    )


def test_resolve_unique_target() -> None:
    resolver = TargetResolver(ProjectLogger("logs/test_target_resolver"))
    cmd = ParsedCommand(
        action=ActionType.PICK_AND_PLACE,
        source=ObjectQuery(raw_text="blue block", category="block", color="blue", shape="block"),
        target=None,
        relation=None,
        grasp_mode="topdown",
        raw_prompt="pick up the blue block",
    )
    resolved = resolver.resolve(cmd, _make_scene())
    assert resolved.source_id == "obj_3"


def test_resolve_prefers_higher_confidence_when_multiple_candidates() -> None:
    resolver = TargetResolver(ProjectLogger("logs/test_target_resolver"))
    cmd = ParsedCommand(
        action=ActionType.PICK_AND_PLACE,
        source=ObjectQuery(raw_text="red cube", category="cube", color="red", shape="cube"),
        target=None,
        relation=None,
        grasp_mode="topdown",
        raw_prompt="pick up the red cube",
    )
    resolved = resolver.resolve(cmd, _make_scene())
    assert resolved.source_id == "obj_2"


def test_resolve_raises_when_not_found() -> None:
    resolver = TargetResolver(ProjectLogger("logs/test_target_resolver"))
    cmd = ParsedCommand(
        action=ActionType.PICK_AND_PLACE,
        source=ObjectQuery(raw_text="green cube", category="cube", color="green", shape="cube"),
        target=None,
        relation=None,
        grasp_mode="topdown",
        raw_prompt="pick up the green cube",
    )
    with pytest.raises(DetectionNotFoundError):
        resolver.resolve(cmd, _make_scene())
