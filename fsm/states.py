from __future__ import annotations

from enum import Enum


class FSMState(str, Enum):
    IDLE = "idle"
    PARSE_COMMAND = "parse_command"
    SENSE_SCENE = "sense_scene"
    DETECT_OBJECTS = "detect_objects"
    BUILD_SCENE_STATE = "build_scene_state"
    RESOLVE_TARGETS = "resolve_targets"
    PLAN = "plan"
    SAFETY_CHECK = "safety_check"
    EXECUTE_PICK = "execute_pick"
    VERIFY_GRASP = "verify_grasp"
    EXECUTE_PLACE = "execute_place"
    VERIFY_PLACE = "verify_place"
    RETRY = "retry"
    SUCCESS = "success"
    FAIL = "fail"
