from __future__ import annotations

from enum import Enum


class RetryDecision(str, Enum):
    RECAPTURE = "recapture"
    REDETECT = "redetect"
    REPARSE = "reparse"
    REPLAN = "replan"
    ABORT = "abort"


class RetryPolicy:
    def __init__(self, max_attempts: int = 3) -> None:
        self._max_attempts = max(1, int(max_attempts))

    def decide(self, failure_stage: str, reason: str, attempt_idx: int) -> RetryDecision:
        _ = reason
        if attempt_idx >= self._max_attempts:
            return RetryDecision.ABORT

        stage = failure_stage.strip().lower()
        if stage in {"sense_scene", "verify_grasp", "verify_place"}:
            return RetryDecision.RECAPTURE
        if stage in {"detect_objects", "build_scene_state", "resolve_targets"}:
            return RetryDecision.REDETECT
        if stage in {"parse_command"}:
            return RetryDecision.REPARSE
        if stage in {"plan", "safety_check", "execute_pick", "execute_place"}:
            return RetryDecision.REPLAN
        return RetryDecision.ABORT
