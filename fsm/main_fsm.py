from __future__ import annotations

from typing import Any

from common.logger import ProjectLogger
from fsm.states import FSMState
from semantic_interface.command_schema import ParsedCommand
from skill_planning.retry_policy import RetryDecision, RetryPolicy


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    log_fn = getattr(logger, level, None)
    if callable(log_fn):
        log_fn(message)


def _call_optional(component: Any, method_name: str, *args: Any) -> None:
    if component is None:
        return
    method = getattr(component, method_name, None)
    if callable(method):
        method(*args)


class Prompt2PoseFSM:
    def __init__(
        self,
        regex_parser,
        frame_provider,
        detector,
        scene_builder,
        target_resolver,
        pick_place_planner,
        safety_guardrail,
        motion_executor,
        gripper_executor,
        grasp_verifier,
        place_verifier,
        logger: ProjectLogger,
        llm_parser=None,
        scene_rechecker=None,
        retry_policy: RetryPolicy | None = None,
        max_attempts: int = 2,
    ) -> None:
        self._regex_parser = regex_parser
        self._llm_parser = llm_parser
        self._frame_provider = frame_provider
        self._detector = detector
        self._scene_builder = scene_builder
        self._target_resolver = target_resolver
        self._pick_place_planner = pick_place_planner
        self._safety_guardrail = safety_guardrail
        self._motion_executor = motion_executor
        self._gripper_executor = gripper_executor
        self._grasp_verifier = grasp_verifier
        self._place_verifier = place_verifier
        self._scene_rechecker = scene_rechecker
        self._retry_policy = retry_policy
        self._logger = logger
        self._max_attempts = max(1, int(max_attempts))

    def _record_state(self, state: FSMState, state_trace: list[str]) -> None:
        state_trace.append(state.value)
        _log(self._logger, "info", f"FSM state -> {state.value}")

    def _parse_prompt(self, prompt: str):
        try:
            return self._regex_parser.parse(prompt)
        except Exception as regex_exc:
            if self._llm_parser is None:
                raise regex_exc
            return self._llm_parser.parse(prompt)

    def _sense_scene(self, candidate_labels: list[str], state_trace: list[str]):
        self._record_state(FSMState.SENSE_SCENE, state_trace)
        frame = self._frame_provider.get_current_frame()

        self._record_state(FSMState.DETECT_OBJECTS, state_trace)
        detections = self._detector.detect(frame.rgb, candidate_labels)

        self._record_state(FSMState.BUILD_SCENE_STATE, state_trace)
        scene_state = self._scene_builder.build(frame, detections)
        return frame, detections, scene_state

    def _resense_for_verification(self, candidate_labels: list[str]):
        if self._scene_rechecker is not None:
            return self._scene_rechecker.reacquire_scene(candidate_labels)
        frame = self._frame_provider.get_current_frame()
        detections = self._detector.detect(frame.rgb, candidate_labels)
        return self._scene_builder.build(frame, detections)

    def _retry_decision(self, failure_stage: str, reason: str, attempt_idx: int) -> RetryDecision:
        if attempt_idx >= self._max_attempts:
            return RetryDecision.ABORT
        if self._retry_policy is None:
            return RetryDecision.REPLAN
        return self._retry_policy.decide(failure_stage, reason, attempt_idx)

    def run_once(self, prompt: str) -> dict:
        result = {
            "status": FSMState.FAIL.value,
            "failure_reason": None,
            "attempts": 0,
            "resolved_ids": {"source_id": None, "target_id": None},
            "states": [],
            "retry_decisions": [],
        }
        state_trace: list[str] = result["states"]

        self._record_state(FSMState.PARSE_COMMAND, state_trace)
        try:
            parsed_cmd = self._parse_prompt(prompt)
        except Exception as exc:
            self._record_state(FSMState.FAIL, state_trace)
            result["failure_reason"] = f"Command parsing failed: {exc}"
            return result

        candidate_labels = self._build_candidate_labels(parsed_cmd)
        last_failure_reason: str | None = None

        for attempt_idx in range(1, self._max_attempts + 1):
            result["attempts"] = attempt_idx
            current_stage = FSMState.SENSE_SCENE.value
            try:
                _, _, scene_before = self._sense_scene(candidate_labels, state_trace)

                current_stage = FSMState.RESOLVE_TARGETS.value
                self._record_state(FSMState.RESOLVE_TARGETS, state_trace)
                resolved_cmd = self._target_resolver.resolve(parsed_cmd, scene_before)
                _call_optional(self._scene_rechecker, "set_resolved_context", resolved_cmd, scene_before)
                result["resolved_ids"] = {
                    "source_id": resolved_cmd.source_id,
                    "target_id": resolved_cmd.target_id,
                }

                current_stage = FSMState.PLAN.value
                self._record_state(FSMState.PLAN, state_trace)
                plan = self._pick_place_planner.build(resolved_cmd, scene_before)
                _call_optional(self._scene_rechecker, "set_plan_context", resolved_cmd, plan, scene_before)

                current_stage = FSMState.SAFETY_CHECK.value
                self._record_state(FSMState.SAFETY_CHECK, state_trace)
                self._safety_guardrail.validate_pick_place_plan(plan)

                current_stage = FSMState.EXECUTE_PICK.value
                self._record_state(FSMState.EXECUTE_PICK, state_trace)
                pick_motion = plan.extras.get("pick_motion", plan.approach_plan)
                if pick_motion is None:
                    raise RuntimeError("Pick motion plan is missing.")
                self._motion_executor.execute_cartesian_plan(pick_motion)

                current_stage = FSMState.VERIFY_GRASP.value
                self._record_state(FSMState.VERIFY_GRASP, state_trace)
                scene_after_pick = self._resense_for_verification(candidate_labels)
                gripper_state = self._gripper_executor.get_state()
                grasp_ok, grasp_reason = self._grasp_verifier.verify(
                    scene_before=scene_before,
                    scene_after=scene_after_pick,
                    source_id=resolved_cmd.source_id,
                    gripper_state=gripper_state,
                )
                if not grasp_ok:
                    last_failure_reason = grasp_reason
                    decision = self._retry_decision(current_stage, grasp_reason, attempt_idx)
                    result["retry_decisions"].append({"stage": current_stage, "decision": decision.value})
                    if decision != RetryDecision.ABORT:
                        self._record_state(FSMState.RETRY, state_trace)
                        continue
                    self._record_state(FSMState.FAIL, state_trace)
                    result["failure_reason"] = grasp_reason
                    return result

                current_stage = FSMState.EXECUTE_PLACE.value
                self._record_state(FSMState.EXECUTE_PLACE, state_trace)
                place_motion = plan.extras.get("place_motion", plan.transfer_plan)
                if place_motion is None:
                    raise RuntimeError("Place motion plan is missing.")
                self._motion_executor.execute_cartesian_plan(place_motion)

                current_stage = FSMState.VERIFY_PLACE.value
                self._record_state(FSMState.VERIFY_PLACE, state_trace)
                scene_after_place = self._resense_for_verification(candidate_labels)
                place_ok, place_reason = self._place_verifier.verify(resolved_cmd, scene_after_place)
                if place_ok:
                    self._record_state(FSMState.SUCCESS, state_trace)
                    result["status"] = FSMState.SUCCESS.value
                    result["failure_reason"] = None
                    return result

                last_failure_reason = place_reason
                decision = self._retry_decision(current_stage, place_reason, attempt_idx)
                result["retry_decisions"].append({"stage": current_stage, "decision": decision.value})
                if decision != RetryDecision.ABORT:
                    self._record_state(FSMState.RETRY, state_trace)
                    continue
                self._record_state(FSMState.FAIL, state_trace)
                result["failure_reason"] = place_reason
                return result

            except Exception as exc:
                last_failure_reason = str(exc)
                decision = self._retry_decision(current_stage, str(exc), attempt_idx)
                result["retry_decisions"].append({"stage": current_stage, "decision": decision.value})
                if decision != RetryDecision.ABORT:
                    self._record_state(FSMState.RETRY, state_trace)
                    continue
                self._record_state(FSMState.FAIL, state_trace)
                result["failure_reason"] = str(exc)
                return result

        self._record_state(FSMState.FAIL, state_trace)
        result["failure_reason"] = last_failure_reason or "Unknown failure."
        return result

    def _build_candidate_labels(self, parsed_cmd: ParsedCommand) -> list[str]:
        labels: list[str] = []

        def add_label(value: str | None) -> None:
            if value is None:
                return
            normalized = value.strip().lower()
            if not normalized:
                return
            if normalized in {"area", "zone", "region"}:
                return
            if normalized not in labels:
                labels.append(normalized)

        add_label(parsed_cmd.source.category)
        add_label(parsed_cmd.source.shape)
        if parsed_cmd.target is not None:
            add_label(parsed_cmd.target.category)
            add_label(parsed_cmd.target.shape)
        for fallback in ["block", "cube"]:
            add_label(fallback)
        return labels
