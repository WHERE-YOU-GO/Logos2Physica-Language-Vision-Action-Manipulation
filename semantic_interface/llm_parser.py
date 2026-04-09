from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.config_loader import load_yaml
from common.exceptions import ProjectError, TargetResolutionError
from common.logger import ProjectLogger
from semantic_interface.command_schema import (
    ActionType,
    ObjectQuery,
    ParsedCommand,
    SpatialRelation,
)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


class LLMCommandParser:
    def __init__(self, api_config_path: str, logger: ProjectLogger) -> None:
        self._api_config_path = str(Path(api_config_path))
        self._logger = logger
        self._config = self._load_config()
        self._client: Any | None = None

    def _load_config(self) -> dict[str, Any]:
        try:
            return load_yaml(self._api_config_path)
        except ProjectError as exc:
            self._logger.warn(f"LLM parser config is unavailable: {exc}")
            return {}

    def _api_section(self) -> dict[str, Any]:
        candidate = self._config.get("llm", self._config)
        return candidate if isinstance(candidate, dict) else {}

    def build_messages(self, prompt: str, scene_summary: dict | None = None) -> list[dict]:
        system_prompt = (
            "You are a robotics command parser. "
            "Output a strict JSON object with keys: action, source, target, relation, grasp_mode. "
            "The action must be 'pick_and_place'. "
            "Source and target may only contain raw_text, category, color, shape. "
            "Do not output xyz, coordinates, joint angles, or full poses."
        )
        user_payload: dict[str, Any] = {"prompt": prompt}
        if scene_summary is not None:
            user_payload["scene_summary"] = scene_summary

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

    def _ensure_configured(self) -> dict[str, Any]:
        section = self._api_section()
        provider = str(section.get("provider", "")).strip()
        api_key = str(section.get("api_key", "")).strip()
        model = str(section.get("model", "")).strip()
        if not provider or not api_key or not model:
            raise RuntimeError(
                "LLM API is not configured. Provide provider, api_key, and model in the parser config."
            )
        return section

    def _get_client(self, section: dict[str, Any]):
        if self._client is not None:
            return self._client
        if OpenAI is None:
            raise RuntimeError(
                "The openai package is required for OpenAI-compatible LLM parsing backends."
            )

        base_url = str(section.get("base_url", "")).strip() or None
        self._client = OpenAI(api_key=section["api_key"], base_url=base_url)
        return self._client

    def parse(self, prompt: str, scene_summary: dict | None = None, image=None) -> ParsedCommand:
        _ = image
        section = self._ensure_configured()
        client = self._get_client(section)
        messages = self.build_messages(prompt, scene_summary=scene_summary)

        request_kwargs = {
            "model": section["model"],
            "messages": messages,
            "temperature": float(section.get("temperature", 0.0)),
        }
        if bool(section.get("json_mode", True)):
            request_kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise RuntimeError("LLM parser returned an empty response.")

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise TargetResolutionError("LLM parser response is not valid JSON.") from exc
        return self._validate_llm_json(payload)

    def _validate_object_query(self, payload: dict[str, Any] | None) -> ObjectQuery | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise TargetResolutionError("LLM object query payload must be a JSON object.")

        allowed_keys = {"raw_text", "category", "color", "shape"}
        unexpected = set(payload) - allowed_keys
        forbidden = {"xyz", "joint", "joints", "pose", "position", "quaternion"}
        if unexpected & forbidden:
            raise TargetResolutionError("LLM output contains forbidden geometry fields.")
        if unexpected:
            raise TargetResolutionError(
                f"LLM output contains unexpected object-query fields: {sorted(unexpected)}"
            )

        raw_text = str(payload.get("raw_text", "")).strip()
        if not raw_text:
            raise TargetResolutionError("LLM object query must include raw_text.")
        return ObjectQuery(
            raw_text=raw_text,
            category=(str(payload["category"]).strip() or None)
            if "category" in payload and payload["category"] is not None
            else None,
            color=(str(payload["color"]).strip() or None)
            if "color" in payload and payload["color"] is not None
            else None,
            shape=(str(payload["shape"]).strip() or None)
            if "shape" in payload and payload["shape"] is not None
            else None,
        )

    def _validate_llm_json(self, payload: dict) -> ParsedCommand:
        if not isinstance(payload, dict):
            raise TargetResolutionError("LLM output must be a JSON object.")

        allowed_keys = {"action", "source", "target", "relation", "grasp_mode", "raw_prompt"}
        unexpected = set(payload) - allowed_keys
        forbidden = {"xyz", "joint", "joints", "pose", "position", "quaternion"}
        if unexpected & forbidden:
            raise TargetResolutionError("LLM output contains forbidden geometry fields.")
        if unexpected:
            raise TargetResolutionError(f"LLM output contains unexpected top-level fields: {sorted(unexpected)}")

        action_raw = str(payload.get("action", "")).strip().lower()
        if action_raw != ActionType.PICK_AND_PLACE.value:
            raise TargetResolutionError(f"Unsupported action from LLM parser: {action_raw!r}")

        relation_raw = payload.get("relation")
        relation = None
        if relation_raw is not None:
            try:
                relation = SpatialRelation(str(relation_raw).strip().lower())
            except ValueError as exc:
                raise TargetResolutionError(f"Unsupported relation from LLM parser: {relation_raw!r}") from exc

        source = self._validate_object_query(payload.get("source"))
        if source is None:
            raise TargetResolutionError("LLM parser did not return a source object query.")
        target = self._validate_object_query(payload.get("target"))
        grasp_mode = str(payload.get("grasp_mode", "topdown")).strip() or "topdown"

        return ParsedCommand(
            action=ActionType.PICK_AND_PLACE,
            source=source,
            target=target,
            relation=relation,
            grasp_mode=grasp_mode,
            raw_prompt=str(payload.get("raw_prompt", source.raw_text)),
        )
