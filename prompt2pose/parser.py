"""Natural-language command parser using an OpenAI-compatible LLM API.

The parser turns a free-form instruction such as
    "Put the red cube on the blue block"
into a strict JSON dict:
    {"action": "pick_and_place",
     "pick_object": "red cube",
     "place_target": "blue block",
     "place_relation": "on"}

Backends supported via a single OpenAI-style client (set base_url + model):
  - OpenAI       (gpt-4o-mini, gpt-4.1, ...)
  - Ollama       (base_url="http://localhost:11434/v1", model="qwen2.5", ...)
  - OpenRouter   (gives access to Claude, Gemini, etc. through one endpoint)
  - vLLM / LiteLLM proxy / any chat-completions-compatible server
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


SYSTEM_PROMPT = """You are a robot command parser. Convert the user's natural-language
instruction into a STRICT JSON object with EXACTLY these keys:

{
  "action":         "pick_and_place",
  "pick_object":    "<short noun phrase the robot should grasp>",
  "place_target":   "<short noun phrase to place onto/at, or null>",
  "place_relation": "on" | "in" | "to" | "at"
}

Rules:
- Always output valid JSON, no commentary, no markdown fences.
- pick_object is required and must be a concrete object the camera can see
  (e.g. "red cube", "blue block", "yellow brick").
- place_target may be null if the user only asked to pick something up.
- place_relation defaults to "on" when stacking, "in" for containers, "at" for
  free locations, and "to" for goal areas.
- Do NOT output coordinates, joint angles, or robot poses.

Examples:
"Put the red cube on the blue block"
-> {"action":"pick_and_place","pick_object":"red cube","place_target":"blue block","place_relation":"on"}

"Pick up the green block"
-> {"action":"pick_and_place","pick_object":"green block","place_target":null,"place_relation":"at"}

"Drop the yellow cube into the basket"
-> {"action":"pick_and_place","pick_object":"yellow cube","place_target":"basket","place_relation":"in"}
"""


@dataclass
class ParsedCommand:
    action: str
    pick_object: str
    place_target: Optional[str]
    place_relation: str
    raw_prompt: str

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "pick_object": self.pick_object,
            "place_target": self.place_target,
            "place_relation": self.place_relation,
            "raw_prompt": self.raw_prompt,
        }


class LLMCommandParser:
    """Thin wrapper around the OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def parse(self, prompt: str) -> ParsedCommand:
        if not prompt or not prompt.strip():
            raise ValueError("prompt is empty")

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt.strip()},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned an empty response")

        payload = json.loads(content)
        return self._validate(payload, prompt)

    @staticmethod
    def _validate(payload: dict, raw_prompt: str) -> ParsedCommand:
        action = str(payload.get("action", "")).strip().lower()
        if action != "pick_and_place":
            raise ValueError(f"Unsupported action: {action!r}")

        pick = payload.get("pick_object")
        if not isinstance(pick, str) or not pick.strip():
            raise ValueError("pick_object must be a non-empty string")

        place = payload.get("place_target")
        if place is not None and not isinstance(place, str):
            raise ValueError("place_target must be a string or null")
        if isinstance(place, str) and not place.strip():
            place = None

        relation = str(payload.get("place_relation", "on")).strip().lower()
        if relation not in {"on", "in", "to", "at"}:
            relation = "on"

        return ParsedCommand(
            action=action,
            pick_object=pick.strip(),
            place_target=place.strip() if isinstance(place, str) else None,
            place_relation=relation,
            raw_prompt=raw_prompt,
        )
