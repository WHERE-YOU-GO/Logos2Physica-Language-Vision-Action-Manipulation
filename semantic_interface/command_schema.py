from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    PICK_AND_PLACE = "pick_and_place"


class SpatialRelation(str, Enum):
    ON = "on"
    TO = "to"
    IN = "in"


@dataclass
class ObjectQuery:
    raw_text: str
    category: str | None = None
    color: str | None = None
    shape: str | None = None


@dataclass
class ParsedCommand:
    action: ActionType
    source: ObjectQuery
    target: ObjectQuery | None
    relation: SpatialRelation | None
    grasp_mode: str
    raw_prompt: str


@dataclass
class ResolvedCommand:
    parsed: ParsedCommand
    source_id: str
    target_id: str | None = None
