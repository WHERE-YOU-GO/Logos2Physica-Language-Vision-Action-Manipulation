from __future__ import annotations

from common.logger import ProjectLogger
from semantic_interface.command_schema import SpatialRelation
from semantic_interface.regex_parser import RegexCommandParser


def test_parse_pick_prompt() -> None:
    parser = RegexCommandParser(ProjectLogger("logs/test_regex_parser"))
    parsed = parser.parse("pick up the red cube")
    assert parsed.source.color == "red"
    assert parsed.source.shape == "cube"
    assert parsed.relation is None


def test_parse_on_relation_prompt() -> None:
    parser = RegexCommandParser(ProjectLogger("logs/test_regex_parser"))
    parsed = parser.parse("put the red cube on the blue block")
    assert parsed.relation == SpatialRelation.ON
    assert parsed.target is not None
    assert parsed.target.color == "blue"


def test_parse_to_relation_prompt() -> None:
    parser = RegexCommandParser(ProjectLogger("logs/test_regex_parser"))
    parsed = parser.parse("move the green cube to the yellow area")
    assert parsed.relation == SpatialRelation.TO
    assert parsed.target is not None
    assert parsed.target.color == "yellow"
    assert parsed.target.category == "area"
