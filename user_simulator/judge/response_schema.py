"""Builds the exact JSON contract for one task or output-criterion call."""

from collections.abc import Sequence
import dataclasses
from typing import Any

from user_simulator.data import judgments
from user_simulator.judge.ratings import output_definitions
from user_simulator.judge.ratings import task_definitions
from user_simulator.judge.ratings import ratings as rating_scale


def _build_rating_schema(
    unit: str | None, maximum: int | None,
) -> dict[str, Any]:
    """Constrains units and ranges, with evidence generated before the value."""
    evidence = {
        "type": "array", "items": {"type": "string", "minLength": 1},
    }
    number = {"type": "integer" if maximum == 4 else "number", "minimum": 0}
    if maximum is not None:
        number["maximum"] = maximum
    return {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "evidence": {**evidence, "minItems": 1},
                    "value": number,
                    "unit": {"const": unit} if unit else {
                        "type": "string", "minLength": 1,
                    },
                },
                "required": ["evidence", "value", "unit"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "evidence": {**evidence, "maxItems": 0},
                    "value": {"type": "null"},
                    "unit": {"const": unit},
                },
                "required": ["evidence", "value", "unit"],
                "additionalProperties": False,
            },
        ],
    }


@dataclasses.dataclass(frozen=True)
class RatingSchema:
    """The name, unit, and upper bound of one LM rating or quantity."""

    name: str
    unit: str | None
    maximum: int | None


@dataclasses.dataclass(frozen=True)
class ResponseSchema:
    """The ordered rating fields and feedback required from one LM call.

    Tuples preserve generation order when the shared JSON logger sorts keys.
    The saved dataclass is sufficient to rebuild the exact native schema.
    """

    ratings: tuple[RatingSchema, ...]
    quantities: tuple[RatingSchema, ...]
    require_feedback: bool

    def __post_init__(self) -> None:
        """Freezes the ordered rating and quantity definitions."""
        object.__setattr__(self, "ratings", tuple(self.ratings))
        object.__setattr__(self, "quantities", tuple(self.quantities))

    def build_json_schema(self) -> dict[str, Any]:
        """Builds the native JSON Schema at the runtime or logging boundary."""
        schema = {
            "type": "object",
            "properties": {
                name: {
                    "type": "object", "properties": {
                        item.name: _build_rating_schema(item.unit, item.maximum)
                        for item in properties
                    },
                    "required": [item.name for item in properties],
                    "additionalProperties": False,
                } for name, properties in (
                    ("ratings", self.ratings), ("quantities", self.quantities)
                )
            },
            "required": ["ratings", "quantities"],
            "additionalProperties": False,
        }
        if self.require_feedback:
            schema["properties"]["feedback"] = {
                "type": "string", "minLength": 1, "maxLength": 400,
            }
            schema["required"].append("feedback")
        return schema


def build_response_schema(
    kind: judgments.JudgmentKind,
    criteria: Sequence[rating_scale.RatingAnchoredCriterion],
) -> ResponseSchema:
    """Returns the typed contract for a task or output-criterion call."""
    if kind is judgments.JudgmentKind.TASK:
        return ResponseSchema(
            ratings=tuple(
                RatingSchema(
                    name=name,
                    unit=(task_definitions.TASK_RATING_UNIT
                          if name in task_definitions.QUALITATIVE else None),
                    maximum=(100 if name in task_definitions.QUALITATIVE
                             else None),
                ) for name in task_definitions.CONDITIONS
            ),
            quantities=(),
            require_feedback=False,
        )
    return ResponseSchema(
        ratings=tuple(
            RatingSchema(criterion.name, rating_scale.OUTPUT_RATING_UNIT, 4)
            for criterion in criteria
        ),
        quantities=tuple(
            RatingSchema(criterion.name, None, None)
            for criterion in criteria
            if criterion.name in output_definitions.COST_COMPONENTS
        ),
        require_feedback=True,
    )
