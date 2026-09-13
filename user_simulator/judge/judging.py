"""Validates LM judgment structure and quotes, not measurement quality."""

from collections.abc import Iterable, Mapping, Sequence
import json
from typing import Any

from user_simulator import config
from user_simulator.data import judgments
from user_simulator.judge.ratings import output_definitions
from user_simulator.judge.ratings import task_definitions
from user_simulator.judge.ratings import ratings as rating_scale


class MalformedLMJudgmentError(ValueError):
    """An LM judgment that violates the schema or evidence-quote contract."""


def _check_keys(value: Any, keys: Iterable[str]) -> None:
    expected = frozenset(keys)
    if not isinstance(value, dict) or set(value) != expected:
        raise MalformedLMJudgmentError(
            f"Expected exactly keys {sorted(expected)!r}"
        )


def _check_evidence(
    item: Mapping[str, Any], sources: Sequence[str], known: bool
) -> None:
    """Checks that known measurements have nonempty, verbatim evidence."""
    evidence = item["evidence"]
    if not isinstance(evidence, list) or (known and not evidence):
        raise MalformedLMJudgmentError(
            "Known measurements need evidence quotes"
        )
    for quote in evidence:
        if not isinstance(quote, str) or not quote.strip():
            raise MalformedLMJudgmentError("Evidence quotes must be nonempty")
        if not any(quote in source for source in sources):
            raise MalformedLMJudgmentError("Evidence must quote supplied data")


def _check_quantity(
    item: Mapping[str, Any],
    sources: Sequence[str],
    qualitative: bool = False,
) -> None:
    """Checks finite quantities, unknown markers, units, and evidence."""
    _check_keys(item, ("value", "unit", "evidence"))
    value = item["value"]
    unit = item["unit"]
    if value is not None:
        if not config.is_number(value) or value < 0:
            raise MalformedLMJudgmentError("Invalid measurement value")
        if qualitative and value > 100:
            raise MalformedLMJudgmentError(
                "Qualitative measurement exceeds 100"
            )
    if qualitative:
        if unit != task_definitions.TASK_RATING_UNIT:
            raise MalformedLMJudgmentError("Expected rating_0_100 unit")
    elif value is None:
        if unit is not None:
            raise MalformedLMJudgmentError(
                "Unknown quantity must have null unit"
            )
    elif not isinstance(unit, str) or not unit.strip():
        raise MalformedLMJudgmentError(
            "Known quantity requires an explicit unit"
        )
    _check_evidence(item, sources, value is not None)


def _build_unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    """Builds a JSON object while rejecting ambiguous duplicate keys."""
    result = {}
    for key, value in items:
        if key in result:
            raise MalformedLMJudgmentError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise MalformedLMJudgmentError(f"Nonfinite JSON number: {value!r}")


def decode_lm_json(raw: str) -> dict[str, Any]:
    """Decodes fresh or cached LM JSON before fields can be discarded.

    Raises:
        MalformedLMJudgmentError: The response is not a JSON object, contains
            duplicate keys, or contains nonfinite JSON constants.
    """
    try:
        data = json.loads(
            raw,
            object_pairs_hook=_build_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, TypeError) as error:
        raise MalformedLMJudgmentError(str(error)) from error
    if not isinstance(data, dict):
        raise MalformedLMJudgmentError("LM judgment must be an object")
    return data


def parse_lm_judgment(
    raw: str,
    kind: judgments.JudgmentKind,
    sources: Sequence[str],
    rating_criteria: tuple[rating_scale.RatingAnchoredCriterion, ...],
) -> judgments.LMJudgment:
    """Returns a validated LM judgment without claiming semantic correctness.

    Args:
        raw: The judge's unmodified JSON response text.
        kind: JudgmentKind.TASK or JudgmentKind.OUTPUT.
        sources: Allowed evidence texts; history only for task LM judgments.
        rating_criteria: Rating definitions from the selected user config.

    Returns:
        The parsed task or output ratings with supporting quotations.

    Raises:
        MalformedLMJudgmentError: JSON, schema, ranges, or quotes are invalid.
    """
    data = decode_lm_json(raw)
    if kind is judgments.JudgmentKind.TASK:
        _check_keys(data, ("ratings", "quantities"))
        _check_keys(data["quantities"], ())
        _check_keys(data["ratings"], task_definitions.CONDITIONS)
        for name, item in data["ratings"].items():
            _check_quantity(
                item, sources, name in task_definitions.QUALITATIVE
            )
    elif kind is judgments.JudgmentKind.OUTPUT:
        _check_keys(data, ("ratings", "quantities"))
        _check_keys(data["ratings"], (item.name for item in rating_criteria))
        for item in data["ratings"].values():
            _check_keys(item, ("value", "unit", "evidence"))
            value = item["value"]
            if item["unit"] != rating_scale.OUTPUT_RATING_UNIT:
                raise MalformedLMJudgmentError("Expected rating_0_4 unit")
            if value is not None:
                if (not config.is_number(value)
                        or value not in rating_scale.RATING_LEVELS):
                    raise MalformedLMJudgmentError(
                        "Output rating must be a whole value from 0–4 or null"
                    )
            _check_evidence(item, sources, value is not None)
        _check_keys(data["quantities"], output_definitions.COST_COMPONENTS)
        for item in data["quantities"].values():
            _check_quantity(item, sources)
    else:
        raise ValueError(f"Unknown LM judgment kind: {kind!r}")
    return judgments.LMJudgment(
        ratings={
            name: judgments.Rating(**item)
            for name, item in data["ratings"].items()
        },
        quantities={
            name: judgments.Rating(**item)
            for name, item in data["quantities"].items()
        },
    )
