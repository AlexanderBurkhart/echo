"""Typed request records and cache identities for simulator scoring."""

from collections.abc import Mapping
import dataclasses
import enum
import json
import pathlib
from typing import Any

from conversation import history
from user_simulator import config
from user_simulator.data import judge_identity
from user_simulator.data import judgments
from user_simulator.data import preference_scores
from user_simulator.judge.ratings import ratings as rating_scale


class ScoreStatus(enum.StrEnum):
    """Execution states of a complete simulator scoring request."""

    RUNNING = "running"
    OK = "ok"
    FAILURE = "failure"


@dataclasses.dataclass(frozen=True)
class JudgeNamespace:
    """The frozen judge and prompt identity defining a cache namespace."""

    judge: judge_identity.LocalJudgeIdentity
    prompts: str
    cache_schema_version: str = "lm-judgments-v3"


@dataclasses.dataclass(frozen=True)
class LMCacheKey:
    """A task or output request; output=None identifies the task judgment."""

    namespace: str
    history: history.ConversationHistory
    output: str | None = None


@dataclasses.dataclass
class LMCallResult:
    """One LM call's response or error, stored inside the score record."""

    raw: str | None = None
    error: str | None = None
    elapsed_seconds: float | None = None


@dataclasses.dataclass
class ScoreRecord:
    """One scoring request and all available diagnostics at its completion.

    Attributes:
        task_cache_hit: Whether cached LM judgments of the general task
            conditions were reused to select the user's preference weights.
        output_cache_hit: Whether cached LM ratings of the output against the
            fixed preference rubrics were reused to calculate the user's score
            deterministically with those weights.

    Cache-hit flags are None until that stage starts, False on a miss,
    invalid entry, or explicit bypass, and True when a valid saved LM judgment
    is reused. Cache errors and invalid contents are saved in this record.
    LM call results appear in execution order under their task or output prompt.
    """

    user_id: str
    history: history.ConversationHistory
    output: str
    # Older archived test runs recorded backend-specific identity maps.
    judge: judge_identity.LocalJudgeIdentity | Mapping[str, Any]
    prompt_hash: str
    user_config: config.UserConfig
    task_cache_key: str
    timestamp: str
    output_cache_key: str | None = None
    output_cache_hit: bool | None = None
    output_cache_error: str | None = None
    output_invalid_cache: str | None = None
    status: ScoreStatus = ScoreStatus.RUNNING
    task_judgment: judgments.LMJudgment | None = None
    task_cache_hit: bool | None = None
    task_cache_error: str | None = None
    task_invalid_cache: str | None = None
    output_judgment: judgments.LMJudgment | None = None
    task_prompt: str | None = None
    output_prompt: str | None = None
    task_lm_calls: list[LMCallResult] = dataclasses.field(default_factory=list)
    output_lm_calls: list[LMCallResult] = dataclasses.field(
        default_factory=list
    )
    result: preference_scores.ScoreResult | None = None
    error: str | None = None
    elapsed_seconds: float | None = None


def load_score_record(path: pathlib.Path) -> ScoreRecord:
    """Decodes a persisted scoring record at the JSON input boundary."""
    value = json.loads(path.read_text(encoding="utf-8"))
    value["status"] = ScoreStatus(value["status"])
    if "model_sha256" in value["judge"]:
        value["judge"] = judge_identity.LocalJudgeIdentity(**value["judge"])
    for field in ("task_lm_calls", "output_lm_calls"):
        value[field] = [LMCallResult(**call) for call in value.get(field, [])]
    value.pop("_record", None)
    # Older records included a redundant user-configuration hash.
    value.pop("config_hash", None)
    value["history"] = history.ConversationHistory(messages=tuple(
        history.Message(
            role=history.Role(message["role"]), content=message["content"]
        )
        for message in value["history"]["messages"]
    ))
    value["user_config"] = config.parse_user(value["user_config"])
    # Accept the earlier nested JSON fields when reading archived records.
    task = value.pop("task", None) or value.get("task_judgment")
    if task is not None:
        parsed = task.get("assessment", task)
        ratings = parsed.get("ratings", parsed.get("measurements"))
        value["task_judgment"] = judgments.LMJudgment(
            ratings={
                name: judgments.Rating(**item)
                for name, item in ratings.items()
            },
            source_record=task["source_record"],
        )
    output = (
        value.pop("output_assessment", None) or value.get("output_judgment")
    )
    if output is not None:
        if "assessment" in output:
            parsed = output["assessment"]
            ratings = parsed["assessments"]
        else:
            parsed = output
            ratings = output["ratings"]
        # Archived output ratings used level without a unit. Normalize only
        # at this file boundary; current LM responses must use value and unit.
        for item in ratings.values():
            if "level" in item:
                item["value"] = item.pop("level")
                item["unit"] = rating_scale.OUTPUT_RATING_UNIT
        value["output_judgment"] = judgments.LMJudgment(
            ratings={
                name: judgments.Rating(**item)
                for name, item in ratings.items()
            },
            quantities={
                name: judgments.Rating(**item)
                for name, item in parsed["quantities"].items()
            },
            source_record=output["source_record"],
        )
    if value.get("result") is not None:
        result = value["result"]
        if "cost_weights" in result:
            # Preserve archived scores while translating their weight layout.
            result.pop("cost_weights")
            result["selected_rule"] = result["selected_rule"] or "always"
            selected = next(
                rule for rule in value["user_config"].rules
                if rule.id == result["selected_rule"]
            )
            result["weights"] = {
                name: weight for name, weight in selected.weights.items()
                if weight > 0
            }
            result["normalized_weights"] = config.normalize_weights(
                result["weights"]
            )
            result["component_values"] = {
                name: result["component_values"][name]
                for name in result["weights"]
            }
            result["unknown_components"] = [
                name for name in result["unknown_components"]
                if name in result["weights"]
            ]
            if "always" not in result["matching_rules"]:
                result["matching_rules"].insert(0, "always")
        value["result"] = preference_scores.ScoreResult(**result)
    return ScoreRecord(**value)
