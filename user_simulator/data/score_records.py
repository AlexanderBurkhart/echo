"""Typed request records and cache identities for simulator scoring."""

from __future__ import annotations

import dataclasses
import enum
import json
import pathlib

from conversation import history
from user_simulator import config
from user_simulator.data import judge_identity
from user_simulator.data import judgments
from user_simulator.data import preference_scores
from user_simulator.judge import response_schema


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
    prompt: str | None = None
    response_schema: response_schema.ResponseSchema | None = None
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
    judge: judge_identity.LocalJudgeIdentity
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
    task_lm_calls: list[LMCallResult] = dataclasses.field(default_factory=list)
    output_lm_calls: list[LMCallResult] = dataclasses.field(
        default_factory=list
    )
    result: preference_scores.ScoreResult | None = None
    error: str | None = None
    elapsed_seconds: float | None = None


def load_score_record(path: pathlib.Path) -> ScoreRecord:
    """Decodes current JSON into typed records at the file input boundary."""
    value = json.loads(path.read_text(encoding="utf-8"))
    value.pop("_record", None)
    value["status"] = ScoreStatus(value["status"])
    value["judge"] = judge_identity.LocalJudgeIdentity(**value["judge"])
    for field in ("task_lm_calls", "output_lm_calls"):
        for call in value[field]:
            schema = call["response_schema"]
            if schema is not None:
                call["response_schema"] = response_schema.ResponseSchema(
                    ratings=tuple(
                        response_schema.RatingSchema(**item)
                        for item in schema["ratings"]
                    ),
                    quantities=tuple(
                        response_schema.RatingSchema(**item)
                        for item in schema["quantities"]
                    ),
                    require_feedback=schema["require_feedback"],
                )
        value[field] = [LMCallResult(**call) for call in value[field]]
    value["history"] = history.ConversationHistory(messages=tuple(
        history.Message(
            role=history.Role(message["role"]), content=message["content"]
        ) for message in value["history"]["messages"]
    ))
    value["user_config"] = config.parse_user(value["user_config"])
    for field in ("task_judgment", "output_judgment"):
        judgment = value[field]
        if judgment is not None:
            value[field] = judgments.LMJudgment(
                ratings={
                    name: judgments.Rating(**item)
                    for name, item in judgment["ratings"].items()
                },
                quantities={
                    name: judgments.Rating(**item)
                    for name, item in judgment["quantities"].items()
                },
                source_record=judgment["source_record"],
            )
    if value["result"] is not None:
        value["result"] = preference_scores.ScoreResult(**value["result"])
    return ScoreRecord(**value)
