"""Builds shared synthetic fixtures for simulator tests."""

import dataclasses
import pathlib

from experiment_logging import records

from user_simulator import config
from user_simulator.data import judge_identity
from user_simulator.data import judgments
from user_simulator.judge.ratings import output_definitions
from user_simulator.judge.ratings import ratings as rating_scale

CASES_PATH = (
    pathlib.Path(__file__).resolve().parent
    / "integration_tests/judge_cases.json"
)


def make_judge_identity(
    backend: str = "test-double", revision: str = "fixture-v1"
) -> judge_identity.LocalJudgeIdentity:
    """Builds an explicit identity for a mocked local judge."""
    return judge_identity.LocalJudgeIdentity(
        backend=backend,
        binding_version="test",
        runtime_revision=revision,
        runtime_files_sha256={},
        model_path="test.gguf",
        model_revision="fixture-v1",
        model_sha256="test-model-digest",
        model_metadata={},
        settings={},
        generation={},
        platform="test-platform",
        machine="test-machine",
    )


def make_task(
    value: float | None = None, unit: str = "rating_0_100"
) -> dict[str, judgments.Rating]:
    """Builds minimal measurements for conditional scoring tests."""
    return {
        "urgency": judgments.Rating(value, unit),
        "uncertainty": judgments.Rating(None, "rating_0_100"),
        "user_cost": judgments.Rating(None, "rating_0_100"),
    }


def make_ratings(
    urgency: int | None = 2,
    uncertainty: int | None = 2,
    money: int | None = 2,
    user_time: int | None = 2,
    effort: int | None = 2,
) -> dict[str, judgments.Rating]:
    """Builds separate qualitative assessments for every pilot component."""
    levels = (urgency, uncertainty, money, user_time, effort)
    return {
        criterion.name: judgments.Rating(
            level, rating_scale.OUTPUT_RATING_UNIT
        )
        for criterion, level in zip(
            output_definitions.DEFAULT_RATING_CRITERIA, levels
        )
    }


def make_conditional_user() -> config.UserConfig:
    """Builds a user whose urgency rule reverses a cost tradeoff."""
    return config.UserConfig(
        user_id="pilot_00",
        rules=(
            config.PreferenceRule(
                id="urgent", precedence=1,
                when=config.Condition(
                    property="urgency",
                    operator=config.ConditionOperator.GREATER,
                    threshold=80,
                ),
                weights={"urgency": 0.8, "uncertainty": 0.1, "money": 0.1},
            ),
            config.PreferenceRule(
                id="always", precedence=0, when=None,
                weights={"urgency": 0.1, "uncertainty": 0.1, "money": 0.8},
            ),
        ),
    )


def serialize_lm_response(judgment: judgments.LMJudgment) -> str:
    """Encodes authored LM ratings without score-record provenance."""
    payload = dataclasses.asdict(judgment)
    payload.pop("source_record")
    return records.serialize_canonical_json(payload)
