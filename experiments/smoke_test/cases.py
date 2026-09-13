"""Loads frozen development or holdout cases for the real-LM smoke test."""

from collections.abc import Mapping
import dataclasses
import enum
import hashlib
import json
import pathlib
import types

from conversation import history
from user_simulator import config

DATA_DIRECTORY = pathlib.Path(__file__).with_name("data")
USERS_PER_SPLIT = 4
CASES_PER_USER = 5


class Split(enum.StrEnum):
    """The development set and the separately gated holdout set."""

    DEVELOPMENT = "development"
    HOLDOUT = "holdout"


@dataclasses.dataclass(frozen=True)
class SmokeCase:
    """One input and its authored expectations, never sent to the LM."""

    name: str
    history: history.ConversationHistory
    output: str
    expected_rule: str
    expected_ratings: Mapping[str, int]
    explanation: str

    def __post_init__(self) -> None:
        """Freezes expected ratings for the duration of an experiment."""
        object.__setattr__(self, "expected_ratings", types.MappingProxyType(
            dict(self.expected_ratings)
        ))


@dataclasses.dataclass(frozen=True)
class UserCases:
    """A test-only user configuration and five separately scored outputs."""

    user: config.UserConfig
    description: str
    cases: tuple[SmokeCase, ...]

    def __post_init__(self) -> None:
        """Rejects incomplete case groups and invalid expected ratings."""
        if len(self.cases) != CASES_PER_USER:
            raise ValueError("Each test user requires five cases")
        if len({case.name for case in self.cases}) != CASES_PER_USER:
            raise ValueError("Case names must be unique within a user")
        rules = {rule.id: rule for rule in self.user.rules}
        for case in self.cases:
            rule = rules[case.expected_rule]
            active = {name for name, weight in rule.weights.items() if weight}
            if set(case.expected_ratings) != active:
                raise ValueError("Expectations must cover active preferences")
            if any(type(value) is not int or value not in range(5)
                   for value in case.expected_ratings.values()):
                raise ValueError("Expected ratings must be integers in [0, 4]")
            if not case.output.strip() or not case.explanation.strip():
                raise ValueError("Cases require an output and an explanation")


def load_cases(split: Split) -> tuple[UserCases, ...]:
    """Loads only the requested split; callers own the holdout release gate."""
    raw = (DATA_DIRECTORY / f"{split}.json").read_bytes()
    manifest = json.loads((DATA_DIRECTORY / "manifest.json").read_text())
    if hashlib.sha256(raw).hexdigest() != manifest[split]:
        raise ValueError("Evaluation fixtures changed after they were frozen")
    raw_groups = json.loads(raw)
    groups = []
    for group in raw_groups:
        inputs = []
        for item in group["cases"]:
            inputs.append(SmokeCase(
                name=item["name"],
                history=history.ConversationHistory(messages=tuple(
                    history.Message(
                        role=history.Role(message["role"]),
                        content=message["content"],
                    ) for message in item["history"]
                )),
                output=item["output"],
                expected_rule=item["expected_rule"],
                expected_ratings=item["expected_ratings"],
                explanation=item["explanation"],
            ))
        groups.append(UserCases(
            user=config.parse_user(group["user"]),
            description=group["description"], cases=tuple(inputs),
        ))
    if len(groups) != USERS_PER_SPLIT:
        raise ValueError("Each split requires four users")
    if len({group.user.user_id for group in groups}) != USERS_PER_SPLIT:
        raise ValueError("User IDs must be unique within a split")
    return tuple(groups)
