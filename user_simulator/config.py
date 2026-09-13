"""Defines, validates, and loads synthetic user configurations."""

from collections.abc import Mapping
import dataclasses
import enum
import json
import math
import pathlib
import random
import types
from typing import Any

from user_simulator.judge.ratings import output_definitions
from user_simulator.judge.ratings import task_definitions
from user_simulator.judge.ratings import ratings as rating_scale

DEFAULT_SEED = 20260907
DEFAULT_USER_COUNT = 8
USERS_DIRECTORY = pathlib.Path("config/users")


class ConditionOperator(enum.StrEnum):
    """Supported comparisons for conditional preference weights."""

    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    EQUAL = "=="


@dataclasses.dataclass(frozen=True)
class Condition:
    """A condition on a task preference rating in the shared 0–100 scale."""

    property: str
    operator: ConditionOperator
    threshold: float

    def __post_init__(self) -> None:
        """Rejects a missing task property name."""
        _validate_nonempty_string(self.property, "Condition property")


@dataclasses.dataclass(frozen=True)
class PreferenceRule:
    """Direct output-preference weights; when=None always matches."""

    id: str
    precedence: int
    when: Condition | None
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        """Validates the rule ID and freezes the supplied weights."""
        _validate_nonempty_string(self.id, "Rule ID")
        object.__setattr__(self, "weights", types.MappingProxyType(
            dict(self.weights)
        ))


@dataclasses.dataclass(frozen=True)
class UserConfig:
    """A stationary user's private preferences and conditional rules."""

    user_id: str
    rules: tuple[PreferenceRule, ...]
    rating_criteria: tuple[rating_scale.RatingAnchoredCriterion, ...] = (
        output_definitions.DEFAULT_RATING_CRITERIA
    )

    def __post_init__(self) -> None:
        """Freezes collections and rejects an invalid user configuration.

        Every rule weight must reference a defined output rating criterion.
        Criteria omitted from a rule are allowed: that rule ignores them.

        Raises:
            ValueError: Criteria, rules, weights, or conditions are invalid.
        """
        object.__setattr__(self, "rules", tuple(self.rules))
        object.__setattr__(self, "rating_criteria", tuple(self.rating_criteria))

        _validate_nonempty_string(self.user_id, "user_id")
        self._validate_rating_criteria()
        self._validate_preference_rules()

    def _validate_rating_criteria(self) -> None:
        """Checks anchor definitions, unique names, and pilot coverage."""
        for criterion in self.rating_criteria:
            _validate_nonempty_string(criterion.name, "Rating criterion name")
            if len(criterion.anchors) != len(
                rating_scale.DEFAULT_RATING_VALUES
            ):
                raise ValueError("Rating anchors must match the value mapping")
            for index, anchor in enumerate(criterion.anchors):
                _validate_nonempty_string(anchor, f"Rating anchors[{index}]")
        criterion_names = {criterion.name for criterion in self.rating_criteria}
        if len(criterion_names) != len(self.rating_criteria):
            raise ValueError("Duplicate rating criteria")
        required = {
            criterion.name
            for criterion in output_definitions.DEFAULT_RATING_CRITERIA
        }
        if not required <= criterion_names:
            raise ValueError("Missing pilot output rating criteria")

    def _validate_preference_rules(self) -> None:
        """Checks coverage, criterion references, weights, and precedence."""
        criterion_names = {criterion.name for criterion in self.rating_criteria}
        if not any(rule.when is None for rule in self.rules):
            raise ValueError(
                "A user requires an always-applicable preference rule"
            )
        precedences = set()
        rule_ids = set()
        for rule in self.rules:
            _validate_weights(rule.weights)
            undefined = rule.weights.keys() - criterion_names
            if undefined:
                raise ValueError(
                    f"Rule {rule.id!r} references undefined rating criteria: "
                    f"{', '.join(sorted(undefined))}"
                )
            precedence = rule.precedence
            if type(precedence) is not int or precedence in precedences:
                raise ValueError("Rule precedences must be distinct integers")
            precedences.add(precedence)
            rule_id = rule.id
            if rule_id in rule_ids:
                raise ValueError("Rule IDs must be unique")
            rule_ids.add(rule_id)
            if rule.when is not None:
                _validate_condition(rule.when)


def is_number(value: object) -> bool:
    """Returns whether a value is finite numeric data, excluding booleans."""
    return type(value) in (int, float) and math.isfinite(value)


def _validate_nonempty_string(value: str, field_name: str) -> None:
    """Rejects nonstrings and blank text without changing the supplied value."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")


def _validate_weights(weights: Mapping[str, float]) -> None:
    """Validates finite nonnegative weights with a positive total."""
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("Weights must be a nonempty mapping")
    for name in weights:
        _validate_nonempty_string(name, "Preference name")
    if any(not is_number(value) or value < 0 for value in weights.values()):
        raise ValueError("Weights must be finite and nonnegative")
    if max(weights.values()) <= 0:
        raise ValueError("Weights must have a positive total")


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Returns normalized weights without overflowing a finite input sum."""
    _validate_weights(weights)
    scale = max(weights.values())
    total = math.fsum(value / scale for value in weights.values())
    return {name: (value / scale) / total for name, value in weights.items()}


def _validate_condition(condition: Condition) -> None:
    """Validates a condition on one of the pilot's 0–100 task ratings."""
    if not isinstance(condition.operator, ConditionOperator):
        raise ValueError("Unsupported condition operator")
    if condition.property not in task_definitions.QUALITATIVE:
        raise ValueError("Conditions require a qualitative task preference")
    threshold = condition.threshold
    if not is_number(threshold) or not 0 <= threshold <= 100:
        raise ValueError("Condition threshold must be finite and in [0, 100]")


def _sample_weights(
    generator: random.Random, names: tuple[str, ...]
) -> dict[str, float]:
    """Returns normalized independent positive uniform draws."""
    return normalize_weights({name: 1.0 - generator.random() for name in names})


def generate_users(
    seed: int = DEFAULT_SEED, count: int = DEFAULT_USER_COUNT
) -> list[UserConfig]:
    """Returns a list of stationary synthetic users.

    Args:
        seed: Saved configuration RNG seed, independent of training seeds.
        count: Positive number of users to generate.

    Returns:
        Each user has one always-applicable rule and three conditional rules.
        Every rule samples weights directly for all output criteria.
        Users are never rejected or resampled to force disagreement.
    """
    if type(seed) is not int or type(count) is not int or count <= 0:
        raise ValueError("Seed and positive user count must be integers")
    generator = random.Random(seed)
    criterion_names = tuple(
        criterion.name
        for criterion in output_definitions.DEFAULT_RATING_CRITERIA
    )
    users = []
    for index in range(count):
        rules = [PreferenceRule(
            id="always", precedence=0, when=None,
            weights=_sample_weights(generator, criterion_names),
        )]
        order = list(task_definitions.PREFERENCES)
        generator.shuffle(order)
        for precedence, name in enumerate(order, 1):
            rules.append(PreferenceRule(
                id=f"{name}_condition",
                precedence=precedence,
                when=Condition(
                    property=name,
                    operator=ConditionOperator.GREATER,
                    threshold=generator.uniform(0, 100),
                ),
                weights=_sample_weights(generator, criterion_names),
            ))
        user = UserConfig(
            user_id=f"pilot_{index:02d}",
            rules=tuple(rules),
        )
        users.append(user)
    return users


def parse_user(value: Mapping[str, Any]) -> UserConfig:
    """Decodes the current user configuration format at the JSON boundary."""
    rules = tuple(
        PreferenceRule(
            id=rule["id"], precedence=rule["precedence"],
            when=Condition(
                property=rule["when"]["property"],
                operator=ConditionOperator(rule["when"]["operator"]),
                threshold=rule["when"]["threshold"],
            ) if rule["when"] is not None else None,
            weights=rule["weights"],
        ) for rule in value["rules"]
    )
    rating_criteria = output_definitions.DEFAULT_RATING_CRITERIA
    if "rating_criteria" in value:
        rating_criteria = tuple(
            rating_scale.RatingAnchoredCriterion(
                name=item["name"], anchors=tuple(item["anchors"]),
            ) for item in value["rating_criteria"]
        )
    return UserConfig(
        user_id=value["user_id"], rules=rules, rating_criteria=rating_criteria,
    )


def load_user(path: str | pathlib.Path) -> UserConfig:
    """Loads one user configuration from its JSON file."""
    value = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return parse_user(value)


def load_users(directory: str | pathlib.Path) -> list[UserConfig]:
    """Loads the user*.json files in a configuration directory."""
    users = [
        load_user(path)
        for path in sorted(pathlib.Path(directory).glob("user*.json"))
    ]
    if not users or len({user.user_id for user in users}) != len(users):
        raise ValueError("Empty or duplicate user IDs")
    return users
