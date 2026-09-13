"""Scores fixed assessments with private conditional preference weights."""

from collections.abc import Mapping, Sequence
import math
import operator

from user_simulator import config
from user_simulator.data import judgments
from user_simulator.data import preference_scores
from user_simulator.judge.ratings import task_definitions
from user_simulator.judge.ratings import ratings as rating_scale

_ENGINE_VERSION = "weighted-values-v3"
_MIN_SCORE = 0.0
_MAX_SCORE = 100.0
_OPERATORS = {
    config.ConditionOperator.GREATER: operator.gt,
    config.ConditionOperator.GREATER_EQUAL: operator.ge,
    config.ConditionOperator.LESS: operator.lt,
    config.ConditionOperator.LESS_EQUAL: operator.le,
    config.ConditionOperator.EQUAL: operator.eq,
}


def _calculate_weighted_value(
    weights: Mapping[str, float], values: Mapping[str, float]
) -> float:
    """Averages values in [0, 1]; Mapping inputs are read without mutation."""
    weights = config.normalize_weights(weights)
    for name in weights:
        value = values.get(name)
        if not config.is_number(value) or not 0 <= value <= 1:
            raise ValueError("Each preference needs a finite value in [0, 1]")
    return math.fsum(weight * values[name] for name, weight in weights.items())


def _find_matching_rules(
    user: config.UserConfig,
    task_ratings: Mapping[str, judgments.Rating],
) -> list[config.PreferenceRule]:
    """Returns existing user rules activated by the supplied task ratings.

    The highest integer precedence wins. Missing or unknown ratings and
    ratings with a different scale unit do not activate conditions.
    """
    matches = []
    for rule in user.rules:
        condition = rule.when
        if condition is None:
            matches.append(rule)
            continue
        rating = task_ratings.get(condition.property)
        if rating is None or rating.value is None:
            continue
        if rating.unit != task_definitions.TASK_RATING_UNIT:
            continue
        if not config.is_number(rating.value) or not 0 <= rating.value <= 100:
            raise ValueError("Task ratings must be finite and in [0, 100]")
        compare = _OPERATORS[condition.operator]
        if compare(rating.value, condition.threshold):
            matches.append(rule)
    return matches


def _map_output_ratings(
    names: set[str],
    output_ratings: Mapping[str, judgments.Rating],
    rating_values: Sequence[float],
) -> preference_scores.MappedRatings:
    """Returns anchored values and names assigned the explicit unknown value."""
    values = {}
    unknown = []
    for name in sorted(names):
        rating = output_ratings[name]
        if rating.unit != rating_scale.OUTPUT_RATING_UNIT:
            raise ValueError("Output ratings require the rating_0_4 unit")
        level = rating.value
        if level is None:
            values[name] = rating_scale.UNKNOWN_VALUE
            unknown.append(name)
        elif config.is_number(level) and level in range(len(rating_values)):
            values[name] = rating_values[int(level)]
        else:
            raise ValueError("Assessment level must be 0–4 or null")
    return preference_scores.MappedRatings(values, tuple(unknown))


def calculate_preference_score(
    user: config.UserConfig,
    task_ratings: Mapping[str, judgments.Rating],
    output_ratings: Mapping[str, judgments.Rating],
    rating_values: Sequence[float] = rating_scale.DEFAULT_RATING_VALUES,
) -> preference_scores.ScoreResult:
    """Returns a simulator score and evaluator-only scoring diagnostics.

    Args:
        user: Validated private config with direct weights in each rule.
        task_ratings: LM task ratings used to activate conditional rules.
        output_ratings: LM output ratings for the configured preferences.
        rating_values: Five nondecreasing values in [0, 1] for levels 0–4.

    Returns:
        A record containing a score in [0, 100], rules, weights, component
        values, unknown flags, and scoring version. No candidate pool is used.
    """
    if len(rating_values) != len(rating_scale.DEFAULT_RATING_VALUES):
        raise ValueError(
            "Rating values must contain "
            f"{len(rating_scale.DEFAULT_RATING_VALUES)} values"
        )
    if any(
        not config.is_number(value) or not 0 <= value <= 1
        for value in rating_values
    ):
        raise ValueError("Rating values must be finite and in [0, 1]")
    if any(
        left > right for left, right in zip(rating_values, rating_values[1:])
    ):
        raise ValueError("Rating values must be monotone")
    matches = _find_matching_rules(user, task_ratings)
    selected = max(matches, key=lambda rule: rule.precedence)
    # Omitted and zero-weight preferences have no effect on the score or its
    # denominator. A low output rating for an active preference still counts.
    weights = {
        name: weight for name, weight in selected.weights.items() if weight > 0
    }
    values = _map_output_ratings(set(weights), output_ratings, rating_values)
    component_values = values.components
    score = _MAX_SCORE * _calculate_weighted_value(weights, component_values)
    score = min(_MAX_SCORE, max(_MIN_SCORE, score))
    return preference_scores.ScoreResult(
        score=score,
        selected_rule=selected.id,
        matching_rules=[rule.id for rule in matches],
        weights=dict(weights),
        normalized_weights=config.normalize_weights(weights),
        component_values=component_values,
        unknown_components=list(values.unknown_components),
        unknown_value=rating_scale.UNKNOWN_VALUE,
        value_mapping=list(rating_values),
        engine_version=_ENGINE_VERSION,
    )
