"""History-only task definitions and their qualitative rating scale."""

import types

from user_simulator.judge.ratings import ratings as rating_scale

PREFERENCES = ("urgency", "uncertainty", "user_cost")
TASK_RATING_UNIT = "rating_0_100"
QUALITATIVE = frozenset(PREFERENCES)

TASK_CRITERIA = (
    rating_scale.RatingScaleCriterion(
        name="urgency",
        description=(
            "Time pressure from deadline, available time, and consequences "
            "of delay."
        ),
    ),
    rating_scale.RatingScaleCriterion(
        name="uncertainty",
        description=(
            "Missing information, unsettled requirements, and possible "
            "changes."
        ),
    ),
    rating_scale.RatingScaleCriterion(
        name="user_cost",
        description=(
            "Resource constraint intensity from budget, known expenses, "
            "user time and effort limits."
        ),
    ),
    rating_scale.RatingScaleCriterion(
        name="budget",
        description=(
            "Explicit monetary budget; preserve currency unit, e.g. USD."
        ),
    ),
    rating_scale.RatingScaleCriterion(
        name="known_expenses",
        description=(
            "Explicit known expenses; preserve currency unit."
        ),
    ),
    rating_scale.RatingScaleCriterion(
        name="available_time",
        description=(
            "Explicit time remaining to deadline; preserve stated time "
            "unit."
        ),
    ),
    rating_scale.RatingScaleCriterion(
        name="user_time_budget",
        description=(
            "Explicit time the user can spend; preserve stated time unit."
        ),
    ),
    rating_scale.RatingScaleCriterion(
        name="effort_budget",
        description=(
            "Explicit numeric effort constraint; preserve stated unit."
        ),
    ),
)
CONDITIONS = types.MappingProxyType({
    criterion.name: criterion for criterion in TASK_CRITERIA
})
