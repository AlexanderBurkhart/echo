"""History-only task definitions and their qualitative rating scale."""

import types

from user_simulator.judge.ratings import ratings as rating_scale

PREFERENCES = ("urgency", "uncertainty", "resource_constraints")
TASK_RATING_UNIT = "rating_0_100"
QUALITATIVE = frozenset(PREFERENCES)

TASK_CRITERIA = (
    rating_scale.RatingScaleCriterion(
        name="urgency",
        description=(
            "Time pressure from deadline, available time, and consequences "
            "of delay. Use 0 when no deadline or delay consequence exists; "
            "25 for a flexible preference to finish soon; 50 for a deadline "
            "with little pressure; 75 for a pressing deadline; 100 for an "
            "imminent deadline whose breach defeats the user's stated goal."
        ),
    ),
    rating_scale.RatingScaleCriterion(
        name="uncertainty",
        description=(
            "Missing information, unsettled requirements, and possible "
            "changes. Use 0 when requirements are explicitly settled; 25 "
            "for minor possible revisions; 50 for one substantive open "
            "decision; 75 for multiple substantive undecided requirements "
            "or likely changes; 100 when essential information is absent "
            "and the task cannot yet be specified."
        ),
    ),
    rating_scale.RatingScaleCriterion(
        name="resource_constraints",
        description=(
            "How limited the resources, personal time, and effort available "
            "to the user are. Assess stated capacity and commitments, not "
            "the quality of a proposed solution. If none of these resource "
            "constraints are stated, value must be null and evidence []. "
            "A deadline alone describes urgency, not a user resource limit. "
            "Do not assign a low default such as 25 when constraints are "
            "unknown. With explicit resources, use 0 for no constraints, "
            "25 for flexible limits, 50 for a meaningful but flexible "
            "allowance, 75 for a strict maximum that cannot be exceeded, "
            "and 100 for almost no remaining user capacity. Intensity "
            "describes how binding the limit is, not the raw amount."
        ),
    ),
    rating_scale.RatingScaleCriterion(
        name="budget",
        description=(
            "Explicit external-resource allowance; preserve its stated unit."
        ),
    ),
    rating_scale.RatingScaleCriterion(
        name="known_expenses",
        description=(
            "Explicit existing external-resource commitments; preserve "
            "their unit."
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
