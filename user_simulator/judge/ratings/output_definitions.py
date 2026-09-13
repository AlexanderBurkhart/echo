"""Default output anchors and cost components used by user configurations.

Each user owns its selected rating criteria. These defaults contain
descriptions only; private preference weights and rules stay in the user
configuration.
"""

from user_simulator.judge.ratings import ratings as rating_scale

COST_COMPONENTS = ("money", "user_time", "effort")


DEFAULT_RATING_CRITERIA = (
    rating_scale.RatingAnchoredCriterion(name="urgency", anchors=(
        "Introduces delay that clearly defeats the stated timing need.",
        "Introduces substantial avoidable delay or poorly addresses timing.",
        (
            'Partially addresses timing, with meaningful avoidable '
            'delay remaining.'
        ),
        "Supports timely completion with only minor avoidable delay.",
        "Directly supports timely completion without unnecessary delay.",
    )),
    rating_scale.RatingAnchoredCriterion(name="uncertainty", anchors=(
        (
            'Ignores explicit unknowns and locks in a choice that '
            'cannot accommodate changes.'
        ),
        "Leaves major known uncertainties unaddressed and costly to revise.",
        "Handles some uncertainties but leaves meaningful adaptation gaps.",
        (
            "Provides a way to revise decisions or outputs, but leaves a "
            "minor stated change or requirement without a way to adapt."
        ),
        (
            "Explicitly acknowledges stated uncertainty and gives a usable "
            "way to revise decisions or outputs if information changes, "
            "without claiming that unknowns are settled."
        ),
    )),
    rating_scale.RatingAnchoredCriterion(name="money", anchors=(
        (
            'Requires spending far beyond the stated budget or a very '
            'heavy documented money burden.'
        ),
        (
            'Requires spending above budget or a substantial documented '
            'money burden.'
        ),
        (
            'Requires a moderate documented money burden relative to '
            'stated resources.'
        ),
        (
            'Requires a small documented money burden relative to '
            'stated resources.'
        ),
        "Requires no user spending or negligible documented money burden.",
    )),
    rating_scale.RatingAnchoredCriterion(name="user_time", anchors=(
        (
            'Requires user time far beyond their limit or a very heavy '
            'documented time burden.'
        ),
        (
            'Requires user time above their limit or a substantial '
            'documented time burden.'
        ),
        (
            'Requires a moderate documented user-time burden relative '
            'to stated resources.'
        ),
        (
            'Requires a small documented user-time burden relative to '
            'stated resources.'
        ),
        "Requires no user time or negligible documented user-time burden.",
    )),
    rating_scale.RatingAnchoredCriterion(name="effort", anchors=(
        (
            'Requires extreme documented user exertion or complexity '
            'beyond their stated capacity.'
        ),
        "Requires substantial documented user exertion or complexity.",
        "Requires moderate documented user exertion or complexity.",
        "Requires minor documented user exertion or complexity.",
        "Requires no or negligible documented user exertion or complexity.",
    )),
)
