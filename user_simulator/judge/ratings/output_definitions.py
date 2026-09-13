"""Default output anchors and cost components used by user configurations.

Each user owns its selected rating criteria. These defaults contain
descriptions only; private preference weights and rules stay in the user
configuration.
"""

from user_simulator.judge.ratings import ratings as rating_scale

COST_COMPONENTS = ("external_resource_demand", "user_time", "effort")


DEFAULT_RATING_CRITERIA = (
    rating_scale.RatingAnchoredCriterion(name="urgency", anchors=(
        "Introduces delay that clearly defeats the stated timing need.",
        (
            "Introduces substantial avoidable delay that puts timely "
            "completion at risk, or offers no usable plan for the timing need."
        ),
        (
            "Provides a timely completion plan but adds an unnecessary wait "
            "or work that consumes a meaningful part of the available time."
        ),
        (
            "Supports timely completion with only a brief unnecessary check "
            "or small extra action, leaving ample time for the stated work."
        ),
        "Directly supports timely completion without unnecessary delay.",
    )),
    rating_scale.RatingAnchoredCriterion(name="uncertainty", anchors=(
        (
            'Ignores explicit unknowns and locks in a choice that '
            'cannot accommodate changes.'
        ),
        (
            "Acknowledges uncertainty but provides no usable revision path "
            "for the substantive unsettled requirements. Unlike level 0, "
            "it does not explicitly prevent all revisions."
        ),
        (
            "Provides a usable revision path for at least one substantive "
            "unsettled requirement but excludes another. Credit the "
            "supported changes even when other changes are locked out."
        ),
        (
            "Supports revisions for all substantive unsettled requirements, "
            "but leaves a gap only for a change the history explicitly "
            "identifies as minor or cosmetic."
        ),
        (
            "Addresses any stated uncertainty and gives a usable "
            "way to revise decisions or outputs if information changes, "
            "without claiming that unknowns are settled."
        ),
    )),
    rating_scale.RatingAnchoredCriterion(
        name="external_resource_demand", anchors=(
            (
                "Requires external resources supplied or committed by the "
                "user far beyond stated capacity, or an extreme documented "
                "burden. Exclude the user's personal time and effort."
            ),
            (
                "Requires external resources supplied or committed by the "
                "user above stated capacity, or a substantial documented "
                "burden. Exclude the user's personal time and effort."
            ),
            (
                "Requires a moderate documented external-resource burden "
                "relative to the user's stated capacity. Exclude the user's "
                "personal time and effort."
            ),
            (
                "Requires a small documented external-resource burden "
                "relative to the user's stated capacity. Exclude the user's "
                "personal time and effort."
            ),
            (
                "Explicitly requires no or negligible external resources "
                "supplied or committed by the user. Exclude the user's "
                "personal time and effort; automation alone is not evidence "
                "of no external-resource use."
            ),
        ),
    ),
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
        (
            "Requires minor user effort: a simple button press, toggle, "
            "confirmation, or a few uncomplicated actions."
        ),
        "Explicitly requires no user action or effort; fully automatic.",
    )),
)
