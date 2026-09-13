"""Simple rating criteria and the synthetic output value mapping."""

import dataclasses

# Anchor index i maps to DEFAULT_RATING_VALUES[i]. Equal spacing is a
# synthetic assumption; these numbers are not measured human satisfaction.
DEFAULT_RATING_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
RATING_LEVELS = range(len(DEFAULT_RATING_VALUES))
OUTPUT_RATING_UNIT = f"rating_0_{RATING_LEVELS.stop - 1}"
UNKNOWN_VALUE = 0.5


@dataclasses.dataclass(frozen=True)
class RatingAnchoredCriterion:
    """An output property with descriptions ordered like DEFAULT_RATING_VALUES.

    The fixed tuple type supports static checks of hardcoded anchor counts.
    UserConfig validates its selected criteria when constructed.
    """

    name: str
    anchors: tuple[str, str, str, str, str]


@dataclasses.dataclass(frozen=True)
class RatingScaleCriterion:
    """A task property described for numeric rating by the LM.

    Task intensities use the shared 0–100 scale. Explicit quantities retain
    their stated units, as specified in each description.
    """

    name: str
    description: str
