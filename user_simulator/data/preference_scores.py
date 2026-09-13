"""Deterministic preference-score results and intermediate values."""

import dataclasses


@dataclasses.dataclass
class MappedRatings:
    """Independently mapped output ratings and explicitly unknown components."""

    components: dict[str, float]
    unknown_components: tuple[str, ...]


@dataclasses.dataclass
class ScoreResult:
    """A preference score and the private values used to calculate it."""

    score: float
    selected_rule: str
    matching_rules: list[str]
    weights: dict[str, float]
    normalized_weights: dict[str, float]
    component_values: dict[str, float]
    unknown_components: list[str]
    unknown_value: float
    value_mapping: list[float]
    engine_version: str
