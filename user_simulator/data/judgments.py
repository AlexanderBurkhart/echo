"""Typed measurements and output assessments returned by the frozen judge."""

import dataclasses
import enum


class JudgmentKind(enum.StrEnum):
    """Identifies which ratings the LM-as-a-judge is asked to provide."""

    TASK = "task"
    OUTPUT = "output"


@dataclasses.dataclass
class Rating:
    """A numeric rating or explicit quantity, with its unit and evidence."""

    value: float | None
    unit: str | None
    evidence: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class LMJudgment:
    """Task or output ratings, explicit quantities, and saved provenance.

    Task conditions, including task quantities, live in ratings; their
    quantities map is empty. Output quantities record proposed user costs.
    """

    ratings: dict[str, Rating]
    quantities: dict[str, Rating] = dataclasses.field(default_factory=dict)
    source_record: str | None = None
