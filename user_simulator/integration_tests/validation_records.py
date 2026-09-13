"""Typed comparison fixtures and LM validation reports."""

import dataclasses
import enum

from conversation import history
from user_simulator import config
from user_simulator.data import judge_identity
from user_simulator.data import judgments


class CandidateSide(enum.StrEnum):
    """Identifies a candidate within a development comparison."""

    A = "A"
    B = "B"


class ComparisonLabel(enum.StrEnum):
    """Preference ordering calculated from two numerical scores."""

    A = "A"
    B = "B"
    TIE = "tie"


class ValidationStatus(enum.StrEnum):
    """Whether evaluation calls completed, not whether the judge is good."""

    NOT_RUN = "not_run"
    RUNNING = "running"
    COMPLETE = "complete"
    COMPLETE_WITH_FAILURES = "complete_with_failures"


@dataclasses.dataclass(frozen=True)
class Candidate:
    """An output and its authored reference LM ratings."""

    side: CandidateSide
    output: str
    reference: judgments.LMJudgment


@dataclasses.dataclass(frozen=True)
class ComparisonCase:
    """One authored comparison, with explicit review provenance."""

    id: str
    tags: tuple[str, ...]
    history: history.ConversationHistory
    first: Candidate
    second: Candidate
    reference_task: judgments.LMJudgment
    review_notes: str
    reviewer: str | None = None
    reviewed_at: str | None = None
    user: config.UserConfig | None = None

    def get_candidates(self) -> tuple[Candidate, Candidate]:
        """Returns the two candidates in their declared order."""
        return self.first, self.second


@dataclasses.dataclass(frozen=True)
class CaseBank:
    """Loaded development cases and the hash of their original JSON."""

    version: str
    human_review_status: str
    cases: tuple[ComparisonCase, ...]
    source_hash: str


@dataclasses.dataclass(frozen=True)
class RatingPerturbation:
    """A changed rating and resulting label; side=None means task conditions."""

    property: str
    delta: int
    label: ComparisonLabel
    side: CandidateSide | None = None


@dataclasses.dataclass(frozen=True)
class SensitivityResult:
    """Label sensitivity to value mappings and small rating errors."""

    baseline: ComparisonLabel
    alternative_mapping_labels: dict[str, ComparisonLabel]
    assessment_perturbations: tuple[RatingPerturbation, ...]
    condition_perturbations: tuple[RatingPerturbation, ...]
    any_label_change: bool


@dataclasses.dataclass(frozen=True)
class ReferenceAgreement:
    """Agreement with authored ratings, separate from repeat consistency."""

    side: CandidateSide
    task_exact: dict[str, bool]
    quantity_exact: dict[str, bool]
    level_exact: dict[str, bool]


@dataclasses.dataclass
class RepetitionResult:
    """Scores and failures from one uncached comparison repetition."""

    scores: dict[CandidateSide, float] = dataclasses.field(default_factory=dict)
    failures: dict[CandidateSide, str] = dataclasses.field(default_factory=dict)
    reference_agreement: list[ReferenceAgreement] = dataclasses.field(
        default_factory=list
    )
    label: ComparisonLabel | None = None
    matches_reference_label: bool | None = None


@dataclasses.dataclass
class CaseReport:
    """Repeated judgments and sensitivity checks for one comparison case."""

    id: str
    reference_label: ComparisonLabel
    runs: list[RepetitionResult] = dataclasses.field(default_factory=list)
    sensitivity: dict[str, dict[str, dict[str, SensitivityResult]]] = (
        dataclasses.field(default_factory=dict)
    )
    reference_ratings_match: bool = False
    score_order_changed: bool = False
    messages: list[str] = dataclasses.field(default_factory=list)
    all_three_labels_consistent: bool = False
    all_three_measurements_consistent: bool = False
    label_counts: dict[ComparisonLabel, int] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class ValidationReport:
    """Validation results without asserting human-reviewed correctness."""

    case_bank_hash: str
    judge: judge_identity.LocalJudgeIdentity
    human_review_status: str
    repetitions: int
    cases: list[CaseReport] = dataclasses.field(default_factory=list)
    status: ValidationStatus = ValidationStatus.RUNNING
    interpretation: str | None = None
