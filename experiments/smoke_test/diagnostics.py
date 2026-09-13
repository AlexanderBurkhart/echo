"""Reports score error and reference diagnostics for simulator evaluation."""

from collections.abc import Mapping, Sequence
import dataclasses
import enum
import itertools
import json
import math
import pathlib
import types

from loguru import logger as log

from experiment_logging import runs
from experiments.smoke_test import cases
from user_simulator import config
from user_simulator import engine
from user_simulator.data import score_records
from user_simulator.judge.ratings import ratings

_COMPONENT = "smoke_test"


class Ordering(enum.StrEnum):
    """Ordering of the first output's score relative to the second's."""

    LESS = "less"
    TIE = "tie"
    GREATER = "greater"


@dataclasses.dataclass(frozen=True)
class Agreement:
    """All eligible observations remain in total, including unavailable ones."""

    total: int
    available: int
    correct: int


@dataclasses.dataclass(frozen=True)
class NumericError:
    """Exact agreement and absolute error over available numeric values only."""

    agreement: Agreement
    mean_absolute_error: float | None


@dataclasses.dataclass(frozen=True)
class PairComparison:
    """One unordered pair of outputs for the same user and identical history."""

    user_id: str
    first_case: str
    second_case: str
    expected: Ordering
    actual: Ordering | None


@dataclasses.dataclass(frozen=True)
class Diagnostics:
    """Evaluation-only agreement metrics and pooled criterion error."""

    rule_selection: Agreement
    task_rule_selection: Agreement
    criteria: Mapping[str, NumericError]
    combined_score: NumericError
    ordering: Agreement
    pairs: tuple[PairComparison, ...]
    failed_cases: tuple[str, ...]
    missing_cases: tuple[str, ...]
    changed_rating_definitions: tuple[str, ...]
    macro_criterion_mean_absolute_error: float | None = dataclasses.field(
        init=False,
    )

    def __post_init__(self) -> None:
        """Freezes the criterion mapping with the rest of this report."""
        object.__setattr__(self, "criteria", types.MappingProxyType(
            dict(self.criteria)
        ))
        errors = [item.mean_absolute_error for item in self.criteria.values()
                  if item.mean_absolute_error is not None]
        object.__setattr__(
            self, "macro_criterion_mean_absolute_error",
            math.fsum(errors) / len(errors) if errors else None,
        )

    def calculate_criterion_error(self) -> NumericError:
        """Pools authored labels with equal weight for every rating."""
        metrics = tuple(self.criteria.values())
        total = sum(item.agreement.total for item in metrics)
        available = sum(item.agreement.available for item in metrics)
        correct = sum(item.agreement.correct for item in metrics)
        error_sum = math.fsum(
            item.mean_absolute_error * item.agreement.available
            for item in metrics if item.mean_absolute_error is not None
        )
        return NumericError(
            agreement=Agreement(total, available, correct),
            mean_absolute_error=error_sum / available if available else None,
        )


def _calculate_expected_score(
    group: cases.UserCases, case: cases.SmokeCase,
) -> float:
    rule = next(rule for rule in group.user.rules
                if rule.id == case.expected_rule)
    return 100 * math.fsum(
        weight * case.expected_ratings[name] / 4
        for name, weight in rule.weights.items() if weight > 0
    ) / math.fsum(rule.weights.values())


def _compare_scores(first: float, second: float) -> Ordering:
    # The simulator treats equal final scores as ties, without binning.
    if first == second:
        return Ordering.TIE
    return Ordering.LESS if first < second else Ordering.GREATER


def _calculate_numeric_error(
    observations: Sequence[tuple[float, float | None]],
) -> NumericError:
    errors = [abs(actual - expected) for expected, actual in observations
              if actual is not None]
    correct = sum(actual is not None and math.isclose(
        expected, actual, abs_tol=1e-9
    ) for expected, actual in observations)
    return NumericError(
        agreement=Agreement(len(observations), len(errors), correct),
        mean_absolute_error=math.fsum(errors) / len(errors) if errors else None,
    )


def _get_available_rating(
    record: score_records.ScoreRecord | None, name: str,
) -> float | None:
    if record is None or record.output_judgment is None:
        return None
    rating = record.output_judgment.ratings.get(name)
    if (rating is None or not config.is_number(rating.value)
            or rating.unit != ratings.OUTPUT_RATING_UNIT
            or rating.value not in range(5)):
        return None
    return rating.value


def _compare_output_pairs(
    user_id: str,
    outputs: Sequence[tuple[cases.SmokeCase, float, float | None]],
) -> list[PairComparison]:
    pairs = []
    for first, second in itertools.combinations(outputs, 2):
        first_case, first_expected, first_actual = first
        second_case, second_expected, second_actual = second
        if first_case.history != second_case.history:
            continue
        actual_order = None
        if first_actual is not None and second_actual is not None:
            actual_order = _compare_scores(first_actual, second_actual)
        pairs.append(PairComparison(
            user_id=user_id, first_case=first_case.name,
            second_case=second_case.name,
            expected=_compare_scores(first_expected, second_expected),
            actual=actual_order,
        ))
    return pairs


def _select_task_rule(record: score_records.ScoreRecord | None) -> str | None:
    if record is None or record.task_judgment is None:
        return None
    # Reuse the production condition checks without inventing output ratings.
    # The engine exposes no public selector independent of output scoring.
    matches = engine._find_matching_rules(  # pylint: disable=protected-access
        record.user_config, record.task_judgment.ratings,
    )
    return max(matches, key=lambda rule: rule.precedence).id


def _calculate_rule_agreement(
    observations: Sequence[tuple[str, str | None]],
) -> Agreement:
    return Agreement(
        len(observations),
        sum(actual is not None for _, actual in observations),
        sum(expected == actual for expected, actual in observations),
    )


def calculate_diagnostics(
    groups: Sequence[cases.UserCases],
    saved_records: Sequence[score_records.ScoreRecord],
) -> Diagnostics:
    """Compares saved records with authored references without model calls.

    Every authored case counts for rules and combined scores; each criterion
    counts only cases with an authored label. Failed/incomplete requests supply
    no combined score. Any valid ratings already saved on those requests still
    contribute to criterion coverage. Missing records remain in all applicable
    totals. Duplicate records for an input are rejected rather than selected.
    """
    indexed = {}
    for record in saved_records:
        key = (record.user_id, record.history, record.output)
        if key in indexed:
            raise ValueError("Multiple saved records for one evaluation input")
        indexed[key] = record
    rule_observations = []
    task_rule_observations = []
    criterion_observations = {}
    score_observations = []
    pairs = []
    failed = []
    missing = []
    changed_definitions = set()
    for group in groups:
        group_scores = []
        for case in group.cases:
            key = (group.user.user_id, case.history, case.output)
            record = indexed.get(key)
            label = f"{group.user.user_id}/{case.name}"
            if record is None:
                missing.append(label)
            elif record.status != score_records.ScoreStatus.OK:
                failed.append(label)
            if record is not None:
                if record.user_config.rules != group.user.rules:
                    raise ValueError(
                        f"Saved rule configuration differs: {label}"
                    )
                if (record.user_config.rating_criteria
                        != group.user.rating_criteria):
                    changed_definitions.add(group.user.user_id)
            result = record.result if record is not None else None
            selected_rule = result.selected_rule if result is not None else None
            rule_observations.append((case.expected_rule, selected_rule))
            task_rule_observations.append((
                case.expected_rule, _select_task_rule(record),
            ))
            for name, expected in case.expected_ratings.items():
                actual = _get_available_rating(record, name)
                criterion_observations.setdefault(name, []).append(
                    (expected, actual)
                )
            actual_score = None
            if (record is not None
                    and record.status == score_records.ScoreStatus.OK
                    and result is not None and config.is_number(result.score)):
                actual_score = result.score
            expected_score = _calculate_expected_score(group, case)
            score_observations.append((expected_score, actual_score))
            group_scores.append((case, expected_score, actual_score))
        pairs.extend(_compare_output_pairs(group.user.user_id, group_scores))
    return Diagnostics(
        rule_selection=_calculate_rule_agreement(rule_observations),
        task_rule_selection=_calculate_rule_agreement(task_rule_observations),
        criteria={name: _calculate_numeric_error(values)
                  for name, values in criterion_observations.items()},
        combined_score=_calculate_numeric_error(score_observations),
        ordering=Agreement(
            len(pairs), sum(pair.actual is not None for pair in pairs),
            sum(pair.actual == pair.expected for pair in pairs),
        ),
        pairs=tuple(pairs), failed_cases=tuple(failed),
        missing_cases=tuple(missing),
        changed_rating_definitions=tuple(sorted(changed_definitions)),
    )


def _format_agreement(metric: Agreement) -> str:
    return (f"{metric.correct}/{metric.total}; "
            f"coverage {metric.available}/{metric.total}")


def format_diagnostics(report: Diagnostics) -> str:
    """Explains denominators and ties alongside the diagnostic metrics."""
    lines = [
        "Criterion MAE and reference diagnostics",
        "Rule selection from completed scores: "
        + _format_agreement(report.rule_selection),
        "Rule selection from saved task judgments: "
        + _format_agreement(report.task_rule_selection),
    ]
    for name, metric in (*report.criteria.items(),
                         ("combined score", report.combined_score)):
        error = (f"{metric.mean_absolute_error:.4f}"
                 if metric.mean_absolute_error is not None else "unavailable")
        agreement = metric.agreement
        lines.append(
            f"{name}: exact {_format_agreement(agreement)}; "
            f"exact errors {agreement.available - agreement.correct}; "
            f"unavailable {agreement.total - agreement.available}; MAE {error}"
        )
    pooled = report.calculate_criterion_error()
    error = pooled.mean_absolute_error
    lines.append(
        f"Pooled criterion MAE (0–4): {error}; "
        "normalized points (0–100): "
        f"{25 * error if error is not None else None}; "
        + _format_agreement(pooled.agreement)
    )
    available_criteria = sum(item.mean_absolute_error is not None
                             for item in report.criteria.values())
    lines.append(
        "Macro criterion MAE (diagnostic; equal weight per criterion): "
        f"{report.macro_criterion_mean_absolute_error}; "
        f"criterion coverage {available_criteria}/{len(report.criteria)}"
    )
    lines.extend((
        "Ordering agreement: " + _format_agreement(report.ordering),
        "Pairs compare only the same user and identical history; ties count "
        "as agreement only when both reference and observed scores tie.",
        "Criterion totals include authored labels only (0–4); combined score "
        "uses 0–100. MAE uses available values only. Failed requests supply no "
        "combined score; saved valid ratings still count toward coverage.",
        f"Failed/incomplete cases: {', '.join(report.failed_cases) or 'none'}",
        f"Missing cases: {', '.join(report.missing_cases) or 'none'}",
        "Saved rating definitions differ from current definitions for: "
        f"{', '.join(report.changed_rating_definitions) or 'none'}. "
        "Stored ratings/scores are compared to the same authored numeric "
        "targets without rescoring.",
    ))
    for pair in report.pairs:
        lines.append(
            f"{pair.user_id}: {pair.first_case} vs {pair.second_case}: "
            f"expected {pair.expected}, actual {pair.actual or 'unavailable'}"
        )
    return "\n".join(lines) + "\n"


def write_diagnostics(
    run_root: pathlib.Path, split: cases.Split,
    groups: Sequence[cases.UserCases],
    score_record_paths: Sequence[pathlib.Path],
) -> Diagnostics:
    """Loads only the supplied score paths and writes both diagnostics."""
    saved = [score_records.load_score_record(path)
             for path in score_record_paths if path.exists()]
    report = calculate_diagnostics(groups, saved)
    summary = format_diagnostics(report)
    runs.write_record(run_root, _COMPONENT, f"{split}/diagnostics.json", report)
    runs.write_text(run_root, _COMPONENT, f"{split}/diagnostics.txt", summary)
    log.info("\n{}", summary)
    return report


def review_saved_development(run_root: pathlib.Path) -> Diagnostics:
    """Adds diagnostics to a prior development run without loading holdout.

    Older results without score_record references count as missing, even when
    their error text mentions a failed request's path. No paths are inferred
    from prose and no other split's records are scanned.
    """
    root = runs.get_component_directory(run_root, _COMPONENT)
    results = json.loads((root / "development/results.json").read_text())
    score_record_paths = [pathlib.Path(outcome["score_record"])
                          for outcome in results["cases"]
                          if outcome.get("score_record") is not None]
    return write_diagnostics(
        run_root, cases.Split.DEVELOPMENT,
        cases.load_cases(cases.Split.DEVELOPMENT), score_record_paths,
    )
