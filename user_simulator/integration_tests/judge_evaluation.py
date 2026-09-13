"""Optional fixture-based judge evaluation for development experiments."""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import json
import pathlib

from conversation import history
from experiment_logging import records
from experiment_logging import runs
from user_simulator import config
from user_simulator import engine
from user_simulator import simulator
from user_simulator.data import judgments
from user_simulator.data import score_records
from user_simulator.integration_tests import validation_records
from user_simulator.judge import judge as judge_interface
from user_simulator.judge import judging
from user_simulator.judge.ratings import output_definitions
from user_simulator.judge.ratings import task_definitions
from user_simulator.judge.ratings import ratings as rating_scale


REPETITIONS = 3
CASE_COUNT = 12
REPORT_FILENAME = "validation.json"


def calculate_comparison_label(
    first: float, second: float
) -> validation_records.ComparisonLabel:
    """Returns A, B, or an exact tie for two already computed scores."""
    if first > second:
        return validation_records.ComparisonLabel.A
    if second > first:
        return validation_records.ComparisonLabel.B
    return validation_records.ComparisonLabel.TIE


def _evaluate_sensitivity(
    user: config.UserConfig,
    task_ratings: Mapping[str, judgments.Rating],
    first: Mapping[str, judgments.Rating],
    second: Mapping[str, judgments.Rating],
) -> validation_records.SensitivityResult:
    """Returns label changes under mappings and one-at-a-time rating errors.

    Both candidates use the supplied common task assessment. Output errors move
    one known level by one anchor; condition errors move one intensity by two.
    Unknown measurements are kept unknown. This is a sensitivity diagnostic,
    not a model of the probability of LM judgment errors.
    """
    def compare(
        task: Mapping[str, judgments.Rating],
        first_ratings: Mapping[str, judgments.Rating],
        second_ratings: Mapping[str, judgments.Rating],
        rating_values: Sequence[float] = rating_scale.DEFAULT_RATING_VALUES,
    ) -> validation_records.ComparisonLabel:
        """Returns the label under this user's fixed configuration."""
        first_score = engine.calculate_preference_score(
            user, task, first_ratings, rating_values
        ).score
        second_score = engine.calculate_preference_score(
            user, task, second_ratings, rating_values
        ).score
        return calculate_comparison_label(first_score, second_score)

    baseline = compare(task_ratings, first, second)
    alternatives = {
        "concave": [0, 0.4, 0.65, 0.85, 1],
        "convex": [0, 0.15, 0.35, 0.6, 1],
    }
    mapping_labels = {
        name: compare(task_ratings, first, second, mapping)
        for name, mapping in alternatives.items()
    }
    changes = []
    for side, ratings in (
        (validation_records.CandidateSide.A, first),
        (validation_records.CandidateSide.B, second),
    ):
        for name, item in ratings.items():
            if item.value is None:
                continue
            for delta in (-1, 1):
                changed = item.value + delta
                if changed not in rating_scale.RATING_LEVELS:
                    continue
                revised = dict(ratings)
                revised[name] = dataclasses.replace(item, value=changed)
                result = compare(
                    task_ratings,
                    (revised if side is validation_records.CandidateSide.A
                     else first),
                    (revised if side is validation_records.CandidateSide.B
                     else second),
                )
                changes.append(validation_records.RatingPerturbation(
                    side=side, property=name, delta=delta, label=result,
                ))
    conditions = []
    for name in task_definitions.PREFERENCES:
        if task_ratings[name].value is None:
            continue
        for delta in (-2, 2):
            revised = dict(task_ratings)
            revised[name] = dataclasses.replace(
                task_ratings[name],
                value=min(100, max(0, task_ratings[name].value + delta)),
            )
            conditions.append(validation_records.RatingPerturbation(
                property=name, delta=delta,
                label=compare(revised, first, second),
            ))
    return validation_records.SensitivityResult(
        baseline=baseline,
        alternative_mapping_labels=mapping_labels,
        assessment_perturbations=tuple(changes),
        condition_perturbations=tuple(conditions),
        any_label_change=(
            any(value != baseline for value in mapping_labels.values())
            or any(value.label != baseline for value in changes + conditions)
        ),
    )


def load_cases(path: str | pathlib.Path) -> validation_records.CaseBank:
    """Decodes the authored case bank at its JSON input boundary."""
    bundle = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    source_hash = records.compute_digest(bundle)
    raw_cases = bundle["cases"]
    if (len(raw_cases) != CASE_COUNT
            or len({case["id"] for case in raw_cases}) != CASE_COUNT):
        raise ValueError("Initial validation requires 12 distinct cases")
    cases = []
    for raw_case in raw_cases:
        user = (
            config.parse_user(raw_case["user"]) if "user" in raw_case else None
        )
        criteria = (
            user.rating_criteria
            if user else output_definitions.DEFAULT_RATING_CRITERIA
        )
        conversation_history = history.ConversationHistory(messages=tuple(
            history.Message(
                role=history.Role(message["role"]), content=message["content"]
            )
            for message in raw_case["history"]
        ))
        sources = [message.content for message in conversation_history.messages]
        task = judging.parse_lm_judgment(
            json.dumps(raw_case["reference_task"]), judgments.JudgmentKind.TASK,
            sources, criteria,
        )
        candidates = []
        for side in validation_records.CandidateSide:
            output = raw_case[side]
            reference = judging.parse_lm_judgment(
                json.dumps(raw_case["reference_outputs"][side]),
                judgments.JudgmentKind.OUTPUT, sources + [output], criteria,
            )
            candidates.append(validation_records.Candidate(
                side=side, output=output, reference=reference,
            ))
        cases.append(validation_records.ComparisonCase(
            id=raw_case["id"], tags=tuple(raw_case["tags"]),
            history=conversation_history,
            first=candidates[0], second=candidates[1],
            reference_task=task, review_notes=raw_case["review_notes"],
            reviewer=raw_case.get("reviewer"),
            reviewed_at=raw_case.get("reviewed_at"),
            user=user,
        ))
    return validation_records.CaseBank(
        version=bundle["version"],
        human_review_status=bundle["human_review_status"],
        cases=tuple(cases), source_hash=source_hash,
    )


def _compare_reference_ratings(
    case: validation_records.ComparisonCase,
    score_record: score_records.ScoreRecord,
) -> validation_records.ReferenceAgreement:
    """Returns task and output agreement, separately from repeat consistency."""
    candidate = next(item for item in case.get_candidates()
                     if item.output == score_record.output)
    measured = score_record.task_judgment.ratings
    assessed = score_record.output_judgment.ratings
    expected_measurements = case.reference_task.ratings
    expected_assessments = candidate.reference.ratings
    task_exact = {}
    for name, value in measured.items():
        expected = expected_measurements[name]
        task_exact[name] = (
            (value.value, value.unit)
            == (expected.value, expected.unit)
        )
    quantities = score_record.output_judgment.quantities
    expected_quantities = candidate.reference.quantities
    quantity_exact = {}
    for name, value in quantities.items():
        expected = expected_quantities[name]
        quantity_exact[name] = (
            (value.value, value.unit)
            == (expected.value, expected.unit)
        )
    return validation_records.ReferenceAgreement(
        side=candidate.side, task_exact=task_exact,
        quantity_exact=quantity_exact,
        level_exact={
            name: value.value == expected_assessments[name].value
            for name, value in assessed.items()
        },
    )


def _have_equal_rating_values(
    first: Mapping[str, judgments.Rating],
    second: Mapping[str, judgments.Rating],
) -> bool:
    """Compares named rating values and units, ignoring evidence wording."""
    return first.keys() == second.keys() and all(
        rating.value == second[name].value
        and rating.unit == second[name].unit
        for name, rating in first.items()
    )


def _have_equal_record_ratings(
    first: score_records.ScoreRecord,
    second: score_records.ScoreRecord,
) -> bool:
    """Compares one output's LM ratings, ignoring evidence and provenance."""
    return (
        first.output == second.output
        and _have_equal_rating_values(
            first.task_judgment.ratings, second.task_judgment.ratings,
        )
        and _have_equal_rating_values(
            first.output_judgment.ratings, second.output_judgment.ratings,
        )
        and _have_equal_rating_values(
            first.output_judgment.quantities, second.output_judgment.quantities,
        )
    )


@dataclasses.dataclass(frozen=True)
class _RepetitionExecution:
    """A repetition's report and the successfully loaded scoring records."""

    result: validation_records.RepetitionResult
    successful: tuple[score_records.ScoreRecord, ...]


def _run_repetition(
    judge: judge_interface.Judge,
    case: validation_records.ComparisonCase,
    user: config.UserConfig,
    run_root: str | pathlib.Path,
    component: str,
) -> _RepetitionExecution:
    """Scores both sides without caching and returns results and diagnostics."""
    run_dir = runs.get_component_directory(run_root, component)
    scorer = simulator.Simulator([user], judge, component=component)
    result = validation_records.RepetitionResult()
    for candidate in case.get_candidates():
        try:
            result.scores[candidate.side] = scorer.score(
                run_root=run_root, user_id=user.user_id,
                conversation_history=case.history, output=candidate.output,
                use_cache=False,
            )
        except Exception as error:
            # score() has already retried malformed responses and saved
            # their diagnostics. Record the failure and try the other output.
            result.failures[candidate.side] = f"{type(error).__name__}: {error}"
    successful = []
    for path in (run_dir / runs.REQUESTS_DIRECTORY).glob("*/score.json"):
        score_record = score_records.load_score_record(path)
        if score_record.status is score_records.ScoreStatus.OK:
            successful.append(score_record)
    if not result.failures:
        result.label = calculate_comparison_label(
            result.scores[case.first.side], result.scores[case.second.side]
        )
    return _RepetitionExecution(result, tuple(successful))


def _evaluate_case_sensitivity(
    case: validation_records.ComparisonCase,
    successful: tuple[score_records.ScoreRecord, ...],
    users: list[config.UserConfig],
) -> dict[str, dict[str, validation_records.SensitivityResult]]:
    """Evaluates sensitivity under each independently rated task context."""
    by_output = {item.output: item for item in successful}
    first = by_output[case.first.output].output_judgment.ratings
    second = by_output[case.second.output].output_judgment.ratings
    return {
        f"common_task_from_{candidate.side}": {
            user.user_id: _evaluate_sensitivity(
                user, by_output[candidate.output].task_judgment.ratings,
                first, second,
            )
            for user in users
        }
        for candidate in case.get_candidates()
    }


def _check_reference_ratings(
    case: validation_records.ComparisonCase,
    executions: list[_RepetitionExecution],
    case_report: validation_records.CaseReport,
) -> None:
    """Records which LM ratings match the authored values in the case file."""
    agreements = []
    for execution in executions:
        result = execution.result
        result.reference_agreement = [
            _compare_reference_ratings(case, item)
            for item in execution.successful
        ]
        agreements.extend(result.reference_agreement)
        if result.label is not None:
            result.matches_reference_label = (
                result.label == case_report.reference_label
            )
    case_report.reference_ratings_match = (
        len(agreements) == REPETITIONS * len(case.get_candidates())
        and all(
            all(item.task_exact.values())
            and all(item.level_exact.values())
            and all(item.quantity_exact.values())
            for item in agreements
        )
    )


def _check_repeated_ratings(
    executions: list[_RepetitionExecution],
    case_report: validation_records.CaseReport,
) -> None:
    """Records whether three successful repeats gave the same ratings/order."""
    completed = [
        execution for execution in executions
        if execution.result.label is not None
    ]
    labels = [execution.result.label for execution in completed]
    repeated_records = [
        sorted(execution.successful, key=lambda item: item.output)
        for execution in completed
    ]
    case_report.all_three_labels_consistent = (
        len(labels) == REPETITIONS and len(set(labels)) == 1
    )
    case_report.all_three_measurements_consistent = (
        len(repeated_records) == REPETITIONS
        and all(
            len(repeated) == len(repeated_records[0])
            and all(
                _have_equal_record_ratings(first, current)
                for first, current in zip(repeated_records[0], repeated)
            )
            for repeated in repeated_records[1:]
        )
    )
    case_report.label_counts = dict(collections.Counter(labels))


def _check_score_changes(
    case: validation_records.ComparisonCase,
    users: list[config.UserConfig],
    executions: list[_RepetitionExecution],
    case_report: validation_records.CaseReport,
) -> None:
    """Changes ratings/mappings in code and records whether A/B/tie changes.

    This makes no LM calls. A repeat with a failed output cannot be compared.
    """
    for repeat, execution in enumerate(executions):
        if execution.result.label is None:
            continue
        changes = _evaluate_case_sensitivity(case, execution.successful, users)
        case_report.sensitivity[str(repeat)] = changes
        for context in changes.values():
            if any(result.any_label_change for result in context.values()):
                case_report.score_order_changed = True


def run_evaluation(
    judge: judge_interface.Judge,
    cases_path: str | pathlib.Path,
    run_root: str | pathlib.Path,
    *,
    component: str = "judge_validation",
) -> validation_records.ValidationReport:
    """Runs optional example checks and saves their results.

    Use this in a smoke test or local benchmark. Production score() does not
    call it, and its results never stop training.

    First load the examples; a broken case file raises before any LM calls.
    Then score each example three times without caching. score() tries a
    malformed response three times; if it still fails, record the error here
    and continue with the remaining outputs and examples.

    Check reference ratings, repeat consistency, and changes in score order.
    A mismatch is a finding to save, not a reason to retry the LM or raise.
    COMPLETE means the calls completed, not that all checks passed.
    File-writing errors raise because the results could not be saved.
    """
    bundle = load_cases(cases_path)
    users = config.generate_users()
    report = validation_records.ValidationReport(
        case_bank_hash=bundle.source_hash, judge=judge.identity,
        human_review_status=bundle.human_review_status, repetitions=REPETITIONS,
    )
    runs.write_record(run_root, component, REPORT_FILENAME, report)
    for case in bundle.cases:
        user = case.user or users[0]
        reference_scores = [
            engine.calculate_preference_score(
                user, case.reference_task.ratings, candidate.reference.ratings,
            ).score
            for candidate in case.get_candidates()
        ]
        case_report = validation_records.CaseReport(
            id=case.id,
            reference_label=calculate_comparison_label(*reference_scores),
        )
        executions = [
            _run_repetition(
                judge, case, user, run_root,
                f"{component}/{case.id}/{repeat}/scoring",
            )
            for repeat in range(REPETITIONS)
        ]
        case_report.runs = [execution.result for execution in executions]
        if any(result.failures for result in case_report.runs):
            case_report.messages.append(
                "Some outputs could not be scored. Their errors are saved; "
                "check the successful outputs and continue to the next case."
            )

        _check_reference_ratings(case, executions, case_report)
        if case_report.reference_ratings_match:
            case_report.messages.append(
                "LM ratings match the case-file ratings."
            )
        else:
            case_report.messages.append(
                "LM ratings differ from the case file, or some are missing. "
                "Keep the differences in the report and continue."
            )

        _check_repeated_ratings(executions, case_report)
        if (
            case_report.all_three_measurements_consistent
            and case_report.all_three_labels_consistent
        ):
            case_report.messages.append(
                "All three repeats gave the same ratings and preference order."
            )
        else:
            case_report.messages.append(
                "Repeated ratings or preference orders changed, or a repeat "
                "failed. Keep the results and continue."
            )

        _check_score_changes(case, users, executions, case_report)
        if not case_report.sensitivity:
            case_report.messages.append(
                "Skipped the rating-change check: no repeat scored "
                "both outputs."
            )
        elif case_report.score_order_changed:
            case_report.messages.append(
                "Changing a rating or value mapping changed which output "
                "was preferred. This case is sensitive to those choices."
            )
        else:
            case_report.messages.append(
                "The tested rating and value-mapping changes kept the same "
                "preference order."
            )

        report.cases.append(case_report)
        runs.write_record(
            run_root, component, f"{case.id}/case.json", case_report
        )
        runs.write_record(run_root, component, REPORT_FILENAME, report)

    if any(run.failures for case in report.cases for run in case.runs):
        report.status = (
            validation_records.ValidationStatus.COMPLETE_WITH_FAILURES
        )
    else:
        report.status = validation_records.ValidationStatus.COMPLETE
    report.interpretation = (
        "These are optional example checks, not a training gate. Agreement "
        "with unreviewed case-file ratings does not prove the LM is correct."
    )
    runs.write_record(run_root, component, REPORT_FILENAME, report)
    return report
