"""Evaluates development, with a separate guarded holdout evaluation entry."""

import dataclasses
import json
import math
import pathlib

from loguru import logger as log

from experiment_logging import records
from experiment_logging import runs
from experiments.smoke_test import cases
from experiments.smoke_test import diagnostics
from experiments.smoke_test import review
from user_simulator import config
from user_simulator import simulator
from user_simulator.judge import judge as judge_interface
from user_simulator.judge import prompts
from user_simulator.judge.ratings import output_definitions

COMPONENT = "smoke_test"
RELEASE_FILENAME = "holdout_release.json"
MAX_CRITERION_MAE = 0.20


@dataclasses.dataclass(frozen=True)
class CaseResult:
    """A test outcome with its authored target and saved production record."""

    user_id: str
    case_name: str
    expected_score: float
    score: float | None
    errors: tuple[str, ...]
    score_record: str | None


@dataclasses.dataclass(frozen=True)
class SplitResult:
    """Saved scores, with exact reference agreement retained as a diagnostic."""

    split: cases.Split
    correct: int
    total: int
    cases: tuple[CaseResult, ...]
    criterion_error: diagnostics.NumericError | None = None

    def calculate_mean_absolute_error(self) -> float | None:
        """Returns combined-score MAE as a diagnostic over finite scores."""
        errors = [abs(case.score - case.expected_score) for case in self.cases
                  if config.is_number(case.score)]
        return math.fsum(errors) / len(errors) if errors else None

    def has_passed(self) -> bool:
        """Requires complete scores/labels and criterion MAE <=0.20 of 4."""
        expected_total = cases.USERS_PER_SPLIT * cases.CASES_PER_USER
        if self.total != expected_total or len(self.cases) != expected_total:
            return False
        if any(not config.is_number(case.score)
               or not 0 <= case.score <= 100 for case in self.cases):
            return False
        if self.criterion_error is None:
            return False
        coverage = self.criterion_error.agreement
        error = self.criterion_error.mean_absolute_error
        return (
            coverage.total > 0 and coverage.available == coverage.total
            and error is not None and 0 <= error <= MAX_CRITERION_MAE
        )


@dataclasses.dataclass(frozen=True)
class HoldoutRelease:
    """Pins the first development-qualified judge and frozen fixture set."""

    prompt_hash: str
    judge_hash: str
    fixture_hash: str
    first_run: str
    max_criterion_mae: float = MAX_CRITERION_MAE


def _run_case(
    scorer: simulator.Simulator, group: cases.UserCases, case: cases.SmokeCase,
    run_root: pathlib.Path, split: cases.Split,
) -> CaseResult:
    """Scores one output and records exact agreement without score retries."""
    rule = next(rule for rule in group.user.rules
                if rule.id == case.expected_rule)
    expected_score = 100 * math.fsum(
        weight * case.expected_ratings[name] / 4
        for name, weight in rule.weights.items() if weight > 0
    ) / math.fsum(rule.weights.values())
    score = None
    score_path = None
    try:
        score = scorer.score(
            run_root=run_root, user_id=group.user.user_id,
            conversation_history=case.history, output=case.output,
        )
        score_path, record = review.load_output_record(
            run_root, case.output, group.user.user_id,
        )
        if record.user_config != group.user:
            errors = ["Scoring used a different user configuration."]
            report = "Result: FAIL\n" + "\n".join(errors) + "\n"
            score = None
        else:
            report, errors = review.review_output(case, record, score)
        report += f"Score record: {score_path}\n"
    except simulator.JudgeError as error:
        score_path = error.diagnostics
        errors = [f"LM request failed: {error}; {score_path}"]
        report = "\n".join(errors) + "\n"
    result = CaseResult(
        user_id=group.user.user_id, case_name=case.name,
        expected_score=expected_score, score=score, errors=tuple(errors),
        score_record=str(score_path) if score_path else None,
    )
    runs.write_text(
        run_root, COMPONENT,
        f"{split}/{group.user.user_id}/{case.name}.txt", report,
    )
    log.log("WARNING" if errors else "INFO", "\n{}", report)
    return result


def _run_split(
    split: cases.Split, judge: judge_interface.Judge, run_root: pathlib.Path,
) -> SplitResult:
    """Loads and scores one group; every case stays in the denominator."""
    groups = cases.load_cases(split)
    scorer = simulator.Simulator(
        [group.user for group in groups], judge,
    )
    outcomes = []
    for group in groups:
        log.info("{}: {} — {}", split, group.user.user_id, group.description)
        for case in group.cases:
            log.info("Scoring {}/{}", group.user.user_id, case.name)
            outcomes.append(_run_case(scorer, group, case, run_root, split))
        correct = sum(not item.errors for item in outcomes
                      if item.user_id == group.user.user_id)
        log.info(
            "{}: {}/5 exact reference matches", group.user.user_id, correct,
        )
    report = diagnostics.write_diagnostics(
        run_root, split, groups,
        [pathlib.Path(item.score_record) for item in outcomes
         if item.score_record is not None],
    )
    result = SplitResult(
        split=split, correct=sum(not item.errors for item in outcomes),
        total=len(outcomes), cases=tuple(outcomes),
        criterion_error=report.calculate_criterion_error(),
    )
    runs.write_record(run_root, COMPONENT, f"{split}/results.json", result)
    error = result.criterion_error.mean_absolute_error
    available = sum(config.is_number(item.score) for item in outcomes)
    summary = [
        f"{split}: criterion MAE {error} on 0–4; "
        f"target <= {MAX_CRITERION_MAE}",
        f"Criterion coverage: {result.criterion_error.agreement.available}/"
        f"{result.criterion_error.agreement.total}",
        f"Combined score MAE (diagnostic): "
        f"{result.calculate_mean_absolute_error()}",
        f"Score coverage: {available}/{result.total}",
        "Exact reference matches (diagnostic): "
        f"{result.correct}/{result.total}",
    ]
    for group in groups:
        correct = sum(not item.errors for item in outcomes
                      if item.user_id == group.user.user_id)
        summary.append(f"{group.user.user_id}: {correct}/5")
    summary.append(f"MAE gate: {'PASS' if result.has_passed() else 'FAIL'}")
    runs.write_text(
        run_root, COMPONENT, f"{split}/summary.txt", "\n".join(summary),
    )
    log.info("\n{}", "\n".join(summary))
    return result


def _release_holdout(
    judge: judge_interface.Judge, run_root: pathlib.Path,
    release_path: pathlib.Path,
) -> None:
    """Prevents testing a revised judge on an already exposed holdout."""
    release = HoldoutRelease(
        prompt_hash=prompts.compute_prompt_fingerprint(
            output_definitions.DEFAULT_RATING_CRITERIA,
        ),
        judge_hash=records.compute_digest(judge.identity),
        fixture_hash=records.compute_digest(json.loads(
            (cases.DATA_DIRECTORY / "manifest.json").read_text()
        )),
        first_run=str(run_root), max_criterion_mae=MAX_CRITERION_MAE,
    )
    if release_path.exists():
        saved = HoldoutRelease(**json.loads(release_path.read_text()))
        if dataclasses.replace(release, first_run=saved.first_run) != saved:
            raise ValueError(
                "Holdout was already exposed to a different judge, prompt, "
                "fixture set, or MAE target. Use a new independently authored "
                "holdout."
            )
        log.info("Repeating frozen holdout evaluation from {}", saved.first_run)
    else:
        records.write_json(release_path, release, exclusive=True)


def run_development_evaluation(
    judge: judge_interface.Judge, run_root: pathlib.Path,
) -> bool:
    """Runs development diagnostics without loading or releasing holdout."""
    development = _run_split(cases.Split.DEVELOPMENT, judge, run_root)
    passed = development.has_passed()
    error = (development.criterion_error.mean_absolute_error
             if development.criterion_error is not None else None)
    message = (
        f"Development criterion MAE: {error}\n"
        f"Required MAE: <= {MAX_CRITERION_MAE} on 0–4, "
        "with all scores and reference ratings available\n"
        f"Development: {'PASS' if passed else 'FAIL'}"
    )
    runs.write_text(run_root, COMPONENT, "summary.txt", message)
    log.log("SUCCESS" if passed else "WARNING", message)
    return passed


def run_evaluation(
    judge: judge_interface.Judge, run_root: pathlib.Path,
    release_path: pathlib.Path,
) -> bool:
    """Runs holdout only after development passes, using the same judge.

    Each split requires all twenty scores, all authored criterion ratings,
    and pooled criterion MAE <=0.20 on the 0–4 scale (5 normalized points).
    Exact criterion/rule agreement and ordering remain diagnostics. Failed
    requests block acceptance; expected labels never enter LM prompts.
    """
    development = _run_split(cases.Split.DEVELOPMENT, judge, run_root)
    if not development.has_passed():
        message = (
            f" MAE <= {MAX_CRITERION_MAE} on the 0–4 scale."
        )
        runs.write_text(run_root, COMPONENT, "summary.txt", message)
        log.warning(message)
        return False
    _release_holdout(judge, run_root, release_path)
    holdout = _run_split(cases.Split.HOLDOUT, judge, run_root)
    message = (
        f"Development criterion MAE: "
        f"{development.criterion_error.mean_absolute_error}\n"
        f"Holdout criterion MAE: "
        f"{holdout.criterion_error.mean_absolute_error}\n"
        f"Required MAE: <= {MAX_CRITERION_MAE} on 0–4, "
        "with all scores and reference ratings available\n"
        f"Overall: {'PASS' if holdout.has_passed() else 'FAIL'}"
    )
    runs.write_text(run_root, COMPONENT, "summary.txt", message)
    log.log("SUCCESS" if holdout.has_passed() else "WARNING", message)
    return holdout.has_passed()
