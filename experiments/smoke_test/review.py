"""Explains real smoke-test results and checks authored expectations."""

import math
import pathlib

from experiment_logging import runs
from experiments.smoke_test import cases
from user_simulator.data import score_records


def load_output_record(
    run_root: pathlib.Path, output: str,
) -> tuple[pathlib.Path, score_records.ScoreRecord]:
    """Finds the score record for one unique output in this fresh smoke run."""
    root = runs.get_component_directory(run_root, "user_simulator")
    for path in (root / runs.REQUESTS_DIRECTORY).glob("*/score.json"):
        record = score_records.load_score_record(path)
        if record.output == output:
            return path, record
    raise FileNotFoundError("No score record was saved for this output")


def review_output(
    case: cases.SmokeCase, record: score_records.ScoreRecord, score: float,
) -> tuple[str, list[str]]:
    """Returns a readable calculation and any smoke-only check failures.

    Expected ratings remain in this module's caller; they never enter judge
    prompts or production records. Arithmetic is checked directly from LM
    ratings and config weights, without calling the scoring engine again.
    """
    failures = []
    rule = next(
        rule for rule in record.user_config.rules
        if rule.id == cases.EXPECTED_RULE
    )
    task_urgency = record.task_judgment.ratings["urgency"]
    if task_urgency.value is None or task_urgency.value < rule.when.threshold:
        failures.append("LM task urgency did not activate the urgent rule.")
    if record.result.selected_rule != rule.id:
        failures.append(f"Expected rule {rule.id} was not selected.")
    if record.result.weights != dict(rule.weights):
        failures.append("Logged weights differ from the expected config rule.")

    expected_score = 100 * math.fsum(
        weight * case.expected_ratings[name] / 4
        for name, weight in rule.weights.items()
    ) / math.fsum(rule.weights.values())
    lines = [
        f"Output: {case.name}", case.output, "",
        f"User: {record.user_id}",
        f"Task urgency: {task_urgency.value}/100",
        f"Task evidence: {task_urgency.evidence}",
        f"Selected rule: {record.result.selected_rule}",
        f"Config weights: {dict(rule.weights)}", "",
    ]
    terms = []
    values = []
    for name, weight in rule.weights.items():
        rating = record.output_judgment.ratings[name]
        expected = case.expected_ratings[name]
        lines.append(
            f"{name}: LM {rating.value}/4; expected {expected}/4; "
            f"weight {weight}; evidence {rating.evidence}"
        )
        if rating.value != expected:
            failures.append(
                f"LM {name} rating was {rating.value}; expected {expected}."
            )
        if rating.value is None:
            failures.append(f"Cannot manually calculate unknown {name} rating.")
            continue
        value = rating.value / 4
        values.append(weight * value)
        terms.append(f"{weight} * ({rating.value}/4)")
        if not math.isclose(record.result.component_values[name], value):
            failures.append(f"Logged mapped value for {name} is incorrect.")
    if len(values) == len(rule.weights):
        calculated = 100 * math.fsum(values) / math.fsum(rule.weights.values())
        lines.append(
            f"Calculation: 100 * ({' + '.join(terms)}) / "
            f"{math.fsum(rule.weights.values())} = {calculated:.2f}"
        )
        if not math.isclose(score, calculated, abs_tol=1e-9):
            failures.append("Returned score differs from manual arithmetic.")
    if not math.isclose(score, record.result.score, abs_tol=1e-9):
        failures.append("Returned score differs from the logged score.")
    if not math.isclose(score, expected_score, abs_tol=1e-9):
        failures.append(f"Score differs from expected {expected_score:.2f}.")
    lines.extend((
        f"Expected score: {expected_score:.2f}/100",
        f"Actual score: {score:.2f}/100",
        f"Scoring time: {record.elapsed_seconds:.2f}s"
        if record.elapsed_seconds is not None else "Scoring time: unavailable",
        f"Task LM cache reused: {record.task_cache_hit}",
        f"Output LM cache reused: {record.output_cache_hit}",
        f"LM calls: task={len(record.task_lm_calls)}, "
        f"output={len(record.output_lm_calls)}",
        f"Result: {'FAIL' if failures else 'PASS'}",
    ))
    lines.extend(f"- {failure}" for failure in failures)
    return "\n".join(lines) + "\n", failures
