"""Tests for judge_evaluation."""

import dataclasses
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from user_simulator import test_utils
from user_simulator.data import score_records
from user_simulator.integration_tests import judge_evaluation
from user_simulator.integration_tests import validation_records
from user_simulator.judge import prompts
from user_simulator.judge.ratings import output_definitions


class _ReferenceJudge:
    """A test judge that returns the authored fixture ratings."""

    def __init__(
        self, cases: tuple[validation_records.ComparisonCase, ...]
    ) -> None:
        self.identity = test_utils.make_judge_identity(
            backend="reference-fixture-replay", revision="v1",
        )
        self.calls = []
        self._lookup = {}
        for case in cases:
            task_prompt = prompts.build_task_prompt(case.history)
            self._lookup[task_prompt] = test_utils.serialize_lm_response(
                case.reference_task
            )
            for candidate in case.get_candidates():
                prompt = prompts.build_output_prompt(
                    case.history, candidate.output,
                    output_definitions.DEFAULT_RATING_CRITERIA,
                )
                self._lookup[prompt] = test_utils.serialize_lm_response(
                    candidate.reference
                )

    def complete(self, prompt: str) -> str:
        """Returns the pre-authored reference for this exact prompt."""
        self.calls.append(prompt)
        return self._lookup[prompt]


class ValidationTest(unittest.TestCase):
    """Checks fixture audits without inferring real model quality."""

    def _make_score_record(self) -> score_records.ScoreRecord:
        """Builds typed LM results for repeat-consistency comparisons."""
        case = judge_evaluation.load_cases(test_utils.CASES_PATH).cases[0]
        return score_records.ScoreRecord(
            user_id="pilot_00",
            history=case.history,
            output=case.first.output,
            judge=test_utils.make_judge_identity(),
            prompt_hash="test_prompt",
            user_config=test_utils.make_conditional_user(),
            task_cache_key="test_task",
            timestamp="test_timestamp",
            task_judgment=case.reference_task,
            output_judgment=case.first.reference,
        )

    def test_consistency_ignores_evidence_and_log_provenance(self) -> None:
        first = self._make_score_record()
        second = self._make_score_record()
        second.timestamp = "another_timestamp"
        for judgment in (second.task_judgment, second.output_judgment):
            judgment.source_record = "another_record"
            for rating in (*judgment.ratings.values(),
                           *judgment.quantities.values()):
                rating.evidence = ["Different supporting quote"]
        self.assertTrue(
            judge_evaluation._have_equal_record_ratings(first, second)
        )

    def test_consistency_detects_changed_ratings_units_and_properties(
        self,
    ) -> None:
        first = self._make_score_record()
        self.assertFalse(judge_evaluation._have_equal_record_ratings(
            first, dataclasses.replace(first, output="Different output"),
        ))
        for group_index in range(3):
            for change in ("value", "unit", "property"):
                with self.subTest(group=group_index, change=change):
                    second = self._make_score_record()
                    groups = (
                        second.task_judgment.ratings,
                        second.output_judgment.ratings,
                        second.output_judgment.quantities,
                    )
                    ratings = groups[group_index]
                    name = next(iter(ratings))
                    if change == "value":
                        ratings[name].value = (
                            1 if ratings[name].value is None else None
                        )
                    elif change == "unit":
                        ratings[name].unit = "different_unit"
                    else:
                        del ratings[name]
                    self.assertFalse(
                        judge_evaluation._have_equal_record_ratings(
                            first, second
                        )
                    )

    def test_reference_replay_verifies_harness_only(self) -> None:
        bundle = judge_evaluation.load_cases(test_utils.CASES_PATH)
        judge = _ReferenceJudge(bundle.cases)
        with tempfile.TemporaryDirectory() as directory:
            report = judge_evaluation.run_evaluation(
                judge, test_utils.CASES_PATH, directory
            )
            self.assertEqual(report.status, "complete")
            self.assertIn("pending", report.human_review_status)
            self.assertEqual(len(judge.calls), 12 * 3 * 4)
            self.assertFalse(list(pathlib.Path(directory).rglob("task_cache")))
            self.assertFalse(
                list(pathlib.Path(directory).rglob("output_cache"))
            )
            for case in report.cases:
                self.assertTrue(case.reference_ratings_match)
                self.assertTrue(case.messages)
                self.assertTrue(case.all_three_labels_consistent)
                self.assertTrue(case.all_three_measurements_consistent)
                for run in case.runs:
                    self.assertTrue(run.matches_reference_label)
                    for agreement in run.reference_agreement:
                        self.assertTrue(all(agreement.task_exact.values()))
                        self.assertTrue(all(agreement.level_exact.values()))
                        self.assertTrue(
                            all(agreement.quantity_exact.values())
                        )

    def test_failed_lm_response_is_saved_and_other_cases_continue(self) -> None:
        bundle = judge_evaluation.load_cases(test_utils.CASES_PATH)
        judge = _ReferenceJudge(bundle.cases)
        complete = judge.complete

        def respond(prompt: str) -> str:
            raw = complete(prompt)
            return "invalid JSON" if len(judge.calls) <= 3 else raw

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(judge, "complete", side_effect=respond):
                report = judge_evaluation.run_evaluation(
                    judge, test_utils.CASES_PATH, directory,
                )
            self.assertEqual(report.status, "complete_with_failures")
            self.assertEqual(len(report.cases), len(bundle.cases))
            failed_case = report.cases[0]
            self.assertIn(
                validation_records.CandidateSide.A,
                failed_case.runs[0].failures,
            )
            self.assertIn(
                validation_records.CandidateSide.B,
                failed_case.runs[0].scores,
            )
            self.assertNotIn("0", failed_case.sensitivity)
            self.assertIn("1", failed_case.sensitivity)
            self.assertFalse(failed_case.all_three_measurements_consistent)
            self.assertTrue(all(
                not run.failures
                for case in report.cases[1:] for run in case.runs
            ))
            saved = json.loads(next(
                pathlib.Path(directory).rglob(judge_evaluation.REPORT_FILENAME)
            ).read_text())
            self.assertEqual(saved["status"], "complete_with_failures")
            self.assertTrue(saved["cases"][0]["runs"][0]["failures"])

    def test_reference_mismatch_is_reported_without_retry(self) -> None:
        bundle = judge_evaluation.load_cases(test_utils.CASES_PATH)
        judge = _ReferenceJudge(bundle.cases)
        case = bundle.cases[0]
        prompt = prompts.build_output_prompt(
            case.history, case.first.output,
            output_definitions.DEFAULT_RATING_CRITERIA,
        )
        response = json.loads(judge._lookup[prompt])
        response["ratings"]["urgency"]["value"] = 3
        judge._lookup[prompt] = json.dumps(response)
        with tempfile.TemporaryDirectory() as directory:
            report = judge_evaluation.run_evaluation(
                judge, test_utils.CASES_PATH, directory,
            )
        self.assertEqual(report.status, "complete")
        self.assertEqual(len(judge.calls), 12 * 3 * 4)
        self.assertFalse(report.cases[0].reference_ratings_match)
        self.assertTrue(report.cases[0].all_three_measurements_consistent)
        self.assertTrue(any(
            "differ from the case file" in message
            for message in report.cases[0].messages
        ))
        self.assertEqual(len(report.cases), len(bundle.cases))

    def test_threshold_and_level_sensitivity_detect_changes(self) -> None:
        result = judge_evaluation._evaluate_sensitivity(
            test_utils.make_conditional_user(), test_utils.make_task(79),
            test_utils.make_ratings(urgency=4, money=0),
            test_utils.make_ratings(urgency=0, money=4),
        )
        self.assertTrue(result.any_label_change)
        self.assertIn(
            "A", [item.label for item in result.condition_perturbations]
        )


if __name__ == "__main__":
    unittest.main()
