"""Tests reference diagnostics without model calls or holdout access."""

import dataclasses
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from conversation import history
from experiment_logging import runs
from experiments.smoke_test import cases
from experiments.smoke_test import diagnostics
from user_simulator import config
from user_simulator import engine
from user_simulator import test_utils
from user_simulator.data import judgments
from user_simulator.data import score_records


class DiagnosticsTest(unittest.TestCase):
    """Protects criterion errors, eligible pairs, and missing denominators."""

    def setUp(self) -> None:
        user = config.UserConfig(
            user_id="diagnostics_test",
            rules=(config.PreferenceRule(
                id="always", precedence=0, when=None,
                weights={"urgency": 0.5, "uncertainty": 0.5},
            ),),
        )
        inputs = []
        for index, value in enumerate((0, 2, 4, 2, 2)):
            task = "shared task" if index < 3 else f"other task {index}"
            inputs.append(cases.SmokeCase(
                name=f"case_{index}",
                history=history.ConversationHistory(messages=(history.Message(
                    role=history.Role.USER, content=task,
                ),)),
                output=f"output {index}", expected_rule="always",
                expected_ratings={"urgency": value, "uncertainty": value},
                explanation="Synthetic reference for diagnostics unit tests.",
            ))
        self.group = cases.UserCases(user, "Test user", tuple(inputs))
        self.saved = [self._make_record(case) for case in self.group.cases]

    def _make_record(self, case: cases.SmokeCase) -> score_records.ScoreRecord:
        output = judgments.LMJudgment(test_utils.make_ratings(
            urgency=case.expected_ratings["urgency"],
            uncertainty=case.expected_ratings["uncertainty"],
        ))
        task = judgments.LMJudgment(test_utils.make_task())
        return score_records.ScoreRecord(
            user_id=self.group.user.user_id, history=case.history,
            output=case.output, judge=test_utils.make_judge_identity(),
            prompt_hash="test", user_config=self.group.user,
            task_cache_key="test", timestamp="test", task_judgment=task,
            output_judgment=output,
            result=engine.calculate_preference_score(
                self.group.user, task.ratings, output.ratings,
            ),
            status=score_records.ScoreStatus.OK,
        )

    def test_compensating_criterion_errors_leave_combined_score_exact(
        self,
    ) -> None:
        record = self.saved[1]
        record.output_judgment.ratings["urgency"].value = 3
        record.output_judgment.ratings["uncertainty"].value = 1
        record.result = engine.calculate_preference_score(
            self.group.user, record.task_judgment.ratings,
            record.output_judgment.ratings,
        )
        report = diagnostics.calculate_diagnostics((self.group,), self.saved)
        self.assertEqual(report.combined_score.mean_absolute_error, 0)
        self.assertEqual(report.combined_score.agreement.correct, 5)
        for metric in report.criteria.values():
            self.assertEqual(metric.agreement, diagnostics.Agreement(5, 5, 4))
            self.assertEqual(metric.mean_absolute_error, 0.2)
        self.assertEqual(report.rule_selection, diagnostics.Agreement(5, 5, 5))

    def test_pools_criterion_labels_instead_of_averaging_criterion_means(
        self,
    ) -> None:
        report = diagnostics.calculate_diagnostics((self.group,), self.saved)
        report = dataclasses.replace(report, criteria={
            "urgency": diagnostics.NumericError(
                diagnostics.Agreement(1, 1, 0), 1.0,
            ),
            "uncertainty": diagnostics.NumericError(
                diagnostics.Agreement(9, 9, 9), 0.0,
            ),
        })
        pooled = report.calculate_criterion_error()
        self.assertEqual(pooled.agreement, diagnostics.Agreement(10, 10, 9))
        self.assertEqual(pooled.mean_absolute_error, 0.1)
        self.assertEqual(report.macro_criterion_mean_absolute_error, 0.5)

    def test_boolean_ratings_are_excluded_from_numeric_coverage(self) -> None:
        for value in (True, False):
            with self.subTest(value=value):
                self.saved[0].output_judgment.ratings["urgency"].value = value
                report = diagnostics.calculate_diagnostics(
                    (self.group,), self.saved,
                )
                self.assertEqual(report.criteria["urgency"].agreement,
                                 diagnostics.Agreement(5, 4, 4))
                self.assertEqual(report.calculate_criterion_error().agreement,
                                 diagnostics.Agreement(10, 9, 9))

    def test_boolean_scores_are_excluded_from_scores_and_ordering(
        self,
    ) -> None:
        for value in (True, False):
            with self.subTest(value=value):
                self.saved[0].result = dataclasses.replace(
                    self.saved[0].result, score=value,
                )
                report = diagnostics.calculate_diagnostics(
                    (self.group,), self.saved,
                )
                self.assertEqual(report.combined_score.agreement,
                                 diagnostics.Agreement(5, 4, 4))
                self.assertEqual(report.ordering,
                                 diagnostics.Agreement(3, 1, 1))

    def test_pooled_mae_uses_available_labels_and_retains_missing_total(
        self,
    ) -> None:
        self.saved[0].output_judgment.ratings["urgency"].value = None
        self.saved[1].output_judgment.ratings["urgency"].value = 3
        report = diagnostics.calculate_diagnostics((self.group,), self.saved)
        pooled = report.calculate_criterion_error()
        self.assertEqual(pooled.agreement, diagnostics.Agreement(10, 9, 8))
        self.assertAlmostEqual(pooled.mean_absolute_error, 1 / 9)

    def test_reversals_and_observed_ties_are_disagreements(self) -> None:
        for record, value in zip(self.saved, (100, 50, 50)):
            record.result = dataclasses.replace(record.result, score=value)
        report = diagnostics.calculate_diagnostics((self.group,), self.saved)
        self.assertEqual(report.ordering, diagnostics.Agreement(3, 3, 0))
        self.assertEqual([pair.actual for pair in report.pairs], [
            diagnostics.Ordering.GREATER, diagnostics.Ordering.GREATER,
            diagnostics.Ordering.TIE,
        ])

    def test_nearby_distinct_scores_remain_strictly_ordered(self) -> None:
        for record, value in zip(self.saved, (50.0, 50.0 + 1e-10, 100.0)):
            record.result = dataclasses.replace(record.result, score=value)
        report = diagnostics.calculate_diagnostics((self.group,), self.saved)
        self.assertEqual(report.pairs[0].actual, diagnostics.Ordering.LESS)

    def test_reference_ties_must_also_be_observed_ties(self) -> None:
        inputs = list(self.group.cases)
        inputs[1] = dataclasses.replace(inputs[1], expected_ratings={
            "urgency": 0, "uncertainty": 0,
        })
        group = dataclasses.replace(self.group, cases=tuple(inputs))
        report = diagnostics.calculate_diagnostics((group,), self.saved)
        self.assertEqual(report.pairs[0].expected, diagnostics.Ordering.TIE)
        self.assertEqual(report.pairs[0].actual, diagnostics.Ordering.LESS)
        self.assertEqual(report.ordering.correct, 2)
        self.saved[1].result = dataclasses.replace(
            self.saved[1].result, score=0,
        )
        report = diagnostics.calculate_diagnostics((group,), self.saved)
        self.assertEqual(report.ordering.correct, 3)
        self.assertEqual(report.pairs[0].actual, diagnostics.Ordering.TIE)

    def test_never_compares_different_histories_or_users(self) -> None:
        other_user = dataclasses.replace(self.group.user, user_id="other_user")
        other_group = dataclasses.replace(self.group, user=other_user)
        other_saved = [dataclasses.replace(
            record, user_id=other_user.user_id, user_config=other_user,
        ) for record in self.saved]
        report = diagnostics.calculate_diagnostics(
            (self.group, other_group), self.saved + other_saved,
        )
        self.assertEqual(report.ordering, diagnostics.Agreement(6, 6, 6))
        for pair in report.pairs:
            self.assertNotIn(pair.first_case, ("case_3", "case_4"))
            self.assertNotIn(pair.second_case, ("case_3", "case_4"))

    def test_failed_and_missing_cases_remain_in_applicable_totals(self) -> None:
        record = self.saved[0]
        record.status = score_records.ScoreStatus.FAILURE
        record.result = None
        record.output_judgment.ratings["urgency"].value = 1
        del record.output_judgment.ratings["uncertainty"]
        report = diagnostics.calculate_diagnostics(
            (self.group,), [record, *self.saved[2:]],
        )
        self.assertEqual(report.rule_selection, diagnostics.Agreement(5, 3, 3))
        self.assertEqual(report.combined_score.agreement,
                         diagnostics.Agreement(5, 3, 3))
        self.assertEqual(report.combined_score.mean_absolute_error, 0)
        self.assertEqual(report.criteria["urgency"].agreement,
                         diagnostics.Agreement(5, 4, 3))
        self.assertEqual(report.criteria["urgency"].mean_absolute_error, 0.25)
        self.assertEqual(report.criteria["uncertainty"].agreement,
                         diagnostics.Agreement(5, 3, 3))
        self.assertEqual(report.ordering, diagnostics.Agreement(3, 0, 0))
        self.assertEqual(report.failed_cases, ("diagnostics_test/case_0",))
        self.assertEqual(report.missing_cases, ("diagnostics_test/case_1",))

    def test_task_rule_remains_visible_when_output_judging_fails(
        self,
    ) -> None:
        rule = config.PreferenceRule(
            id="urgent", precedence=1,
            when=config.Condition(
                property="urgency", threshold=80,
                operator=config.ConditionOperator.GREATER_EQUAL,
            ),
            weights={"urgency": 0.5, "uncertainty": 0.5},
        )
        user = dataclasses.replace(
            self.group.user, rules=(*self.group.user.rules, rule),
        )
        group = dataclasses.replace(
            self.group, user=user, cases=tuple(dataclasses.replace(
                case, expected_rule="urgent",
            ) for case in self.group.cases),
        )
        for saved in self.saved:
            saved.user_config = user
            saved.task_judgment.ratings["urgency"].value = 95
            saved.result = dataclasses.replace(
                saved.result, selected_rule="urgent",
            )
        record = self.saved[0]
        record.status = score_records.ScoreStatus.FAILURE
        record.result = None
        record.output_judgment = None
        report = diagnostics.calculate_diagnostics((group,), self.saved)
        self.assertEqual(report.rule_selection, diagnostics.Agreement(5, 4, 4))
        self.assertEqual(report.task_rule_selection,
                         diagnostics.Agreement(5, 5, 5))
        record.task_judgment = None
        report = diagnostics.calculate_diagnostics((group,), self.saved)
        self.assertEqual(report.task_rule_selection,
                         diagnostics.Agreement(5, 4, 4))

    def test_no_records_has_unavailable_mae_and_zero_coverage(self) -> None:
        report = diagnostics.calculate_diagnostics((self.group,), ())
        self.assertIsNone(report.combined_score.mean_absolute_error)
        pooled = report.calculate_criterion_error()
        self.assertIsNone(pooled.mean_absolute_error)
        self.assertIsNone(report.macro_criterion_mean_absolute_error)
        self.assertEqual(pooled.agreement, diagnostics.Agreement(10, 0, 0))
        self.assertEqual(report.combined_score.agreement,
                         diagnostics.Agreement(5, 0, 0))
        self.assertEqual(report.ordering, diagnostics.Agreement(3, 0, 0))
        self.assertIn("MAE unavailable", diagnostics.format_diagnostics(report))

    def test_rejects_ambiguous_records_and_changed_user_configuration(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "Multiple saved"):
            diagnostics.calculate_diagnostics(
                (self.group,), [*self.saved, self.saved[0]],
            )
        self.saved[0].user_config = test_utils.make_conditional_user()
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            diagnostics.calculate_diagnostics((self.group,), self.saved)

    def test_keeps_historical_ratings_and_discloses_definition_changes(
        self,
    ) -> None:
        record = self.saved[0]
        criteria = list(record.user_config.rating_criteria)
        anchors = list(criteria[0].anchors)
        anchors[0] = "Earlier wording of the same anchor."
        criteria[0] = dataclasses.replace(criteria[0], anchors=tuple(anchors))
        record.user_config = dataclasses.replace(
            record.user_config, rating_criteria=tuple(criteria),
        )
        report = diagnostics.calculate_diagnostics((self.group,), self.saved)
        self.assertEqual(report.changed_rating_definitions,
                         (self.group.user.user_id,))
        self.assertEqual(report.combined_score.mean_absolute_error, 0)
        self.assertEqual(report.criteria["urgency"].agreement.correct, 5)

    def test_writes_diagnostics_from_only_supplied_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            paths = [runs.write_record(
                root, "user_simulator", f"requests/{index}/score.json", record,
            ) for index, record in enumerate(self.saved)]
            # Unreferenced scores exist, but only this split's paths count.
            report = diagnostics.write_diagnostics(
                root, cases.Split.DEVELOPMENT, (self.group,),
                [paths[0], root / "missing-score.json"],
            )
            self.assertEqual(report.combined_score.agreement,
                             diagnostics.Agreement(5, 1, 1))
            self.assertEqual(report.missing_cases, tuple(
                f"diagnostics_test/case_{index}" for index in range(1, 5)
            ))
            folder = runs.get_component_directory(root, "smoke_test")
            self.assertFalse((folder / "development/results.json").exists())
            self.assertTrue((folder / "development/diagnostics.json").exists())
            self.assertTrue((folder / "development/diagnostics.txt").exists())

    def test_archived_review_ignores_error_paths_and_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            paths = [runs.write_record(
                root, "user_simulator", f"requests/{index}/score.json", record,
            ) for index, record in enumerate(self.saved)]
            runs.write_record(root, "smoke_test", "development/results.json", {
                "cases": [
                    {"score_record": str(paths[0])},
                    {"errors": [f"LM request failed: {paths[1]}"]},
                    {"score_record": None},
                    {"score_record": str(root / "missing-score.json")},
                ],
            })
            runs.write_record(root, "smoke_test", "holdout/results.json", {
                "cases": [{"score_record": str(path)} for path in paths[1:]],
            })
            with mock.patch.object(cases, "load_cases",
                                   return_value=(self.group,)) as load_cases:
                report = diagnostics.review_saved_development(root)
            load_cases.assert_called_once_with(cases.Split.DEVELOPMENT)
            self.assertEqual(report.combined_score.agreement,
                             diagnostics.Agreement(5, 1, 1))
            self.assertEqual(len(report.missing_cases), 4)

    def test_saved_development_helper_writes_both_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            outcomes = []
            for index, record in enumerate(self.saved):
                path = runs.write_record(
                    root, "user_simulator", f"requests/{index}/score.json",
                    record,
                )
                outcomes.append({"score_record": str(path)})
            runs.write_record(root, "smoke_test", "development/results.json", {
                "cases": outcomes,
            })
            with mock.patch.object(cases, "load_cases",
                                   return_value=(self.group,)) as load_cases:
                diagnostics.review_saved_development(root)
            load_cases.assert_called_once_with(cases.Split.DEVELOPMENT)
            folder = runs.get_component_directory(root, "smoke_test")
            saved = json.loads((folder / "development/diagnostics.json")
                               .read_text())
            self.assertEqual(saved["ordering"]["total"], 3)
            summary = (folder / "development/diagnostics.txt").read_text()
            self.assertIn("coverage 5/5", summary)


if __name__ == "__main__":
    unittest.main()
