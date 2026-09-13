"""Checks holdout gating without accessing holdout fixtures or a real model."""

import dataclasses
import json
import math
import pathlib
import tempfile
import unittest
from unittest import mock

from experiments.smoke_test import cases
from experiments.smoke_test import diagnostics
from experiments.smoke_test import evaluation
from user_simulator import simulator
from user_simulator import test_utils
from user_simulator.judge import prompts


class EvaluationTest(unittest.TestCase):
    """Protects the fixed denominator and prevents holdout reuse for tuning."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name)
        self.release = self.root / "release.json"
        self.judge = mock.Mock(identity=test_utils.make_judge_identity())

    def _make_split(
        self, split: cases.Split = cases.Split.DEVELOPMENT,
        criterion_mae: float = 0.20,
    ) -> evaluation.SplitResult:
        outcomes = tuple(evaluation.CaseResult(
            user_id=f"user_{index // 5}", case_name=f"case_{index % 5}",
            expected_score=50, score=75, errors=("Reference mismatch",),
            score_record=None,
        ) for index in range(20))
        return evaluation.SplitResult(
            split=split, correct=0, total=20, cases=outcomes,
            criterion_error=diagnostics.NumericError(
                diagnostics.Agreement(40, 40, 32), criterion_mae,
            ),
        )

    def test_split_saves_completed_result_once_with_outcome_paths(self) -> None:
        group = cases.load_cases(cases.Split.DEVELOPMENT)[0]
        outcomes = tuple(dataclasses.replace(
            outcome, user_id=group.user.user_id,
            score_record=(str(self.root / f"score-{index}.json")
                          if index else None),
        ) for index, outcome in enumerate(self._make_split().cases[:5]))
        metric = diagnostics.NumericError(
            diagnostics.Agreement(10, 8, 7), 0.125,
        )
        report = mock.Mock()
        report.calculate_criterion_error.return_value = metric
        with (
            mock.patch.object(cases, "load_cases", return_value=(group,)),
            mock.patch.object(evaluation.simulator, "Simulator"),
            mock.patch.object(evaluation, "_run_case", side_effect=outcomes),
            mock.patch.object(diagnostics, "write_diagnostics",
                              return_value=report) as write_diagnostics,
            mock.patch.object(
                evaluation.runs, "write_record",
                wraps=evaluation.runs.write_record,
            ) as write_record,
        ):
            result = evaluation._run_split(
                cases.Split.DEVELOPMENT, self.judge, self.root,
            )
        write_diagnostics.assert_called_once_with(
            self.root, cases.Split.DEVELOPMENT, (group,),
            [pathlib.Path(outcome.score_record) for outcome in outcomes[1:]],
        )
        write_record.assert_called_once_with(
            self.root, evaluation.COMPONENT, "development/results.json", result,
        )
        self.assertEqual(result.cases, outcomes)
        self.assertEqual(result.criterion_error, metric)
        folder = evaluation.runs.get_component_directory(
            self.root, evaluation.COMPONENT,
        )
        saved = json.loads((folder / "development/results.json").read_text())
        self.assertEqual(saved["criterion_error"]["mean_absolute_error"], 0.125)
        self.assertEqual(saved["criterion_error"]["agreement"]["available"], 8)
        self.assertFalse(result.has_passed())

    def test_above_target_never_loads_or_releases_holdout(self) -> None:
        result = self._make_split(criterion_mae=math.nextafter(0.20, math.inf))
        with mock.patch.object(evaluation, "_run_split", return_value=result
                               ) as run_split:
            self.assertFalse(evaluation.run_evaluation(
                self.judge, self.root, self.release,
            ))
        run_split.assert_called_once_with(
            cases.Split.DEVELOPMENT, self.judge, self.root,
        )
        self.assertFalse(self.release.exists())

    def test_development_only_never_releases_holdout_even_after_passing(
        self,
    ) -> None:
        for error in (0.20, math.nextafter(0.20, math.inf)):
            with self.subTest(criterion_mae=error):
                with (
                    mock.patch.object(
                        evaluation, "_run_split",
                        return_value=self._make_split(criterion_mae=error),
                    ) as run_split,
                    mock.patch.object(evaluation, "_release_holdout")
                    as release_holdout,
                ):
                    passed = evaluation.run_development_evaluation(
                        self.judge, self.root,
                    )
                self.assertEqual(passed, error <= 0.20)
                run_split.assert_called_once_with(
                    cases.Split.DEVELOPMENT, self.judge, self.root,
                )
                release_holdout.assert_not_called()
                folder = evaluation.runs.get_component_directory(
                    self.root, evaluation.COMPONENT,
                )
                self.assertIn(
                    f"Development: {'PASS' if passed else 'FAIL'}",
                    (folder / "summary.txt").read_text(),
                )

    def test_exact_threshold_releases_same_judge_to_holdout(self) -> None:
        development = self._make_split()
        for error in (0.20, math.nextafter(0.20, math.inf)):
            with self.subTest(holdout_mae=error):
                holdout = self._make_split(cases.Split.HOLDOUT, error)
                with mock.patch.object(evaluation, "_run_split",
                                       side_effect=[development, holdout]
                                       ) as run_split:
                    passed = evaluation.run_evaluation(
                        self.judge, self.root, self.release,
                    )
                self.assertEqual(passed, error <= 0.20)
                self.assertEqual(run_split.call_args_list, [
                    mock.call(cases.Split.DEVELOPMENT, self.judge, self.root),
                    mock.call(cases.Split.HOLDOUT, self.judge, self.root),
                ])
        self.assertTrue(self.release.exists())

    def test_score_error_and_exact_case_count_remain_diagnostics(self) -> None:
        result = self._make_split()
        self.assertEqual(result.correct, 0)
        self.assertEqual(result.calculate_mean_absolute_error(), 25)
        self.assertTrue(result.has_passed())
        exact = dataclasses.replace(
            result, correct=20,
            criterion_error=dataclasses.replace(
                result.criterion_error, mean_absolute_error=0.21,
            ),
        )
        self.assertFalse(exact.has_passed())

    def test_changed_prompt_cannot_reuse_released_holdout(self) -> None:
        evaluation._release_holdout(self.judge, self.root, self.release)
        with (
            mock.patch.object(prompts, "PROMPT_VERSION", "changed"),
            mock.patch.object(evaluation, "_run_split",
                              return_value=self._make_split()) as run_split,
        ):
            with self.assertRaisesRegex(ValueError, "already exposed"):
                evaluation.run_evaluation(self.judge, self.root, self.release)
        run_split.assert_called_once()

    def test_changed_mae_target_cannot_reuse_released_holdout(self) -> None:
        evaluation._release_holdout(self.judge, self.root, self.release)
        with (
            mock.patch.object(evaluation, "MAX_CRITERION_MAE", 0.25),
            mock.patch.object(evaluation, "_run_split",
                              return_value=self._make_split()) as run_split,
        ):
            with self.assertRaisesRegex(ValueError, "already exposed"):
                evaluation.run_evaluation(self.judge, self.root, self.release)
        run_split.assert_called_once()

    def test_missing_cases_cannot_improve_gate(self) -> None:
        result = self._make_split()
        for total in (19, 20):
            with self.subTest(total=total):
                self.assertFalse(dataclasses.replace(
                    result, total=total, cases=result.cases[:-1],
                ).has_passed())

    def test_unavailable_or_invalid_score_blocks_release(self) -> None:
        result = self._make_split()
        for score in (
            None, True, False, math.nan, math.inf, -math.inf, -1, 101,
        ):
            with self.subTest(score=score):
                outcome = dataclasses.replace(result.cases[0], score=score)
                incomplete = dataclasses.replace(
                    result, cases=(outcome, *result.cases[1:]),
                )
                self.assertFalse(incomplete.has_passed())
                with mock.patch.object(evaluation, "_run_split",
                                       return_value=incomplete) as run_split:
                    self.assertFalse(evaluation.run_evaluation(
                        self.judge, self.root, self.release,
                    ))
                run_split.assert_called_once()
        self.assertFalse(self.release.exists())

    def test_incomplete_gold_ratings_block_gate_despite_zero_error(
        self,
    ) -> None:
        result = self._make_split()
        for metric in (
            None,
            diagnostics.NumericError(diagnostics.Agreement(40, 39, 39), 0),
            diagnostics.NumericError(diagnostics.Agreement(0, 0, 0), None),
            diagnostics.NumericError(diagnostics.Agreement(40, 40, 40), None),
        ):
            with self.subTest(metric=metric):
                incomplete = dataclasses.replace(result, criterion_error=metric)
                self.assertFalse(incomplete.has_passed())

    def test_failed_request_keeps_its_diagnostics_link(self) -> None:
        group = cases.load_cases(cases.Split.DEVELOPMENT)[0]
        diagnostics = self.root / "failed-score.json"
        scorer = mock.Mock()
        scorer.score.side_effect = simulator.JudgeError(
            "Malformed LM judgment", diagnostics,
        )
        result = evaluation._run_case(
            scorer, group, group.cases[0], self.root, cases.Split.DEVELOPMENT,
        )
        self.assertEqual(result.score_record, str(diagnostics))
        self.assertIsNone(result.score)
        self.assertTrue(result.errors)

    def test_wrong_user_configuration_cannot_print_pass(self) -> None:
        group = cases.load_cases(cases.Split.DEVELOPMENT)[0]
        scorer = mock.Mock()
        scorer.score.return_value = 100.0
        record = mock.Mock(user_config=None)
        with (
            mock.patch.object(evaluation.review, "load_output_record",
                              return_value=(self.root / "score.json", record)),
            mock.patch.object(evaluation.runs, "write_text") as write_text,
        ):
            result = evaluation._run_case(
                scorer, group, group.cases[0], self.root,
                cases.Split.DEVELOPMENT,
            )
        self.assertTrue(result.errors)
        self.assertIsNone(result.score)
        self.assertIn("Result: FAIL", write_text.call_args.args[-1])
        self.assertNotIn("Result: PASS", write_text.call_args.args[-1])

    def test_development_has_four_distinct_users_and_twenty_cases(self) -> None:
        groups = cases.load_cases(cases.Split.DEVELOPMENT)
        self.assertEqual(len(groups), 4)
        self.assertEqual(sum(len(group.cases) for group in groups), 20)
        self.assertEqual(len({group.user.user_id for group in groups}), 4)
        for group in groups:
            self.assertEqual(len({case.expected_rule
                                  for case in group.cases}), 2)


if __name__ == "__main__":
    unittest.main()
