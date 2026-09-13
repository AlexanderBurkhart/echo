"""Tests smoke-only checks without claiming real LM inference."""

import dataclasses
import unittest

from conversation import history
from experiments.smoke_test import cases
from experiments.smoke_test import review
from user_simulator import config
from user_simulator import engine
from user_simulator import test_utils
from user_simulator.data import judgments
from user_simulator.data import score_records


class SmokeReviewTest(unittest.TestCase):
    """Checks that the smoke report exposes incorrect ratings and scores."""

    def setUp(self) -> None:
        self.case = cases.SmokeCase(
            name="example", history=history.ConversationHistory(),
            output="Send now and keep revisions possible.",
            expected_rule="prefer_speed_when_urgent",
            expected_ratings={"urgency": 4, "uncertainty": 4},
            explanation="Immediate and adaptable.",
        )
        user = config.UserConfig(
            user_id="smoke_test",
            rules=(
                config.PreferenceRule(
                    id="always", precedence=0, when=None,
                    weights={"urgency": 0.5, "uncertainty": 0.5},
                ),
                config.PreferenceRule(
                    id=self.case.expected_rule, precedence=1,
                    when=config.Condition(
                        property="urgency",
                        operator=config.ConditionOperator.GREATER_EQUAL,
                        threshold=80,
                    ),
                    weights={"urgency": 0.8, "uncertainty": 0.2},
                ),
            ),
        )
        task = judgments.LMJudgment(test_utils.make_task(95))
        output = judgments.LMJudgment(
            test_utils.make_ratings(urgency=4, uncertainty=4)
        )
        self.record = score_records.ScoreRecord(
            user_id=user.user_id, history=self.case.history,
            output=self.case.output, judge=test_utils.make_judge_identity(),
            prompt_hash="test-prompt", user_config=user,
            task_cache_key="test-task", timestamp="test-time",
            task_judgment=task, output_judgment=output,
            result=engine.calculate_preference_score(
                user, task.ratings, output.ratings
            ),
            status=score_records.ScoreStatus.OK,
        )

    def test_reports_expected_score_and_manual_calculation(self) -> None:
        report, failures = review.review_output(self.case, self.record, 100)
        self.assertEqual(failures, [])
        self.assertIn("100 * (0.8 * (4/4) + 0.2 * (4/4)) / 1.0", report)

    def test_reports_unexpected_lm_rating_even_with_correct_arithmetic(
        self,
    ) -> None:
        self.record.output_judgment.ratings["urgency"].value = 3
        self.record.result = engine.calculate_preference_score(
            self.record.user_config, self.record.task_judgment.ratings,
            self.record.output_judgment.ratings,
        )
        _, failures = review.review_output(self.case, self.record, 80)
        self.assertIn("LM urgency rating was 3; expected 4.", failures)
        self.assertNotIn(
            "Returned score differs from manual arithmetic.", failures
        )

    def test_equal_score_does_not_hide_incorrect_active_ratings(self) -> None:
        case = dataclasses.replace(
            self.case, expected_ratings={"urgency": 3, "uncertainty": 0}
        )
        self.record.output_judgment.ratings["urgency"].value = 2
        self.record.output_judgment.ratings["uncertainty"].value = 4
        self.record.result = engine.calculate_preference_score(
            self.record.user_config, self.record.task_judgment.ratings,
            self.record.output_judgment.ratings,
        )
        _, failures = review.review_output(case, self.record, 60)
        self.assertIn("LM urgency rating was 2; expected 3.", failures)
        self.assertIn("LM uncertainty rating was 4; expected 0.", failures)
        self.assertFalse(any("Score differs" in error for error in failures))

    def test_reports_wrong_rule_weights_and_returned_score(self) -> None:
        self.record.result = dataclasses.replace(
            self.record.result, selected_rule="always",
            weights={"urgency": 0.5, "uncertainty": 0.5},
        )
        _, failures = review.review_output(self.case, self.record, 50)
        self.assertIn(
            "Returned score differs from manual arithmetic.", failures
        )
        self.assertIn(
            "Logged weights differ from the expected config rule.", failures
        )
        self.assertIn(
            f"Expected rule {self.case.expected_rule} was not selected.",
            failures,
        )

    def test_reports_unknown_rating_instead_of_silently_accepting_it(
        self,
    ) -> None:
        self.record.output_judgment.ratings["urgency"].value = None
        self.record.result = engine.calculate_preference_score(
            self.record.user_config, self.record.task_judgment.ratings,
            self.record.output_judgment.ratings,
        )
        _, failures = review.review_output(self.case, self.record, 60)
        self.assertIn(
            "Cannot manually calculate unknown urgency rating.", failures
        )


if __name__ == "__main__":
    unittest.main()
