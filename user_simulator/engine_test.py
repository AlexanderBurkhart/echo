"""Tests for deterministic preference scoring and conditional weights."""

import copy
import dataclasses
import random
import unittest

from user_simulator import config
from user_simulator import engine
from user_simulator import test_utils
from user_simulator.data import judgments
from user_simulator.integration_tests import judge_evaluation
from user_simulator.judge.ratings import ratings as rating_scale


class EngineTest(unittest.TestCase):
    """Validates conditional utility and numerical invariants."""

    def test_threshold_boundaries_and_unknown(self) -> None:
        user = test_utils.make_conditional_user()
        for value, expected in (
            (79.999, "always"), (80, "always"),
            (80.001, "urgent"), (None, "always"),
        ):
            with self.subTest(value=value):
                result = engine.calculate_preference_score(
                    user, test_utils.make_task(value),
                    test_utils.make_ratings(),
                )
                self.assertEqual(result.selected_rule, expected)
        for task in ({}, test_utils.make_task(90, "hours")):
            result = engine.calculate_preference_score(
                user, task, test_utils.make_ratings(),
            )
            self.assertEqual(result.selected_rule, "always")

    def test_all_comparison_operators(self) -> None:
        for operation, expected in (
            (">", False), (">=", True), ("<", False),
            ("<=", True), ("==", True),
        ):
            user = test_utils.make_conditional_user()
            rule = user.rules[0]
            user = dataclasses.replace(user, rules=(dataclasses.replace(
                rule, when=dataclasses.replace(
                    rule.when, operator=config.ConditionOperator(operation),
                ),
            ), user.rules[1]))
            result = engine.calculate_preference_score(
                user, test_utils.make_task(80), test_utils.make_ratings(),
            )
            self.assertEqual(result.selected_rule == "urgent", expected)

    def test_highest_precedence_wins_independent_of_order(self) -> None:
        user = test_utils.make_conditional_user()
        other = dataclasses.replace(
            user.rules[0], id="higher", precedence=5,
            weights=user.rules[1].weights,
        )
        for rules in (user.rules + (other,), (other,) + user.rules):
            result = engine.calculate_preference_score(
                dataclasses.replace(user, rules=rules),
                test_utils.make_task(90), test_utils.make_ratings(),
            )
            self.assertEqual(result.selected_rule, "higher")

    def test_conditions_only_use_the_shared_task_rating_scale(self) -> None:
        user = test_utils.make_conditional_user()
        for unit in ("rating_0_1", "rating_0_4", "USD", "hours", None):
            with self.subTest(unit=unit):
                result = engine.calculate_preference_score(
                    user, test_utils.make_task(90, unit),
                    test_utils.make_ratings(),
                )
                self.assertEqual(result.selected_rule, "always")

    def test_invalid_task_rating_values_are_rejected(self) -> None:
        user = test_utils.make_conditional_user()
        for value in (-1, 101, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Task ratings"):
                    engine.calculate_preference_score(
                        user, test_utils.make_task(value),
                        test_utils.make_ratings(),
                    )

    def test_expected_conditional_preference_reversal(self) -> None:
        user = test_utils.make_conditional_user()
        first = test_utils.make_ratings(urgency=4, money=0)
        second = test_utils.make_ratings(urgency=0, money=4)
        for urgency, first_preferred in ((79, False), (81, True)):
            task = test_utils.make_task(urgency)
            first_score = engine.calculate_preference_score(
                user, task, first,
            ).score
            second_score = engine.calculate_preference_score(
                user, task, second,
            ).score
            self.assertNotEqual(first_score, second_score)
            self.assertEqual(first_score > second_score, first_preferred)

    def test_scores_are_bounded_and_monotone(self) -> None:
        generator = random.Random(5)
        for user in config.generate_users():
            for _ in range(20):
                ratings = test_utils.make_ratings(
                    *(generator.randrange(5) for _ in range(5))
                )
                task = test_utils.make_task(generator.uniform(0, 100))
                baseline = engine.calculate_preference_score(
                    user, task, ratings,
                ).score
                self.assertGreaterEqual(baseline, 0)
                self.assertLessEqual(baseline, 100)
                for name in ratings:
                    improved = copy.deepcopy(ratings)
                    improved[name].value = min(4, ratings[name].value + 1)
                    score = engine.calculate_preference_score(
                        user, task, improved,
                    ).score
                    self.assertGreaterEqual(score, baseline)

    def test_common_scale_invariance_and_large_finite_weights(self) -> None:
        values = {"a": 0.2, "b": 0.8}
        baseline = engine._calculate_weighted_value({"a": 1, "b": 2}, values)
        for scale in (1e-200, 0.25, 10, 1e300):
            self.assertAlmostEqual(
                engine._calculate_weighted_value(
                    {"a": scale, "b": 2 * scale}, values,
                ),
                baseline,
            )
        self.assertAlmostEqual(
            engine._calculate_weighted_value(
                {"a": 1e308, "b": 1e308}, values,
            ),
            0.5,
        )

    def test_cost_preferences_are_weighted_directly(self) -> None:
        user = test_utils.make_conditional_user()
        rule = dataclasses.replace(
            user.rules[1], weights={"money": 1, "user_time": 2, "effort": 1},
        )
        result = engine.calculate_preference_score(
            dataclasses.replace(user, rules=(rule,)), test_utils.make_task(),
            test_utils.make_ratings(money=0, user_time=4, effort=2),
        )
        self.assertEqual(result.score, 62.5)
        self.assertEqual(result.weights, dict(rule.weights))
        self.assertNotIn("user_cost", result.component_values)

    def test_inactive_preferences_do_not_change_score_or_denominator(
        self,
    ) -> None:
        user = test_utils.make_conditional_user()
        rule = dataclasses.replace(
            user.rules[0], weights={"urgency": 2, "money": 0},
        )
        user = dataclasses.replace(user, rules=(rule, user.rules[1]))
        ratings = test_utils.make_ratings(urgency=4, uncertainty=0, money=0)
        result = engine.calculate_preference_score(
            user, test_utils.make_task(90), ratings,
        )
        self.assertEqual(result.score, 100)
        self.assertEqual(result.normalized_weights, {"urgency": 1})
        self.assertEqual(result.component_values, {"urgency": 1})
        ratings["urgency"].value = 0
        self.assertEqual(engine.calculate_preference_score(
            user, test_utils.make_task(90), ratings,
        ).score, 0)

    def test_unknown_output_is_explicit_and_neutral(self) -> None:
        result = engine.calculate_preference_score(
            test_utils.make_conditional_user(), test_utils.make_task(),
            test_utils.make_ratings(None, None, None, None, None),
        )
        self.assertEqual(result.score, 50)
        self.assertEqual(len(result.unknown_components), 3)

    def test_generic_engine_supports_extra_preferences(self) -> None:
        user = test_utils.make_conditional_user()
        user = dataclasses.replace(
            user, rules=(dataclasses.replace(
                user.rules[1], weights={**user.rules[1].weights, "extra": 3},
            ),),
            rating_criteria=user.rating_criteria + (
                rating_scale.RatingAnchoredCriterion(
                    name="extra", anchors=("a", "b", "c", "d", "e"),
                ),
            ),
        )
        ratings = test_utils.make_ratings()
        ratings["extra"] = judgments.Rating(
            4, rating_scale.OUTPUT_RATING_UNIT
        )
        self.assertEqual(
            engine.calculate_preference_score(
                user, test_utils.make_task(), ratings,
            ).score,
            87.5,
        )

    def test_generated_users_disagree_without_resampling(self) -> None:
        labels = set()
        for user in config.generate_users():
            first = engine.calculate_preference_score(
                user, test_utils.make_task(),
                test_utils.make_ratings(4, 0, 0, 0, 0),
            )
            second = engine.calculate_preference_score(
                user, test_utils.make_task(),
                test_utils.make_ratings(0, 4, 4, 4, 4),
            )
            labels.add(judge_evaluation.calculate_comparison_label(
                first.score, second.score,
            ))
        self.assertEqual(labels, {"A", "B"})

    def test_float_anchors_score_equally_and_other_units_fail(self) -> None:
        user = test_utils.make_conditional_user()
        task = test_utils.make_task()
        ratings = test_utils.make_ratings(urgency=4)
        expected = engine.calculate_preference_score(user, task, ratings).score
        ratings["urgency"].value = 4.0
        self.assertEqual(
            engine.calculate_preference_score(user, task, ratings).score,
            expected,
        )
        ratings["urgency"].unit = "USD"
        with self.assertRaisesRegex(ValueError, "rating_0_4"):
            engine.calculate_preference_score(user, task, ratings)

    def test_invalid_values_and_mappings_fail(self) -> None:
        for value in (-1, 2, None, float("nan")):
            with self.assertRaises(ValueError):
                engine._calculate_weighted_value({"a": 1}, {"a": value})
        for values in ((0, 1), (0, 0.5, 0.2, 0.8, 1), (0, 0.2, 0.5, 1, 2)):
            with self.assertRaises(ValueError):
                engine.calculate_preference_score(
                    test_utils.make_conditional_user(), test_utils.make_task(),
                    test_utils.make_ratings(), values,
                )

    def test_invalid_weights_are_rejected(self) -> None:
        for weights in (
            {}, {"a": 0}, {"a": -1}, {"a": True},
            {"a": float("nan")}, {"a": float("inf")},
        ):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    engine._calculate_weighted_value(weights, {"a": 0.5})


if __name__ == "__main__":
    unittest.main()
