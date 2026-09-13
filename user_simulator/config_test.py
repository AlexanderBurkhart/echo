"""Tests for config."""

import dataclasses
import json
import pathlib
import tempfile
import unittest

from experiment_logging import records
from user_simulator import config
from user_simulator import test_utils
from user_simulator.judge import prompts
from user_simulator.judge.ratings import output_definitions


class ConfigTest(unittest.TestCase):
    """Validates reproducible private configuration generation."""

    def test_seeded_configs_are_reproducible(self) -> None:
        first = config.generate_users()
        self.assertEqual(first, config.generate_users())
        self.assertNotEqual(first, config.generate_users(7))
        self.assertEqual(len(first), 8)
        for index, user in enumerate(first):
            self.assertEqual(user.user_id, f"pilot_{index:02d}")
            self.assertEqual(
                {
                    rule.when.property for rule in user.rules
                    if rule.when is not None
                },
                {"urgency", "uncertainty", "resource_constraints"},
            )
            weights = [rule.weights for rule in user.rules]
            for group in weights:
                self.assertEqual(
                    set(group),
                    {
                        "urgency", "uncertainty", "external_resource_demand",
                        "user_time", "effort",
                    },
                )
                self.assertTrue(all(value > 0 for value in group.values()))
                self.assertAlmostEqual(sum(group.values()), 1)
            self.assertTrue(any(rule.when is None for rule in user.rules))
            for rule in user.rules:
                if rule.when is not None:
                    self.assertGreaterEqual(rule.when.threshold, 0)
                    self.assertLessEqual(rule.when.threshold, 100)

    def test_user_files_load_and_preserve_config_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "config/users"
            users = config.generate_users()
            for index, user in enumerate(users):
                records.write_json(path / f"user{index}.json", user)
            self.assertEqual(config.load_users(path), users)
            self.assertEqual(
                config.load_user(path / "user0.json"), users[0]
            )
            self.assertEqual(len(list(path.glob("user*.json"))), 8)
            saved_user = json.loads((path / "user0.json").read_text())
            self.assertNotIn("default_weights", saved_user)
            self.assertNotIn("cost_weights", saved_user)
            self.assertEqual(
                {item["name"] for item in saved_user["rating_criteria"]},
                {
                    "urgency", "uncertainty", "external_resource_demand",
                    "user_time", "effort",
                },
            )
            for rule in saved_user["rules"]:
                self.assertNotIn("money", rule["weights"])
                if rule["when"] is not None:
                    self.assertNotEqual(rule["when"]["property"], "user_cost")
                    self.assertEqual(
                        set(rule["when"]),
                        {"property", "operator", "threshold"},
                    )

    def test_generated_rules_do_not_share_fixed_resource_ratios(self) -> None:
        user = config.generate_users()[0]
        ratios = {
            rule.weights["external_resource_demand"] / rule.weights["user_time"]
            for rule in user.rules
        }
        self.assertEqual(len(ratios), len(user.rules))

    def test_parse_user_preserves_authored_criteria_and_weights(self) -> None:
        user = config.generate_users()[0]
        criterion = dataclasses.replace(
            user.rating_criteria[0], anchors=("zero", "one", "two", "three", "four"),
        )
        user = dataclasses.replace(
            user, rating_criteria=(criterion,) + user.rating_criteria[1:],
        )
        payload = json.loads(records.serialize_canonical_json(user))
        before = json.dumps(payload)
        self.assertEqual(config.parse_user(payload), user)
        self.assertEqual(json.dumps(payload), before)

    def test_custom_criteria_use_the_same_name_and_weight_checks(self) -> None:
        user = test_utils.make_conditional_user()
        for name in ("resource_constraints", "custom_preference"):
            with self.subTest(name=name):
                criterion = dataclasses.replace(
                    user.rating_criteria[0], name=name,
                )
                rule = dataclasses.replace(user.rules[1], weights={name: 1})
                updated = dataclasses.replace(
                    user, rules=(rule,),
                    rating_criteria=user.rating_criteria + (criterion,),
                )
                self.assertEqual(updated.rules[0].weights, {name: 1})

    def test_identifiers_reject_nonstring_and_blank_values(self) -> None:
        user = test_utils.make_conditional_user()
        for value in (None, 12, "", " \t"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "user_id"):
                    dataclasses.replace(user, user_id=value)
                with self.assertRaisesRegex(ValueError, "Rule ID"):
                    dataclasses.replace(user.rules[1], id=value)

    def test_condition_properties_reject_nonstring_and_blank_values(
        self,
    ) -> None:
        for value in (None, 12, "", " ", []):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Condition property"):
                    config.Condition(
                        property=value,
                        operator=config.ConditionOperator.GREATER,
                        threshold=80,
                    )
                payload = json.loads(records.serialize_canonical_json(
                    test_utils.make_conditional_user()
                ))
                payload["rules"][0]["when"]["property"] = value
                with self.assertRaisesRegex(ValueError, "Condition property"):
                    config.parse_user(payload)

    def test_conditions_require_qualitative_properties(self) -> None:
        user = test_utils.make_conditional_user()
        for name in ("budget", "available_time", "unknown"):
            with self.subTest(property=name):
                rule = dataclasses.replace(
                    user.rules[0],
                    when=dataclasses.replace(user.rules[0].when, property=name),
                )
                with self.assertRaisesRegex(ValueError, "task preference"):
                    dataclasses.replace(user, rules=(rule, user.rules[1]))

    def test_condition_thresholds_use_the_shared_scale(self) -> None:
        user = test_utils.make_conditional_user()
        for threshold in (0, 100):
            rule = dataclasses.replace(
                user.rules[0],
                when=dataclasses.replace(
                    user.rules[0].when, threshold=threshold,
                ),
            )
            dataclasses.replace(user, rules=(rule, user.rules[1]))
        for threshold in (-1, 101, float("nan"), float("inf"), True):
            with self.subTest(threshold=threshold):
                rule = dataclasses.replace(
                    user.rules[0],
                    when=dataclasses.replace(
                        user.rules[0].when, threshold=threshold,
                    ),
                )
                with self.assertRaisesRegex(ValueError, "Condition threshold"):
                    dataclasses.replace(user, rules=(rule, user.rules[1]))

    def test_duplicate_precedence_and_missing_coverage_are_rejected(
        self,
    ) -> None:
        user = test_utils.make_conditional_user()
        with self.assertRaisesRegex(ValueError, "precedences"):
            dataclasses.replace(user, rules=user.rules + (
                dataclasses.replace(user.rules[0], id="second"),
            ))
        for rules in ((), (user.rules[0],)):
            with self.subTest(rules=rules):
                with self.assertRaisesRegex(ValueError, "always-applicable"):
                    dataclasses.replace(user, rules=rules)

    def test_rule_weights_need_positive_total_and_defined_criteria(
        self,
    ) -> None:
        user = test_utils.make_conditional_user()
        for weights in (
            {}, {"external_resource_demand": 0},
            {"external_resource_demand": -1}, {"undefined": 1},
        ):
            with self.subTest(weights=weights):
                rule = dataclasses.replace(user.rules[1], weights=weights)
                with self.assertRaises(ValueError):
                    dataclasses.replace(user, rules=(rule,))

    def test_loaded_rating_anchors_match_mapping_and_are_nonempty(self) -> None:
        for anchors in (
            ("a",) * 4, ("a",) * 6, ("a",) * 4 + (" ",),
            ("a",) * 4 + (None,),
        ):
            with self.subTest(anchors=anchors):
                value = json.loads(records.serialize_canonical_json(
                    test_utils.make_conditional_user()
                ))
                value["rating_criteria"][0]["anchors"] = anchors
                with self.assertRaisesRegex(ValueError, "anchors"):
                    config.parse_user(value)

    def test_duplicate_or_missing_rating_criteria_are_rejected(self) -> None:
        user = test_utils.make_conditional_user()
        for criteria in (
            user.rating_criteria + (user.rating_criteria[0],), (),
        ):
            with self.subTest(criteria=criteria):
                with self.assertRaises(ValueError):
                    dataclasses.replace(user, rating_criteria=criteria)

    def test_constructor_names_undefined_rule_criteria(self) -> None:
        rule = config.PreferenceRule(
            id="test_rule", precedence=0, when=None,
            weights={"undefined": 1},
        )
        with self.assertRaisesRegex(
            ValueError, "Rule 'test_rule'.*undefined rating criteria: undefined"
        ):
            config.UserConfig(user_id="test_user", rules=(rule,))

    def test_criteria_are_immutable_and_define_the_prompt_identity(
        self,
    ) -> None:
        user = test_utils.make_conditional_user()
        criterion = user.rating_criteria[0]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            criterion.name = "changed"
        original = prompts.compute_prompt_fingerprint(user.rating_criteria)
        user = dataclasses.replace(
            user,
            rating_criteria=(dataclasses.replace(
                criterion, anchors=("a", "b", "c", "d", "e"),
            ),) + user.rating_criteria[1:],
        )
        self.assertNotEqual(
            original, prompts.compute_prompt_fingerprint(user.rating_criteria),
        )
        self.assertEqual(
            config.parse_user(json.loads(
                records.serialize_canonical_json(user)
            )).rating_criteria,
            user.rating_criteria,
        )

    def test_user_and_nested_preferences_are_immutable(self) -> None:
        user = test_utils.make_conditional_user()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            user.user_id = "changed"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            user.rules[0].precedence = 99
        with self.assertRaises(dataclasses.FrozenInstanceError):
            user.rules[0].when.threshold = 0
        for weights in (rule.weights for rule in user.rules):
            with self.assertRaises(TypeError):
                weights[next(iter(weights))] = 0
        self.assertIsInstance(user.rules, tuple)
        self.assertIsInstance(user.rating_criteria, tuple)

    def test_mutating_constructor_inputs_cannot_change_user(self) -> None:
        original = test_utils.make_conditional_user()
        replacement = dict(original.rules[0].weights)
        rule = dataclasses.replace(original.rules[0], weights=replacement)
        rules = [rule, original.rules[1]]
        criteria = list(original.rating_criteria)
        user = dataclasses.replace(
            original, rules=rules, rating_criteria=criteria,
        )
        replacement.clear()
        rules.clear()
        criteria.clear()
        self.assertEqual(user, original)


if __name__ == "__main__":
    unittest.main()
