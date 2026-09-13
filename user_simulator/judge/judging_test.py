"""Tests for judging."""

import copy
import json
import unittest

from user_simulator import test_utils
from user_simulator.data import judgments
from user_simulator.integration_tests import judge_evaluation
from user_simulator.judge import judging
from user_simulator.judge.ratings import output_definitions


class LMJudgmentTest(unittest.TestCase):
    """Rejects malformed measurements and fabricated quotations."""

    def test_bad_json_and_schema_are_rejected(self) -> None:
        samples = ("{}", "[]", '{"x": 1, "x": 2}', '{"x": NaN}',
                   '```json\n{}\n```', "null")
        for raw in samples:
            with self.subTest(raw=raw):
                with self.assertRaises(judging.MalformedLMJudgmentError):
                    judging.parse_lm_judgment(
                        raw, judgments.JudgmentKind.TASK,
                        ["source"], output_definitions.DEFAULT_RATING_CRITERIA,
                    )

    def test_unsupported_evidence_and_boolean_rating_are_rejected(self) -> None:
        case = judge_evaluation.load_cases(test_utils.CASES_PATH).cases[0]
        for level, evidence in (
            (True, [case.first.output]), (5, [case.first.output]),
            (2.5, [case.first.output]),
            (4, ["fabricated quotation"]), (4, []),
        ):
            output = copy.deepcopy(case.first.reference)
            output.ratings["urgency"].value = level
            output.ratings["urgency"].evidence = evidence
            with self.assertRaises(judging.MalformedLMJudgmentError):
                judging.parse_lm_judgment(
                    test_utils.serialize_lm_response(output),
                    judgments.JudgmentKind.OUTPUT,
                    [case.first.output],
                    output_definitions.DEFAULT_RATING_CRITERIA,
                )

    def test_task_evidence_cannot_come_from_output(self) -> None:
        case = judge_evaluation.load_cases(test_utils.CASES_PATH).cases[0]
        task = copy.deepcopy(case.reference_task)
        task.ratings["urgency"].evidence = [case.first.output]
        with self.assertRaises(judging.MalformedLMJudgmentError):
            judging.parse_lm_judgment(
                test_utils.serialize_lm_response(task),
                judgments.JudgmentKind.TASK,
                [message.content for message in case.history.messages],
                output_definitions.DEFAULT_RATING_CRITERIA,
            )

    def test_unified_judgments_preserve_ratings_and_quantities(self) -> None:
        case = json.loads(test_utils.CASES_PATH.read_text())["cases"][0]
        sources = [message["content"] for message in case["history"]]
        for kind, raw, evidence in (
            (judgments.JudgmentKind.TASK, case["reference_task"], sources),
            (judgments.JudgmentKind.OUTPUT, case["reference_outputs"]["A"],
             sources + [case["A"]]),
        ):
            with self.subTest(kind=kind):
                parsed = judging.parse_lm_judgment(
                    json.dumps(raw), kind, evidence,
                    output_definitions.DEFAULT_RATING_CRITERIA,
                )
                self.assertIsInstance(parsed, judgments.LMJudgment)
                self.assertEqual(
                    json.loads(test_utils.serialize_lm_response(parsed)), raw
                )

    def test_float_anchor_values_and_rating_units(self) -> None:
        case = judge_evaluation.load_cases(test_utils.CASES_PATH).cases[0]
        output = copy.deepcopy(case.first.reference)
        output.ratings["urgency"].value = 4.0
        parsed = judging.parse_lm_judgment(
            test_utils.serialize_lm_response(output),
            judgments.JudgmentKind.OUTPUT, [case.first.output],
            output_definitions.DEFAULT_RATING_CRITERIA,
        )
        self.assertEqual(parsed.ratings["urgency"].value, 4.0)
        output.ratings["urgency"].unit = "USD"
        with self.assertRaisesRegex(
            judging.MalformedLMJudgmentError, "rating_0_4",
        ):
            judging.parse_lm_judgment(
                test_utils.serialize_lm_response(output),
                judgments.JudgmentKind.OUTPUT, [case.first.output],
                output_definitions.DEFAULT_RATING_CRITERIA,
            )

    def test_missing_rating_lists_expected_names_in_error(self) -> None:
        raw = '{"ratings": {}, "quantities": {}}'
        with self.assertRaisesRegex(
            judging.MalformedLMJudgmentError, "effort.*money.*urgency",
        ):
            judging.parse_lm_judgment(
                raw, judgments.JudgmentKind.OUTPUT, [],
                output_definitions.DEFAULT_RATING_CRITERIA,
            )


if __name__ == "__main__":
    unittest.main()
