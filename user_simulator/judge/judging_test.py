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
            judging.MalformedLMJudgmentError,
            "effort.*external_resource_demand.*urgency",
        ):
            judging.parse_lm_judgment(
                raw, judgments.JudgmentKind.OUTPUT, [],
                output_definitions.DEFAULT_RATING_CRITERIA,
            )

    def test_unknown_values_require_empty_evidence(self) -> None:
        case = judge_evaluation.load_cases(test_utils.CASES_PATH).cases[0]
        sources = [message.content for message in case.history.messages]
        sources.append(case.first.output)
        for kind, reference, group, name, unit in (
            (judgments.JudgmentKind.TASK, case.reference_task,
             "ratings", "urgency", "rating_0_100"),
            (judgments.JudgmentKind.TASK, case.reference_task,
             "ratings", "available_time", None),
            (judgments.JudgmentKind.OUTPUT, case.first.reference,
             "ratings", "urgency", "rating_0_4"),
            (judgments.JudgmentKind.OUTPUT, case.first.reference,
             "quantities", "user_time", None),
        ):
            for evidence in ([], [sources[0]], None, "", {}):
                with self.subTest(kind=kind, name=name, evidence=evidence):
                    raw = json.loads(
                        test_utils.serialize_lm_response(reference)
                    )
                    raw[group][name] = {
                        "value": None, "unit": unit, "evidence": evidence,
                    }
                    if evidence == []:
                        parsed = judging.parse_lm_judgment(
                            json.dumps(raw), kind, sources,
                            output_definitions.DEFAULT_RATING_CRITERIA,
                        )
                        self.assertIsNone(getattr(parsed, group)[name].value)
                    else:
                        with self.assertRaises(
                            judging.MalformedLMJudgmentError
                        ):
                            judging.parse_lm_judgment(
                                json.dumps(raw), kind, sources,
                                output_definitions.DEFAULT_RATING_CRITERIA,
                            )

    def test_feedback_accepts_one_to_400_characters(self) -> None:
        case = judge_evaluation.load_cases(test_utils.CASES_PATH).cases[0]
        for feedback in ("x", "é" * 400):
            with self.subTest(length=len(feedback)):
                raw = json.loads(
                    test_utils.serialize_lm_response(case.first.reference)
                )
                raw["feedback"] = feedback
                parsed = judging.parse_lm_judgment(
                    json.dumps(raw), judgments.JudgmentKind.OUTPUT,
                    [case.first.output],
                    output_definitions.DEFAULT_RATING_CRITERIA,
                    require_feedback=True,
                )
                self.assertEqual(parsed.ratings, case.first.reference.ratings)

    def test_missing_blank_and_overlong_feedback_are_rejected(self) -> None:
        case = judge_evaluation.load_cases(test_utils.CASES_PATH).cases[0]
        for feedback in (None, "", " ", 1, "x" * 401, "x" * 400 + " "):
            with self.subTest(feedback=feedback):
                raw = json.loads(
                    test_utils.serialize_lm_response(case.first.reference)
                )
                if feedback is not None:
                    raw["feedback"] = feedback
                with self.assertRaises(judging.MalformedLMJudgmentError):
                    judging.parse_lm_judgment(
                        json.dumps(raw), judgments.JudgmentKind.OUTPUT,
                        [case.first.output],
                        output_definitions.DEFAULT_RATING_CRITERIA,
                        require_feedback=True,
                    )


if __name__ == "__main__":
    unittest.main()
