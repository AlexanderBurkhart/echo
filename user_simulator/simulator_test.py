"""Tests for simulator."""

from collections.abc import Iterable
import copy
import dataclasses
import json
import tempfile
import unittest

from conversation import history
from experiment_logging import runs
from user_simulator import simulator
from user_simulator import test_utils
from user_simulator.data import judgments
from user_simulator.data import score_records
from user_simulator.integration_tests import judge_evaluation
from user_simulator.judge.ratings import ratings as rating_scale


class _SequenceJudge:
    """A test judge returning predetermined responses or failures."""

    def __init__(self, responses: Iterable[str | Exception]) -> None:
        self.identity = test_utils.make_judge_identity()
        self.calls = []
        self._responses = iter(responses)

    def complete(self, prompt: str) -> str:
        """Returns the next fixture response and records the visible prompt."""
        self.calls.append(prompt)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


class SimulatorTest(unittest.TestCase):
    """Checks judge isolation, cache behavior, and failure records."""

    def setUp(self) -> None:
        """Creates a private temporary directory and development fixture."""
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.run_log = runs.RunLogger(self._temporary.name)
        self.root = self.run_log.get_component_directory("user_simulator")
        self.case = judge_evaluation.load_cases(test_utils.CASES_PATH).cases[0]
        self.user = test_utils.make_conditional_user()
        self.task_raw = test_utils.serialize_lm_response(
            self.case.reference_task
        )
        self.output_raw = test_utils.serialize_lm_response(
            self.case.first.reference
        )

    def _score(
        self, scorer: simulator.Simulator, *, use_cache: bool = True
    ) -> float:
        return scorer.score(
            run_root=self.run_log.root,
            user_id=self.user.user_id,
            conversation_history=self.case.history,
            output=self.case.first.output,
            use_cache=use_cache,
        )

    def test_one_score_records_all_private_diagnostics(self) -> None:
        judge = _SequenceJudge([self.task_raw, self.output_raw])
        scorer = simulator.Simulator([self.user], judge)
        score = self._score(scorer)
        self.assertIsInstance(score, float)
        path = next(self.root.glob("requests/*/score.json"))
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertIs(
            score_records.load_score_record(path).status,
            score_records.ScoreStatus.OK,
        )
        self.assertEqual(record["result"]["score"], score)
        self.assertEqual(record["result"]["selected_rule"], "urgent")
        self.assertIn("component_values", record["result"])
        self.assertIn("user_config", record)
        self.assertIn("elapsed_seconds", record)
        self.assertEqual(record["_record"]["run_id"], self.run_log.run_id)
        self.run_log.log_event("training", "start_update")
        journal = self.run_log.root / "logs" / "events.jsonl"
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        self.assertEqual(
            {event["component"] for event in events},
            {"user_simulator", "training"},
        )
        self.assertTrue(all(
            event["run_id"] == self.run_log.run_id for event in events
        ))
        saved = score_records.load_score_record(path)
        self.assertEqual(saved.judge, judge.identity)
        self.assertIs(scorer._identity, judge.identity)
        self.assertEqual(saved.task_lm_calls[0].raw, self.task_raw)
        self.assertEqual(saved.output_lm_calls[0].raw, self.output_raw)
        self.assertIsNone(saved.task_lm_calls[0].error)
        self.assertIsNone(saved.output_lm_calls[0].error)
        self.assertEqual(saved.task_prompt, judge.calls[0])
        self.assertEqual(saved.output_prompt, judge.calls[1])
        self.assertEqual(saved.task_judgment.source_record, str(path.resolve()))
        self.assertEqual(list(path.parent.iterdir()), [path])
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_lm_caches_survive_restart_and_scores_use_each_user(self) -> None:
        judge = _SequenceJudge([self.task_raw, self.output_raw])
        scorer = simulator.Simulator([self.user], judge)
        baseline = self._score(scorer)
        self.assertEqual(self._score(scorer), baseline)
        other = dataclasses.replace(
            self.user, user_id="different_user",
            rules=(dataclasses.replace(
                self.user.rules[1], weights={"money": 1},
            ),),
        )
        scorer = simulator.Simulator([other], judge)
        other_score = scorer.score(
            run_root=self.run_log.root,
            user_id="different_user",
            conversation_history=self.case.history,
            output=self.case.first.output,
        )
        self.assertEqual(len(judge.calls), 2)
        self.assertNotEqual(other_score, baseline)
        saved = [
            score_records.load_score_record(path)
            for path in self.root.glob("requests/*/score.json")
        ]
        reused = next(item for item in saved if item.user_id == other.user_id)
        self.assertTrue(reused.task_cache_hit)
        self.assertTrue(reused.output_cache_hit)
        self.assertEqual(reused.task_lm_calls, [])
        self.assertEqual(reused.output_lm_calls, [])

    def test_cache_bypass_ignores_existing_entry(self) -> None:
        judge = _SequenceJudge([self.task_raw, self.output_raw] * 3)
        scorer = simulator.Simulator([self.user], judge)
        self._score(scorer)
        self._score(scorer, use_cache=False)
        self._score(scorer, use_cache=False)
        self.assertEqual(len(judge.calls), 6)
        self.assertEqual(len(list(self.root.glob("task_cache/*.json"))), 1)
        self.assertEqual(len(list(self.root.glob("output_cache/*.json"))), 1)

    def test_bypass_does_not_create_cache(self) -> None:
        judge = _SequenceJudge([self.task_raw, self.output_raw])
        self._score(
            simulator.Simulator([self.user], judge),
            use_cache=False,
        )
        self.assertFalse((self.root / "task_cache").exists())
        self.assertFalse((self.root / "output_cache").exists())

    def test_config_is_reused_and_private_weights_stay_out_of_prompts(
        self,
    ) -> None:
        private = dataclasses.replace(
            self.user,
            rules=(dataclasses.replace(
                self.user.rules[0], id="PRIVATE_SENTINEL_318",
                weights={
                    **self.user.rules[0].weights,
                    "urgency": 0.1234567890123,
                },
            ), self.user.rules[1]),
        )
        judge = _SequenceJudge([self.task_raw, self.output_raw] * 2)
        scorer = simulator.Simulator([private], judge)
        first = self._score(scorer, use_cache=False)
        self.assertIs(scorer._users[private.user_id], private)
        with self.assertRaises(TypeError):
            private.rules[0].weights["urgency"] = 0
        self.assertEqual(self._score(scorer, use_cache=False), first)
        for prompt in judge.calls:
            self.assertNotIn("PRIVATE_SENTINEL_318", prompt)
            self.assertNotIn("0.1234567890123", prompt)
            self.assertNotIn(private.user_id, prompt)
            self.assertNotIn("default_weights", prompt)
        self.assertNotIn(self.case.first.output, judge.calls[0])
        self.assertNotIn(self.task_raw, judge.calls[1])

    def test_three_malformed_attempts_save_explicit_failure(self) -> None:
        judge = _SequenceJudge(["not json"] * 3)
        scorer = simulator.Simulator([self.user], judge)
        with self.assertRaises(simulator.JudgeError) as caught:
            self._score(scorer)
        self.assertTrue(caught.exception.diagnostics.exists())
        self.assertEqual(len(judge.calls), 3)
        self.assertFalse((self.root / "task_cache").exists())
        self.assertFalse((self.root / "output_cache").exists())
        path = next(self.root.glob("requests/*/score.json"))
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "failure")
        self.assertIs(
            score_records.load_score_record(path).status,
            score_records.ScoreStatus.FAILURE,
        )

    def test_output_retry_retains_task_assessment(self) -> None:
        judge = _SequenceJudge([self.task_raw, "{", "bad", self.output_raw])
        self._score(simulator.Simulator([self.user], judge))
        self.assertEqual(len(judge.calls), 4)
        self.assertEqual(judge.calls[1:], [judge.calls[1]] * 3)

    def test_runtime_failure_is_not_retried(self) -> None:
        judge = _SequenceJudge([RuntimeError("runtime unavailable")])
        with self.assertRaises(simulator.JudgeError) as caught:
            self._score(simulator.Simulator([self.user], judge))
        self.assertEqual(len(judge.calls), 1)
        saved = score_records.load_score_record(caught.exception.diagnostics)
        self.assertIn("runtime unavailable", saved.task_lm_calls[0].error)
        self.assertIsNone(saved.task_lm_calls[0].raw)
        self.assertIs(saved.status, score_records.ScoreStatus.FAILURE)

    def test_low_scores_do_not_retry(self) -> None:
        output = copy.deepcopy(self.case.first.reference)
        for item in output.ratings.values():
            item.value = 0
            item.evidence = [self.case.first.output]
        judge = _SequenceJudge([
            self.task_raw, test_utils.serialize_lm_response(output)
        ])
        scorer = simulator.Simulator([self.user], judge)
        score = self._score(scorer)
        self.assertEqual(score, 0)
        self.assertEqual(len(judge.calls), 2)

    def test_identity_change_invalidates_cache_and_is_frozen_per_instance(
        self,
    ) -> None:
        judge = _SequenceJudge([self.task_raw, self.output_raw] * 2)
        scorer = simulator.Simulator([self.user], judge)
        self._score(scorer)
        judge.identity = dataclasses.replace(
            judge.identity, runtime_revision="fixture-v2",
        )
        with self.assertRaises(ValueError):
            self._score(scorer)
        self._score(simulator.Simulator([self.user], judge))
        self.assertEqual(len(list(self.root.glob("task_cache/*.json"))), 2)
        self.assertEqual(len(list(self.root.glob("output_cache/*.json"))), 2)

    def test_history_change_invalidates_cache(self) -> None:
        judge = _SequenceJudge([self.task_raw, self.output_raw] * 2)
        scorer = simulator.Simulator([self.user], judge)
        self._score(scorer)
        scorer.score(
            run_root=self.run_log.root,
            user_id=self.user.user_id,
            conversation_history=history.ConversationHistory(
                messages=self.case.history.messages + (
                    history.Message(
                        role=history.Role.USER, content="Extra context."
                    ),
                ),
            ),
            output=self.case.first.output,
        )
        self.assertEqual(len(judge.calls), 4)

    def test_extra_preference_uses_rating_criteria_in_user_config(self) -> None:
        rule = dataclasses.replace(
            self.user.rules[1],
            weights={**self.user.rules[1].weights, "extra": 1},
        )
        with self.assertRaisesRegex(ValueError, "undefined rating criteria"):
            dataclasses.replace(self.user, rules=(rule,))
        user = dataclasses.replace(
            self.user, rules=(rule,),
            rating_criteria=self.user.rating_criteria + (
                rating_scale.RatingAnchoredCriterion(
                    name="extra", anchors=("a", "b", "c", "d", "e"),
                ),
            ),
        )
        output = copy.deepcopy(self.case.first.reference)
        output.ratings["extra"] = judgments.Rating(
            value=4, unit=rating_scale.OUTPUT_RATING_UNIT,
            evidence=[self.case.first.output],
        )
        judge = _SequenceJudge([
            self.task_raw, test_utils.serialize_lm_response(output),
        ])
        scorer = simulator.Simulator(
            [user], judge,
        )
        score = self._score(scorer)
        self.assertGreater(score, 0)
        self.assertIn(
            '"extra":{"0":"a","1":"b","2":"c","3":"d","4":"e"}',
            judge.calls[1],
        )
        saved = score_records.load_score_record(
            next(self.root.glob("requests/*/score.json"))
        )
        self.assertEqual(saved.output_judgment.ratings["extra"].value, 4)

    def test_different_user_rating_criteria_do_not_share_cached_ratings(
        self,
    ) -> None:
        criterion = self.user.rating_criteria[0]
        other = dataclasses.replace(
            self.user, user_id="other",
            rating_criteria=(dataclasses.replace(
                criterion,
                anchors=criterion.anchors[:4] + ("CUSTOM_RATING_DESCRIPTION",),
            ),) + self.user.rating_criteria[1:],
        )
        judge = _SequenceJudge([self.task_raw, self.output_raw] * 2)
        scorer = simulator.Simulator([self.user, other], judge)
        self._score(scorer)
        scorer.score(
            run_root=self.run_log.root,
            user_id=other.user_id,
            conversation_history=self.case.history,
            output=self.case.first.output,
        )
        self.assertEqual(len(judge.calls), 4)
        self.assertNotIn("CUSTOM_RATING_DESCRIPTION", judge.calls[1])
        self.assertIn("CUSTOM_RATING_DESCRIPTION", judge.calls[3])

    def test_duplicate_user_id_is_rejected_during_initialization(self) -> None:
        judge = _SequenceJudge([])
        with self.assertRaisesRegex(ValueError, "Duplicate user ID"):
            simulator.Simulator([self.user, self.user], judge)
        self.assertEqual(judge.calls, [])

    def test_missing_user_is_rejected_when_scoring(self) -> None:
        judge = _SequenceJudge([])
        scorer = simulator.Simulator([], judge)
        with self.assertRaises(KeyError):
            self._score(scorer)
        self.assertEqual(judge.calls, [])

    def test_scoring_selects_the_run_without_a_logger_object(self) -> None:
        judge = _SequenceJudge([self.task_raw, self.output_raw] * 2)
        scorer = simulator.Simulator([self.user], judge)
        first_root = self.run_log.root / "first"
        second_root = self.run_log.root / "second"
        self.assertFalse(first_root.exists())
        for root in (first_root, first_root, second_root):
            scorer.score(
                run_root=root,
                user_id=self.user.user_id,
                conversation_history=self.case.history,
                output=self.case.first.output,
            )
        self.assertEqual(len(judge.calls), 4)
        first = json.loads((first_root / "run.json").read_text())
        second = json.loads((second_root / "run.json").read_text())
        self.assertNotEqual(first["run_id"], second["run_id"])

    def test_corrupt_task_cache_requests_and_reuses_replacement(self) -> None:
        judge = _SequenceJudge([
            self.task_raw, self.output_raw, self.task_raw,
        ])
        scorer = simulator.Simulator([self.user], judge)
        baseline = self._score(scorer)
        cache_path = next(self.root.glob("task_cache/*.json"))
        cache_path.write_text("{}", encoding="utf-8")
        self.assertEqual(self._score(scorer), baseline)
        saved_records = (
            score_records.load_score_record(path)
            for path in self.root.glob("requests/*/score.json")
        )
        saved = next(item for item in saved_records if item.task_cache_error)
        self.assertEqual(saved.task_invalid_cache, "{}")
        self.assertIn("source_record", saved.task_cache_error)
        self.assertEqual(saved.task_lm_calls[0].raw, self.task_raw)
        self.assertIsNone(saved.output_cache_error)
        self.assertIs(saved.task_cache_hit, False)
        self.assertIs(saved.output_cache_hit, True)
        self.assertEqual(self._score(scorer), baseline)
        self.assertEqual(len(judge.calls), 3)
        self.assertEqual(judge.calls[0], judge.calls[2])

    def test_changed_output_reuses_task_but_requests_new_lm_assessment(
        self,
    ) -> None:
        second_raw = test_utils.serialize_lm_response(
            self.case.second.reference
        )
        judge = _SequenceJudge([self.task_raw, self.output_raw, second_raw])
        scorer = simulator.Simulator([self.user], judge)
        self._score(scorer)
        for _ in range(2):
            scorer.score(
                run_root=self.run_log.root,
                user_id=self.user.user_id,
                conversation_history=self.case.history,
                output=self.case.second.output,
            )
        self.assertEqual(len(judge.calls), 3)
        self.assertEqual(len(list(self.root.glob("task_cache/*.json"))), 1)
        self.assertEqual(len(list(self.root.glob("output_cache/*.json"))), 2)

    def test_corrupt_output_cache_requests_and_reuses_replacement(self) -> None:
        judge = _SequenceJudge([
            self.task_raw, self.output_raw, self.output_raw,
        ])
        scorer = simulator.Simulator([self.user], judge)
        baseline = self._score(scorer)
        path = next(self.root.glob("output_cache/*.json"))
        cached = json.loads(path.read_text(encoding="utf-8"))
        cached["ratings"]["urgency"]["evidence"] = [
            "Fabricated quotation absent from the history and output."
        ]
        raw = json.dumps(cached)
        path.write_text(raw, encoding="utf-8")
        self.assertEqual(self._score(scorer), baseline)
        saved_records = (
            score_records.load_score_record(path)
            for path in self.root.glob("requests/*/score.json")
        )
        saved = next(item for item in saved_records if item.output_cache_error)
        self.assertEqual(saved.output_invalid_cache, raw)
        self.assertIn("Evidence must quote", saved.output_cache_error)
        self.assertEqual(saved.output_lm_calls[0].raw, self.output_raw)
        self.assertIs(saved.task_cache_hit, True)
        self.assertIsNone(saved.task_cache_error)
        self.assertIs(saved.output_cache_hit, False)
        self.assertEqual(self._score(scorer), baseline)
        self.assertEqual(len(judge.calls), 3)
        self.assertEqual(judge.calls[1], judge.calls[2])

    def test_invalid_cache_json_requests_fresh_lm_judgment(self) -> None:
        invalid_entries = ("{", "[]", "null")
        judge = _SequenceJudge([
            self.task_raw, self.output_raw,
            *([self.task_raw] * len(invalid_entries)),
        ])
        scorer = simulator.Simulator([self.user], judge)
        baseline = self._score(scorer)
        path = next(self.root.glob("task_cache/*.json"))
        for raw in invalid_entries:
            with self.subTest(raw=raw):
                path.write_text(raw, encoding="utf-8")
                self.assertEqual(self._score(scorer), baseline)
        self.assertEqual(len(judge.calls), 2 + len(invalid_entries))
        saved = [
            score_records.load_score_record(path)
            for path in self.root.glob("requests/*/score.json")
        ]
        self.assertEqual(
            {item.task_invalid_cache
             for item in saved if item.task_cache_error},
            set(invalid_entries),
        )

    def test_duplicate_cache_keys_request_and_reuse_replacement(self) -> None:
        for kind, response in (
            (judgments.JudgmentKind.TASK, self.task_raw),
            (judgments.JudgmentKind.OUTPUT, self.output_raw),
        ):
            judge = _SequenceJudge([
                self.task_raw, self.output_raw, response, response,
            ])
            scorer = simulator.Simulator([self.user], judge)
            baseline = self._score(scorer)
            path = next(self.root.glob(f"{kind.value}_cache/*.json"))
            valid = path.read_text(encoding="utf-8")
            invalid_entries = (
                '{"ratings":{},' + valid[1:],
                valid.replace('"urgency":{', '"urgency":{"value":0,', 1),
            )
            for index, raw in enumerate(invalid_entries):
                with self.subTest(kind=kind, index=index):
                    path.write_text(raw, encoding="utf-8")
                    self.assertEqual(self._score(scorer), baseline)
                    saved_records = (
                        score_records.load_score_record(record_path)
                        for record_path in self.root.glob(
                            "requests/*/score.json"
                        )
                    )
                    if kind is judgments.JudgmentKind.TASK:
                        saved = next(
                            item for item in saved_records
                            if item.task_invalid_cache == raw
                        )
                        self.assertIn(
                            "Duplicate JSON key", saved.task_cache_error
                        )
                        self.assertFalse(saved.task_cache_hit)
                        self.assertTrue(saved.output_cache_hit)
                        self.assertEqual(saved.task_lm_calls[0].raw, response)
                    else:
                        saved = next(
                            item for item in saved_records
                            if item.output_invalid_cache == raw
                        )
                        self.assertIn(
                            "Duplicate JSON key", saved.output_cache_error
                        )
                        self.assertFalse(saved.output_cache_hit)
                        self.assertTrue(saved.task_cache_hit)
                        self.assertEqual(saved.output_lm_calls[0].raw, response)
                    self.assertEqual(len(judge.calls), 3 + index)
                    self.assertEqual(self._score(scorer), baseline)
                    self.assertEqual(len(judge.calls), 3 + index)
            # Give the next judgment kind a fresh pair of caches.
            for cache_path in self.root.glob("*_cache/*.json"):
                cache_path.unlink()

    def test_invalid_cache_recovery_uses_normal_lm_retry_limit(self) -> None:
        judge = _SequenceJudge([
            self.task_raw, self.output_raw, "bad", "bad", "bad",
        ])
        scorer = simulator.Simulator([self.user], judge)
        self._score(scorer)
        cache_path = next(self.root.glob("output_cache/*.json"))
        cache_path.write_text("{", encoding="utf-8")
        with self.assertRaises(simulator.JudgeError) as caught:
            self._score(scorer)
        self.assertEqual(len(judge.calls), 5)
        self.assertTrue(caught.exception.diagnostics.exists())
        saved = score_records.load_score_record(caught.exception.diagnostics)
        self.assertEqual(
            [call.raw for call in saved.output_lm_calls], ["bad"] * 3
        )
        self.assertTrue(all(call.error for call in saved.output_lm_calls))
        self.assertEqual(saved.output_invalid_cache, "{")
        self.assertIs(saved.output_cache_hit, False)
        self.assertIsNotNone(saved.output_cache_error)
        self.assertIs(saved.status, score_records.ScoreStatus.FAILURE)
        self.assertIsNone(saved.result)
        self.assertEqual(cache_path.read_text(encoding="utf-8"), "{")

    def test_failed_lm_output_assessment_is_not_cached(self) -> None:
        judge = _SequenceJudge([
            self.task_raw, "bad", "bad", "bad", self.output_raw,
        ])
        scorer = simulator.Simulator([self.user], judge)
        with self.assertRaises(simulator.JudgeError):
            self._score(scorer)
        self.assertFalse((self.root / "output_cache").exists())
        self._score(scorer)
        self.assertEqual(len(judge.calls), 5)
        self.assertEqual(len(list(self.root.glob("output_cache/*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
