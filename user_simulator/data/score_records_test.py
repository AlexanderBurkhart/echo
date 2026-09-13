"""Tests the production score-record decoding contract."""

import json
import pathlib
import tempfile
import unittest

from conversation import history
from experiment_logging import records
from user_simulator import engine
from user_simulator import test_utils
from user_simulator.data import judgments
from user_simulator.data import score_records
from user_simulator.judge import response_schema


class ScoreRecordsTest(unittest.TestCase):
    """Checks that old test identities cannot leak into production records."""

    def test_current_record_preserves_scores_schemas_and_raw_calls(self) -> None:
        user = test_utils.make_conditional_user()
        task = test_utils.make_task(90)
        task["resource_constraints"].value = 75
        output = test_utils.make_ratings(
            urgency=4, external_resource_demand=0,
        )
        result = engine.calculate_preference_score(user, task, output)
        schema = response_schema.build_response_schema(
            judgments.JudgmentKind.OUTPUT, tuple(reversed(user.rating_criteria)),
        )
        saved = score_records.ScoreRecord(
            user_id=user.user_id, history=history.ConversationHistory(),
            output="Use the prepared response.",
            judge=test_utils.make_judge_identity(), prompt_hash="old-hash",
            user_config=user, task_cache_key="old-cache", timestamp="old-time",
            task_judgment=judgments.LMJudgment(task),
            output_judgment=judgments.LMJudgment(output), result=result,
            output_lm_calls=[score_records.LMCallResult(
                raw='{"ratings": "raw LM response"}',
                response_schema=schema,
            )],
            status=score_records.ScoreStatus.OK,
        )
        raw = records.serialize_canonical_json(saved)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "score.json"
            path.write_text(raw)
            loaded = score_records.load_score_record(path)
            self.assertEqual(path.read_text(), raw)
        self.assertEqual(loaded.result, result)
        self.assertEqual(loaded.user_config, user)
        self.assertEqual(loaded.task_judgment.ratings, task)
        self.assertEqual(loaded.output_judgment.ratings, output)
        self.assertEqual(loaded.output_lm_calls, saved.output_lm_calls)
        self.assertEqual(
            json.dumps(
                loaded.output_lm_calls[0].response_schema.build_json_schema()
            ),
            json.dumps(schema.build_json_schema()),
        )
        self.assertEqual(loaded.prompt_hash, "old-hash")

    def test_loading_rejects_legacy_test_identity(self) -> None:
        user = test_utils.make_conditional_user()
        score_record = score_records.ScoreRecord(
            user_id=user.user_id,
            history=history.ConversationHistory(),
            output="Send the prepared draft now.",
            judge=test_utils.make_judge_identity(),
            prompt_hash="test-prompt",
            user_config=user,
            task_cache_key="test-task",
            timestamp="2026-09-12T00:00:00+00:00",
        )
        payload = json.loads(records.serialize_canonical_json(score_record))
        payload["judge"] = {"backend": "old-test-double", "revision": "v1"}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "score.json"
            records.write_json(path, payload)
            with self.assertRaises(TypeError):
                score_records.load_score_record(path)


if __name__ == "__main__":
    unittest.main()
