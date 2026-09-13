"""Tests the production score-record decoding contract."""

import json
import pathlib
import tempfile
import unittest

from conversation import history
from experiment_logging import records
from user_simulator import test_utils
from user_simulator.data import score_records


class ScoreRecordsTest(unittest.TestCase):
    """Checks that old test identities cannot leak into production records."""

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
