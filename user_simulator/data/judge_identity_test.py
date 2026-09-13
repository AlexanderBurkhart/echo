"""Tests immutable judge identity snapshots and their JSON representation."""

import dataclasses
import json
import unittest

from experiment_logging import records
from user_simulator import test_utils
from user_simulator.data import judge_identity


class LocalJudgeIdentityTest(unittest.TestCase):
    """Checks nested immutability and stable cache/log serialization."""

    def test_nested_metadata_is_frozen_and_detached_from_runtime(self) -> None:
        metadata = {"layers": [{"name": "layer0"}]}
        generation = {"response_format": {"type": "json_object"}}
        settings = {"n_ctx": 8192}
        hashes = {"runtime.so": "original"}
        identity = dataclasses.replace(
            test_utils.make_judge_identity(),
            model_metadata=metadata, generation=generation,
            settings=settings, runtime_files_sha256=hashes,
        )
        original = records.serialize_canonical_json(identity)
        metadata["layers"][0]["name"] = "changed"
        metadata["layers"].append({"name": "layer1"})
        generation["response_format"]["type"] = "text"
        settings["n_ctx"] = 1
        hashes["runtime.so"] = "changed"
        self.assertEqual(records.serialize_canonical_json(identity), original)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            identity.model_revision = "changed"
        with self.assertRaises(TypeError):
            identity.model_metadata["layers"][0]["name"] = "changed"
        with self.assertRaises(TypeError):
            identity.generation["response_format"]["type"] = "text"
        with self.assertRaises(TypeError):
            identity.settings["n_ctx"] = 1
        with self.assertRaises(TypeError):
            identity.runtime_files_sha256["runtime.so"] = "changed"
        self.assertIsInstance(identity.model_metadata["layers"], tuple)

    def test_json_round_trip_preserves_identity_and_cache_digest(self) -> None:
        identity = dataclasses.replace(
            test_utils.make_judge_identity(),
            model_metadata={"layers": [{"size": 64}]},
            generation={"response_format": {"type": "json_object"}},
        )
        payload = json.loads(records.serialize_canonical_json(identity))
        restored = judge_identity.LocalJudgeIdentity(**payload)
        self.assertEqual(restored, identity)
        self.assertEqual(
            records.compute_digest(identity), records.compute_digest(payload)
        )
        self.assertEqual(
            records.compute_digest(identity), records.compute_digest(restored)
        )


if __name__ == "__main__":
    unittest.main()
