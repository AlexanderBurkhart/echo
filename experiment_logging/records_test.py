"""Checks dataclass persistence and deterministic cache-key serialization."""

import dataclasses
import json
import pathlib
import tempfile
import unittest

from experiment_logging import records


@dataclasses.dataclass(frozen=True)
class _Payload:
    name: str
    value: float


@dataclasses.dataclass(frozen=True)
class _Batch:
    entries: tuple[_Payload, ...]


class RecordsTest(unittest.TestCase):
    """Exercises nested dataclasses at the shared persistence boundary."""

    def test_nested_dataclasses_round_trip_through_json_files(self) -> None:
        payload = _Batch(entries=(_Payload("cost", 2.5),))
        expected = {"entries": [{"name": "cost", "value": 2.5}]}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "record.json"
            records.write_json(path, {"payload": payload})
            self.assertEqual(
                json.loads(path.read_text()), {"payload": expected}
            )
        self.assertEqual(payload.entries, (_Payload("cost", 2.5),))

    def test_digest_matches_json_fields_and_changes_with_values(self) -> None:
        payload = _Payload("cost", 2.5)
        self.assertEqual(
            records.compute_digest(payload),
            records.compute_digest({"value": 2.5, "name": "cost"}),
        )
        self.assertNotEqual(
            records.compute_digest(payload),
            records.compute_digest(_Payload("cost", 3.0)),
        )

    def test_unsupported_and_nonfinite_fields_are_rejected(self) -> None:
        for value in (object(), _Payload):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    records.serialize_canonical_json({"payload": value})
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    records.serialize_canonical_json(_Payload("cost", value))


if __name__ == "__main__":
    unittest.main()
