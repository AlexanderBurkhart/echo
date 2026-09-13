"""Checks shared run persistence and isolation without invoking model code."""

from concurrent import futures
import dataclasses
import json
import pathlib
import tempfile
import threading
from typing import Any
import unittest
from unittest import mock

from loguru import logger as log

from experiment_logging import records
from experiment_logging import runs


@dataclasses.dataclass(frozen=True)
class _TestRecord:
    """An example component-owned payload."""

    score: float


class RunLoggerTest(unittest.TestCase):
    """Validates shared provenance, private artifacts, and journal writes."""

    def setUp(self) -> None:
        """Creates a fresh run isolated from existing experiment artifacts."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name) / "run"
        self.logger = runs.RunLogger(self.root)

    def _read_events(self) -> list[dict[str, Any]]:
        path = self.root / "logs" / "events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_run_capture_isolates_concurrent_runs(self) -> None:
        roots = (self.root, self.root.parent / "other")
        barrier = threading.Barrier(2)

        def write_messages(root: pathlib.Path) -> None:
            with runs.capture_logs(root):
                barrier.wait(timeout=5)
                log.info("Message for {}", root.name)
                runs.write_record(root, "training", "step.json", {"step": 1})

        with futures.ThreadPoolExecutor(max_workers=2) as executor:
            pending = [executor.submit(write_messages, root) for root in roots]
            for result in pending:
                result.result()
        for root in roots:
            content = (root / "logs" / runs.RUNTIME_LOG_FILENAME).read_text()
            self.assertIn(f"Message for {root.name}", content)
            other = next(item for item in roots if item != root)
            self.assertNotIn(f"Message for {other.name}", content)
            self.assertEqual(content.count("training: write_record"), 1)

    def test_run_capture_closes_on_failure_and_preserves_other_sinks(
        self,
    ) -> None:
        messages = []
        sink_id = log.add(messages.append)
        self.addCleanup(log.remove, sink_id)
        with self.assertRaisesRegex(ValueError, "failed run"):
            with runs.capture_logs(self.root):
                log.error("Before failure")
                raise ValueError("failed run")
        path = self.root / "logs" / runs.RUNTIME_LOG_FILENAME
        original = path.read_text()
        # Explicit binding would reach a leaked sink even after context exit.
        log.bind(run_id=self.logger.run_id).info("After capture")
        self.assertEqual(path.read_text(), original)
        self.assertTrue(any("After capture" in item for item in messages))
        with runs.capture_logs(self.root):
            log.info("Reopened run")
        self.assertEqual(path.read_text().count("Reopened run"), 1)
        self.assertIn("Before failure", path.read_text())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_reopening_preserves_run_identity_and_previous_events(self) -> None:
        self.logger.log_event("training", "start_update")
        reopened = runs.RunLogger(self.root)
        reopened.log_event("evaluation", "start_evaluation")
        self.assertEqual(reopened.run_id, self.logger.run_id)
        self.assertEqual(len(self._read_events()), 2)

    def test_components_share_journal_but_keep_payloads_separate(self) -> None:
        original = {"status": "ok", "private_score": 12.5}
        first = self.logger.write_record(
            "user_simulator", "score.json", original, request_id="task-1"
        )
        second = self.logger.write_record(
            "training", "update.json", {"status": "ok", "loss": 0.4},
            request_id="task-1",
        )
        self.assertNotEqual(first.parent, second.parent)
        self.assertNotIn("_record", original)
        payload = json.loads(first.read_text())
        self.assertEqual(payload["_record"]["run_id"], self.logger.run_id)
        events = self._read_events()
        self.assertEqual(len(events), 2)
        for event in events:
            self.assertEqual(event["request_id"], "task-1")
            self.assertNotIn("private_score", event)
            self.assertTrue((self.root / event["artifact"]).exists())
        self.assertEqual(first.stat().st_mode & 0o777, 0o600)
        self.assertEqual((self.root / "logs").stat().st_mode & 0o777, 0o700)

    def test_direct_writer_infers_request_id_and_preserves_overrides(
        self,
    ) -> None:
        component = "evaluation/scoring"
        directory = runs.get_component_directory(self.root, component)
        paths = (
            (directory / "requests/task-1/score.json", None, "task-1"),
            (pathlib.Path("requests/task-2/score.json"),
             "override", "override"),
            (directory / "task_cache/cached.json", None, None),
        )
        for path, supplied_id, expected_id in paths:
            with self.subTest(path=path):
                saved = runs.write_record(
                    self.root, component, path, _TestRecord(score=12.5),
                    request_id=supplied_id,
                )
                payload = json.loads(saved.read_text())
                self.assertEqual(payload["score"], 12.5)
                self.assertEqual(payload["_record"]["request_id"], expected_id)
                self.assertEqual(
                    payload["_record"]["run_id"], self.logger.run_id
                )
                event = self._read_events()[-1]
                self.assertEqual(event["request_id"], expected_id)
                self.assertEqual(self.root / event["artifact"], saved)

    def test_direct_writer_rejects_paths_outside_component(self) -> None:
        outside = self.root.parent / "outside.json"
        other_component = self.root / "logs/training/outside.json"
        for path in (outside, other_component, pathlib.Path("../outside.json")):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    runs.write_record(
                        self.root, "evaluation", path, _TestRecord(score=1),
                    )
        self.assertFalse(outside.exists())
        self.assertFalse(other_component.exists())
        self.assertFalse((self.root / "logs/events.jsonl").exists())

    def test_snapshot_replacement_keeps_journal_history(self) -> None:
        path = self.logger.write_record(
            "judge", "attempt.json", {"status": "start"}
        )
        self.logger.write_record("judge", "attempt.json", {"status": "ok"})
        self.assertEqual(json.loads(path.read_text())["status"], "ok")
        self.assertEqual(
            [event["status"] for event in self._read_events()], ["start", "ok"]
        )

    def test_parallel_logger_instances_do_not_interleave_json_lines(
        self,
    ) -> None:
        loggers = [runs.RunLogger(self.root) for _ in range(4)]
        with futures.ThreadPoolExecutor(max_workers=4) as executor:
            pending = [
                executor.submit(
                    loggers[index % 4].log_event,
                    "training", "complete_step", request_id=str(index),
                )
                for index in range(40)
            ]
            for result in pending:
                result.result()
        events = self._read_events()
        self.assertEqual(len(events), 40)
        self.assertEqual(len({event["request_id"] for event in events}), 40)

    def test_component_and_artifact_paths_cannot_escape_run(self) -> None:
        for component in ("../outside", "/tmp/outside", "."):
            with self.subTest(component=component):
                with self.assertRaises(ValueError):
                    self.logger.write_record(component, "x.json", {})
        for path in ("../outside.json", "/tmp/outside.json"):
            with self.assertRaises(ValueError):
                self.logger.write_record("training", path, {})

    def test_symlink_cannot_redirect_component_outside_logs_root(self) -> None:
        outside = self.root.parent / "outside"
        outside.mkdir()
        (self.root / "logs" / "redirect").symlink_to(outside)
        with self.assertRaises(ValueError):
            self.logger.write_record("redirect", "x.json", {})
        self.assertFalse((outside / "x.json").exists())

    def test_provenance_cannot_be_overridden(self) -> None:
        with self.assertRaises(ValueError):
            self.logger.write_record("training", "x.json", {"_record": {}})

    def test_failed_artifact_write_does_not_log_success(self) -> None:
        with mock.patch.object(records, "write_json", side_effect=OSError):
            with self.assertRaises(OSError):
                self.logger.write_record("training", "x.json", {})
        self.assertFalse((self.root / "logs" / "events.jsonl").exists())

    def test_text_reports_share_journal_and_path_constraints(self) -> None:
        path = runs.write_text(
            self.root, "evaluation", "report.md", "results\n"
        )
        self.assertEqual(path.read_text(), "results\n")
        event = self._read_events()[0]
        self.assertEqual(event["run_id"], self.logger.run_id)
        self.assertEqual(self.root / event["artifact"], path)
        with self.assertRaises(ValueError):
            runs.write_text(self.root, "evaluation", "../outside.md", "bad")
        with mock.patch.object(records, "write_text", side_effect=OSError):
            with self.assertRaises(OSError):
                runs.write_text(self.root, "evaluation", "failed.md", "failed")
        self.assertEqual(len(self._read_events()), 1)


if __name__ == "__main__":
    unittest.main()
