"""Tests for benchmark."""

import json
import pathlib
import tempfile
import unittest
from unittest import mock

from temp.scripts import benchmark_local_judge as benchmark
from temp.scripts import benchmark_records
from user_simulator import config
from user_simulator import test_utils
from user_simulator.integration_tests import judge_evaluation
from user_simulator.judge import local_judge


class BenchmarkTest(unittest.TestCase):
    """Checks benchmark preflight and failure reports without a model."""

    def test_benchmark_preflight_does_not_load_a_model(self) -> None:
        with mock.patch.object(local_judge, "LocalLlamaJudge") as loader:
            report = benchmark._run_preflight()
            loader.assert_not_called()
            self.assertFalse(report.target_verified)

    def test_failed_benchmark_saves_standalone_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = benchmark_records.BenchmarkConfig(
                output=pathlib.Path(directory) / "run",
                model=pathlib.Path("missing.gguf"), model_revision="test",
                users=pathlib.Path("users"),
                cases=pathlib.Path("cases.json"),
                runtime_revision="test", threads=4,
            )
            with self.assertRaises(FileNotFoundError):
                benchmark._run_benchmark(settings)
            root = (
                settings.output / "logs" / "judge_benchmark"
            )
            self.assertTrue((root / "report.md").exists())
            report = json.loads(
                (root / "benchmark.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "failed")


    def test_completed_scores_survive_later_request_failure(self) -> None:
        scorer = mock.Mock()
        scorer.score.side_effect = [10.0, 20.0, RuntimeError("inference")]
        measurements = benchmark_records.BenchmarkMeasurements()
        case = judge_evaluation.load_cases(test_utils.CASES_PATH).cases[0]
        with self.assertRaisesRegex(RuntimeError, "inference"):
            benchmark._measure_requests(
                scorer, "pilot_00", case, mock.Mock(), pathlib.Path("unused"),
                measurements,
            )
        self.assertEqual(measurements.single_request.scores, [10.0])
        self.assertEqual(
            measurements.four_outputs_cached_history.scores, [20.0]
        )
        self.assertIsNotNone(measurements.single_request.seconds)
        self.assertIsNotNone(measurements.four_outputs_cached_history.seconds)

    def test_report_write_failure_still_closes_model(self) -> None:
        judge = mock.Mock(
            identity=test_utils.make_judge_identity(), load_seconds=0.01,
        )
        with tempfile.TemporaryDirectory() as directory:
            settings = benchmark_records.BenchmarkConfig(
                output=pathlib.Path(directory) / "run",
                model=pathlib.Path("unused.gguf"), model_revision="test",
                users=pathlib.Path("unused_users"), cases=test_utils.CASES_PATH,
                runtime_revision="test",
            )
            with (
                mock.patch.object(
                    local_judge, "LocalLlamaJudge", return_value=judge,
                ),
                mock.patch.object(
                    config, "load_users",
                    return_value=config.generate_users(),
                ),
                mock.patch.object(benchmark, "_MemoryMonitor") as monitor,
                mock.patch.object(benchmark, "_measure_requests"),
                mock.patch.object(
                    benchmark, "_write_report", side_effect=OSError("report"),
                ),
            ):
                monitor.return_value.stop.return_value = []
                with self.assertRaisesRegex(OSError, "report"):
                    benchmark._run_benchmark(settings)
        judge.close.assert_called_once_with()
        monitor.return_value.stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
