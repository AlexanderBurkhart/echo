"""Checks the default smoke entry point without opening a model or holdout."""

import pathlib
import tempfile
import unittest
from unittest import mock

from experiments.smoke_test import main


class MainTest(unittest.TestCase):
    """Keeps routine smoke runs confined to development."""

    def test_default_runs_only_development_and_propagates_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = pathlib.Path(temporary) / "other.gguf"
            model.touch()
            for passed in (True, False):
                with self.subTest(passed=passed):
                    judge = mock.Mock(load_seconds=0.0)
                    with (
                        mock.patch.object(main, "RUNS_ROOT",
                                          pathlib.Path(temporary)),
                        mock.patch.object(main.local_judge, "LocalLlamaJudge",
                                          return_value=judge) as load,
                        mock.patch.object(main.evaluation,
                                          "run_development_evaluation",
                                          return_value=passed) as evaluate,
                        mock.patch.object(main.evaluation, "run_evaluation")
                        as evaluate_holdout,
                    ):
                        if passed:
                            main.run_smoke_test(
                                model, model_revision="other@revision",
                                runtime_revision="actual CPU build", threads=2,
                            )
                        else:
                            with self.assertRaises(SystemExit) as error:
                                main.run_smoke_test(
                                    model, model_revision="other@revision",
                                    runtime_revision="actual CPU build",
                                    threads=2,
                                )
                            self.assertEqual(error.exception.code, 1)
                    load.assert_called_once_with(
                        model, model_revision="other@revision",
                        runtime_revision="actual CPU build", threads=2,
                    )
                    evaluate.assert_called_once()
                    self.assertIs(evaluate.call_args.args[0], judge)
                    self.assertEqual(evaluate.call_args.args[1].parent,
                                     pathlib.Path(temporary))
                    evaluate_holdout.assert_not_called()
                    judge.close.assert_called_once()

    def test_missing_model_fails_before_creating_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with (
                mock.patch.object(main.runs, "capture_logs") as capture,
                mock.patch.object(main.local_judge, "LocalLlamaJudge") as load,
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError, "No download attempted",
                ):
                    main.run_smoke_test(
                        pathlib.Path(temporary) / "missing.gguf",
                        model_revision="test", runtime_revision="test",
                    )
            load.assert_not_called()
            capture.assert_not_called()

    def test_invalid_settings_fail_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = pathlib.Path(temporary) / "model.gguf"
            model.touch()
            for model_revision, runtime_revision, threads in (
                (" ", "runtime", 8), ("model", " ", 8), ("model", "runtime", 0),
            ):
                with self.subTest(
                    model=model_revision, runtime=runtime_revision,
                    threads=threads,
                ):
                    with mock.patch.object(
                        main.local_judge, "LocalLlamaJudge",
                    ) as load:
                        with self.assertRaises(ValueError):
                            main.run_smoke_test(
                                model, model_revision=model_revision,
                                runtime_revision=runtime_revision,
                                threads=threads,
                            )
                        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
