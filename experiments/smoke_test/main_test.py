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
            for passed in (True, False):
                with self.subTest(passed=passed):
                    judge = mock.Mock(load_seconds=0.0)
                    with (
                        mock.patch.object(main, "RUNS_ROOT",
                                          pathlib.Path(temporary)),
                        mock.patch.object(main.local_judge, "LocalLlamaJudge",
                                          return_value=judge),
                        mock.patch.object(main.evaluation,
                                          "run_development_evaluation",
                                          return_value=passed) as evaluate,
                        mock.patch.object(main.evaluation, "run_evaluation")
                        as evaluate_holdout,
                    ):
                        if passed:
                            main.run_smoke_test()
                        else:
                            with self.assertRaises(SystemExit) as error:
                                main.run_smoke_test()
                            self.assertEqual(error.exception.code, 1)
                    evaluate.assert_called_once()
                    self.assertIs(evaluate.call_args.args[0], judge)
                    self.assertEqual(evaluate.call_args.args[1].parent,
                                     pathlib.Path(temporary))
                    evaluate_holdout.assert_not_called()
                    judge.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
