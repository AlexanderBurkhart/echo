"""Checks the local runtime boundary without loading a model."""

import pathlib
import tempfile
import types
import unittest
from unittest import mock

from experiment_logging import records
from user_simulator.judge import local_judge


class LocalLlamaJudgeTest(unittest.TestCase):
    """Verifies fixed settings, fresh contexts, and native resource cleanup."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.model_path = pathlib.Path(temporary.name) / "model.gguf"
        self.model_path.write_bytes(b"test model")
        self.model = mock.Mock(metadata={"name": "test"})
        self.module = types.SimpleNamespace(
            Llama=mock.Mock(return_value=self.model),
            __version__="test",
            __file__=str(self.model_path.parent / "binding.py"),
        )
        patcher = mock.patch.object(
            local_judge, "llama_cpp", self.module,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _load_judge(self) -> local_judge.LocalLlamaJudge:
        return local_judge.LocalLlamaJudge(
            self.model_path, model_revision="model-test",
            runtime_revision="runtime-test", threads=2,
        )

    def test_calls_reset_context_and_preserve_frozen_settings(self) -> None:
        self.model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "{}"}}],
        }
        judge = self._load_judge()
        self.addCleanup(judge.close)
        for prompt in ("first task", "second output"):
            self.assertEqual(judge.complete(prompt), "{}")
            arguments = self.model.create_chat_completion.call_args.kwargs
            self.assertEqual(arguments["messages"], [
                {"role": "user", "content": prompt},
            ])
            self.assertEqual(arguments["temperature"], 0)
            self.assertEqual(arguments["max_tokens"], 1024)
            self.assertEqual(
                arguments["response_format"]["type"], "json_object"
            )
        self.assertIn("schema", arguments["response_format"])
        self.assertEqual(
            records.compute_digest(arguments["response_format"]),
            records.compute_digest(judge.identity.generation["response_format"]),
        )
        self.assertEqual(self.model.reset.call_count, 2)
        settings = self.module.Llama.call_args.kwargs
        self.assertEqual(settings["n_ctx"], 8192)
        self.assertEqual(settings["n_threads"], 2)
        self.assertEqual(judge.identity.settings["n_ctx"], 8192)
        self.assertEqual(judge.identity.generation["temperature"], 0)

    def test_invalid_arguments_fail_before_checking_model_path(self) -> None:
        missing = self.model_path.parent / "missing.gguf"
        for model_revision, runtime_revision, threads, error in (
            ("", "runtime-test", 2, "revisions"),
            ("model-test", "", 2, "revisions"),
            ("model-test", "runtime-test", 0, "threads"),
        ):
            with self.subTest(error=error, model_revision=model_revision):
                with self.assertRaisesRegex(ValueError, error):
                    local_judge.LocalLlamaJudge(
                        missing, model_revision=model_revision,
                        runtime_revision=runtime_revision, threads=threads,
                    )
        self.module.Llama.assert_not_called()

    def test_failed_identity_capture_closes_loaded_model(self) -> None:
        with mock.patch.object(
            local_judge, "_compute_runtime_hashes", side_effect=OSError("read"),
        ):
            with self.assertRaisesRegex(OSError, "read"):
                self._load_judge()
        self.model.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
