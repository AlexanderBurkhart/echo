"""Checks the local runtime boundary without loading a model."""

import pathlib
import tempfile
import types
import unittest
from unittest import mock

from experiment_logging import records
from user_simulator.data import judgments
from user_simulator.judge import local_judge
from user_simulator.judge import response_schema
from user_simulator.judge.ratings import output_definitions


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
        schema = response_schema.build_response_schema(
            judgments.JudgmentKind.TASK,
            output_definitions.DEFAULT_RATING_CRITERIA,
        )
        for prompt in ("first task", "second task"):
            self.assertEqual(
                judge.complete(prompt, response_schema=schema), "{}",
            )
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
            arguments["response_format"]["schema"], schema.build_json_schema(),
        )
        self.assertEqual(self.model.reset.call_count, 2)
        settings = self.module.Llama.call_args.kwargs
        self.assertEqual(settings["n_ctx"], 8192)
        self.assertEqual(settings["n_threads"], 2)
        self.assertEqual(judge.identity.settings["n_ctx"], 8192)
        self.assertEqual(judge.identity.generation["temperature"], 0)

    def test_each_call_uses_its_schema_without_changing_frozen_settings(self) -> None:
        self.model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "{}"}}],
        }
        judge = self._load_judge()
        self.addCleanup(judge.close)
        identity = records.compute_digest(judge.identity)
        schema = response_schema.build_response_schema(
            judgments.JudgmentKind.OUTPUT,
            (output_definitions.DEFAULT_RATING_CRITERIA[0],),
        )
        judge.complete("one criterion", response_schema=schema)
        arguments = self.model.create_chat_completion.call_args.kwargs
        self.assertEqual(
            arguments["response_format"]["schema"], schema.build_json_schema(),
        )
        task_schema = response_schema.build_response_schema(
            judgments.JudgmentKind.TASK,
            output_definitions.DEFAULT_RATING_CRITERIA,
        )
        judge.complete("task", response_schema=task_schema)
        arguments = self.model.create_chat_completion.call_args.kwargs
        self.assertEqual(
            arguments["response_format"]["schema"],
            task_schema.build_json_schema(),
        )
        self.assertNotIn("response_format", judge.identity.generation)
        self.assertEqual(records.compute_digest(judge.identity), identity)

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
