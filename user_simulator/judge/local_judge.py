"""Runs an existing local GGUF through the required llama.cpp bindings."""

import dataclasses
import hashlib
import json
import pathlib
import platform
import time
from typing import Any

import llama_cpp

from conversation import history
from user_simulator.data import judge_identity
from user_simulator.judge import runtime_settings

HASH_BLOCK_BYTES = 1024 * 1024
RESPONSE_SCHEMA_PATH = pathlib.Path(__file__).with_name("response_schema.json")


def _compute_file_sha256(path: str | pathlib.Path) -> str:
    """Returns a file digest without loading the entire file into memory."""
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(HASH_BLOCK_BYTES), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _compute_runtime_hashes(package: pathlib.Path) -> dict[str, str]:
    """Returns hashes of the installed wrapper and bundled native libraries."""
    hashes = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".so", ".dylib", ".dll"):
            hashes[str(path.relative_to(package))] = _compute_file_sha256(path)
    return hashes


class LocalLlamaJudge:
    """A resident model with a fresh context for each LM judgment.

    Attributes:
        identity: Model/runtime hashes, revisions, metadata, and fixed settings.
        load_seconds: Model-constructor duration, excluding subsequent hashing.
        last_response: Complete runtime response from the last LM judgment.
    """

    def __init__(
        self,
        model_path: str | pathlib.Path,
        *,
        model_revision: str,
        runtime_revision: str,
        threads: int = 4,
    ) -> None:
        """Loads authorized, already-installed assets without network access.

        Args:
            model_path: Existing local GGUF file.
            model_revision: Exact source revision of the supplied model.
            runtime_revision: Exact llama.cpp build revision and provenance.
            threads: Positive number of CPU execution and batch threads.

        Raises:
            ValueError: Revisions are missing or threads is not positive.
            OSError: Local files or model resources cannot be accessed.
        """
        if not model_revision or not runtime_revision:
            raise ValueError("Exact model and runtime revisions are required")
        if type(threads) is not int or threads <= 0:
            raise ValueError("threads must be a positive integer")
        path = pathlib.Path(model_path).resolve(strict=True)
        # Capture the actual external runtime arguments as backend metadata.
        self._settings = dict(
            runtime_settings.MODEL_OPTIONS,
            n_threads=threads, n_threads_batch=threads,
        )
        self._generation = dict(
            runtime_settings.GENERATION_OPTIONS,
            response_format={
                "type": runtime_settings.RESPONSE_FORMAT,
                "schema": json.loads(
                    RESPONSE_SCHEMA_PATH.read_text(encoding="utf-8")
                ),
            },
        )
        started = time.perf_counter()
        self._model = llama_cpp.Llama(
            model_path=str(path), **self._settings
        )
        self.load_seconds = time.perf_counter() - started
        try:
            self.identity = judge_identity.LocalJudgeIdentity(
                backend="llama-cpp-python (direct in-process llama.cpp)",
                binding_version=llama_cpp.__version__,
                runtime_revision=runtime_revision,
                runtime_files_sha256=_compute_runtime_hashes(
                    pathlib.Path(llama_cpp.__file__).parent
                ),
                model_path=str(path),
                model_revision=model_revision,
                model_sha256=_compute_file_sha256(path),
                model_metadata=self._model.metadata,
                settings=self._settings,
                generation=self._generation,
                platform=platform.platform(),
                machine=platform.machine(),
            )
        except Exception:
            self._model.close()
            raise
        self.last_response: dict[str, Any] | None = None

    def complete(self, prompt: str) -> str:
        """Returns raw LM judgment text after resetting the model context."""
        self._model.reset()
        # A single user turn uses the model's pinned GGUF chat template.
        self.last_response = self._model.create_chat_completion(
            messages=[dataclasses.asdict(history.Message(
                role=history.Role.USER, content=prompt,
            ))],
            **self._generation,
        )
        return self.last_response["choices"][0]["message"]["content"]

    def close(self) -> None:
        """Releases the resident model and native runtime resources."""
        self._model.close()
