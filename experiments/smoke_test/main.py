"""Evaluates four development users without opening the exposed holdout.

Run from the repository root:
    .venv/bin/python -m experiments.smoke_test.main
"""

import contextlib
import datetime
import pathlib

from loguru import logger as log

from experiment_logging import runs
from experiments.smoke_test import evaluation
from user_simulator.judge import local_judge

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
MODEL_REVISION = (
    "bartowski/Qwen2.5-14B-Instruct-GGUF@"
    "05244aa5d871c661c80082a15d3bce44714d068d"
)
RUNTIME_REVISION = (
    "llama-cpp-python v0.3.35; "
    "llama.cpp 4df29be4f4c3673f428170fda944a5b19f743bb8; "
    "GGML_CUDA=ON; CUDA architecture 89"
)
RUNS_ROOT = PROJECT_ROOT / "artifacts/smoke_test_real_judge"


def run_smoke_test() -> None:
    """Runs the real judge on development with the existing MAE gate."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    run_root = RUNS_ROOT / timestamp
    with runs.capture_logs(run_root):
        log.info("Logs: {}", run_root)
        log.info("Loading real LM judge: {}", MODEL_PATH)
        with contextlib.closing(local_judge.LocalLlamaJudge(
            MODEL_PATH, model_revision=MODEL_REVISION,
            runtime_revision=RUNTIME_REVISION, threads=8,
        )) as judge:
            log.info("Model loaded in {:.2f}s", judge.load_seconds)
            passed = evaluation.run_development_evaluation(judge, run_root)
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    run_smoke_test()
