"""Evaluates four development users without opening the exposed holdout.

Run from the repository root:
    python -m experiments.smoke_test.main --help
"""

import argparse
import contextlib
import datetime
import pathlib

from loguru import logger as log

from experiment_logging import runs
from experiments.smoke_test import evaluation
from user_simulator.judge import local_judge

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "artifacts/smoke_test_real_judge"


def run_smoke_test(
    model_path: pathlib.Path, *, model_revision: str,
    runtime_revision: str, threads: int = 8,
) -> None:
    """Runs development with local assets and explicit provenance.

    No model is downloaded. Revisions describe the caller's installation;
    the judge also records hashes of the actual model and runtime files.
    """
    if not model_path.is_file():
        raise FileNotFoundError(
            f"GGUF model not found: {model_path}. Supply an existing local "
            "file with --model; see docs/model-setup.md. No download attempted."
        )
    if not model_revision.strip() or not runtime_revision.strip():
        raise ValueError("Nonempty model and runtime revisions are required")
    if type(threads) is not int or threads <= 0:
        raise ValueError("threads must be a positive integer")
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    run_root = RUNS_ROOT / timestamp
    with runs.capture_logs(run_root):
        log.info("Logs: {}", run_root)
        log.info("Loading real LM judge: {}", model_path)
        with contextlib.closing(local_judge.LocalLlamaJudge(
            model_path, model_revision=model_revision,
            runtime_revision=runtime_revision, threads=threads,
        )) as judge:
            log.info("Model loaded in {:.2f}s", judge.load_seconds)
            passed = evaluation.run_development_evaluation(judge, run_root)
        if not passed:
            raise SystemExit(1)


def _run_cli() -> None:
    """Parses explicit asset provenance for a development-only evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, required=True,
                        help="Existing local GGUF; never downloaded")
    parser.add_argument("--model-revision", required=True,
                        help="Model source repository@revision")
    parser.add_argument("--runtime-revision", required=True,
                        help="Actual llama.cpp revision and build provenance")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    run_smoke_test(
        args.model, model_revision=args.model_revision,
        runtime_revision=args.runtime_revision, threads=args.threads,
    )


if __name__ == "__main__":
    _run_cli()
