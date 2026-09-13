"""Scores five authored outputs through the real local LM judge.

Run from the repository root:
    .venv/bin/python -m experiments.smoke_test.main
"""

import contextlib
import datetime
import pathlib

from loguru import logger as log

from experiment_logging import runs
from experiments.smoke_test import cases
from experiments.smoke_test import review
from user_simulator import config
from user_simulator import simulator
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
USER_CONFIG_PATH = PROJECT_ROOT / config.USERS_DIRECTORY / "user0.json"
RUNS_ROOT = PROJECT_ROOT / "artifacts/smoke_test_real_judge"


def run_smoke_test() -> None:
    """Loads one user and real judge, scores five outputs, and saves reviews."""
    user = config.load_user(USER_CONFIG_PATH)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%fZ"
    )
    run_root = RUNS_ROOT / timestamp
    with runs.capture_logs(run_root):
        log.info("User config: {}", USER_CONFIG_PATH)
        log.info("Logs: {}", run_root)
        log.info("Task: {}", cases.CONVERSATION_HISTORY.messages[-1].content)
        log.info("Loading real LM judge: {}", MODEL_PATH)
        summary = []
        failures = []
        with contextlib.closing(local_judge.LocalLlamaJudge(
            MODEL_PATH, model_revision=MODEL_REVISION,
            runtime_revision=RUNTIME_REVISION, threads=8,
        )) as judge:
            scorer = simulator.Simulator([user], judge)
            log.info("Model loaded in {:.2f}s", judge.load_seconds)
            for index, case in enumerate(cases.CASES, 1):
                log.info(
                    "[{}/{}] Scoring {}...", index, len(cases.CASES), case.name
                )
                try:
                    score = scorer.score(
                        run_root=run_root, user_id=user.user_id,
                        conversation_history=cases.CONVERSATION_HISTORY,
                        output=case.output,
                    )
                except simulator.JudgeError as error:
                    message = (
                        f"{case.name}: LM call failed; {error.diagnostics}"
                    )
                    failures.append(message)
                    summary.append(message)
                    log.error(message)
                    continue
                path, record = review.load_output_record(
                    run_root, case.output
                )
                report, errors = review.review_output(case, record, score)
                report += f"Score record: {path}\n"
                runs.write_text(
                    run_root, "smoke_test", f"{case.name}.txt", report
                )
                log.log("ERROR" if errors else "INFO", "\n{}", report)
                failures.extend(errors)
                summary.append(
                    f"{case.name}: {score:.2f}/100 "
                    f"{'FAIL' if errors else 'PASS'}"
                )
        summary.append(f"Overall: {'FAIL' if failures else 'PASS'}")
        summary.append(f"Logs: {run_root}")
        runs.write_text(
            run_root, "smoke_test", "summary.txt", "\n".join(summary)
        )
        log.log("ERROR" if failures else "SUCCESS", "\n{}", "\n".join(summary))


if __name__ == "__main__":
    run_smoke_test()
