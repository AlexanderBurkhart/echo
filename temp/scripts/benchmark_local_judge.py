"""Benchmarks a local judge without an API server or automatic downloads."""

import argparse
import importlib.util
import os
import pathlib
import platform
import queue
import resource
import subprocess
import threading
import time

from loguru import logger as log

from experiment_logging import records
from experiment_logging import runs
from temp.scripts import benchmark_records
from user_simulator import config
from user_simulator import simulator
from user_simulator.integration_tests import judge_evaluation
from user_simulator.integration_tests import validation_records
from user_simulator.judge import local_judge

COMPONENT = "judge_benchmark"
REPORT_FILENAME = "benchmark.json"
REPORT_MARKDOWN_FILENAME = "report.md"
COMMAND_TIMEOUT_SECONDS = 3
MEMORY_SAMPLE_INTERVAL_SECONDS = 1


def _run_command(command: list[str]) -> benchmark_records.CommandResult:
    """Returns a bounded read-only command result, preserving failures."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True,
            timeout=COMMAND_TIMEOUT_SECONDS, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return benchmark_records.CommandResult(error=str(error))
    return benchmark_records.CommandResult(
        stdout=result.stdout, stderr=result.stderr,
        returncode=result.returncode,
    )


class _MemoryMonitor:
    """Collects process and OS memory observations in a background thread."""

    def __init__(self) -> None:
        self._samples: list[benchmark_records.MemorySample] = []
        self._phases: queue.Queue[benchmark_records.MemoryPhase] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_memory, daemon=True)

    def set_phase(self, phase: benchmark_records.MemoryPhase) -> None:
        """Queues a phase change without relying on atomic Python assignment."""
        self._phases.put(phase)

    def _sample_memory(self) -> None:
        """Collects approximate memory samples until explicitly stopped."""
        phase = benchmark_records.MemoryPhase.STARTUP
        while not self._stop.is_set():
            try:
                while True:
                    phase = self._phases.get_nowait()
            except queue.Empty:
                pass
            is_mac = platform.system() == "Darwin"
            self._samples.append(benchmark_records.MemorySample(
                monotonic_seconds=time.perf_counter(), phase=phase,
                rss_kib=_run_command([
                    "ps", "-o", "rss=", "-p", str(os.getpid()),
                ]),
                vm_stat=_run_command(["vm_stat"]) if is_mac else None,
                swap_usage=(
                    _run_command(["sysctl", "vm.swapusage"]) if is_mac else None
                ),
                memory_pressure_query=(
                    _run_command(["memory_pressure", "-Q"]) if is_mac else None
                ),
            ))
            self._stop.wait(MEMORY_SAMPLE_INTERVAL_SECONDS)

    def start(self) -> None:
        """Starts the sampling thread once."""
        self._thread.start()

    def stop(self) -> list[benchmark_records.MemorySample]:
        """Stops the thread before returning its collected samples."""
        self._stop.set()
        self._thread.join()
        return self._samples


def _run_preflight() -> benchmark_records.PreflightReport:
    """Returns host observations without importing a model runtime."""
    hardware = {}
    if platform.system() == "Darwin":
        for key in ("hw.model", "hw.memsize", "machdep.cpu.brand_string"):
            hardware[key] = _run_command(["sysctl", "-n", key])
    return benchmark_records.PreflightReport(
        platform=platform.platform(), machine=platform.machine(),
        python=platform.python_version(),
        binding_installed=importlib.util.find_spec("llama_cpp") is not None,
        hardware=hardware,
    )


def _measure_requests(
    scorer: simulator.Simulator,
    user_id: str,
    case: validation_records.ComparisonCase,
    monitor: _MemoryMonitor,
    run_root: pathlib.Path,
    measurements: benchmark_records.BenchmarkMeasurements,
) -> None:
    """Records inference timing, preserving completed scores if a call fails."""
    monitor.set_phase(benchmark_records.MemoryPhase.SINGLE_REQUEST)
    measurements.single_request = benchmark_records.RequestTiming(
        expected_judge_calls_without_retries=2,
    )
    started = time.perf_counter()
    try:
        measurements.single_request.scores.append(scorer.score(
            run_root=run_root, user_id=user_id,
            conversation_history=case.history, output=case.first.output,
        ))
    finally:
        measurements.single_request.seconds = time.perf_counter() - started
    monitor.set_phase(benchmark_records.MemoryPhase.FOUR_OUTPUTS)
    outputs = (
        "Send the prepared notice immediately, without waiting.",
        case.second.output,
        "Deliver the prepared notice right now.",
        "Leave the prepared notice waiting for several days.",
    )
    measurements.four_outputs_cached_history = benchmark_records.RequestTiming(
        expected_judge_calls_without_retries=4,
    )
    started = time.perf_counter()
    try:
        for output in outputs:
            measurements.four_outputs_cached_history.scores.append(scorer.score(
                run_root=run_root, user_id=user_id,
                conversation_history=case.history, output=output,
            ))
    finally:
        measurements.four_outputs_cached_history.seconds = (
            time.perf_counter() - started
        )


def _run_benchmark(settings: benchmark_records.BenchmarkConfig) -> None:
    """Runs explicitly supplied local assets and saves results or a failure."""
    root = settings.output
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    with runs.capture_logs(root):
        report = benchmark_records.BenchmarkReport(preflight=_run_preflight())
        runs.write_record(root, COMPONENT, REPORT_FILENAME, report)
        monitor = _MemoryMonitor()
        judge = None
        monitor.start()
        try:
            monitor.set_phase(benchmark_records.MemoryPhase.LOAD)
            judge = local_judge.LocalLlamaJudge(
                settings.model, model_revision=settings.model_revision,
                runtime_revision=settings.runtime_revision,
                threads=settings.threads,
            )
            report.judge = judge.identity
            report.measurements.cold_model_load_seconds = judge.load_seconds
            users = config.load_users(settings.users)
            scorer = simulator.Simulator(
                users, judge, component=f"{COMPONENT}/scoring"
            )
            case = judge_evaluation.load_cases(settings.cases).cases[0]
            _measure_requests(
                scorer, users[0].user_id, case, monitor, root,
                report.measurements,
            )
            if settings.validate:
                monitor.set_phase(benchmark_records.MemoryPhase.VALIDATION)
                result = judge_evaluation.run_evaluation(
                    judge, settings.cases, root
                )
                report.validation_status = result.status
                report.human_review_status = result.human_review_status
            report.status = benchmark_records.BenchmarkStatus.MEASURED
        except Exception as error:
            report.status = benchmark_records.BenchmarkStatus.FAILED
            report.error = f"{type(error).__name__}: {error}"
            log.error("Benchmark failed: {}", report.error)
            raise
        finally:
            # Release the model even if monitoring or logging fails.
            try:
                report.memory_samples = monitor.stop()
                multiplier = 1 if platform.system() == "Darwin" else 1024
                report.process_peak_rss_bytes = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * multiplier
                )
                runs.write_record(root, COMPONENT, REPORT_FILENAME, report)
                _write_report(root, report)
            finally:
                if judge is not None:
                    judge.close()


def _write_report(
    run_root: pathlib.Path, report: benchmark_records.BenchmarkReport
) -> None:
    """Writes the readable benchmark report through shared persistence."""
    limitations = "\n".join("- " + item for item in report.limitations)
    measurements = records.serialize_canonical_json(report.measurements)
    text = (
        f"# Local judge benchmark\n\nStatus: {report.status}\n\n"
        "These are simulator preference scores based on frozen-LM "
        "measurements. Objective correctness is separate.\n\n"
        f"Full diagnostics: `{REPORT_FILENAME}`. LM inputs and ratings: "
        "`scoring/`.\n\n"
        f"```json\n{measurements}\n```\n\n{limitations}\n\n"
        "Hardware selection and API development require review of these "
        "observations and the separate LM judgment quality audit.\n"
    )
    runs.write_text(run_root, COMPONENT, REPORT_MARKDOWN_FILENAME, text)


def _run_cli() -> None:
    """Runs preflight or benchmarks explicitly supplied local assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--model", type=pathlib.Path, help="Existing local GGUF"
    )
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--runtime-revision", help="Exact llama.cpp build revision"
    )
    parser.add_argument("--users", type=pathlib.Path,
                        default=config.USERS_DIRECTORY)
    parser.add_argument(
        "--cases", type=pathlib.Path,
        default=pathlib.Path(
            "user_simulator/integration_tests/judge_cases.json"
        ),
    )
    parser.add_argument("--output", type=pathlib.Path,
                        default=pathlib.Path("artifacts/mac-judge-benchmark"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--validate", action="store_true",
                        help="Run 12 cases, three uncached repetitions")
    args = parser.parse_args()
    if args.preflight:
        log.info("Preflight: {}", records.serialize_canonical_json(
            _run_preflight()
        ))
        return
    if not args.model or not args.model_revision or not args.runtime_revision:
        parser.error("Existing model and exact model/runtime revisions needed")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    _run_benchmark(benchmark_records.BenchmarkConfig(
        model=args.model, model_revision=args.model_revision,
        runtime_revision=args.runtime_revision, users=args.users,
        cases=args.cases, output=args.output, threads=args.threads,
        validate=args.validate,
    ))


if __name__ == "__main__":
    _run_cli()
