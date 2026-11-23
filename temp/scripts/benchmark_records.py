"""Typed local benchmark settings, observations, and reports."""

import dataclasses
import enum
import pathlib

from user_simulator.data import judge_identity
from user_simulator.integration_tests import validation_records


class BenchmarkStatus(enum.StrEnum):
    """Execution outcome; successful measurements still require review."""

    RUNNING = "running"
    FAILED = "failed"
    MEASURED = "measured; quality and hardware review required"


class MemoryPhase(enum.StrEnum):
    """The benchmark activity associated with a memory sample."""

    STARTUP = "startup"
    LOAD = "load"
    SINGLE_REQUEST = "single_uncached_history"
    FOUR_OUTPUTS = "four_cached_history"
    VALIDATION = "judge_validation"


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    """Explicit local assets and settings supplied at the CLI boundary."""

    model: pathlib.Path
    model_revision: str
    runtime_revision: str
    users: pathlib.Path
    cases: pathlib.Path
    output: pathlib.Path
    threads: int = 4
    validate: bool = False


@dataclasses.dataclass(frozen=True)
class CommandResult:
    """Captured process output or a failure to run the diagnostic command."""

    stdout: str | None = None
    stderr: str | None = None
    returncode: int | None = None
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class MemorySample:
    """Raw process and OS memory observations with their collection phase."""

    monotonic_seconds: float
    phase: MemoryPhase
    rss_kib: CommandResult
    vm_stat: CommandResult | None = None
    swap_usage: CommandResult | None = None
    memory_pressure_query: CommandResult | None = None


@dataclasses.dataclass(frozen=True)
class PreflightReport:
    """Host observations, without claiming target feasibility or LM quality."""

    platform: str
    machine: str
    python: str
    binding_installed: bool
    hardware: dict[str, CommandResult]
    note: str = "Preflight only; no inference or model availability claim."


@dataclasses.dataclass
class RequestTiming:
    """Timing and completed scores for a batch, including partial failures."""

    expected_judge_calls_without_retries: int
    scores: list[float] = dataclasses.field(default_factory=list)
    seconds: float | None = None


@dataclasses.dataclass
class BenchmarkMeasurements:
    """Model loading and inference observations collected by the benchmark."""

    cold_model_load_seconds: float | None = None
    single_request: RequestTiming | None = None
    four_outputs_cached_history: RequestTiming | None = None


@dataclasses.dataclass
class BenchmarkReport:
    """One benchmark's progress, failure details, and review limitations."""

    preflight: PreflightReport
    measurements: BenchmarkMeasurements = dataclasses.field(
        default_factory=BenchmarkMeasurements
    )
    status: BenchmarkStatus = BenchmarkStatus.RUNNING
    judge: judge_identity.LocalJudgeIdentity | None = None
    error: str | None = None
    validation_status: validation_records.ValidationStatus = (
        validation_records.ValidationStatus.NOT_RUN
    )
    human_review_status: str | None = None
    memory_samples: list[MemorySample] = dataclasses.field(default_factory=list)
    process_peak_rss_bytes: int | None = None
    limitations: tuple[str, ...] = (
        "Expected calls assume an initially empty cache and no retries; "
        "the four-output batch reuses the first request's history and "
        "any repeated outputs. Counts are expectations, not measurements.",
        "Cold model construction, not guaranteed cold filesystem cache.",
        "RSS is not total memory pressure; OS diagnostics are saved.",
        "Memory sampling has overhead and may miss short peaks.",
        "Single timings are observations, not stable latency estimates.",
    )
