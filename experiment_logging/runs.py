"""Stores component artifacts and a shared evaluator-only event journal."""

from collections.abc import Iterator
import contextlib
import dataclasses
import datetime
import fcntl
import json
import os
import pathlib
from typing import Any
import uuid

from loguru import logger as log

from experiment_logging import records
from experiment_logging.data import run_records

REQUESTS_DIRECTORY = "requests"
LOGS_DIRECTORY = "logs"
MANIFEST_FILENAME = "run.json"
JOURNAL_FILENAME = "events.jsonl"
RUNTIME_LOG_FILENAME = "run.log"
LOCK_FILENAME = ".run.lock"
SCHEMA_VERSION = "run-log-v1"


def _get_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _validate_relative_path(value: str | pathlib.Path) -> pathlib.Path:
    """Returns a relative path that stays within its component directory."""
    path = pathlib.Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Expected a nonempty relative path: {value!r}")
    return path


def get_component_directory(
    run_root: str | pathlib.Path, component: str
) -> pathlib.Path:
    """Returns a component's artifact path without creating any files."""
    logs_root = pathlib.Path(run_root).resolve() / LOGS_DIRECTORY
    relative = _validate_relative_path(component)
    path = logs_root / relative
    if not path.resolve().is_relative_to(logs_root.resolve()):
        raise ValueError("Component path escapes the run logs directory")
    return path


def write_record(
    run_root: str | pathlib.Path,
    component: str,
    path: str | pathlib.Path,
    value: Any,
    *,
    request_id: str | None = None,
) -> pathlib.Path:
    """Writes a component record and journals it under its run identity.

    Args:
        run_root: Run directory; run.json is created or reopened on demand.
        component: Producer namespace within the run.
        path: Component-relative path or absolute path within that component.
        value: Component-owned dataclass or JSON mapping.
        request_id: Explicit correlation ID. When omitted, paths under
            requests/<request_id>/ supply it; other artifacts have no ID.

    Returns:
        The saved artifact's absolute path.

    Raises:
        ValueError: The artifact path escapes its component directory.
        OSError: Artifact persistence or journaling fails.
    """
    relative_path = pathlib.Path(path)
    if relative_path.is_absolute():
        relative_path = relative_path.relative_to(
            get_component_directory(run_root, component)
        )
    if (request_id is None and len(relative_path.parts) >= 3
            and relative_path.parts[0] == REQUESTS_DIRECTORY):
        request_id = relative_path.parts[1]
    return RunLogger(run_root).write_record(
        component, relative_path, value, request_id=request_id,
    )


def write_text(
    run_root: str | pathlib.Path,
    component: str,
    relative_path: str | pathlib.Path,
    text: str,
) -> pathlib.Path:
    """Writes a component-relative text artifact and journals its location."""
    return RunLogger(run_root).write_text(component, relative_path, text)


@contextlib.contextmanager
def capture_logs(run_root: str | pathlib.Path) -> Iterator[None]:
    """Saves Loguru messages from this run's context to logs/run.log.

    Configure this once around a run's entry point. Components use the shared
    Loguru logger directly; the context supplies their run identity. The file
    sink is removed on exit without removing any caller-owned sinks.
    """
    run = RunLogger(run_root)
    sink_id = log.add(
        run.root / LOGS_DIRECTORY / RUNTIME_LOG_FILENAME,
        filter=lambda record: record["extra"].get("run_id") == run.run_id,
        encoding="utf-8", diagnose=False, catch=False,
        opener=lambda path, flags: os.open(path, flags, 0o600),
    )
    try:
        with log.contextualize(run_id=run.run_id):
            yield
    finally:
        log.remove(sink_id)


class RunLogger:
    """A shared run identity, artifact root, and chronological event journal.

    Each component retains its own artifact schema. The journal contains only
    operational metadata and artifact references; it is still evaluator-only.
    POSIX locks coordinate journal writers and run initialization. Artifact
    destinations must have a single owner; same-path writes replace snapshots.

    Attributes:
        root: Absolute run directory.
        run_id: Persisted identity reused when reopening the run.
    """

    def __init__(self, root: str | pathlib.Path) -> None:
        """Creates or reopens a run without replacing its identity.

        Args:
            root: Directory shared by all components of one experiment run.
        """
        self.root = pathlib.Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._logs_root = self.root / LOGS_DIRECTORY
        self._logs_root.mkdir(exist_ok=True, mode=0o700)
        self._journal = self._logs_root / JOURNAL_FILENAME
        with self._lock_run():
            manifest_path = self.root / MANIFEST_FILENAME
            if manifest_path.exists():
                manifest = run_records.RunManifest(**json.loads(
                    manifest_path.read_text(encoding="utf-8")
                ))
                if manifest.schema_version != SCHEMA_VERSION:
                    raise ValueError("Unsupported run log schema")
            else:
                manifest = run_records.RunManifest(
                    schema_version=SCHEMA_VERSION,
                    run_id=uuid.uuid4().hex,
                    created_at=_get_timestamp(),
                )
                records.write_json(manifest_path, manifest)
            self.run_id = manifest.run_id

    @contextlib.contextmanager
    def _lock_run(self) -> Iterator[None]:
        """Serializes manifest initialization and journal append operations."""
        descriptor = os.open(
            self._logs_root / LOCK_FILENAME, os.O_CREAT | os.O_RDWR, 0o600
        )
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def get_component_directory(self, component: str) -> pathlib.Path:
        """Returns the artifact directory for a component namespace."""
        return get_component_directory(self.root, component)

    def log_event(
        self,
        component: str,
        event: str,
        *,
        status: str = "ok",
        artifact: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Appends a synchronized event without copying artifact payloads.

        Args:
            component: Producer namespace, such as user_simulator or training.
            event: Verb describing the operation, such as write_record.
            status: Operational outcome, not an objective task-success score.
            artifact: Optional path relative to the run root.
            request_id: Optional correlation ID for related component events.

        Raises:
            OSError: The mandatory event journal cannot be written.
        """
        self.get_component_directory(component)
        if artifact is not None:
            _validate_relative_path(artifact)
        with self._lock_run():
            entry = run_records.RunEvent(
                run_id=self.run_id,
                timestamp=_get_timestamp(),
                component=component,
                event=event,
                status=status,
                artifact=artifact,
                request_id=request_id,
            )
            descriptor = os.open(
                self._journal, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
            )
            with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                stream.write(records.serialize_canonical_json(entry) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        log.bind(
            run_id=self.run_id, component=component, request_id=request_id,
            artifact=artifact, status=status,
        ).debug("{}: {} ({})", component, event, status)

    def _get_artifact_path(
        self, component: str, relative_path: str | pathlib.Path
    ) -> pathlib.Path:
        """Resolves a destination that remains within its component."""
        directory = self.get_component_directory(component)
        path = directory / _validate_relative_path(relative_path)
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Artifact path escapes its component directory")
        return path

    def write_text(
        self, component: str, relative_path: str | pathlib.Path, text: str
    ) -> pathlib.Path:
        """Writes a text snapshot and journals it after success."""
        path = self._get_artifact_path(component, relative_path)
        records.write_text(path, text)
        self.log_event(
            component, "write_text", artifact=str(path.relative_to(self.root)),
        )
        return path

    def write_record(
        self,
        component: str,
        relative_path: str | pathlib.Path,
        value: Any,
        *,
        request_id: str | None = None,
    ) -> pathlib.Path:
        """Writes a JSON snapshot and then journals its relative location.

        Args:
            component: Producer namespace within this run.
            relative_path: Artifact location within the component directory.
            value: Component-owned dataclass or JSON mapping; it is not mutated.
            request_id: Optional request correlation ID.

        Returns:
            The absolute saved artifact path. The object includes _record
            provenance separate from its component-specific fields.

        Raises:
            OSError: Artifact persistence or subsequent journaling fails.
        """
        path = self._get_artifact_path(component, relative_path)
        # This is the persistence boundary for component-owned dataclasses.
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            payload = {
                field.name: getattr(value, field.name)
                for field in dataclasses.fields(value)
            }
        else:
            payload = dict(value)
        if "_record" in payload:
            raise ValueError("_record is reserved for run provenance")
        payload["_record"] = run_records.RecordMetadata(
            run_id=self.run_id,
            component=component,
            request_id=request_id,
            saved_at=_get_timestamp(),
        )
        records.write_json(path, payload)
        self.log_event(
            component,
            "write_record",
            status=str(payload.get("status", "written")),
            artifact=str(path.relative_to(self.root)),
            request_id=request_id,
        )
        return path
