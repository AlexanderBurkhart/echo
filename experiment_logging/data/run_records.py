"""Typed identities and journal entries shared by experiment components."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class RunManifest:
    """Persistent identity and creation time of an experiment run."""

    schema_version: str
    run_id: str
    created_at: str


@dataclasses.dataclass(frozen=True)
class RecordMetadata:
    """Provenance attached to a component artifact at persistence time."""

    run_id: str
    component: str
    request_id: str | None
    saved_at: str


@dataclasses.dataclass(frozen=True)
class RunEvent:
    """One operational event in the shared chronological journal."""

    run_id: str
    timestamp: str
    component: str
    event: str
    status: str
    artifact: str | None
    request_id: str | None
