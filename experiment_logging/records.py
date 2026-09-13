"""Serializes records and writes atomic JSON or text files for run logging."""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any


def _serialize_record(value: Any) -> dict[str, Any]:
    """Converts dataclasses and immutable mappings at the JSON boundary."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def serialize_canonical_json(value: Any) -> str:
    """Serializes JSON values and dataclasses, rejecting nonfinite numbers."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_serialize_record,
    )


def compute_digest(value: Any) -> str:
    """Returns the SHA-256 of the canonical JSON encoding of a value."""
    encoded = serialize_canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(
    path: pathlib.Path, value: Any, exclusive: bool = False
) -> None:
    """Writes JSON with owner-only permissions and atomic replacement.

    Args:
        path: Destination file; parent directories are created as needed.
        value: JSON-compatible data or dataclasses to persist.
        exclusive: Creates a new file only; an existing file raises an error.

    Raises:
        OSError: Creation, writing, syncing, or replacement fails.
    """
    write_text(path, serialize_canonical_json(value) + "\n", exclusive)


def write_text(
    path: pathlib.Path, text: str, exclusive: bool = False
) -> None:
    """Writes text atomically with owner-only access for new artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if exclusive:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        return
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
