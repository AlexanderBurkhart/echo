"""Immutable identity of the local LM judge used for caching and logging."""

from collections.abc import Mapping
import dataclasses
import types
from typing import Any


def _freeze_metadata(value: Any) -> Any:
    """Copies JSON metadata into read-only maps and tuples."""
    if isinstance(value, Mapping):
        return types.MappingProxyType({
            key: _freeze_metadata(item) for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("Judge metadata must contain JSON-compatible values")


@dataclasses.dataclass(frozen=True)
class LocalJudgeIdentity:
    """The exact model, runtime, settings, and host recorded for a local judge.

    Dynamic metadata is copied and frozen recursively so changes to the
    runtime's original dictionaries cannot change a recorded identity.
    """

    backend: str
    binding_version: str
    runtime_revision: str
    runtime_files_sha256: Mapping[str, str]
    model_path: str
    model_revision: str
    model_sha256: str
    model_metadata: Mapping[str, Any]
    settings: Mapping[str, Any]
    generation: Mapping[str, Any]
    platform: str
    machine: str

    def __post_init__(self) -> None:
        """Freezes the nested metadata owned by this identity."""
        for name in (
            "runtime_files_sha256", "model_metadata", "settings", "generation",
        ):
            object.__setattr__(
                self, name, _freeze_metadata(getattr(self, name))
            )
