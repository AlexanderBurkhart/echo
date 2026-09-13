"""Defines the interface implemented by an LM judge."""

from typing import Protocol

from user_simulator.data import judge_identity


class Judge(Protocol):
    """An LM judge whose calls each use a fresh context.

    Attributes:
        identity: Fixed model, runtime, settings, and revisions.
    """

    identity: judge_identity.LocalJudgeIdentity

    def complete(self, prompt: str) -> str:
        """Returns raw LM judgment text for a single fresh-context prompt."""
        ...
