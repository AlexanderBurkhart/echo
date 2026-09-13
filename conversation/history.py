"""Defines the shared conversation data contract."""

import dataclasses
import enum


class Role(enum.StrEnum):
    """Identifies the speaker of a conversation message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclasses.dataclass(frozen=True)
class Message:
    """One visible conversation turn.

    Attributes:
        role: Speaker category.
        content: Visible text, including serialized tool observations.
    """

    role: Role
    content: str


@dataclasses.dataclass(frozen=True)
class ConversationHistory:
    """An immutable ordered sequence of messages supplied by the producer.

    Attributes:
        messages: Tuple of Message objects, in conversation order.
    """

    messages: tuple[Message, ...] = ()
