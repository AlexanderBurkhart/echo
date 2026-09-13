"""Five authored outputs and expected ratings for the real-LM smoke test."""

from collections.abc import Mapping
import dataclasses
import types

from conversation import history


@dataclasses.dataclass(frozen=True)
class SmokeCase:
    """One candidate and its smoke-only expectations, never sent to the LM."""

    name: str
    output: str
    expected_ratings: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "expected_ratings",
            types.MappingProxyType(dict(self.expected_ratings)),
        )


EXPECTED_RULE = "prefer_speed_when_urgent"
CONVERSATION_HISTORY = history.ConversationHistory(messages=(
    history.Message(
        role=history.Role.USER,
        content=(
            "The announcement draft is complete and approved for release. "
            "I have the editable source document ready."
        ),
    ),
    history.Message(
        role=history.Role.ASSISTANT,
        content="The approved draft is ready to send without additional work.",
    ),
    history.Message(
        role=history.Role.USER,
        content=(
            "This is urgent: I must send the announcement in the next "
            "2 minutes. Missing this deadline cancels today's launch. "
            "The launch details are still uncertain and may change after "
            "sending, so I need to be able to revise and send a correction. "
            "What should I do?"
        ),
    ),
))
CASES = (
    SmokeCase(
        name="fast_and_adaptable",
        output=(
            "Send the approved announcement immediately, within the "
            "2-minute deadline. Keep the editable source and a copy of the "
            "sent version. Because the launch details may change, revise "
            "the source and send a correction if new details arrive."
        ),
        expected_ratings={"urgency": 4, "uncertainty": 4},
    ),
    SmokeCase(
        name="brief_delay_but_adaptable",
        output=(
            "Pause for 15 seconds to reopen the already approved draft, "
            "although it needs no further review. Then send it within the "
            "2-minute deadline. Keep the editable source and a copy of the "
            "sent version. Because the launch details may change, revise "
            "the source and send a correction if new details arrive."
        ),
        expected_ratings={"urgency": 3, "uncertainty": 4},
    ),
    SmokeCase(
        name="longer_delay_but_adaptable",
        output=(
            "Wait 45 seconds before sending the already approved "
            "announcement, just in case another idea occurs to you. "
            "No further review is needed. Send it in the remaining "
            "75 seconds before the deadline. Keep the editable source "
            "and a copy of the sent version. Because the launch details "
            "may change, revise the source and send a correction if new "
            "details arrive."
        ),
        expected_ratings={"urgency": 2, "uncertainty": 4},
    ),
    SmokeCase(
        name="fast_but_incomplete_corrections",
        output=(
            "Send the approved announcement immediately, within the "
            "2-minute deadline. Keep the editable source and a copy of "
            "the sent version. If the launch date changes, revise the "
            "source and send a correction to the original recipients. "
            "However, if other launch details change, leave those parts "
            "of the announcement as they were and do not correct them."
        ),
        expected_ratings={"urgency": 4, "uncertainty": 2},
    ),
    SmokeCase(
        name="delayed_and_rigid",
        output=(
            "Wait until tomorrow to send the announcement, even though this "
            "misses the 2-minute deadline and cancels today's launch. Then "
            "permanently delete the editable source and all copies. Treat "
            "the announcement as final: ignore later changes to the launch "
            "details and do not send corrections."
        ),
        expected_ratings={"urgency": 0, "uncertainty": 0},
    ),
)
