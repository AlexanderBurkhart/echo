"""Runs one hardcoded example through the actual LM judge and simulator."""

import contextlib
import pathlib

from conversation import history
from user_simulator import config
from user_simulator import simulator
from user_simulator.judge import local_judge

# Set these to the installed judge model and its actual source/build revisions.
MODEL_PATH = pathlib.Path("models/judge.gguf")
MODEL_REVISION = ""
RUNTIME_REVISION = ""
RUN_ROOT = pathlib.Path("artifacts/smoke_test_real_judge")
# Handwritten smoke-test user; the generated pilot users live in users/pilot.
USER_CONFIG_PATH = config.USERS_DIRECTORY / "user0.json"

CONVERSATION_HISTORY = history.ConversationHistory(messages=(
    history.Message(
        role=history.Role.USER,
        content="I have a prepared, approved notice template ready to use.",
    ),
    history.Message(
        role=history.Role.ASSISTANT,
        content="We can use that template when you are ready to send it.",
    ),
    history.Message(
        role=history.Role.USER,
        content=(
            "I need to send the notice within 10 minutes or miss today's "
            "delivery window. The wording might need corrections later. "
            "What should I do?"
        ),
    ),
))
OUTPUT = (
    "Send the prepared notice now using the approved template. Keep the "
    "source document editable so you can prepare a corrected notice "
    "if the wording changes."
)


def run_smoke_test() -> None:
    """Loads the real judge, scores one output, and prints the result."""
    user = config.load_user(USER_CONFIG_PATH)
    with contextlib.closing(local_judge.LocalLlamaJudge(
        MODEL_PATH, model_revision=MODEL_REVISION,
        runtime_revision=RUNTIME_REVISION,
    )) as judge:
        scorer = simulator.Simulator([user], judge)
        score = scorer.score(
            run_root=RUN_ROOT, user_id=user.user_id,
            conversation_history=CONVERSATION_HISTORY, output=OUTPUT,
            use_cache=False,
        )
    print(f"Preference score: {score:.2f}/100")
    print(f"Logs: {RUN_ROOT.resolve()}")


if __name__ == "__main__":
    run_smoke_test()
