"""Shows conditional scoring with authored measurements, without an LM.

Run from the repository root: python3 -m examples.offline_scoring
"""

from user_simulator import config
from user_simulator import engine
from user_simulator.data import judgments
from user_simulator.judge.ratings import ratings
from user_simulator.judge.ratings import task_definitions


def run_example() -> None:
    """Scores fixed alternatives before and after urgency activates a rule."""
    user = config.UserConfig(
        user_id="example_user",
        rules=(
            config.PreferenceRule(
                id="prefer_flexibility", precedence=0, when=None,
                weights={"urgency": 0.2, "uncertainty": 0.8},
            ),
            config.PreferenceRule(
                id="prefer_speed_when_urgent", precedence=1,
                when=config.Condition(
                    property="urgency",
                    operator=config.ConditionOperator.GREATER_EQUAL,
                    threshold=80,
                ),
                weights={"urgency": 0.8, "uncertainty": 0.2},
            ),
        ),
    )
    alternatives = {
        "Ship a fixed solution now": {"urgency": 4, "uncertainty": 0},
        "Wait to deliver a revisable solution": {
            "urgency": 0, "uncertainty": 4,
        },
    }
    print("Authored measurements; no model inference or task execution.")
    print("Output ratings (0-4):")
    for output, levels in alternatives.items():
        print(f"  {output}: urgency={levels['urgency']}, "
              f"uncertainty={levels['uncertainty']}")
    for urgency, expected_rule, expected_scores in (
        (20, "prefer_flexibility", (20.0, 80.0)),
        (90, "prefer_speed_when_urgent", (80.0, 20.0)),
    ):
        task = {
            "urgency": judgments.Rating(
                urgency, task_definitions.TASK_RATING_UNIT,
            ),
        }
        print(f"\nTask urgency: {urgency}/100")
        for (output, levels), expected_score in zip(
            alternatives.items(), expected_scores,
        ):
            measurements = {
                name: judgments.Rating(level, ratings.OUTPUT_RATING_UNIT)
                for name, level in levels.items()
            }
            result = engine.calculate_preference_score(user, task, measurements)
            assert result.selected_rule == expected_rule
            assert result.score == expected_score
            print(f"  {output}: {result.score:.0f}/100 "
                  f"(rule: {result.selected_rule})")
    print("\nUrgency changes the weights and reverses the ranking.")
    print("Score = 100 * weighted mean of rating/4; "
          "these rules ignore other criteria.")
    print("This demonstrates scoring, not judge accuracy, "
          "personalization learning, or human calibration.")


if __name__ == "__main__":
    run_example()
