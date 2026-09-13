"""Checks that prompt response shapes match the runtime LM contract."""

import dataclasses
import json
import unittest

from conversation import history
from user_simulator import config
from user_simulator.data import judgments
from user_simulator.judge import judging
from user_simulator.judge import prompts
from user_simulator.judge.ratings import ratings as rating_scale


class PromptsTest(unittest.TestCase):
    """Keeps required response fields aligned without adding expected scores."""

    def test_response_shapes_parse_for_task_and_extra_output_criterion(
        self,
    ) -> None:
        user = config.generate_users(count=1)[0]
        extra = rating_scale.RatingAnchoredCriterion(
            name="clarity",
            anchors=("opaque", "poor", "mixed", "clear", "exact"),
        )
        user = dataclasses.replace(
            user, rating_criteria=user.rating_criteria + (extra,)
        )
        for kind, prompt in (
            (judgments.JudgmentKind.TASK,
             prompts.build_task_prompt(history.ConversationHistory())),
            (judgments.JudgmentKind.OUTPUT,
             prompts.build_output_prompt(
                 history.ConversationHistory(), "example", user.rating_criteria
             )),
        ):
            with self.subTest(kind=kind):
                response = prompt.split(
                    "Do not add or omit fields.\n", 1
                )[1].split("\nHISTORY:\n", 1)[0]
                parsed = judging.parse_lm_judgment(
                    response, kind, [], user.rating_criteria
                )
                self.assertTrue(all(
                    rating.value is None for rating in parsed.ratings.values()
                ))
                self.assertNotIn("source_record", json.loads(response))
                if kind is judgments.JudgmentKind.OUTPUT:
                    self.assertIn("clarity", parsed.ratings)


if __name__ == "__main__":
    unittest.main()
