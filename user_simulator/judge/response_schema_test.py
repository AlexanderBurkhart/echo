"""Checks native grammar compilation and exact per-call JSON fields."""

import dataclasses
import json
import unittest

import llama_cpp

from user_simulator.data import judgments
from user_simulator.judge import response_schema
from user_simulator.judge.ratings import output_definitions


class ResponseSchemaTest(unittest.TestCase):
    """Keeps quantity fields distinct from criterion quality levels."""

    def test_schema_and_nested_rating_definitions_are_immutable(self) -> None:
        rating = response_schema.RatingSchema("urgency", "rating_0_4", 4)
        supplied = [rating]
        schema = response_schema.ResponseSchema(supplied, (), True)
        supplied.clear()
        self.assertEqual(schema.ratings[0], rating)
        with self.assertRaises(TypeError):
            schema.ratings[0] = rating
        with self.assertRaises(dataclasses.FrozenInstanceError):
            rating.maximum = 100

    def test_task_grammar_requires_empty_quantities(self) -> None:
        schema = response_schema.build_response_schema(
            judgments.JudgmentKind.TASK,
            output_definitions.DEFAULT_RATING_CRITERIA,
        )
        schema = schema.build_json_schema()
        self.assertEqual(schema["properties"]["quantities"]["properties"], {})
        quantity_schema = schema["properties"]["quantities"]
        self.assertFalse(quantity_schema["additionalProperties"])
        llama_cpp.LlamaGrammar.from_json_schema(
            json.dumps(schema), verbose=False,
        )

    def test_each_criterion_grammar_has_only_its_own_fields(self) -> None:
        for criterion in output_definitions.DEFAULT_RATING_CRITERIA:
            with self.subTest(criterion=criterion.name):
                schema = response_schema.build_response_schema(
                    judgments.JudgmentKind.OUTPUT, (criterion,),
                )
                schema = schema.build_json_schema()
                fields = schema["properties"]
                self.assertEqual(
                    fields["ratings"]["required"], [criterion.name],
                )
                quantity_names = (
                    [criterion.name]
                    if criterion.name in output_definitions.COST_COMPONENTS
                    else []
                )
                self.assertEqual(
                    fields["quantities"]["required"], quantity_names,
                )
                rating = fields["ratings"]["properties"][criterion.name]
                known = rating["oneOf"][0]["properties"]
                self.assertEqual(known["value"]["maximum"], 4)
                self.assertEqual(known["unit"], {"const": "rating_0_4"})
                llama_cpp.LlamaGrammar.from_json_schema(
                    json.dumps(schema), verbose=False,
                )



if __name__ == "__main__":
    unittest.main()
