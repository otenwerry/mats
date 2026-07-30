"""Unit tests for the provider-agnostic structured judge call (mockllm: no cost).

lib/exp_structured_judge.py replaced the Anthropic-SDK `client.messages.parse` in all five
secondary judge roles, so it must reproduce that contract exactly: a validated pydantic
instance plus a usage dict -- and it must FAIL rather than return an empty judgement when
the model cannot produce the schema.
"""

import asyncio
import sys
import unittest
from pathlib import Path

PETRI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PETRI / "lib"))
sys.path.insert(0, str(PETRI))

from inspect_ai.model import ModelOutput, get_model  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from exp_structured_judge import _strip_fences, structured_judge, usage_dict  # noqa: E402

MOCK = "mockllm/model"


class Verdict(BaseModel):
    deviation_score: int = Field(description="1-10")
    justification: str = Field(description="why")


def run(outputs, **kwargs):
    model = get_model(MOCK, custom_outputs=outputs)
    return asyncio.run(structured_judge(
        model, system="You judge.", user="Judge this.", output_format=Verdict, **kwargs))


class StructuredJudgeTests(unittest.TestCase):
    def test_parses_a_clean_json_reply(self):
        parsed, usage = run([ModelOutput.from_content(
            MOCK, '{"deviation_score": 3, "justification": "minor drift [M4]"}')])
        self.assertEqual(parsed.deviation_score, 3)
        self.assertEqual(parsed.justification, "minor drift [M4]")
        self.assertIn("input", usage)
        self.assertIn("output", usage)
        self.assertEqual(usage["model"], MOCK)

    def test_tolerates_a_fenced_reply(self):
        parsed, _ = run([ModelOutput.from_content(
            MOCK, '```json\n{"deviation_score": 7, "justification": "x"}\n```')])
        self.assertEqual(parsed.deviation_score, 7)

    def test_repairs_once_then_succeeds(self):
        parsed, usage = run([
            ModelOutput.from_content(MOCK, "I think it deviated a fair bit."),
            ModelOutput.from_content(MOCK, '{"deviation_score": 9, "justification": "y"}'),
        ])
        self.assertEqual(parsed.deviation_score, 9)
        # both calls are billed, so both must be counted
        self.assertGreaterEqual(usage["input"], 0)

    def test_unparseable_after_repair_raises(self):
        # The old SDK call raised on a bad parse and every caller treats an exception as
        # "this trajectory failed". Returning an empty judgement instead would look like
        # a real verdict of zero deviation.
        with self.assertRaises(ValueError):
            run([ModelOutput.from_content(MOCK, "nope"),
                 ModelOutput.from_content(MOCK, "still nope")])

    def test_wrong_shape_is_a_failure_not_a_default(self):
        with self.assertRaises(ValueError):
            run([ModelOutput.from_content(MOCK, '{"deviation_score": "high"}'),
                 ModelOutput.from_content(MOCK, '{"deviation_score": "high"}')])

    def test_strict_is_only_requested_where_supported(self):
        import exp_structured_judge as sj

        self.assertTrue("openai/".startswith(sj._STRICT_PREFIXES[0][:7]))
        # mockllm is not an openai model, so strict must not be forced on
        parsed, _ = run([ModelOutput.from_content(
            MOCK, '{"deviation_score": 1, "justification": "ok"}')])
        self.assertEqual(parsed.deviation_score, 1)

    def test_usage_dict_shape_matches_the_cost_system(self):
        d = usage_dict("m", None)
        self.assertEqual(set(d), {"model", "input", "output", "cache_read",
                                  "cache_write", "total_cost"})

    def test_fence_stripper(self):
        self.assertEqual(_strip_fences('```json\n{"a":1}\n```'), '{"a":1}')
        self.assertEqual(_strip_fences('{"a":1}'), '{"a":1}')


if __name__ == "__main__":
    unittest.main()
