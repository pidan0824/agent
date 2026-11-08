"""Test memory generation through the configured model interface."""

import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from agent_simulation.agents.memory.reflection import SummaryGenerator
from agent_simulation.inference.client import ModelOutputError
from agent_simulation.agents.memory.validation import ReflectionDraftError, read_draft
from agent_simulation.agents.memory.prompts import SUMMARY_INSTRUCTION
from agent_simulation.agents.memory.reflection import SUMMARY_PROMPTS


def seller_payload():
    def event(source, origin, time, item, price, exposure, click, purchase):
        return {
            "source_id": source,
            "origin": origin,
            "time_index" if origin == "observed" else "settled_step": time,
            "content": [
                {
                    "item_id": item,
                    "price_raw": price,
                    "price_status": "unknown" if price is None else "observed",
                    "exposure_records": exposure,
                    "click_records": click,
                    "purchase_records": purchase,
                }
            ],
        }

    return {
        "actor_role": "seller",
        "profile": {
            "seller_id": 1,
            "products": [
                {
                    "item_id": 10,
                    "item_title": "商品甲",
                    "reference_price_raw": 999,
                    "reference_price_time_index": 10000,
                },
                {"item_id": 20, "item_title": "商品乙"},
            ],
        },
        "experiences": [
            event("new", "observed", 50, 10, 80, 2, 1, 1),
            event("other", "observed", 40, 20, None, 3, 0, 0),
            event("old", "observed", 10, 10, 90, 1, 0, 0),
            event("sim", "simulated", 1, 10, 70, 1, 1, 0),
        ],
        "new_evidence_ids": ["new", "sim"],
        "previous_long_term_memory": [],
    }


class SummaryGeneratorTests(unittest.TestCase):
    def model(self, reply):
        self.calls = []

        def complete(request, *, call_id):
            self.calls.append((request, call_id))
            return reply

        return SimpleNamespace(tokenizer=object(), max_new_tokens=2048, complete=complete)

    def messages(self):
        return [
            {"role": "system", "content": "Role instructions"},
            {"role": "user", "content": json.dumps(seller_payload())},
        ]

    def prepare(self, messages, tools, tokenizer, **kwargs):
        return {"messages": messages, "tools": tools}, 10

    def test_summary_and_original_evidence_are_preserved(self):
        raw = {"action": "add",
            "content": "商品甲的所选观测价格下降；这不证明降价造成购买。",
            "source_ids": ["old", "new"],
        }
        model = self.model({"role": "assistant", "content": json.dumps({"action":"add", "content":"May prefer this product; evidence is limited.", "evidence_refs":["E3","E1"]})})
        generate = SummaryGenerator(model)
        messages = self.messages()
        with patch(
            "agent_simulation.agents.memory.reflection.prepare_request",
            side_effect=self.prepare,
        ):
            result = generate(messages)
        self.assertEqual(result["source_ids"], raw["source_ids"])
        self.assertEqual(result["content"], "May prefer this product; evidence is limited.")
        self.assertNotIn("这不证明", result["content"])
        self.assertEqual(json.loads(messages[1]["content"]), seller_payload())
        request = self.calls[0][0]
        self.assertEqual(len(json.loads(request["messages"][1]["content"])["evidence"]), 4)
        self.assertEqual(request["memory_generation_method"], "explicit-memory-operation-v2")
        self.assertIn("json_schema", request)
        self.assertNotIn("memory_fact_catalog", request)
        self.assertTrue(request["messages"][0]["content"].startswith("Role instructions"))
        self.assertTrue(self.calls[0][1].startswith("memory:"))

    def test_both_role_prompts_reach_the_model(self):
        for role in ("customer", "seller"):
            payload = seller_payload()
            if role == "customer":
                payload = {
                    "actor_role": role,
                    "profile": {"user_id": 7},
                    "experiences": [
                        {
                            "source_id": "customer-new",
                            "origin": "observed",
                            "time_index": 10,
                            "content": {
                                "query": "水杯",
                                "clicked_items": [],
                                "purchased_items": [],
                            },
                        }
                    ],
                    "new_evidence_ids": ["customer-new"],
                    "previous_long_term_memory": [],
                }
            messages = [
                {"role": "system", "content": SUMMARY_PROMPTS[role] + SUMMARY_INSTRUCTION},
                {"role": "user", "content": json.dumps(payload)},
            ]
            model = self.model({"role": "assistant", "content": '{"action":"no_update"}'})
            with patch(
                "agent_simulation.agents.memory.reflection.prepare_request",
                side_effect=self.prepare,
            ):
                result = SummaryGenerator(model)(messages)
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(result, {"action": "no_update", "content": "", "source_ids": []})
            self.assertEqual(self.calls[0][0]["messages"][0], messages[0])

    def test_invalid_schema_and_unavailable_sources_are_rejected(self):
        invalid = [
            {"content": "Wrong schema"},
            {
                "category": "pricing_and_sales",
                "fact_ids": ["F1"],
                "condition": "",
                "prediction": "",
                "alternative": "",
            },
            {"action": "add", "content": "Summary", "source_ids": ["missing"]},
            {"action": "add", "content": "Summary", "source_ids": ["new", "new"]},
            {"action": "add", "content": "Summary", "source_ids": [1]},
            {"action": "add", "content": None, "source_ids": []},
        ]
        for raw in invalid:
            model = self.model({"role": "assistant", "content": json.dumps(raw)})
            with (
                patch(
                    "agent_simulation.agents.memory.reflection.prepare_request",
                    side_effect=self.prepare,
                ),
                self.assertRaises(ModelOutputError),
            ):
                read_draft(SummaryGenerator(model)(self.messages()), seller_payload()["experiences"], [])

    def test_overflow_never_calls_model(self):
        model = self.model({})
        with (
            patch(
                "agent_simulation.agents.memory.reflection.prepare_request",
                side_effect=ValueError("overflow"),
            ),
            self.assertRaisesRegex(ValueError, "overflow"),
        ):
            SummaryGenerator(model)(self.messages())
        self.assertEqual(self.calls, [])

    def test_invalid_tool_calls_field_rejects_an_otherwise_valid_summary(self):
        reply = {"role": "assistant", "content": '{"action":"no_update"}', "tool_calls": {}}
        with patch("agent_simulation.agents.memory.reflection.prepare_request", side_effect=self.prepare):
            with self.assertRaises(ModelOutputError) as caught:
                SummaryGenerator(self.model(reply))(self.messages())
        self.assertEqual(caught.exception.raw_output, reply)

    def test_parse_failure_preserves_raw_output_on_failure(self):
        raw = '{"content":'
        model = self.model({})

        def fail(*args, **kwargs):
            raise ModelOutputError("Incomplete JSON", raw_output=raw)

        model.complete = fail
        with patch(
            "agent_simulation.agents.memory.reflection.prepare_request",
            side_effect=self.prepare,
        ):
            with self.assertRaises(ModelOutputError) as caught:
                SummaryGenerator(model)(self.messages())
        self.assertEqual(caught.exception.raw_output, raw)

    def test_markdown_fences_and_truncated_outputs_are_rejected_without_rewriting(self):
        raw = (
            "```json\n"
            + json.dumps({"action": "add", "content": "Summary fixture", "source_ids": ["old", "new"]})
            + "\n```"
        )
        for finished, text in ((True, raw), (False, raw), (True, "Here is the answer: " + raw)):
            model = self.model({})

            def fail(*args, **kwargs):
                raise ModelOutputError("JSON envelope", raw_output=text, finished=finished)

            model.complete = fail
            with patch(
                "agent_simulation.agents.memory.reflection.prepare_request",
                side_effect=self.prepare,
            ):
                with self.assertRaises(ModelOutputError):
                    SummaryGenerator(model)(self.messages())
