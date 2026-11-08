"""Test explicit memory operations, evidence references and failed updates."""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock

import agent_simulation.agents.memory.reflection as memory
from agent_simulation.agents.memory.validation import quality_issues
from agent_simulation.tests.fixtures import MarketFixture
from agent_simulation.inference.client import ModelOutputError
from agent_simulation.agents.memory.validation import ReflectionDraftError


class ReflectionQualityTests(MarketFixture, unittest.TestCase):
    def test_direct_draft_requires_an_explicit_action(self):
        from agent_simulation.agents.memory.validation import read_draft
        with self.assertRaisesRegex(ReflectionDraftError, "schema"):
            read_draft({"content": "A possible preference.", "source_ids": ["new"]},
                       [{"source_id": "new"}], [])

    def state_and_profile(self):
        state = self.runtime.snapshot()
        state["historical_cutoff_exclusive"] = 50
        first = state["initial_seller_memory"]["10"][0]
        state["initial_seller_memory"]["10"] = [
            {**deepcopy(first), "time_index": i,
             "content": [{**deepcopy(first["content"][0]), "time_index": i}]}
            for i in range(1, 21)]
        profile = {"seller_id": 10, "products": [p for p in state["products"].values() if p["seller_id"] == 10]}
        return state, profile

    def update(self, state, profile, generate):
        return memory.update(state, "seller", 10, profile, generate, lambda t: [1., 0.], "test", "test")




    def test_invalid_summary_raises_once_without_consuming_history(self):
        state, profile = self.state_and_profile()
        before = deepcopy(state)
        generate = Mock(spec=lambda request: None, side_effect=lambda request: {"action": "add",
            "content": "An unsupported claim.",
            "source_ids": ["unavailable"],
        })
        with self.assertRaises(ValueError):
            self.update(state, profile, generate)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(state, before)

    def test_model_error_is_propagated_without_repair_or_retry(self):
        state, profile = self.state_and_profile()
        before = deepcopy(state)
        generate = Mock(spec=lambda request: None,
            side_effect=ModelOutputError("unfinished", raw_output="{"))
        with self.assertRaisesRegex(ReflectionDraftError, "unfinished") as caught:
            self.update(state, profile, generate)
        self.assertEqual(caught.exception.output, "{")
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(state, before)

    def test_device_failure_is_not_converted_to_a_rejected_summary(self):
        import torch

        state, profile = self.state_and_profile()
        before = deepcopy(state)
        generate = Mock(spec=lambda request: None, side_effect=torch.OutOfMemoryError("fixture device failure"))
        with self.assertRaises(torch.OutOfMemoryError):
            self.update(state, profile, generate)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(state, before)

    def test_generator_programming_errors_propagate_unchanged(self):
        for error in (TypeError("wrong internal argument"), ValueError("invalid configuration")):
            state, profile = self.state_and_profile()
            before = deepcopy(state)
            with self.assertRaises(type(error)) as caught:
                self.update(state, profile, Mock(side_effect=error))
            self.assertIs(caught.exception, error)
            self.assertEqual(state, before)

    def test_malformed_json_is_a_rejected_draft_and_preserves_raw_text(self):
        state, profile = self.state_and_profile()
        before = deepcopy(state)
        raw = '  {"source_ids":'
        with self.assertRaises(ReflectionDraftError) as caught:
            self.update(state, profile, lambda _: raw)
        self.assertEqual(caught.exception.output, raw)
        self.assertTrue(caught.exception.request)
        self.assertEqual(state, before)

    def test_no_update_consumes_batch_without_creating_a_memory(self):
        state, profile = self.state_and_profile()
        result = self.update(state, profile, lambda r: {"action": "no_update", "content": "", "source_ids": []})
        self.assertEqual(result["status"], "no_update")
        self.assertEqual(result["operation"], "no_update")
        self.assertFalse(state["long_term"])
        self.assertEqual(state["memory_processed"]["seller:10"], 10)

    def test_only_exact_add_duplicates_are_rejected(self):
        observation = "这件夹克在多次曝光中吸引了顾客点击，值得继续观察购买反馈。"
        old = observation + "降低报价可能带来更多购买。"
        previous = [{"content": old}]
        draft = {"action": "add", "content": f" {old} ", "source_ids": ["new"]}
        self.assertEqual(quality_issues(draft, ["new"], previous, action="add"),
                         ["Content exactly repeats an active memory."])
        self.assertEqual(quality_issues(draft, ["new"], previous, action="revise"), [])
        draft["content"] = observation + "最近几次报价与购买之间没有稳定关系。"
        for action in ("add", "revise"):
            with self.subTest(action=action):
                self.assertEqual(quality_issues(draft, ["new"], previous, action=action), [])

    def test_add_and_revise_require_a_new_source(self):
        draft = {"action": "add", "content": "A possible association.", "source_ids": ["older"]}
        for action in ("add", "revise"):
            with self.subTest(action=action):
                self.assertEqual(quality_issues(draft, ["new"], [], action=action),
                                 ["The update cites no new evidence."])



