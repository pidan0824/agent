"""Frozen evaluation hides targets and cannot mutate the observed request."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent_simulation.evaluation.micro.customer_agentic import (
    FrozenSampleTools, load_baseline, main, planning_usage, predict_request,
)
from agent_simulation.inference.client import ModelOutputError
from agent_simulation.inference.requests import ContextCapacityError


def request():
    context = {"profile": {"user_id": 7}, "history_cutoff_exclusive": 10,
               "short_term_memory": {"search_history": [
                   {"session_id": 1, "time_index": 1, "query": "old query",
                    "clicked_items": [], "purchased_items": []}]}, "long_term_memory": []}
    return {"session_id": 11, "assigned_split": "test", "query_input": context,
            "query_target": {"query": "SECRET TARGET"},
            "response_input": {**deepcopy(context), "query": "SECRET TARGET",
                               "displayed_items": [{"item_id": 4, "item_title": "Cup", "price_raw": 8}]},
            "response_target": {"clicked_item_ids": [4], "purchased_item_ids": [4]}}


class RecordingModel:
    def __init__(self, response):
        self.response, self.requests = response, []

    async def acomplete(self, value, *, call_id):
        self.requests.append(deepcopy(value))
        envelope = {"rationale": {"hypothesis": "fixture behavior", "evidence": ["prior query"],
                                  "uncertainty": "only a fixture"}, "action": self.response}
        return {"role": "assistant", "content": json.dumps(envelope), "tool_calls": []}


class FrozenCustomerEvaluationTests(unittest.TestCase):
    def test_infrastructure_and_programming_failures_are_not_model_predictions(self):
        class BrokenModel:
            async def acomplete(self, value, *, call_id):
                raise failure

        for failure in (RuntimeError("server down"), ValueError("broken config"),
                        PermissionError("denied"), TypeError("broken adapter")):
            with self.subTest(error=type(failure).__name__), self.assertRaises(type(failure)) as caught:
                asyncio.run(predict_request(request(), "query", BrokenModel()))
            self.assertIs(caught.exception, failure)
        for failure in (ModelOutputError("bad JSON"), ContextCapacityError("too long")):
            with self.subTest(error=type(failure).__name__):
                result = asyncio.run(predict_request(request(), "query", BrokenModel()))
                self.assertEqual(result["status"], "invalid")
                self.assertNotIn("prediction", result)
                self.assertEqual(result["trace"]["status"], "failed")

    def test_multiple_trajectories_do_not_silently_repeat_greedy_decoding(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "positive temperature"):
                main(["--dataset", directory, "--output", directory + "/results",
                      "--samples-per-decision", "2"])

    def test_planning_usage_counts_successful_theory_reads_separately(self):
        records = [{"trace": {"model_calls": [{}, {}], "tools": [
            {"name": "read_skill_reference", "arguments": {"name": "decision-types"}},
            {"name": "read_skill_reference", "arguments": {"name": "decision-types"}, "error": "disabled"},
            {"name": "record_behavior_hypothesis", "arguments": {}},
        ]}}, {"trace": {"model_calls": [{}], "tools": []}}]
        usage = planning_usage(records)
        self.assertEqual(usage["decisions"], 2)
        self.assertEqual(usage["decisions_with_tool_calls"], 1)
        self.assertEqual(usage["model_calls"], 3)
        self.assertEqual(usage["tool_calls"], 3)
        self.assertEqual(usage["failed_tool_calls"], 1)
        self.assertEqual(usage["theory_reference_reads"], 1)
        self.assertEqual(usage["recorded_behavior_hypotheses"], 1)

    def test_query_target_and_current_display_never_reach_model(self):
        observed = request()
        original = deepcopy(observed)
        model = RecordingModel({"query": "杯子"})
        result = asyncio.run(predict_request(observed, "query", model))
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("SECRET TARGET", json.dumps(model.requests))
        self.assertNotIn('"query_displayed_products"', json.dumps(model.requests))
        self.assertEqual(observed, original)

    def test_response_outcomes_hidden_and_invalid_ids_rejected(self):
        observed = request()
        model = RecordingModel({"clicked_item_ids": [999], "purchased_item_ids": []})
        result = asyncio.run(predict_request(observed, "response", model))
        self.assertEqual(result["status"], "invalid")
        visible = json.loads(next(message["content"] for message in model.requests[0]["messages"]
                                  if message["role"] == "user"))
        self.assertNotIn("target", visible)
        self.assertEqual(visible["query"], "SECRET TARGET")

    def test_queries_are_frozen_bounded_and_return_copies(self):
        observed = request()
        tools = FrozenSampleTools(observed, "query")
        first = tools.execute("query_customer_history", {"before_time_index": 10, "limit": 5})
        first["observed_history"][0]["query"] = "mutated"
        second = tools.execute("query_customer_history", {"before_time_index": 10, "limit": 5})
        self.assertEqual(second["observed_history"][0]["query"], "old query")
        with self.assertRaises(ValueError):
            tools.execute("query_customer_history", {"before_time_index": 11, "limit": 5})
        with self.assertRaises(PermissionError):
            tools.execute("submit_response", {"clicked_item_ids": [4], "purchased_item_ids": []})
        tools.guard()

    def test_baseline_response_targets_ignore_batch_order_but_reject_duplicates(self):
        expected = {"response:11": {"stage": "response", "target": {
            "clicked_item_ids": [4, 7], "purchased_item_ids": [7, 4]}}}
        row = {"sample_id": "response:11", "stage": "response", "target": {
            "clicked_item_ids": [7, 4], "purchased_item_ids": [4, 7]}}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(load_baseline(path, expected), [row])
            for malformed, message in (([7, 4, 4], "Duplicate"), ([7, 8], "targets differ")):
                row["target"]["clicked_item_ids"] = malformed
                path.write_text(json.dumps(row) + "\n")
                with self.assertRaisesRegex(ValueError, message):
                    load_baseline(path, expected)

    def test_baseline_queries_still_require_exact_text_and_coverage(self):
        expected = {"query:11": {"stage": "query", "target": {"query": "杯子"}}}
        row = {"sample_id": "query:11", "stage": "query", "target": {"query": "杯子 "}}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.jsonl"
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "targets differ"):
                load_baseline(path, expected)
            row["target"]["query"] = "杯子"
            for rows in ([], [row, row]):
                path.write_text("".join(json.dumps(value) + "\n" for value in rows))
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    load_baseline(path, expected)

    def test_bad_baseline_fails_before_model_creation_or_output(self):
        connection = Mock()
        connection.execute.return_value.fetchall.return_value = [(11,)]
        catalog = SimpleNamespace(connection=connection, request=Mock(return_value=request()))
        data = SimpleNamespace(catalog=catalog, close=Mock(), tokenizer=object())
        with TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.jsonl"
            baseline.write_text(json.dumps({"sample_id": "query:11", "stage": "query",
                                            "target": {"query": "different target"}}) + "\n")
            output = root / "new-output"
            with patch("agent_simulation.datasets.labels.SFTDataset", return_value=data), \
                 patch("agent_simulation.inference.agentic.AgenticClient") as client:
                with self.assertRaisesRegex(ValueError, "targets differ"):
                    main(["--dataset", str(root), "--output", str(output), "--request-id", "11",
                          "--stage", "query", "--baseline", str(baseline)])
                client.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
