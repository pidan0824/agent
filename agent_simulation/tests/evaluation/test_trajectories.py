"""Outcome filtering preserves label isolation and each branch's causal transcript."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent_simulation.tests.ekb_fixture import rationale
from agent_simulation.agents.agentic import AgenticOptions
from agent_simulation.evaluation.micro.customer_agentic import FrozenSampleTools
from agent_simulation.evaluation.trajectories import (
    FrozenSellerSampleTools, OutcomeScorer, generate_trajectory, parser, score_group,
    summarize_group, training_rows,
)
from agent_simulation.tests.evaluation.test_customer_agentic import request


def envelope(action):
    return {"rationale": rationale(), "action": action}


def branch(identity, action, evidence, *, split="train"):
    prompt = [{"role": "system", "content": "available evidence only"}, {"role": "user", "content": "frozen state"}]
    return {"role": "customer", "actor_id": 7, "stage": "response", "initial_state_id": "frozen-1",
            "record_id": 11, "trajectory_id": identity, "source_split": split,
            "status": "ok", "final_action": action, "prompt": prompt, "tools": [],
            "messages": [*deepcopy(prompt), {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call-1", "content": evidence},
                {"role": "system", "content": "remaining budget"},
                {"role": "assistant", "content": json.dumps(envelope(action))}],
            "trace": {"tools": [{"name": "lookup"}]}}


class TrajectoryTests(unittest.TestCase):
    def test_generation_hides_target_and_retains_exact_tool_and_budget_context(self):
        observed = request()
        binding = FrozenSampleTools(observed, "query")
        calls = []

        class Model:
            seed, model = 42, "fixture"

            async def acomplete(self, value, *, call_id):
                calls.append(deepcopy(value))
                if len(calls) == 1:
                    return {"role": "assistant", "content": None, "tool_calls": [{
                        "id": "history-1", "type": "function", "function": {
                            "name": "query_customer_history",
                            "arguments": '{"before_time_index":10,"limit":1}'}}]}
                return {"role": "assistant", "content": json.dumps(envelope({"query": "杯子"})), "tool_calls": []}

        result = asyncio.run(generate_trajectory(
            binding, Model(), role="customer", actor_id=7, stage="query", record_id=11,
            source_split="train", catalog_identity="catalog", options=AgenticOptions(max_model_calls=3), sample_index=0))
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("SECRET TARGET", json.dumps(calls))
        self.assertNotIn("target", result)
        self.assertEqual(result["final_action"], {"query": "杯子"})
        self.assertEqual(result["messages"][:len(result["prompt"])], result["prompt"])
        self.assertEqual(result["messages"][:-1], calls[1]["messages"])
        self.assertEqual(sum(message["role"] == "tool" for message in result["messages"]), 1)
        self.assertEqual(result["tools"], calls[0]["tools"])

    def test_sft_and_dpo_keep_distinct_actual_tool_results_and_train_provenance(self):
        target = {"clicked_item_ids": [4], "purchased_item_ids": [4]}
        good = branch("good", target, "chosen-only evidence")
        bad = branch("bad", {"clicked_item_ids": [], "purchased_item_ids": []}, "rejected-only evidence")
        trajectories = [good, bad, {**deepcopy(good), "trajectory_id": "duplicate"}]
        score_group(trajectories, target, OutcomeScorer())
        sft, pairs, reason = training_rows(trajectories)
        self.assertEqual((len(sft), len(pairs), reason), (1, 1, "paired"))
        self.assertEqual(sft[0]["source_split"], "train")
        self.assertIn("chosen-only evidence", json.dumps(pairs[0]["chosen"]))
        self.assertNotIn("chosen-only evidence", json.dumps(pairs[0]["rejected"]))
        self.assertIn("rejected-only evidence", json.dumps(pairs[0]["rejected"]))
        self.assertEqual(pairs[0]["chosen"][0]["role"], "assistant")
        self.assertEqual(pairs[0]["chosen_score"], 1)
        self.assertEqual(pairs[0]["rejected_score"], 0)
        for row in trajectories:
            row["source_split"] = "test"
        self.assertEqual(training_rows(trajectories), ([], [], "held_out_no_training_export"))

    def test_ties_mixed_actors_and_all_failures_are_not_repaired_into_preferences(self):
        target = {"clicked_item_ids": [], "purchased_item_ids": []}
        good = branch("one", target, "same evidence")
        duplicate = {**deepcopy(good), "trajectory_id": "two"}
        score_group([good, duplicate], target, OutcomeScorer())
        self.assertEqual(training_rows([good, duplicate])[2], "tie_or_insufficient_score_gap")
        for key, changed in (("actor_id", 9), ("initial_state_id", "another"), ("stage", "query")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "identity"):
                training_rows([good, {**duplicate, key: changed}])
        failed = {**deepcopy(good), "status": "invalid", "error": "budget exhausted"}
        score_group([failed], target, OutcomeScorer())
        sft, pairs, reason = training_rows([failed])
        report = summarize_group([failed], sft, pairs, reason)
        self.assertTrue(report["all_failed"])
        self.assertTrue(report["no_accepted_trajectory"])
        self.assertEqual((sft, pairs), ([], []))
        self.assertIsNone(failed["score"])
        self.assertEqual(failed["target"], target)

    def test_outcome_scoring_uses_observed_actions_not_rationale(self):
        scorer = OutcomeScorer()
        self.assertEqual(scorer.score("query", {"query": "杯子"}, {"query": "杯子"})["score"], 1)
        self.assertEqual(scorer.score("seller", {"prices": [{"item_id": 1, "price_raw": 110}]},
                                      {"prices": [{"item_id": 1, "price_raw": 100}]})["score"], -0.1)
        metrics = scorer.score("response", {"clicked_item_ids": [1], "purchased_item_ids": []},
                               {"clicked_item_ids": [1, 2], "purchased_item_ids": []})
        self.assertAlmostEqual(metrics["score"], (2 / 3 + 1) / 2)

    def test_seller_binding_omits_label_and_rejects_future_tool_windows(self):
        sample = {"input": {"profile": {"seller_id": 7, "seller_name": "shop"},
                            "history_cutoff_exclusive": 10, "requested_price_item_ids": [4],
                            "owned_products": [{"item_id": 4, "last_observed_price_raw": 12}]},
                  "target": {"prices": [{"item_id": 4, "price_raw": 99999}]}}
        catalog = SimpleNamespace(connection=Mock(), config=object())
        history = [{"seller_id": 7, "item_id": 4, "time_index": 9, "price_status": "observed", "price_raw": 12}]
        with patch("agent_simulation.evaluation.trajectories.DataRepository") as repository, \
             patch("agent_simulation.evaluation.trajectories.rows", return_value=history):
            binding = FrozenSellerSampleTools(catalog, sample)
            self.assertNotIn("99999", json.dumps(binding.initial_context()))
            self.assertNotIn("target", vars(binding))
            with self.assertRaisesRegex(ValueError, "bounds"):
                binding.execute("query_business", {"before_time_index": 11, "limit": 1, "item_ids": [4]})
            repository.return_value.business.assert_not_called()
            binding.execute("query_business", {"before_time_index": 10, "limit": 1, "item_ids": [4]})
            repository.return_value.business.assert_called_once_with(
                7, before_time_index=10, limit=1, item_ids=[4], ranking_connection=catalog.connection)

    def test_training_defaults_use_official_train_and_stochastic_multi_sampling(self):
        args = parser().parse_args(["--dataset", "fixture", "--output", "output"])
        self.assertEqual(args.split, "train")
        self.assertGreater(args.samples_per_decision, 1)
        self.assertGreater(args.temperature, 0)


if __name__ == "__main__":
    unittest.main()
