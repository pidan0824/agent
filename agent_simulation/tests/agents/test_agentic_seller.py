import json
import unittest

from agent_simulation.agents.agentic import AgenticBudgetExceeded, FrozenSellerTools, validate_teacher_final
from agent_simulation.agents.actions import price_schema
from agent_simulation.agents.seller import SellerAgent
from agent_simulation.agents.context.seller_features import seller_behavior_features
from agent_simulation.tests.fixtures import MarketFixture, ScriptedModel, TokenizerFixture, answer, tool


def quote(item, price):
    return answer({"rationale": {"hypothesis": "reference-price anchoring",
        "evidence": ["The owned item has prior quotations"], "uncertainty": "No observed costs"},
        "action": {"prices": [{"item_id": item, "price_raw": price}]}})


class AgenticSellerTests(MarketFixture, unittest.TestCase):
    def external(self, replies):
        model = ScriptedModel(replies)
        model.tokenizer = TokenizerFixture()
        model.max_new_tokens = 512
        model.generator_id = "fixture-qwen32b"
        return model

    def test_one_policy_reads_own_history_and_commits_complete_review(self):
        external = self.external([tool("query_business", {
            "item_ids": [1], "before_time_index": 10, "limit": 3}), quote(1, 125), quote(3, 200)])
        baseline = ScriptedModel([])
        run = SellerAgent(self.runtime, baseline, TokenizerFixture(), agentic_model=external).run(10)
        self.assertFalse(baseline.requests)
        self.assertEqual(run["engine"], "seller-agentic-frozen-v1")
        self.assertEqual(self.runtime.snapshot()["active_prices"]["1"]["price_raw"], 125)
        self.assertEqual(len(run["agentic_traces"]), 2)
        first = run["agentic_traces"][0]
        self.assertEqual(first["role"], "seller")
        self.assertEqual(first["actor_id"], 10)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(first["tools"][0]["name"], "query_business")
        self.assertIn("rationale", first)
        # Later model calls retain their exact earlier conversation prefix.
        earlier = first["model_calls"][0]["messages"]
        self.assertEqual(first["model_calls"][1]["messages"][:len(earlier)], earlier)
        self.assertEqual(self.runtime.db.execute("SELECT count(*) FROM memory_evidence").fetchone()[0], 0)

    def test_frozen_queries_reject_writes_and_foreign_private_records(self):
        before = self.runtime.snapshot()
        bound = FrozenSellerTools(self.runtime, 10, before)
        with self.assertRaises(PermissionError):
            bound.execute("submit_prices", {"prices": []})
        with self.assertRaises(PermissionError):
            bound.execute("query_business", {"item_ids": [2], "before_time_index": 10, "limit": 3})
        self.assertEqual(self.runtime.snapshot(), before)

    def test_budget_failure_does_not_publish_partial_prices(self):
        before = self.runtime.snapshot()
        external = self.external([tool("query_shop_products", {"after_item_id": 0, "limit": 2})])
        with self.assertRaises(AgenticBudgetExceeded):
            SellerAgent(self.runtime, ScriptedModel([]), TokenizerFixture(), agentic_model=external,
                        agentic_options={"max_model_calls": 1}).run(10)
        self.assertEqual(self.runtime.snapshot(), before)

    def test_visible_rationale_required_and_duplicate_targets_rejected(self):
        action = {"prices": [{"item_id": 1, "price_raw": 100}]}
        with self.assertRaises(ValueError):
            validate_teacher_final(action, [price_schema([1])], trace={})
        value = json.loads(quote(1, 100)["content"])
        value["action"]["prices"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate price target"):
            validate_teacher_final(value, [price_schema([1, 3])], trace={})

    def test_features_exclude_target_time_without_claiming_strategy(self):
        records = self.runtime.data.records(1)
        features = seller_behavior_features(records, seller_id=10, cutoff=4)
        self.assertEqual(features["products"][0]["last_valid_price_raw"], 100)
        self.assertEqual(features["products"][0]["observations"], 1)
        self.assertIsNone(features["products"][0]["observed_quote_change_fraction"])


if __name__ == "__main__":
    unittest.main()
