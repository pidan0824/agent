from agent_simulation.tests.fixtures import ScriptedModel
from agent_simulation.agents.context.shopping import project_customer

from agent_simulation.agents.seller import SellerAgent
from agent_simulation.agents.prompts.customer import customer_decision_messages
"""Test catalog-backed samples and market integration."""

from agent_simulation.agents.prompts.customer import decode_input as decode_customer

import json
import unittest
from copy import deepcopy
from unittest.mock import patch
from agent_simulation.tests.data.fixtures import CatalogFixture
from agent_simulation.datasets.samples.customer import customer_sample
from agent_simulation.datasets.samples.seller import seller_sample
from agent_simulation.agents.context.shopping import context
from agent_simulation.agents.prompts.customer import messages
from agent_simulation.tests.seller_context_fixture import decision_context
from agent_simulation.agents.prompts.seller import decode_input
from agent_simulation.environment.market import MarketEnvironment
from agent_simulation.data.catalog import DataRepository


class IntegratedTests(CatalogFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.path = self.root / (self.id().split(".")[-1] + ".sqlite")
        self.runtime = MarketEnvironment.create(
            self.path,
            DataRepository(self.catalog.connection, self.catalog.config, 5),
            [1, 2, 3],
            {1: []},
            top_k=3,
            catalog=self.catalog,
        )
        self.addCleanup(self.runtime.close)

    def settle(self):
        self.runtime.execute("customer", 1, "search_products", {"query": "痘痘贴"})
        self.runtime.execute(
            "customer", 1, "submit_response", {"clicked_item_ids": [1], "purchased_item_ids": []}
        )

    def test_query_response_pair_and_runtime_same_real_history(self):
        q, r = customer_sample(self.catalog, 5)
        self.assertEqual(q["input"], self.runtime.inputs.customer(1))
        self.assertEqual(q["split"], r["split"])
        self.assertEqual(q["input"]["short_term_memory"], r["input"]["short_term_memory"])
        self.assertNotIn("query", q["input"])
        self.assertNotIn("is_clicked", json.dumps(r["input"]["displayed_items"]))
        self.assertEqual(decode_customer(q["messages"][1]["content"]), project_customer(q["input"]))
        self.assertEqual(r["messages"], messages(r["input"], r["target"], response=True))

    def test_background_is_prior_frozen_and_not_an_event(self):
        before = self.runtime.inputs.customer(1, observe=False)
        summary = before["behavioral_summary"]
        self.assertEqual(summary["observed_time_index"], 4)
        self.assertEqual(summary["recent_purchased_item_ids"], [2, 2])
        self.assertEqual(summary["statistics_30d"]["orders"], 2)
        self.assertEqual([p["item_id"] for p in summary["item_details"]], [1, 2])
        self.assertNotIn("price_raw", summary["item_details"][0])
        self.assertTrue(all(not e["purchased_items"] for e in before["short_term_memory"]["search_history"]))
        self.assertIsNone(self.catalog.customer_summary(1, 1))
        self.assertIsNone(self.catalog.customer_summary(999, 5))
        self.settle()
        after = self.runtime.inputs.customer(1, observe=False)
        self.assertEqual(after["behavioral_summary"], summary)
        self.assertGreater(len(after["short_term_memory"]["search_history"]),
                           len(before["short_term_memory"]["search_history"]))
        with self.assertRaisesRegex(ValueError, "Future Customer background"):
            context({"user_id": 1}, [], cutoff=4, history_embedder=None, behavioral_summary=summary)

    def test_stream_background_matches_direct_and_excludes_same_time(self):
        from agent_simulation.datasets.samples.customer import customer_records, CustomerHistory
        samples = list(customer_records(self.catalog, 0, 1, split="test", stage="response"))
        self.assertEqual(samples[0][1]["input"]["behavioral_summary"],
                         customer_sample(self.catalog, 5, stage="response")[1]["input"]["behavioral_summary"])
        state = CustomerHistory(1, "fixture")
        products = self.catalog.request(5)["response_input"]["displayed_items"]
        row = dict(session_id=100, user_id=1, time_index=8, split="train", assigned_split="train",
                   query="a", impressed_item_ids=[p["item_id"] for p in products],
                   clicked_item_ids=[], purchased_item_ids=[], displayed_products=products,
                   user_profile={"user_id": 1}, background=dict(time_index=8,
                       recently_clicked_item_ids=[], recently_purchased_item_ids=[1],
                       user_statistical_features={}, product_details=None))
        outputs = list(state.samples([row, {**row, "session_id": 101}], stage="query", render=False))
        self.assertTrue(all(q["input"]["behavioral_summary"] is None for q, _ in outputs))
        later = list(state.samples([{**row, "session_id": 102, "time_index": 9}], stage="query", render=False))
        self.assertEqual(later[0][0]["input"]["behavioral_summary"]["observed_time_index"], 8)

    def test_seller_background_reaches_prompt_without_public_counts(self):
        from agent_simulation.agents.context.pricing_evidence import project_seller
        sample = seller_sample(self.catalog, 10, 1, 5, history_embedder=None)
        visible, _ = project_seller(sample["input"], include_audit=False)
        encoded = json.dumps(visible)
        self.assertIn('"business_summary"', encoded)
        summary = sample["input"]["owned_products"][0]["business_summary"]
        self.assertEqual(summary["observed_time_index"], 4)
        self.assertEqual(summary["statistics_30d"], {"exposures": 1.0, "clicks": 0.0, "orders": 0.0})
        self.assertNotIn('"user_order_cnt_30d_hist"', encoded)
        runtime, _ = self.runtime.inputs.seller(10, retrieve=False, observe=False, include_market=False)
        self.assertEqual(runtime["owned_products"][0]["business_summary"], summary)

    def test_cached_observed_history_preserves_catalog_precedence_and_seed_corrections(self):
        state = self.runtime.snapshot()
        seed = state["customers"]["1"]["observed_history"]
        canonical = dict(seed[-1])
        seed[-1]["query"] = "seed version of a catalog session"
        seed.append({**canonical, "session_id": 99, "query": "additional seed session"})

        with patch.object(self.runtime.catalog, "customer_history",
                          wraps=self.runtime.catalog.customer_history) as read, \
             patch.object(self.runtime.catalog, "customer_summary",
                          wraps=self.runtime.catalog.customer_summary) as read_summary:
            initial = self.runtime.inputs.customer(1, state=state, observe=False)
            history = initial["short_term_memory"]["search_history"]
            self.assertEqual(history[-2]["query"], canonical["query"])
            self.assertEqual(history[-1]["query"], "additional seed session")
            self.assertEqual(
                self.runtime.inputs.customer(1, state=deepcopy(state), observe=False), initial,
            )
            self.assertEqual(read.call_count, 1)
            self.assertEqual(read_summary.call_count, 1)

            seed[-1]["query"] = "corrected seed session"
            updated = self.runtime.inputs.customer(1, state=state, observe=False)
            self.assertEqual(updated["short_term_memory"]["search_history"][-1]["query"],
                             "corrected seed session")

    def test_settled_history_exposes_only_decision_visible_product_fields(self):
        from agent_simulation.agents.context.shopping import model_product
        from agent_simulation.agents.memory.evidence import experiences
        from agent_simulation.agents.memory.evidence import eligible_evidence
        self.settle()
        state = self.runtime.snapshot()
        stored = state["customers"]["1"]["simulation_history"][-1]
        self.assertIn("price_provenance", stored["clicked_items"][0])
        expected = model_product(stored["clicked_items"][0])
        history = self.runtime.execute("customer", 1, "query_customer_history",
                                       {"before_time_index": 5, "limit": 10})
        recent = self.runtime.inputs.customer(1)["short_term_memory"]["search_history"][-1]
        experience = experiences(state, "customer", 1)[-1]["content"]
        for event in (recent, experience, history["simulation_history"][-1]):
            self.assertEqual(event["clicked_items"], [expected])
        query_evidence = {"owner": "customer:1", "kind": "query_customer_history", "event_count": 1,
                  "available_step": state["simulation_step"], "historical_cutoff_exclusive": 5,
                  "content": {"result": {"simulation_history": [stored]}}}
        entry, = eligible_evidence([query_evidence], "customer:1", 1, state)
        self.assertEqual(entry["content"]["result"]["simulation_history"][0]["clicked_items"], [expected])
        self.assertIn("price_provenance", stored["clicked_items"][0])

    def test_single_target_with_shop_history(self):
        s = seller_sample(self.catalog, 10, 1, 5)
        self.assertEqual(len(s["target"]["prices"]), 1)
        self.assertEqual(s["input"]["requested_price_item_ids"], [1])
        self.assertEqual({p["item_id"] for p in s["input"]["owned_products"]}, {1})
        self.assertNotIn("shop_summary", s["input"])
        self.assertEqual(s["input"]["public_product_pages"], [])
        self.assertEqual(decode_input(s["messages"][1]["content"]), decision_context(s["input"]))
        records = [r for e in s["input"]["short_term_memory"] for r in e["content"]]
        self.assertTrue(all(r["time_index"] < 5 for r in records))
        self.assertEqual({r["item_id"] for r in records}, {1})
        queried = self.runtime.execute(
            "seller", 10, "query_business", {"item_ids": [2], "before_time_index": 5, "limit": 10}
        )
        self.assertIsNone(next(r["price_raw"] for r in queried["series"] if r["time_index"] == 2))
        self.assertNotIn("coverage", s)
        with self.assertRaises(ValueError):
            seller_sample(self.catalog, 10, 2, 2)

    def test_default_shop_overview_and_query_match_sft_target_history(self):
        sample = seller_sample(self.catalog, 10, 1, 5)
        initial = next(self.runtime.inputs.seller_targets(10, state=self.runtime.snapshot()))
        historical_products = deepcopy(initial["owned_products"])
        for product in historical_products:
            product["market_position"].pop("simulation")
            for peer in product['market_position']['competitor_examples']['products']:
                peer.pop('simulation_price', None)
        self.assertEqual(historical_products, sample["input"]["owned_products"])
        self.assertTrue(initial["short_term_memory"])
        self.assertEqual(initial["requested_price_item_ids"], [1])
        queried = self.runtime.execute(
            "seller", 10, "query_business", {"item_ids": [1], "before_time_index": 5, "limit": 10}
        )
        self.assertEqual(
            queried["series"],
            [r for e in sample["input"]["short_term_memory"] for r in e["content"]],
        )
        expanded, _ = self.runtime.inputs.seller(10, detail_item_ids=[1])
        self.assertEqual(
            expanded["short_term_memory"],
            sample["input"]["short_term_memory"],
        )
        self.assertEqual(self.runtime.snapshot()["price_actions"], [])
        self.settle()
        updated, _ = self.runtime.inputs.seller(10)
        self.assertTrue(updated["short_term_memory"])
        self.assertEqual(updated["owned_products"][0]["simulation_click_records"], 1)
        self.assertEqual(updated["owned_products"][0]["simulation_points"], 1)
        self.assertEqual(updated["owned_products"][0]["observed_points"], 4)


    def test_catalog_history_not_limited_to_initialization(self):
        self.assertEqual(len(self.runtime.snapshot()["customers"]["1"]["observed_history"]), 4)
        found = self.runtime.execute(
            "customer", 1, "query_customer_history", {"before_time_index": 3, "limit": 10}
        )
        self.assertEqual([e["session_id"] for e in found["observed_history"]], [1, 2])
        self.settle()
        self.assertEqual(
            self.runtime.inputs.customer(1)["short_term_memory"]["search_history"][-1]["query"],
            "痘痘贴",
        )
        no_future = self.runtime.execute(
            "customer",
            1,
            "query_customer_history",
            {"before_time_index": 5, "limit": 10, "before_simulation_step": 1},
        )
        self.assertEqual(no_future["simulation_history"], [])

    def test_shared_rolling_window_does_not_remove_other_topics(self):
        past = [{"session_id": i, "time_index": i, "query": "unrelated"} for i in range(40)]
        result = context({}, past, [{"settled_step": 100, "query": "current"}], limit=30, cutoff=50, simulation_before=101)
        self.assertEqual(len(result["short_term_memory"]["search_history"]), 30)
        self.assertEqual(result["short_term_memory"]["search_history"][0]["session_id"], 11)

    def test_reports_aggregate_only_settled_feedback_without_consuming_it(self):
        call = lambda name, args: self.runtime.execute("seller", 10, name, args)
        self.assertEqual(call("read_business_report", {})["shop_exposed_requests"], 0)
        self.settle()
        self.settle()
        report = call("read_business_report", {})
        self.assertEqual(report["shop_exposed_requests"], 2)
        self.assertEqual(report["products"][0]["click_records"], 2)
        self.assertEqual(call("read_business_report", {}), report)
        with self.assertRaises(PermissionError):
            call("schedule_review", {"after_settled_requests": 1})

    def test_report_includes_shop_with_zero_exposure(self):
        shown = self.runtime.execute("customer", 1, "search_products", {"query": "公开"})
        self.assertEqual([p["item_id"] for p in shown["displayed_items"]], [3])
        self.runtime.execute(
            "customer", 1, "submit_response", {"clicked_item_ids": [], "purchased_item_ids": []}
        )
        report = self.runtime.execute("seller", 10, "read_business_report", {})
        self.assertNotIn("settled_requests",report)
        self.assertEqual(report["shop_exposed_requests"], 0)
        self.assertEqual(sum(p["exposure_records"] for p in report["products"]), 0)
        from agent_simulation.agents.memory.evidence import experiences, with_settled_offers
        state = self.runtime.snapshot()
        event = experiences(state, "customer", 1)[-1]
        enriched, = with_settled_offers([event], state["feedback"])
        self.assertEqual(enriched["displayed_offers"], [{
            "item_id": 3, "price_raw": shown["displayed_items"][0]["price_raw"],
            "click_records": 0, "purchase_records": 0,
        }])

    def test_memory_due_and_saved_prompt_content_restore(self):
        for _ in range(5):
            self.settle()
        self.assertFalse(self.runtime.long_term_memory.status("seller", 10)["generation_due"])
        self.settle()
        self.assertTrue(self.runtime.long_term_memory.status("seller", 10)["generation_due"])
        # Fixtures test wiring only; they are not real-model or semantic retrieval evidence.
        embed = lambda text: [1.0, 0.5]

        def generate(request):
            evidence = json.loads(request[1]["content"])["experiences"]
            self.assertEqual(len(evidence), 10)
            if role == "customer":
                settled = [e for e in evidence if e["origin"] == "simulated"]
                self.assertTrue(settled)
                for event in settled:
                    rows = [r for r in self.runtime.snapshot()["feedback"]
                            if r["request_id"] == event["content"]["request_id"]]
                    self.assertEqual(event["displayed_offers"], [
                        {k: r[k] for k in ("item_id", "price_raw", "click_records", "purchase_records")}
                        for r in rows
                    ])
                    self.assertTrue(any(not offer["purchase_records"] for offer in event["displayed_offers"]))
            return {"action": "add",
                "content": "Observed repeated clicks; their cause is unknown.",
                "source_ids": [evidence[0]["source_id"]],
            }

        for role, actor in [("seller", 10), ("customer", 1)]:
            self.runtime.long_term_memory.update(
                role,
                actor,
                generate=generate,
                embedder=embed,
                model_id="test-fixture",
                encoder_id="test-vector",
            )
            self.assertFalse(self.runtime.long_term_memory.status(role, actor)["generation_due"])
        seller, _ = self.runtime.inputs.seller(10)
        self.assertIn("repeated clicks", seller["long_term_memory"][0]["content"])
        self.assertTrue({"simulation_step", "content", "source_ids", "evidence_times", "memory_id"}
                        <= set(seller["long_term_memory"][0]))
        self.assertTrue(seller["long_term_memory"][0]["source_ids"])
        self.assertTrue(self.runtime.inputs.customer(1)["long_term_memory"])
        self.runtime.execute("customer", 1, "search_products", {"query": "痘痘贴"})
        self.assertIn(
            "repeated clicks", customer_decision_messages(self.runtime.inputs.customer(1, response=True), response=True)[1]["content"]
        )
        other, _ = self.runtime.inputs.seller(20)
        self.assertEqual(other["long_term_memory"], [])
        restored = MarketEnvironment(self.path, self.runtime.data)
        try:
            with self.assertRaisesRegex(ValueError, "embedding model"):
                restored.inputs.seller(10)
            restored.long_term_memory.bind_encoder(embed, "test-vector")
            self.assertEqual(
                restored.inputs.seller(10)[0]["long_term_memory"], seller["long_term_memory"]
            )
        finally:
            restored.close()

    def test_model_handoff_rejects_overflow_without_changing_memory(self):
        class TokenizerFixture:
            chat_template = "fixture"

            def apply_chat_template(self, *args, **kwargs):
                return [1] * 32768

        before = deepcopy(self.runtime.snapshot())
        with self.assertRaisesRegex(ValueError, "context overflow"):
            list(SellerAgent(self.runtime, ScriptedModel([]), TokenizerFixture()).preview_requests(10))
        self.assertEqual(before, self.runtime.snapshot())

    def test_ten_event_batches_are_not_skipped(self):
        for _ in range(21):
            self.settle()
        kwargs = dict(
            generate=lambda _: dict(action="no_update", content="", source_ids=[]),
            embedder=lambda _: [1.0, 0.0],
            model_id="test-fixture",
            encoder_id="test-vector",
        )
        self.runtime.long_term_memory.update("seller", 10, **kwargs)
        self.assertEqual(self.runtime.long_term_memory.status("seller", 10)["processed_events"], 10)
        self.assertTrue(self.runtime.long_term_memory.status("seller", 10)["generation_due"])
        self.runtime.long_term_memory.update("seller", 10, **kwargs)
        self.assertEqual(self.runtime.long_term_memory.status("seller", 10)["processed_events"], 20)
        self.assertFalse(self.runtime.long_term_memory.status("seller", 10)["generation_due"])

    def test_long_term_chain_preserves_original_evidence(self):
        for _ in range(10):
            self.settle()

        def first(request):
            event = json.loads(request[1]["content"])["experiences"][0]
            return {"action": "add", "content": "An observed experience.", "source_ids": [event["source_id"]]}

        args = dict(
            embedder=lambda _: [1.0, 0.0], model_id="test-fixture", encoder_id="test-vector"
        )
        original = self.runtime.long_term_memory.update("seller", 10, generate=first, **args)["saved"]
        for _ in range(10):
            self.settle()

        def second(request):
            payload = json.loads(request[1]["content"])
            prior = payload["previous_long_term_memory"][0]
            return {"action": "add",
                "content": "The new observation revises the earlier hypothesis.",
                "source_ids": [*prior["source_ids"], payload["new_evidence_ids"][0]],
            }

        saved = self.runtime.long_term_memory.update("seller", 10, generate=second, **args)["saved"]
        self.assertTrue(set(original["source_ids"]) < set(saved["source_ids"]))
        self.assertEqual(saved["parent_memory_event_counts"], [10])
        with self.assertRaisesRegex(ValueError, "model differs"):
            self.runtime.long_term_memory.bind_encoder(lambda _: [1.0, 0.0], "different-encoder")

    def test_query_owned_product_outside_active_market(self):
        path = self.root / "smaller_market.sqlite"
        runtime = MarketEnvironment.create(path, self.runtime.data, [1, 3], {1: []}, catalog=self.catalog)
        try:
            data = runtime.execute(
                "seller",
                10,
                "query_business",
                {"item_ids": [2], "before_time_index": 5, "limit": 10},
            )
            self.assertEqual({r["item_id"] for r in data["series"]}, {2})
            with self.assertRaises(PermissionError):
                runtime.execute(
                    "seller",
                    10,
                    "query_business",
                    {"item_ids": [3], "before_time_index": 5, "limit": 10},
                )
        finally:
            runtime.close()

    def test_failed_summary_never_overwrites_state(self):
        for _ in range(10):
            self.settle()
        before = deepcopy(self.runtime.snapshot())
        with self.assertRaisesRegex(ValueError, "unavailable evidence"):
            self.runtime.long_term_memory.update(
                "seller",
                10,
                generate=lambda _: dict(action="add", content="invented", source_ids=["foreign"]),
                embedder=lambda _: [1.0, 0.0],
                model_id="test-fixture",
                encoder_id="test-vector",
            )
        self.assertEqual(before, self.runtime.snapshot())
        self.assertIn(
            "unavailable evidence",
            self.runtime.db.execute(
                "SELECT error FROM calls WHERE name='generate_long_term'"
            ).fetchone()[0],
        )

    def test_empty_summary_is_legal(self):
        for _ in range(10):
            self.settle()
        self.runtime.long_term_memory.update(
            "seller",
            10,
            generate=lambda _: dict(action="no_update", content="", source_ids=[]),
            embedder=lambda _: [1.0, 0.0],
            model_id="test-fixture",
            encoder_id="test-vector",
        )
        self.assertEqual(self.runtime.inputs.seller(10)[0]["long_term_memory"], [])
        self.assertFalse(self.runtime.long_term_memory.status("seller", 10)["generation_due"])
