"""Test memory lineage, triggering and persistence."""

from agent_simulation.agents.retrieval.long_term import retrieve_long_term_memory
from agent_simulation.agents.prompts.customer import decode_input as decode_customer

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import agent_simulation.agents.memory.reflection as runtime_memory
import agent_simulation.agents.retrieval.bge_encoder as memory_retrieval
from agent_simulation.tests.fixtures import MarketFixture
from agent_simulation.tests.fixtures import ProfileCatalogFixture, ScriptedModel, TokenizerFixture, answer
from agent_simulation.tests.fixtures import AgentFixture
from agent_simulation.environment.market import MarketEnvironment
from agent_simulation.agents.memory.reflection import SummaryGenerator


def summarize(request):
    evidence = json.loads(request[1]["content"])["experiences"]
    return {"action": "add", "content": "Test summary, not Qwen output.", "source_ids": [evidence[0]["source_id"]]}


class RuntimeMemoryTests(MarketFixture, unittest.TestCase):

    def test_unrelated_old_facts_are_optional(self):
        from agent_simulation.agents.memory.reflection import retrieve_prior_events

        encoder = Mock(spec=["encode_query"], return_value=[0., 1.])
        encoder.encode_query.return_value = [1., 0.]
        self.assertEqual(retrieve_prior_events([], "customer", {}, "鞋", encoder), ([], {}))
        encoder.encode_query.assert_not_called()
        event = {"source_id": "old", "content": {"query": "饼干"}}
        self.assertEqual(retrieve_prior_events([event], "customer", {}, "鞋", encoder), ([], {}))

    def test_restored_role_summary_is_saved_and_retrieved_without_rewriting(self):
        # Explicit model fixtures verify the protocol, not Qwen summary accuracy.
        state = self.runtime.snapshot()
        state["historical_cutoff_exclusive"] = 50
        self.data.cutoff = 50
        first = state["initial_seller_memory"]["10"][0]
        state["initial_seller_memory"]["10"] = [
            {**deepcopy(first), "time_index": i} for i in range(1, 11)]
        first = state["customers"]["7"]["observed_history"][0]
        state["customers"]["7"]["observed_history"] = [
            {**deepcopy(first), "session_id": i, "time_index": i} for i in range(1, 11)]
        with self.runtime.db:
            self.runtime.db.execute("UPDATE state SET value=?", (json.dumps(state),))
        self.runtime.catalog = ProfileCatalogFixture(self.runtime)
        for role, actor in (("customer", 7), ("seller", 10)):
            original = runtime_memory.experiences(self.runtime.snapshot(), role, actor)
            content = f"{role} summary fixture: observed patterns and tentative interpretations."
            requests = []
            def complete(request, **kwargs):
                requests.append(deepcopy(request))
                payload = json.loads(request["messages"][1]["content"])
                return answer({"action":"add", "content":content, "evidence_refs":[e["ref"] for e in payload["evidence"] if e["new"]][:1]})
            model = SimpleNamespace(tokenizer=object(), max_new_tokens=100, complete=complete)
            with patch("agent_simulation.agents.memory.reflection.prepare_request",
                       side_effect=lambda messages, *a, **kw: ({"messages": messages}, 10)):
                result = self.runtime.long_term_memory.update(
                    role, actor, generate=SummaryGenerator(model), model_id="fixture",
                    embedder=lambda text: [1., 0.], encoder_id="fixture")
            saved = self.runtime.snapshot()
            self.assertEqual(result["status"], "saved")
            audit, = self.runtime.db.execute(
                "SELECT result FROM calls WHERE rowid=?", (result["audit_id"],)
            ).fetchone()
            audit = json.loads(audit)
            self.assertEqual(json.loads(audit["request"][1]["content"])["actor_role"], role)
            self.assertNotIn("attempts", audit)
            content = result["saved"]["content"]
            self.assertIn("summary fixture", content)
            draft = audit["raw_output"]
            self.assertEqual(draft["content"], content)
            self.assertEqual(draft["action"], "add")
            self.assertEqual(result["operation"], "add")
            self.assertNotIn("action", result["saved"])
            self.assertEqual(result["saved"]["content"], content)
            self.assertEqual(saved["long_term"][f"{role}:{actor}"][0]["content"], content)
            visible = self.runtime.execute(role, actor, "query_long_term_memory", {"query": "history"})
            self.assertEqual(self.runtime.snapshot(), saved)
            self.assertEqual(visible[0]["content"], content)
            self.assertEqual(visible[0]["owner"], f"{role}:{actor}")
            self.assertEqual(visible[0]["evidence_kind"], "model_summary_or_hypothesis")
            self.assertEqual(result["saved"]["content_format"], "role_experience_summary_v1")
            self.assertTrue(visible[0]["evidence_times"])
            self.assertEqual(runtime_memory.experiences(saved, role, actor), original)
            supplied = json.loads(requests[0]["messages"][1]["content"])
            self.assertTrue(all(e in original for e in json.loads(audit["request"][1]["content"])["experiences"]))
            self.assertIn("profile", supplied)
            self.assertIn("previous_memory", supplied)

    def test_seller_reflection_profile_contains_only_selected_evidence_products(self):
        profile = {"seller_id": 10, "products": [
            {"item_id": i, "item_title": f"Product {i}"} for i in range(1000)
        ]}
        selected = [{"content": [{"item_id": 2}, {"item_id": 5}]},
                    {"content": [{"item_id": 2}]}]
        original = deepcopy(profile)
        result = runtime_memory.reflection_profile("seller", profile, selected)
        self.assertEqual(result["shop_product_count"], 1000)
        self.assertEqual(result["products"], [profile["products"][2], profile["products"][5]])
        self.assertEqual(profile, original)
        result["products"][0]["item_title"] = "Changed copy"
        self.assertEqual(profile, original)
        self.assertEqual(runtime_memory.reflection_profile("customer", profile, selected), profile)
        with self.assertRaisesRegex(ValueError, "description"):
            runtime_memory.reflection_profile("seller", {"products": []}, selected)

    def test_role_specific_reflection_reaches_generator_with_original_evidence(self):
        requests = {}
        for role, actor in (("customer", 7), ("seller", 10)):
            state = self.runtime.snapshot()
            state["historical_cutoff_exclusive"] = 30
            if role == "customer":
                first = state["customers"]["7"]["observed_history"][0]
                state["customers"]["7"]["observed_history"] = [
                    {**deepcopy(first), "session_id": i, "time_index": i}
                    for i in range(1, 22)
                ]
            else:
                first = state["initial_seller_memory"]["10"][0]
                state["initial_seller_memory"]["10"] = [
                    {**deepcopy(first), "time_index": i} for i in range(1, 22)
                ]
            original = runtime_memory.experiences(state, role, actor)

            def generate(request):
                requests[role] = deepcopy(request)
                return summarize(request)

            profile = {"id": actor, "products": [
                p for p in state["products"].values() if p["seller_id"] == actor
            ]}
            result = runtime_memory.update(
                state, role, actor, profile, generate,
                lambda text: [1.0, 0.0], "test-generator", "test-encoder",
            )
            payload = json.loads(requests[role][1]["content"])
            self.assertEqual(payload["actor_role"], role)
            self.assertEqual(payload["experiences"], original[:10])
            self.assertEqual(payload["new_evidence_ids"], [e["source_id"] for e in original[:10]])
            if role == "seller":
                selected_ids = {row["item_id"] for e in payload["experiences"] for row in e["content"]}
                self.assertEqual({p["item_id"] for p in payload["profile"]["products"]}, selected_ids)
            self.assertEqual(runtime_memory.experiences(state, role, actor), original)
            self.assertEqual(result["saved"]["owner"], f"{role}:{actor}")
            self.assertEqual(result["processed_events"], 10)
            memories = deepcopy(state["long_term"])

            def embed(text):
                return [1., 0.]

            embed.encode_query = Mock(side_effect=embed)
            embed.encode_many = Mock(side_effect=lambda texts: [embed(text) for text in texts])
            supplement = {
                "source_id": "context:fixture", "owner": f"{role}:{actor}", "kind": "query_displayed_products" if role == "customer" else "query_shop_products",
                "event_count": 10, "available_step": 0,
                "historical_cutoff_exclusive": 30, "content": {"result": {"products": []}},
            }
            second = runtime_memory.update(
                state, role, actor, profile, lambda _: {"action": "no_update", "content": "", "source_ids": []},
                embed, "test-generator", "test-encoder", [supplement],
            )
            packet = json.loads(second["request"][1]["content"])
            self.assertTrue(packet["previous_long_term_memory"])
            self.assertEqual(packet["supplemental_evidence"], [supplement])
            embed.encode_query.assert_called_once_with(second["retrieval_query"])
            self.assertEqual([len(call.args[0]) for call in embed.encode_many.call_args_list], [10, 1])
            self.assertEqual(packet["experiences"][:10], original[10:20])
            self.assertEqual(packet["new_evidence_ids"], [e["source_id"] for e in original[10:20]])
            self.assertEqual(packet["experiences"][10:], list(reversed(original[5:10])))
            self.assertEqual(second["status"], "no_update")
            self.assertEqual(state["memory_processed"][f"{role}:{actor}"], 20)
            self.assertEqual(state["long_term"], memories)
            self.assertEqual(runtime_memory.experiences(state, role, actor), original)
        customer = requests["customer"][0]["content"]
        seller = requests["seller"][0]["content"]
        self.assertNotEqual(customer, seller)

    def test_initial_sources_keep_ownership_cutoff_and_values(self):
        state = self.runtime.snapshot()
        seller = runtime_memory.experiences(state, "seller", 10)
        customer = runtime_memory.experiences(state, "customer", 7)
        self.assertEqual([e["time_index"] for e in seller], [2, 3, 4])
        self.assertEqual({r["item_id"] for e in seller for r in e["content"]}, {1, 3})
        self.assertEqual(customer[0]["content"]["query"], "水杯")
        self.assertEqual(runtime_memory.experiences(state, "customer", 8), [])
        public_seller = runtime_memory.experiences(state, "seller", 20)
        self.assertIsNone(public_seller[-1]["content"][0]["price_raw"])
        self.assertEqual(
            self.runtime.long_term_memory.status("seller", 10),
            {
                "initial_events": 3,
                "settled_events": 0,
                "total_events": 3,
                "processed_events": 0,
                "generation_due": False,
            },
        )

    def test_initial_history_can_generate_before_any_simulated_action(self):
        state = self.runtime.snapshot()
        state["historical_cutoff_exclusive"] = 20
        state["customers"]["7"]["observed_history"] = [
            {
                "session_id": i,
                "time_index": i,
                "query": f"fixture {i}",
                "clicked_items": [],
                "purchased_items": [],
            }
            for i in range(1, 13)
        ]
        original = deepcopy(state["customers"])
        result = runtime_memory.update(
            state,
            "customer",
            7,
            {"user_id": 7},
            summarize,
            lambda text: [1.0, 0.0],
            "test-generator",
            "test-encoder",
        )
        evidence = json.loads(result["request"][1]["content"])["experiences"]
        self.assertEqual([e["time_index"] for e in evidence], list(range(1, 11)))
        self.assertTrue(all(e["origin"] == "observed" and e["time_index"] <= 10 for e in evidence))
        self.assertEqual(result["saved"]["generated_step"], 0)
        self.assertEqual(result["processed_events"], 10)
        self.assertEqual(original, state["customers"])
        self.assertEqual(state["simulation_step"], 0)
        visible = retrieve_long_term_memory(
            state, "customer", 7, "shopping", lambda text: [1.0, 0.0], "test-encoder"
        )
        self.assertEqual(
            [{k: e[k] for k in ("simulation_step", "content")} for e in visible],
            [{"simulation_step": 0, "content": "Test summary, not Qwen output."}]
        )

    def test_rolling_window_does_not_delete_source_pool(self):
        initial = runtime_memory.experiences(self.runtime.snapshot(), "seller", 10)
        for _ in range(3):
            self.runtime.execute("customer", 7, "search_products", {"query": "保温杯"})
            self.runtime.execute(
                "customer",
                7,
                "submit_response",
                {"clicked_item_ids": [], "purchased_item_ids": [1]},
            )
        context, _ = self.runtime.inputs.seller(10, detail_item_ids=[1, 3])
        self.assertTrue(all("simulation_step" in e for e in context["short_term_memory"]))
        pool = runtime_memory.experiences(self.runtime.snapshot(), "seller", 10)
        self.assertEqual(pool[:3], initial)
        self.assertEqual([e["origin"] for e in pool], ["observed"] * 3 + ["simulated"] * 3)
        self.assertEqual(pool[-1]["content"][0]["purchase_records"], 1)
        self.assertEqual(pool[-1]["content"][0]["click_records"], 0)

    def test_default_encoder_is_connected_without_a_generation_model(self):
        # Retrieval cannot invent an experience when none has been generated.
        with patch.object(memory_retrieval, "default_encoder") as encoder:
            self.assertEqual(self.runtime.long_term_memory.context("seller", 10, "price"), [])
            encoder.assert_not_called()
        self.assertEqual(self.runtime.snapshot()["long_term"], {})

    def test_loop_uses_default_encoder_and_keeps_generated_memory_private(self):
        self.runtime.catalog = ProfileCatalogFixture(self.runtime)
        model = ScriptedModel(
            [
                answer({"query": "保温杯"}),
                answer({"clicked_item_ids": [], "purchased_item_ids": []}),
            ]
            * 10
        )
        with patch.object(
            memory_retrieval,
            "default_encoder",
            return_value=(lambda text: [1.0, 0.0], "test-default"),
        ) as encoder:
            loop = AgentFixture(
                self.runtime,
                model,
                TokenizerFixture(),
                summary_models={"generate": summarize, "model_id": "test-generator"},
            )
            for _ in range(10):
                loop.customer(7)
            encoder.assert_called_once()
        state = self.runtime.snapshot()
        self.assertEqual(
            set(state["long_term"]), {"customer:7", "seller:10", "seller:20", "seller:40"}
        )
        self.assertEqual(self.runtime.long_term_memory.context("customer", 8, "shopping"), [])
        query = decode_customer(model.requests[-2]["messages"][1]["content"])
        response = decode_customer(model.requests[-1]["messages"][1]["content"])
        self.assertTrue(query["long_term_memory"])
        self.assertEqual(query["long_term_memory"], response["long_term_memory"])
        restored = MarketEnvironment(self.path, self.data)
        try:
            restored.long_term_memory.bind_encoder(lambda text: [1.0, 0.0], "test-default")
            self.assertEqual(
                restored.long_term_memory.status("seller", 10), self.runtime.long_term_memory.status("seller", 10)
            )
            self.assertEqual(
                restored.long_term_memory.context("seller", 10, "price"),
                self.runtime.long_term_memory.context("seller", 10, "price"),
            )
        finally:
            restored.close()

    def test_source_pool_rejects_future_and_duplicate_evidence(self):
        state = self.runtime.snapshot()
        state["customers"]["7"]["observed_history"][0]["time_index"] = 10
        with self.assertRaisesRegex(ValueError, "Future initialization"):
            runtime_memory.experiences(state, "customer", 7)
        state = self.runtime.snapshot()
        state["customers"]["7"]["observed_history"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate memory"):
            runtime_memory.experiences(state, "customer", 7)

    def test_initial_batch_is_processed_before_first_query_when_generator_is_supplied(self):
        state = self.runtime.snapshot()
        original = state["customers"]["7"]["observed_history"][0]
        state["customers"]["7"]["observed_history"] = [
            {**original, "session_id": i} for i in range(100, 112)
        ]
        with self.runtime.db:
            self.runtime.db.execute("UPDATE state SET value=? WHERE id=1", (json.dumps(state),))
        self.runtime.catalog = ProfileCatalogFixture(self.runtime)
        model = ScriptedModel(
            [
                answer({"query": "保温杯"}),
                answer({"clicked_item_ids": [], "purchased_item_ids": []}),
            ]
        )
        tokenizer = TokenizerFixture()
        loop = AgentFixture(
            self.runtime,
            model,
            tokenizer,
            summary_models={
                "generate": summarize,
                "model_id": "test-summary",
                "embedder": lambda text: [1.0, 0.0],
                "encoder_id": "test-encoder",
            },
        )
        with patch.object(tokenizer, "encode", wraps=tokenizer.encode) as encode, \
                patch("agent_simulation.agents.base.perf_counter", side_effect=[100., 102.]):
            result = loop.customer(7)
        self.assertEqual(encode.call_count, 2)
        first = decode_customer(model.requests[0]["messages"][1]["content"])
        self.assertEqual(len(first["short_term_memory"]["search_history"]), 10)
        self.assertEqual(first["long_term_memory"][0]["simulation_step"], 0)
        self.assertNotIn("source_ids", first["long_term_memory"][0])
        audited, = result["context_provenance"][0]["memories"]
        self.assertEqual(audited["memory_id"], "customer:7:10")
        self.assertTrue(audited["source_ids"])
        self.assertEqual(result["memory_updates"][0]["result"]["processed_events"], 10)
        self.assertEqual(result["memory_updates"][0]["seconds"], 2.)
        self.assertEqual(self.runtime.long_term_memory.status("customer", 7)["total_events"], 13)

    def test_local_encoder_missing_files_fails_without_fallback(self):
        from agent_simulation.agents.retrieval.bge_encoder import bge_encoder

        with self.assertRaisesRegex(ValueError, "Incomplete local Chinese BGE"):
            bge_encoder(self.temp.name)


if __name__ == "__main__":
    unittest.main()
