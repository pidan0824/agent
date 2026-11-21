"""Bounded tool use preserves Customer isolation and never silently settles a failure."""

import asyncio
from copy import deepcopy
import json
import unittest

from agent_simulation.agents.agentic import (
    AgenticBudgetExceeded, AgenticOptions, FrozenCustomerTools, run_agentic_workflow,
)
from agent_simulation.agents.customer import CustomerAgent
from agent_simulation.agents.schemas import function
from agent_simulation.inference.client import ModelOutputError
from agent_simulation.tests.fixtures import (
    MarketFixture, ProfileCatalogFixture, ScriptedModel, TokenizerFixture, answer, tool,
)


def tool_reply(name="read", args=None, identifier="t1"):
    return tool(name, {} if args is None else args, identifier)({}, "unused")


class AgenticLoopTests(unittest.TestCase):
    def run_loop(self, replies, *, options=None, execute=None, tools=None, trace=None):
        replies = iter(replies)
        self.requests = []

        def complete(messages, schemas):
            self.requests.append((messages, schemas))
            return next(replies)

        return asyncio.run(run_agentic_workflow(
            messages=[{"role": "user", "content": "Choose."}],
            tools=tools or [function("read", "Read facts.", {}, [])],
            complete=complete, execute_tool=execute or (lambda *_: {"fact": 1}),
            validate_final=lambda value: value, options=options, trace=trace))

    def test_exact_tool_conversation_and_result_are_recorded(self):
        trace = {}
        self.assertEqual(self.run_loop([tool_reply(args="{}"), answer({"query": "杯子"})], trace=trace),
                         {"query": "杯子"})
        self.assertEqual(trace["status"], "complete")
        self.assertEqual(trace["tools"][0]["arguments"], {})
        self.assertEqual(trace["tools"][0]["result"], {"fact": 1})
        visible = next(m for m in reversed(self.requests[1][0]) if m["role"] == "tool")
        self.assertEqual(visible["tool_call_id"], "t1")
        self.assertEqual(json.loads(visible["content"]), {"fact": 1})
        for messages, _ in self.requests:
            budgets = [m for m in messages if str(m.get("content", "")).startswith("Runtime budget for this decision stage: ")]
            self.assertTrue(budgets)
            self.assertTrue(all(m["role"] == "user" for m in budgets))

    def test_last_call_requires_final_and_has_no_tools(self):
        trace = {}
        with self.assertRaises(AgenticBudgetExceeded):
            self.run_loop([tool_reply(), tool_reply(identifier="t2")],
                          options={"max_model_calls": 2}, trace=trace)
        self.assertEqual(self.requests[-1][1], self.requests[0][1])
        self.assertFalse(trace["model_calls"][-1]["tools_enabled"])
        self.assertEqual(len(trace["tools"]), 1)
        self.assertEqual(trace["status"], "failed")

    def test_excess_parallel_calls_execute_nothing(self):
        response = tool_reply()
        response["tool_calls"] += tool_reply(identifier="t2")["tool_calls"]
        touched = []
        with self.assertRaises(AgenticBudgetExceeded):
            self.run_loop([response], options={"max_tools_per_call": 1},
                          execute=lambda *args: touched.append(args))
        self.assertFalse(touched)

    def test_unknown_write_tool_is_never_dispatched(self):
        touched, trace = [], {}
        self.run_loop([tool_reply("submit_response"), answer({"query": "杯子"})],
                      execute=lambda *args: touched.append(args), trace=trace)
        self.assertFalse(touched)
        self.assertIn("PermissionError", trace["tools"][0]["error"])

    def test_truncation_is_explicit_and_raw_result_is_retained(self):
        trace = {}
        self.run_loop([tool_reply(), answer({"query": "杯子"})],
                      execute=lambda *_: {"large": "水" * 1000},
                      options={"max_tool_output_chars": 160}, trace=trace)
        entry = trace["tools"][0]
        self.assertTrue(entry["truncated"])
        self.assertTrue(json.loads(entry["returned_content"])["truncated"])
        self.assertLessEqual(len(entry["returned_content"]), 160)
        self.assertEqual(len(entry["result"]["large"]), 1000)

    def test_reused_tool_call_id_fails_without_second_execution(self):
        trace = {}
        with self.assertRaises(ModelOutputError):
            self.run_loop([tool_reply(), tool_reply()], trace=trace)
        self.assertEqual(len(trace["tools"]), 1)

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            AgenticOptions(max_model_calls=0)

    def test_cancelled_generation_preserves_a_terminal_trace(self):
        trace = {}

        async def cancelled(*_):
            raise asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(run_agentic_workflow(
                messages=[{"role": "user", "content": "Choose."}], tools=[],
                complete=cancelled, execute_tool=lambda *_: None,
                validate_final=lambda value: value, trace=trace))
        self.assertEqual(trace["status"], "failed")
        self.assertIn("CancelledError", trace["error"])


class AgenticCustomerTests(MarketFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.runtime.catalog = ProfileCatalogFixture(self.runtime)
        self.runtime.inputs.history_embedder = None

    def external(self, replies):
        model = ScriptedModel(replies)
        model.tokenizer = TokenizerFixture()
        model.max_new_tokens = 512
        model.generator_id = "fixture-qwen32b"
        original = model.complete

        def complete(request, *, call_id):
            reply = original(request, call_id=call_id)
            if reply.get("content") and not reply.get("tool_calls"):
                value = json.loads(reply["content"])
                if "rationale" not in value:
                    reply["content"] = json.dumps({"rationale": {
                        "hypothesis": "routine or unknown", "evidence": ["Prior shopping observations"],
                        "uncertainty": "Limited observations"}, "action": value})
            return reply

        model.complete = complete
        return model

    def evidence_count(self):
        return self.runtime.db.execute("SELECT count(*) FROM memory_evidence").fetchone()[0]

    def test_frozen_tools_do_not_change_market_or_memory_evidence(self):
        state = self.runtime.snapshot()
        before = self.evidence_count()
        tools = FrozenCustomerTools(self.runtime, 7, state, "query")
        state["customers"]["7"]["observed_history"].clear()
        result = tools.execute("query_customer_history", {"before_time_index": 10, "limit": 2})
        self.assertEqual(len(result["observed_history"]), 1)
        self.assertEqual(self.evidence_count(), before)
        self.assertEqual(self.runtime.revision, 0)
        with self.assertRaises(PermissionError):
            tools.execute("search_products", {"query": "杯子"})
        with self.assertRaises(PermissionError):
            tools.execute("query_displayed_products", {"item_ids": [1]})
        with self.assertRaises(ValueError):
            tools.execute("query_customer_history", {"before_time_index": 11, "limit": 2})

    def test_one_search_then_same_agent_response_and_single_settlement(self):
        def respond(request, _):
            state = json.loads(request["messages"][-1]["content"])
            self.assertEqual(state["stage"], "response")
            self.assertTrue(any(m["role"] == "tool" for m in request["messages"]))
            ids = [p["item_id"] for p in state["context"]["displayed_items"]]
            return answer({"clicked_item_ids": ids, "purchased_item_ids": []})

        external = self.external([
            tool("query_customer_history", {"before_time_index": 10, "limit": 2}),
            answer({"query": "水杯"}), respond,
        ])
        baseline = ScriptedModel([])
        before = self.evidence_count()
        run = CustomerAgent(self.runtime, baseline, TokenizerFixture(), agentic_model=external).run(7)
        self.assertEqual(run["engine"], "customer-agentic-frozen-v1")
        self.assertFalse(baseline.requests)
        self.assertEqual([trace["stage"] for trace in run["agentic_traces"]], ["query", "response"])
        self.assertIn("rationale", run["agentic_traces"][0])
        self.assertEqual(run["agentic_traces"][0]["final_action"], {"query": "水杯"})
        self.assertEqual(len(run["result"]["clicked_item_ids"]), 4)  # No ranker top-three cap.
        self.assertEqual(len(self.runtime.snapshot()["customers"]["7"]["simulation_history"]), 1)
        self.assertEqual(self.evidence_count(), before)
        self.assertEqual([t["name"] for t in run["tools"]].count("search_products"), 1)
        self.assertEqual([t["name"] for t in run["tools"]].count("submit_response"), 1)

    def test_budget_failure_leaves_existing_display_unsettled(self):
        self.runtime.execute("customer", 7, "search_products", {"query": "水杯"})
        before = self.runtime.snapshot()
        external = self.external([tool("query_displayed_products", {"item_ids": [1]})])
        agent = CustomerAgent(self.runtime, ScriptedModel([]), TokenizerFixture(),
                              agentic_model=external, agentic_options={"max_model_calls": 1})
        with self.assertRaises(AgenticBudgetExceeded):
            agent.run(7)
        self.assertEqual(self.runtime.snapshot(), before)

    def test_preview_matches_first_agentic_request_without_writing_evidence(self):
        external = self.external([answer({"query": "水杯"})])
        agent = CustomerAgent(self.runtime, ScriptedModel([]), TokenizerFixture(), agentic_model=external)
        state, evidence = self.runtime.snapshot(), self.evidence_count()
        preview = next(agent.prepare_requests(7))
        self.assertEqual(self.runtime.snapshot(), state)
        self.assertEqual(self.evidence_count(), evidence)
        run = agent.prepare_decision(7)
        asyncio.run(agent.choose_query(run))
        self.assertEqual(preview["messages"], external.requests[0]["messages"])
        self.assertEqual(preview["tools"], external.requests[0]["tools"])

    def test_changed_market_rejects_the_final_proposal(self):
        self.runtime.execute("customer", 7, "search_products", {"query": "水杯"})

        def change_market(request, _):
            self.runtime.execute("seller", 10, "submit_prices", {"prices": [{"item_id": 1, "price_raw": 99}]})
            return answer({"clicked_item_ids": [], "purchased_item_ids": []})

        external = self.external([change_market])
        agent = CustomerAgent(self.runtime, ScriptedModel([]), TokenizerFixture(), agentic_model=external)
        with self.assertRaisesRegex(ValueError, "Market changed"):
            agent.run(7)
        self.assertIn("7", self.runtime.snapshot()["pending"])
        self.assertEqual(self.runtime.snapshot()["customers"]["7"]["simulation_history"], [])

    def test_stale_phase_snapshot_is_rejected_before_model_dispatch(self):
        external = self.external([answer({"query": "水杯"})])
        agent = CustomerAgent(self.runtime, ScriptedModel([]), TokenizerFixture(), agentic_model=external)
        run = agent.prepare_decision(7)
        stale = self.runtime.snapshot()
        self.runtime.execute("seller", 10, "submit_prices", {"prices": [{"item_id": 1, "price_raw": 99}]})
        with self.assertRaisesRegex(ValueError, "Market changed before"):
            asyncio.run(agent.choose_query(run, state=stale))
        self.assertEqual(external.requests, [])
        self.assertNotIn("search_query", run)

    def test_agentic_stage_uses_an_atomic_market_snapshot(self):
        from unittest.mock import patch

        external = self.external([answer({"query": "水杯"})])
        agent = CustomerAgent(self.runtime, ScriptedModel([]), TokenizerFixture(), agentic_model=external)
        run = agent.prepare_decision(7)
        # Snapshot and revision must not be acquired in two independently
        # scheduled reads; freeze returns the pair in one SQLite statement.
        with patch.object(self.runtime, "snapshot", side_effect=AssertionError("non-atomic read")):
            asyncio.run(agent.choose_query(run))
        self.assertEqual(run["search_query"], {"query": "水杯"})


if __name__ == "__main__":
    unittest.main()
