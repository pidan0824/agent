"""Test Tick scheduling, Agent actions and market settlement."""

import asyncio
import json
import unittest
from unittest.mock import patch

from agent_simulation.tests.fixtures import MarketFixture
from agent_simulation.tests.fixtures import ScriptedModel, TokenizerFixture, ProfileCatalogFixture, answer
from agent_simulation.agents.customer import CustomerAgent
from agent_simulation.agents.prompts.customer import decode_input
from agent_simulation.agents.seller import SellerAgent
from agent_simulation.simulation.episode import Episode

WAIT = answer({"action": "no_action"})
EXIT = answer({"action": "exit"})


class EpisodeTests(MarketFixture, unittest.TestCase):
    def test_customer_phases_share_state_without_mutating_it(self):
        response = answer({'clicked_item_ids': [], 'purchased_item_ids': []})
        episode = self.make_episode([WAIT] * 3, [answer({'query': '水杯'})] * 2 + [response] * 2, ticks=1)
        episode.concurrency = 2
        build = self.runtime.inputs.customer
        snapshots, contents = {}, {}

        def customer(actor, **kwargs):
            stage = 'response' if kwargs.get('response') else 'query'
            state = kwargs['state']
            self.assertIsNotNone(state)
            if stage not in snapshots:
                snapshots[stage] = state
                contents[stage] = json.dumps(state, sort_keys=True)
            self.assertIs(state, snapshots[stage])
            return build(actor, **kwargs)

        with patch.object(self.runtime.inputs, 'customer', side_effect=customer):
            episode.step()
        self.assertIsNot(snapshots['query'], snapshots['response'])
        self.assertFalse(snapshots['query']['pending'])
        self.assertEqual(set(snapshots['response']['pending']), {'7', '8'})
        for stage, state in snapshots.items():
            self.assertEqual(json.dumps(state, sort_keys=True), contents[stage])

    def test_malformed_tool_calls_cannot_become_a_seller_wait(self):
        self.make_episode([], [], ticks=1)
        for malformed in ({}, "", False, 0, None, [{}]):
            with self.subTest(malformed=malformed):
                self.seller.model = ScriptedModel([{**WAIT, "tool_calls": malformed}])
                with self.assertRaises(ValueError):
                    self.seller.propose(10)
        self.seller.model = ScriptedModel([{**WAIT, "tool_calls": []}])
        self.assertEqual(self.seller.propose(10)["result"], {"action": "no_action"})

    def test_rejected_quote_commit_marks_staged_sellers_failed(self):
        quote = answer({"prices": [{"item_id": 1, "price_raw": 777}]})
        episode = self.make_episode([quote, WAIT, WAIT], [], ticks=1)
        prices = self.runtime.snapshot()["active_prices"]
        with patch.object(self.runtime, "apply_seller_quotes", side_effect=ValueError("commit rejected")):
            with self.assertRaisesRegex(ValueError, "commit rejected"):
                episode.step()
        self.assertEqual(self.runtime.snapshot()["active_prices"], prices)
        self.assertEqual(self.runtime.snapshot()["clock"]["settled_phases"], [])
        logs = self.runtime.db.execute("SELECT status,error FROM agent_runs WHERE role='seller'").fetchall()
        self.assertEqual(len(logs), 3)
        self.assertTrue(all(status == "failed" and "commit rejected" in error for status, error in logs))

    def test_staged_seller_quote_cannot_overwrite_a_newer_price(self):
        quote = answer({"prices": [{"item_id": 1, "price_raw": 777}]})
        episode = self.make_episode([quote, WAIT, WAIT], [WAIT, WAIT], ticks=1)
        apply = self.runtime.apply_seller_quotes

        def change_then_commit(quotes, **kwargs):
            self.runtime.execute("seller", 10, "submit_prices", {"prices": [{"item_id": 1, "price_raw": 650}]})
            return apply(quotes, **kwargs)

        with patch.object(self.runtime, "apply_seller_quotes", side_effect=change_then_commit):
            with self.assertRaisesRegex(ValueError, "Market changed during Seller"):
                episode.step()
        self.assertEqual(self.runtime.snapshot()["active_prices"]["1"]["price_raw"], 650)

    def test_tick_finalization_preserves_actions_committed_before_its_transaction(self):
        episode = self.make_episode([WAIT] * 3, [WAIT] * 2, ticks=1)
        record = self.runtime.record_tick

        def commit_before_record(*args, **kwargs):
            self.runtime.execute("seller", 10, "submit_prices",
                                 {"prices": [{"item_id": 1, "price_raw": 650}]})
            return record(*args, **kwargs)

        with patch.object(self.runtime, "record_tick", side_effect=commit_before_record):
            trace = episode.step()

        state = self.runtime.snapshot()
        self.assertEqual(state["active_prices"]["1"]["price_raw"], 650)
        self.assertEqual(len(state["price_actions"]), 1)
        self.assertEqual(state["clock"]["status"], "complete")
        self.assertEqual(trace["status"], "complete")

    def test_nonfinal_tick_fails_if_a_search_is_still_pending(self):
        episode = self.make_episode([WAIT] * 3, [WAIT] * 2, ticks=2)
        settle = self.customer.settle_staged

        def settle_then_search(runs, **kwargs):
            settle(runs, **kwargs)
            self.runtime.execute("customer", 7, "search_products", {"query": "杯子"})

        with patch.object(self.customer, "settle_staged", side_effect=settle_then_search):
            with self.assertRaisesRegex(ValueError, "Cannot finish Tick with unsettled searches"):
                episode.step()
        state = self.runtime.snapshot()
        self.assertEqual((state["clock"]["tick"], state["clock"]["status"]), (0, "failed"))
        self.assertIn("7", state["pending"])
        trace = json.loads(self.runtime.db.execute("SELECT trace FROM ticks WHERE tick=0").fetchone()[0])
        self.assertEqual(trace["status"], "failed")

    def test_second_quote_from_same_seller_rejects_whole_batch(self):
        before = self.runtime.snapshot()
        for second_item in (1, 3):
            with self.subTest(second_item=second_item):
                with self.assertRaisesRegex(ValueError, "Duplicate Seller"):
                    self.runtime.apply_seller_quotes(iter([
                        (20, {"prices": [{"item_id": 2, "price_raw": 500}]}),
                        (10, {"prices": [{"item_id": 1, "price_raw": 910}]}),
                        (10, {"prices": [{"item_id": second_item, "price_raw": 920}]}),
                    ]))
                self.assertEqual(self.runtime.snapshot(), before)
                self.assertEqual(self.runtime.db.execute(
                    "SELECT count(*) FROM calls WHERE name='submit_prices'").fetchone()[0], 0)

    def test_reflection_error_records_failed_run_and_committed_settlements(self):
        query = answer({"query": "杯子"})
        response = answer({"clicked_item_ids": [1], "purchased_item_ids": [1]})
        episode = self.make_episode([WAIT] * 3, [query, query, response, response], ticks=1)
        error = TypeError("bad reflection argument")
        with patch.object(self.customer, "reflect_on_feedback", side_effect=error) as reflect:
            with self.assertRaises(TypeError) as caught:
                episode.step()
        self.assertIs(caught.exception, error)
        reflect.assert_called_once()
        state = self.runtime.snapshot()
        self.assertEqual(state["clock"]["status"], "failed")
        self.assertEqual(state["pending"], {})
        trace = json.loads(self.runtime.db.execute("SELECT trace FROM ticks").fetchone()[0])
        self.assertEqual(trace["settled_phases"], ["seller", "customer"])
        rows = self.runtime.db.execute(
            "SELECT actor,status,error,trace FROM agent_runs WHERE role='customer' ORDER BY actor"
        ).fetchall()
        self.assertEqual([(actor, status) for actor, status, _, _ in rows],
                         [(7, "failed"), (8, "settled")])
        self.assertIn("Post-settlement reflection: TypeError: bad reflection argument", rows[0][2])
        for actor, _, _, raw in rows:
            receipt = json.loads(raw)["settled_result"]
            history = state["customers"][str(actor)]["simulation_history"]
            self.assertEqual(len(history), 1)
            self.assertEqual(receipt["request_id"], history[0]["request_id"])
            self.assertEqual(receipt["settled_step"], history[0]["settled_step"])
            self.assertEqual(receipt["purchased_item_ids"], [1])

    def test_customer_messages_and_settlement_are_independent_of_concurrency(self):
        def replay(limit):
            case = EpisodeTests()
            case.setUp()
            try:
                prompts = {}
                def decide(request, call_id):
                    ctx = decode_input(request["messages"][1]["content"])
                    response = "displayed_items" in ctx
                    prompts[(ctx["profile"]["user_id"], response)] = request["messages"]
                    if not response:
                        return answer({"query": "水杯"})
                    return answer({"clicked_item_ids": [ctx["displayed_items"][0]["item_id"]],
                                   "purchased_item_ids": []})
                episode = case.make_episode([WAIT] * 3, [decide] * 4, ticks=1)
                episode.concurrency = limit
                episode.step()
                state = case.runtime.snapshot()
                return prompts, state["customers"], state["feedback"]
            finally:
                case.tearDown()
        self.assertEqual(replay(1), replay(2))

    def test_customers_overlap_in_each_stage_and_searches_observe_actor_order(self):
        episode = self.make_episode([WAIT] * 3, [], ticks=1)
        episode.concurrency = 2
        counts = {"query": 0, "response": 0}
        gates = {stage: asyncio.Event() for stage in counts}
        class CustomerFixture:
            async def acomplete(inner, request, *, call_id):
                self.assertEqual(request["agent_role"], "customer")
                context = decode_input(request["messages"][1]["content"])
                stage = "response" if "displayed_items" in context else "query"
                counts[stage] += 1
                if counts[stage] == 2:
                    gates[stage].set()
                await asyncio.wait_for(gates[stage].wait(), 2)
                state = self.runtime.snapshot()
                self.assertFalse(state["feedback"])
                if stage == "query":
                    self.assertFalse(state["pending"])
                    return answer({"query": "水杯"})
                self.assertLess(state["pending"]["7"]["request_id"], state["pending"]["8"]["request_id"])
                return answer({"clicked_item_ids": [], "purchased_item_ids": []})
        self.customer.model = CustomerFixture()
        episode.step()
        self.assertEqual(counts, {"query": 2, "response": 2})
        self.assertFalse(self.runtime.snapshot()["pending"])

    def test_stale_request_rolls_back_other_customers_in_the_same_settlement(self):
        self.make_episode([], [], ticks=1)
        response = {"clicked_item_ids": [], "purchased_item_ids": []}
        actors = (7, 8)
        displays = {actor: self.runtime.execute("customer", actor, "search_products", {"query": "杯子"})
                    for actor in actors}
        self.runtime.execute("customer", 8, "submit_response", response)
        self.runtime.execute("customer", 8, "search_products", {"query": "杯子"})
        runs = [self.customer.logs.start("customer", actor) for actor in actors]
        for run in runs:
            run.update(displayed_list=displays[run["actor"]], result={"status": "staged", "response": response})
        before = self.runtime.snapshot()
        with self.assertRaisesRegex(ValueError, "displayed request changed"):
            self.customer.settle_staged(runs)
        self.assertEqual(self.runtime.snapshot(), before)
        self.assertEqual(self.runtime.db.execute(
            "SELECT count(*) FROM agent_runs WHERE status='failed'").fetchone()[0], 2)

    def test_one_customer_cannot_respond_and_exit_in_the_same_settlement(self):
        self.make_episode([], [], ticks=1)
        self.runtime.execute("customer", 7, "search_products", {"query": "杯子"})
        before = self.runtime.snapshot()
        with self.assertRaisesRegex(ValueError, "Duplicate Customer"):
            self.runtime.settle_customers([
                (7, "submit_response", {"clicked_item_ids": [], "purchased_item_ids": []}),
                (7, "exit_customer", {}),
            ])
        self.assertEqual(self.runtime.snapshot(), before)

    def test_customer_settlement_accepts_a_single_pass_action_stream(self):
        self.make_episode([], [], ticks=1)
        receipts = self.runtime.settle_customers(iter([(7, "exit_customer", {})]))
        self.assertEqual([r["action"] for r in receipts], ["exit"])
        self.assertIsNotNone(self.runtime.snapshot()["customers"]["7"]["exit"])

    def test_empty_logical_window_inside_idle_span_has_no_intervals(self):
        self.make_episode([WAIT] * 9, [WAIT] * 6).run()
        result = self.runtime.execute("seller", 10, "query_business", {
            "item_ids": [1], "before_time_index": 10, "limit": 1,
            "start_logical_time": 15, "before_logical_time": 15})
        timeline, = result["simulation"]["activity_timeline"]
        self.assertEqual(timeline["intervals"], [])
        self.assertIsNone(timeline["next_before_logical_time"])

    def test_customer_settlement_rolls_back_earlier_exit_on_late_invalid_response(self):
        self.make_episode([], [], ticks=1)
        self.runtime.execute("customer", 8, "search_products", {"query": "杯子"})
        before = self.runtime.snapshot()
        with self.assertRaises(ValueError):
            self.runtime.settle_customers([
                (7, "exit_customer", {}),
                (8, "submit_response", {"clicked_item_ids": [9999], "purchased_item_ids": []}),
            ])
        self.assertEqual(self.runtime.snapshot(), before)
        self.assertEqual(self.runtime.db.execute(
            "SELECT count(*) FROM calls WHERE name='exit_customer'").fetchone()[0], 0)

    def test_out_of_order_quotes_publish_before_first_customer(self):
        episode = self.make_episode([], [WAIT, WAIT], ticks=1)
        episode.concurrency = 2
        initial = self.runtime.snapshot()
        finished = []
        quote_items = {}
        class Quotes:
            async def acomplete(inner, request, *, call_id):
                actor = json.loads(request["messages"][1]["content"])["seller"]["seller_id"]
                await asyncio.sleep(0.03 if actor == 10 else 0)
                self.assertEqual(self.runtime.snapshot()["active_prices"], initial["active_prices"])
                item = next(int(k) for k, v in initial["products"].items() if v["seller_id"] == actor)
                quote_items[item] = 900 + actor
                finished.append(actor)
                prices = [{"item_id": int(k), "price_raw": 900 + actor if int(k) == item else initial["active_prices"][k]["price_raw"]}
                          for k, v in initial["products"].items() if v["seller_id"] == actor]
                return answer({"prices": prices})
        self.seller.model = Quotes()
        def customer(request, call_id):
            state = self.runtime.snapshot()
            self.assertEqual(len(finished), 3)
            for item, price in quote_items.items():
                self.assertEqual(state["active_prices"][str(item)]["price_raw"], price)
            return WAIT
        self.customer.model = ScriptedModel([customer, customer])
        trace = episode.step()
        self.assertNotEqual(finished, sorted(finished))
        self.assertEqual([r["actor"] for r in trace["sellers"]], sorted(finished))

    def test_concurrent_quotes_share_prices_and_failure_publishes_nothing(self):
        episode = self.make_episode([], [], ticks=1)
        episode.concurrency = 2
        old_prices = self.runtime.snapshot()["active_prices"]
        active = peak = 0
        class Overlap:
            async def acomplete(inner, request, *, call_id):
                nonlocal active, peak
                active += 1
                peak = max(active, peak)
                try:
                    context = json.loads(request["messages"][1]["content"])
                    await asyncio.sleep(0.015)
                    self.assertEqual(self.runtime.snapshot()["active_prices"], old_prices)
                    if context["seller"]["seller_id"] == 10:
                        return answer({"prices": [{"item_id": 1, "price_raw": 901}, {"item_id": 3, "price_raw": 200}]})
                    raise TimeoutError("fixture deadline")
                finally:
                    active -= 1
        self.seller.model = Overlap()
        with self.assertRaisesRegex(TimeoutError, "fixture deadline"):
            episode.step()
        self.assertEqual(peak, 2)
        self.assertEqual(self.runtime.snapshot()["active_prices"], old_prices)
        self.assertEqual(self.runtime.snapshot()["clock"]["status"], "failed")
        self.assertFalse(self.customer_model.requests)

    def test_late_invalid_quote_rolls_back_entire_price_phase(self):
        before = self.runtime.snapshot()
        with self.assertRaises(PermissionError):
            self.runtime.apply_seller_quotes([
                (10, {"prices": [{"item_id": 1, "price_raw": 910}]}),
                (20, {"prices": [{"item_id": 1, "price_raw": 920}]}),
            ])
        self.assertEqual(self.runtime.snapshot(), before)
        self.assertEqual(self.runtime.db.execute(
            "SELECT count(*) FROM calls WHERE name='submit_prices'").fetchone()[0], 0)
    def test_customer_summary_failure_is_shared_and_can_recover_next_tick(self):
        state = self.runtime.snapshot()
        first = state['customers']['7']['observed_history'][0]
        state['customers']['7']['observed_history'] = [
            {**first, 'session_id': i + 1, 'time_index': i} for i in range(10)
        ]
        with self.runtime.db:
            self.runtime.db.execute('UPDATE state SET value=?', (json.dumps(state),))
        episode = self.make_episode([WAIT] * 6, [WAIT] * 4, ticks=2)
        self.runtime.inputs.history_embedder = None
        attempts = []

        def generate(messages):
            attempts.append(json.loads(messages[1]['content'])['new_evidence_ids'])
            if len(attempts) == 1:
                return {"action": "add", 'content': 'Unsupported draft', 'source_ids': ['missing']}
            return {"action": "no_update", 'content': '', 'source_ids': []}

        memory = {'generate': generate, 'model_id': 'fixture',
                  'embedder': lambda text: [1., 0.], 'encoder_id': 'fixture'}
        self.seller.summary_models = self.customer.summary_models = memory
        run = {'memory_updates': []}
        self.seller._update_long_term_memory(run, [('customer', 7)])
        self.customer._update_long_term_memory(run, [('customer', 7)])
        self.assertEqual(len(attempts), 1)
        self.assertTrue(self.runtime.long_term_memory.status('customer', 7)['generation_due'])
        episode.step()
        self.assertEqual(len(attempts), 1)
        self.customer._update_long_term_memory(run, [('customer', 7)])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], attempts[1])
        self.assertFalse(self.runtime.long_term_memory.status('customer', 7)['generation_due'])
        self.assertEqual(self.runtime.snapshot()['memory_processed']['customer:7'], 10)
        self.assertFalse(self.runtime.snapshot()['long_term'])

    def test_one_rejected_factual_batch_is_attempted_once_per_tick_across_agents(self):
        state = self.runtime.snapshot()
        first = state["initial_seller_memory"]["10"][0]
        state["initial_seller_memory"]["10"] = [
            {**first, "time_index": i, "content": [{**first["content"][0], "time_index": i}]}
            for i in range(10)
        ]
        with self.runtime.db:
            self.runtime.db.execute("UPDATE state SET value=?", (json.dumps(state),))
        response = answer({"clicked_item_ids": [], "purchased_item_ids": []})
        episode = self.make_episode(
            [WAIT] * 6,
            [answer({"query": "杯子"}), answer({"query": "杯子"}), response, response, WAIT, WAIT],
            ticks=2,
        )
        self.runtime.inputs.history_embedder = None
        attempts = []

        def invalid(messages):
            attempts.append(json.loads(messages[1]["content"])["new_evidence_ids"])
            return {"action": "add", "content": "Invalid draft", "source_ids": ["missing"]}

        memory = {"generate": invalid, "model_id": "fixture", "embedder": lambda text: [1., 0.], "encoder_id": "fixture"}
        self.seller.summary_models = self.customer.summary_models = memory
        episode.step()
        self.assertEqual(len(attempts), 1)
        self.assertEqual(len(self.runtime.snapshot()["feedback"]), 8)
        self.assertTrue(self.runtime.long_term_memory.status("seller", 10)["generation_due"])
        episode.step()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], attempts[1])
        self.assertEqual(self.runtime.snapshot()["clock"]["status"], "complete")
        self.assertNotIn("seller:10", self.runtime.snapshot()["memory_processed"])
    def test_rejected_post_settlement_summary_preserves_outcomes_and_completes_tick(self):
        state = self.runtime.snapshot()
        first = state["customers"]["7"]["observed_history"][0]
        state["customers"]["7"]["observed_history"] = [
            {**first, "session_id": i + 1, "time_index": i} for i in range(9)
        ]
        with self.runtime.db:
            self.runtime.db.execute("UPDATE state SET value=?", (json.dumps(state),))
        episode = self.make_episode(
            [WAIT] * 3,
            [answer({"query": "水杯"}), WAIT, answer({"clicked_item_ids": [1], "purchased_item_ids": [1]})],
            ticks=1,
        )
        self.runtime.inputs.history_embedder = None

        def invalid(request):
            return {"action": "add", "content": "Unsupported summary", "source_ids": ["unavailable"]}

        self.customer.summary_models = {
            "generate": invalid, "model_id": "test",
            "embedder": lambda text: [1., 0.], "encoder_id": "test",
        }
        trace = episode.step()
        final = self.runtime.snapshot()
        self.assertEqual(final["clock"]["status"], "complete")
        self.assertEqual(final["clock"]["tick"], 1)
        self.assertEqual(len(final["feedback"]), 4)
        self.assertEqual(sum(r["purchase_records"] for r in final["feedback"]), 1)
        self.assertEqual(final["pending"], {})
        self.assertFalse(final["long_term"])
        self.assertNotIn("customer:7", final["memory_processed"])
        failure = trace["customers"][0]["memory_updates"][0]["result"]
        self.assertEqual(failure["status"], "failed")
        self.assertGreaterEqual(trace["customers"][0]["memory_updates"][0]["seconds"], 0.)
        audit, = self.runtime.db.execute("SELECT result FROM calls WHERE rowid=?",
                                         (failure["audit_id"],)).fetchone()
        self.assertEqual(json.loads(audit)["raw_output"]["source_ids"], ["unavailable"])
        self.assertTrue(json.loads(audit)["request"])

    def test_invalid_reflection_is_recorded_without_failing_tick(self):
        episode = self.make_episode([WAIT] * 3, [WAIT] * 2, ticks=1)
        state = self.runtime.snapshot()
        first = state["initial_seller_memory"]["10"][0]
        state["initial_seller_memory"]["10"] = [
            {**first, "time_index": i, "content": [{**first["content"][0], "time_index": i}]}
            for i in range(10)]
        with self.runtime.db:
            self.runtime.db.execute("UPDATE state SET value=?", (json.dumps(state),))
        calls = []
        def invalid(request):
            calls.append(request)
            return {"action": "add", "content": "An unsupported claim.", "source_ids": ["unavailable"]}
        self.seller.summary_models = {"generate": invalid, "model_id": "test",
                                    "embedder": lambda t: [1., 0.], "encoder_id": "test"}
        episode.run()
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.runtime.snapshot()["clock"]["status"], "complete")
        trace = json.loads(self.runtime.db.execute("SELECT trace FROM agent_runs WHERE role='seller' AND actor=10").fetchone()[0])
        self.assertEqual(trace["memory_updates"][0]["result"]["status"], "failed")
        final = self.runtime.snapshot()
        self.assertEqual(final["simulation_step"], 0)
        self.assertFalse(final["long_term"])
        self.assertFalse(final["feedback"])
        self.assertEqual(final["clock"]["time"], final["clock"]["start"] + 5)

    def make_episode(self, sellers, customers, ticks=3):
        self.runtime.catalog = ProfileCatalogFixture(self.runtime)
        self.runtime.catalog.connection = None
        from agent_simulation.tests.fixtures import explicit_shop_model
        self.seller_model, self.customer_model = explicit_shop_model(self.runtime, sellers), ScriptedModel(customers)
        self.seller = SellerAgent(self.runtime, self.seller_model, TokenizerFixture(), price_batch_size=16)
        self.customer = CustomerAgent(self.runtime, self.customer_model, TokenizerFixture())
        return Episode(self.runtime, self.seller, self.customer, tick_size=5, max_ticks=ticks)

    def test_response_clock_includes_current_search_before_first_settlement(self):
        episode = self.make_episode([WAIT] * 6, [WAIT] * 2, ticks=2)
        episode.step()
        listing = self.runtime.execute("customer", 7, "search_products", {"query": "杯子"})
        messages = next(self.customer.preview_requests(7, response=True))["messages"]
        seen = decode_input(messages[1]["content"])
        self.assertEqual(seen["last_search_time"], listing["logical_time"])
        self.assertEqual(seen["no_new_search_interval"],
                         {"start_inclusive": 10, "end_exclusive": listing["logical_time"]})
        self.assertNotIn("purchased_item_ids", seen)

    def test_waits_advance_time_without_business_or_memory_events(self):
        episode = self.make_episode([WAIT] * 9, [WAIT] * 6)
        initial = self.runtime.snapshot()
        self.assertEqual(episode.run()["time"], 25)
        state = self.runtime.snapshot()
        self.assertEqual(state["clock"]["status"], "complete")
        self.assertEqual(state["clock"]["termination_reason"], "max_ticks")
        for key in ("feedback", "price_actions", "customers", "simulation_step", "memory_processed"):
            self.assertEqual(state[key], initial[key])
        self.assertEqual(len(self.seller_model.requests), 9)
        self.assertEqual(len(self.customer_model.requests), 6)
        overview, _ = self.runtime.inputs.seller(10)
        first = next(p for p in overview["owned_products"] if p["item_id"] == 1)
        self.assertEqual(first["activity_intervals"], [{"start_inclusive": 10, "end_exclusive": 25,
            "price_raw": 120, "exposure_records": 0, "click_records": 0,
            "purchase_records": 0, "estimated_units_sold": 0}])
        customer = self.runtime.inputs.customer(7)
        self.assertEqual(customer["no_new_search_interval"], {"start_inclusive": 10, "end_exclusive": 25})
        self.assertEqual(customer["short_term_memory"]["search_history"], initial["customers"]["7"]["observed_history"])
        with self.assertRaisesRegex(ValueError, "not ready"):
            episode.step()

    def test_terminal_episode_blocks_writes_but_reports_remain_available(self):
        episode = self.make_episode([WAIT] * 3, [WAIT] * 2, ticks=1)
        episode.run()
        state = self.runtime.snapshot()
        for name, role, actor, args in (
            ("submit_prices", "seller", 10, {"prices": [{"item_id": 1, "price_raw": 999}]}),
            ("search_products", "customer", 7, {"query": "水杯"}),
            ("exit_customer", "customer", 7, {}),
            ("submit_response", "customer", 7, {"clicked_item_ids": [], "purchased_item_ids": []}),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "closed"):
                self.runtime.execute(role, actor, name, args)
        with self.assertRaisesRegex(ValueError, "closed"):
            self.seller.propose(10)
        with self.assertRaisesRegex(ValueError, "closed"):
            self.customer.run(7)
        self.runtime.execute("seller", 10, "read_business_report", {})
        self.assertEqual(state, self.runtime.snapshot())

    def test_exit_is_separate_from_wait_and_an_empty_search_response(self):
        def leave(request, call_id):
            context = decode_input(request["messages"][1]["content"])
            self.assertIn("profile", context)
            self.assertIn("short_term_memory", context)
            self.assertIn('"action":"exit"', request["messages"][0]["content"])
            return EXIT

        def other_customer_search(request, call_id):
            # Exit is staged until every Customer has made this Tick's decision.
            self.assertNotIn("exit", self.runtime.snapshot()["customers"]["7"])
            return answer({"query": "水杯"})

        episode = self.make_episode([WAIT] * 9, [leave, other_customer_search,
            answer({"clicked_item_ids": [], "purchased_item_ids": []}), WAIT, EXIT], ticks=8)
        initial = self.runtime.snapshot()
        episode.step()
        state = self.runtime.snapshot()
        self.assertEqual(state["customers"]["7"]["exit"], {"tick": 0, "logical_time": 10})
        self.assertNotIn("exit", state["customers"]["8"])
        self.assertEqual(state["customers"]["7"]["simulation_history"], [])
        event = state["customers"]["8"]["simulation_history"][0]
        self.assertEqual(event["query"], "水杯")
        self.assertEqual(event["clicked_items"], [])
        self.assertEqual(event["purchased_items"], [])
        self.assertEqual(len(state["feedback"]), 4)
        trace = episode.step()
        self.assertEqual([run["actor"] for run in trace["customers"]], [8])
        context = self.runtime.inputs.customer(8)
        self.assertEqual(context["no_new_search_interval"], {"start_inclusive": 15, "end_exclusive": 20})
        self.assertEqual(context["short_term_memory"]["search_history"], [event])
        clock = episode.run()
        self.assertEqual((clock["tick"], clock["termination_reason"]), (3, "all_customers_exited"))
        state = self.runtime.snapshot()
        self.assertEqual(state["customers"]["8"]["exit"], {"tick": 2, "logical_time": 20})
        self.assertEqual(len(self.customer_model.requests), 5)
        self.assertEqual(len(self.seller_model.requests), 9)
        self.assertEqual(len(state["feedback"]), 4)
        self.assertEqual(state["pending"], {})
        for actor in ("7", "8"):
            self.assertEqual(state["customers"][actor]["observed_history"], initial["customers"][actor]["observed_history"])
        persisted = json.loads(self.runtime.db.execute("SELECT value FROM state WHERE id=1").fetchone()[0])
        self.assertEqual(persisted["customers"], state["customers"])
        for name, args in (("search_products", {"query": "水杯"}), ("exit_customer", {})):
            with self.assertRaisesRegex(ValueError, "already exited"):
                self.runtime.execute("customer", 7, name, args)
        with self.assertRaisesRegex(ValueError, "already exited"):
            self.customer.run(7)
        self.assertEqual(len(self.customer_model.requests), 5)

    def test_replay_tracks_only_completed_ticks_and_the_actual_display_order(self):
        from agent_simulation.simulation.replay import page
        from agent_simulation.simulation.replay import read_timeline
        episode = self.make_episode([WAIT] * 6, [EXIT, answer({"query": "水杯"}),
            answer({"clicked_item_ids": [], "purchased_item_ids": [1]}), WAIT], ticks=2)
        self.assertFalse(read_timeline(self.path)["frames"])
        episode.step()
        first = read_timeline(self.path)
        self.assertIsNone(read_timeline(self.path, first["stamp"]))
        leave, search = first["frames"][0]["customers"]
        self.assertEqual(leave["action"], "exit")
        self.assertEqual(search["query"], "水杯")
        listing = json.loads(self.runtime.db.execute(
            "SELECT result FROM calls WHERE name='search_products'").fetchone()[0])
        self.assertEqual([i["id"] for i in search["items"]], [i["item_id"] for i in listing["displayed_items"]])
        self.assertEqual(sum(i["buy"] for i in search["items"]), 1)
        self.assertFalse(any(i["click"] for i in search["items"]))
        episode.step()
        final = read_timeline(self.path, first["stamp"])
        self.assertEqual(final["frames"][1]["customers"], [{"id": 8, "action": "no_action"}])
        self.assertNotIn("observed_history", page(final))
        self.assertNotIn("</script><img", page({"query": "</script><img onerror=test>"}))

    def test_fixed_horizon_keeps_running_after_all_customers_exit(self):
        episode = self.make_episode([WAIT] * 9, [EXIT, EXIT], ticks=3)
        episode.fixed_horizon = True
        clock = episode.run()
        self.assertEqual((clock["tick"], clock["termination_reason"]), (3, "max_ticks"))
        self.assertEqual(len(self.customer_model.requests), 2)

    def test_exit_cannot_discard_an_unsettled_response(self):
        self.make_episode([], [])
        self.runtime.execute("customer", 7, "search_products", {"query": "水杯"})
        with self.assertRaisesRegex(ValueError, "Settle"):
            self.runtime.execute("customer", 7, "exit_customer", {})
        state = self.runtime.snapshot()
        self.assertIn("7", state["pending"])
        self.assertNotIn("exit", state["customers"]["7"])

    def test_exit_is_simulation_only_and_never_a_response_label(self):
        with self.assertRaises(PermissionError):
            self.runtime.execute("customer", 7, "exit_customer", {})
        episode = self.make_episode([WAIT] * 3, [answer({"query": "水杯"}), WAIT, EXIT])
        with self.assertRaises(ValueError):
            episode.step()
        state = self.runtime.snapshot()
        self.assertEqual(state["clock"]["status"], "failed")
        self.assertIn("7", state["pending"])
        self.assertNotIn("exit", state["customers"]["7"])
        self.assertEqual(state["feedback"], [])
        with self.assertRaises(PermissionError):
            self.runtime.execute("seller", 10, "exit_customer", {})

    def test_failed_tick_does_not_commit_a_staged_exit(self):
        episode = self.make_episode([WAIT] * 3, [EXIT, answer({"action": "invalid"})])
        with self.assertRaises(ValueError):
            episode.step()
        self.assertNotIn("exit", self.runtime.snapshot()["customers"]["7"])
        self.assertEqual(self.runtime.snapshot()["clock"]["status"], "failed")

    def test_direct_tick_customer_run_applies_exit_without_a_search(self):
        self.make_episode([], [EXIT])
        run = self.customer.run(7)
        self.assertEqual(run["result"], {"action": "exit"})
        self.assertEqual(run["applied_result"]["logical_time"], 10)
        self.assertEqual(self.runtime.snapshot()["pending"], {})
        self.assertEqual(self.runtime.snapshot()["feedback"], [])

    def test_phase_barriers_and_next_tick_context(self):
        def second_seller(request, call_id):
            self.assertEqual(self.runtime.snapshot()["active_prices"]["1"]["price_raw"], 120)
            return WAIT

        seen_customer_contexts = []
        def search(request, call_id):
            state = self.runtime.snapshot()
            self.assertEqual(state["feedback"], [])
            self.assertEqual(state["active_prices"]["1"]["price_raw"], 110)
            seen_customer_contexts.append(decode_input(request["messages"][1]["content"]))
            return answer({"query": "水杯"})

        episode = self.make_episode(
            [answer({"prices": [{"item_id": 1, "price_raw": 110}]}), second_seller, WAIT]
            + [WAIT] * 3,
            [search, search, answer({"clicked_item_ids": [], "purchased_item_ids": [1]}),
             answer({"clicked_item_ids": [1], "purchased_item_ids": []})] + [WAIT] * 2, ticks=2)
        episode.step()
        state = self.runtime.snapshot()
        self.assertEqual(state["clock"]["time"], 15)
        self.assertEqual(len(state["feedback"]), 8)
        self.assertEqual(state["pending"], {})
        self.assertEqual(state["price_actions"][0]["logical_time"], 10)
        event = state["customers"]["7"]["simulation_history"][0]
        self.assertEqual((event["search_time"], event["settled_time"]), (10, 10))
        self.assertEqual(event["purchased_items"][0]["price_raw"], 110)
        self.assertEqual(seen_customer_contexts[1]["short_term_memory"]["search_history"], [])
        context, _ = self.runtime.inputs.seller(10, detail_item_ids=[1])
        self.assertEqual(context["business_report"]["shop_totals"]["exposure_records"], 4)
        self.assertEqual(context["business_report"]["shop_totals"]["purchase_records"], 1)
        rows = [r for e in context["short_term_memory"] for r in e["content"] if "settled_time" in r]
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["settled_time"] for r in rows}, {10})
        # Reading extra reports does not advance the decision's feedback cursor.
        a = self.runtime.execute("seller", 10, "read_business_report", {})
        b = self.runtime.execute("seller", 10, "read_business_report", {})
        self.assertEqual(a, b)
        c = self.runtime.inputs.customer(7)
        self.assertEqual(c["short_term_memory"]["search_history"][-1]["query"], "水杯")
        episode.step()
        context, _ = self.runtime.inputs.seller(10)
        self.assertEqual(context["business_report"]["shop_totals"]["exposure_records"], 0)
        self.assertEqual(context["business_report"]["shop_cumulative_totals"]["exposure_records"], 4)
        self.assertEqual(self.runtime.snapshot()["historical_cutoff_exclusive"], 10)
        self.assertNotIn(999, [r["price_raw"] for e in context["short_term_memory"] for r in e["content"]])
        query = {"item_ids": [1], "before_time_index": 10, "limit": 1}
        recent = self.runtime.execute("seller", 10, "query_business", query)["simulation"]["activity_timeline"][0]
        self.assertTrue(recent["has_more"])
        self.assertEqual(recent["next_before_logical_time"], 15)
        older = self.runtime.execute("seller", 10, "query_business", {
            **query, "before_logical_time": 15, "metrics": ["purchase_rate"]})["simulation"]
        self.assertEqual(older["activity_timeline"][0]["intervals"][0]["purchase_records"], 1)
        self.assertEqual(older["statistics"]["items"][0]["purchase_records"], 1)
        empty = self.runtime.execute("seller", 10, "query_business", {
            **query, "start_logical_time": 15, "metrics": ["purchase_rate"]})["simulation"]
        self.assertEqual(empty["settled_series"], [])
        self.assertEqual(empty["statistics"]["items"][0]["purchase_records"], 0)
        for extra in ({"before_logical_time": 25}, {"start_logical_time": 11},
                      {"before_logical_time": 15, "before_simulation_step": 2}):
            with self.assertRaises(ValueError):
                self.runtime.execute("seller", 10, "query_business", {**query, **extra})

    def test_pending_exposure_is_not_idle_and_settlement_does_not_count_it_twice(self):
        from agent_simulation.environment.operations import actions, reporting

        self.make_episode([], [])
        self.runtime.execute("customer", 7, "search_products", {"query": "水杯"})
        state = self.runtime.snapshot()
        state["clock"]["time"] = 15
        report = reporting.read(state, 10)
        product = next(p for p in report["products"] if p["item_id"] == 1)
        self.assertEqual([(s["start_inclusive"], s["end_exclusive"], s["exposure_records"])
                          for s in product["activity_intervals"]], [(10, 15, 1)])
        self.assertEqual(product["purchase_records"], 0)

        actions.submit_response(state, 7, {"clicked_item_ids": [1], "purchased_item_ids": [1]})
        state["clock"]["time"] = 20
        product = next(p for p in reporting.read(state, 10)["products"] if p["item_id"] == 1)
        self.assertEqual([(s["start_inclusive"], s["exposure_records"], s["purchase_records"])
                          for s in product["activity_intervals"]], [(10, 1, 0), (15, 0, 1)])
        self.assertEqual(sum(s["exposure_records"] for s in product["price_intervals"]), 1)

    def test_idle_spans_split_at_price_changes(self):
        episode = self.make_episode([WAIT] * 3 + [
            answer({"prices": [{"item_id": 1, "price_raw": 90}]}), WAIT, WAIT] + [WAIT] * 3,
            [WAIT] * 6)
        episode.run()
        report = self.runtime.execute("seller", 10, "read_business_report", {})
        spans = next(p for p in report["products"] if p["item_id"] == 1)["activity_intervals"]
        self.assertEqual([(s["start_inclusive"], s["end_exclusive"], s["price_raw"]) for s in spans],
                         [(10, 15, 120), (15, 25, 90)])
        self.assertTrue(all(s["exposure_records"] == 0 for s in spans))

    def test_invalid_action_stops_instead_of_becoming_wait(self):
        episode = self.make_episode([answer({"action": "maybe"})], [])
        with self.assertRaises(ValueError):
            episode.step()
        state = self.runtime.snapshot()
        self.assertEqual(state["clock"]["status"], "failed")
        self.assertEqual(state["clock"]["time"], 10)
        self.assertEqual(state["price_actions"], [])
        trace = json.loads(self.runtime.db.execute("SELECT trace FROM ticks").fetchone()[0])
        self.assertIn("error", trace)
        with self.assertRaisesRegex(ValueError, "not ready"):
            episode.step()

    def test_customer_failure_does_not_settle_other_customers(self):
        quote = answer({"prices": [{"item_id": 1, "price_raw": 90}]})
        episode = self.make_episode([quote, WAIT, WAIT], [answer({"query": "水杯"}), answer({"query": "水杯"}),
            answer({"clicked_item_ids": [1], "purchased_item_ids": []}), answer({"action": "invalid"})])
        with self.assertRaises(ValueError):
            episode.step()
        state = self.runtime.snapshot()
        self.assertEqual(state["feedback"], [])
        self.assertIn("7", state["pending"])
        self.assertEqual(state["clock"]["status"], "failed")
        self.assertEqual(state["active_prices"]["1"]["price_raw"], 90)
        self.assertEqual(state["clock"]["settled_phases"], ["seller"])
        trace = json.loads(self.runtime.db.execute("SELECT trace FROM ticks").fetchone()[0])
        self.assertEqual(trace["settled_phases"], ["seller"])

    def test_tick_tools_exclude_scheduling_and_batch_prices_wait(self):

        def second_batch(request, call_id):
            from agent_simulation.agents.prompts.seller import decode_input as seller_input
            self.assertEqual(len(request["action_output"]["final_schemas"]), 1)
            self.assertEqual(request["action_output"]["final_schemas"][0]["properties"]["prices"]["items"]["properties"]["item_id"]["enum"], [3])
            self.assertNotIn('"action":"no_action"', request["messages"][0]["content"])
            self.assertNotIn("empty", seller_input(request["messages"][1]["content"])["decision"]["price_scope"])
            self.assertEqual(self.runtime.snapshot()["active_prices"]["1"]["price_raw"], 120)
            names = [t["function"]["name"] for t in request["tools"]]
            self.assertNotIn("schedule_review", names)
            self.assertNotIn("schedule_review", request["messages"][0]["content"])
            return answer({"prices": [{"item_id": 3, "price_raw": 190}]})

        episode = self.make_episode([
            answer({"prices": [{"item_id": 1, "price_raw": 110}]}),
            second_batch, WAIT, WAIT], [WAIT] * 2, ticks=1)
        self.seller.price_batch_size = 1
        episode.run()
        state = self.runtime.snapshot()
        self.assertEqual([state["active_prices"][str(i)]["price_raw"] for i in (1, 3)], [110, 190])
        self.assertEqual({a["logical_time"] for a in state["price_actions"]}, {10})

    def test_reject_abandoning_a_partially_reviewed_shop(self):
        episode = self.make_episode([answer({"prices": [{"item_id": 1, "price_raw": 110}]}), WAIT], [])
        self.seller.price_batch_size = 1
        with self.assertRaisesRegex(ValueError, "Only the first Seller batch can wait"):
            episode.step()
        self.assertEqual(self.runtime.snapshot()["price_actions"], [])

    def test_tick_and_horizon_must_be_positive_integers(self):
        self.runtime.catalog = ProfileCatalogFixture(self.runtime)
        seller = SellerAgent(self.runtime, ScriptedModel([]), TokenizerFixture(), price_batch_size=16)
        customer = CustomerAgent(self.runtime, ScriptedModel([]), TokenizerFixture())
        for delta, horizon in ((0, 2), (True, 2), (5, 0), (5, 1.5)):
            with self.assertRaises(ValueError):
                Episode(self.runtime, seller, customer, tick_size=delta, max_ticks=horizon)


if __name__ == "__main__":
    unittest.main()
