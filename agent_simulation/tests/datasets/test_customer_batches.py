"""Test request-level batch accounting and label-independent partitioning."""
import asyncio
import json
import unittest
from agent_simulation.agents.context.shopping import project_customer

from copy import deepcopy
from unittest.mock import patch
from agent_simulation.tests.fixtures import MarketFixture
from agent_simulation.tests.fixtures import ScriptedModel, TokenizerFixture, ProfileCatalogFixture, answer, tool
from agent_simulation.agents.customer import CustomerAgent
from agent_simulation.agents.prompts.customer import decode_input
import agent_simulation.environment.batching as batching
from agent_simulation.environment.batching import response_batches
from agent_simulation.datasets.samples.customer import split_supervision
from agent_simulation.agents.context.shopping import displayed_details


class BatchTokenizer(TokenizerFixture):
    def apply_chat_template(self, messages, tools, **kwargs):
        state = decode_input(messages[1]['content'])
        return list(range(100 + 1500 * len(state.get('displayed_items', []))))


POLICY = {'context_limit': 4500, 'reserved_output_tokens': 2048}


class BatchTests(MarketFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.runtime.catalog = ProfileCatalogFixture(self.runtime)

    def pending(self):
        self.runtime.execute('customer', 7, 'search_products', {'query': '保温杯'})
        return self.runtime.inputs.customer(7, response=True)

    def test_partition_has_no_capacity_probes_and_uses_configured_item_limit(self):
        state = self.pending()
        state['displayed_items'] = [{'item_id':i} for i in range(1459)]
        with patch('agent_simulation.agents.prompts.customer.messages',
                   side_effect=AssertionError('Batch planning must not render a prompt')):
            batches = response_batches(state)
        self.assertEqual(len(batches),23)
        self.assertTrue(all(len(b['displayed_items']) <= 64 for b in batches))
        self.assertEqual([p for b in batches for p in b['displayed_items']],state['displayed_items'])
        self.assertEqual(response_batches({**state,'displayed_items':[]}),[{**state,'displayed_items':[]}])

    def test_partition_does_not_use_labels_and_reconstructs_all_facts(self):
        state = self.pending()
        ids = [p['item_id'] for p in state['displayed_items']]
        original = deepcopy(state)
        sample = {'sample_id': 'response:fixture', 'split': 'test', 'input': state,
                  'target': {'clicked_item_ids': ids[:1], 'purchased_item_ids': ids[-1:]},
                  'tools': []}
        with patch.object(batching, "CUSTOMER_RESPONSE_BATCH_SIZE", 1):
            rows = split_supervision(sample)
            changed = split_supervision({**sample, 'target': {'clicked_item_ids': [], 'purchased_item_ids': []}})
        self.assertGreater(len(rows), 1)
        self.assertEqual([r['input'] for r in rows], [r['input'] for r in changed])
        self.assertEqual([p for r in rows for p in r['input']['displayed_items']], state['displayed_items'])
        self.assertEqual(state, original)
        self.assertAlmostEqual(sum(r['batching']['request_weight'] for r in rows), 1)
        for row in rows:
            self.assertEqual(row['input']['short_term_memory'], state['short_term_memory'])
            self.assertEqual(row['input']['long_term_memory'], state['long_term_memory'])
            self.assertEqual(row['split'], 'test')
            self.assertEqual(decode_input(row['messages'][1]['content']), project_customer(row['input']))
            self.assertEqual([m['role'] for m in row['messages']], ['system', 'user', 'assistant'])

    def test_replaced_display_is_rejected_before_constructing_a_response_prompt(self):
        empty = {"clicked_item_ids": [], "purchased_item_ids": []}
        first = self.runtime.execute("customer", 7, "search_products", {"query": "保温杯"})
        self.runtime.execute("customer", 7, "submit_response", empty)
        self.runtime.execute("customer", 7, "search_products", {"query": "保温杯"})
        model = ScriptedModel([answer(empty)])
        agent = CustomerAgent(self.runtime, model, TokenizerFixture())
        run = agent.prepare_decision(7)
        run["displayed_list"] = first
        with patch.object(self.runtime.long_term_memory, "observe_context") as record:
            with self.assertRaisesRegex(ValueError, "displayed request changed"):
                asyncio.run(agent.choose_response(run))
            record.assert_not_called()
        self.assertEqual(model.requests, [])

    def test_pending_response_prompt_keeps_the_displayed_price_after_a_requote(self):
        empty = {"clicked_item_ids": [], "purchased_item_ids": []}
        first = self.runtime.execute("customer", 7, "search_products", {"query": "保温杯"})
        product = first["displayed_items"][0]
        item = product["item_id"]
        new_price = product["price_raw"] + 25
        self.runtime.execute("seller", product["seller_id"], "submit_prices",
                             {"prices": [{"item_id": item, "price_raw": new_price}]})
        model = ScriptedModel([answer(empty)])
        CustomerAgent(self.runtime, model, TokenizerFixture()).run(7)
        prompt = decode_input(model.requests[0]["messages"][1]["content"])
        offer = next(p for p in prompt["displayed_items"] if p["item_id"] == item)
        self.assertEqual(offer["price_raw"], product["price_raw"])
        self.assertNotIn("price_provenance", offer)
        self.assertEqual(self.runtime.snapshot()["active_prices"][str(item)]["price_raw"], new_price)

    def test_old_customer_decision_cannot_settle_a_new_display(self):
        empty = {'clicked_item_ids': [], 'purchased_item_ids': []}
        self.pending()
        agent = CustomerAgent(self.runtime, ScriptedModel([answer(empty)]), TokenizerFixture())
        run = agent.run(7, defer_settlement=True)
        self.runtime.execute('customer', 7, 'submit_response', empty)
        self.pending()  # Same customer, query and products, different request.
        before = self.runtime.snapshot()
        with self.assertRaisesRegex(ValueError, "displayed request changed"):
            agent.settle_staged([run])
        self.assertEqual(self.runtime.snapshot(), before)

    def test_deferred_settlement_records_all_receipts_before_reflecting(self):
        empty = {"clicked_item_ids": [], "purchased_item_ids": []}
        agent = CustomerAgent(self.runtime, ScriptedModel([answer(empty), answer(empty)]),
                              TokenizerFixture())
        for actor in (7, 8):
            self.runtime.execute("customer", actor, "search_products", {"query": "保温杯"})
        runs = [agent.run(actor, defer_settlement=True) for actor in (7, 8)]

        def reflect(run):
            self.assertFalse(self.runtime.snapshot()["pending"])
            for settled in runs:
                self.assertEqual(settled["settled_result"]["request_id"],
                                 settled["displayed_list"]["request_id"])

        with patch.object(agent, "reflect_on_feedback", side_effect=reflect) as recorded:
            agent.settle_staged(iter(runs))
        self.assertEqual([call.args[0]["actor"] for call in recorded.call_args_list], [7, 8])
        self.assertEqual(self.runtime.db.execute(
            "SELECT count(*) FROM agent_runs WHERE status='complete'").fetchone()[0], 2)
        with self.assertRaisesRegex(ValueError, "already applied"):
            agent.settle_staged(runs)

    def test_invalid_settlement_receipts_are_never_acknowledged(self):
        self.pending()
        empty = {"clicked_item_ids": [], "purchased_item_ids": []}
        agent = CustomerAgent(self.runtime, ScriptedModel([answer(empty)]), TokenizerFixture())
        run = agent.run(7, defer_settlement=True)
        for receipts, error in (([], "result count mismatch"), ([{"request_id": -1}], "request ID mismatch")):
            with self.subTest(receipts=receipts), \
                 patch.object(self.runtime, "settle_customers", return_value=receipts), \
                 patch.object(agent, "reflect_on_feedback") as reflect:
                with self.assertRaisesRegex(ValueError, error):
                    agent.settle_staged([run])
                self.assertNotIn("settled_result", run)
                reflect.assert_not_called()

    def test_failed_search_retains_the_selected_query_for_retry(self):
        agent = CustomerAgent(self.runtime, ScriptedModel([]), TokenizerFixture())
        run = agent.prepare_decision(7)
        run["search_query"] = {"query": "保温杯"}
        with patch.object(self.runtime, "execute", side_effect=RuntimeError("search failed")):
            with self.assertRaisesRegex(RuntimeError, "search failed"):
                agent.open_display(run)
        self.assertEqual(run["search_query"], {"query": "保温杯"})
        self.assertNotIn("displayed_list", run)
        agent.open_display(run)
        self.assertNotIn("search_query", run)
        self.assertEqual(run["displayed_list"]["query"], "保温杯")

    def test_standalone_reflection_failure_keeps_settlement_receipt(self):
        self.pending()
        agent = CustomerAgent(self.runtime, ScriptedModel([
            answer({'clicked_item_ids': [], 'purchased_item_ids': []})
        ]), TokenizerFixture())
        with patch.object(agent, "reflect_on_feedback", side_effect=RuntimeError("fixture reflection")):
            with self.assertRaisesRegex(RuntimeError, "fixture reflection"):
                agent.run(7)
        status, trace = self.runtime.db.execute("SELECT status,trace FROM agent_runs").fetchone()
        self.assertEqual(status, "failed")
        run = json.loads(trace)
        self.assertEqual(run["settled_result"], run["result"])
        history = self.runtime.snapshot()["customers"]["7"]["simulation_history"]
        self.assertEqual([e["request_id"] for e in history], [run["settled_result"]["request_id"]])

    def test_many_responses_have_one_settlement_and_one_history_event(self):
        state = self.pending()
        ids = [p['item_id'] for p in state['displayed_items']]
        before = self.runtime.snapshot()
        replies = []
        for item in ids:
            replies.extend([
                            answer({'clicked_item_ids': [item], 'purchased_item_ids': [item] if item == ids[-1] else []})])
        model = ScriptedModel(replies)
        agent = CustomerAgent(self.runtime, model, BatchTokenizer(), response_batch_size=1)
        with patch.object(batching, "CUSTOMER_RESPONSE_BATCH_SIZE", 1):
            result = agent.run(7)
        after = self.runtime.snapshot()
        self.assertEqual(result['result']['clicked_item_ids'], ids)
        self.assertEqual(result['result']['purchased_item_ids'], ids[-1:])
        self.assertEqual(after['simulation_step'], before['simulation_step'] + 1)
        self.assertEqual(len(after['feedback']) - len(before['feedback']), len(ids))
        self.assertEqual(len(after['customers']['7']['simulation_history']), 1)
        self.assertEqual([t['name'] for t in result['tools']].count('submit_response'), 1)
        traces = [entry for entry in result['context_provenance'] if entry['stage'] == 'response']
        self.assertEqual(len(traces), len(model.requests))
        expected_memory = project_customer(state)['short_term_memory']
        recent_ids = [event.get('session_id') for event in state['short_term_memory']['search_history']]
        segments = state.get('retrieved_context', {}).get('history', {}).get('segments', [])
        retrieved_ids = [[event.get('session_id') for event in segment['records']] for segment in segments]
        for request, trace in zip(model.requests, traces, strict=True):
            self.assertEqual(decode_input(request['messages'][1]['content'])['short_term_memory'], expected_memory)
            self.assertEqual(trace['recent_session_ids'], recent_ids)
            self.assertEqual(trace['retrieved_session_ids'], retrieved_ids)

    def test_failure_has_no_partial_settlement_and_resume_does_not_search_again(self):
        state = self.pending()
        ids = [p['item_id'] for p in state['displayed_items']]
        before = self.runtime.snapshot()
        model = ScriptedModel([answer({'clicked_item_ids': ids[:1], 'purchased_item_ids': []}), RuntimeError('probe')])
        with patch.object(batching, "CUSTOMER_RESPONSE_BATCH_SIZE", 1):
            with self.assertRaisesRegex(RuntimeError, 'probe'):
                CustomerAgent(self.runtime, model, BatchTokenizer()).run(7)
            self.assertEqual(self.runtime.snapshot(), before)
            model = ScriptedModel([answer({'clicked_item_ids': [], 'purchased_item_ids': []}) for _ in ids])
            result = CustomerAgent(self.runtime, model, BatchTokenizer()).run(7)
        self.assertNotIn('search_products', [t['name'] for t in result['tools']])
        self.assertEqual(self.runtime.snapshot()['simulation_step'], before['simulation_step'] + 1)

    def test_cross_batch_answer_still_fails_without_settlement(self):
        state = self.pending()
        other = state['displayed_items'][-1]['item_id']
        before = self.runtime.snapshot()
        reply = answer({'clicked_item_ids': [other], 'purchased_item_ids': []})
        with patch.object(batching, "CUSTOMER_RESPONSE_BATCH_SIZE", 1):
            with self.assertRaises(ValueError):
                CustomerAgent(self.runtime, ScriptedModel([reply]), BatchTokenizer()).run(7)
        self.assertEqual(self.runtime.snapshot(), before)

    def test_details_use_displayed_snapshot_and_never_return_labels(self):
        state = self.pending()
        p = deepcopy(state['displayed_items'][0])
        p.update(is_clicked=1, is_purchased=1, secret='private')
        result = displayed_details([p], [p['item_id']])['products'][0]
        self.assertEqual(result['price_raw'], p['price_raw'])
        self.assertNotIn('secret', result)
        self.assertNotIn('is_clicked', result)
        self.assertNotIn('is_purchased', result)
        before = self.runtime.snapshot()
        with self.assertRaises(PermissionError):
            self.runtime.execute('customer', 7, 'query_displayed_products', {'item_ids': [999999]})
        with self.assertRaises(ValueError):
            self.runtime.execute('customer', 7, 'query_displayed_products', {'item_ids': []})
        self.assertEqual(self.runtime.snapshot(), before)

    def test_detail_price_does_not_change_after_another_seller_action(self):
        state = self.pending()
        item = next(p for p in state['displayed_items'] if p['item_id'] == 1)
        self.runtime.execute('seller', 10, 'submit_prices',
                             {'prices': [{'item_id': 1, 'price_raw': 9999}]})
        result = self.runtime.execute('customer', 7, 'query_displayed_products', {'item_ids': [1]})
        self.assertEqual(result['products'][0]['price_raw'], item['price_raw'])
        self.assertNotEqual(result['products'][0]['price_raw'], 9999)

    def test_query_stage_has_no_product_detail_tool(self):
        model = ScriptedModel([tool('query_displayed_products', {'item_ids': [1]})])
        before = self.runtime.snapshot()
        with self.assertRaises(PermissionError):
            CustomerAgent(self.runtime, model, TokenizerFixture()).run(7)
        self.assertEqual(self.runtime.snapshot(), before)

    def test_seller_queries_complete_attributes_omitted_from_overview(self):
        context, _ = self.runtime.inputs.seller(10)
        self.assertEqual({p['item_id'] for p in context['product_details']}, {1, 3})
        self.assertNotIn('brand_name', context['owned_products'][0])
        result = self.runtime.execute('seller', 10, 'query_business',
                                      {'item_ids': [1], 'before_time_index': 10, 'limit': 2})
        self.assertIn('brand_name', result['products'][0])
        self.assertIn('category_level3_id', result['products'][0])
        self.assertLess(result['products'][0]['reference_price_time_index'], 10)
