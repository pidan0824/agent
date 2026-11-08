"""Test shop coverage, frozen batch context and atomic price application."""

from copy import deepcopy
import json
import unittest
from agent_simulation.tests.seller_context_fixture import memory as cited_memory
from unittest.mock import patch

from agent_simulation.tests.fixtures import MarketFixture
from agent_simulation.tests.fixtures import ScriptedModel, TokenizerFixture, answer
from agent_simulation.agents.seller import SellerAgent
from agent_simulation.agents.context.pricing import PricingSnapshot
from agent_simulation.agents.prompts.seller import decode_input
from agent_simulation.environment.operations.actions import SellerReview



class SellerBatchTests(MarketFixture, unittest.TestCase):

    def test_batch_keeps_description_for_selected_cross_item_evidence(self):
        context, _ = self.runtime.inputs.seller(10, retrieve=False, observe=False)
        item, other = context["owned_products"]
        context["retrieved_context"] = {"history": {"segments": [{"item_id": other["item_id"]}]}}
        context["public_product_pages"] = [{"item_id": p["item_id"], "candidates": []} for p in (item, other)]
        context["unused_shop_payload"] = {"products": [other]}
        projected = PricingSnapshot(context).view([item['item_id']])
        self.assertEqual([p["item_id"] for p in projected["owned_products"]], [item["item_id"]])
        self.assertEqual(projected["product_details"], context["product_details"])
        self.assertEqual(projected["public_product_pages"], context["public_product_pages"][:1])
        self.assertNotIn("unused_shop_payload", projected)
        self.assertNotIn("pricing_context", projected)


    def test_each_batch_retrieves_for_its_products_from_the_whole_shop(self):
        self.runtime.inputs.history_embedder = None
        for item in (1, 3):
            original = self.data.records(item)[0]
            self.data._histories[item] = [
                {**deepcopy(original), 'time_index': index, 'price_raw': 100 + item + index}
                for index in range(1, 9)
            ]
        queries = []

        def memory(role, actor, context, **kwargs):
            focal = context['history_presentation']['detail_item_ids']
            queries.append(focal)
            return [cited_memory(f'Experience for {focal}')]

        with patch.object(self.runtime.long_term_memory, 'context', side_effect=memory):
            run, model = self.run_batches([
                answer({'prices': [{'item_id': 1, 'price_raw': 120}]}),
                answer({'prices': [{'item_id': 3, 'price_raw': 200}]}),
            ])
        self.assertEqual(queries, [[1], [3]])
        contexts = [decode_input(r['messages'][1]['content']) for r in model.requests]
        for item, context in zip((1, 3), contexts):
            segments = context['history']['similar_evidence']
            self.assertTrue(any(s['item_id'] == item for s in segments))
            self.assertTrue(all('match' not in s for s in segments))
            self.assertTrue(any(s['item_id'] != item for s in segments))
            self.assertEqual(context['history']['long_term_memory'][0]['content'], f'Experience for {[item]}')
            self.assertTrue(all(r['observation_range'][1] < 10 for s in segments for r in s['price_runs']))
        self.assertEqual(run['result']['reviewed_products'], 2)

    def run_batches(self, replies, **kwargs):
        model = ScriptedModel(replies)
        run = SellerAgent(self.runtime, model, TokenizerFixture(), **kwargs).run(10)
        return run, model

    def test_all_products_one_commit_with_frozen_context(self):
        baseline = self.runtime.snapshot()

        def final(request, call_id):
            self.assertEqual(self.runtime.snapshot(), baseline)
            value = json.loads(request['messages'][-1]['content'])
            return answer({'prices': [{'item_id': 3, 'price_raw': 222}]})

        run, model = self.run_batches([
            answer({'prices': [{'item_id': 1, 'price_raw': 111}]}),
            final,
        ])
        after = self.runtime.snapshot()
        self.assertEqual(after['simulation_step'], baseline['simulation_step'] + 1)
        self.assertEqual([a['price_raw'] for a in after['price_actions']], [111, 222])
        self.assertEqual(len({a['simulation_step'] for a in after['price_actions']}), 1)
        self.assertEqual(run['result']['reviewed_products'], 2)
        self.assertEqual(self.runtime.db.execute("SELECT count(*) FROM calls WHERE name='submit_prices'").fetchone()[0], 1)
        contexts = [decode_input(r['messages'][1]['content']) for r in model.requests]
        self.assertTrue(all(c['seller'] == contexts[0]['seller'] and c['current_state']['observed_shop_period'] == contexts[0]['current_state']['observed_shop_period']
                            and c['history'].get('long_term_memory',[]) == contexts[0]['history'].get('long_term_memory',[]) for c in contexts))
        self.assertEqual(contexts[-1]['current_state']['products'][0]['active_quote']['price_raw'], 200)
        self.assertNotIn('111', json.dumps(contexts[-1]['current_state']['products'][0]['recent']))

    def test_later_failure_discards_prices(self):
        before = self.runtime.snapshot()
        with self.assertRaises(ValueError):
            self.run_batches([answer({'prices': [{'item_id': 1, 'price_raw': 99}]}),
                              answer({'prices': []})])
        self.assertEqual(self.runtime.snapshot(), before)
        self.assertEqual(self.runtime.db.execute("SELECT status FROM agent_runs").fetchone()[0], 'failed')
        # A new run starts a fresh review, never applies the failed draft.
        run, _ = self.run_batches([answer({'prices': [{'item_id': 1, 'price_raw': 120}]}),
                                  answer({'prices': [{'item_id': 3, 'price_raw': 200}]})])
        self.assertEqual(run['result']['kept_products'], 2)
        self.assertEqual(self.runtime.snapshot(), before)

    def test_foreign_or_out_of_batch_price_cannot_be_staged(self):
        before = self.runtime.snapshot()
        for item in (2, 3):
            with self.subTest(item=item), self.assertRaises(ValueError):
                self.run_batches([answer({'prices': [{'item_id': item, 'price_raw': 99}]})])
            self.assertEqual(self.runtime.snapshot(), before)

    def test_market_change_rejects_stale_review_without_clobbering_new_state(self):
        review = SellerReview(self.runtime, 10, self.runtime.freeze())
        review.stage({'prices': [{'item_id': 1, 'price_raw': 111}]}, [1])
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            review.commit('incomplete')
        review.stage({'prices': [{'item_id': 3, 'price_raw': 222}]}, [3])
        self.runtime.execute('seller', 10, 'submit_prices', {'prices': [{'item_id': 1, 'price_raw': 130}]})
        newer = self.runtime.snapshot()
        with self.assertRaisesRegex(ValueError, 'Market changed'):
            review.commit('stale')
        self.assertEqual(self.runtime.snapshot(), newer)

    def test_directory_is_owned_paginated_and_commit_does_not_change_pending_prices(self):
        page = self.runtime.execute('seller', 10, 'query_shop_products', {'after_item_id': 0, 'limit': 1})
        self.assertEqual([p['item_id'] for p in page['products']], [1])
        page2 = self.runtime.execute('seller', 10, 'query_shop_products', {'after_item_id': page['next_after_item_id'], 'limit': 1})
        self.assertEqual([p['item_id'] for p in page2['products']], [3])
        self.assertFalse(page2['has_more'])
        pending = self.runtime.execute('customer', 7, 'search_products', {'query': '保温杯'})
        self.run_batches([answer({'prices': [{'item_id': 1, 'price_raw': 111}]}),
                          answer({'prices': [{'item_id': 3, 'price_raw': 200}]})])
        self.assertEqual(self.runtime.snapshot()['pending']['7'], pending)
        with self.assertRaises(PermissionError):
            self.runtime.execute('customer', 7, 'query_shop_products', {'after_item_id': 0, 'limit': 1})

    def test_null_only_keeps_unknown_and_never_becomes_a_published_price(self):
        state = self.runtime.snapshot()
        state["active_prices"]["3"]["price_raw"] = None
        with self.runtime.db:
            self.runtime.db.execute("UPDATE state SET value=?", (json.dumps(state),))
        review = SellerReview(self.runtime, 10, self.runtime.freeze())
        with self.assertRaisesRegex(ValueError, "unknown active price"):
            review.stage({"prices": [{"item_id": 1, "price_raw": None}]}, [1])
        review.stage({"prices": [{"item_id": 1, "price_raw": 130}, {"item_id": 3, "price_raw": None}]}, [1, 3])
        result = review.commit("fixture")
        self.assertEqual(result["updated"], [{"item_id": 1, "price_raw": 130}])
        after = self.runtime.snapshot()
        self.assertIsNone(after["active_prices"]["3"]["price_raw"])
        next_review = SellerReview(self.runtime, 10, self.runtime.freeze())
        next_review.stage({"prices": [{"item_id": 3, "price_raw": 200}]}, [3])
        next_review.stage({"prices": [{"item_id": 1, "price_raw": 130}]}, [1])
        next_review.commit("fixture-next")
        self.assertEqual(self.runtime.snapshot()["active_prices"]["3"]["price_raw"], 200)

