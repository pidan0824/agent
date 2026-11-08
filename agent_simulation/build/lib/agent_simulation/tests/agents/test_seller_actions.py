"""Seller submits one final action; backend queries are not model actions."""

import unittest

from agent_simulation.agents.seller import SellerAgent
from agent_simulation.agents.prompts.seller import decode_input
from agent_simulation.tests.fixtures import MarketFixture, ScriptedModel, TokenizerFixture, tool, answer


class SellerActionTests(MarketFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.runtime.inputs.history_embedder = None

    def test_one_generation_quotes_the_complete_batch(self):
        model = ScriptedModel([answer({'prices': [
            {'item_id': 1, 'price_raw': 130}, {'item_id': 3, 'price_raw': 200}]})])
        agent = SellerAgent(self.runtime, model, TokenizerFixture(), price_batch_size=2)
        run = agent.run(10)
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(model.requests[0]['tools'], [])
        context = decode_input(model.requests[0]['messages'][1]['content'])
        self.assertEqual(context['decision']['requested_item_ids'], [1, 3])
        self.assertEqual(run['result']['reviewed_products'], 2)
        self.assertEqual(self.runtime.snapshot()['active_prices']['1']['price_raw'], 130)
        self.assertEqual([q['name'] for q in run['tools']], ['submit_prices'])

    def test_backend_queries_are_rejected_as_model_actions_without_retry(self):
        cases = [
            ('query_market_position', {'item_id': 1, 'before_time_index': 10}),
            ('discover_public_products', {'focal_item_id': 1, 'before_time_index': 10}),
            ('query_public_products', {'item_ids': [2], 'before_time_index': 10, 'limit': 1}),
            ('query_business', {'item_ids': [1, 3], 'before_time_index': 10, 'limit': 1}),
            ('query_shop_products', {'after_item_id': 0, 'limit': 1}),
        ]
        before = self.runtime.snapshot()
        for name, arguments in cases:
            with self.subTest(name=name):
                model = ScriptedModel([tool(name, arguments)])
                agent = SellerAgent(self.runtime, model, TokenizerFixture(), price_batch_size=2)
                with self.assertRaises(PermissionError):
                    agent.run(10)
                self.assertEqual(len(model.requests), 1)
                self.assertEqual(self.runtime.snapshot(), before)
                self.runtime.execute('seller', 10, name, arguments)
                self.assertEqual(self.runtime.snapshot(), before)
