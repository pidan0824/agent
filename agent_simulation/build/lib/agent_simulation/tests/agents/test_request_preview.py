"""Previews and actual calls share target scope, batching, prompts and schemas."""
import asyncio
import unittest
from unittest.mock import patch

from agent_simulation.agents.customer import CustomerAgent
from agent_simulation.agents.seller import SellerAgent
from agent_simulation.agents.prompts.customer import decode_input, prepare_history
from agent_simulation.tests.fixtures import MarketFixture, ProfileCatalogFixture, ScriptedModel, TokenizerFixture, answer


class RequestPreviewTests(MarketFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.runtime.inputs.history_embedder = None

    def test_seller_preview_matches_every_actual_target_without_writes(self):
        model = ScriptedModel([answer({'prices': [{'item_id': 1, 'price_raw': 120}]}),
                               answer({'prices': [{'item_id': 3, 'price_raw': 200}]})])
        agent = SellerAgent(self.runtime, model, TokenizerFixture())
        revision = self.runtime.revision
        changes = self.runtime.db.total_changes
        preview = list(agent.preview_requests(10))
        self.assertEqual(self.runtime.db.total_changes, changes)
        self.assertEqual(self.runtime.revision, revision)
        self.assertEqual([r['action_output']['final_schemas'][0]['properties']['prices']['items']['properties']['item_id']['enum'] for r in preview], [[1], [3]])
        agent.run(10)
        self.assertEqual(preview, model.requests)

    def test_customer_preview_matches_query_and_all_response_batches(self):
        self.runtime.catalog = ProfileCatalogFixture(self.runtime)
        query_model = ScriptedModel([answer({'query': '保温杯'})])
        agent = CustomerAgent(self.runtime, query_model, TokenizerFixture(), response_batch_size=2)
        preview = list(agent.preview_requests(7))
        run = agent.prepare_decision(7)
        asyncio.run(agent.choose_query(run))
        self.assertEqual(preview, query_model.requests)
        agent.open_display(run)
        model = ScriptedModel([answer({'clicked_item_ids': [], 'purchased_item_ids': []})] * 2)
        agent.model = model
        changes = self.runtime.db.total_changes
        with patch("agent_simulation.agents.customer.prepare_history",
                   wraps=prepare_history) as history:
            preview = list(agent.preview_requests(7, response=True))
            self.assertEqual(self.runtime.db.total_changes, changes)
            history.assert_called_once()
            asyncio.run(agent.choose_response(run))
            self.assertEqual(history.call_count, 2)
        self.assertEqual(len(preview), 2)
        self.assertEqual(preview, model.requests)
        scopes = []
        for request in preview:
            displayed = sorted(x['item_id'] for x in decode_input(request['messages'][1]['content'])['displayed_items'])
            schema = request['action_output']['final_schemas'][0]
            for field in schema['properties'].values():
                self.assertEqual(field['items']['enum'], displayed)
            scopes.append(set(displayed))
        self.assertFalse(scopes[0] & scopes[1])
