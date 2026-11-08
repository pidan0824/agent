"""Check actual composed instructions across SFT and the automatic workflow."""
import json
import unittest
from agent_simulation.tests.seller_context_fixture import seller_state
from agent_simulation.tests.seller_context_fixture import decision_context
from copy import deepcopy

from agent_simulation.agents.prompts.customer import messages, decode_input
from agent_simulation.agents.prompts.seller import pricing_messages
from agent_simulation.agents.prompts.seller import decode_input as seller_decode
from agent_simulation.agents.prompts.customer import customer_decision_messages
from agent_simulation.agents.prompts.seller import seller_decision_messages
from agent_simulation.agents.prompts.instructions import FACT_RULES
from agent_simulation.agents.memory.reflection import SUMMARY_PROMPTS


class PromptCompositionTests(unittest.TestCase):
    def test_customer_stage_permissions_are_exclusive(self):
        state = {'profile': {'user_id': 1}, 'short_term_memory': {'search_history': []},
                 'long_term_memory': [], 'history_cutoff_exclusive': 20}
        original = deepcopy(state)
        training = messages(state, {'query': '保温杯'})
        runtime = customer_decision_messages({**state, 'decision_time': {'logical_time': 21}})
        self.assertNotIn('"action":"exit"', training[0]['content'])
        self.assertIn('"action":"exit"', runtime[0]['content'])
        self.assertNotIn('Return only {"query"', runtime[0]['content'])
        response = customer_decision_messages({**state, 'query': '保温杯', 'displayed_items': []}, response=True)
        self.assertNotIn('"action":"exit"', response[0]['content'])
        self.assertEqual(response[0]['content'].count('"clicked_item_ids"'), 1)
        self.assertEqual(decode_input(training[1]['content']), state)
        self.assertEqual(json.loads(training[-1]['content']), {'query': '保温杯'})
        self.assertEqual(state, original)
        for request in (training, runtime, response):
            self.assertEqual(request[0]['content'].count(FACT_RULES), 1)
            self.assertNotIn('Optional queries', request[0]['content'])

    def test_seller_protocol_and_shared_evidence_survive_composition(self):
        state = seller_state()
        target = {'prices': [{'item_id': 1, 'price_raw': 100}]}
        sft = pricing_messages(state, target)
        tick = {**state, "decision_time": {}, "product_batch": {"index": 1, "count": 2}}
        first = seller_decision_messages(tick)[0]["content"]
        later = seller_decision_messages({**tick, "product_batch": {"index": 2, "count": 2}})[0]["content"]
        self.assertEqual(seller_decode(sft[1]['content']), decision_context(state))
        self.assertEqual(json.loads(sft[-1]['content']), target)
        for system in (sft[0]["content"], first, later):
            self.assertEqual(system.count(FACT_RULES), 1)
        self.assertIn('Return only the JSON object.', sft[0]['content'])
        self.assertIn('An empty prices list is invalid', sft[0]['content'])
        self.assertEqual(first, later)
        for system in (first, later):
            self.assertNotIn('"action":"no_action"', system)
            self.assertIn('Return one JSON object with the requested prices.', system)
            self.assertIn('Include each requested ID exactly once.', system)
            self.assertIn('send null to keep an unknown active price', system)
            self.assertNotIn('Return only the JSON object.', system)


