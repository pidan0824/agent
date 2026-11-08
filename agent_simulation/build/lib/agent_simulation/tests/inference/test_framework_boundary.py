"""Test model-client dispatch and market action validation."""

import unittest

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from agent_simulation.agents.customer import CustomerAgent
from agent_simulation.inference.client import FrameworkClient
from agent_simulation.inference.client import ModelOutputError, finish_reply
from agent_simulation.simulation.episode import Episode
from agent_simulation.inference.client import response_tool_calls
from agent_simulation.agents.seller import SellerAgent
from agent_simulation.agents.schemas import strict_json, validate_arguments
from agent_simulation.tests.fixtures import MarketFixture, ProfileCatalogFixture, ScriptedModel, TokenizerFixture, answer


class FrameworkBoundaryTests(MarketFixture, unittest.TestCase):
    def test_native_provider_drives_customer_and_seller(self):
        actions = iter([TextPart('{"query":"保温杯"}'),
                        TextPart('{"clicked_item_ids":[],"purchased_item_ids":[1]}'),
                        TextPart('{"prices":[{"item_id":1,"price_raw":130},{"item_id":3,"price_raw":200}]}')])
        stages = []
        def respond(messages, info):
            stages.append([t.name for t in info.function_tools])
            return ModelResponse(parts=[next(actions)])
        tokenizer = TokenizerFixture()
        client = FrameworkClient(FunctionModel(respond), tokenizer, 2048)
        self.runtime.catalog = ProfileCatalogFixture(self.runtime)
        customer = CustomerAgent(self.runtime, client, tokenizer).run(7)
        self.assertEqual(customer['result']['purchased_item_ids'], [1])
        SellerAgent(self.runtime, client, tokenizer, price_batch_size=16).run(10)
        self.assertEqual(stages, [[], [], []])
        self.assertEqual(self.runtime.snapshot()['active_prices']['1']['price_raw'], 130)

    def test_native_provider_wait_exit_and_invalid_output_are_single_calls(self):
        stages = []
        def respond(messages, info):
            stages.append([t.name for t in info.function_tools])
            action = "no_action" if len(stages) <= 3 else "exit"
            return ModelResponse(parts=[TextPart(answer({"action": action})["content"])])
        client = FrameworkClient(FunctionModel(respond), TokenizerFixture(), 2048)
        self.runtime.catalog = ProfileCatalogFixture(self.runtime)
        self.runtime.catalog.connection = None
        seller = SellerAgent(self.runtime, client, client.tokenizer, price_batch_size=16)
        customer = CustomerAgent(self.runtime, client, client.tokenizer)
        Episode(self.runtime, seller, customer, tick_size=1, max_ticks=2).run()
        self.assertEqual(self.runtime.snapshot()['clock']['termination_reason'], 'all_customers_exited')
        self.assertEqual(stages, [[]] * 5)
        # The same text is invalid under the declared memory-summary schema.
        with self.assertRaises(ModelOutputError):
            client.complete({'messages':[{'role':'user','content':'Summarize.'}], 'tools':[],
                             'json_schema':{'type':'object','required':['content','source_ids']}},
                            call_id='invalid-summary')
        self.assertEqual(len(stages), 6)

    def test_local_provider_keeps_the_original_tokenizer_request(self):
        backend = ScriptedModel([answer({'prices':[{'item_id':1,'price_raw':120},{'item_id':3,'price_raw':200}]})])
        backend.generator_id = 'local-fixture'
        run = SellerAgent(self.runtime, backend, TokenizerFixture(), price_batch_size=16).run(10)
        self.assertEqual(backend.requests, [run['model_calls'][0]['request']])
        self.assertEqual(run['result']['updated'], [])

    def test_assistant_envelope_rejects_malformed_calls(self):
        for malformed in ({}, [{"type": "wrong", "id": "a",
                                "function": {"name": "submit_prices", "arguments": {"prices": []}}}]):
            with self.subTest(malformed=malformed):
                reply = {"role": "assistant", "content": '{"action":"no_action"}', "tool_calls": malformed}
                with self.assertRaises(ModelOutputError) as caught:
                    response_tool_calls(reply)
                self.assertEqual(caught.exception.raw_output, reply)

    def test_partial_draft_keeps_text_and_provider_failure_propagates(self):
        reasons = iter(["length", "error"])
        def respond(messages, info):
            return ModelResponse(parts=[TextPart('{"content":')], finish_reason=next(reasons))
        client = FrameworkClient(FunctionModel(respond), TokenizerFixture(), 64)
        request = {"messages": [{"role": "user", "content": "Summarize."}], "tools": [],
                   "memory_generation_method": "role-experience-summary-v1"}
        with self.assertRaises(ModelOutputError) as truncated:
            client.complete(request, call_id="partial")
        self.assertEqual(truncated.exception.raw_output, '{"content":')
        self.assertIs(truncated.exception.finished, False)
        with self.assertRaisesRegex(RuntimeError, "provider"):
            client.complete(request, call_id="provider-failure")

    def test_strict_tool_contract(self):
        with self.assertRaises(ValueError):
            validate_arguments(1.0, {'type':'integer'})
        for x in (True, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                validate_arguments(x, {'type':'number'})
        with self.assertRaises(ValueError):
            validate_arguments('  ', {'type':'string','minLength':1})

    def test_strict_json_rejects_invalid_model_output(self):
        for literal in ("1e999", "-1e999"):
            with self.subTest(literal=literal):
                with self.assertRaisesRegex(ValueError, "Nonfinite JSON"):
                    strict_json('{"prices":[' + literal + ']}')
        self.assertEqual(strict_json('{"prices":[1.5e2]}'), {"prices": [150.0]})
        for raw in ('{"x":1,"x":2}', "```json\n{}\n```", "<tool_call>{}</tool_call>", "because {}"):
            with self.subTest(raw=raw), self.assertRaises(ModelOutputError):
                finish_reply(raw, finished=True, request={})
