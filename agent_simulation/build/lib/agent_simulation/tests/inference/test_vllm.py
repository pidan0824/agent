"""Exercise vLLM transport, exact prompts, model routing and rejected responses."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from agent_simulation import PROJECT_ROOT
from agent_simulation.agents.decisions import action_output
from agent_simulation.inference.client import ModelOutputError
from agent_simulation.inference.requests import load_tokenizer, prepare_request
from agent_simulation.inference.vllm import VLLMClient, structured_output


class VLLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_tokenizer(str(PROJECT_ROOT / 'models/Qwen2.5-3B-Instruct-tokenizer'),
                                       local_files_only=True)

    def request(self, stage='seller'):
        request, _ = prepare_request([{'role':'system','content':'Decide.'},
                                      {'role':'user','content':'保温杯'}], [], self.tokenizer)
        request.update(action_output=action_output(stage, allow_control_actions=True, displayed_ids=[1, 3],
                                                  price_ids=[1], allow_unknown_prices=True),
                       agent_role='seller' if stage == 'seller' else 'customer')
        return request

    def client(self, handler, **kwargs):
        backend = VLLMClient('http://vllm.test/v1', 'base', self.tokenizer,
                             role_models={'seller':'seller-lora'}, **kwargs)
        backend.http.close()
        backend.http = httpx.Client(base_url='http://vllm.test/v1/', transport=httpx.MockTransport(handler))
        self.addCleanup(backend.close)
        return backend

    def test_exact_tokens_and_role_routing(self):
        sent=[]
        def handler(request):
            if request.url.path == '/v1/models':
                return httpx.Response(200,json={'data':[{'id':'base'},{'id':'seller-lora'}]})
            sent.append(json.loads(request.content))
            self.assertEqual(request.url.path, '/v1/completions')
            return httpx.Response(200,json={'choices':[{
                'text':'{"prices":[{"item_id":1,"price_raw":100}]}',
                'finish_reason':'stop'}], 'usage':{'prompt_tokens':100,'completion_tokens':30}})
        backend=self.client(handler)
        self.assertFalse(backend.check_models()['remote_weights_verified'])
        original=self.request()
        with patch.object(self.tokenizer,'apply_chat_template',side_effect=AssertionError('retokenized')):
            reply=backend.complete(original,call_id='quote')
        self.assertEqual(sent[0]['prompt'],original['_prompt_tokens']['ids'])
        self.assertEqual(sent[0]['model'],'seller-lora')
        self.assertEqual(json.loads(reply['content']), {'prices': [{'item_id': 1, 'price_raw': 100}]})
        choices=sent[0]['structured_outputs']['json']['anyOf']
        self.assertEqual(len(choices),2) # wait or final quote
        prices=choices[0]['properties']['prices']
        self.assertEqual((prices['minItems'],prices['maxItems']), (1,1))
        self.assertEqual(prices['items']['properties']['item_id']['enum'],[1])

    def test_customer_json_and_reflection_use_distinct_contracts(self):
        request=self.request('query')
        self.assertEqual(len(structured_output(request)['json']['anyOf']),2)
        schema={'const':{"action": "no_update", 'content':'','source_ids':[]}}
        self.assertEqual(structured_output({'json_schema':schema}), {'json':schema})
        with self.assertRaisesRegex(ValueError,'one output contract'):
            structured_output({**request,'json_schema':schema})

    def test_response_display_scope_reaches_remote_schema(self):
        request = self.request('response')
        for ids in ([1, 3], []):
            request['action_output'] = action_output('response', displayed_ids=ids)
            schema = structured_output(request)['json']
            if ids:
                for field in schema['properties'].values():
                    self.assertEqual(field['items']['enum'], ids)
            else:
                self.assertEqual(schema, {'const': {'clicked_item_ids': [], 'purchased_item_ids': []}})

    def test_failed_and_truncated_responses_are_not_retried_or_repaired(self):
        for code,raw,finish,error in [(503,'',None,httpx.HTTPStatusError),
                                      (200,'{"query":"杯子"}', 'length',ModelOutputError),
                                      (200,'not json','stop',ModelOutputError),
                                      (200,'{"query":123}','stop',ModelOutputError),
                                      (200,'{"query":"杯子","extra":1}','stop',ModelOutputError)]:
            calls=[]
            def handler(request):
                calls.append(request)
                return httpx.Response(code,json={'choices':[{'text':raw,'finish_reason':finish}],
                                                 'usage':{'prompt_tokens':10,'completion_tokens':5}})
            with self.subTest(code=code,finish=finish,raw=raw):
                backend=self.client(handler)
                with self.assertRaises(error):
                    backend.complete(self.request('query'),call_id='bad')
                self.assertEqual(len(calls),1)

    def test_raw_truncation_is_logged_before_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model_calls.jsonl'
            backend=self.client(lambda _: httpx.Response(200,json={
                'choices':[{'text':'{"query":','finish_reason':'length'}],
                'usage':{'prompt_tokens':10,'completion_tokens':5}}),log_path=path)
            with self.assertRaises(ModelOutputError):
                asyncio.run(backend.acomplete(self.request('query'),call_id='cut'))
            logged=json.loads(path.read_text())
            self.assertEqual(logged['raw_output'],'{"query":')
            self.assertFalse(logged['finished'])

    def test_simulation_loader_uses_remote_without_loading_local_weights(self):
        from agent_simulation.simulation.run import load_default_model, parser

        answers = {
            'qwen': {'action': 'no_update'},
            'seller-sft': {'prices': [{'item_id': 1, 'price_raw': 100}]},
            'customer-sft': {'query': '杯子'},
        }
        sent = []

        def handler(request):
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': name} for name in answers]})
            name = json.loads(request.content)['model']
            sent.append(name)
            return httpx.Response(200, json={
                'choices': [{'text': json.dumps(answers[name]), 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 10},
            })

        with tempfile.TemporaryDirectory() as output:
            args = parser().parse_args([
                '--catalog', 'unused', '--model', str(PROJECT_ROOT/'models/Qwen2.5-3B-Instruct-tokenizer'),
                '--output', output, '--customers', '1', '--inference-backend', 'vllm',
                '--customer-served-model', 'customer-sft', '--seller-served-model', 'seller-sft',
            ])
            http = httpx.Client(base_url='http://vllm.test/v1/', transport=httpx.MockTransport(handler))
            with patch('agent_simulation.inference.vllm.httpx.Client', return_value=http), \
                 patch('agent_simulation.simulation.run.verify_checkpoint', side_effect=AssertionError('local weights')):
                with load_default_model(args) as (client, report):
                    self.assertEqual(report['role_models'], {'customer': 'customer-sft', 'seller': 'seller-sft'})
                    self.assertEqual(report['model'], 'qwen')
                    client.complete(self.request('seller'), call_id='seller')
                    client.complete(self.request('query'), call_id='customer')
                    reflection = self.request('query')
                    reflection.pop('agent_role')
                    reflection.pop('action_output')
                    reflection.update(json_schema={'const': {'action': 'no_update'}},
                                      memory_generation_method='fixture')
                    client.complete(reflection, call_id='reflection')
            self.assertTrue(http.is_closed)
            self.assertEqual(sent, ['seller-sft', 'customer-sft', 'qwen'])
