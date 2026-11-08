"""Reuse capacity-check tokens through direct model dispatch."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from agent_simulation import PROJECT_ROOT
from agent_simulation.inference.requests import load_tokenizer, prepare_request, read_tokens
from agent_simulation.inference.local.qwen import QwenLocal
from agent_simulation.inference.client import complete_once
from agent_simulation.tests.fixtures import answer


class PromptTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_tokenizer(str(PROJECT_ROOT/'models/Qwen2.5-3B-Instruct-tokenizer'), local_files_only=True)

    def request(self, text='保温杯'):
        return prepare_request([{'role':'system','content':'Decide.'}, {'role':'user','content':text}],
                               [], self.tokenizer)

    def test_backend_reuses_exact_tokens_without_rendering_or_tokenizing(self):
        request, count = self.request()
        backend = object.__new__(QwenLocal)
        backend.tokenizer, backend.device, backend._torch = self.tokenizer, 'cpu', torch
        with patch.object(self.tokenizer, 'apply_chat_template', side_effect=AssertionError('duplicate template')):
            inputs = backend._inputs([request, request])
        self.assertEqual(inputs.input_ids.tolist(), [request['_prompt_tokens']['ids']]*2)
        self.assertEqual(inputs.attention_mask.sum().item(), count*2)
        changed = deepcopy(request)
        changed['messages'][1]['content'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'content changed'):
            backend._inputs([changed])

    def test_direct_dispatch_preserves_prompt_identity(self):
        request, _ = self.request()
        received = []
        class Sink:
            generator_id = 'fixture'
            def complete(inner, value, *, call_id):
                received.append(read_tokens(value, self.tokenizer))
                return answer({'prices':[{'item_id':1,'price_raw':100}]})
        asyncio.run(complete_once(Sink(), request, call_id='prepared'))
        self.assertEqual(received, [request['_prompt_tokens']['ids']])

    def test_same_common_prefix_with_different_roles_has_separate_kv(self):
        from peft import LoraConfig, get_peft_model
        from agent_simulation.inference.local.qwen import PrefixCache
        from agent_simulation.tests.fixtures import tiny_base
        model = get_peft_model(tiny_base(), LoraConfig(task_type='CAUSAL_LM', r=2,
                               target_modules=['q_proj','v_proj']), adapter_name='seller')
        model.add_adapter('customer', LoraConfig(task_type='CAUSAL_LM', r=2,
                          target_modules=['q_proj','v_proj']))
        model.eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if 'lora_B' in name:
                    parameter.normal_(std=0.2)
        backend = SimpleNamespace(model=model, device='cpu', tokenizer=SimpleNamespace(pad_token_id=0),
                  _adapter_args=lambda req: {'adapter_names':[r['agent_role'] for r in req]})
        from transformers import BatchEncoding
        inputs = BatchEncoding(dict(input_ids=torch.tensor([[3,4,5,6]]), attention_mask=torch.ones(1,4,dtype=torch.long)))
        cache = PrefixCache(max_tokens=3, min_tokens=2)
        for role in ('seller','customer','seller'):
            request = [{'agent_role':role}]
            with torch.inference_mode():
                expected = model.generate(**inputs, max_new_tokens=3, do_sample=False, **backend._adapter_args(request))
                reused, kv = cache.prepare(backend, inputs, request)
                actual = model.generate(**reused, **kv, max_new_tokens=3, do_sample=False, **backend._adapter_args(request))
            self.assertTrue(torch.equal(expected,actual))
        self.assertEqual(cache.misses,2)
        self.assertEqual(cache.hits,1)
