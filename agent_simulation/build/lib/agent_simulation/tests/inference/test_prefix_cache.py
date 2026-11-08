"""Cached prefill must preserve real Qwen logits and greedy continuations."""
from types import SimpleNamespace
import unittest
import torch
from transformers import BatchEncoding
from agent_simulation.inference.local.qwen import PrefixCache
from agent_simulation.tests.fixtures import tiny_base


class PrefixCacheTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_base().eval()
        self.backend = SimpleNamespace(model=self.model, device='cpu', tokenizer=SimpleNamespace(pad_token_id=0),
                                       _adapter_args=lambda requests: {})
        self.cache = PrefixCache(max_tokens=8, max_entries=2, min_tokens=2)

    def inputs(self, rows):
        width = max(map(len, rows))
        return BatchEncoding(dict(input_ids=torch.tensor([[0]*(width-len(row))+row for row in rows]),
            attention_mask=torch.tensor([[0]*(width-len(row))+[1]*len(row) for row in rows])))

    def compare(self, rows):
        inputs = self.inputs(rows)
        with torch.inference_mode():
            expected = self.model.generate(**inputs, max_new_tokens=5, do_sample=False,
                                           return_dict_in_generate=True, output_scores=True)
            reused, kwargs = self.cache.prepare(self.backend, inputs, [{}]*len(rows))
            actual = self.model.generate(**reused, max_new_tokens=5, do_sample=False,
                                         return_dict_in_generate=True, output_scores=True, **kwargs)
        width = inputs.input_ids.shape[1]
        self.assertTrue(torch.equal(expected.sequences[:,width:], actual.sequences[:,width:]))
        for a, b in zip(expected.scores, actual.scores, strict=True):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        self.assertTrue(all(value.get_seq_length() <= 8 for value in self.cache.entries.values()))

    def test_variable_lengths_cached_and_uncached_generate_identical_tokens(self):
        self.compare([[3,4,5,6], [3,4,5,8,9]])
        self.assertEqual(self.cache.misses, 1)
        self.compare([[3,4,5,10,11,12], [3,4,5,7]])
        self.assertEqual(self.cache.hits, 1)
        self.assertGreater(self.cache.reused_tokens, 0)
        self.compare([[3,4,5,9]])
        self.assertEqual(self.cache.hits, 2)

    def test_unmatched_prefixes_eviction_and_disabled_cache(self):
        for prefix in ([3,4], [8,9], [11,12]):
            self.compare([prefix+[20], prefix+[21]])
        self.assertEqual(len(self.cache.entries), 2)
        inputs = self.inputs([[3,4,5]])
        disabled = PrefixCache(max_tokens=0)
        self.assertIs(disabled.prepare(self.backend, inputs, [{}])[0], inputs)
        self.backend._adapter_args = lambda req: {'adapter_names': [r['role'] for r in req]}
        self.assertIs(self.cache.prepare(self.backend, inputs, [{'role':'seller'}, {'role':'customer'}])[0], inputs)
