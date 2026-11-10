"""Official-template answer masking, padding and request-weight preservation."""

from agent_simulation import PROJECT_ROOT
from contextlib import closing
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from agent_simulation.inference.requests import load_tokenizer
from agent_simulation.datasets.tokenization import encode_sample, collate, IGNORE_INDEX
from agent_simulation.datasets.labels import SFTDataset


class BatchTests(unittest.TestCase):
    def test_capacity_and_training_share_one_tokenization_without_exporting_the_cache(self):
        original = deepcopy(self.samples['query_14'])
        data = SFTDataset.__new__(SFTDataset)
        data.tokenizer = self.tokenizer
        with patch.object(self.tokenizer, 'apply_chat_template', wraps=self.tokenizer.apply_chat_template) as render:
            prepared = data.check_sample(original)
            encoded = encode_sample(prepared, self.tokenizer)
            actual = collate([prepared], self.tokenizer)
            self.assertEqual(render.call_count, 2)
        self.assertEqual(encoded, encode_sample(original, self.tokenizer))
        self.assertEqual(actual, collate([original], self.tokenizer))
        self.assertEqual(json.loads(json.dumps(prepared)), dict(prepared))
        self.assertEqual(prepared['capacity']['input_tokens'], encoded['answer_start'])

    def test_cached_tokens_are_recomputed_for_a_different_tokenizer_or_edited_messages(self):
        data = SFTDataset.__new__(SFTDataset)
        data.tokenizer = self.tokenizer
        sample = data.check_sample(deepcopy(self.samples['response_12758_batch_1']))
        other = load_tokenizer(str(PROJECT_ROOT / 'models/Qwen2.5-3B-Instruct-tokenizer'), local_files_only=True)
        with patch.object(other, 'apply_chat_template', wraps=other.apply_chat_template) as render:
            encoded = encode_sample(sample, other)
            self.assertEqual(render.call_count, 2)
        self.assertEqual(encoded, encode_sample(dict(sample), self.tokenizer))
        sample['messages'][1]['content'] += ' Additional observed information.'
        with patch.object(self.tokenizer, 'apply_chat_template', wraps=self.tokenizer.apply_chat_template) as render:
            changed = encode_sample(sample, self.tokenizer)
            self.assertEqual(render.call_count, 2)
        self.assertNotEqual(changed['input_ids'], encoded['input_ids'])
        self.assertEqual(changed['sample_weight'], encoded['sample_weight'])

    @classmethod
    def setUpClass(cls):
        root = PROJECT_ROOT
        cls.tokenizer = load_tokenizer(str(root / 'models/Qwen2.5-3B-Instruct-tokenizer'), local_files_only=True)
        with closing(SFTDataset(root / 'data/kuaisearch/lite/sft', cls.tokenizer)) as data:
            ids = ['query:14', 'response:14', 'response:12758:batch:1',
                   'seller:108629:579227:850769']
            cls.samples = {key.replace(':', '_'): dict(data.sample(key)) for key in ids}

    def test_only_final_answer_tokens_are_supervised_for_every_stage(self):
        for name in ['query_14', 'response_14', 'seller_108629_579227_850769']:
            with self.subTest(stage=name):
                sample = self.samples[name]
                encoded = encode_sample(sample, self.tokenizer)
                prefix = self.tokenizer.apply_chat_template(sample['messages'][:-1], tools=sample['tools'],
                    tokenize=True, return_dict=False, add_generation_prompt=True)
                self.assertEqual(encoded['answer_start'], len(prefix))
                self.assertEqual(encoded['labels'][:len(prefix)], [IGNORE_INDEX]*len(prefix))
                self.assertEqual(encoded['labels'][len(prefix):], encoded['input_ids'][len(prefix):])
                answer = self.tokenizer.decode(encoded['labels'][len(prefix):])
                self.assertIn('prices' if name.startswith('seller') else ('clicked_item_ids' if name.startswith('response') else 'query'), answer)

    def test_external_source_and_provenance_never_change_model_tokens(self):
        sample = self.samples['seller_108629_579227_850769']
        other = deepcopy(sample)
        other['source'] = {'secret_audit_value': 'must not enter the model'}
        other['provenance'] = [{'generator': 'not a prompt'}]
        other['capacity'] = {'this_is_not_model_input': True}
        self.assertEqual(encode_sample(sample, self.tokenizer), encode_sample(other, self.tokenizer))

    def test_padding_is_masked_and_weights_are_not_model_kwargs(self):
        samples = [self.samples[n] for n in ['query_14', 'response_14', 'response_12758_batch_1']]
        batch = collate(samples, self.tokenizer)
        self.assertEqual(set(batch['model_inputs']), {'input_ids', 'attention_mask', 'labels'})
        self.assertEqual(len({len(r) for r in batch['model_inputs']['input_ids']}), 1)
        self.assertIn(0, batch['model_inputs']['attention_mask'][0])
        for mask, labels in zip(batch['model_inputs']['attention_mask'], batch['model_inputs']['labels']):
            self.assertTrue(all(y == IGNORE_INDEX for m, y in zip(mask, labels) if m == 0))
        self.assertEqual(batch['sample_weights'][0], 1)
        k = samples[-1]['batching']['count']
        self.assertAlmostEqual(batch['sample_weights'][-1], 1/k)
        self.assertAlmostEqual(sum([batch['sample_weights'][-1]]*k), 1)

    def test_invalid_weight_and_invented_query_trajectory_are_rejected(self):
        sample = deepcopy(self.samples['response_12758_batch_1'])
        sample['batching']['request_weight'] = 0
        with self.assertRaisesRegex(ValueError, 'Response weight'):
            encode_sample(sample, self.tokenizer)
        sample = deepcopy(self.samples['query_14'])
        sample['messages'].insert(2, {'role': 'tool', 'content': 'invented trajectory'})
        with self.assertRaisesRegex(ValueError, 'invented tool trajectories'):
            encode_sample(sample, self.tokenizer)

    def test_overflow_raises_without_truncating_answer_or_context(self):
        sample = deepcopy(self.samples['query_14'])
        sample['messages'][1]['content'] += ' very long context' * 20000
        original = deepcopy(sample)
        with self.assertRaisesRegex(ValueError, 'overflow'):
            encode_sample(sample, self.tokenizer)
        self.assertEqual(sample, original)

class AnchoredEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=load_tokenizer(str(PROJECT_ROOT/'models/Qwen2.5-3B-Instruct-tokenizer'),local_files_only=True)

    def test_single_full_encoding_matches_official_prefix_and_complete_for_unicode_and_special_text(self):
        from agent_simulation.datasets.tokenization import _anchored_tokens
        for text in ['保温杯','emoji 🧪\n换行\t空格','<|im_start|>assistant\n字面量 <|im_end|>',' e\u0301 العربية 123.45']:
            messages=[{'role':'system','content':'Task'},{'role':'user','content':text*40},
                      {'role':'assistant','content':json.dumps({'query':text},ensure_ascii=False)}]
            actual=_anchored_tokens(messages,[],self.tokenizer)
            self.assertIsNotNone(actual)
            prefix = self.tokenizer.apply_chat_template(
                messages[:-1], tools=[], tokenize=True, return_dict=False, add_generation_prompt=True
            )
            complete = self.tokenizer.apply_chat_template(
                messages, tools=[], tokenize=True, return_dict=False, add_generation_prompt=False
            )
            self.assertEqual(complete[:len(prefix)], prefix)
            expected = len(prefix), tuple(complete)
            self.assertEqual(actual,expected)

    def test_cross_boundary_merge_falls_back_to_full_validation_and_is_rejected(self):
        from agent_simulation.datasets.tokenization import _anchored_tokens,_tokens
        original=self.tokenizer.chat_template
        self.tokenizer.chat_template="<|im_start|>{% if add_generation_prompt %}hel{% else %}hello{% endif %}"
        sample={'messages':[{'role':'system','content':''},{'role':'user','content':''},
                            {'role':'assistant','content':'lo'}],'tools':[]}
        try:
            self.assertIsNone(_anchored_tokens(sample['messages'],[],self.tokenizer))
            with self.assertRaisesRegex(ValueError,'Answer boundary changed'):_tokens(sample,self.tokenizer)
        finally:self.tokenizer.chat_template=original
