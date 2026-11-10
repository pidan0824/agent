"""Explicit memory actions, exact evidence binding and retirement targets."""
import json
import unittest
from agent_simulation import PROJECT_ROOT
from agent_simulation.agents.memory.protocol import prepare, decode
from agent_simulation.inference.local.qwen import build_prefix_allowed_tokens_fn


class ReflectionGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        from agent_simulation.inference.requests import load_tokenizer
        cls.tokenizer = load_tokenizer(PROJECT_ROOT / 'models/Qwen2.5-3B-Instruct', local_files_only=True)
        cls.tokens = build_token_enforcer_tokenizer_data(cls.tokenizer)

    def setUp(self):
        self.payload = {'actor_role':'customer', 'profile':{'user_id':7},
            'experiences':[{'source_id':f's{i}', 'origin':'observed','time_index':i,
                'content':{'query':'水饺','purchased_items':[{'item_id':i,'seller_id':10+i}], 'clicked_items':[]}}
                for i in range(1,5)], 'new_evidence_ids':['s4'],
            'previous_long_term_memory':[{'memory_id':'customer:7:10', 'generated_step':0,
                                         'content':'Old claim.', 'source_ids':['s1']}]}
        self.visible, self.schema, self.references = prepare(self.payload)

    def allowed(self, value):
        import torch
        from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
        parser = JsonSchemaParser(self.schema, config=CharacterLevelParserConfig(alphabet=self.tokens.tokenizer_alphabet))
        prefix = build_prefix_allowed_tokens_fn(self.tokens, parser)
        sequence = [0]
        for token in self.tokenizer.encode(json.dumps(value), add_special_tokens=False):
            if token not in prefix(0, torch.tensor(sequence)):
                return False
            sequence.append(token)
        return self.tokenizer.eos_token_id in prefix(0, torch.tensor(sequence))

    def test_explicit_operations_bind_sources_and_old_memory(self):
        for action in ('add','revise','retract','no_update'):
            draft = {'action':action}
            if action in ('add','revise'):
                draft['evidence_refs'] = ['E1']
                draft['content'] = 'May prefer item_id 4; observations are limited.'
            if action in ('revise','retract'): draft['memory_refs'] = ['M1']
            self.assertTrue(self.allowed(draft))
            result = decode(json.dumps(draft), self.schema, self.references)
            self.assertEqual(result['action'], action)
            self.assertEqual(result['source_ids'], ['s4'] if action in ('add','revise') else [])
            self.assertEqual('supersedes' in result, action in ('revise','retract'))
            self.assertEqual(bool(result['content']), action in ('add','revise'))
            if result['content']:
                self.assertEqual(result['content'], draft['content'])
                self.assertNotIn('source_ids', result['content'])
                self.assertIn('May prefer item_id 4', result['content'])

    def test_invalid_actions_targets_and_prose_are_not_repaired(self):
        cases = [
            {'evidence_refs':['E4']},
            {'action':'revise','evidence_refs':['E4']},
            {'action':'retract','evidence_refs':['E4'],'memory_refs':['foreign']},
            {'action':'add','evidence_refs':['E9']},
            {'action':'add','evidence_refs':['E1']},
            {'action':'add','evidence_refs':['E1','E2','E3','E4']},
            {'action':'add','evidence_refs':['E4','E4']},
        ]
        for draft in cases:
            with self.subTest(draft=draft):
                with self.assertRaises(ValueError): decode(json.dumps(draft), self.schema, self.references)
        for draft in cases[:-1]: self.assertFalse(self.allowed(draft))

    def test_purchase_index_keeps_product_and_shop_separate_without_mutation(self):
        original = json.dumps(self.payload, sort_keys=True)
        self.assertEqual(self.visible['purchase_evidence'][0],
                         {'item_id':4,'seller_id':14,'evidence_refs':['E1']})
        self.assertEqual(json.dumps(self.payload,sort_keys=True), original)

    def test_duplicate_source_id_between_experience_and_supplement_is_rejected(self):
        duplicate = {**self.payload['experiences'][0], 'content': {'query': 'different'}}
        with self.assertRaisesRegex(ValueError, 'Duplicate reflection evidence source_id'):
            prepare({**self.payload, 'supplemental_evidence': [duplicate]})

    def test_only_no_update_is_available_without_new_evidence(self):
        for events in ([], self.payload['experiences']):
            _, self.schema, self.references = prepare(
                {**self.payload, 'experiences': events, 'new_evidence_ids': []}
            )
            self.assertEqual(len(self.schema['anyOf']), 1)
            self.assertTrue(self.allowed({'action': 'no_update'}))
            self.assertFalse(self.allowed({'action': 'retract', 'memory_refs': ['M1']}))
