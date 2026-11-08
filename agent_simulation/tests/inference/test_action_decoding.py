"""Test Qwen action grammar and JSON output constraints."""

from agent_simulation import PROJECT_ROOT

import json
from types import SimpleNamespace
import unittest
from unittest.mock import call, patch

from agent_simulation.agents.decisions import action_output
from agent_simulation.inference.local.qwen import action_parser
from agent_simulation.inference.local.qwen import QwenLocal
from agent_simulation.inference.client import output_schema


class ActionGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        from agent_simulation.inference.requests import load_tokenizer

        cls.tokenizer = load_tokenizer(
            PROJECT_ROOT / "models/Qwen2.5-3B-Instruct-tokenizer",
            local_files_only=True,
        )
        cls.tokens = build_token_enforcer_tokenizer_data(cls.tokenizer)

    def allowed(self, raw, stage, control=False, batch_ids=None, displayed_ids=None, allow_unknown=False):
        from agent_simulation.inference.local.qwen import build_prefix_allowed_tokens_fn
        import torch

        output = action_output(stage, allow_control_actions=control,
                               displayed_ids=displayed_ids if displayed_ids is not None else list(range(1, 56)),
                               price_ids=batch_ids, allow_unknown_prices=allow_unknown)
        prefix = build_prefix_allowed_tokens_fn(
            self.tokens,
            action_parser(
                output_schema({"action_output": output}),
                2048,
                self.tokens.tokenizer_alphabet,
            ),
        )
        sequence = [0]
        for token in self.tokenizer.encode(raw, add_special_tokens=False):
            if token not in prefix(0, torch.tensor(sequence)):
                return False
            sequence.append(token)
        return self.tokenizer.eos_token_id in prefix(0, torch.tensor(sequence))

    def test_chinese_query_and_exact_keys(self):
        self.assertTrue(self.allowed('{"query":"轻便保温杯"}', "query"))
        for raw in (
            '>{"query":"杯"}',
            '{}\n{"query":"杯"}',
            '{"query_string":"杯"}',
            "Here is my answer:",
        ):
            self.assertFalse(self.allowed(raw, "query"))

    def test_many_choices_not_capped_at_twenty(self):
        raw = json.dumps({"clicked_item_ids": list(range(1, 56)), "purchased_item_ids": []})
        self.assertTrue(self.allowed(raw, "response"))
        # Generation is restricted to this batch; final business validation still runs.
        self.assertFalse(
            self.allowed('{"clicked_item_ids":[99999],"purchased_item_ids":[]}', "response")
        )

    def test_chinese_query_and_free_string_cache(self):
        raw = '{"query":"通勤保温杯偏好"}'
        self.assertTrue(self.allowed(raw, "query"))
        parser = action_parser(
            output_schema({"action_output": action_output("query")}), 2048, self.tokens.tokenizer_alphabet
        )
        prefix = raw.split("通勤")[0]
        for character in prefix:
            parser = parser.add_character(character)
        self.assertEqual(parser.shortcut_key()[0], "json_freetext")

    def test_response_ids_are_bound_to_this_batch_not_history_or_other_batches(self):
        for item, allowed in ((5715820, True), (3590059, True), (5715815, False), (571582, False)):
            raw = json.dumps({"clicked_item_ids": [item], "purchased_item_ids": []})
            self.assertEqual(self.allowed(raw, 'response', displayed_ids=[3590059, 5715820]), allowed)
        self.assertTrue(self.allowed('{"clicked_item_ids":[],"purchased_item_ids":[]}',
                                     'response', displayed_ids=[]))
        self.assertFalse(self.allowed('{"clicked_item_ids":[1],"purchased_item_ids":[]}',
                                      'response', displayed_ids=[]))
        with self.assertRaisesRegex(ValueError, 'displayed IDs'):
            action_output('response')

    def test_seller_json_without_explanation_or_tool_envelope(self):
        raw = '{"prices":[{"item_id":1,"price_raw":780}]}'
        self.assertTrue(self.allowed(raw, "seller", batch_ids=[1]))
        for invalid in ("I will change prices. " + raw, raw[:-1],
                        '<tool_call>{"name":"submit_prices","arguments":' + raw + '}</tool_call>'):
            self.assertFalse(self.allowed(invalid, "seller", batch_ids=[1]))

    def test_runtime_controls_do_not_leak_into_sft(self):
        self.assertFalse(self.allowed('{"action":"exit"}', "query"))
        self.assertTrue(self.allowed('{"action":"exit"}', "query", control=True))
        self.assertFalse(self.allowed('{"action":"no_action"}', "seller", batch_ids=[7]))
        self.assertFalse(self.allowed('{"action":"no_action"}', "seller", control=True, batch_ids=[7]))
        self.assertFalse(self.allowed('{"action":"exit"}', "seller", control=True, batch_ids=[7]))

    def test_seller_supervision_decoding_matches_exact_target_and_runtime_can_keep_unknown(self):
        def permits(prices, *, allow_unknown=False):
            return self.allowed(json.dumps({"prices": prices}), "seller",
                                batch_ids=[7], allow_unknown=allow_unknown)

        self.assertTrue(permits([{"item_id": 7, "price_raw": 100}]))
        for prices in ([], [{"item_id": 8, "price_raw": 100}], [{"item_id": 7, "price_raw": None}],
                       [{"item_id": 7, "price_raw": 1}] * 2):
            self.assertFalse(permits(prices))
        self.assertTrue(permits([{"item_id": 7, "price_raw": None}], allow_unknown=True))

    def test_seller_batch_constrains_count_and_visible_ids(self):
        def permits(ids):
            args = {"prices": [{"item_id": i, "price_raw": 780} for i in ids]}
            return self.allowed(json.dumps(args), "seller", batch_ids=[1, 3])

        self.assertTrue(permits([1, 3]))
        self.assertFalse(permits([]))
        self.assertFalse(permits([1]))
        self.assertFalse(permits([1, 9]))


class DecodingDispatchTests(unittest.TestCase):
    def test_unknown_role_is_not_silently_run_on_base_model(self):
        model = object.__new__(QwenLocal)
        model.role_adapters = ("customer",)
        with self.assertRaisesRegex(ValueError, "Unknown inference agent role"):
            model._adapter_args([{"agent_role": "costumer"}])
        self.assertEqual(model._adapter_args([{"agent_role": "seller"}, {}]),
                         {"adapter_names": ["__base__", "__base__"]})

    def test_plain_requests_and_conflicting_contracts_do_not_build_a_grammar(self):
        model = object.__new__(QwenLocal)
        self.assertEqual(model._decoding({}), {})
        with self.assertRaisesRegex(ValueError, "one output contract"):
            model._decoding({"action_output": {}, "json_schema": {}})

    def test_tokenizer_metadata_is_shared_but_each_request_gets_its_own_parser(self):
        model = object.__new__(QwenLocal)
        model.tokenizer = object()
        model.max_new_tokens = 100
        model._grammar_tokenizer = None
        metadata = SimpleNamespace(tokenizer_alphabet="abc{}[]:,\"0123456789 ")
        integration = "lmformatenforcer.integrations.transformers."
        with patch(integration + "build_token_enforcer_tokenizer_data", return_value=metadata) as build, \
             patch("agent_simulation.inference.local.qwen.build_prefix_allowed_tokens_fn") as wrap, \
             patch("agent_simulation.inference.local.qwen.action_parser") as action:
            decision = {"const": {"query": "杯"}}
            model._decoding({"action_output": {"final_schemas": [decision]}, "tools": []})
            model._decoding({"json_schema": {"type": "object"}, "memory_generation_method": "fixture"})
            model._decoding({"json_schema": {"type": "object"}})
            build.assert_called_once_with(model.tokenizer)
            self.assertEqual(action.call_args_list, [
                call(schema, 100, metadata.tokenizer_alphabet)
                for schema in (decision, {"type": "object"}, {"type": "object"})
            ])
            self.assertEqual(wrap.call_count, 3)


if __name__ == "__main__":
    unittest.main()
