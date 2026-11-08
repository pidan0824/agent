"""Check factual equivalence across model encodings and source boundary cases."""
import json
import unittest
from copy import deepcopy
from agent_simulation.agents.prompts.customer import encode_input, decode_input, messages, factor_history
from agent_simulation.agents.prompts.context_format import pack, unpack


class RepresentationTests(unittest.TestCase):
    def test_shared_history_preserves_versions_and_current_list(self):
        old = {'item_id': 8, 'item_title': '保温杯', 'seller_id': 3,
               'seller_name': '小店', 'brand_id': 4, 'brand_name': None,
               'category_level1_id': 5, 'category_level1_name': '日用品',
               'price_raw': None}
        changed = {**old, 'item_title': '新版保温杯', 'price_raw': 800.0}
        state = {'short_term_memory': {'search_history': [
            {'session_id': 1, 'time_index': 5, 'query': '杯子',
             'clicked_items': [old, old], 'purchased_items': []},
            {'request_id': 2, 'search_step': 10, 'settled_step': 11,
             'source': 'simulation:customer_response', 'query': '保温杯',
             'clicked_items': [changed], 'purchased_items': [changed]},
        ]}, 'displayed_items': [{**changed, 'price_raw': 900}],
            'long_term_memory': [{'content': '经验', 'time_index': 4}]}
        before = deepcopy(state)
        compact = factor_history(state)
        memory = compact['short_term_memory']
        self.assertEqual(len(memory['item_attributes']), 2)
        self.assertEqual(len(memory['attribute_sets']['sellers']), 1)
        self.assertEqual(memory['search_history'][0]['clicked_item_refs'], [0, 0])
        self.assertEqual(memory['search_history'][1]['purchased_item_refs'], [1])
        self.assertEqual(compact['displayed_items'], before['displayed_items'])
        self.assertEqual(decode_input(encode_input(state)), before)
        self.assertEqual(state, before)
        self.assertEqual(decode_input(json.dumps(pack(before))), before)

    def test_recent_and_retrieved_snapshots_and_current_price_remain_distinct(self):
        recent = {"session_id": 3, "clicked_items": [{"item_id": 8, "price_raw": 520}], "purchased_items":[]}
        old = {"session_id": 1, "clicked_items": [{"item_id": 8, "price_raw": None}], "purchased_items":[]}
        state = {"short_term_memory":{"search_history":[recent]},
                 "retrieved_context":{"history":{"segments":[{"records":[old]}]}},
                 "displayed_items":[{"item_id":8,"price_raw":600}]}
        compact = factor_history(state)
        self.assertEqual(compact["retrieved_context"]["history"]["segments"][0]["records"][0]["clicked_item_refs"],[1])
        self.assertEqual(decode_input(json.dumps(pack(compact))),state)
        self.assertEqual(decode_input(encode_input(state)),state)

    def test_shared_history_rejects_collisions_and_invalid_references(self):
        state = {'short_term_memory': {'search_history': [
            {'query': '杯子', 'clicked_items': [{'item_id': 1}], 'purchased_items': []}]}}
        compact = factor_history(state)
        compact['short_term_memory']['search_history'][0]['clicked_item_refs'] = [-1]
        with self.assertRaisesRegex(ValueError, 'reference'):
            decode_input(json.dumps(pack(compact)))
        with self.assertRaisesRegex(ValueError, 'Reserved'):
            encode_input({'short_term_memory': {'search_history': [], 'item_attributes': []}})

    def test_preserves_empty_unknown_types_order_and_mixed_schemas(self):
        cases = [[], {}, None, [None, 0, False, 0.0],
                 [{'id': 3, 'price': None}, {'id': 1, 'price': 0}],
                 [{'id': 1}, {'id': 2, 'time': 8}],
                 [{'v': False}, {'v': 0}, {'v': 0.0}],
                 [{'id': 1, 'events': [{'t': 5}, {'t': 9}]},
                  {'id': 2, 'events': []}]]
        for value in cases:
            state = {"value": value}
            before = json.dumps(state, ensure_ascii=False, sort_keys=True)
            self.assertEqual(json.dumps(decode_input(encode_input(state)), ensure_ascii=False,
                                       sort_keys=True), before)
            self.assertEqual(json.dumps(state, ensure_ascii=False, sort_keys=True), before)
        with self.assertRaises(ValueError):
            pack({'$table': 'reserved'})

    def test_tables_are_opt_in_and_must_reduce_json_size(self):
        small = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        rows = [{"item_id": i, "item_title": "保温杯", "price_raw": 900} for i in range(5)]
        original = {"history": rows, "offers": rows, "small": small}
        packed = pack(original, table_paths={("offers",), ("small",)})
        self.assertEqual(packed["history"], rows)
        self.assertEqual(packed["small"], small)
        self.assertIn("$table", packed["offers"])
        self.assertEqual(unpack(packed), original)
        with self.assertRaisesRegex(ValueError, "Reserved"):
            pack([{"$table": 1}, {"$table": 2}], table_paths={()})

    def test_large_exposure_collection_is_separate_from_complete_history(self):
        state = {'profile': {'user_id': 7}, 'query': '打底衫',
                 'short_term_memory': {'search_history': [
                     {'query': '浴室柜', 'time_index': i, 'clicked_items': [], 'purchased_items': []}
                     for i in range(30)]}, 'long_term_memory': [],
                 'displayed_items': [{'item_id': i, 'item_title': '中文商品', 'price_raw': None,
                                      'brand_name': '无品牌'} for i in range(500)]}
        snapshot = deepcopy(state)
        target = {'clicked_item_ids': [2, 4], 'purchased_item_ids': [4, 8]}
        result = messages(state, target, response=True)
        self.assertEqual(decode_input(result[1]['content']), snapshot)
        self.assertEqual(state, snapshot)
        self.assertEqual(json.loads(result[-1]['content']), target)
        table = json.loads(result[1]['content'])['displayed_items']['$table']
        self.assertIsInstance(json.loads(result[1]['content'])['short_term_memory']['search_history'], list)
        self.assertEqual(len(table['rows']), 500)
        self.assertEqual(table['shared']['price_raw'], None)
        self.assertLess(len(result[1]['content']), len(json.dumps(state, ensure_ascii=False)))
        self.assertEqual(unpack(table['shared']), table['shared'])
