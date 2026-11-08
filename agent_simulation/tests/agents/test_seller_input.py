"""Lossless diagnostic representations must preserve facts, unknowns and provenance."""

import json
import unittest
from agent_simulation.tests.seller_context_fixture import seller_state,memory

from agent_simulation.agents.prompts.context_format import canonical, pack, unpack
from agent_simulation.tests.seller_context_fixture import decision_context
from agent_simulation.agents.prompts.seller import encode_input, decode_input, seller_messages


class RepresentationTests(unittest.TestCase):
    def test_production_prompt_preserves_long_term_experience_and_target_separation(self):
        state = seller_state()
        state['long_term_memory']=[memory('A tentative pricing hypothesis.',step=3)]
        target = {"prices": [{"item_id": 1, "price_raw": 999.5}]}
        messages = seller_messages(state, target)
        visible = decision_context(state)
        self.assertEqual(decode_input(encode_input(visible)), visible)
        self.assertEqual(set(messages[-1]), {"role", "content"})
        self.assertEqual(json.loads(messages[-1]["content"]), target)
        self.assertNotIn("999.5", messages[1]["content"])
        self.assertEqual(messages[:2], seller_messages(state))

    def test_roundtrip_preserves_types_unicode_and_row_order(self):
        rows = [
            {"id": 1, "x": None, "v": False, "zero": 0.0, "s": '中文\\"'},
            {"id": 2, "x": 0, "v": 0, "zero": -0.0, "s": '中文\\"'},
            {"id": 2, "x": 0.0, "v": 0.0, "zero": 0.0, "s": '中文\\"'},
        ]
        packed = pack(rows, table_paths={()})
        self.assertEqual(canonical(unpack(packed)), canonical(rows))
        self.assertNotIn("v", packed["$table"]["shared"])


    def test_encoder_rejects_unprojected_context(self):
        with self.assertRaisesRegex(ValueError, "Expected a projected Seller Evidence View"):
            encode_input(seller_state())

    def test_empty_histories_and_empty_long_term_are_valid(self):
        state = decision_context({**seller_state(), "short_term_memory": []})
        self.assertEqual(decode_input(encode_input(state)), state)


    def test_reserved_markers_and_table_width_fail(self):
        with self.assertRaisesRegex(ValueError, "Reserved"):
            pack({"$table": "literal source"})
        with self.assertRaisesRegex(ValueError, "width"):
            unpack({"$table": {"columns": ["x"], "shared": {}, "rows": [[1, 2]]}})


if __name__ == "__main__":
    unittest.main()
