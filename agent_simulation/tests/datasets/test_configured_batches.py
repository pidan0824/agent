"""Customer Response uses the complete prompt and reserved output to plan batches."""

import unittest

from agent_simulation.agents.prompts.customer import decode_input, messages
from agent_simulation.environment.batching import response_batches
from agent_simulation.inference.requests import ContextCapacityError, input_token_budget, prepare_request
from agent_simulation.tests.fixtures import TokenizerFixture


class ConfiguredBatchTests(unittest.TestCase):
    def test_entire_display_over_64_items_fits_and_keeps_its_prompt(self):
        state = {"displayed_items": [{"item_id": i} for i in range(129)], "history": ["frozen"]}
        (batch, prompt), = response_batches(state, TokenizerFixture())
        self.assertEqual(batch, state)
        self.assertEqual(prompt, messages(state, response=True))
        batch["history"].append("changed")
        self.assertEqual(state["history"], ["frozen"])

    def test_long_text_and_output_reserve_split_a_short_display(self):
        tokenizer = TokenizerFixture()
        state = {
            "displayed_items": [{"item_id": i, "item_title": chr(97 + i) * 14000} for i in range(4)],
            "short_term_memory": {"search_history": [{"query": "b" * 8000}]},
        }
        self.assertEqual(len(response_batches(state, tokenizer)), 1)
        reserve = 26000
        batches = response_batches(state, tokenizer, max_output_tokens=reserve)
        self.assertEqual(len(batches), 4)
        self.assertEqual([p for batch, _ in batches for p in batch["displayed_items"]],
                         state["displayed_items"])
        for index, (batch, prompt) in enumerate(batches, 1):
            self.assertEqual(batch["candidate_batch"], {"index": index, "count": 4, "request_item_count": 4})
            self.assertEqual(prompt, messages(batch, response=True))
            visible = decode_input(prompt[1]["content"])
            self.assertEqual(visible["short_term_memory"], state["short_term_memory"])
            self.assertEqual(visible["candidate_batch"], {"request_item_count": 4})
            _, count = prepare_request(prompt, [], tokenizer, max_output_tokens=reserve)
            self.assertLessEqual(count, input_token_budget(reserve))

    def test_empty_display_and_impossible_context(self):
        tokenizer = TokenizerFixture()
        state = {"displayed_items": []}
        (batch, _), = response_batches(state, tokenizer)
        self.assertEqual(batch, state)
        with self.assertRaises(ContextCapacityError):
            response_batches({"displayed_items": [{"item_id": 1, "item_title": "a" * 140000}]}, tokenizer)
        with self.assertRaises(ContextCapacityError):
            response_batches({**state, "history": ["a" * 140000]}, tokenizer)
