"""Customer history keeps factual choices and excludes source exposure lists."""

from copy import deepcopy
import unittest
from agent_simulation.data.history_records import shopping_event
from agent_simulation.agents.context.shopping import context, shopping_history_event
from agent_simulation.agents.retrieval.history import prepare_customer_history
from agent_simulation.agents.prompts.customer import messages
from agent_simulation.agents.actions import validate_response


class CustomerInputTests(unittest.TestCase):
    def setUp(self):
        self.items = {1: {"item_id": 1, "item_title": "中文商品"}}
        self.row = {
            "session_id": 4,
            "time_index": 9,
            "query": "保温杯",
            "impressed_item_ids": [1],
            "clicked_item_ids": [],
            "purchased_item_ids": [1],
        }

    def test_purchase_without_click_is_preserved_and_exposure_is_not_memory(self):
        event = shopping_event(self.row, self.items)
        self.assertEqual(event["clicked_items"], [])
        self.assertEqual(event["purchased_items"], [self.items[1]])
        self.assertNotIn("impressed_item_ids", event)
        event["purchased_items"][0]["item_title"] = "changed"
        self.assertEqual(self.items[1]["item_title"], "中文商品")

    def test_real_label_validation_and_sft_prompt_are_not_rewritten(self):
        target = {"clicked_item_ids": [], "purchased_item_ids": [1]}
        before = deepcopy(target)
        self.assertEqual(validate_response(target, [1]), before)
        sample_messages = messages(
            {"query": "q", "displayed_items": [{"item_id": 1}]}, target, response=True
        )
        self.assertEqual(target, before)
        self.assertNotIn("In this simulation", sample_messages[0]["content"])
        self.assertIn('"clicked_item_ids": []', sample_messages[-1]["content"])

    def test_invalid_history_fails_instead_of_inventing_records(self):
        for change in ({"query": " "}, {"purchased_item_ids": [1, 1]}, {"clicked_item_ids": [2]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                shopping_event({**self.row, **change}, self.items)
        with self.assertRaises(ValueError):
            shopping_event(self.row, {})

    def test_invalid_owner_or_cutoff_rejected_for_recent_history(self):
        for change in ({"user_id": 8}, {"time_index": 10}):
            event = {**shopping_event(self.row, self.items), **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                context({"user_id": 7}, [event], cutoff=10)

    def test_prepared_snapshot_is_the_source_of_both_history_windows(self):
        observed = [{**shopping_event(self.row, self.items), "user_id": 7,
                     "session_id": i, "time_index": 900 + i} for i in range(1, 4)]
        simulated = [{"user_id": 7, "request_id": 1, "settled_step": 2, "query": "本轮",
                      "clicked_items": [], "purchased_items": []}]
        pool = prepare_customer_history(observed, simulated, user_id=7, cutoff=1000,
                                        recent_n=2, simulation_before=3)
        # The redundant caller lists may differ from the validated snapshot.
        state = context({"user_id": 7}, [dict(observed[0], user_id=8)],
                        [dict(simulated[0], settled_step=3)], limit=2, cutoff=1000,
                        simulation_before=3, prepared_history=pool, history_embedder=None)
        recent = state["short_term_memory"]["search_history"]
        self.assertEqual(recent, [observed[-1], simulated[0]])
        segments = state["retrieved_context"]["history"]["segments"]
        self.assertEqual({seg["records"][0]["session_id"] for seg in segments}, {1, 2})
        self.assertTrue(all("match" not in seg for seg in segments))
        recent[0]["query"] = "changed"
        self.assertEqual(pool.recent_history(), [observed[-1], simulated[0]])

    def test_failed_snapshot_update_keeps_previous_recent_and_retrieved_facts(self):
        first = shopping_event(self.row, self.items)
        next_event = dict(first, session_id=5, time_index=10)
        pool = prepare_customer_history([first], (), cutoff=10, recent_n=1, user_id=7)
        for observed in ([first, next_event], [dict(first, query="corrected"), next_event]):
            with self.assertRaisesRegex(ValueError, "Future Customer history"):
                pool.update(observed, [dict(first, settled_step=3)], cutoff=11, simulation_before=3)
            self.assertEqual(pool.cutoff, 10)
            self.assertEqual(pool.recent_history(), [first])
            self.assertEqual(pool.retrieve(embedder=None)["segments"], [])
        pool.update([first, next_event], (), cutoff=11)
        self.assertEqual(pool.recent_history(), [next_event])
        segment, = pool.retrieve(embedder=None)["segments"]
        self.assertEqual(segment["records"], [first])

    def test_simulated_snapshots_keep_unknown_price_reason_when_available(self):
        event = {"query": "q", "clicked_items": [
            {"item_id": 1, "price_raw": None, "price_status": "conflicting_prices",
             "price_provenance": {"private": True}},
            {"item_id": 2, "price_raw": 800, "price_status": "observed"},
            {"item_id": 3, "price_raw": None},
        ]}
        self.assertEqual(shopping_history_event(event)["clicked_items"], [
            {"item_id": 1, "price_raw": None, "price_status": "conflicting_prices"},
            {"item_id": 2, "price_raw": 800},
            {"item_id": 3, "price_raw": None},
        ])

    def test_query_and_response_use_prior_context_without_target_leakage(self):
        state = context({"user_id": 7}, [shopping_event(self.row, self.items)], cutoff=10)
        before = deepcopy(state)
        query = messages(state, {"query": "下次搜索"})
        self.assertNotIn("下次搜索", query[1]["content"])
        displayed = {**state, "query": "本次搜索", "displayed_items": [self.items[1]]}
        response = messages(
            displayed, {"clicked_item_ids": [], "purchased_item_ids": []}, response=True
        )
        self.assertIn("本次搜索", response[1]["content"])
        self.assertEqual(state, before)
        self.assertEqual(query[0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
