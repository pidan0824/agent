"""Random wiring backend remains reproducible and obeys market price snapshots."""

from contextlib import closing
from agent_simulation.environment.market import MarketEnvironment
import json
import unittest
from copy import deepcopy

from agent_simulation.tests.fixtures import MarketFixture
from agent_simulation.platform.search import random_list


class RandomSearchTests(MarketFixture, unittest.TestCase):
    def enable(self):
        state = self.runtime.snapshot()
        state["search_config"] = {"backend": "random", "seed": 123, "candidate_size": 3}
        state["top_k"] = 2
        with self.runtime.db:
            self.runtime.db.execute("UPDATE state SET value=? WHERE id=1", (json.dumps(state),))
        return state

    def test_random_market_opens_and_searches_without_a_title_index(self):
        self.assertIsNotNone(self.runtime.search)
        self.enable()
        with closing(MarketEnvironment(self.path, self.data)) as reopened:
            self.assertIsNone(reopened.search)
            result = reopened.execute("customer", 7, "search_products", {"query": "杯子"})
            self.assertEqual(result["backend"], "random_recall_random_ranking")

    def test_replay_candidates_order_and_no_hidden_label_access(self):
        state = self.enable()
        expected = random_list(state, 7, "unrelated query")
        self.assertEqual(expected, random_list(deepcopy(state), 7, "unrelated query"))
        self.assertEqual(len(expected[0]), 2)
        self.assertEqual(len(set(expected[1])), 3)
        self.assertTrue(set(expected[0]) <= set(expected[1]) <= {1, 2, 3, 4})
        # Price, title and hidden outcomes cannot steer a random wiring backend.
        other = deepcopy(state)
        other["simulation_step"] += 10
        other["customers"]["8"]["simulation_history"].append({"query": "other shopper"})
        other["active_prices"] = {}
        other["products"] = {key: {} for key in state["products"]}
        self.assertEqual(expected, random_list(other, 7, "unrelated query"))
        result = self.runtime.execute("customer", 7, "search_products", {"query": "unrelated query"})
        self.assertEqual(result["backend"], "random_recall_random_ranking")
        self.assertEqual(result["candidate_item_ids"], expected[1])
        self.assertEqual([p["item_id"] for p in result["displayed_items"]], expected[0])

    def test_settlement_uses_displayed_prices_and_only_exposed_products(self):
        self.enable()
        listing = self.runtime.execute("customer", 7, "search_products", {"query": "test"})
        product = listing["displayed_items"][0]
        before = product["price_raw"]
        self.runtime.execute("seller", product["seller_id"], "submit_prices",
                             {"prices": [{"item_id": product["item_id"], "price_raw": 777}]})
        self.runtime.execute("customer", 7, "submit_response",
                             {"clicked_item_ids": [], "purchased_item_ids": [product["item_id"]]})
        state = self.runtime.snapshot()
        self.assertEqual(len(state["feedback"]), 2)
        feedback = next(e for e in state["feedback"] if e["item_id"] == product["item_id"])
        self.assertEqual(feedback["price_raw"], before)
        self.assertEqual(feedback["purchase_records"], 1)
        self.assertEqual(state["active_prices"][str(product["item_id"])]["price_raw"], 777)
        self.assertFalse(state["pending"])


if __name__ == "__main__":
    unittest.main()
