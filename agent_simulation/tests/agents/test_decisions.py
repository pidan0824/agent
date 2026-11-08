"""Market statistics and Seller timeout behavior."""

import unittest

from agent_simulation.tests.fixtures import MarketFixture, ScriptedModel, TokenizerFixture, answer
from agent_simulation.agents.seller import SellerAgent
from agent_simulation.data.market.statistics import distribution
from agent_simulation.agents.context.market import simulation_position


class MarketTests(MarketFixture, unittest.TestCase):
    def test_same_category_other_shop_cutoff_and_unknown_latest_price(self):
        result = self.data.market_position(10, 1, 10)
        self.assertEqual(result["peer_products"], 1)
        self.assertEqual(result["price_raw"]["own_value"], 120)  # Not the target at index 10.
        self.assertEqual(result["price_raw"]["observed_products"], 0)  # No carry-forward of 80.
        self.assertIsNone(result["price_raw"]["own_percentile"])
        self.assertEqual(result["purchase_records"]["own_value"], 1)
        self.assertIsNone(result["purchase_records"]["median"])
        self.assertEqual(result["coverage"]["comparable_products"], 0)
        self.assertIsNone(result["purchase_records"]["own_percentile"])
        self.assertNotIn("candidates", result)

    def test_common_window_and_ownership(self):
        result = self.data.market_position(10, 1, 10, 6)
        self.assertIsNone(result["purchase_records"]["own_value"])
        self.assertEqual(result["purchase_records"]["observed_products"], 0)
        with self.assertRaises(PermissionError):
            self.data.market_position(10, 2, 10)
        with self.assertRaises(ValueError):
            self.data.market_position(10, 1, 11)
        batch = self.data.market_positions(10, [1, 3], 10)
        self.assertEqual(batch, {i: self.data.market_position(10, i, 10) for i in [1, 3]})

    def test_decision_prices_match_explicit_common_window(self):
        from agent_simulation.agents.context.market import decision_market
        from agent_simulation.data.market.quotes import RECENT_MARKET_SPAN
        for cutoff in (9, 10):
            compact = self.data.market_positions(10, [1, 3], cutoff, decision_only=True)
            start = max(0, cutoff - RECENT_MARKET_SPAN)
            for item, value in compact.items():
                full = self.data.market_position(10, item, cutoff, start)
                visible = decision_market(value)
                self.assertEqual(visible['recent']['median_price'], full['price_raw']['median'])
                self.assertEqual(visible['recent']['products'], full['recent_price_scope']['products'])
                self.assertEqual(visible['recent']['shops'], full['recent_price_scope']['shops'])
                self.assertEqual(visible['window'], {'start_inclusive': start, 'end_exclusive': cutoff})
                # These cutoffs start at zero: no quote can belong to the older reference window.
                self.assertNotIn('historical_reference', visible)

    def test_quantiles_ties_and_simulated_settlement(self):
        d = distribution([10, 20, 20, 30, None], 20)
        self.assertEqual((d["p25"], d["median"], d["p75"]), (17.5, 20, 22.5))
        self.assertEqual(d["own_percentile"], .5)
        state = self.runtime.snapshot()
        initial = simulation_position(state, 1)
        self.assertEqual(initial["purchase_records"]["own_value"], 0)
        self.assertIn("records settled within the window", initial["scope"])
        state["feedback"].append({"item_id": 2, "purchase_records": 1, "click_records": 1, "exposure_records": 1, "settled_step": 1})
        state["active_prices"]["2"]["price_raw"] = 150
        state["simulation_step"] += 1
        updated = simulation_position(state, 1)
        self.assertEqual(updated["price_raw"]["median"], 150)
        self.assertEqual(updated["purchase_records"]["own_percentile"], 0)
        self.assertEqual(updated["peer_products"], 1)


class WorkflowTests(MarketFixture, unittest.TestCase):
    def seller(self, replies):
        model = ScriptedModel(replies)
        return SellerAgent(self.runtime, model, TokenizerFixture(), price_batch_size=16), model

    def test_model_failure_does_not_retry(self):
        before = self.runtime.snapshot()
        agent, model = self.seller([TimeoutError("fixture timeout"), answer({"prices": [{"item_id": 1, "price_raw": 120}, {"item_id": 3, "price_raw": 200}]})])
        with self.assertRaisesRegex(TimeoutError, "fixture timeout"):
            agent.run(10)
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(self.runtime.snapshot(), before)



if __name__ == "__main__":
    unittest.main()
