"""Historical features obey the same visibility boundaries as decisions."""

from copy import deepcopy
import unittest

from agent_simulation.agents.context.customer_features import customer_behavior_features
from agent_simulation.agents.prompts.customer_agentic import (
    agentic_customer_instructions, read_reference, skill_reference_names,
)


def event(index, *, item=1, brand="A", user_id=7):
    product = {"item_id": item, "brand_name": brand, "price_raw": 10,
               "category_level3_name": "cups"}
    return dict(time_index=index, session_id=index, user_id=user_id, query="杯子",
                clicked_items=[product], purchased_items=[product])


class CustomerFeatureTests(unittest.TestCase):
    def test_current_future_outcomes_and_unrelated_fields_cannot_change_features(self):
        history = [event(1), event(2)]
        before = deepcopy(history)
        expected = customer_behavior_features(history, cutoff=3, user_id=7)
        changed = history + [event(3, brand="secret"), event(4, brand="secret")]
        changed[0]["target"] = {"query": "hidden"}
        self.assertEqual(expected, customer_behavior_features(changed, cutoff=3, user_id=7))
        del history[0]["target"]
        self.assertEqual(history, before)

    def test_clocks_and_background_are_separate(self):
        simulated = event(100)
        simulated["settled_step"] = 1
        future = {**simulated, "settled_step": 2}
        background = {"observed_time_index": 1, "statistics_30d": {"orders": 100}}
        features = customer_behavior_features([event(1)], cutoff=3,
            simulated=[simulated, future], simulation_before=2, behavioral_summary=background)
        self.assertEqual(features["observed"]["requests_with_purchases"], 1)
        self.assertEqual(features["simulated"]["requests_with_purchases"], 1)
        self.assertEqual(features["background_snapshot"]["statistics_30d"]["orders"], 100)

    def test_unknowns_do_not_become_zero_traits_and_duplicate_foreign_records_fail(self):
        empty = customer_behavior_features([], cutoff=3)
        self.assertIsNone(empty["observed"]["median_purchased_price_raw"])
        self.assertIsNone(empty["observed"]["consecutive_purchase_brand_overlap_rate"])
        for history in ([event(1), event(1)], [event(1, user_id=8)]):
            with self.assertRaises(ValueError):
                customer_behavior_features(history, cutoff=3, user_id=7)
        with self.assertRaises(ValueError):
            customer_behavior_features([], cutoff=3, behavioral_summary={"observed_time_index": 3})

    def test_theory_switch_does_not_remove_evidence_rules(self):
        self.assertNotIn("decision-types", skill_reference_names(theory=False))
        with self.assertRaises(ValueError):
            read_reference("decision-types", theory=False)
        with self.assertRaises(ValueError):
            read_reference("../../instructions")
        evidence = read_reference("evidence")
        self.assertIn(evidence, agentic_customer_instructions(theory=True))
        self.assertIn(evidence, agentic_customer_instructions(theory=False))


if __name__ == "__main__":
    unittest.main()
