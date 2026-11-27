"""End-to-end Seller projection keeps decision evidence, scopes and shared inputs."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent_simulation.agents.context.seller_features import seller_behavior_features
from agent_simulation.evaluation.seller_context import observed_seller_context
from agent_simulation.evaluation.seller_quality import check_seller_evidence
from agent_simulation.evaluation.trajectories import FrozenSellerSampleTools
from experiments.cot_sft_pilot import decision_request, seller_case


def fixture():
    history = [dict(seller_id=7, item_id=4, time_index=i, price_raw=p, price_status="observed",
                    exposure_records=1, click_records=0, purchase_records=0)
               for i, p in ((1, 20), (2, 20), (3, 22))]
    state = {"profile": {"seller_id": 7}, "history_cutoff_exclusive": 10,
        "requested_price_item_ids": [4],
        "product_details": [{"item_id": 4, "seller_id": 7, "item_title": "双层杯",
                             "brand_name": "甲", "category_level2_name": "杯子"}],
        "owned_products": [{"item_id": 4, "item_title": "双层杯",
            "last_observed_price_raw": 22, "last_observed_time_index": 3, "price_status": "observed",
            "business_summary": {"observed_time_index": 3, "statistics_30d": {
                "exposures": 100, "clicks": 10, "orders": 2}},
            "market_position": {"window": {"start_inclusive": 0, "end_exclusive": 10},
                "recent": {"median_price": 15, "products": 20},
                "competitor_examples": {"selection": {"eligible_products": 20}, "products": [
                    {"item_id": i, "seller_id": 8, "item_title": "双层杯", "brand_name": "甲",
                     "reference_price_raw": 22, "reference_price_time_index": 3,
                     "price_status": "observed", "purchase_records": 999, "simulation_price": {}}
                    for i in (5, 6, 7, 8)]}}}]}
    features = seller_behavior_features(history, seller_id=7, cutoff=10)
    return state, history, features


class SellerContextTests(unittest.TestCase):
    def test_attributes_business_and_competitors_survive_with_separate_scopes(self):
        state, history, features = fixture()
        before = deepcopy((state, history, features))
        view = observed_seller_context(state, features, history)
        own = view["owned_products"][0]
        self.assertEqual(own["brand_name"], "甲")
        self.assertEqual(own["category_level2_name"], "杯子")
        self.assertEqual(own["business_summary"], state["owned_products"][0]["business_summary"])
        self.assertEqual(view["behavior_features"]["products"][0]["purchase_records"], 0)
        self.assertEqual(own["recent_quote_observations"][-1]["price_raw"], 22)
        peers = own["market_position"]["competitor_examples"]
        self.assertEqual((peers["available"], peers["shown"]), (4, 3))
        self.assertEqual([p["item_id"] for p in peers["products"]], [5, 6, 7])
        self.assertTrue(all("purchase_records" not in p and "simulation_price" not in p for p in peers["products"]))
        self.assertEqual((state, history, features), before)

    def test_invalid_owner_time_and_product_are_rejected(self):
        for part in ("business", "competitor_time", "competitor_owner", "details", "history"):
            state, history, features = fixture()
            own = state["owned_products"][0]
            if part == "business":
                own["business_summary"]["observed_time_index"] = 10
            elif part == "competitor_time":
                own["market_position"]["competitor_examples"]["products"][-1]["reference_price_time_index"] = 10
            elif part == "competitor_owner":
                own["market_position"]["competitor_examples"]["products"][0]["seller_id"] = 7
            elif part == "details":
                state["product_details"][0]["seller_id"] = 99
            else:
                history[0]["seller_id"] = 99
            with self.subTest(part=part), self.assertRaises(ValueError):
                observed_seller_context(state, features, history)

    def test_direct_and_rationale_share_inputs_and_targets_do_not_enter_them(self):
        state, history, features = fixture()
        case = {"stage": "seller", "role": "seller", "context": observed_seller_context(state, features, history),
                "target": {"prices": [{"item_id": 4, "price_raw": 99999}]}}
        direct, cot = (decision_request(case, style) for style in ("direct", "concise"))
        self.assertEqual(direct["messages"][1:], cot["messages"][1:])
        self.assertNotIn("99999", json.dumps(direct))
        changed = deepcopy(case)
        changed["target"]["prices"][0]["price_raw"] = 77777
        self.assertEqual(decision_request(changed, "concise"), cot)

    def test_pilot_uses_the_same_adapter_as_generic_seller(self):
        state, history, _ = fixture()
        sample = {"input": state, "target": {"prices": [{"item_id": 4, "price_raw": 99999}]}}
        catalog = SimpleNamespace(connection=object(), config=object())
        label = dict(seller_id=7, item_id=4, history_cutoff_exclusive=10, previous_price_raw=22,
                     target_price_raw=99999, sample_id="seller:7:4:10", direction="up")
        with patch("agent_simulation.datasets.samples.seller.seller_sample", return_value=sample), \
             patch("agent_simulation.evaluation.trajectories.DataRepository"), \
             patch("agent_simulation.evaluation.trajectories.rows", return_value=history):
            case = seller_case(catalog, label, "fit", "fixture")
            expected = FrozenSellerSampleTools(catalog, sample).initial_context()
        self.assertEqual(case["context"], expected)
        self.assertNotIn("99999", json.dumps(expected))
        self.assertEqual(case["target"]["prices"][0]["price_raw"], 99999)

    def test_per_product_citation_is_checked_against_actual_nested_source(self):
        state, history, features = fixture()
        view = observed_seller_context(state, features, history)
        ref = view["owned_products"][0]["evidence_id"]
        rationale = {"evidence": [f"[{ref}] Historical quote is 22."]}
        action = {"prices": [{"item_id": 4, "price_raw": 22}]}
        def trajectory(context):
            return {"rationale": rationale, "final_action": action, "messages": [
                {"role": "user", "content": json.dumps(context)},
                {"role": "assistant", "content": json.dumps({"rationale": rationale, "action": action})}]}
        self.assertTrue(check_seller_evidence(trajectory(view))["accepted"])
        bad = deepcopy(view)
        bad["owned_products"][0]["last_observed_price_raw"] = 999
        self.assertIn("unknown_evidence_reference", check_seller_evidence(trajectory(bad))["reasons"])
