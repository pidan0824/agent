"""Traditional pricing comparisons preserve observation cutoff and train-only fitting."""
import unittest

from agent_simulation.evaluation.micro.seller_baselines import evidence_features, FittedBaselines


def observed(index, price):
    return {"time_index": index, "price_raw": price, "price_status": "observed",
            "exposure_records": 10, "click_records": 2, "purchase_records": 1}


class SellerBaselineTests(unittest.TestCase):
    def test_following_market_preserves_past_relative_price_and_rank(self):
        facts = evidence_features([observed(1, 80), observed(2, 100)], [100, 200, 300],
                                  [50, 100, 150], 3, alpha=.5)
        self.assertEqual(facts["rules"]["persistence"], 100)
        self.assertEqual(facts["rules"]["rolling_median"], 90)
        self.assertEqual(facts["rules"]["ewma"], 90)
        self.assertEqual(facts["rules"]["competitor_relative"], 200)
        self.assertEqual(facts["rules"]["competitor_quantile"], 200)
        self.assertFalse(facts["competitor_fallback"])

    def test_same_time_is_rejected_and_missing_market_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Future"):
            evidence_features([observed(3, 999)], [], [], 3)
        facts = evidence_features([observed(1, 100)], [], [], 3)
        self.assertTrue(facts["competitor_fallback"])
        self.assertIsNone(facts["features"]["log_market_median_ratio"])
        self.assertEqual(facts["rules"]["competitor_relative"], 100)

    def test_supervised_models_reject_test_fit_before_importing_sklearn(self):
        with self.assertRaisesRegex(ValueError, "only train"):
            FittedBaselines().fit([{"split": "test"}])

    def test_all_hold_training_does_not_invent_changes(self):
        facts = evidence_features([observed(1, 100)], [], [], 2)
        records = [dict(facts, split="train", target_price_raw=100) for _ in range(4)]
        result = FittedBaselines().fit(records).predict(records)
        self.assertEqual(result["two_stage_linear"].tolist(), [100.] * 4)
        self.assertEqual(result["gbdt"].tolist(), [100.] * 4)
