import unittest

from agent_simulation.evaluation.customer_request_metrics import response_with_baseline


def row(actual=(), pred=(), purchase=(), bought=(), status="ok"):
    return {"target": {"clicked_item_ids": list(actual), "purchased_item_ids": list(purchase)},
            "prediction": {"clicked_item_ids": list(pred), "purchased_item_ids": list(bought)}
            if status == "ok" else None, "status": status}


class ResponseMetricsTests(unittest.TestCase):
    def test_micro_counts_items_and_macro_counts_scenes(self):
        report = response_with_baseline([
            row((1, 2, 3), (1, 2)), row((4,), (5,)), row(), row((), (9,))])
        click = report["model"]["by_head"]["clicked_item_ids"]
        self.assertEqual((click["tp"], click["fp"], click["fn"]), (2, 2, 2))
        self.assertEqual(click["micro_f1"], .5)
        self.assertAlmostEqual(click["scene_macro_f1_empty_match_one"], .45)
        self.assertEqual(click["positive_scene_macro_f1"], .4)
        self.assertEqual(click["false_action_rate_on_empty"], .5)
        self.assertEqual(click["predicted_action_rate"], .75)
        self.assertEqual(report["always_no_action"]["by_head"]["clicked_item_ids"]["micro_f1"], 0)

    def test_invalid_cannot_receive_empty_credit_or_drop_positive(self):
        r = response_with_baseline([row(status="invalid"), row((1,), status="error")])["model"]
        c = r["by_head"]["clicked_item_ids"]
        self.assertEqual(r["invalid"], 2)
        self.assertEqual(c["fn"], 1)
        self.assertEqual(c["scene_macro_f1_empty_match_one"], 0)
        self.assertEqual(c["empty_accuracy_all_attempts"], 0)
        self.assertEqual(c["predicted_action_rate"], 0)
        self.assertEqual(c["false_action_rate_on_empty"], 0)

    def test_empty_population_ratios_and_purchase_without_click(self):
        r = response_with_baseline([])["model"]
        self.assertIsNone(r["pooled_action_micro_f1"])
        r = response_with_baseline([row(purchase=(7,), bought=(7,))])["model"]
        self.assertIsNone(r["by_head"]["clicked_item_ids"]["micro_f1"])
        self.assertEqual(r["by_head"]["purchased_item_ids"]["micro_recall"], 1)
        self.assertEqual(r["predicted_any_action_rate"], 1)
        self.assertEqual(r["pooled_action_micro_f1"], 1)

    def test_duplicate_ids_do_not_inflate_hits(self):
        c = response_with_baseline([row((1,), (1, 1))])["model"]["by_head"]["clicked_item_ids"]
        self.assertEqual(c["tp"], 1)
        self.assertEqual(c["fp"], 0)

    def test_formal_summary_pools_across_behavior_and_candidate_counts(self):
        from agent_simulation.evaluation.customer_curation import summarize
        records = []
        for index, r in enumerate([row((1, 2), (1,)), row()]):
            records.append({"stage": "response", "mode": "fixed", "policy": "full",
                "variant": "strong", "choice_cnt": index + 2,
                "trajectory": {"target": r["target"], "final_action": r["prediction"],
                               "status": "ok", "trace": {}}})
        summary = summarize(records, [])
        self.assertEqual(len(summary["arms"]), 2)
        self.assertEqual(len(summary["response_metrics"]), 1)
        pooled = summary["response_metrics"][0]
        self.assertAlmostEqual(pooled["model"]["by_head"]["clicked_item_ids"]["micro_f1"], 2 / 3)
        self.assertEqual(pooled["always_no_action"]["by_head"]["clicked_item_ids"]["fn"], 2)
