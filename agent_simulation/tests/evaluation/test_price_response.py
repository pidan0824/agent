from copy import deepcopy
import unittest

from agent_simulation.agents.prompts.customer import messages, decode_input
from agent_simulation.evaluation.micro.price_response import compare_prices, observed_price_gradient, population


class PriceResponseTest(unittest.TestCase):
    def test_repeated_low_price_does_not_hide_observed_high_price(self):
        report = observed_price_gradient({5: [(780, 1, 1), (780, 1, 0), (936, 0, 0)]})
        self.assertEqual(report["comparable_items"], 1)
        self.assertEqual(report["exposure_counts_by_item"], {5: {"low": 2, "high": 1}})
        self.assertEqual(report["high_minus_low_purchase_rate"], {"prediction": -1, "target": -0.5})

    def test_focal_product_is_independent_of_response_partition(self):
        items = [{"item_id": i, "price_raw": 100} for i in range(1, 10)]
        ok = {"status": "ok", "prediction": {"purchased_item_ids": []}}
        def trial(groups, seed):
            samples = []
            for i, group in enumerate(groups, 1):
                state = {"displayed_items": group}
                sid = "response:7" + (f":batch:{i}" if len(groups) > 1 else "")
                samples.append({"sample_id": sid, "input": state, "messages": messages(state, {}, response=True)})
            return compare_prices(samples, [ok] * len(samples), None, lambda *_: ok, seed)["item_id"]
        for seed in range(4):
            self.assertEqual(trial([items], seed), trial([items[:4], items[4:]], seed))

    def test_failed_baseline_is_distinct_from_missing_prices(self):
        sample = {"sample_id": "response:7:batch:1", "input": {"displayed_items": []}}
        failed = {"sample_id": sample["sample_id"], "status": "invalid", "error": "bad JSON"}
        trial = compare_prices([sample], [failed], None, None, 1)
        self.assertEqual(trial["status"], "invalid")
        self.assertEqual(trial["failures"], [{"sample_id": sample["sample_id"], "error": "bad JSON"}])
        skipped = compare_prices([sample], [{"status": "ok"}], None, None, 1)
        self.assertEqual(skipped["status"], "ineligible")

    def test_only_visible_price_changes_and_invalid_arms_are_excluded(self):
        state = {"displayed_items": [{"item_id": 5, "price_raw": 780}, {"item_id": 6, "price_raw": None}],
                 "short_term_memory": {"search_history": [
                     {"session_id": 6, "time_index": 10, "query": "杯子",
                      "clicked_items": [], "purchased_items": [{"item_id": 5, "price_raw": 750}]}
                 ]}}
        sample = {"sample_id": "response:7", "input": state,
                  "messages": messages(state, {}, response=True)}
        original = deepcopy(sample)
        visible = decode_input(sample["messages"][1]["content"])
        baseline = {"stage": "response", "status": "ok",
                    "prediction": {"purchased_item_ids": []}, "target": {"purchased_item_ids": [5]}}
        seen = []
        def fake(variant, model):
            self.assertEqual(variant["messages"][-1], sample["messages"][-1])
            view = decode_input(variant["messages"][1]["content"])
            price = view["displayed_items"][0]["price_raw"]
            seen.append(price)
            view["displayed_items"][0]["price_raw"] = 780
            self.assertEqual(view, visible)
            if price > 780:
                return {"status": "invalid", "error": "invalid test response"}
            return {"status": "ok", "prediction": {"purchased_item_ids": [5]}}
        trial = compare_prices([sample], [baseline], None, fake, seed=1)
        self.assertEqual(sample, original)
        self.assertEqual(seen, [624, 936])
        self.assertEqual([a["purchased"] for a in trial["arms"]], [True, False, None])
        self.assertEqual([a["purchased_item_ids"] for a in trial["arms"]], [[5], [], None])
        report = population([baseline], {5: "14"}, [trial])
        self.assertEqual(report["valid_pairs"], 0)
        self.assertEqual(report["request_purchase_rates"], [None, None, None])
        self.assertIsNone(report["purchase_distributions"]["item"]["total_variation"])
        self.assertEqual(report["purchase_distributions"]["category"]["counts"]["target"], {"14": 1})

    def test_full_request_purchases_include_unmodified_batches(self):
        samples = []
        for index, items in enumerate((
            [{"item_id": 7, "price_raw": None}],
            [{"item_id": 5, "price_raw": 100}, {"item_id": 6, "price_raw": None}],
        ), 1):
            state = {"displayed_items": items}
            samples.append({"sample_id": f"response:7:batch:{index}", "input": state,
                            "messages": messages(state, {}, response=True)})

        def ok(*purchases):
            return {"status": "ok", "prediction": {"purchased_item_ids": list(purchases)}}

        calls = []

        def predict(variant, _model):
            calls.append(variant["sample_id"])
            return ok(5) if variant["input"]["displayed_items"][0]["price_raw"] == 80 else ok()

        trial = compare_prices(samples, [ok(7), ok(6)], None, predict, seed=1)
        self.assertEqual(calls, ["response:7:batch:2:price-0.8", "response:7:batch:2:price-1.2"])
        self.assertEqual([a["purchased_item_ids"] for a in trial["arms"]], [[7, 5], [7, 6], [7]])

        without_other_batch_purchase = compare_prices(samples, [ok(), ok(6)], None, predict, seed=1)
        report = population([], {}, [trial, without_other_batch_purchase])
        self.assertEqual(report["valid_pairs"], 2)
        self.assertEqual(report["focal_purchase_rates"], [1, 0, 0])
        self.assertEqual(report["other_purchase_rates"], [0.5, 1, 0.5])
        self.assertEqual(report["request_purchase_rates"], [1, 1, 0.5])
