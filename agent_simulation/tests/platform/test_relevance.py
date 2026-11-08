"""Check relevance denominators, score identity and absence of invented labels."""

import unittest
from agent_simulation.evaluation.micro.relevance import evaluate_pairs, summarize_lists, auc_metrics


class RelevanceTests(unittest.TestCase):
    def test_auc_ties_and_average_precision(self):
        roc, ap = auc_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        self.assertEqual(roc, 0.75)
        self.assertAlmostEqual(ap, 5 / 6)
        self.assertEqual(auc_metrics([0, 1], [0.5, 0.5]), (0.5, 0.5))
        self.assertEqual(auc_metrics([0, 0], [0.1, 0.2]), (None, None))

    def test_exact_pair_identity_and_duplicate_rows(self):
        refs = [
            {"query": "杯子", "text": "便携杯", "label": 1},
            {"query": "杯子", "text": "鞋", "label": 0},
        ]
        predictions = [
            dict(source_row=i, query=r["query"], text=r["text"], score=1 - i)
            for i, r in enumerate(refs)
        ]
        for bad in (False, 0.0):
            with self.subTest(source_row=bad), self.assertRaisesRegex(ValueError, "source_row"):
                evaluate_pairs(refs, [{**predictions[0], "source_row": bad}, predictions[1]])
        result = evaluate_pairs(refs, predictions, 0.5)
        self.assertEqual(result["macro_f1"], 1)
        self.assertEqual(result["roc_auc"], 1)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            evaluate_pairs(refs, [{**predictions[0], "text": "其他商品"}, predictions[1]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_pairs(refs, [predictions[0], predictions[0]])
        with self.assertRaisesRegex(ValueError, "binary"):
            evaluate_pairs([{**refs[0], "label": 3}, refs[1]], predictions)

    def test_unscored_list_is_unknown_not_irrelevant(self):
        rows = [
            {"request_id": "a", "item_id": 1, "score": None},
            {"request_id": "a", "item_id": 2, "score": None},
        ]
        result = summarize_lists(rows)
        self.assertEqual(result["score_coverage"], 0)
        self.assertEqual(result["request_score_coverage"], {"a": 0})
        self.assertIsNone(result["mean_estimated_score_per_request"])
        rows[0]["score"] = 0.8
        rows.append({"request_id": "b", "item_id": 1, "score": 0.4})
        result = summarize_lists(rows)
        self.assertAlmostEqual(result["score_coverage"], 2 / 3)
        self.assertEqual(result["request_score_coverage"], {"a": 0.5, "b": 1})
        self.assertEqual(result["request_estimated_scores"], {"a": 0.8, "b": 0.4})
        self.assertAlmostEqual(result["mean_estimated_score_per_request"], 0.6)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_lists(rows + rows)


if __name__ == "__main__":
    unittest.main()
