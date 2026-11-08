"""Test platform stage boundaries and market settlement."""

from copy import deepcopy
import json
import unittest

from agent_simulation.environment.market import MarketEnvironment
from agent_simulation.platform.pipeline import RecommendationPipeline
from agent_simulation.platform.search import request_context
from agent_simulation.tests.fixtures import MarketFixture


class PipelineTests(MarketFixture, unittest.TestCase):
    def bind(self, rerank=None):
        self.seen = []

        def recall(context, count):
            self.seen.append(("recall", deepcopy(context)))
            return [{"item_id": i, "recall_score": 4 - i} for i in [1, 2, 3]][:count]

        def ranking(context, ids):
            self.seen.append(("ranking", list(ids)))
            return [0.2, 0.9, 0.5][: len(ids)]

        options = {"rerank": rerank, "reranker_id": "fixture:reverse"} if rerank else {}
        pipeline = RecommendationPipeline(recall, ranking, identity="fixture:stages", **options)
        state = self.runtime.snapshot()
        state["top_k"] = 2
        state["search_config"] = {
            "backend": "kuaisearch",
            "candidate_size": 3,
            "pipeline_id": pipeline.identity,
            "reranker_id": pipeline.reranker_id,
        }
        with self.runtime.db:
            self.runtime.db.execute("UPDATE state SET value=? WHERE id=1", (json.dumps(state),))
        self.runtime.close()
        self.runtime = MarketEnvironment(self.path, self.data, search_pipeline=pipeline)
        return pipeline

    def test_recall_ranking_rerank_and_price_snapshot_settle_in_one_runtime(self):
        def rerank(context, ranked, top_k):
            self.seen.append(("rerank", [r["item_id"] for r in ranked]))
            self.assertEqual(context["active_prices"]["1"]["price_raw"], 333)
            return [r["item_id"] for r in reversed(ranked)][:top_k]

        self.bind(rerank)
        self.runtime.execute(
            "seller", 10, "submit_prices", {"prices": [{"item_id": 1, "price_raw": 333}]}
        )
        listing = self.runtime.execute("customer", 7, "search_products", {"query": "杯子"})
        self.assertEqual([row[0] for row in self.seen], ["recall", "ranking", "rerank"])
        trace = listing["platform_trace"]
        self.assertEqual(trace["reranker_id"], "fixture:reverse")
        self.assertEqual([r["item_id"] for r in trace["recall"]], [1, 2, 3])
        self.assertEqual([r["item_id"] for r in trace["ranking"]], [2, 3, 1])
        self.assertEqual(trace["reranked_item_ids"], [1, 3])
        self.assertEqual([p["item_id"] for p in listing["displayed_items"]], [1, 3])
        self.runtime.execute(
            "seller", 10, "submit_prices", {"prices": [{"item_id": 1, "price_raw": 444}]}
        )
        self.runtime.execute(
            "customer", 7, "submit_response", {"clicked_item_ids": [1], "purchased_item_ids": [1]}
        )
        state = self.runtime.snapshot()
        feedback = {e["item_id"]: e for e in state["feedback"]}
        self.assertEqual(set(feedback), {1, 3})
        self.assertEqual(feedback[1]["price_raw"], 333)
        self.assertEqual(feedback[1]["purchase_records"], 1)
        context = request_context(state, 7, "再次搜索")
        self.assertEqual(context["recently_clicked_item_ids"], [1])
        self.assertNotIn("purchased_items", context)

    def test_identity_keeps_ranking_order_and_requires_explicit_binding_on_reopen(self):
        self.bind()
        with self.assertRaisesRegex(ValueError, "bound"):
            MarketEnvironment(self.path, self.data)
        listing = self.runtime.execute("customer", 7, "search_products", {"query": "杯子"})
        self.assertEqual([p["item_id"] for p in listing["displayed_items"]], [2, 3])
        self.assertEqual(listing["platform_trace"]["reranker_id"], "identity")

    def test_custom_reranker_requires_its_own_id(self):
        def rerank(context, ranked, top_k):
            return []

        for options in ({}, {"reranker_id": "identity"}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "explicit ID"):
                RecommendationPipeline(None, None, identity="fixture", rerank=rerank, **options)

    def test_equal_ranking_scores_use_item_id_even_if_recall_order_changes(self):
        recalled_ids = [3, 1, 2]

        def recall(context, count):
            return [{"item_id": item_id} for item_id in recalled_ids[:count]]

        def ranking(context, ids):
            return [0.7 if item_id == 3 else 0.8 for item_id in ids]

        pipeline = RecommendationPipeline(recall, ranking, identity="fixture")
        context = {"market_item_ids": [1, 2, 3]}
        for order in ([3, 1, 2], [2, 1, 3]):
            recalled_ids[:] = order
            with self.subTest(order=order):
                selected, trace = pipeline.search(context, 3, 1)
                self.assertEqual(selected, [1])
                self.assertEqual([r["item_id"] for r in trace["recall"]], order)
                self.assertEqual([r["item_id"] for r in trace["ranking"]], [1, 2, 3])

    def test_invalid_rerank_is_audited_without_committing_a_request(self):
        self.bind(lambda context, ranked, top_k: [1, 9999])
        before = self.runtime.snapshot()
        with self.assertRaisesRegex(ValueError, "Re-ranking"):
            self.runtime.execute("customer", 7, "search_products", {"query": "杯子"})
        self.assertEqual(self.runtime.snapshot(), before)
        error = self.runtime.db.execute(
            "SELECT error FROM calls ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
        self.assertIn("Re-ranking", error)

    def test_unknown_backend_does_not_fall_back_to_bm25(self):
        state = self.runtime.snapshot()
        state["search_config"]["backend"] = "unknown"
        with self.runtime.db:
            self.runtime.db.execute("UPDATE state SET value=? WHERE id=1", (json.dumps(state),))
        with self.assertRaisesRegex(ValueError, "Unknown search backend"):
            self.runtime.execute("customer", 7, "search_products", {"query": "杯子"})
        self.assertEqual(self.runtime.snapshot(), state)

    def test_recall_cannot_expand_its_authorized_market(self):
        context = request_context(self.runtime.snapshot(), 7, "杯子")
        original = deepcopy(context)
        def recall(stage, count):
            stage["market_item_ids"].append(9999)
            return [{"item_id": 9999}]
        pipeline = RecommendationPipeline(recall, lambda c, ids: [1], identity="fixture")
        with self.assertRaisesRegex(ValueError, "Recall"):
            pipeline.search(context, 1, 1)
        self.assertEqual(context, original)

    def test_ranking_cannot_reassign_scores_by_reordering_ids(self):
        def ranking(context, ids):
            ids.reverse()
            return [1, 2]
        pipeline = RecommendationPipeline(lambda c, n: [{"item_id": 1}, {"item_id": 2}],
                                          ranking, identity="fixture")
        with self.assertRaisesRegex(ValueError, "candidate ID order"):
            pipeline.search(request_context(self.runtime.snapshot(), 7, "杯子"), 2, 1)

    def test_invalid_candidates_and_nonfinite_scores_are_rejected(self):
        context = request_context(self.runtime.snapshot(), 7, "杯子")
        for candidates, scores, error in [
            ([1, 1], [0.5, 0.7], "Recall"),
            ([9999], [0.5], "Recall"),
            ([1, 2, 3], [0.5, 0.7, 0.9], "Recall"),
            ([1], [float("nan")], "Ranking"),
        ]:
            pipeline = RecommendationPipeline(
                lambda c, n: [{"item_id": i} for i in candidates],
                lambda c, ids: scores,
                identity="fixture:invalid",
            )
            with (
                self.subTest(candidates=candidates, error=error),
                self.assertRaisesRegex(ValueError, error),
            ):
                pipeline.search(context, 2, 2)


if __name__ == "__main__":
    unittest.main()
