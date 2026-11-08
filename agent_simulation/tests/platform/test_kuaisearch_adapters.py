"""Test pinned source reuse and inference feature semantics."""

from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agent_simulation import PACKAGE_ROOT
from agent_simulation.data.common import sha256
from agent_simulation.simulation.run import parser, run
from agent_simulation.agents.context.shopping import model_product
from agent_simulation.environment.market import MarketEnvironment
from agent_simulation.platform.pipeline import RecommendationPipeline
from agent_simulation.platform.search import request_context
from agent_simulation.tests.fixtures import MarketFixture
from agent_simulation.platform.pipeline import load_pipeline
from agent_simulation.platform.ranking import OfficialRanking, models
from agent_simulation.platform.ranking.datasets import TrainingDataProcessor
from agent_simulation.platform.recall import DPRRecall


PLATFORM = PACKAGE_ROOT / "platform"


class Batch(dict):
    def to(self, device):
        return self


class Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return Batch(input_ids=torch.ones(1, 2, dtype=torch.long))


class Encoder:
    def __call__(self, **kwargs):
        return SimpleNamespace(last_hidden_state=torch.tensor([[[1.0, 0.5]]]))


def ranking_fixture(model):
    processor = TrainingDataProcessor.__new__(TrainingDataProcessor)
    processor.max_history_len = 2
    processor.user_features = {21: {"gender": "F", "age": "24-30"}}
    processor.item_corpus = {
        i: {"category_level1_id": 1, "category_level2_id": 2, "category_level3_id": 3}
        for i in [11, 12]
    }
    processor.item_emb = np.array([[0.25, 0.75], [0.5, 1.0]], dtype=np.float16)
    processor.item_id2idx = {11: 0, 12: 1}
    tokenizer = Tokenizer()
    ranking = OfficialRanking(model, processor, Encoder(), tokenizer, {1: 11, 2: 12}, {7: 21})
    return ranking, processor, tokenizer


class OfficialTests(unittest.TestCase):
    def test_every_vendored_file_matches_pinned_hash_and_models_import_without_training(self):
        spec = json.loads((PLATFORM / "kuaisearch.json").read_text())
        self.assertEqual(set(spec["source_sha256"]), {"ranking/datasets.py", "ranking/models.py"})
        for name, digest in spec["source_sha256"].items():
            self.assertEqual(sha256(PLATFORM / name), digest, name)
        self.assertEqual(spec["official_commit"], "88a545d6ada3a9a237e2b231b3321ddb83deb0a2")
        self.assertEqual(models.DINModel.__module__, models.__name__)

    def test_dpr_uses_official_cls_inner_product_and_only_market_products(self):
        tokenizer = Tokenizer()
        recall = DPRRecall(
            Encoder(), tokenizer, np.array([[1.0, 2.0], [2.0, 1.0], [100, 100]]), [1, 2, 3]
        )
        result = recall({"query": "a’杯", "market_item_ids": [1, 2]}, 5)
        self.assertEqual(
            result, [{"item_id": 2, "recall_score": 2.5}, {"item_id": 1, "recall_score": 2.0}]
        )
        self.assertEqual(
            tokenizer.calls,
            [
                (
                    ["a'杯"],
                    {
                        "max_length": 64,
                        "truncation": True,
                        "padding": "max_length",
                        "return_tensors": "pt",
                    },
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "lacks"):
            recall({"query": "杯", "market_item_ids": [999]}, 1)

    def test_ranking_reuses_official_collator_and_only_consumes_observed_history(self):
        def model(query, user, item):
            self.assertEqual(query["query_emb"].tolist(), [[1.0, 0.5]] * 2)
            self.assertEqual(user["user_id"].tolist(), [21, 21])
            self.assertEqual(user["gender"].tolist(), [1, 1])
            self.assertEqual(user["age"].tolist(), [3, 3])
            self.assertEqual(user["recent_clicked_items"].tolist(), [[11, 12]] * 2)
            self.assertEqual(item["item_id"].tolist(), [11, 12])
            self.assertEqual(item["title_emb"].tolist(), [[0.25, 0.75], [0.5, 1.0]])
            return torch.tensor([0.0, math.log(3)])

        ranking, processor, tokenizer = ranking_fixture(model)
        context = {
            "user_id": 7,
            "request_id": 100,
            "query": "新搜索词",
            "recently_clicked_item_ids": [999, 1, 2],
        }
        np.testing.assert_allclose(ranking(context, [1, 2]), [0.5, 0.75], rtol=1e-6)
        self.assertEqual(tokenizer.calls[0][0], ["新搜索词"])
        self.assertEqual(tokenizer.calls[0][1]["max_length"], 32)
        self.assertFalse(hasattr(processor, "query_emb"))
        with self.assertRaisesRegex(ValueError, "Product absent"):
            ranking.features(context, [999])

    def test_missing_assets_fail_before_models_or_training_are_loaded(self):
        example = PLATFORM / "assets.example.json"
        with (
            patch("transformers.BertModel.from_pretrained", side_effect=AssertionError("loaded")),
            self.assertRaisesRegex(ValueError, "Missing trained/prepared platform asset"),
        ):
            load_pipeline(example)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            args = parser().parse_args(
                [
                    "--catalog",
                    "unused",
                    "--model",
                    "unused",
                    "--customers",
                    "7",
                    "--output",
                    str(output),
                    "--search-backend",
                    "kuaisearch",
                    "--platform-assets",
                    str(example),
                ]
            )
            with (
                redirect_stdout(io.StringIO()),
                patch(
                    "agent_simulation.simulation.run.load_default_model", side_effect=AssertionError("loaded")
                ),
                self.assertRaisesRegex(ValueError, "Missing trained/prepared platform asset"),
            ):
                run(args)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse((output / "runtime.sqlite").exists())


class FeedbackTests(MarketFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()

        def fixed_model(query_features, user_features, item_features):
            last_click = user_features["recent_clicked_items"][:, -1]
            return (item_features["item_id"] == last_click).float()

        self.ranking, _, _ = ranking_fixture(fixed_model)
        pipeline = RecommendationPipeline(
            lambda context, count: [{"item_id": i, "recall_score": 0.0} for i in [1, 2]],
            self.ranking,
            identity="fixture:fixed-feedback-ranker",
        )
        self.runtime.close()
        self.runtime = MarketEnvironment.create(
            self.path.with_name("feedback.sqlite"),
            self.data,
            [1, 2, 3, 4],
            {7: [], 8: []},
            top_k=2,
            search_pipeline=pipeline,
            search_config={
                "backend": "kuaisearch",
                "candidate_size": 2,
                "pipeline_id": pipeline.identity,
                "reranker_id": pipeline.reranker_id,
            },
        )

    def search(self):
        return self.runtime.execute("customer", 7, "search_products", {"query": "杯子"})

    def test_price_reaches_customer_and_settled_click_changes_next_ranking_input(self):
        self.assertIsNone(self.runtime.search)
        before = self.search()
        before_scores = before["platform_trace"]["ranking"]
        self.runtime.execute(
            "seller", 10, "submit_prices", {"prices": [{"item_id": 1, "price_raw": 333}]}
        )
        self.runtime.execute(
            "customer", 7, "submit_response", {"clicked_item_ids": [], "purchased_item_ids": []}
        )
        after_price = self.search()
        self.assertEqual(after_price["platform_trace"]["ranking"], before_scores)
        displayed = {p["item_id"]: model_product(p) for p in after_price["displayed_items"]}
        self.assertEqual(displayed[1]["price_raw"], 333)
        context = request_context(self.runtime.snapshot(), 7, "杯子")
        self.assertEqual(context["recently_clicked_item_ids"], [])
        self.runtime.execute(
            "customer", 7, "submit_response", {"clicked_item_ids": [2], "purchased_item_ids": [2]}
        )
        context = request_context(self.runtime.snapshot(), 7, "杯子")
        _, user_features, _ = self.ranking.features(context, [1, 2])
        self.assertEqual(user_features["recent_clicked_items"].tolist(), [[0, 12], [0, 12]])
        after_click = self.search()
        self.assertEqual([p["item_id"] for p in before["displayed_items"]], [1, 2])
        self.assertEqual([p["item_id"] for p in after_click["displayed_items"]], [2, 1])
        self.assertNotEqual(after_click["platform_trace"]["ranking"], before_scores)
        other = request_context(self.runtime.snapshot(), 8, "杯子")
        self.assertEqual(other["recently_clicked_item_ids"], [])

    def test_purchase_without_click_is_preserved_without_inventing_a_ranking_click(self):
        before = self.search()
        self.runtime.execute(
            "customer", 7, "submit_response", {"clicked_item_ids": [], "purchased_item_ids": [2]}
        )
        state = self.runtime.snapshot()
        event = state["customers"]["7"]["simulation_history"][-1]
        self.assertEqual([p["item_id"] for p in event["purchased_items"]], [2])
        self.assertEqual(request_context(state, 7, "杯子")["recently_clicked_item_ids"], [])
        self.assertEqual(
            self.search()["platform_trace"]["ranking"], before["platform_trace"]["ranking"]
        )


if __name__ == "__main__":
    unittest.main()
