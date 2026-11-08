"""Catalog-backed supervised samples and preview exports."""

from agent_simulation.agents.context.shopping import project_customer

from collections import Counter
from contextlib import closing
import json
import unittest

from agent_simulation import PROJECT_ROOT
from agent_simulation.datasets.labels import build_labels
from agent_simulation.datasets.labels import SFTDataset
from agent_simulation.datasets.labels import record_samples
from agent_simulation.tests.fixtures import sample_identity
from agent_simulation.datasets.tokenization import encode_sample
from agent_simulation.training.data_loader import SampleStream
from agent_simulation.agents.prompts.customer import decode_input as decode_customer
from agent_simulation.datasets.labels import export
from agent_simulation.tests.data.fixtures import CatalogFixture


class CatalogSampleTests(CatalogFixture, unittest.TestCase):
    def test_raw_catalog_to_training_stream_matches_every_frozen_record(self):
        tok = PROJECT_ROOT / "models/Qwen2.5-3B-Instruct-tokenizer"
        release = self.root / "rolling_sft"
        build_labels(self.output, release)
        with closing(SFTDataset(release, tok)) as data:
            for role in ("seller", "customer"):
                key = "sample_id" if role == "seller" else "session_id"
                for split in ("train", "test"):
                    with self.subTest(role=role, split=split):
                        path = release / role / split / "labels.parquet"
                        ids = data.catalog.connection.execute(
                            f"SELECT {key} FROM read_parquet(?)", [str(path)]
                        ).fetchall()
                        self.assertTrue(ids)
                        expected = Counter(sample_identity(sample)["payload_sha256"]
                                           for record, in ids for sample in record_samples(data, role, record))
                        actual = Counter()
                        for sample in SampleStream(release, data.tokenizer, role, split):
                            self.assertEqual(sample["split"], split)
                            if role == "customer":
                                self.assertNotIn("history_request_ids", sample["source"])
                                history = sample["input"]["short_term_memory"]["search_history"]
                                self.assertTrue(all(e["time_index"] < sample["key"]["time_index"]
                                                    for e in history))
                            encoded = encode_sample(sample, data.tokenizer)
                            n = encoded["answer_start"]
                            self.assertTrue(0 < n < len(encoded["labels"]))
                            self.assertEqual(encoded["labels"][:n], [-100] * n)
                            self.assertEqual(encoded["labels"][n:], encoded["input_ids"][n:])
                            actual[sample_identity(sample)["payload_sha256"]] += 1
                        self.assertEqual(actual, expected)

    def test_catalog_sft_pair_keeps_cutoff_and_shared_memory(self):
        from agent_simulation.datasets.samples.customer import customer_sample

        query, response = customer_sample(self.catalog, 5)
        self.assertEqual(query["input"]["history_cutoff_exclusive"], 5)
        self.assertEqual(
            query["input"]["short_term_memory"], response["input"]["short_term_memory"]
        )
        self.assertEqual(query["split"], response["split"])
        self.assertNotIn("query", query["input"])
        self.assertEqual(response["input"]["query"], query["target"]["query"])
        self.assertEqual(query["input"]["long_term_memory"], [])
        self.assertEqual(decode_customer(query["messages"][1]["content"]), project_customer(query["input"]))

    def test_catalog_sft_has_one_price_label_and_shop_context(self):
        from agent_simulation.datasets.samples.seller import seller_sample
        from agent_simulation.tests.seller_context_fixture import decision_context
        from agent_simulation.agents.prompts.seller import decode_input

        sample = seller_sample(self.catalog, 10, 1, 5)
        self.assertEqual(sample["target"], {"prices": [{"item_id": 1, "price_raw": 101}]})
        self.assertEqual({p["item_id"] for p in sample["input"]["owned_products"]}, {1})
        self.assertNotIn("shop_summary", sample["input"])
        self.assertEqual(sample["input"]["public_product_pages"], [])
        self.assertEqual(decode_input(sample["messages"][1]["content"]), decision_context(sample["input"]))
        self.assertTrue(
            all(
                r["time_index"] < 5
                for e in sample["input"]["short_term_memory"]
                for r in e["content"]
            )
        )
        sample = seller_sample(self.catalog, 10, 2, 3)
        product, = sample["input"]["owned_products"]
        self.assertIsNone(product["last_observed_price_raw"])
        self.assertEqual(product["last_observed_time_index"], 2)
        self.assertEqual(product["last_valid_reference"]["price_raw"], 102)
        self.assertEqual(product["last_valid_reference"]["time_index"], 1)

    def test_preview_exports_paired_samples_from_the_label_release(self):
        tok = PROJECT_ROOT / "models/Qwen2.5-3B-Instruct-tokenizer"
        release = self.root / "preview_sft"
        output = self.root / "preview"
        build_labels(self.output, release)
        export(release, tok, ["query:5", "response:5", "seller:10:1:5"], output)

        query = json.loads((output / "query_5.json").read_text())
        response = json.loads((output / "response_5.json").read_text())
        seller = json.loads((output / "seller_10_1_5.json").read_text())
        self.assertEqual(query["key"], response["key"])
        self.assertEqual(seller["target"], {"prices": [{"item_id": 1, "price_raw": 101.0}]})
        self.assertEqual(seller["split"], "test")
