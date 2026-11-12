"""Catalog tests cover source joins, time boundaries, ownership, and real missing values."""

import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import duckdb
from agent_simulation.tests.data.fixtures import CatalogFixture
from agent_simulation.data.build.catalog import build, verify_catalog


class CatalogTests(CatalogFixture, unittest.TestCase):
    def test_customer_history_preserves_topics_and_excludes_current(self):
        data = self.catalog.customer_history(1, 3)
        self.assertEqual(
            [x["query"] for x in data["short_term_memory"]["search_history"]], ["浴室柜", "痘痘贴"]
        )
        self.assertEqual(data["long_term_memory"], [])
        self.assertNotIn("impressed_item_ids", data["short_term_memory"]["search_history"][0])
        self.assertEqual(
            data["short_term_memory"]["search_history"][0]["clicked_items"][0]["item_id"], 1
        )
        with self.assertRaisesRegex(ValueError, "profile absent"):
            self.catalog.customer_history(2, 3)

    def test_response_labels_not_in_context(self):
        data = self.catalog.request(1)
        self.assertEqual(data["response_target"]["clicked_item_ids"], [1])
        self.assertEqual(data["query_input"]["short_term_memory"]["search_history"], [])
        self.assertNotIn("is_clicked", data["response_input"]["displayed_items"][0])
        self.assertNotIn("query", data["query_input"])

    def test_same_shop_history_missing_price_and_permissions(self):
        data = self.catalog.business(10, [1, 2], 4, 3)
        self.assertEqual(len(data["series"]), 6)
        missing = next(r for r in data["series"] if r["item_id"] == 2 and r["time_index"] == 2)
        self.assertIsNone(missing["price_raw"])
        self.assertEqual(missing["price_status"], "invalid_price")
        with self.assertRaises(PermissionError):
            self.catalog.business(10, [3], 4)
        page = self.catalog.candidates(10, 1, 4)
        self.assertEqual([r["item_id"] for r in page["candidates"]], [3])
        self.assertNotIn("click_records", page["candidates"][0])

    def test_public_history_hides_private_outcomes(self):
        result = self.catalog.public_history(10, [1, 2], [3], 4, 3)
        history = result["products"][0]["reference_price_history"]
        self.assertEqual([r["time_index"] for r in history], [1, 2, 3])
        self.assertTrue(all(set(r) == {"time_index", "price_raw", "price_status"} for r in history))
        with self.assertRaises(PermissionError):
            self.catalog.public_history(10, [1], [2], 4)

    def test_reject_overwrite(self):
        with self.assertRaisesRegex(ValueError, "new catalog"):
            build(self.cfg, self.root / "users.jsonl", self.splits, self.output)

    def test_corrupt_feedback_detected_even_if_totals_preserved(self):
        c = duckdb.connect()
        for name in (
            "products",
            "users",
            "requests",
            "ranking_observations",
            "customer_splits",
            "seller_targets",
        ):
            if name == "customer_splits":
                path = self.splits / "customer_requests.parquet"
            elif name == "seller_targets":
                path = self.splits / "seller_targets/*.parquet"
            else:
                path = self.output / "facts" / (name + ".parquet")
            c.execute(f"CREATE TABLE {name} AS SELECT * FROM read_parquet('{path}')")
        c.execute(
            "UPDATE ranking_observations SET is_clicked=1-is_clicked WHERE session_id=1 AND item_id IN (1,2)"
        )
        with self.assertRaisesRegex(ValueError, "keys or responses differ"):
            verify_catalog(c, self.cfg)
        c.close()


if __name__ == "__main__":
    unittest.main()


class BackgroundRebuildTests(unittest.TestCase):
    def test_rebuild_updates_manifest_to_match_written_background(self):
        from agent_simulation.data.build.catalog import build_customer_background
        from agent_simulation.data.common import sha256
        with tempfile.TemporaryDirectory() as temp, duckdb.connect() as c:
            root = Path(temp)
            (root / 'facts').mkdir()
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'files': {'customer_background': {'sha256': 'old'}},
                                            'dataset': 'unchanged'}))
            c.execute("CREATE TABLE requests AS SELECT 1 session_id,2 user_id,3 time_index,'train' split")
            c.execute("CREATE TABLE source_rows AS SELECT *,[4] recently_clicked_item_ids,"
                      "[5] recently_purchased_item_ids,'{}' user_statistical_features FROM requests")
            for items in ('[4]', '[4,6]'):
                c.execute('UPDATE source_rows SET recently_clicked_item_ids=' + items)
                with patch('agent_simulation.data.build.catalog.raw_ranking', return_value='source_rows'):
                    result = build_customer_background(c, root, root)
                saved = json.loads(manifest.read_text())
                self.assertEqual(saved['files']['customer_background'], result)
                self.assertEqual(saved['dataset'], 'unchanged')
                self.assertEqual(result['sha256'], sha256(root / 'facts/customer_background.parquet'))
                self.assertEqual(result['rows'], 1)
