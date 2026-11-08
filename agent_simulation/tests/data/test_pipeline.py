"""Observed prices, source projections and historical visibility."""

from contextlib import closing
import json
from pathlib import Path
import tempfile
import unittest
from agent_simulation.data.common import Config, connect, bind_config, copy_parquet
from agent_simulation.data.build.seller_tables import observations_sql, business_sql, sources, ranking, business
from agent_simulation.data.market.products import public_candidates, history


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kuaisearch_pipeline_test_")
        self.root = Path(self.temporary.name)
        self.cfg = Config("test-only", self.root / "output", {}, 2)

    def tearDown(self):
        self.temporary.cleanup()

    def connection_with_prices(self, prices):
        c = connect(self.cfg)
        c.execute("CREATE TABLE profiles AS SELECT 10::INTEGER item_id,1::INTEGER seller_id")
        c.execute(
            """CREATE TABLE ranking(item_id INTEGER,time_index BIGINT,price DOUBLE,is_clicked INTEGER,
            is_purchased INTEGER,split VARCHAR,shows_30d DOUBLE,clicks_30d DOUBLE,orders_30d DOUBLE)"""
        )
        c.executemany(
            "INSERT INTO ranking VALUES(10,5,?,0,0,'train',1,0,0)", [(p,) for p in prices]
        )
        c.execute("CREATE TABLE observations AS " + observations_sql())
        return c

    def test_unchanged_observed_price_and_zero_feedback(self):
        c = self.connection_with_prices([100.0, 100.0])
        row = c.execute(
            "SELECT price_raw,price_status,exposure_records,click_records FROM ("
            + business_sql()
            + ")"
        ).fetchone()
        self.assertEqual(row, (100.0, "observed", 2, 0))
        c.close()

    def test_conflicting_prices_are_not_labels(self):
        c = self.connection_with_prices([100.0, 120.0])
        self.assertEqual(
            c.execute("SELECT price_raw,price_status FROM (" + business_sql() + ")").fetchone(),
            (None, "conflicting_prices"),
        )
        c.close()

    def test_invalid_or_missing_prices_remain_unknown(self):
        for prices in ([None], [0], [-1], [float("inf")], [100.0, None]):
            with self.subTest(prices=prices):
                c = self.connection_with_prices(prices)
                self.assertEqual(
                    c.execute(
                        "SELECT price_raw,price_status FROM (" + business_sql() + ")"
                    ).fetchone(),
                    (None, "invalid_price"),
                )
                c.close()

    def test_mixed_null_historical_feature_is_detected(self):
        c = self.connection_with_prices([100.0])
        c.execute("INSERT INTO ranking VALUES(10,5,100,0,0,'train',NULL,0,0)")
        self.assertEqual(
            c.execute("SELECT context_variants FROM (" + observations_sql() + ")").fetchone()[0], 2
        )
        c.close()

    def test_configuration_mixing_rejected(self):
        bind_config(self.cfg)
        with self.assertRaisesRegex(ValueError, "different build configuration"):
            bind_config(Config("different-dataset", self.cfg.output, {}, 2))

    def test_invalid_or_mixed_official_split_cannot_publish_business(self):
        with closing(self.connection_with_prices([100.0])) as c:
            copy_parquet(c, "SELECT * FROM profiles", self.cfg.work / "profiles.parquet")
            copy_parquet(c, "SELECT 1 AS session_id", self.cfg.work / "requests.parquet")
            ranking_path = self.cfg.work / "ranking.parquet"
            for split, error in (("validation", "official split"), ("test", "splits at item/time")):
                with self.subTest(split=split):
                    copy_parquet(c, f"""SELECT 1 AS session_id, * FROM ranking
                        UNION ALL SELECT 2 AS session_id, * REPLACE('{split}' AS split) FROM ranking""",
                        ranking_path)
                    with closing(connect(self.cfg)) as builder, self.assertRaisesRegex(ValueError, error):
                        business(builder, self.cfg)
                    self.assertFalse((self.cfg.data / "seller_business").exists())
                    ranking_path.unlink()

    def test_end_to_end_noncontiguous_ids_empty_buckets_and_cutoff(self):
        items = []
        for item, seller, category in [(10, 1, 7), (20, 2, 7), (30, 1, 7), (2**31 + 40, 3, 8), (50, 4, 0)]:
            items.append(
                dict(
                    item_id=item,
                    seller_id=seller,
                    seller_name=f"shop{seller}",
                    item_title=f"商品{item}",
                    brand_id=1,
                    brand_name="test",
                    category_level1_name="one",
                    category_level2_name="two",
                    category_level1_id=1,
                    category_level2_id=2,
                    category_level3_id=category,
                    category_level3_name="UNKNOWN" if category == 0 else str(category),
                )
            )
        requests = [
            dict(
                session_id=100,
                time_index=5,
                impressed_item_ids=[10, 20, 30, 2**31 + 40, 50],
                clicked_item_ids=[10],
                purchased_item_ids=[],
                split="train",
            ),
            dict(
                session_id=9000000,
                time_index=9,
                impressed_item_ids=[10, 20],
                clicked_item_ids=[20],
                purchased_item_ids=[20],
                split="test",
            ),
        ]
        rank = []
        for request in requests:
            for item in request["impressed_item_ids"]:
                rank.append(
                    dict(
                        session_id=request["session_id"],
                        target_item_id=item,
                        time_index=request["time_index"],
                        target_item_price=100.0 if request["time_index"] == 5 else 120.0,
                        is_clicked=int(item in request["clicked_item_ids"]),
                        is_purchased=int(item in request["purchased_item_ids"]),
                        split=request["split"],
                        target_item_statistical_features=dict(
                            item_show_cnt_30d_hist=1,
                            item_click_cnt_30d_hist=0,
                            item_order_cnt_30d_hist=0,
                        ),
                    )
                )
        inputs = {}
        for name, records in [("items", items), ("recall", requests), ("ranking", rank)]:
            path = self.root / f"{name}.jsonl"
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            inputs[name] = {"path": str(path)}
        cfg = Config("test-only", self.cfg.output, inputs, 2)
        bind_config(cfg)
        for stage in (sources, ranking, business):
            c = connect(cfg)
            stage(c, cfg)
            c.close()
        c = connect(cfg)
        self.assertEqual(public_candidates(c, cfg, 10, 5), (None, []))
        _, peers = public_candidates(c, cfg, 10, 9)
        self.assertEqual([r["item_id"] for r in peers], [20])
        self.assertEqual(peers[0]["reference_price_raw"], 100.0)
        self.assertEqual(len(history(c, cfg, 10, 9)), 1)
        self.assertEqual(public_candidates(c, cfg, 50, 10)[1], [])
        c.close()
        c = connect(cfg)
        with self.assertRaises(FileExistsError):
            business(c, cfg)
        c.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
