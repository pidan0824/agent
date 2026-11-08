"""Public candidate discovery, pagination, detail access and privacy contract tests."""

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from agent_simulation.data.catalog import DataRepository
from agent_simulation.data.common import Config, connect, literal, save_json
from agent_simulation.data.market.quotes import CategorySnapshots
from agent_simulation.data.market.products import (
    PUBLIC_FIELDS,
    public_candidates,
    snapshot,
    candidate_page,
    candidate_detail,
    discover_candidates,
)


class PublicCompetitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cfg = Config("fixture", Path(self.temp.name), {}, 1)
        self.c = connect(self.cfg)
        self.c.execute(
            """CREATE TABLE events(item_id BIGINT,seller_id BIGINT,seller_name VARCHAR,
            item_title VARCHAR,brand_id BIGINT,brand_name VARCHAR,category_level1_id BIGINT,
            category_level1_name VARCHAR,category_level2_id BIGINT,category_level2_name VARCHAR,
            category_level3_id BIGINT,category_level3_name VARCHAR,time_index BIGINT,
            price_raw DOUBLE,price_status VARCHAR,click_records BIGINT,purchase_records BIGINT)"""
        )
        for item, seller, category, timestamp, price, status in [
            (1, 1, 3, 2, 100, "observed"),
            (2, 2, 3, 3, 80, "observed"),
            (2, 2, 3, 8, None, "invalid_price"),
            (3, 1, 3, 3, 90, "observed"),
            (4, 4, 4, 3, 90, "observed"),
            (5, 5, 3, 10, 90, "observed"),
            (6, 6, 3, 4, 90, "observed"),
            (7, 7, 3, 4, 90, "observed"),
        ]:
            self.c.execute(
                "INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    item,
                    seller,
                    f"shop{seller}",
                    f"通用商品{item}",
                    None if item == 6 else item,
                    None if item == 6 else f"品牌{item}",
                    1,
                    "一级",
                    2,
                    "二级",
                    category,
                    str(category),
                    timestamp,
                    price,
                    status,
                    999,
                    888,
                ],
            )
        # Item 7 shares only the final category ID, not the complete category path.
        self.c.execute("UPDATE events SET category_level1_id=9 WHERE item_id=7")
        self.path = self.cfg.data / "seller_business/part_00.parquet"
        self.path.parent.mkdir(parents=True)
        self.export()

    def export(self):
        self.c.execute(f"COPY events TO {literal(self.path)} (FORMAT PARQUET)")

    def tearDown(self):
        self.c.close()
        self.temp.cleanup()

    def test_eligibility_without_any_coexposure_table(self):
        own, peers = public_candidates(self.c, self.cfg, 1, 10)
        self.assertEqual([r["item_id"] for r in peers], [2, 6])
        self.assertEqual(peers[0]["reference_price_time_index"], 8)
        self.assertIsNone(peers[0]["reference_price_raw"])
        self.assertIsNone(peers[1]["brand_name"])

    def test_no_future_focal_observation(self):
        self.assertEqual(public_candidates(self.c, self.cfg, 1, 2), (None, []))

    def test_incomplete_partition_set_fails(self):
        cfg = Config("fixture", self.cfg.output, {}, 2)
        with self.assertRaisesRegex(ValueError, "Missing or unexpected business partitions"):
            public_candidates(self.c, cfg, 1, 10)

    def test_unknown_category_yields_empty_candidates(self):
        self.c.execute("UPDATE events SET category_level2_id=NULL WHERE item_id=1")
        self.export()
        self.assertEqual(public_candidates(self.c, self.cfg, 1, 10)[1], [])

    def test_latest_incomplete_peer_category_excludes_that_item(self):
        self.c.execute("ALTER TABLE events ADD COLUMN exposure_records BIGINT DEFAULT 1000")
        self.c.execute("""INSERT INTO events SELECT * REPLACE(
            9 AS time_index, 'UNKNOWN' AS category_level2_name
        ) FROM events WHERE item_id=6""")
        self.export()
        data = DataRepository(self.c, self.cfg, 10, category_snapshots=CategorySnapshots())
        for cutoff, expected_ids in ((9, [2, 6]), (10, [2])):
            with self.subTest(cutoff=cutoff):
                _, candidates = public_candidates(self.c, self.cfg, 1, cutoff)
                pool = data.public_pool(1, 1, cutoff)
                self.assertEqual(pool["candidates"], candidates)
                self.assertEqual([row["item_id"] for row in candidates], expected_ids)

    def test_only_public_fields_and_all_candidates_retained(self):
        result = snapshot(self.c, self.cfg, 1, 10)
        self.assertEqual([row["item_id"] for row in result["candidates"]], [2, 6])
        self.assertEqual(set(result), {"item_id", "history_cutoff_exclusive", "candidates"})
        for row in result["candidates"]:
            self.assertEqual(set(row), set(PUBLIC_FIELDS))

    def test_more_than_three_candidates_are_not_truncated(self):
        for item in (8, 9, 10):
            self.c.execute(
                f"""INSERT INTO events SELECT * REPLACE({item} AS item_id,{item} AS seller_id)
                FROM events WHERE item_id=1"""
            )
        self.export()
        self.assertEqual(
            [row["item_id"] for row in snapshot(self.c, self.cfg, 1, 10)["candidates"]],
            [2, 6, 8, 9, 10],
        )

    def test_page_result_cannot_mutate_bound_pool(self):
        pool = snapshot(self.c, self.cfg, 1, 10)
        page = candidate_page(pool)
        page["candidates"][0]["reference_price_raw"] = 12345
        self.assertIsNone(candidate_detail(pool, 2)["candidate"]["reference_price_raw"])

    def test_cli_export_without_neural_model_imports(self):
        config = self.cfg.output / "cli_config.json"
        save_json(
            config,
            {
                "dataset": "fixture",
                "output": str(self.cfg.output),
                "item_parts": 1,
                "sources": {
                    name: {"path": str(self.cfg.output / f"{name}.jsonl")}
                    for name in ("items", "recall", "ranking")
                },
            },
        )
        guard = """import runpy,sys
from pathlib import Path
class NoNeuralImports:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','transformers','sentence_transformers','huggingface_hub'}:
            raise ImportError('Neural dependency is forbidden in this test: '+fullname)
sys.meta_path.insert(0,NoNeuralImports())
sys.argv=['agent_simulation.data.market.products',*sys.argv[2:]]
runpy.run_module('agent_simulation.data.market.products',run_name='__main__')
"""
        output = self.cfg.output / "cli_export"
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                guard,
                str(Path(__file__).parent),
                "--config",
                str(config),
                "--item-id",
                "1",
                "--cutoff",
                "10",
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(process.stdout)["eligible_candidates"], 2)
        saved = json.loads((output / "candidate_pool.json").read_text())
        self.assertEqual(saved, snapshot(self.c, self.cfg, 1, 10))

    def test_private_feedback_cannot_change_selection(self):
        before = snapshot(self.c, self.cfg, 1, 10)
        self.c.execute("UPDATE events SET click_records=0,purchase_records=0")
        self.export()
        self.assertEqual(before, snapshot(self.c, self.cfg, 1, 10))

    def test_page_walk_and_empty_last_page(self):
        pool = snapshot(self.c, self.cfg, 1, 10)
        first = candidate_page(pool, page_size=1)
        second = candidate_page(pool, after_item_id=first["next_after_item_id"], page_size=1)
        self.assertEqual(first["total_candidates"], 2)
        self.assertEqual(first["candidates"] + second["candidates"], pool["candidates"])
        self.assertIsNone(second["next_after_item_id"])
        self.assertEqual(candidate_page(pool, after_item_id=6)["candidates"], [])

    def test_empty_pool_is_valid(self):
        pool = snapshot(self.c, self.cfg, 1, 2)
        page = candidate_page(pool)
        self.assertEqual(page["total_candidates"], 0)
        self.assertEqual(page["candidates"], [])
        self.assertIsNone(page["next_after_item_id"])

    def test_detail_access_enforces_full_pool_not_first_page(self):
        pool = snapshot(self.c, self.cfg, 1, 10)
        self.assertEqual(candidate_detail(pool, 6)["candidate"]["item_id"], 6)
        self.assertIsNone(candidate_detail(pool, 2)["candidate"]["reference_price_raw"])
        for invalid_id in (1, 3, 4, 5, 7, 999):
            with self.subTest(invalid_id=invalid_id), self.assertRaises(PermissionError):
                candidate_detail(pool, invalid_id)

    def test_pagination_arguments_fail_fast(self):
        pool = snapshot(self.c, self.cfg, 1, 10)
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                candidate_page(pool, page_size=size)
        for cursor in (-1, True, 1.5):
            with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                candidate_page(pool, after_item_id=cursor)

    def test_titles_and_brands_do_not_rank_or_exclude(self):
        before = [row["item_id"] for row in snapshot(self.c, self.cfg, 1, 10)["candidates"]]
        self.c.execute(
            "UPDATE events SET item_title=NULL,brand_id=NULL,brand_name=NULL WHERE item_id=2"
        )
        self.export()
        after = [row["item_id"] for row in snapshot(self.c, self.cfg, 1, 10)["candidates"]]
        self.assertEqual(before, after)

    def test_no_price_fallback_for_inconsistent_state(self):
        self.c.execute("UPDATE events SET price_status='observed' WHERE price_raw IS NULL")
        self.export()
        with self.assertRaisesRegex(ValueError, "Invalid observed price"):
            public_candidates(self.c, self.cfg, 1, 10)

    def test_discovery_filters_before_pagination_and_preserves_unknowns(self):
        pool = snapshot(self.c, self.cfg, 1, 10)
        page = discover_candidates(pool, query="商品", page_size=1)
        self.assertEqual(page["eligible_candidates"], 2)
        self.assertEqual(page["total_candidates"], 2)
        self.assertEqual(page["next_after_item_id"], 2)
        next_page = discover_candidates(pool, query="商品", page_size=1, after_item_id=2)
        self.assertEqual([r["item_id"] for r in next_page["candidates"]], [6])
        self.assertIsNone(next_page["next_after_item_id"])
        self.assertIsNone(page["candidates"][0]["reference_price_raw"])
        self.assertEqual(discover_candidates(pool, brand_id=2)["total_candidates"], 1)
        self.assertEqual(discover_candidates(pool, query="商品6 品牌2")["candidates"], [])

    def test_discovery_is_literal_public_text_not_outcome_ranking(self):
        pool = snapshot(self.c, self.cfg, 1, 10)
        pool["candidates"][0]["item_title"] = "ＡＢＣ 便携水杯"
        self.assertEqual(discover_candidates(pool, query="abc 便携")["total_candidates"], 1)
        self.assertEqual(discover_candidates(pool, query="不存在")["total_candidates"], 0)
        self.assertNotIn("click_records", json.dumps(discover_candidates(pool)))
        page = discover_candidates(pool)
        page["candidates"][0]["item_title"] = "changed"
        self.assertEqual(pool["candidates"][0]["item_title"], "ＡＢＣ 便携水杯")

    def test_discovery_rejects_invalid_filters(self):
        pool = snapshot(self.c, self.cfg, 1, 10)
        for args in (
            {"page_size": 101},
            {"page_size": True},
            {"query": None},
            {"brand_id": False},
            {"brand_id": 0},
            {"after_item_id": -1},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                discover_candidates(pool, **args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
