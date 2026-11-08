"""Test chronology, source binding and target membership with small real files."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import duckdb
from agent_simulation.data.build.splits import (
    build,
    validate_sources,
)
from agent_simulation.data.common import Config


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recall = self.root / "recall.jsonl"
        self.records = [
            dict(
                session_id=i + 1,
                user_id=1,
                time_index=t,
                split="test" if t == 4 else "train",
                query="杯子",
                impressed_item_ids=[1],
                clicked_item_ids=[],
                purchased_item_ids=[],
            )
            for i, t in enumerate([1, 2, 2, 3, 4])
        ]
        self.write_requests()
        parts = self.root / "business/data/seller_business"
        parts.mkdir(parents=True)
        self.cfg = Config(
            "fixture", self.root / "business", {"recall": {"path": str(self.recall)}}, 2
        )
        c = duckdb.connect()
        c.execute(
            """CREATE TABLE e(item_id BIGINT,seller_id BIGINT,time_index BIGINT,
                    price_raw DOUBLE,price_status VARCHAR,source_split VARCHAR)"""
        )
        c.executemany(
            "INSERT INTO e VALUES(?,?,?,?,?,?)",
            [
                (1, 10, 1, 100, "observed", "train"),
                (1, 10, 2, 100, "observed", "train"),
                (1, 10, 3, None, "invalid_price", "train"),
                (1, 10, 4, 110, "observed", "test"),
                (2, 20, 1, 200, "observed", "train"),
                (2, 20, 3, 200, "observed", "train"),
                (3, 30, 1, None, "invalid_price", "train"),
                (3, 30, 2, 300, "observed", "train"),
            ],
        )
        for bucket in (0, 1):
            c.execute(
                f"COPY (SELECT * FROM e WHERE item_id%2={bucket}) TO '{parts/f'part_{bucket:02d}.parquet'}' (FORMAT PARQUET)"
            )
        c.close()

    def write_requests(self):
        self.recall.write_text("".join(json.dumps(r) + "\n" for r in self.records))

    def test_memberships_ties_unchanged_unknown_and_reconciliation(self):
        output = self.root / "split"
        m = build(self.cfg, output)
        self.assertEqual(m["rule"]["last_train_time_index"], 3)
        self.assertEqual(
            {r["assigned_split"]: r["n"] for r in m["customer_request_pools"]},
            {"train": 4, "test": 1},
        )
        self.assertEqual(
            {k: v["n"] for k, v in m["seller_label_pools"].items()},
            {"train": 2, "test": 1},
        )
        self.assertEqual(
            m["seller_exclusions"],
            {"all_observations": 8, "first_observation": 3, "invalid_target_after_first": 1,
             "valid_target_without_prior_valid_price": 1},
        )
        c = duckdb.connect()
        target = c.execute(
            "SELECT item_id,time_index,previous_time,assigned_split FROM read_parquet(?) ORDER BY item_id,time_index",
            [str(output / "seller_targets/*.parquet")],
        ).fetchall()
        self.assertEqual(
            target, [(1, 2, 1, "train"), (1, 4, 3, "test"), (2, 3, 1, "train")]
        )
        c.close()
        validate_sources(self.cfg, m)

    def test_duplicate_request_rejected(self):
        self.records[1]["session_id"] = self.records[0]["session_id"]
        self.write_requests()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build(self.cfg, self.root / "split")

        self.assertFalse((self.root / "split").exists())

    def test_default_preserves_official_train_test_without_validation(self):
        m = build(self.cfg, self.root / 'official')
        self.assertEqual(m['rule']['last_train_time_index'], 3)
        self.assertEqual(m['rule']['membership'], 'official_source_split')
        self.assertEqual({r['assigned_split']:r['n'] for r in m['customer_request_pools']},
                         {'train':4, 'test':1})
        self.assertEqual({s:r['n'] for s,r in m['seller_label_pools'].items()}, {'train':2,'test':1})

    def test_overlapping_official_periods_rejected(self):
        self.records[-1]["time_index"] = 2
        self.write_requests()
        with self.assertRaisesRegex(ValueError, "overlap"):
            build(self.cfg, self.root / "split")

    def test_source_release_guard(self):
        m = build(self.cfg, self.root / "split")
        changed = deepcopy(m)
        changed["dataset"] = "other release"
        with self.assertRaisesRegex(ValueError, "another release"):
            validate_sources(self.cfg, changed)


if __name__ == "__main__":
    unittest.main()
