"""Regression checks for portable configuration, source binding and explicit sample selection."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import shutil
from agent_simulation.data.common import Config, business_parts, sha256
from agent_simulation.data.build.catalog import build as build_catalog
from agent_simulation.tests.data.fixtures import CatalogFixture
import duckdb


class ConfigTests(unittest.TestCase):
    def test_full_and_lite_relative_paths_ignore_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, count in [("Full", 32), ("Lite", 8)]:
                payload = {
                    "dataset": "KuaiSearch " + name,
                    "output": "../data/" + name,
                    "sources": {
                        k: {"path": "../raw/" + k + ".jsonl"}
                        for k in ("items", "recall", "ranking")
                    },
                }
                if name == "Full":
                    payload["item_parts"] = count
                path = root / (name + ".json")
                path.write_text(json.dumps(payload))
                cfg = Config.load(path)
                self.assertEqual(cfg.output, (root / "../data" / name).resolve())
                self.assertEqual(
                    cfg.sources["items"]["path"], str((root / "../raw/items.jsonl").resolve())
                )
                self.assertEqual(cfg.item_parts, count)

    def test_empty_wrong_partition_cannot_replace_missing_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Config("fixture", Path(directory), {}, 2)
            path = cfg.data / "seller_business"
            path.mkdir(parents=True)
            (path / "part_00.parquet").touch()
            (path / "part_99.parquet").touch()
            with self.assertRaisesRegex(ValueError, "partitions"):
                business_parts(cfg)


class SourceBindingTests(CatalogFixture, unittest.TestCase):
    def test_user_hash_and_explicit_path_are_enforced(self):
        for case in ("hash", "path"):
            cfg = deepcopy(self.cfg)
            cfg.sources["users"]["sha256"] = "0" * 64
            if case == "path":
                cfg.sources["users"]["path"] = str(self.root / "another_user.jsonl")
            with self.assertRaisesRegex(ValueError, "source hash mismatch|User source"):
                build_catalog(
                    cfg, self.root / "users.jsonl", self.splits, self.root / ("bad_" + case)
                )




if __name__ == "__main__":
    unittest.main()
