"""Complete label-pool export and on-demand reading."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import duckdb

from agent_simulation.datasets.labels import SFTDataset
from agent_simulation.datasets.labels import build_labels
from agent_simulation.tests.fixtures import TokenizerFixture


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory = self.root / 'catalog'; self.directory.mkdir()
        self.tokenizer = self.root / 'tokenizer'; self.tokenizer.mkdir()
        self.output = self.root / 'sft'
        manifest = {'files': {}, 'business_sources': [], 'split_files': []}
        (self.directory / 'manifest.json').write_text(json.dumps(manifest))
        directory = self.directory
        class Stub:
            def __init__(self, path):
                self.directory = directory
                self.config = SimpleNamespace(dataset='fixture')
                self.manifest = manifest
                self.connection = duckdb.connect()
                self.connection.execute("CREATE TABLE seller_targets AS SELECT 10 seller_id,i item_id,10 time_index,'train' assigned_split FROM (VALUES (1),(3)) x(i)")
                self.connection.execute("CREATE TABLE seller_history AS SELECT 10 seller_id,i item_id,10 time_index,p price_raw,'observed' price_status FROM (VALUES (1,100),(3,200)) x(i,p)")
                self.connection.execute("INSERT INTO seller_history SELECT * REPLACE(1 AS time_index) FROM seller_history")
                self.connection.execute("CREATE TABLE requests AS SELECT 7 session_id,8 user_id,10 time_index,'杯子' query,[1,3] impressed_item_ids,[]::INTEGER[] clicked_item_ids,[3] purchased_item_ids,'train' split")
                self.connection.execute("INSERT INTO requests SELECT * REPLACE (8 AS session_id, 11 AS time_index, 'test' AS split) FROM requests")
                self.connection.execute("CREATE TABLE customer_splits AS SELECT 7 session_id,'train' assigned_split UNION ALL SELECT 8,'test'")
            def close(self):
                self.connection.close()
        self.Stub = Stub

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self):
        with patch('agent_simulation.datasets.labels.Catalog', self.Stub):
            return build_labels(self.directory, self.output)

    def test_release_rejects_missing_fact_and_can_retry_at_same_path(self):
        catalog = self.Stub(self.directory)
        catalog.connection.execute("DELETE FROM seller_history WHERE item_id=1 AND time_index=10")
        with patch("agent_simulation.datasets.labels.Catalog", return_value=catalog):
            with self.assertRaises(ValueError):
                build_labels(self.directory, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.prepare()["seller"]["train"], 2)

    def test_one_request_cannot_be_published_in_two_splits(self):
        catalog = self.Stub(self.directory)
        catalog.connection.execute("INSERT INTO customer_splits VALUES (7, 'validation')")
        with patch("agent_simulation.datasets.labels.Catalog", return_value=catalog):
            with self.assertRaisesRegex(ValueError, "split"):
                build_labels(self.directory, self.output)
        self.assertFalse(self.output.exists())

    def test_all_labels_and_purchase_without_click_are_preserved(self):
        counts = self.prepare()
        self.assertEqual(counts['seller'], {'train': 2, 'test': 0})
        self.assertEqual(counts['customer'], {'train': 1, 'test': 1})
        c = duckdb.connect()
        try:
            actual = c.execute('SELECT clicked_item_ids,purchased_item_ids FROM read_parquet(?)', [str(self.output / 'customer/train/labels.parquet')]).fetchone()
            self.assertEqual(actual, ([], [3]))
        finally:
            c.close()
        manifest = json.loads((self.output / 'manifest.json').read_text())
        self.assertTrue(manifest['complete_label_pools'])
        self.assertFalse(manifest['all_prompts_materialized'])
        self.assertFalse(manifest['all_samples_capacity_checked'])
        with self.assertRaisesRegex(ValueError, 'new dataset'):
            self.prepare()

    def test_on_demand_read_and_optional_label_integrity(self):
        self.prepare()
        with patch('agent_simulation.datasets.labels.Catalog', self.Stub), \
             patch('agent_simulation.datasets.labels.load_tokenizer', return_value=TokenizerFixture()):
            dataset = SFTDataset(self.output, self.tokenizer, verify_hashes=True)
            dataset.close()
            string_paths = SFTDataset(str(self.output), str(self.tokenizer))
            self.assertEqual(string_paths.manifest, dataset.manifest)
            string_paths.close()
            p = self.output / 'seller/train/labels.parquet'
            p.write_bytes(b'corrupt fixture')
            with self.assertRaisesRegex(ValueError, 'Label pool changed'):
                SFTDataset(self.output, self.tokenizer, verify_hashes=True)

    def test_initialization_failure_closes_the_opened_catalog(self):
        self.prepare()
        opened = []
        def catalog(path):
            instance = self.Stub(path)
            opened.append(instance)
            return instance
        with patch('agent_simulation.datasets.labels.Catalog', side_effect=catalog), \
             patch('agent_simulation.datasets.labels.load_tokenizer', side_effect=ValueError('broken tokenizer')):
            with self.assertRaisesRegex(ValueError, 'broken tokenizer'):
                SFTDataset(self.output, self.tokenizer)
        self.assertEqual(len(opened), 1)
        with self.assertRaises(duckdb.ConnectionException):
            opened[0].connection.execute('SELECT 1')

    def test_default_iteration_covers_all_buckets_and_explicit_bucket_is_bounded(self):
        data = SFTDataset.__new__(SFTDataset)
        data.catalog, data.tokenizer = object(), object()
        data.check_sample = lambda sample: sample
        def source(catalog, bucket, buckets, *, split=None):
            return iter([{'bucket': bucket, 'split': 'train'}])
        with patch('agent_simulation.datasets.labels.seller_records', side_effect=source):
            self.assertEqual([s['bucket'] for s in data.iter_samples('seller', buckets=4)], [0,1,2,3])
            self.assertEqual([s['bucket'] for s in data.iter_samples('seller', bucket=2, buckets=4)], [2])
