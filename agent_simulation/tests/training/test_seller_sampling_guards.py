"""Seller resampling must retain frozen inputs and Train-only comparison pools."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import duckdb

from agent_simulation.data.common import literal, sha256
from agent_simulation.datasets import sampling


class SellerSamplingGuardTests(unittest.TestCase):
    def test_thinning_rejects_changed_source_candidates_before_opening_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, source = root / 'dataset', root / 'source'
            dataset.mkdir()
            source.mkdir()
            (dataset / 'manifest.json').write_text('{}')
            roles = {role: {} for role in ('customer', 'seller')}
            for role, splits in roles.items():
                for split in ('train', 'test'):
                    path = source / f'{role}-{split}-candidates.parquet'
                    path.write_bytes(b'frozen source')
                    splits[split] = dict(file=path.name, sha256=sha256(path))
            manifest = source / 'manifest.json'
            manifest.write_text(json.dumps(dict(schema=sampling.SCHEMA, seed=42, roles=roles,
                                                dataset_manifest_sha256=sha256(dataset / 'manifest.json'))))
            for role, splits in roles.items():
                for split, entry in splits.items():
                    with self.subTest(role=role, split=split):
                        path = source / entry['file']
                        path.write_bytes(b'changed source')
                        with patch.object(sampling, 'Catalog') as catalog, self.assertRaisesRegex(
                                ValueError, 'Frozen source candidates changed'):
                            sampling.select_seller_training(dataset, manifest, root / 'selected')
                        catalog.assert_not_called()
                        self.assertFalse((root / 'selected').exists())
                        path.write_bytes(b'frozen source')

    def test_quote_gap_sampling_rejects_test_labels(self):
        with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as c:
            root = Path(tmp)
            labels = root / 'labels.parquet'
            c.execute("COPY (SELECT 'test' split) TO ? (FORMAT PARQUET)", [str(labels)])
            with self.assertRaisesRegex(ValueError, 'only official Train'):
                sampling.broaden_seller_quote_gaps(c, labels, root / 'unused', root / 'out')
            self.assertFalse((root / 'out').exists())

    def test_quote_gap_sampling_keeps_direction_counts_with_distinct_train_products(self):
        with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as c:
            root = Path(tmp)
            c.execute("""CREATE TABLE seller_history AS
                SELECT 1 seller_id,item_id,time_index,price_raw,'observed' price_status FROM (VALUES
                  (1,1,100.),(1,2,110.),(2,1,100.),(2,2,110.),
                  (3,1,100.),(3,20,110.),(4,1,100.),(4,20,110.)
                ) t(item_id,time_index,price_raw)""")
            labels, previous, output = (root / name for name in ('labels.parquet', 'previous.parquet', 'out.parquet'))
            c.execute("""COPY (SELECT 'train' split,'seller:1:'||item_id||':'||time_index sample_id,
                        seller_id,item_id,time_index history_cutoff_exclusive,price_raw target_price_raw
                        FROM seller_history WHERE time_index>1) TO ? (FORMAT PARQUET)""", [str(labels)])
            c.execute(f"COPY (SELECT sample_id record_id FROM read_parquet(?) WHERE item_id<=2) TO {literal(previous)} (FORMAT PARQUET)",
                      [str(labels)])
            result = sampling.broaden_seller_quote_gaps(c, labels, previous, output, gap_limit=5)
            self.assertEqual(result['direction_counts'], {'up': 2})
            self.assertEqual(c.execute('SELECT count(*),count(DISTINCT item_id) FROM read_parquet(?)',
                                       [str(output)]).fetchone(), (2, 2))

    def test_quote_gap_sampling_rejects_missing_and_duplicate_previous_records(self):
        for previous_ids, message in ((['seller:1:1:2', 'not-in-train'], 'belong to the Train pool'),
                                      (['seller:1:1:2', 'seller:1:1:2'], 'must be unique'),
                                      (['seller:1:1:2', 'seller:1:1:3'], 'repeat a product')):
            with self.subTest(ids=previous_ids), tempfile.TemporaryDirectory() as tmp, duckdb.connect() as c:
                root = Path(tmp)
                c.execute("""CREATE TABLE seller_history AS
                    SELECT 1 seller_id,1 item_id,i time_index,100.+i price_raw,'observed' price_status
                    FROM range(1,4) t(i)""")
                labels, previous = root / 'labels.parquet', root / 'previous.parquet'
                c.execute("""COPY (SELECT 'train' split,'seller:1:1:'||time_index sample_id,
                    seller_id,item_id,time_index history_cutoff_exclusive,price_raw target_price_raw
                    FROM seller_history WHERE time_index>1) TO ? (FORMAT PARQUET)""", [str(labels)])
                c.execute(f'COPY (SELECT unnest(?::VARCHAR[]) record_id) TO {literal(previous)} (FORMAT PARQUET)',
                          [previous_ids])
                with self.assertRaisesRegex(ValueError, message):
                    sampling.broaden_seller_quote_gaps(c, labels, previous, root / 'out', gap_limit=5)


if __name__ == '__main__':
    unittest.main()
