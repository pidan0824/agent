"""Configurable fixed batches keep exact coverage, weights and cheap epoch counts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import duckdb

import agent_simulation.environment.batching as batching
from agent_simulation.environment.batching import response_batches
from agent_simulation.datasets.labels import customer_rows
from agent_simulation.datasets.samples.customer import split_supervision
from agent_simulation.training.data_loader import SampleStream


class ConfiguredBatchTests(unittest.TestCase):
    def test_64_and_128_keep_complete_disjoint_exposures(self):
        state = {'displayed_items': [{'item_id': i} for i in range(257)], 'history': ['frozen']}
        for size, lengths in ((64, [64, 64, 64, 64, 1]), (128, [128, 128, 1])):
            with self.subTest(size=size):
                batches = response_batches(state, max_items=size)
                self.assertEqual([len(b['displayed_items']) for b in batches], lengths)
                self.assertEqual([p for b in batches for p in b['displayed_items']], state['displayed_items'])
                self.assertTrue(all(b['candidate_batch']['count'] == len(lengths) for b in batches))
                batches[0]['history'].append('changed')
                self.assertEqual(state['history'], ['frozen'])
                self.assertEqual(batches[1]['history'], ['frozen'])
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                response_batches(state, max_items=size)

    def test_label_counts_match_batches_without_reading_inputs(self):
        with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as db:
            root = Path(tmp)
            path = root / 'customer/train/labels.parquet'
            path.parent.mkdir(parents=True)
            db.execute('CREATE TABLE labels AS SELECT range(n)::BIGINT[] impressed_item_ids '
                       'FROM (VALUES (0), (1), (64), (65), (128), (129)) t(n)')
            db.execute('COPY labels TO ? (FORMAT PARQUET)', [str(path)])
            for size, total in ((64, 16), (128, 13)):
                with patch.object(batching, "CUSTOMER_RESPONSE_BATCH_SIZE", size), \
                     patch.object(SampleStream, '__iter__', side_effect=AssertionError('No construction')):
                    self.assertEqual(customer_rows(root, 'train'), total)
                    self.assertEqual(customer_rows(root, 'train', response_only=True), total - 6)
                    stream = SampleStream(root, 'unused', 'customer', 'train')
                    self.assertEqual(stream.measure_epoch(6), total)
                    stream.limit = 4
                    self.assertEqual(stream.measure_epoch(6), 4)

    def test_batch_size_preserves_labels_and_total_response_weight(self):
        sample = {'sample_id': 'response:1',
                  'input': {'displayed_items': [{'item_id': i} for i in range(257)]},
                  'target': {'clicked_item_ids': [0, 63, 64, 127, 128, 256],
                             'purchased_item_ids': [64, 128]}}
        for size, count in ((64, 5), (128, 3)):
            with self.subTest(size=size), patch.object(batching, "CUSTOMER_RESPONSE_BATCH_SIZE", size):
                # The renderer is independent of label intersection and sample weights.
                with patch('agent_simulation.datasets.samples.customer.messages', return_value=[]):
                    rows = split_supervision(sample)
                self.assertEqual(len(rows), count)
                self.assertAlmostEqual(sum(row['batching']['request_weight'] for row in rows), 1)
                for key, ids in sample['target'].items():
                    self.assertEqual([i for row in rows for i in row['target'][key]], ids)
