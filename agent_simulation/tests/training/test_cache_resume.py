"""Crash recovery skips committed decisions while retaining their factual history."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import duckdb
import numpy as np

from agent_simulation.agents.retrieval.vector_cache import VectorCache
from agent_simulation.datasets.tokens import build_splits
from agent_simulation.datasets.tokens import PreparedRow, RecordWriter, read_shard
from agent_simulation.datasets.tokens import initialize_vector_cache
from agent_simulation.data.common import literal


def rows(record):
    return [PreparedRow(f'{stage}:{record}', np.array([1, 2, 3], dtype=np.int32), 2, 1.)
            for stage in ('query', 'response')]


class CacheResumeTests(unittest.TestCase):
    def test_resume_keeps_committed_shards_and_failures_but_rebuilds_uncommitted_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp), reserve_gib=0)
            writer = RecordWriter(args, 'customer', 'train', 0, 4)
            writer.add('1', rows('1'))
            writer.flush()
            first = writer.dest / '000000.npz'
            original_mtime = first.stat().st_mtime_ns
            writer.add('2', rows('2'))  # Not committed before the simulated crash.
            writer.reject('3', ValueError('oversized'))
            resumed = RecordWriter(args, 'customer', 'train', 0, 4)
            self.assertEqual(resumed.restored, {'1', '3'})
            self.assertEqual(resumed.seen, 2)
            resumed.add('2', rows('2'))
            resumed.add('4', rows('4'))
            result = resumed.finish()
            self.assertEqual(first.stat().st_mtime_ns, original_mtime)
            self.assertEqual((result['rows'], result['records'], result['excluded_records']), (6, 3, 1))
            ids = [r.sample_id for f in result['files'] for r in read_shard(resumed.dest / f['file'])]
            self.assertEqual(ids, [r.sample_id for i in ('1', '2', '4') for r in rows(i)])
            self.assertEqual(len(resumed.failures.read_text().splitlines()), 1)

    def test_orphan_shard_is_replaced_after_checkpoint_write_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp), reserve_gib=0)
            writer = RecordWriter(args, 'customer', 'train', 0, 1)
            writer.add('1', rows('1'))
            with patch('agent_simulation.datasets.tokens.write_report', side_effect=OSError('crash')):
                with self.assertRaisesRegex(OSError, 'crash'):
                    writer.flush()
            resumed = RecordWriter(args, 'customer', 'train', 0, 1)
            self.assertEqual(resumed.restored, set())
            resumed.add('1', rows('1'))
            self.assertEqual(resumed.finish()['rows'], 2)

    def test_resume_requires_shards_and_only_verifies_checksums_on_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp), reserve_gib=0)
            writer = RecordWriter(args, 'customer', 'train', 0, 2)
            writer.add('1', rows('1'))
            writer.flush()
            shard = writer.dest / '000000.npz'
            shard.write_bytes(b'changed')
            self.assertEqual(RecordWriter(args, 'customer', 'train', 0, 2).restored, {'1'})
            args.verify = True
            with self.assertRaisesRegex(ValueError, 'Committed shard changed'):
                RecordWriter(args, 'customer', 'train', 0, 2)
            shard.unlink()
            args.verify = False
            with self.assertRaisesRegex(ValueError, 'Missing committed shard'):
                RecordWriter(args, 'customer', 'train', 0, 2)

    def test_resumed_builder_only_encodes_pending_record_with_complete_history(self):
        with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as c:
            root = Path(tmp)
            args = SimpleNamespace(output=root/'out', dataset=root, tokenizer=root,
                candidates=root/'manifest.json', buckets=1, reserve_gib=0)
            c.execute('CREATE TABLE requests AS SELECT 7 user_id,i session_id,i time_index FROM range(1,5) t(i)')
            c.execute("CREATE TABLE customer_splits AS SELECT session_id,'train' assigned_split FROM requests")
            path = root/'customer-train-candidates.parquet'
            c.execute(f"COPY (SELECT 7 actor_id,session_id,time_index history_cutoff_exclusive,CAST(session_id AS VARCHAR) record_id FROM requests WHERE session_id<=3) TO {literal(path)} (FORMAT PARQUET)")
            writer = RecordWriter(args, 'customer', 'train', 0, 3)
            writer.add('1', rows('1'))
            writer.flush()
            writer.reject('2', ValueError('oversized'))
            data = SimpleNamespace(catalog=SimpleNamespace(connection=c), tokenizer=None, close=lambda: None)
            def source(*args, **kwargs):
                self.assertEqual(c.execute('SELECT session_id FROM requests ORDER BY 1').fetchall(), [(1,), (2,), (3,)])
                pending = c.execute('SELECT session_id FROM customer_splits WHERE assigned_split IS NOT NULL').fetchall()
                self.assertEqual(pending, [(3,)])
                return iter([(dict(key={'session_id': 3}), {})])
            samples = [dict(split='train', row=r) for r in rows('3')]
            with patch('agent_simulation.datasets.tokens.SFTDataset', return_value=data), \
                 patch('agent_simulation.datasets.tokens.customer_records', side_effect=source), \
                 patch('agent_simulation.datasets.tokens.expand_record', return_value=samples) as expand, \
                 patch('agent_simulation.datasets.tokens.PreparedRow.from_sample', side_effect=lambda s,t:s['row']) as encode:
                result = build_splits(args, 'customer', 0, ('train',))['train']
            self.assertEqual(expand.call_count, 1)
            self.assertEqual(encode.call_count, 2)
            self.assertEqual((result['rows'], result['candidate_records'], result['excluded_records']), (4, 3, 1))


class CacheInitializationTests(unittest.TestCase):
    def test_independent_build_parents_initialize_one_database_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'vectors.sqlite3'
            local, barrier = threading.local(), threading.Barrier(16)
            def worker(i):
                local.cache = VectorCache(path, 'fixture', reserve_bytes=0)
                barrier.wait()
                try:
                    initialize_vector_cache()
                    local.cache.write({(False, str(i)): (1., 0.)})
                    return local.cache.read([(False, str(i))])
                finally:
                    local.cache.connection.close()
            with patch('agent_simulation.agents.retrieval.bge_encoder.default_encoder',
                       side_effect=lambda: (SimpleNamespace(disk_cache=local.cache), 'fixture')):
                with ThreadPoolExecutor(16) as pool:
                    result = list(pool.map(worker, range(16)))
            self.assertEqual(result, [{(False, str(i)): (1., 0.)} for i in range(16)])
