"""Real SQLite capacity, admission, encoder separation and restart behavior."""
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent_simulation.agents.retrieval.vector_cache import VectorCache


class BoundedCacheTests(unittest.TestCase):
    def test_minimum_budget_accepts_large_batches_and_reopens(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.db'
            budget = 64 * 1024
            vector = tuple(i / 512 for i in range(512))
            cache = VectorCache(path, 'encoder', max_bytes=budget, reserve_bytes=0)
            for start in (0, 64):
                cache.write({(False, f'商品 {i}'): vector for i in range(start, start + 64)})
                self.assertEqual(cache.read([(False, f'商品 {start + 63}')]),
                                 {(False, f'商品 {start + 63}'): vector})
            self.assertEqual(cache.skipped_writes, 0)
            self.assertEqual(cache.read([(False, '商品 0')]), {})
            self.assertLessEqual(cache.connection.execute('PRAGMA page_count').fetchone()[0]
                                 * cache.page_size, budget)
            cache.connection.close()
            cache = VectorCache(path, 'encoder', max_bytes=budget, reserve_bytes=0)
            self.assertEqual(cache.read([(False, '商品 127')]), {(False, '商品 127'): vector})
            cache.connection.close()

    def test_larger_vectors_can_evict_many_small_rows_without_retrying_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = VectorCache(Path(tmp) / 'cache.db', 'encoder', max_bytes=256 * 1024, reserve_bytes=0)
            cache.write({(False, f'small {i}'): (1., 0.) for i in range(2000)})
            vector = (0.25,) * 3000
            cache.write({(False, f'large {i}'): vector for i in range(64)})
            self.assertEqual(cache.skipped_writes, 0)
            self.assertEqual(cache.read([(False, 'large 63')]), {(False, 'large 63'): vector})
            cache.connection.close()

    def test_fifo_reuses_pages_and_preserves_newest_exact_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache.db'
            budget = 256 * 1024
            cache = VectorCache(path, 'encoder-A', max_bytes=budget, reserve_bytes=0)
            vector = tuple(i / 512 for i in range(512))
            for start in range(0, 1000, 16):
                cache.write({(False, f'商品 {i}'): vector for i in range(start, start + 16)})
            self.assertLessEqual(path.stat().st_size, budget)
            self.assertLessEqual(cache.connection.execute('PRAGMA page_count').fetchone()[0] * 4096, budget)
            self.assertEqual(cache.read([(False, '商品 1007')]), {(False, '商品 1007'): vector})
            self.assertEqual(cache.read([(False, '商品 0')]), {})
            self.assertLess(cache.connection.execute('SELECT count(*) FROM vectors').fetchone()[0], 1000)
            self.assertEqual(cache.skipped_writes, 0)
            cache.connection.close()
            reopened = VectorCache(path, 'encoder-A', max_bytes=budget, reserve_bytes=0)
            self.assertEqual(reopened.read([(False, '商品 1007')]), {(False, '商品 1007'): vector})
            reopened.connection.close()

    def test_query_and_disk_reserve_skip_admission_without_changing_existing_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = VectorCache(Path(tmp) / 'cache.db', 'encoder', reserve_bytes=1024)
            cache.write({(False, 'doc'): (1., 0.), (True, 'query'): (0., 1.)})
            with patch('agent_simulation.agents.retrieval.vector_cache.shutil.disk_usage',
                       return_value=SimpleNamespace(free=1024)):
                cache.write({(False, 'later'): (0., 1.)})
            self.assertEqual(cache.read([(False, 'doc'), (False, 'later'), (True, 'query')]),
                             {(False, 'doc'): (1., 0.)})
            self.assertEqual(cache.skipped_writes, 1)
            cache.connection.close()

    def test_legacy_cache_is_never_migrated_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.db'
            with sqlite3.connect(path) as c:
                c.execute('CREATE TABLE vectors(encoder TEXT, query INTEGER, text TEXT, vector BLOB)')
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'new vectors-v2'):
                VectorCache(path, 'encoder').read([(False, 'test')])
            self.assertEqual(path.read_bytes(), original)
