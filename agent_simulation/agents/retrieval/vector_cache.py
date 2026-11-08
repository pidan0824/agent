"""Bounded process-shared exact vectors; cache admission never limits sample coverage.

A rowid table keeps 512-d float32 vectors inline. Encoder names are stored once;
fixed-size text hashes index lookups, with original text checked on every hit.
FIFO eviction runs only near capacity. Reads never update SQLite (no LRU writes).
"""

from functools import cached_property
import hashlib
from pathlib import Path
import shutil
import sqlite3
import struct


class VectorCache:
    def __init__(self, path, identity, *, max_bytes=2 * 1024**3,
                 reserve_bytes=8 * 1024**3, persist_queries=False):
        if max_bytes < 64 * 1024 or reserve_bytes < 0:
            raise ValueError("Vector cache needs at least 64 KiB and a nonnegative disk reserve")
        self.path, self.identity = Path(path), identity
        self.max_bytes, self.reserve_bytes = int(max_bytes), int(reserve_bytes)
        self.persist_queries = persist_queries
        self.skipped_writes = 0

    @cached_property
    def connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(self.path, timeout=30)
        # Deliberately refuse the old unbounded file, rather than migrating it in place.
        version = c.execute("PRAGMA user_version").fetchone()[0]
        tables = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if version not in (0, 2) or (version == 0 and tables):
            c.close()
            raise ValueError("Incompatible vector cache; use a new vectors-v2.sqlite3 path")
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA wal_autocheckpoint=256")
        c.execute("PRAGMA journal_size_limit=1048576")
        self.page_size = c.execute("PRAGMA page_size").fetchone()[0]
        self.max_pages = self.max_bytes // self.page_size
        actual = c.execute(f"PRAGMA max_page_count={self.max_pages}").fetchone()[0]
        if actual > self.max_pages:
            c.close()
            raise ValueError("Existing vector cache exceeds configured budget; use a new path")
        with c:
            c.execute("CREATE TABLE IF NOT EXISTS encoders (id INTEGER PRIMARY KEY, identity TEXT UNIQUE)")
            c.execute("""CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY, encoder INTEGER, query INTEGER,
                digest BLOB, text TEXT, vector BLOB, UNIQUE(encoder,query,digest))""")
            c.execute("PRAGMA user_version=2")
            c.execute("INSERT OR IGNORE INTO encoders(identity) VALUES (?)", (self.identity,))
        self.encoder_id = c.execute("SELECT id FROM encoders WHERE identity=?", (self.identity,)).fetchone()[0]
        return c

    @staticmethod
    def _digest(text):
        return hashlib.sha256(text.encode()).digest()

    def read(self, keys):
        result = {}
        for query, text in keys:
            if query and not self.persist_queries:
                continue
            row = self.connection.execute(
                "SELECT text,vector FROM vectors WHERE encoder=? AND query=? AND digest=?",
                (self.encoder_id, query, self._digest(text))).fetchone()
            if row is not None and row[0] == text:
                result[query, text] = struct.unpack(f"<{len(row[1]) // 4}f", row[1])
        return result

    def _batches(self, entries):
        """Bound transactions by estimated page/WAL space as well as row count."""
        batch, needed = [], 0
        for row in entries:
            size = (len(row[2].encode()) + len(row[3]) + self.page_size) * 2
            if batch and (len(batch) == 64 or needed + size > self.max_bytes // 4):
                yield batch, needed
                batch, needed = [], 0
            batch.append(row)
            needed += size
        if batch:
            yield batch, needed

    # Admission and eviction are serialized with other writers, in small transactions.
    def write(self, values):
        entries = []
        for (query, text), vector in values.items():
            if query and not self.persist_queries:
                continue
            size = len(text.encode()) + 4 * len(vector)
            if size > min(self.max_bytes // 16, 1024 * 1024):
                self.skipped_writes += 1
                continue
            entries.append((query, self._digest(text), text, struct.pack(f"<{len(vector)}f", *vector)))
        if not entries:
            return
        c = self.connection
        for batch, needed in self._batches(entries):
            # Allow room for both data pages and the WAL transaction above the reserve.
            if shutil.disk_usage(self.path.parent).free < self.reserve_bytes + needed:
                self.skipped_writes += len(batch)
                continue
            try:
                with c:
                    c.execute("BEGIN IMMEDIATE")
                    used = c.execute("PRAGMA page_count").fetchone()[0] - c.execute("PRAGMA freelist_count").fetchone()[0]
                    pages = (needed + self.page_size - 1) // self.page_size
                    while used + pages >= self.max_pages * .9:
                        count = c.execute("SELECT count(*) FROM vectors").fetchone()[0]
                        if not count:
                            break
                        c.execute("DELETE FROM vectors WHERE id IN (SELECT id FROM vectors ORDER BY id LIMIT ?)",
                                  (max(len(batch), count // 10),))
                        used = c.execute("PRAGMA page_count").fetchone()[0] - c.execute("PRAGMA freelist_count").fetchone()[0]
                    c.executemany("""INSERT OR IGNORE INTO vectors(encoder,query,digest,text,vector)
                                     VALUES (?,?,?,?,?)""",
                                  [(self.encoder_id, *row) for row in batch])
            except sqlite3.OperationalError as error:
                # A cache at its hard page cap is a miss; all other SQLite errors surface.
                if error.sqlite_errorcode != sqlite3.SQLITE_FULL:
                    raise
                self.skipped_writes += len(batch)
