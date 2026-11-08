"""Transactional market persistence with a monotonic state revision."""
from contextlib import contextmanager
import json
from typing import NamedTuple

from agent_simulation.data.common import require


class MarketSnapshot(NamedTuple):
    """One consistent database read; decision callers share state without mutating it."""
    state: dict
    revision: int


class MarketStore:
    def __init__(self, db):
        self.db = db
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS state_revision '
                       '(id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL)')
            db.execute('INSERT OR IGNORE INTO state_revision VALUES (1,0)')
            # Covers every UPDATE, including a second connection and change-then-restore.
            db.execute('''CREATE TRIGGER IF NOT EXISTS advance_state_revision
                AFTER UPDATE OF value ON state WHEN NEW.value IS NOT OLD.value
                BEGIN UPDATE state_revision SET revision=revision+1 WHERE id=1; END''')

    @property
    def revision(self):
        return self.db.execute('SELECT revision FROM state_revision WHERE id=1').fetchone()[0]

    def snapshot(self):
        return json.loads(self.db.execute('SELECT value FROM state WHERE id=1').fetchone()[0])

    def freeze(self):
        value, revision = self.db.execute(
            'SELECT value,revision FROM state JOIN state_revision USING(id) WHERE id=1'
        ).fetchone()
        return MarketSnapshot(json.loads(value), revision)

    @contextmanager
    def transaction(self):
        require(not self.db.in_transaction, 'Nested market transaction')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            yield

    def write(self, state):
        require(self.db.in_transaction, 'Market writes require a transaction')
        self.db.execute('UPDATE state SET value=? WHERE id=1',
                        (json.dumps(state, ensure_ascii=False, allow_nan=False),))

    def write_tick_trace(self, trace):
        require(self.db.in_transaction, 'Tick traces require a transaction')
        audit = {k: v for k, v in trace.items() if k not in ('sellers', 'customers')}
        for role in ('sellers', 'customers'):
            audit[role] = [{k: run[k] for k in ('id', 'actor', 'result', 'applied_result', 'settled_result')
                            if k in run} for run in trace[role]]
        self.db.execute('INSERT OR REPLACE INTO ticks VALUES (?,?)',
                        (trace['tick'], json.dumps(audit, ensure_ascii=False, allow_nan=False)))
