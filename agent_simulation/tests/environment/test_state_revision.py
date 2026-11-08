"""Revision guards detect writes, preserve rollback, and ignore read-only audits."""
import asyncio
import json
import sqlite3
import unittest
from unittest.mock import patch

from agent_simulation.agents.customer import CustomerAgent
from agent_simulation.environment.operations.actions import SellerReview
from agent_simulation.tests.fixtures import MarketFixture, ScriptedModel, TokenizerFixture, answer


class StateRevisionTests(MarketFixture, unittest.TestCase):
    def test_frozen_snapshot_keeps_its_original_version(self):
        frozen = self.runtime.freeze()
        self.assertEqual(frozen.state, self.runtime.snapshot())
        self.assertEqual(frozen.revision, self.runtime.revision)
        self.runtime.execute('seller', 10, 'submit_prices', {'prices': [{'item_id': 1, 'price_raw': 130}]})
        self.assertEqual(frozen.state['active_prices']['1']['price_raw'], 120)
        # A review created later must keep the snapshot's version, not the live one.
        review = SellerReview(self.runtime, 10, frozen)
        review.stage({'prices': [{'item_id': 1, 'price_raw': 140}]}, [1])
        review.stage({'prices': [{'item_id': 3, 'price_raw': 200}]}, [3])
        with self.assertRaisesRegex(ValueError, 'Market changed'):
            review.commit('stale-before-review')
        self.assertEqual(self.runtime.snapshot()['active_prices']['1']['price_raw'], 130)

    def test_staged_review_rejects_change_then_restore_before_state_read(self):
        frozen = self.runtime.freeze()
        review = SellerReview(self.runtime, 10, frozen)
        review.stage({'prices': [{'item_id': 1, 'price_raw': 111}]}, [1])
        review.stage({'prices': [{'item_id': 3, 'price_raw': 200}]}, [3])
        with sqlite3.connect(self.path) as db:
            changed = {**frozen.state, 'simulation_step': 99}
            db.execute('UPDATE state SET value=? WHERE id=1', (json.dumps(changed),))
            db.execute('UPDATE state SET value=? WHERE id=1', (json.dumps(frozen.state),))
        self.assertEqual(self.runtime.snapshot(), frozen.state)
        with patch.object(self.runtime, 'snapshot', side_effect=AssertionError('No full snapshot read')):
            with self.assertRaisesRegex(ValueError, 'Market changed'):
                review.commit('restored')
        self.assertEqual(self.runtime.snapshot(), frozen.state)

    def test_read_only_tools_and_unchanged_prices_do_not_advance_revision(self):
        revision = self.runtime.revision
        self.runtime.execute('seller', 10, 'query_shop_products', {'after_item_id': 0, 'limit': 1})
        self.assertEqual(self.runtime.db.execute('SELECT count(*) FROM calls').fetchone()[0], 0)
        self.runtime.execute('seller', 10, 'submit_prices', {'prices': [{'item_id': 1, 'price_raw': 120}]})
        self.assertEqual(self.runtime.revision, revision)
        self.runtime.execute('seller', 10, 'submit_prices', {'prices': [{'item_id': 1, 'price_raw': 130}]})
        self.assertEqual(self.runtime.revision, revision + 1)

    def test_rollback_restores_state_and_revision(self):
        before, revision = self.runtime.snapshot(), self.runtime.revision
        with self.assertRaisesRegex(RuntimeError, 'rollback'):
            with self.runtime.store.transaction():
                state = self.runtime.snapshot()
                state['active_prices']['1']['price_raw'] = 150
                self.runtime.store.write(state)
                self.assertEqual(self.runtime.revision, revision + 1)
                raise RuntimeError('rollback')
        self.assertEqual(self.runtime.snapshot(), before)
        self.assertEqual(self.runtime.revision, revision)

    def test_generation_detects_change_then_restore_from_another_connection(self):
        before, revision = self.runtime.snapshot(), self.runtime.revision
        def respond(request, call_id):
            with sqlite3.connect(self.path) as db:
                changed = {**before, 'simulation_step': 99}
                db.execute('UPDATE state SET value=? WHERE id=1', (json.dumps(changed, ensure_ascii=False),))
                db.execute('UPDATE state SET value=? WHERE id=1', (json.dumps(before, ensure_ascii=False),))
            return answer({'query': '水杯'})
        agent = CustomerAgent(self.runtime, ScriptedModel([respond]), TokenizerFixture())
        run = self.runtime.logs.start('customer', 7)
        with patch.object(self.runtime, 'snapshot', side_effect=AssertionError('No full snapshot read')):
            with self.assertRaisesRegex(ValueError, 'State changed during model generation'):
                asyncio.run(agent._complete(run, [{'role':'user','content':'query'}], [], 'query'))
        self.assertEqual(self.runtime.snapshot(), before)
        self.assertEqual(self.runtime.revision, revision + 2)
