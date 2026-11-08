"""Preserve facts and boundaries when removing duplicate work and diagnostics."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from agent_simulation.agents.retrieval.vector_cache import VectorCache
from agent_simulation.agents.retrieval.history import _rank
from agent_simulation.agents.retrieval.long_term import budget_memories
from agent_simulation.tests.seller_context_fixture import decision_context


class ContextEfficiencyTests(unittest.TestCase):
    def test_vector_cache_reopens_and_separates_model_text_and_query_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'vectors.sqlite3'
            a = VectorCache(path, 'encoder-A', persist_queries=True)
            a.write({(False, 'same'): (.25, .5), (True, 'same'): (.5, .25)})
            b = VectorCache(path, 'encoder-A', persist_queries=True)
            self.assertEqual(b.read([(False, 'same'), (True, 'same'), (False, 'changed')]),
                             {(False, 'same'): (.25, .5), (True, 'same'): (.5, .25)})
            self.assertEqual(VectorCache(path, 'encoder-B').read([(False, 'same')]), {})

    def test_duplicate_history_preserves_different_business_states_items_and_clocks(self):
        def segment(item, index, price=100, clock='observed_time_index'):
            return dict(item_id=item, clock=clock, end_inclusive=index, search_text='product',
                        match=dict(entity_match=1, numeric_distance=0), records=[dict(
                            price_raw=price, price_status='observed', exposure_records=1,
                            click_records=0, purchase_records=0)])
        candidates = [segment(1,1),segment(1,2),segment(2,3),segment(1,4,price=120),
                      segment(1,1,clock='simulation_settled_step')]
        result = _rank(candidates, 'query', None, 5, cutoff=5, simulation_before=2, deduplicate=True)
        self.assertEqual(result['returned_segments'], 4)
        self.assertIn(('observed_time_index',2), [(s['clock'],s['end_inclusive']) for s in result['segments']])
        self.assertNotIn(('observed_time_index',1), [(s['clock'],s['end_inclusive']) for s in result['segments']])

    def test_projection_keeps_recent_facts_and_audit_context_unchanged(self):
        from agent_simulation.tests.seller_context_fixture import seller_state
        state=seller_state();original=deepcopy(state)
        visible=decision_context(state)
        self.assertEqual(state,original)
        recent=visible['current_state']['products'][0]['recent'][0]['price_runs']
        self.assertEqual([r['price_raw'] for r in recent],[100,120])
        self.assertEqual(sum(r['exposure_records'] for r in recent),2)
        self.assertNotIn('indices',str(visible))
        self.assertNotIn('pricing_context',visible)

    def test_memory_budget_skips_whole_oversized_entry_and_keeps_later_citations(self):
        class Tokenizer:
            def encode(self, text, **kwargs): return list(text)
        memories=[dict(content='x'*500,source_ids=['a']),dict(content='useful',source_ids=['b'])]
        original=deepcopy(memories)
        self.assertEqual(budget_memories(memories,tokenizer=Tokenizer(),limit=100),[memories[1]])
        self.assertEqual(memories,original)


class PurchaseIndexTests(unittest.TestCase):
    def test_exact_public_counts_and_cutoff_from_one_connection_local_table(self):
        import duckdb
        from agent_simulation.data.market.quotes import purchase_bands
        with duckdb.connect() as c:
            c.execute('CREATE TABLE products AS SELECT * FROM (VALUES (1,10,1,2,3),(2,20,1,2,3)) p(item_id,seller_id,category_level1_id,category_level2_id,category_level3_id)')
            c.execute('CREATE TABLE ranking_observations AS SELECT * FROM (VALUES (1,1,100,1),(2,2,100,1),(2,3,100,0),(2,4,6000,1),(2,9,100,1)) o(item_id,time_index,price_raw,is_purchased)')
            a=purchase_bands(c,10,0,4,categories=[(1,2,3)])
            self.assertEqual(a,{((1,2,3),0):dict(purchases=1,purchased_products=1,selling_shops=1)})
            b=purchase_bands(c,10,4,9,categories=[(1,2,3)])
            self.assertEqual(b,{((1,2,3),2):dict(purchases=1,purchased_products=1,selling_shops=1)})
            self.assertEqual(c.execute('SELECT count(*) FROM context_public_purchases').fetchone()[0],4)
            self.assertEqual(c.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name='context_public_purchases' AND temporary").fetchone()[0],1)
