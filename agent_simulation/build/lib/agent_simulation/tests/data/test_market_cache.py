"""Materialized market facts preserve queries and reject stale or altered sources."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import duckdb

from agent_simulation.data.common import Config, literal
from agent_simulation.data.market.quotes import build_cache, attach_cache
from agent_simulation.data.catalog import DataRepository
from agent_simulation.data.market.quotes import owned_profiles, shop_period, _decision_positions, category_panels


class MarketCacheTests(unittest.TestCase):
    def fixture(self, root):
        cfg = Config('test', root/'facts', {}, item_parts=1)
        parts = cfg.data/'seller_business';parts.mkdir(parents=True)
        with duckdb.connect() as c:
            c.execute('''CREATE TABLE business(item_id BIGINT,seller_id BIGINT,time_index BIGINT,
                category_level1_id BIGINT,category_level2_id BIGINT,category_level3_id BIGINT,
                price_raw DOUBLE,price_status VARCHAR,exposure_records BIGINT,
                click_records BIGINT,purchase_records BIGINT)''')
            c.executemany('INSERT INTO business VALUES (?,?,?,?,?,?,?,?,?,?,?)', [
                (1,10,1,1,2,3,1000,'observed',10,2,1),
                (2,20,1,1,2,3,1200,'observed',20,3,1),
                (2,20,3,1,2,3,None,'invalid_price',4,1,0),
                (3,30,2,1,2,3,900,'observed',15,5,2),
                (4,40,50,1,2,3,1,'observed',100,90,80),
            ])
            c.execute(f"""COPY (SELECT *, 'shop' AS seller_name, 'cup' AS item_title,
                1 AS brand_id, 'brand' AS brand_name, 'c1' AS category_level1_name,
                'c2' AS category_level2_name, 'c3' AS category_level3_name FROM business)
                TO {literal(parts/'part_00.parquet')} (FORMAT PARQUET)""")
        return cfg

    def test_default_market_uses_recent_window_and_full_query_keeps_old_quotes(self):
        from agent_simulation.data.market.quotes import band_summary, historical_reference
        with tempfile.TemporaryDirectory() as temp, duckdb.connect() as c:
            cfg = self.fixture(Path(temp))
            data = DataRepository(c, cfg, 10040, history_points=2)
            result = dict(_decision_positions(data, {(1,2,3): [dict(item_id=1,recent_start=1)]}, 10, 10040))[1]
            self.assertEqual(result['window'], dict(start_inclusive=40,end_exclusive=10040))
            self.assertEqual(result['recent_price_scope']['products'], 1)
            self.assertNotIn('historical_reference', result)
            _,panel=next(category_panels(data,[(1,2,3)],10040))
            reference=historical_reference(panel,10)
            self.assertEqual(reference['products'], 3)
            self.assertEqual(reference['observation_range'], [1,50])
            band = band_summary({((1,2,3),1): dict(purchases=2,purchased_products=1,selling_shops=1)},
                                (1,2,3), 1000, 40, 10040)
            self.assertEqual(band['evidence_status'], 'sparse')
            self.assertEqual(band['start_inclusive'], result['window']['start_inclusive'])
            self.assertEqual(band['end_exclusive'], result['window']['end_exclusive'])

    def test_exact_facts_and_historical_queries_with_separate_cursor(self):
        with tempfile.TemporaryDirectory() as temp, duckdb.connect() as c:
            root=Path(temp);cfg=self.fixture(root);cache=root/'cache'
            def queries(connection):
                data=DataRepository(connection, cfg, 4, history_points=2)
                return (owned_profiles(data,10,4),shop_period(data,10,4),
                        dict(_decision_positions(data,{(1,2,3):[dict(item_id=1,recent_start=0)]},10,4)),
                        list(category_panels(data,[(1,2,3)],4)))
            expected=queries(c)
            self.assertEqual(build_cache(cfg,cache)['rows'],{'by_category':5,'by_seller':5})
            attach_cache(c,cfg,cache)
            cursor=c.cursor()
            try:self.assertEqual(queries(cursor),expected)
            finally:cursor.close()
            for table in ('by_category','by_seller'):
                source=f'read_parquet({literal(cfg.data/"seller_business/part_*.parquet")})'
                cached=f'seller_market_cache.{table}'
                for left,right in ((source,cached),(cached,source)):
                    self.assertEqual(c.execute(f'SELECT * FROM {left} EXCEPT ALL SELECT * FROM {right}').fetchall(),[])

    def test_preflight_verified_worker_attachment_does_not_rehash_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg=self.fixture(root);cache=root/'cache'
            build_cache(cfg,cache)
            with duckdb.connect() as parent:
                attach_cache(parent,cfg,cache)
            with duckdb.connect() as worker,patch('agent_simulation.data.market.quotes.sha256',
                    side_effect=AssertionError('Worker must not rehash verified files')):
                attach_cache(worker,cfg,cache,verify_hashes=False)
                self.assertEqual(worker.execute('SELECT count(*) FROM seller_market_cache.by_seller').fetchone(),(5,))

    def test_changed_database_or_source_fails_instead_of_using_stale_cache(self):
        for altered in ('database','source'):
            with self.subTest(altered=altered),tempfile.TemporaryDirectory() as temp,duckdb.connect() as c:
                root=Path(temp);cfg=self.fixture(root);cache=root/'cache';build_cache(cfg,cache)
                path=cache/'market.duckdb' if altered=='database' else cfg.data/'seller_business/part_00.parquet'
                with path.open('ab') as f:f.write(b'changed')
                with self.assertRaisesRegex(ValueError,'Market cache (changed|sources differ)'):
                    attach_cache(c,cfg,cache)


if __name__=='__main__':unittest.main()

class IncrementalQuoteTests(unittest.TestCase):
    fixture=MarketCacheTests.fixture
    def test_incremental_full_history_and_rewind_match_fresh_sql(self):
        from agent_simulation.data.market.quotes import CategorySnapshots,category_snapshot
        with tempfile.TemporaryDirectory() as temp,duckdb.connect() as c:
            cfg=self.fixture(Path(temp));cache=CategorySnapshots();previous=None
            for before in (4,6,50,51,10040,2,52):
                data=DataRepository(c,cfg,before,category_snapshots=cache)
                actual=category_snapshot(data,(1,2,3),before)
                expected=category_snapshot(DataRepository(c,cfg,before),(1,2,3),before)
                self.assertEqual(actual,expected)
                self.assertTrue(all(r['price_time_index']<before for r in actual))
                if before==4:previous=list(actual)
                if before==6:self.assertEqual(previous,actual)
            self.assertEqual(previous[0]['price_time_index'],1)

    def test_recent_quotes_match_filtered_full_snapshot_at_boundaries(self):
        from agent_simulation.data.market.quotes import category_snapshot,RECENT_MARKET_SPAN
        with tempfile.TemporaryDirectory() as temp,duckdb.connect() as c:
            cfg=self.fixture(Path(temp))
            for before in (1,4,50,51,10040,10050,10051):
                data=DataRepository(c,cfg,before);start=max(0,before-RECENT_MARKET_SPAN)
                full=category_snapshot(data,(1,2,3),before)
                recent=category_snapshot(data,(1,2,3),before,start=start)
                fields=('item_id','seller_id','price_raw','price_time_index','price_status','_public')
                expected=[{k:r[k] for k in fields} for r in full if r['price_time_index']>=start]
                self.assertEqual([{k:r[k] for k in fields} for r in recent],expected)
                self.assertTrue(all(start<=r['price_time_index']<before for r in recent))
                if before==4:
                    # A newer invalid quote is not replaced by an older valid one.
                    self.assertIsNone(next(r for r in recent if r['item_id']==2)['price_raw'])
                if before==10051:self.assertEqual(recent,[])

    def test_summary_and_competitors_share_one_bounded_sql_read(self):
        from agent_simulation.data.market import quotes as queries
        with tempfile.TemporaryDirectory() as temp,duckdb.connect() as c:
            cfg=self.fixture(Path(temp));data=DataRepository(c,cfg,10040)
            with patch.object(queries,'rows',wraps=queries.rows) as read:
                summary=dict(_decision_positions(data,{(1,2,3):[dict(item_id=1)]},10,10040))[1]
                snapshot=data.competitor_snapshot(10,1,10040)
                self.assertEqual(read.call_count,1)
                sql=read.call_args.args[1];parameters=read.call_args.args[2]
                self.assertIn('time_index>=?',sql)
                self.assertEqual(parameters[-2:],[10040,40])
                self.assertNotIn('AS last_valid_price',sql)
            self.assertEqual(summary['recent_price_scope']['products'],1)
            self.assertEqual([r['item_id'] for r in snapshot],[4])
            # Explicit historical discovery retains the full past independently.
            self.assertEqual({p['item_id'] for p in data.competitor_pool(10,1,10040)},{2,3,4})

    def test_pricing_batch_reads_each_category_once(self):
        from agent_simulation.agents.context.market import load_market_context
        from agent_simulation.data.market import quotes as queries

        with tempfile.TemporaryDirectory() as temp, duckdb.connect() as c:
            cfg = self.fixture(Path(temp))
            path = cfg.data / 'seller_business/part_00.parquet'
            c.execute('CREATE TABLE original AS SELECT * FROM read_parquet(?)', [str(path)])
            c.execute(f"""COPY (SELECT * REPLACE(
                CASE WHEN seller_id=10 THEN 100+i ELSE 200+i END AS item_id,
                100+i AS category_level3_id
            ) FROM original, range(16) t(i) WHERE item_id IN (1,3))
            TO {literal(path)} (FORMAT PARQUET)""")
            data = DataRepository(c, cfg, 4)
            context = {
                'profile': {'seller_id': 10},
                'history_cutoff_exclusive': 4,
                'owned_products': [{'item_id': 100+i} for i in range(16)],
            }
            with patch.object(queries, 'rows', wraps=queries.rows) as read:
                load_market_context(context, data, history_embedder=None)
            scans = [call for call in read.call_args_list if 'arg_max(struct_pack' in call.args[1]]
            self.assertEqual(len(scans), 16)
            for product in context['owned_products']:
                position = product['market_position']
                self.assertEqual(position['recent']['median_price'], 900)
                peer, = position['competitor_examples']['products']
                self.assertEqual(peer['item_id'], product['item_id'] + 100)

    def test_recent_window_keeps_complete_equal_time_cohort(self):
        from agent_simulation.data.market.quotes import category_snapshot
        with tempfile.TemporaryDirectory() as temp,duckdb.connect() as c:
            cfg=self.fixture(Path(temp));path=cfg.data/'seller_business/part_00.parquet'
            c.execute('CREATE TABLE original AS SELECT * FROM read_parquet(?)',[str(path)])
            c.execute(f"""COPY (SELECT * EXCLUDE(item_id,time_index),
                i+100 AS item_id,10000 AS time_index FROM original,range(4100) t(i)
                WHERE item_id=3) TO {literal(path)} (FORMAT PARQUET)""")
            data=DataRepository(c,cfg,10001)
            self.assertEqual(len(category_snapshot(data,(1,2,3),10001,start=1)),4100)
            self.assertEqual(category_snapshot(data,(1,2,3),10000,start=0),[])
