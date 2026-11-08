"""Test factual retrieval and memory injection into decision inputs."""

from copy import deepcopy
import unittest
from agent_simulation.tests.fixtures import MarketFixture
from agent_simulation.agents.retrieval.history import seller_history, customer_history
from agent_simulation.data.market.statistics import market_position
from agent_simulation.agents.retrieval.long_term import retrieve_long_term_memory


class RetrievalTests(MarketFixture, unittest.TestCase):
    def test_cached_market_windows_match_raw_records(self):
        from agent_simulation.data.market.quotes import category_rows, category_panels, owned_profiles, COUNTS
        profile = owned_profiles(self.data, 10, 10)[0]
        panel = None
        for start in (0, 2, 4, 5, 6, 9, 2):
            result = category_rows(self.data, profile, 10, start)
            cached = next(iter(self.data._market_panels.values()))
            if panel is not None:
                self.assertIs(panel, cached)
            panel = cached
            for product in result:
                history = [r for r in self.data.records(product['item_id']) if r['time_index'] < 10]
                window = [r for r in history if r['time_index'] >= start]
                for metric in COUNTS:
                    self.assertEqual(product[metric], sum(r[metric] for r in window) if window else None)
                    self.assertEqual(product['cumulative_'+metric], sum(r[metric] for r in history))
                if product['item_id'] == 2:
                    self.assertIsNone(product['price_raw'])
        panels = dict(category_panels(self.data, [(1, 2, 3), (1, 2, 9)], 10))
        self.assertEqual({p['item_id'] for p in panels[(10, 1, 2, 3)]}, {1, 2, 3})
        self.assertEqual({p['item_id'] for p in panels[(10, 1, 2, 9)]}, {4})

    def test_older_history_is_not_recent_memory_or_future(self):
        events = []
        base = self.data.records(1)[0]
        for i in range(1, 12):
            events.append({**base, 'time_index': i, 'price_raw': 100 + i})
        original = deepcopy(events)
        result = seller_history(events, seller_id=10, cutoff=10, recent_n=2, focus_ids=[1])
        indices = [r['index'] for e in result['segments'] for r in e['records']]
        self.assertEqual(set(indices), set(range(1, 8)))
        self.assertEqual(events, original)
        with self.assertRaises(ValueError):
            seller_history(events, seller_id=20, cutoff=10, recent_n=2)

    def test_bge_callback_can_rank_raw_history_without_summaries(self):
        events = [{**self.data.records(1)[0], 'time_index': i} for i in range(1, 9)]
        calls = []
        def encode(text):
            calls.append(text)
            return [1., .5]
        result = seller_history(events, seller_id=10, cutoff=9, recent_n=2, embedder=encode)
        self.assertEqual(result['ranking'], 'rrf_bge')
        self.assertTrue(calls)
        self.assertTrue(all('semantic_similarity' in e['match'] for e in result['segments']))
        self.assertTrue(all(e['evidence_kind'] == 'observed_fact' for e in result['segments']))

    def test_customer_habits_remain_and_future_is_rejected(self):
        events = [{'session_id': i, 'time_index': i, 'query': q,
                   'clicked_items': [], 'purchased_items': []}
                  for i,q in enumerate(['鞋子', '牛奶', '杯子'], 1)]
        result = customer_history(events, (), cutoff=4, recent_n=1, query='保温杯', user_id=7)
        self.assertEqual({e['records'][0]['query'] for e in result['segments']}, {'鞋子', '牛奶'})
        with self.assertRaises(ValueError):
            customer_history(events, (), cutoff=3, recent_n=1)
        with self.assertRaises(ValueError):
            customer_history([{**events[0], 'user_id': 8}], (), cutoff=4, recent_n=1, user_id=7)

    def test_same_cohort_rates_coverage_and_cumulative(self):
        def row(price, exposures, clicks, buys):
            return {'price_raw': price, 'exposure_records': exposures, 'click_records': clicks,
                    'purchase_records': buys, 'cumulative_exposure_records': 100,
                    'cumulative_click_records': 20, 'cumulative_purchase_records': 5}
        own = row(100, 10, 2, 1)
        result = market_position(own, [row(90, 20, 4, 2), row(None, 10, 0, 0),
                                       row(80, None, None, None)],
                                 window={'start_inclusive': 3, 'end_exclusive': 10}, clock='observed')
        self.assertEqual(result['coverage']['comparable_products'], 1)
        for metric in ('price_raw', 'exposure_records', 'click_records', 'purchase_records', 'ctr', 'cvr'):
            self.assertEqual(result[metric]['observed_products'], 1)
        self.assertEqual(result['ctr']['own_value'], .2)
        self.assertEqual(result['cvr']['own_value'], .1)
        self.assertEqual(result['cumulative']['purchase_records']['own_value'], 5)

    def test_memory_filters_before_semantic_ranking(self):
        state = self.runtime.snapshot()
        state['long_term']['customer:7'] = [{
            'owner': 'customer:7', 'generated_step': 9, 'event_count': 99,
            'embedding_model': 'test', 'embedding': [1., 0.],
            'content': 'future', 'source_ids': []}]
        def forbidden(text):
            self.fail('Future memory must be filtered before embedding')
        self.assertEqual(retrieve_long_term_memory(state, 'customer', 7, 'shopping', forbidden, 'test'), [])
        state['long_term']['customer:7'][0]['owner'] = 'customer:8'
        with self.assertRaises(ValueError):
            retrieve_long_term_memory(state, 'customer', 7, 'shopping', forbidden, 'test')


if __name__ == '__main__':
    unittest.main()
