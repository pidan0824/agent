"""Test historical segment selection and retrieval ranking."""
import unittest
from unittest.mock import patch

from agent_simulation.agents.retrieval.history import SellerItemWindow, customer_history, seller_history
from agent_simulation.tests.fixtures import MarketFixture
from agent_simulation.tests.fixtures import ProfileCatalogFixture


def shopping(index, simulated=False, category=None):
    return {('settled_step' if simulated else 'time_index'): index,
            ('request_id' if simulated else 'session_id'): index, 'query': '水杯',
            'clicked_items': [] if category is None else [{'category_level3_id': category}],
            'purchased_items': []}


def business(item, index, title='保温杯', clicks=2, simulated=False):
    return {'seller_id': 10, 'item_id': item, 'item_title': title,
            ('settled_step' if simulated else 'time_index'): index,
            'category_level1_id': 1, 'category_level2_id': 2, 'category_level3_id': 3,
            'price_raw': 100, 'price_status': 'observed',
            'exposure_records': 10, 'click_records': clicks, 'purchase_records': 1}


class Encoder:
    def __init__(self):
        self.queries = []

    def encode_query(self, text):
        self.queries.append(text)
        return [1., 0.]

    def __call__(self, text):
        return [1., 0.]


class RetrievalRegressionTests(unittest.TestCase):
    def test_repeated_outcomes_at_different_times_remain_distinct_evidence(self):
        result = seller_history([business(1, t) for t in range(1, 19)],
                                seller_id=10, cutoff=19, recent_n=3,
                                focus_ids=[1], embedder=None)
        self.assertEqual(len(result['segments']), 5)
        self.assertEqual([(s['start_inclusive'], s['end_inclusive']) for s in result['segments']],
                         [(13,15), (10,12), (7,9), (4,6), (1,3)])
        self.assertEqual({r['index'] for s in result['segments'] for r in s['records']}, set(range(1,16)))

    def test_only_identical_observation_references_are_deduplicated(self):
        from copy import deepcopy
        from agent_simulation.agents.retrieval.history import seller_segment, PreparedSellerHistory
        current = business(1, 4)
        observed = seller_segment(1, 'observed_time_index', [business(1, 1)])
        settled = seller_segment(1, 'simulation_settled_step', [business(1, 1, simulated=True)])
        result = PreparedSellerHistory([observed, deepcopy(observed), settled],
            {1:[current]}, {1:current}, seller_id=10, cutoff=5, simulation_before=2,
        ).retrieve([1], embedder=None)
        self.assertEqual(len(result['segments']), 2)
        self.assertEqual({s['clock'] for s in result['segments']},
                         {'observed_time_index','simulation_settled_step'})

    def test_repeated_patterns_share_comparison_work_but_keep_all_dates(self):
        from agent_simulation.agents.retrieval.history import _distance
        rows = [business(1,60)] + [business(2,t) for t in range(1,61)]
        with patch('agent_simulation.agents.retrieval.history._distance', wraps=_distance) as compare:
            result = seller_history(rows, seller_id=10, cutoff=61, recent_n=2,
                                    focus_ids=[1], embedder=None, limit=100)
        self.assertLessEqual(compare.call_count, 3)
        self.assertEqual({r['index'] for s in result['segments'] for r in s['records']}, set(range(1,61)))

    def test_window_eviction_across_separate_clocks(self):
        window = SellerItemWindow(1, recent_n=2, stride=2)
        for i in (40, 50, 60, 70):
            window.append("observed_time_index", business(1, i))
        for i in range(1, 7):
            window.append("simulation_settled_step", business(1, i, simulated=True))
        self.assertEqual([r["settled_step"] for r in window.recent_rows()], [5, 6])
        self.assertEqual(
            [(s["clock"], [r["index"] for r in s["records"]]) for s in window.segments()],
            [("observed_time_index", [40, 50, 60]), ("observed_time_index", [60, 70]),
             ("simulation_settled_step", [1, 2, 3]), ("simulation_settled_step", [3, 4])],
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            SellerItemWindow(1, recent_n=0)

    def test_other_products_include_recent_and_older_but_target_recent_is_direct(self):
        rows = [business(item, t) for item in (1, 2) for t in range(1, 7)]
        rows += [business(3, 6), business(3, 7)]  # Cold product and cutoff row.
        result = seller_history(rows, seller_id=10, cutoff=7, recent_n=2,
                                focus_ids=[1], embedder=None, limit=100)
        by_item = {}
        for segment in result["segments"]:
            by_item.setdefault(segment["item_id"], set()).update(r["index"] for r in segment["records"])
        self.assertEqual(by_item, {1: set(range(1, 5)), 2: set(range(1, 7)), 3: {6}})

    def test_unchanged_window_reuses_recent_segments(self):
        from agent_simulation.agents.retrieval.history import seller_segment
        window = SellerItemWindow(1, recent_n=2)
        window.append("observed_time_index", business(1, 1))
        with patch("agent_simulation.agents.retrieval.history.seller_segment", wraps=seller_segment) as build:
            first = list(window.segments(include_recent=True))
            self.assertEqual(list(window.segments(include_recent=True)), first)
            self.assertEqual(build.call_count, 1)
            window.append("observed_time_index", business(1, 2))
            self.assertEqual(len(list(window.segments(include_recent=True))[0]["records"]), 2)
            self.assertEqual(build.call_count, 2)

    def test_incremental_seller_text_features_match_cold_build_after_changes(self):
        from agent_simulation.agents.retrieval.history import (
            _feature_text, _features, PreparedSellerHistory,
        )
        windows = {item: SellerItemWindow(item, recent_n=2) for item in (1, 2)}
        observed = [business(item, t, clicks=t) for t in range(1, 9) for item in (1, 2)]
        for row in observed:
            windows[row['item_id']].append('observed_time_index', row)

        def retrieve(focus, cutoff):
            recent = {item: window.recent_rows() for item, window in windows.items()}
            return PreparedSellerHistory(
                [s for window in windows.values() for s in window.segments(include_recent=True)],
                recent, {item: rows[-1] for item, rows in recent.items()}, seller_id=10,
                cutoff=cutoff,
                window_features={item: w.recent_features() for item, w in windows.items()}
            ).retrieve([focus], embedder=Encoder(), limit=100)

        retrieve(1, 9)
        with patch('agent_simulation.agents.retrieval.history._feature_text', wraps=_feature_text) as text, \
             patch('agent_simulation.agents.retrieval.history._features', wraps=_features) as features:
            first = retrieve(1, 9)
            self.assertEqual(text.call_count, 1)  # Current query only, no document regeneration.
            self.assertEqual(features.call_count, 0)
            second = retrieve(1, 10009)
            self.assertNotEqual(first['segments'][0]['match']['freshness_factor'],
                                second['segments'][0]['match']['freshness_factor'])
        for focus in (1, 2):
            self.assertEqual(retrieve(focus, 9), seller_history(observed, seller_id=10,
                cutoff=9, recent_n=2, focus_ids=[focus], embedder=Encoder(), limit=100))
        changed = business(2, 9, title='新标题', clicks=3)
        observed.append(changed)
        windows[2].append('observed_time_index', changed)
        self.assertEqual(retrieve(1, 10), seller_history(observed, seller_id=10,
            cutoff=10, recent_n=2, focus_ids=[1], embedder=Encoder(), limit=100))
        first['segments'][0]['records'][0]['price_raw'] = -1
        self.assertTrue(all(r['price_raw'] > 0 for seg in retrieve(1, 10)['segments'] for r in seg['records']))

    def test_unknown_categories_cast_no_match_vote_for_either_role(self):
        for category, name in ((0, "UNKNOWN"), (3, "UNKNOWN")):
            events = [shopping(i, category=category) for i in (1, 2)]
            for e in events:
                e["clicked_items"][0]["category_level3_name"] = name
            segment, = customer_history(events, [], cutoff=3, recent_n=1, embedder=None)["segments"]
            self.assertEqual(segment["match"]["entity_match"], 0)
            rows = [business(1, 1), business(2, 1), business(2, 2)]
            for r in rows:
                r.update(category_level3_id=category, category_level3_name=name)
            segments = seller_history(rows, seller_id=10, cutoff=3, recent_n=1,
                                      focus_ids=[1], embedder=None)["segments"]
            self.assertTrue(segments)
            self.assertTrue(all(s["match"]["entity_match"] == 0 for s in segments))

    def test_unsorted_prefix_uses_latest_profile_and_order_independent_hits(self):
        rows = [business(item, t, f"版本{t}") for t in (1, 2) for item in (1, 2, 3)]
        def retrieve(records):
            encoder = Encoder()
            hits = seller_history(records, seller_id=10, cutoff=3, recent_n=1,
                                  embedder=encoder)["segments"]
            return hits, encoder.queries
        self.assertEqual(retrieve(rows), retrieve(list(reversed(rows))))

    def test_prepared_segments_reject_foreign_or_future_before_encoding(self):
        from agent_simulation.agents.retrieval.history import seller_segment, PreparedSellerHistory

        current = business(1, 2)
        for row in (dict(current, seller_id=11), dict(current, time_index=3)):
            segment = seller_segment(1, "observed_time_index", [row])
            for encoder in (None, Encoder()):
                with self.subTest(row=row, encoder=encoder), self.assertRaises(ValueError):
                    PreparedSellerHistory([segment], {1: [current]}, {1: current},
                                         seller_id=10, cutoff=3).retrieve(embedder=encoder)
                if encoder:
                    self.assertEqual(encoder.queries, [])

    def test_customer_settled_records_require_a_cutoff_even_inside_recent_window(self):
        with self.assertRaises(ValueError):
            customer_history([], [shopping(5, True)], cutoff=10, recent_n=2, embedder=None)

    def test_response_memory_query_uses_visible_offer_prices(self):
        from agent_simulation.agents.retrieval.retrieval_text import query_text
        for price in (780, 1560):
            query = query_text("customer", {"query": "痘痘贴", "displayed_items": [
                {"category_level3_name": "祛痘贴片", "price_raw": price}, {"price_raw": None}]})
            self.assertIn("祛痘贴片", query)
            self.assertIn(f"中位 {price}", query)

    def test_customer_freshness_uses_separate_clock_units(self):
        result = customer_history([shopping(900000)],
            [shopping(i, True) for i in (1, 2, 3)], cutoff=900001, recent_n=1,
            simulation_before=4, user_id=7)
        self.assertEqual([(s['clock'], s['end_inclusive']) for s in result['segments']],
                         [('observed_time_index', 900000), ('simulation_settled_step', 2),
                          ('simulation_settled_step', 1)])

    def test_clock_recency_does_not_override_entity_relevance(self):
        result = customer_history([shopping(900000, category=3)],
            [shopping(1, True), shopping(2, True, category=3)],
            cutoff=900001, recent_n=1, simulation_before=3, user_id=7)
        self.assertEqual(result['segments'][0]['clock'], 'observed_time_index')

    def test_seller_freshness_uses_separate_clock_units(self):
        result = seller_history([business(1, 900000)], seller_id=10, cutoff=900001,
            recent_n=1, simulated=[business(1, i, simulated=True) for i in range(1, 6)],
            simulation_before=6, embedder=Encoder())
        self.assertEqual([(s['clock'], s['end_inclusive']) for s in result['segments']],
                         [('observed_time_index', 900000), ('simulation_settled_step', 4),
                          ('simulation_settled_step', 3)])
        self.assertNotIn(5, [r['index'] for s in result['segments'] for r in s['records']])

    def test_overlap_covers_a_boundary_without_repeating_the_tail(self):
        rows = [business(1, i) for i in range(1, 7)]
        for i in (3, 4, 5):
            rows[i]['price_raw'] = 90
        def windows(stride):
            result = seller_history(rows, seller_id=10, cutoff=7, recent_n=1,
                                    embedder=None, segment_stride=stride)
            return sorted(tuple(r['index'] for r in s['records']) for s in result['segments'])
        self.assertEqual(windows(3), [(1, 2, 3), (4, 5)])
        self.assertEqual(windows(2), [(1, 2, 3), (3, 4, 5)])

    def test_cross_item_query_uses_focal_title_and_state(self):
        encoder = Encoder()
        rows = [business(1, 8, '目标保温杯', 4), business(2, 1, '同店陶瓷杯'),
                business(2, 4, '同店陶瓷杯')]
        result = seller_history(rows, seller_id=10, cutoff=9, recent_n=1,
                                focus_ids=[1], embedder=encoder)
        self.assertEqual(result['segments'][0]['item_id'], 2)
        self.assertEqual(len(encoder.queries), 1)
        self.assertIn('目标保温杯', encoder.queries[0])
        self.assertIn('点击/曝光：0.4', encoder.queries[0])
        self.assertNotIn('同店陶瓷杯', encoder.queries[0])

    def test_multi_focus_uses_same_numeric_target_for_semantics(self):
        rows = [business(1, 8, '高点击商品', 8), business(3, 8, '低点击商品', 2),
                business(2, 1, '待检索商品'), business(2, 4, '待检索商品')]
        for focus in ([1, 3], [3, 1]):
            encoder = Encoder()
            result = seller_history(rows, seller_id=10, cutoff=9, recent_n=1,
                                    focus_ids=focus, embedder=encoder)
            self.assertIn('低点击商品', encoder.queries[0])
            self.assertEqual(result['segments'][0]['match']['numeric_distance'], 0)

    def test_same_item_query_still_uses_own_recent_state(self):
        encoder = Encoder()
        seller_history([business(1, 1), business(1, 4, clicks=6)], seller_id=10,
                       cutoff=9, recent_n=1, focus_ids=[1], embedder=encoder)
        self.assertIn('点击/曝光：0.6', encoder.queries[0])

    def test_cross_item_missing_category_uses_focal_query_at_lower_tier(self):
        encoder = Encoder()
        rows = [business(1, 8, '目标商品'), business(2, 1, '待检索商品'),
                business(2, 4, '待检索商品')]
        rows[1]['category_level3_id'] = rows[2]['category_level3_id'] = None
        result = seller_history(rows, seller_id=10, cutoff=9, recent_n=1,
                                focus_ids=[1], embedder=encoder)
        self.assertEqual(result['segments'][0]['match']['entity_match'], 0)
        self.assertIn('目标商品', encoder.queries[0])

    def test_multi_focus_numeric_tie_is_stable(self):
        rows = [business(1, 8, '目标一'), business(3, 8, '目标三'),
                business(2, 1), business(2, 4)]
        for focus in ([1, 3], [3, 1]):
            encoder = Encoder()
            seller_history(rows, seller_id=10, cutoff=9, recent_n=1,
                           focus_ids=focus, embedder=encoder)
            self.assertIn('目标一', encoder.queries[0])


class RuntimeRetrievalOnceTests(MarketFixture, unittest.TestCase):
    def test_runtime_each_stage_retrieves_once(self):
        self.runtime.catalog = ProfileCatalogFixture(self.runtime)
        for response in (False, True):
            if response:
                self.runtime.execute('customer', 7, 'search_products', {'query': '保温杯'})
            from agent_simulation.agents.retrieval.history import PreparedCustomerHistory
            with patch.object(PreparedCustomerHistory, 'retrieve', autospec=True,
                              side_effect=PreparedCustomerHistory.retrieve) as calls:
                result = self.runtime.inputs.customer(7, response=response)
            self.assertEqual(calls.call_count, 1)
            self.assertEqual(calls.call_args.args[1], '保温杯' if response else '')
            self.assertEqual(len(result['short_term_memory']['search_history']), 1)
