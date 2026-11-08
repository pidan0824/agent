"""Test rank fusion and default BGE selection."""
from copy import deepcopy
import math
import unittest
from agent_simulation.agents.retrieval.ranking import average_ranks
from unittest.mock import patch

from agent_simulation.agents.retrieval.history import _rank, _distance, customer_history, seller_history
from agent_simulation.tests.agents.retrieval.test_history_retrieval_regressions import shopping, business


class SemanticEncoder:
    def encode_query(self, text):
        return [1., 0.]

    def __call__(self, text):
        score = {'high': .99, 'low': .1, 'medium': .5}[text]
        return [score, math.sqrt(1-score*score)]


def candidate(item, entity, semantic, distance=None, index=1):
    return {'item_id':item, 'clock':'observed_time_index', 'end_inclusive':index,
            'match':{'entity_match':entity, 'numeric_distance':distance}, 'search_text':semantic}


class HistoryFusionTests(unittest.TestCase):
    def test_freshness_is_bounded_and_does_not_remove_strong_old_evidence(self):
        from agent_simulation.agents.retrieval.ranking import freshness_factor
        self.assertEqual(freshness_factor(0, 10001), .95)
        self.assertEqual(freshness_factor(0, 11, "simulation_settled_step"), .95)
        self.assertGreaterEqual(freshness_factor(0, 10000000), .9)
        result = _rank([candidate(1, 3, 'high', 0, index=1),
                        candidate(2, 0, 'low', None, index=99999)],
                       'query', SemanticEncoder(), 2, cutoff=100000)
        self.assertEqual(result['segments'][0]['item_id'], 1)
        self.assertLess(result['segments'][0]['match']['freshness_factor'], 1)

    def test_semantics_and_numeric_similarity_can_outweigh_same_item(self):
        result = _rank([candidate(1, 2, 'low', 1), candidate(2, 1, 'high', 0)],
                       'current product', SemanticEncoder(), 5)
        self.assertEqual([s['item_id'] for s in result['segments']], [2,1])
        self.assertGreater(result['segments'][0]['match']['retrieval_score'], result['segments'][1]['match']['retrieval_score'])

    def test_customer_index_preserves_action_roles_without_modifying_records(self):
        old = dict(shopping(1), clicked_items=[{"item_title": "夹克"}],
                   purchased_items=[{"item_title": "猕猴桃"}])
        encoded = []

        def embed(text):
            encoded.append(text)
            return [1., 0.]

        result = customer_history([old, shopping(2)], (), cutoff=3, recent_n=1, user_id=7,
                                  query="水果", embedder=embed)
        self.assertIn("搜索：水杯\n点击：夹克\n购买：猕猴桃", encoded)
        self.assertEqual(result["segments"][0]["records"], [old])

    def test_customer_category_overlap_is_not_absolute_priority(self):
        result = _rank([candidate(1, 1, 'low'), candidate(2, 0, 'high')],
                       'current search', SemanticEncoder(), 5)
        self.assertEqual(result['segments'][0]['item_id'], 2)

    def test_numeric_unknown_is_not_a_perfect_match(self):
        self.assertIsNone(_distance((None,)*6, (None,)*6))
        result = _rank([candidate(1, 1, 'high', None), candidate(2, 1, 'high', .1)],
                       'query', SemanticEncoder(), 5)
        self.assertEqual(result['segments'][0]['item_id'], 2)

    def test_equal_signal_values_receive_equal_ranks(self):
        self.assertEqual(average_ranks([4,4,1,None]), [1.5,1.5,3.,None])
        result = _rank([candidate(1,1,'high',0,index=2), candidate(2,1,'high',0,index=1)],
                       'query', SemanticEncoder(), 5)
        a,b=result['segments']
        self.assertEqual(a['match']['retrieval_score'], b['match']['retrieval_score'])
        self.assertEqual(a['item_id'], 1)

    def test_default_resolves_local_encoder_and_structured_is_explicit(self):
        encoder=lambda text:[1.,0.]
        with patch('agent_simulation.agents.retrieval.bge_encoder.default_encoder', return_value=(encoder,'fixture')) as load:
            automatic = customer_history([shopping(1),shopping(2)], (), cutoff=3, recent_n=1, user_id=7)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(automatic['ranking'], 'rrf_bge')
            explicit = customer_history([shopping(1),shopping(2)], (), cutoff=3, recent_n=1, user_id=7, embedder=None)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(explicit['ranking'], 'rrf_structured')

    def test_future_and_foreign_records_never_reach_encoder(self):
        def encode(text):
            self.assertNotIn('future secret', text)
            return [1.,0.]
        events = [business(1,1),business(1,2),business(1,10,'future secret')]
        before = deepcopy(events)
        result = seller_history(events,seller_id=10,cutoff=3,recent_n=1,embedder=encode)
        self.assertEqual([r['index'] for s in result['segments'] for r in s['records']], [1])
        self.assertEqual(events,before)
        with patch('agent_simulation.agents.retrieval.bge_encoder.default_encoder', side_effect=AssertionError('Must filter first')):
            with self.assertRaises(ValueError):
                customer_history([{**shopping(1),'user_id':8}],(),cutoff=2,recent_n=1,user_id=7)
            with self.assertRaises(ValueError):
                customer_history([shopping(2)],(),cutoff=2,recent_n=1,user_id=7)

    def test_unavailable_default_encoder_does_not_silently_change_the_ranking(self):
        with patch('agent_simulation.agents.retrieval.bge_encoder.default_encoder', side_effect=ValueError('Missing BGE')):
            with self.assertRaisesRegex(ValueError, 'Missing BGE'):
                customer_history([shopping(1), shopping(2)], (), cutoff=3, recent_n=1, user_id=7)
