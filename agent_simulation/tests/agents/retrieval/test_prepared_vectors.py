"""Cache invariant math while applying each query's authorization and clock bounds."""
import unittest
from unittest.mock import patch
import numpy as np
from agent_simulation.agents.retrieval.similarity import PreparedVector, cosine_scores
from agent_simulation.agents.retrieval.history import SellerItemWindow, seller_segment
from agent_simulation.tests.agents.retrieval.test_history_retrieval_regressions import business


class PreparedVectorTests(unittest.TestCase):
    def test_only_eligible_embeddings_are_validated(self):
        embeddings = [None, PreparedVector([1., 0., 0.]), [1., 0.]]
        metadata = [{'owner': 2, 'time': 0}, {'owner': 1, 'time': 5}, {'owner': 1, 'time': 0}]
        predicates = [('owner', '==', 1), ('time', '<', 5)]
        self.assertEqual(
            cosine_scores([1., 0.], embeddings, metadata=metadata, predicates=predicates),
            {2: 1.0},
        )
        predicates[-1] = ('time', '<=', 5)
        with self.assertRaisesRegex(ValueError, 'incompatible embedding'):
            cosine_scores([1., 0.], embeddings, metadata=metadata, predicates=predicates)

    def test_shared_vectors_reuse_scores_without_merging_records_or_authorization(self):
        prepared = PreparedVector([1.,2.])
        vectors = [prepared] * 5
        meta = [{'owner':1,'time':t} for t in range(4)] + [{'owner':2,'time':0}]
        options = dict(metadata=meta, predicates=[('owner','==',1),('time','<',3)])
        for query in ([2.,1.],[-1.,2.]):
            expected = cosine_scores(query, [p.values for p in vectors], **options)
            with patch('numpy.dot', wraps=np.dot) as dot:
                self.assertEqual(cosine_scores(query, vectors, **options), expected)
                self.assertEqual(dot.call_count, 1)
            self.assertEqual(set(expected), {0,1,2})

    def test_norms_are_reused_but_query_authorization_is_not(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(30, 512)); vectors[0] = 0
        prepared = [PreparedVector(v) for v in vectors]
        meta = [dict(owner=i % 2, time=i) for i in range(30)]
        for owner, cutoff in ((0, 10), (1, 20), (0, 30)):
            query = rng.normal(size=512)
            options = dict(metadata=meta, predicates=[('owner', '==', owner), ('time', '<', cutoff)])
            expected = cosine_scores(query, vectors, **options)
            with patch('numpy.linalg.norm', wraps=np.linalg.norm) as norm:
                self.assertEqual(cosine_scores(query, prepared, **options), expected)
                self.assertEqual(norm.call_count, 1)
        original = prepared[1].values.copy()
        vectors[1] *= 2
        np.testing.assert_array_equal(prepared[1].values, original)
        self.assertFalse(prepared[1].values.flags.writeable)
        for invalid in ([float('nan')], [], [[1, 2]]):
            with self.assertRaises(ValueError): PreparedVector(invalid)
        with self.assertRaises(ValueError): cosine_scores([1], prepared)

    def test_unqueried_partial_segments_are_not_prepared_repeatedly(self):
        window = SellerItemWindow(1, recent_n=10)
        with patch('agent_simulation.agents.retrieval.history.seller_segment', wraps=seller_segment) as build:
            for index in range(1, 1002):
                window.append('observed_time_index', business(1, index))
            self.assertEqual(build.call_count, 330)
            segments = list(window.segments())
            self.assertEqual(build.call_count, 331)
            self.assertEqual(len(segments[-1]['records']), 1)
            self.assertEqual(list(window.segments()), segments)
            self.assertEqual(build.call_count, 331)
