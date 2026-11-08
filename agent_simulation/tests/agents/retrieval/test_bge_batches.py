"""Batch text chunks without truncation; immutable text vectors survive request boundaries."""
from unittest.mock import patch
import unittest
import numpy as np
from agent_simulation import PROJECT_ROOT
from agent_simulation.agents.retrieval.bge_encoder import LocalBGE
from agent_simulation.inference import resolve_device
from agent_simulation.agents.retrieval.history import CachedHistoryEncoder


class BGEBatchTests(unittest.TestCase):
    def test_auto_device_prefers_cuda_then_mps_and_explicit_gpu_fails_when_unavailable(self):
        for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")):
            with (self.subTest(expected=expected),
                  patch("torch.cuda.is_available", return_value=cuda),
                  patch("torch.backends.mps.is_available", return_value=mps)):
                self.assertEqual(resolve_device("auto").type, expected)
                self.assertEqual(resolve_device("cpu").type, "cpu")
        with patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "MPS is unavailable"):
                resolve_device("mps")

    def test_batched_vectors_match_single_text_and_cache_misses_share_forward(self):
        encoder = LocalBGE(PROJECT_ROOT/'models/bge-small-zh-v1.5')
        self.assertEqual(next(encoder.model().parameters()).device.type, encoder.device.type)
        texts = ['保温杯耐用方便清洗', '轻便运动鞋', '卧室台灯护眼', '苹果手机透明壳', '长文本保温杯。'*350]
        with patch.object(encoder.model(), 'forward', wraps=encoder.model().forward) as forward:
            vectors = encoder.encode_many(texts)
            self.assertTrue(all(v.dtype == np.float32 and v.nbytes == 512 * 4
                                and not v.flags.writeable for v in vectors))
            count = forward.call_count
            self.assertLess(count, len(texts))
            self.assertEqual(encoder.encode_many(texts[::-1]), vectors[::-1])
            self.assertEqual(forward.call_count, count)
            encoder.encode_query(texts[0])
            self.assertEqual(forward.call_count, count+1)  # Query instruction has a separate cache key.
        encoder._vectors.clear()
        single = [encoder(text) for text in texts]
        np.testing.assert_allclose(vectors, single, atol=2e-6, rtol=2e-5)
        self.assertTrue(all(abs(np.linalg.norm(v)-1) < 1e-5 for v in vectors))

    def test_bounded_cache_does_not_reencode_evicted_vectors_within_one_call(self):
        class Encoder:
            def __init__(self): self.calls = []
            def encode_many(self, texts):
                self.calls.append(texts)
                return [[len(text),1.] for text in texts]
        encoder = Encoder()
        cache = CachedHistoryEncoder(encoder, max_entries=2)
        vectors = cache.encode_many(['a','bb','ccc','a'])
        self.assertEqual(encoder.calls, [['a','bb','ccc']])
        self.assertTrue(np.array_equal(vectors[0], vectors[-1]))
        self.assertEqual(len(cache.documents),2)
        self.assertFalse(vectors[0].flags.writeable)
