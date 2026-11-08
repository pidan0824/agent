"""Concurrent target quotes preserve the frozen market and atomic publication."""
import asyncio
import unittest

from agent_simulation.tests.fixtures import MarketFixture, TokenizerFixture, answer
from agent_simulation.agents.seller import SellerAgent
from agent_simulation.agents.prompts.seller import decode_input
from agent_simulation.agents.context.market import simulation_market_cache, simulation_position
from agent_simulation.inference.local.service import LocalInferenceService


def quote(request, call_id, price):
    item, = decode_input(request['messages'][1]['content'])['decision']['requested_item_ids']
    return answer({'prices': [{'item_id': item, 'price_raw': price}]})


class QuoteBackend:
    generator_id = 'batch-fixture'

    def complete_batch(self, jobs):
        # Encode batch size in the quoted price to observe the real worker dispatch.
        return [quote(request, call_id, 100 + len(jobs)) for call_id, request, _ in jobs]


class AsyncModel:
    def complete(self, request, *, call_id):
        raise AssertionError("Expected async dispatch")


class SellerParallelTests(MarketFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.runtime.inputs.history_embedder = None

    def test_overlapping_quotes_finish_out_of_order_and_publish_once(self):
        before = self.runtime.snapshot()
        started, finished, call_ids = [], [], []
        gate = asyncio.Event()
        class Model(AsyncModel):
            async def acomplete(inner, request, *, call_id):
                item, = decode_input(request['messages'][1]['content'])['decision']['requested_item_ids']
                started.append(item)
                call_ids.append(call_id)
                if len(started) == 2:
                    gate.set()
                await asyncio.wait_for(gate.wait(), 2)
                if item == 1:
                    await asyncio.sleep(0.01)
                self.assertEqual(self.runtime.snapshot(), before)
                finished.append(item)
                return quote(request, call_id, 100 + item)
        run = SellerAgent(self.runtime, Model(), TokenizerFixture(), price_concurrency=2).run(10)
        self.assertEqual(finished, [3, 1])
        self.assertEqual(len(set(call_ids)), 2)
        self.assertEqual([call["call_id"] for call in run["model_calls"]], call_ids)
        self.assertEqual([r['item_ids'] for r in run['price_batches']], [[1], [3]])
        after = self.runtime.snapshot()
        self.assertEqual(after['simulation_step'], before['simulation_step'] + 1)
        self.assertEqual({r['price_raw'] for r in after['price_actions']}, {101, 103})

    def test_failed_quote_cancels_sibling_and_publishes_nothing(self):
        before = self.runtime.snapshot()
        started, cancelled = asyncio.Event(), []
        class Model(AsyncModel):
            async def acomplete(inner, request, *, call_id):
                item, = decode_input(request['messages'][1]['content'])['decision']['requested_item_ids']
                if item == 1:
                    await asyncio.wait_for(started.wait(), 2)
                    raise TimeoutError('quote failed')
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.append(item)
        with self.assertRaisesRegex(TimeoutError, 'quote failed'):
            SellerAgent(self.runtime, Model(), TokenizerFixture()).run(10)
        self.assertEqual(cancelled, [3])
        self.assertEqual(self.runtime.snapshot(), before)

    def test_shared_request_limit_applies_across_shops_and_targets(self):
        active = peak = 0
        before = self.runtime.snapshot()
        class Model(AsyncModel):
            async def acomplete(inner, request, *, call_id):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                try:
                    await asyncio.sleep(0.01)
                    self.assertEqual(self.runtime.snapshot(), before)
                    return quote(request, call_id, 123)
                finally:
                    active -= 1
        agent = SellerAgent(self.runtime, Model(), TokenizerFixture(), price_concurrency=4)
        runs = [agent.prepare_review(actor) for actor in (10, 20, 40)]
        baseline = self.runtime.freeze()
        async def collect():
            slots = asyncio.Semaphore(2)
            return await asyncio.gather(*(agent.review_prices(run, baseline=baseline, request_slots=slots) for run in runs))
        proposals = asyncio.run(collect())
        self.assertEqual(peak, 2)
        self.assertEqual(sum(len(p['result']['prices']) for p in proposals), 4)
        self.assertEqual(self.runtime.snapshot(), before)

    def test_target_requests_reach_existing_inference_batch(self):
        service = LocalInferenceService(QuoteBackend, {}, concurrency=2, timeout=5)
        self.addCleanup(service.close)
        SellerAgent(self.runtime, service, TokenizerFixture()).run(10)
        self.assertEqual({r['price_raw'] for r in self.runtime.snapshot()['price_actions']}, {102})

    def test_indexed_market_matches_uncached_market(self):
        state = self.runtime.snapshot()
        cache = simulation_market_cache(state)
        for item in (1, 3, 2, 4):
            self.assertEqual(simulation_position(state, item, cache=cache), simulation_position(state, item))
