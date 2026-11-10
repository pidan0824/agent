"""Independent response batches overlap without partial Customer settlement."""
import asyncio
import unittest

from agent_simulation.agents.customer import CustomerAgent
from agent_simulation.agents.prompts.customer import decode_input
from agent_simulation.tests.fixtures import MarketFixture, ProfileCatalogFixture, DisplayBudgetTokenizer, answer


class AsyncModel:
    def complete(self, request, *, call_id):
        raise AssertionError('Expected async dispatch')


class CustomerParallelTests(MarketFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.runtime.catalog = ProfileCatalogFixture(self.runtime)
        self.runtime.inputs.history_embedder = None
        self.pending = self.runtime.execute('customer', 7, 'search_products', {'query': '水杯'})
        self.ids = [p['item_id'] for p in self.pending['displayed_items']]

    def test_next_batch_starts_while_first_batch_is_still_running(self):
        third_started = asyncio.Event()
        before = self.runtime.snapshot()

        class Model(AsyncModel):
            async def acomplete(inner, request, *, call_id):
                context = decode_input(request['messages'][1]['content'])
                index = self.ids.index(context['displayed_items'][0]['item_id']) + 1
                if index == 1:
                    await asyncio.wait_for(third_started.wait(), 1)
                if index == 3:
                    third_started.set()
                self.assertEqual(self.runtime.snapshot(), before)
                item = context['displayed_items'][0]['item_id']
                return answer({'clicked_item_ids': [item], 'purchased_item_ids': []})

        run = CustomerAgent(self.runtime, Model(), DisplayBudgetTokenizer(1),
                            response_concurrency=2).run(7)
        ids = [p['item_id'] for p in self.pending['displayed_items']]
        self.assertEqual([r['item_ids'][0] for r in run['response_batches']], ids)
        self.assertEqual(run['result']['clicked_item_ids'], ids)
        self.assertEqual(len(self.runtime.snapshot()['customers']['7']['simulation_history']), 1)

    def test_overlap_order_and_single_settlement(self):
        initial = self.runtime.snapshot()
        seen, finished, call_ids = [], [], []
        gate = asyncio.Event()
        class Model(AsyncModel):
            async def acomplete(inner, request, *, call_id):
                context = decode_input(request['messages'][1]['content'])
                item = context['displayed_items'][0]['item_id']
                seen.append(item)
                call_ids.append(call_id)
                if len(seen) == 2:
                    gate.set()
                await asyncio.wait_for(gate.wait(), 2)
                if item == self.ids[0]:
                    await asyncio.sleep(0.01)
                self.assertEqual(self.runtime.snapshot(), initial)
                finished.append(item)
                return answer({'clicked_item_ids': [item], 'purchased_item_ids': []})
        run = CustomerAgent(self.runtime, Model(), DisplayBudgetTokenizer(1)).run(7)
        ids = [p['item_id'] for p in self.pending['displayed_items']]
        self.assertNotEqual(finished, ids)
        self.assertEqual([r['item_ids'][0] for r in run['response_batches']], ids)
        self.assertEqual(run['result']['clicked_item_ids'], ids)
        self.assertEqual(len(set(call_ids)), len(ids))
        self.assertEqual([call["call_id"] for call in run["model_calls"]], call_ids)
        after = self.runtime.snapshot()
        self.assertEqual(len(after['customers']['7']['simulation_history']), 1)
        self.assertEqual(after['simulation_step'], initial['simulation_step'] + 1)

    def test_failed_batch_cancels_pending_sibling_and_keeps_search_unsettled(self):
        initial = self.runtime.snapshot()
        started, cancelled = asyncio.Event(), []
        class Model(AsyncModel):
            async def acomplete(inner, request, *, call_id):
                context = decode_input(request['messages'][1]['content'])
                batch = self.ids.index(context['displayed_items'][0]['item_id']) + 1
                if batch == 1:
                    await asyncio.wait_for(started.wait(), 2)
                    raise TimeoutError('batch failed')
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.append(batch)
        with self.assertRaisesRegex(TimeoutError, 'batch failed'):
            CustomerAgent(self.runtime, Model(), DisplayBudgetTokenizer(1)).run(7)
        self.assertEqual(cancelled, [2])
        self.assertEqual(self.runtime.snapshot(), initial)

    def test_shared_limit_caps_multiple_customers_and_batches(self):
        other = self.runtime.execute('customer', 8, 'search_products', {'query': '水杯'})
        initial = self.runtime.snapshot()
        active = peak = 0
        class Model(AsyncModel):
            async def acomplete(inner, request, *, call_id):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                try:
                    await asyncio.sleep(0.01)
                    self.assertEqual(self.runtime.snapshot(), initial)
                    return answer({'clicked_item_ids': [], 'purchased_item_ids': []})
                finally:
                    active -= 1
        agent = CustomerAgent(self.runtime, Model(), DisplayBudgetTokenizer(1), response_concurrency=4)
        runs = [agent.prepare_decision(actor) for actor in (7, 8)]
        for run, pending in zip(runs, (self.pending, other)):
            run['displayed_list'] = pending
        async def collect():
            slots = asyncio.Semaphore(2)
            await asyncio.gather(*(agent.choose_response(run, request_slots=slots) for run in runs))
        asyncio.run(collect())
        self.assertEqual(peak, 2)
        self.assertTrue(all(r['result']['status'] == 'staged' for r in runs))
        self.assertEqual(self.runtime.snapshot(), initial)
