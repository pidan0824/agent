"""Workers refill free slots without reordering results or leaking failed tasks."""

import asyncio
import unittest

from agent_simulation.agents.concurrency import map_decisions


class DecisionWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_concurrency_rejects_nonempty_work(self):
        async def decide(item):
            return item

        for concurrency in (0, -1, True, 1.5):
            with self.subTest(concurrency=concurrency):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    await map_decisions([1], decide, concurrency=concurrency)

    async def test_free_slot_advances_past_a_slow_first_item(self):
        third_started = asyncio.Event()
        finished = []
        active = peak = 0

        async def decide(item):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if item == 0:
                await asyncio.wait_for(third_started.wait(), 1)
            if item == 2:
                third_started.set()
            await asyncio.sleep(0)
            finished.append(item)
            active -= 1
            return item * 10

        results = await map_decisions(iter(range(5)), decide, concurrency=2)
        self.assertEqual(results, [0, 10, 20, 30, 40])
        self.assertLess(finished.index(2), finished.index(0))
        self.assertEqual(peak, 2)

    async def test_input_failure_cancels_inflight_work(self):
        cancelled = []

        def items():
            yield 1
            raise ValueError('context failed')

        async def decide(item):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(item)

        with self.assertRaisesRegex(ValueError, 'context failed'):
            await map_decisions(items(), decide, concurrency=2)
        self.assertEqual(cancelled, [1])
