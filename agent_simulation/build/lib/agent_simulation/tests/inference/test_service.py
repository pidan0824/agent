"""Exercise the spawned inference worker using a batch-aware test double."""
import asyncio
import os
import time
import unittest

from agent_simulation.inference.local.service import LocalInferenceService


class BatchFixture:
    def __init__(self):
        self.generator_id = "fixture"

    def complete_batch(self, jobs):
        time.sleep(max(req.get("delay", 0) for _, req, _ in jobs))
        return [(call_id, len(jobs), os.getpid()) for call_id, _, _ in jobs]


class ServiceTests(unittest.TestCase):
    def test_shared_worker_and_request_routing(self):
        service = LocalInferenceService(BatchFixture, {}, concurrency=2, timeout=5)
        self.addCleanup(service.close)
        async def submit():
            return await asyncio.gather(*(service.acomplete({}, call_id=str(i)) for i in range(8)))
        results = asyncio.run(submit())
        self.assertEqual([r[0] for r in results], list(map(str, range(8))))
        self.assertEqual(max(r[1] for r in results), 2)
        self.assertEqual({r[2] for r in results}, {service.process.pid})

    def test_deadline_includes_queued_requests_and_discards_late_replies(self):
        service = LocalInferenceService(BatchFixture, {}, concurrency=1, timeout=0.1)
        self.addCleanup(service.close)
        async def submit():
            return await asyncio.gather(service.acomplete({"delay": 0.3}, call_id="slow"),
                                        service.acomplete({"delay": 0.3}, call_id="queued"),
                                        return_exceptions=True)
        self.assertTrue(all(isinstance(e, TimeoutError) for e in asyncio.run(submit())))
        time.sleep(0.35)
        self.assertEqual(service.complete({}, call_id="next")[0], "next")
