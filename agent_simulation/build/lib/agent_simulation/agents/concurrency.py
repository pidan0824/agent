"""Run bounded decision workers; preserve input order and cancel on failure."""

import asyncio

from agent_simulation.data.common import require


async def map_decisions(items, decide, *, concurrency):
    require(type(concurrency) is int and concurrency > 0, "concurrency must be a positive integer")
    remaining = iter(items)
    results = []

    async def worker():
        for item in remaining:
            index = len(results)
            results.append(None)
            results[index] = await decide(item)

    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*tasks)
        return results
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
