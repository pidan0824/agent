"""Per-product selection, cold histories and independently checked source fidelity."""

import unittest

from agent_simulation.data.history_records import select_business_history
from agent_simulation.tests.data.fixtures import event


class MemoryTests(unittest.TestCase):
    def test_busy_product_cannot_evict_sparse_product(self):
        events = [event(1, t, 100) for t in range(1, 21)] + [event(2, 1, 200)]
        memory, coverage = select_business_history(events, owner_id=10, cutoff=21, observations_per_item=3)
        rows = [r for m in memory for r in m["content"]]
        self.assertEqual(
            [(r["item_id"], r["time_index"]) for r in rows], [(2, 1), (1, 18), (1, 19), (1, 20)]
        )
        self.assertEqual([c["retained_points"] for c in coverage], [3, 1])

    def test_same_time_distinct_products_and_no_gap_filling(self):
        events = [event(1, 2, 100), event(2, 2, None, "invalid_price"), event(1, 99, 100)]
        memory, _ = select_business_history(events, owner_id=10, cutoff=100, observations_per_item=5)
        self.assertEqual([m["time_index"] for m in memory], [2, 99])
        self.assertEqual(len(memory[0]["content"]), 2)
        self.assertIsNone(memory[0]["content"][1]["price_raw"])

    def test_future_same_index_and_foreign_memory(self):
        rows = [event(1, 1, 100), event(1, 10, 999), event(1, 11, 888)]
        memory, _ = select_business_history(rows, owner_id=10, cutoff=10, observations_per_item=30)
        self.assertEqual(len(memory), 1)
        with self.assertRaisesRegex(ValueError, "Foreign"):
            select_business_history(
                rows + [event(2, 2, 200, seller=20)],
                owner_id=10,
                cutoff=10,
                observations_per_item=1,
            )

    def test_duplicate_and_invalid_n_fail(self):
        row = event(1, 1, 100)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_business_history([row, row], owner_id=10, cutoff=10, observations_per_item=1)
        for n in (0, -1, True):
            with self.assertRaises(ValueError):
                select_business_history([row], owner_id=10, cutoff=10, observations_per_item=n)

    def test_one_total_limit_across_real_and_simulated_history(self):
        observed = [event(1, t, 100) for t in (1, 2, 3)] + [event(2, 1, 200)]
        feedback = [
            {**event(1, 0, 90), "request_id": 1, "search_step": 1, "settled_step": 2},
            {**event(1, 0, 80), "request_id": 3, "search_step": 3, "settled_step": 4},
        ]
        memory, coverage = select_business_history(
            observed,
            owner_id=10,
            cutoff=10,
            observations_per_item=2,
            simulated=feedback,
            simulation_cutoff=4,
        )
        self.assertEqual(coverage[0]["retained_points"], 2)
        self.assertEqual(coverage[0]["observed_points"], 1)
        self.assertEqual(coverage[0]["simulated_points"], 1)
        self.assertEqual(coverage[1]["retained_points"], 1)
        self.assertEqual(memory[-1]["simulation_step"], 2)


if __name__ == "__main__":
    unittest.main()
