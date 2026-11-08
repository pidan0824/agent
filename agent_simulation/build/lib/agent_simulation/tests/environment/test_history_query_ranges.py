"""Test observed and simulated history query windows."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from agent_simulation.environment.operations.queries import query_business
from agent_simulation.environment.operations.api import dispatch


class HistoryRangeTests(unittest.TestCase):
    def setUp(self):
        self.data = SimpleNamespace(cutoff=100, business=Mock(), discover=Mock(), public=Mock())
        self.state = {"simulation_step": 2}
        self.args = {"item_ids": [1], "before_time_index": 100, "limit": 1}

    def test_invalid_ranges_are_rejected_before_source_reads(self):
        bad = [
            {"before_time_index": 101},
            {"start_time_index": 101},
            {"before_simulation_step": 100},
            {"start_simulation_step": 3, "before_simulation_step": 2},
            {"before_logical_time": 1},
        ]
        for change in bad:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    query_business(self.state, self.data, 1, {**self.args, **change})
        self.data.business.assert_not_called()

    def test_invalid_tick_ranges_are_rejected_before_source_reads(self):
        state = {**self.state, "clock": {"start": 100, "time": 120, "tick_size": 10}}
        for change in [
            {"before_logical_time": 130},
            {"before_logical_time": 115},
            {"before_logical_time": 120, "before_simulation_step": 2},
        ]:
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    query_business(state, self.data, 1, {**self.args, **change})
        self.data.business.assert_not_called()

    def test_public_ranges_are_checked_before_catalog_access(self):
        runtime = SimpleNamespace(data=self.data)
        for name, args in [
            ("discover_public_products", {"focal_item_id": 1, "before_time_index": 101}),
            ("query_public_products", {**self.args, "before_time_index": 101}),
        ]:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    dispatch(runtime, self.state, "seller", 1, [1], name, args)
        self.data.discover.assert_not_called()
        self.data.public.assert_not_called()


if __name__ == "__main__":
    unittest.main()
