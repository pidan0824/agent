"""Shared context tests preserve records, clocks, source alignment, and action freedom."""

from copy import deepcopy
import json
import unittest
from agent_simulation.tests.seller_context_fixture import memory
from agent_simulation.tests.seller_context_fixture import decision_context

from agent_simulation.tests.data.fixtures import event
from agent_simulation.agents.context.pricing import build_pricing_context
from agent_simulation.agents.context.pricing import overview_context
from agent_simulation.agents.prompts.seller import pricing_messages
from agent_simulation.agents.prompts.seller import decode_input


class OverviewTests(unittest.TestCase):
    def source(self, **kwargs):
        history = [
            event(1, 1, 100),
            event(1, 3, None, "invalid_price"),
            event(1, 7, 120),
            event(2, 2, 200),
            event(2, 7, 200),
            event(1, 9, 999),
        ]
        return build_pricing_context(
            history,
            owner_id=10,
            cutoff=9,
            observations_per_item=2,
            public_pages=(),
            requested_item_ids=[1],
            **kwargs
        )[0]

    def test_builder_rejects_foreign_history_and_unregistered_settlements(self):
        observed = [event(1, 1, 100)]
        feedback = {**event(1, 0, 100), "request_id": 1, "search_step": 1, "settled_step": 2}
        params = dict(owner_id=10, cutoff=9, observations_per_item=2,
                      public_pages=(), requested_item_ids=[1])
        with self.assertRaisesRegex(ValueError, "Foreign private business records"):
            build_pricing_context([*observed, event(2, 2, 200, seller=11)], **params)
        with self.assertRaisesRegex(ValueError, "Foreign private business records"):
            build_pricing_context(observed, simulated=[dict(feedback, seller_id=11)],
                                  simulation_cutoff=3, **params)
        with self.assertRaisesRegex(ValueError, "previously observed owned product"):
            build_pricing_context(observed, simulated=[dict(feedback, item_id=2)],
                                  simulation_cutoff=3, **params)
        with self.assertRaisesRegex(ValueError, "Simulation cutoff is required"):
            build_pricing_context(observed, simulated=[feedback], **params)

    def test_builder_consumes_observations_once_and_keeps_exact_cutoffs(self):
        observed = [event(1, 1, 100), event(1, 7, None, "invalid_price"), event(1, 9, 999)]
        feedback = {**event(1, 0, 90), "request_id": 1, "search_step": 1, "settled_step": 2}
        state, coverage = build_pricing_context(
            iter(reversed(observed)), owner_id=10, cutoff=9, observations_per_item=2,
            public_pages=(), requested_item_ids=[1], simulation_cutoff=3,
            simulated=[feedback, dict(feedback, settled_step=3, price_raw=999)],
        )
        product, = state["owned_products"]
        self.assertEqual(product["last_observed_time_index"], 7)
        self.assertEqual(product["last_valid_reference"], {"price_raw": 100, "time_index": 1})
        self.assertEqual(product["observed_price_history"]["all_observations"]["observations"], 2)
        self.assertEqual(product["settled_price_history"]["all_observations"]["observations"], 1)
        self.assertEqual(coverage[0]["simulated_points"], 1)

    def test_overview_uses_exact_retained_windows_not_calendar_gaps(self):
        original = self.source()
        before = deepcopy(original)
        projected = overview_context(original, [1])
        a, b = projected["owned_products"]
        self.assertEqual(
            (a["observed_points"], a["observed_window_start"], a["observed_window_end"]), (2, 3, 7)
        )
        self.assertEqual(
            (b["observed_points"], b["observed_window_start"], b["observed_window_end"]), (2, 2, 7)
        )
        self.assertEqual(a["observed_exposure_records"], 2)
        self.assertEqual(original, before)
        self.assertEqual(len(original["short_term_memory"][-1]["content"]), 2)
        self.assertEqual(len(projected["short_term_memory"][-1]["content"]), 1)
        self.assertEqual(projected["short_term_memory"][-1]["source_ids"], ["item:1:time:7"])
        self.assertIsNone(projected["short_term_memory"][0]["content"][0]["price_raw"])
        self.assertEqual(projected["owned_products"][0]["last_observed_price_raw"], 120)
        self.assertNotIn("observed_estimated_units_sold", a)
        projected["short_term_memory"][-1]["content"][0]["price_raw"] = 999
        self.assertEqual(original, before)

    def test_sft_and_runtime_share_overview_and_retrievable_details(self):
        original = self.source()
        sft = overview_context(original, [1])
        runtime = overview_context(original, [])
        after = overview_context(original, [1])
        self.assertEqual(sft["owned_products"], runtime["owned_products"])
        self.assertEqual(sft["short_term_memory"], after["short_term_memory"])
        self.assertEqual(runtime["short_term_memory"], [])
        self.assertEqual(sft["requested_price_item_ids"], [1])

    def test_other_product_history_can_be_expanded_without_selecting_prices(self):
        result = overview_context(self.source(), [2])
        self.assertEqual(
            {r["item_id"] for e in result["short_term_memory"] for r in e["content"]}, {2}
        )
        with self.assertRaises(ValueError):
            overview_context(self.source(), [20])
        with self.assertRaises(ValueError):
            overview_context(self.source(), [1, 1])
        self.assertEqual(overview_context(self.source(), [])["short_term_memory"], [])

    def test_runtime_observations_and_settlements_keep_separate_clocks(self):
        simulated = [
            {**event(1, 0, 90), "request_id": 1, "search_step": 1, "settled_step": 2},
            {**event(1, 0, 80), "request_id": 3, "search_step": 3, "settled_step": 4},
        ]
        result = overview_context(
            self.source(simulated=simulated, simulation_cutoff=5), [1]
        )
        a = result["owned_products"][0]
        self.assertEqual(a["observed_points"], 0)
        self.assertIsNone(a["observed_click_records"])
        self.assertIsNone(a["observed_window_start"])
        self.assertEqual(a["simulation_points"], 2)
        self.assertEqual(a["simulation_exposure_records"], 2)
        self.assertEqual(a["last_observed_price_raw"], 120)
        self.assertEqual(result["short_term_memory"][-1]["content"][0]["price_raw"], 80)

    def test_official_input_representation_roundtrips_and_preserves_long_memory(self):
        original = self.source(simulation_cutoff=4)
        original['long_term_memory']=[memory('Fixture experience, not a model output.',step=1)]
        for simulation, ids in ((False, [1]), (True, []), (True, [2])):
            state = overview_context(original, ids)
            messages = pricing_messages(state)
            self.assertEqual(decode_input(messages[1]["content"]), decision_context(state))
            self.assertEqual(state["long_term_memory"], original["long_term_memory"])

    def test_seller_projection_keeps_sources_in_input_and_audit(self):
        state = overview_context(self.source(simulation_cutoff=4),[1])
        state['long_term_memory']=[dict(memory('A hypothesis, not an observed fact.',step=3),generator='test:model')]
        before = deepcopy(state)
        audit = {}
        messages = pricing_messages(state, audit_sink=audit)
        self.assertEqual(state, before)
        self.assertEqual(audit["memories"][0]["source_ids"], state["long_term_memory"][0]["source_ids"])
        for field in ("source_ids", "source_templates", "generator", "test:model", "event:test"):
            self.assertNotIn(field, messages[1]["content"])

    def test_price_supervision_is_one_submission_without_query_labels(self):
        state = overview_context(self.source(), [1])
        target = {"prices": [{"item_id": 1, "price_raw": 120.5}]}
        messages = pricing_messages(state, target)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        self.assertEqual(set(messages[-1]), {"role", "content"})
        self.assertEqual(json.loads(messages[-1]["content"]), target)
        self.assertEqual(messages[:2], pricing_messages(state))
        for invalid in (0, -1, float("nan"), True, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                pricing_messages(state, {"prices": [{"item_id": 1, "price_raw": invalid}]})


if __name__ == "__main__":
    unittest.main()
