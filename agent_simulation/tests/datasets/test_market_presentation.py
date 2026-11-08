"""Test sparse market statistics and time domains."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from agent_simulation.data.market.statistics import market_position
from agent_simulation.agents.context.market import (
    context_view, load_market_context, simulation_market_cache, simulation_position,
)
from agent_simulation.tests.seller_context_fixture import decision_context
from agent_simulation.agents.prompts.seller import decode_input, encode_input
from agent_simulation.tests.fixtures import MarketFixture


def row(price=100, exposures=10, clicks=0, purchases=0, index=8):
    return {'price_raw': price, 'price_time_index': index, 'price_status': 'observed',
            'exposure_records': exposures, 'click_records': clicks, 'purchase_records': purchases,
            'cumulative_exposure_records': exposures, 'cumulative_click_records': clicks,
            'cumulative_purchase_records': purchases}


class MarketPresentationTests(unittest.TestCase):
    def market(self, peers, own=None):
        return market_position(own or row(), peers, window={'start_inclusive':5,'end_exclusive':10},
                               clock='observed_time_index')

    def test_temporal_supplement_includes_cutoff_ties_and_keeps_recent_group(self):
        from agent_simulation.data.market.quotes import extended_price_sample
        peers=[dict(item_id=i,seller_id=i%2,price_raw=i,price_time_index=t)
               for i,t in enumerate([2,3,3,8],1)]
        recent=[peers[-1]]
        expanded=extended_price_sample(peers,recent,10,6,minimum=2)
        self.assertEqual((expanded["products"],expanded["shops"],expanded["added_products"]),(3,2,2))
        self.assertEqual(expanded["observation_range"],[3,8])
        self.assertEqual(expanded["median"],3)
        self.assertEqual(len(recent),1)
        self.assertFalse(extended_price_sample(peers,recent,10,6)["target_reached"])

    def test_sparse_peers_hide_tied_ranks_but_keep_nonzero_coverage_and_own_count(self):
        raw=self.market([row() for _ in range(8)]+[row(clicks=1),row(clicks=2)],row(clicks=3))
        original=deepcopy(raw)
        shown=context_view(raw)
        self.assertEqual(shown["click_records"], {"own_value":3})
        self.assertEqual(shown["exposure_records"], {"own_value":10})
        self.assertEqual(shown["cumulative"]["click_records"], {"own_value":3})
        self.assertEqual(shown["purchase_records"]["p75"],0)
        self.assertEqual(shown["purchase_records"]["nonzero_products"],0)
        self.assertEqual(raw,original)
        self.assertIn("p75",shown["price_raw"])

    def test_peer_traffic_cannot_change_any_agent_visible_market_field(self):
        own = row(clicks=2,purchases=1)
        initial = self.market([row(exposures=8,clicks=3,purchases=1)],own)
        changed = self.market([row(exposures=50,clicks=30,purchases=1)],own)
        self.assertNotEqual(initial["ctr"],changed["ctr"])
        for v in (initial,changed):
            v["simulation"] = deepcopy(v)
        self.assertEqual(context_view(initial),context_view(changed))
        self.assertNotIn("own_percentile",context_view(initial)["price_raw"])

    def test_unpriced_peers_do_not_reveal_recent_traffic_presence(self):
        absent = self.market([row(price=None, exposures=None)])
        present = self.market([row(price=None, exposures=1)])
        self.assertNotEqual(absent["coverage"], present["coverage"])
        self.assertEqual(context_view(absent), context_view(present))

    def test_rates_have_exact_numerator_denominator_and_no_peer_quantile_repetition(self):
        shown=context_view(self.market([row(clicks=2,purchases=1)],row(clicks=4,purchases=2)))
        for key,num,expected in [('ctr',4,.4),('cvr',2,.2)]:
            self.assertEqual(shown[key]['own_value'],expected)
            self.assertEqual(shown[key]['numerator'],num)
            self.assertEqual(shown[key]['denominator'],10)
            self.assertEqual(shown[key]['denominator_field'],'exposure_records')
            self.assertNotIn('p25',shown[key])
            self.assertNotIn('own_percentile',shown[key])

    def test_zero_exposure_and_no_comparables_do_not_become_observed_zero_rates(self):
        shown=context_view(self.market([row(exposures=0)],row(exposures=0)))
        self.assertIsNone(shown['ctr']['own_value'])
        self.assertEqual(shown['ctr']['denominator'],0)
        missing=context_view(self.market([row(price=None)]))
        self.assertEqual(missing['click_records'], {'own_value': 0})
        self.assertNotIn('nonzero_products',missing['click_records'])
        self.assertIsNone(missing['price_raw']['median'])

    def test_recent_cumulative_and_reference_price_clocks_remain_distinct(self):
        raw=self.market([row(index=6),row(index=9),row(exposures=None,index=1)])
        shown=context_view(raw)
        self.assertEqual(shown['window'],{'start_inclusive':5,'end_exclusive':10})
        self.assertEqual(shown['current_reference']['observation_index'],8)
        self.assertEqual(shown['price_time_basis']['cutoff_exclusive'],10)
        self.assertEqual(shown['price_time_basis']['comparable_peer_observation_range'],
                         {'earliest_index':6,'latest_index':9})
        self.assertEqual(shown['cumulative']['window']['end_exclusive'],10)
        self.assertEqual(shown['cumulative']['window']['selection'],'all_available_history_per_product_before_cutoff')


class SimulationAndRepresentationTests(MarketFixture,unittest.TestCase):
    def test_simulated_active_price_and_cumulative_window_are_explicit(self):
        state=self.runtime.snapshot()
        state['clock']={'start':1000,'time':1030,'last_decisions':{'seller:10':{'time':1020}}}
        state['active_prices']['1'].update(source='simulation:submitted_price',logical_time=1010)
        shown=context_view(simulation_position(state,1))
        self.assertIn("whose exposure occurred within the window", shown["scope"])
        self.assertEqual(shown['window']['start_inclusive'],1020)
        self.assertEqual(shown['window']['end_exclusive'],1030)
        self.assertEqual(shown['price_time_basis'],{'clock':'simulation_logical_time',
            'selection':'active_price_at_decision','as_of':1030})
        self.assertEqual(shown['cumulative']['window']['start_inclusive'],1000)
        self.assertEqual(shown['current_reference']['active_price_basis']['logical_time'],1010)

    def test_compact_simulation_matches_full_market_and_reuses_same_shop_category(self):
        state = self.runtime.snapshot()
        state["clock"] = {"start": 1000, "time": 1030, "last_decisions": {"seller:10": {"time": 1020}}}
        state["products"]["98"] = dict(state["products"]["1"], item_id=98)
        state["products"]["99"] = dict(state["products"]["2"], item_id=99, seller_id=99)
        state["active_prices"]["99"] = {"price_raw": None}
        state["feedback"] = [
            dict(item_id=item, search_time=t, purchase_records=n, exposure_records=n, click_records=n)
            for item, t, n in [(2, 1019, 10), (2, 1020, 3), (2, 1030, 20), (99, 1020, 500)]
        ]
        cache = simulation_market_cache(state)
        full = simulation_position(state, 1, cache=cache)
        with patch("agent_simulation.agents.context.market.market_position",
                   side_effect=AssertionError("Pricing must not compute full distributions")):
            compact = simulation_position(state, 1, cache=cache, decision_only=True)
            repeated = simulation_position(state, 98, cache=cache, decision_only=True)
        self.assertEqual(compact, repeated)
        self.assertEqual(len(cache["summaries"]), 1)
        self.assertEqual(compact, {
            "active_products": full["peer_products"],
            "price_as_of": full["price_time_basis"],
            "median_active_price": full["price_raw"]["median"],
            "purchase_window": full["window"],
            "median_purchase_records": full["purchase_records"]["median"],
        })
        self.assertEqual(compact["median_purchase_records"], 3)
        compact["purchase_window"]["start_inclusive"] = 0
        self.assertEqual(repeated["purchase_window"]["start_inclusive"], 1020)

    def test_missing_observed_category_keeps_available_simulation_market(self):
        state = self.runtime.snapshot()
        context = {
            "profile": {"seller_id": 10},
            "history_cutoff_exclusive": self.data.cutoff,
            "owned_products": [{"item_id": 1}],
        }
        missing = {"item_id": 1, "status": "missing_category"}
        with patch("agent_simulation.agents.context.market.iter_decision_positions", return_value=[(1, missing)]):
            load_market_context(context, self.data, simulation=state)
        shown = context["owned_products"][0]["market_position"]
        self.assertEqual(shown["status"], "missing_category")
        self.assertEqual(shown["simulation"]["median_active_price"], state["active_prices"]["2"]["price_raw"])
        full = simulation_position(state, 1)
        self.assertEqual(context_view({**missing, "simulation": full})["simulation"], context_view(full))

    def test_shared_category_keeps_distinct_item_windows(self):
        state,_=self.runtime.inputs.seller(10)
        a,b=state['owned_products']
        a['market_position']['window']={'start_inclusive':1,'end_exclusive':10}
        b['market_position']['window']={'start_inclusive':7,'end_exclusive':10}
        visible = decode_input(encode_input(decision_context(state)))
        self.assertEqual(
            {item["item_id"]: item["observed_market"]["window"] for item in visible["external_reference"]},
            {item["item_id"]: item["market_position"]["window"] for item in (a, b)},
        )
        for item in (a,b):
            self.assertNotIn("cumulative",item["market_position"])
            self.assertNotIn("exposure_records",item["market_position"])
