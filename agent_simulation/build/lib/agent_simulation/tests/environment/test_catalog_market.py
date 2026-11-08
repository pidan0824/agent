"""Market queries and Seller quotes against catalog facts."""

from pathlib import Path
import json
import tempfile
import unittest

from agent_simulation.tests.data.fixtures import CatalogFixture


class CatalogMarketTests(CatalogFixture, unittest.TestCase):
    def test_recent_summary_and_competitors_share_read_full_discovery_is_separate(self):
        from unittest.mock import patch
        from agent_simulation.data.catalog import DataRepository
        from agent_simulation.data.market.quotes import rows, historical_reference
        from agent_simulation.data.market.products import public_candidates
        data = DataRepository(self.catalog.connection, self.cfg, 4)
        with patch('agent_simulation.data.market.quotes.rows', wraps=rows) as read, \
             patch('agent_simulation.data.market.quotes.historical_reference', wraps=historical_reference) as summary:
            first = data.market_positions(10, [1], 4, decision_only=True)
            first[1]['recent_price_scope']['caller_marker'] = True
            data.competitor_snapshot(10,1,4)
            self.assertEqual(len([c for c in read.call_args_list if 'arg_max(struct_pack' in c.args[1]]),1)
            # The summary's latest target profile must serve discovery without history SQL.
            with patch.object(data, 'records', side_effect=AssertionError('duplicate target history scan')):
                pool = data.public_pool(10, 1, 4)
            second = data.market_positions(10, [2], 4, decision_only=True)
            self.assertNotIn('caller_marker', second[2]['recent_price_scope'])
            self.assertEqual(summary.call_count, 0)
            data.public_pool(10, 2, 4)
            scans = [c for c in read.call_args_list if 'arg_max(struct_pack' in c.args[1]]
            self.assertEqual(len(scans), 2)
            data.public_pool(10, 1, 3)
            scans = [c for c in read.call_args_list if 'arg_max(struct_pack' in c.args[1]]
            self.assertEqual(len(scans), 3)
        _, expected = public_candidates(self.catalog.connection, self.cfg, 1, 4)
        self.assertEqual(pool['candidates'], expected)
        self.assertTrue(all(p['seller_id'] != 10 and p['reference_price_time_index'] < 4
                            for p in pool['candidates']))

    def test_market_queries_and_seller_quote(self):
        from contextlib import closing
        from agent_simulation.data.catalog import DataRepository
        from agent_simulation.environment.market import MarketEnvironment
        from agent_simulation.agents.seller import SellerAgent
        from agent_simulation.tests.fixtures import ScriptedModel, TokenizerFixture, answer

        data = DataRepository(self.catalog.connection, self.cfg, 4)
        with tempfile.TemporaryDirectory() as tmp, closing(MarketEnvironment.create(
                Path(tmp)/"state.db", data,[1,2,3],{},catalog=self.catalog)) as market:
            market.execute("seller",10,"discover_public_products", {"focal_item_id":1,"before_time_index":4})
            details = market.execute("seller",10,"query_public_products", {"item_ids":[3],"before_time_index":4,"limit":3})
            model = ScriptedModel([answer({"prices":[{"item_id":1,"price_raw":102},{"item_id":2,"price_raw":102}]})])
            agent = SellerAgent(market,model,TokenizerFixture(), price_batch_size=16)
            run=agent.run(10)
            self.assertEqual([a["name"] for a in run["tools"]], ["submit_prices"])
            visible=details["products"][0]
            self.assertEqual(visible["simulation_current_price"]["price_raw"],103)
            self.assertEqual(visible["profile"]["item_title"],"公开竞品")
            self.assertNotIn("exposure_records",str(visible))
            self.assertNotIn("click_records",str(visible))
            stat=market.execute("seller",10,"query_market_position",{"item_id":1,"before_time_index":4})
            for v in (stat,stat["simulation"]):
                for scope in (v,v["cumulative"]):
                    for metric in ("exposure_records","click_records"):
                        self.assertEqual(set(scope[metric]), {"own_value"})
                self.assertNotIn("median",v["cvr"])
                self.assertIn("median",v["purchase_records"])
            self.assertEqual(model.requests[0]["tools"], [])
            self.assertEqual(market.snapshot()["active_prices"]["1"]["price_raw"],102)
            evidence = [json.loads(payload) for (payload,) in market.db.execute(
                "SELECT payload FROM memory_evidence WHERE owner=?", ("seller:10",))]
            self.assertTrue(any(e["kind"]=="query_public_products" for e in evidence))
            for e in evidence:
                if e["kind"]=="market_position":
                    value=e["content"]["market_position"]
                    self.assertIn("recent",value)
                    self.assertNotIn("exposure_records",str(value))
                    self.assertNotIn("click_records",str(value))
                    continue
                elif e["kind"]=="query_market_position":
                    value=e["content"]["result"]
                else:
                    continue
                for window in (value, value["simulation"], value["cumulative"]):
                    self.assertEqual(set(window["click_records"]), {"own_value"})
                    self.assertEqual(set(window["exposure_records"]), {"own_value"})
            state,_=market.inputs.seller(10,retrieve=False,observe=False)
            product = next(p for p in state["owned_products"] if p["item_id"]==1)
            self.assertNotIn("own_price_band", product["market_position"])
            self.assertEqual(product["market_position"]["window"]["end_exclusive"],4)

