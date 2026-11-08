"""Real tool dispatch, temporal visibility, settlement and transaction tests."""

from agent_simulation.agents.prompts.seller import seller_decision_messages

import json
from copy import deepcopy
import unittest
import sqlite3
from unittest.mock import patch

from agent_simulation.data.common import literal
from agent_simulation.data.catalog import DataRepository
from agent_simulation.environment.market import MarketEnvironment
from agent_simulation.agents.context.pricing import build_pricing_context
from agent_simulation.tests.fixtures import MarketFixture


class RuntimeTests(MarketFixture, unittest.TestCase):
    def test_retired_runtime_schemas_are_rejected(self):
        state = self.runtime.snapshot()
        for version in ("kuaisearch-runtime-v4", "kuaisearch-runtime-v3"):
            state["schema"] = version
            with self.runtime.db:
                self.runtime.db.execute("UPDATE state SET value=?", (json.dumps(state),))
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "Unsupported runtime schema"):
                MarketEnvironment(self.path, self.data)

    def test_repeating_price_is_audited_without_mutating_market(self):
        before = self.runtime.snapshot()
        same = {"item_id": 1, "price_raw": before["active_prices"]["1"]["price_raw"]}
        result = self.call("seller", 10, "submit_prices", prices=[same])
        self.assertEqual(result["status"], "no_change")
        self.assertEqual(self.runtime.snapshot(), before)
        args, = self.runtime.db.execute("SELECT arguments FROM calls WHERE name='submit_prices'").fetchone()
        self.assertEqual(json.loads(args)["prices"], [same])
        changed = {"item_id": 3, "price_raw": 234}
        result = self.call("seller", 10, "submit_prices", prices=[same, changed])
        self.assertEqual(result["updated"], [changed])
        self.assertEqual(self.runtime.snapshot()["simulation_step"], before["simulation_step"] + 1)

    def call(self, role, actor, name, **args):
        return self.runtime.execute(role, actor, name, args)

    def search(self, user=7):
        return self.call("customer", user, "search_products", query="保温杯")

    def report(self):
        return self.call(
            "seller", 10, "query_business", item_ids=[1, 3], before_time_index=10, limit=30
        )

    def test_failed_initialization_closes_database_connections(self):
        connect = sqlite3.connect
        for stage in ("cutoff", "search_index"):
            opened = []

            class TrackingConnection(sqlite3.Connection):
                def execute(self, sql, *args, **kwargs):
                    if stage == "search_index" and sql.startswith("CREATE VIRTUAL TABLE"):
                        raise sqlite3.OperationalError("search initialization failed")
                    return super().execute(sql, *args, **kwargs)

            def track(*args, **kwargs):
                db = connect(*args, **kwargs, factory=TrackingConnection)
                opened.append(db)
                return db

            data = DataRepository(self.c, self.cfg, 9) if stage == "cutoff" else self.data
            with self.subTest(stage=stage), patch(
                "agent_simulation.environment.market.sqlite3.connect", side_effect=track
            ):
                with self.assertRaises((ValueError, sqlite3.OperationalError)):
                    MarketEnvironment(self.path, data)
                try:
                    self.assertEqual(len(opened), 1 if stage == "cutoff" else 2)
                    for db in opened:
                        with self.assertRaises(sqlite3.ProgrammingError):
                            db.execute("SELECT 1")
                finally:
                    for db in opened:
                        db.close()

    def test_initialization_rejects_invalid_history_and_explicit_empty_search_config(self):
        path = self.path.with_name("invalid.sqlite")
        record = self.runtime.snapshot()["customers"]["7"]["observed_history"][0]
        for histories, options, error in (
            ({7: [{**record, "user_id": 8}]}, {}, "Foreign customer history"),
            ({7: [{**record, "time_index": self.data.cutoff}]}, {}, "Future event in history store"),
            ({7: [record, record]}, {}, "Duplicate customer history"),
            ({7: []}, {"search_config": {}}, "Unknown search backend"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                MarketEnvironment.create(path, self.data, [1], histories, **options)
            self.assertFalse(path.exists())

    def test_seed_history_is_ordered_once_and_queries_preserve_cutoff_and_isolation(self):
        record = self.runtime.snapshot()["customers"]["7"]["observed_history"][0]
        seeds = [{**record, "time_index": t, "session_id": s}
                 for t, s in [(8, 4), (2, 3), (6, 2), (2, 1)]]
        runtime = MarketEnvironment.create(self.path.with_name("seed-order.sqlite"), self.data, [1], {7: seeds})
        self.addCleanup(runtime.close)
        before = runtime.snapshot()
        self.assertEqual([e["session_id"] for e in before["customers"]["7"]["observed_history"]],
                         [1, 3, 2, 4])
        result = runtime.execute("customer", 7, "query_customer_history",
                                 {"before_time_index": 6, "limit": 1})
        self.assertEqual([e["session_id"] for e in result["observed_history"]], [3])
        result["observed_history"][0]["query"] = "changed return value"
        self.assertEqual(runtime.snapshot(), before)
        self.assertEqual(runtime.execute("customer", 7, "query_customer_history",
                                        {"before_time_index": 0, "limit": 1})["observed_history"], [])

    def test_trajectory_summary_and_grouped_limit(self):
        report = self.report()
        self.assertEqual([r["time_index"] for r in report["series"]], [2, 3, 4])
        self.assertEqual(report["summary_of_returned_series"][0]["exposure_records"], 2)
        recent = self.call(
            "seller", 10, "query_business", item_ids=[1, 3], before_time_index=10, limit=1
        )
        self.assertEqual([r["price_raw"] for r in recent["series"]], [200, 120])
        self.assertTrue(recent["has_earlier_records"])
        self.assertEqual(recent["summary_of_returned_series"][0]["exposure_records"], 1)

    def test_tick_history_read_of_own_product_outside_simulated_inventory(self):
        from agent_simulation.simulation.episode import Episode
        from types import SimpleNamespace
        runtime = MarketEnvironment.create(self.path.with_name("subset.sqlite"), self.data, [1], {})
        self.addCleanup(runtime.close)
        agent = SimpleNamespace(runtime=runtime)
        Episode(runtime, agent, agent, tick_size=1, max_ticks=2)
        args = {"item_ids": [1, 3], "before_time_index": 10, "limit": 2}
        result = runtime.execute("seller", 10, "query_business", args)
        self.assertEqual({r["item_id"] for r in result["series"]}, {1, 3})
        self.assertEqual(result["simulation"]["outside_market_item_ids"], [3])
        for key in ("summary", "settled_coverage", "price_action_coverage", "activity_timeline"):
            self.assertEqual([row["item_id"] for row in result["simulation"][key]], [1])

    def test_market_excludes_items_without_strictly_prior_valid_price(self):
        from contextlib import closing
        history = self.data.records(2)
        self.data._histories[2] = [history[-1], {**history[0], "time_index": self.data.cutoff}]
        path = self.path.with_name("filtered.sqlite")
        with self.assertRaisesRegex(ValueError, "valid observed price before cutoff"):
            MarketEnvironment.create(path, self.data, [2], {})
        self.assertFalse(path.exists())
        with closing(MarketEnvironment.create(path, self.data, [1, 2], {})) as market:
            state = market.snapshot()
            self.assertEqual(state["excluded_items_without_initial_price"], [2])
            for key in ("products", "active_prices"):
                self.assertEqual(set(state[key]), {"1"})
            self.assertEqual(set(state["initial_seller_memory"]), {"10"})

    def test_unknown_reference_remains_unknown(self):
        result = self.call(
            "seller", 10, "query_public_products", item_ids=[2], before_time_index=10, limit=30
        )
        self.assertEqual(
            [r["price_raw"] for r in result["products"][0]["reference_price_history"]], [80, None]
        )
        self.assertIsNone(result["products"][0]["profile"]["reference_price_raw"])
        self.assertNotIn("purchase_records", json.dumps(result))
        shown = self.search()
        self.assertEqual(next(p["price_raw"] for p in shown["displayed_items"] if p["item_id"] == 2), 80)
        self.assertEqual(self.runtime.snapshot()["active_prices"]["2"]["time_index"], 3)

    def test_public_pool_cache_cannot_grant_or_deny_visibility_by_call_order(self):
        real_product = self.data.product
        def profile(item, before=None):
            p = real_product(item, before)
            if item == 3:
                p["category_level3_name"] = "UNKNOWN"
            return p
        with patch.object(self.data, "product", side_effect=profile):
            for order in ((1, 3), (3, 1)):
                self.data._decision_cache.clear()
                for item in order:
                    page = self.call("seller", 10, "discover_public_products",
                                     focal_item_id=item, before_time_index=10,
                                     after_item_id=0, page_size=5)
                    self.assertEqual([p["item_id"] for p in page["candidates"]],
                                     [2] if item == 1 else [])

    def test_known_public_id_needs_no_discovery_and_keeps_permissions(self):
        before = self.runtime.snapshot()
        result = self.call(
            "seller", 10, "query_public_products", item_ids=[2], before_time_index=4, limit=1
        )
        self.assertEqual(result["products"][0]["reference_price_history"],
                         [{"time_index": 3, "price_raw": 80, "price_status": "observed"}])
        self.assertEqual(self.runtime.snapshot(), before)
        self.assertNotIn("purchase_records", json.dumps(result))
        for forbidden in [1, 4]:
            with self.subTest(item_id=forbidden), self.assertRaises(PermissionError):
                self.call("seller", 10, "query_public_products", item_ids=[forbidden],
                          before_time_index=4, limit=1)
        self.assertEqual(self.runtime.snapshot(), before)

    def test_per_product_history_continuation_and_empty_window(self):
        recent = self.call("seller", 10, "query_business", item_ids=[1, 3],
                           before_time_index=10, limit=1)
        first, second = recent["item_coverage"]
        self.assertEqual((first["matched_points"], first["returned_points"]), (2, 1))
        self.assertEqual(first["next_before_time_index"], 4)
        self.assertIsNone(second["next_before_time_index"])
        older = self.call("seller", 10, "query_business", item_ids=[1],
                          before_time_index=first["next_before_time_index"], limit=1)
        self.assertEqual([r["time_index"] for r in older["series"]], [2])
        empty = self.call("seller", 10, "query_business", item_ids=[1],
                          before_time_index=2, limit=1)
        self.assertEqual(empty["item_coverage"][0]["matched_points"], 0)
        self.assertIsNone(empty["item_coverage"][0]["next_before_time_index"])
        public = self.call("seller", 10, "query_public_products", item_ids=[2],
                           before_time_index=10, limit=1)["products"][0]
        self.assertEqual(public["coverage"]["next_before_time_index"], 5)
        self.assertIsNone(public["reference_price_history"][0]["price_raw"])

    def test_price_receipt_and_independent_simulation_cursors(self):
        self.search()
        receipt = self.call("seller", 10, "submit_prices", prices=[{"item_id": 1, "price_raw": 90}])
        self.assertEqual(receipt["status"], "applied")
        self.assertEqual(receipt["effective_from_simulation_step"], 2)
        self.call("customer", 7, "submit_response", clicked_item_ids=[1], purchased_item_ids=[])
        self.call("seller", 10, "submit_prices", prices=[{"item_id": 1, "price_raw": 95}])
        self.search()
        self.call("customer", 7, "submit_response", clicked_item_ids=[], purchased_item_ids=[])
        result = self.call("seller", 10, "query_business", item_ids=[1], before_time_index=10, limit=1)
        sim = result["simulation"]
        self.assertEqual(sim["settled_coverage"][0]["next_before_simulation_step"], 6)
        self.assertEqual(sim["price_action_coverage"][0]["next_before_simulation_step"], 4)
        earlier = self.call("seller", 10, "query_business", item_ids=[1], before_time_index=10,
                            before_simulation_step=6, limit=1)
        self.assertEqual(earlier["simulation"]["settled_series"][0]["price_raw"], 120)
        unchanged = self.call("seller", 10, "submit_prices", prices=[])
        self.assertEqual(unchanged["status"], "no_change")
        self.assertIsNone(unchanged["effective_from_simulation_step"])

    def test_discovery_to_detail_finds_products_outside_initial_market_page(self):
        target = self.cfg.data / "seller_business/part_00.parquet"
        self.c.execute(f"CREATE TABLE expanded AS SELECT * FROM read_parquet({literal(target)})")
        self.c.execute(
            """INSERT INTO expanded SELECT * REPLACE(6 AS item_id,60 AS seller_id,
            6 AS time_index,'便携水杯' AS item_title,8 AS brand_id,'新品牌' AS brand_name)
            FROM expanded WHERE item_id=2 AND time_index=3"""
        )
        self.c.execute(
            """INSERT INTO expanded SELECT * REPLACE(7 AS item_id,70 AS seller_id,
            10 AS time_index) FROM expanded WHERE item_id=6"""
        )
        self.c.execute(f"COPY expanded TO {literal(target)} (FORMAT PARQUET)")
        before = self.runtime.snapshot()
        first = self.call(
            "seller",
            10,
            "discover_public_products",
            focal_item_id=1,
            before_time_index=10,
            page_size=1,
        )
        self.assertEqual(first["candidates"][0]["simulation_current_price"], {"price_raw": 80})
        self.assertEqual(first["total_candidates"], 2)
        self.assertEqual([p["item_id"] for p in first["candidates"]], [2])
        second = self.call(
            "seller",
            10,
            "discover_public_products",
            focal_item_id=1,
            before_time_index=10,
            page_size=1,
            after_item_id=first["next_after_item_id"],
        )
        self.assertEqual([p["item_id"] for p in second["candidates"]], [6])
        self.assertIsNone(second["candidates"][0]["simulation_current_price"])
        detail = self.call(
            "seller", 10, "query_public_products", item_ids=[6], before_time_index=10, limit=10
        )
        self.assertEqual(detail["products"][0]["reference_price_history"][0]["time_index"], 6)
        filtered = self.call(
            "seller",
            10,
            "discover_public_products",
            focal_item_id=1,
            before_time_index=10,
            query="便携",
            brand_id=8,
        )
        self.assertEqual([p["item_id"] for p in filtered["candidates"]], [6])
        self.assertEqual(before, self.runtime.snapshot())
        self.assertNotIn("purchase_records", json.dumps(filtered))

    def test_discovery_permissions_cutoff_and_empty_results(self):
        for role, actor, args, error in [
            ("customer", 7, {"focal_item_id": 1, "before_time_index": 10}, PermissionError),
            ("seller", 10, {"focal_item_id": 2, "before_time_index": 10}, PermissionError),
            ("seller", 10, {"focal_item_id": 1, "before_time_index": 11}, ValueError),
            (
                "seller",
                10,
                {"focal_item_id": 1, "before_time_index": 10, "page_size": 101},
                ValueError,
            ),
            (
                "seller",
                10,
                {"focal_item_id": 1, "before_time_index": 10, "query": "  "},
                ValueError,
            ),
        ]:
            with self.subTest(args=args), self.assertRaises(error):
                self.call(role, actor, "discover_public_products", **args)
        page = self.call(
            "seller",
            10,
            "discover_public_products",
            focal_item_id=1,
            before_time_index=10,
            query="完全不相关",
        )
        self.assertEqual(page["candidates"], [])
        self.assertIsNone(page["next_after_item_id"])

    def test_two_customers_settle_into_one_shop_report(self):
        self.search(7)
        self.search(8)
        self.assertEqual(self.report()["simulation"]["settled_series"], [])
        self.call("customer", 7, "submit_response", clicked_item_ids=[1, 3], purchased_item_ids=[1])
        self.call("customer", 8, "submit_response", clicked_item_ids=[1], purchased_item_ids=[])
        summary = self.report()["simulation"]["summary"]
        self.assertEqual(
            summary[0],
            {"item_id": 1, "exposure_records": 2, "click_records": 2, "purchase_records": 1,
             "estimated_units_sold": 1},
        )
        memory = self.call("customer", 7, "query_customer_history", before_time_index=10, limit=30)
        self.assertEqual(len(memory["observed_history"]), 1)
        self.assertEqual(memory["simulation_history"][0]["query"], "保温杯")
        self.assertNotIn("displayed_items", memory["simulation_history"][0])

    def test_isolated_report_reads_do_not_reset_history(self):
        self.search(7)
        self.call("customer", 7, "submit_response", clicked_item_ids=[1], purchased_item_ids=[1])
        first = self.call("seller", 10, "read_business_report")
        first_step = first["through_step_inclusive"]
        self.assertEqual(first["products"][0]["estimated_units_sold"], 1)
        self.search(8)
        self.call("customer", 8, "submit_response", clicked_item_ids=[], purchased_item_ids=[1])
        second = self.call("seller", 10, "read_business_report")
        item = second["products"][0]
        self.assertEqual(second["after_step_exclusive"], 0)
        self.assertEqual((item["exposure_records"], item["click_records"], item["estimated_units_sold"]), (2, 1, 2))
        self.assertEqual(item["cumulative"], {"exposure_records": 2, "click_records": 1,
                                             "purchase_records": 2, "estimated_units_sold": 2})
        empty = self.call("seller", 10, "read_business_report")
        self.assertEqual(empty, second)
        self.assertEqual(empty["shop_cumulative_totals"]["estimated_units_sold"], 2)
        recent = self.call("seller", 10, "query_business", item_ids=[1], before_time_index=10, limit=1)
        self.assertEqual(recent["simulation"]["summary"][0]["purchase_records"], 1)
        self.assertEqual(recent["simulation"]["cumulative_series"][0]["estimated_units_sold"], 2)
        past = self.call("seller", 10, "query_business", item_ids=[1], before_time_index=10,
                         before_simulation_step=first_step + 1, limit=1)
        self.assertEqual(past["simulation"]["cumulative_series"][0]["estimated_units_sold"], 1)

    def test_price_intervals_use_event_time_not_price_attribution(self):
        self.search(7)
        before = self.runtime.snapshot()
        pending = self.call("seller", 10, "read_business_report")
        initial = pending["products"][0]["price_intervals"][0]
        self.assertEqual((initial["exposure_records"], initial["purchase_records"]), (1, 0))
        change = self.call("seller", 10, "submit_prices", prices=[{"item_id": 1, "price_raw": 90}])
        self.call("customer", 7, "submit_response", clicked_item_ids=[1], purchased_item_ids=[1])
        report = self.call("seller", 10, "read_business_report")
        old, new = report["products"][0]["price_intervals"]
        self.assertEqual(old["end_step_exclusive"], change["simulation_step"])
        self.assertEqual((old["exposure_records"], old["click_records"], old["estimated_units_sold"]), (1, 0, 0))
        self.assertEqual((new["exposure_records"], new["click_records"], new["estimated_units_sold"]), (0, 1, 1))
        self.assertEqual(self.runtime.snapshot()["feedback"][0]["price_raw"], 120)
        self.call("seller", 10, "submit_prices", prices=[{"item_id": 1, "price_raw": 90}])
        unchanged = self.call("seller", 10, "read_business_report")
        self.assertEqual(len(unchanged["products"][0]["price_intervals"]), 2)
        self.call("seller", 10, "submit_prices", prices=[{"item_id": 1, "price_raw": 120}])
        changed_back = self.call("seller", 10, "read_business_report")
        self.assertEqual([w["price_raw"] for w in changed_back["products"][0]["price_intervals"]], [120, 90, 120])
        self.assertEqual(len(changed_back["products"][1]["price_intervals"]), 1)
        from agent_simulation.environment.operations.reporting import price_intervals
        past = price_intervals(self.runtime.snapshot(), 10, before=before["simulation_step"] + 1)
        self.assertEqual(len(past[1]), 1)
        self.assertEqual((past[1][0]["exposure_records"], past[1][0]["purchase_records"]), (1, 0))

    def test_price_changes_apply_to_next_search_not_pending_list(self):
        shown = self.search()
        old = next(p["price_raw"] for p in shown["displayed_items"] if p["item_id"] == 1)
        self.call("seller", 10, "submit_prices", prices=[{"item_id": 1, "price_raw": 90}])
        self.call("customer", 7, "submit_response", clicked_item_ids=[1], purchased_item_ids=[1])
        self.assertEqual(self.report()["simulation"]["settled_series"][0]["price_raw"], old)
        current = self.search()
        self.assertEqual(
            next(p["price_raw"] for p in current["displayed_items"] if p["item_id"] == 1), 90
        )

    def test_new_price_replaces_unknown_display_status_without_rewriting_history(self):
        self.call("seller", 20, "submit_prices", prices=[{"item_id": 2, "price_raw": 95}])
        shown = next(p for p in self.search()["displayed_items"] if p["item_id"] == 2)
        self.assertEqual(shown["price_raw"], 95)
        self.assertEqual(shown["price_status"], "observed")
        self.assertIsNone(self.data.product(2)["reference_price_raw"])
        self.assertEqual(self.data.product(2)["price_status"], "invalid_price")

    def test_invalid_batch_rolls_back_and_is_audited(self):
        before = self.runtime.snapshot()
        with self.assertRaises(PermissionError):
            self.call(
                "seller",
                10,
                "submit_prices",
                prices=[{"item_id": 1, "price_raw": 90}, {"item_id": 2, "price_raw": 80}],
            )
        self.assertEqual(before, self.runtime.snapshot())
        self.assertIn(
            "PermissionError", self.runtime.db.execute("SELECT error FROM calls").fetchone()[0]
        )

    def test_cutoff_and_private_permissions(self):
        cases = [
            (
                "query_business",
                {"item_ids": [2], "before_time_index": 10, "limit": 30},
                PermissionError,
            ),
            ("query_business", {"item_ids": [1], "before_time_index": 11, "limit": 30}, ValueError),
            (
                "query_public_products",
                {"item_ids": [4], "before_time_index": 10, "limit": 30},
                PermissionError,
            ),
        ]
        for name, args, error in cases:
            with self.subTest(name=name, args=args), self.assertRaises(error):
                self.runtime.execute("seller", 10, name, args)

    def test_invalid_response_does_not_settle_or_consume_request(self):
        self.search()
        before = self.runtime.snapshot()
        with self.assertRaises(ValueError):
            self.call(
                "customer", 7, "submit_response", clicked_item_ids=[999], purchased_item_ids=[]
            )
        self.assertEqual(before, self.runtime.snapshot())
        self.call("customer", 7, "submit_response", clicked_item_ids=[], purchased_item_ids=[])
        with self.assertRaisesRegex(ValueError, "No pending"):
            self.call("customer", 7, "submit_response", clicked_item_ids=[], purchased_item_ids=[])

    def test_no_match_and_empty_response_are_valid(self):
        result = self.call("customer", 7, "search_products", query="xxxyyyzzz")
        self.assertEqual(result["displayed_items"], [])
        self.call("customer", 7, "submit_response", clicked_item_ids=[], purchased_item_ids=[])
        self.assertEqual(self.runtime.snapshot()["feedback"], [])

    def test_resume_keeps_prices_pending_list_and_history(self):
        self.call("seller", 10, "submit_prices", prices=[{"item_id": 1, "price_raw": 90}])
        self.search()
        before = self.runtime.snapshot()
        self.runtime.close()
        self.runtime = MarketEnvironment(self.path, self.data)
        self.assertEqual(before, self.runtime.snapshot())
        self.call("customer", 7, "submit_response", clicked_item_ids=[1], purchased_item_ids=[1])
        self.assertEqual(self.report()["simulation"]["summary"][0]["purchase_records"], 1)

    def test_contract_rejects_extra_fields_wrong_roles_and_bad_prices(self):
        with self.assertRaises(ValueError):
            self.call("customer", 7, "search_products", query="杯", user_id=8)
        with self.assertRaises(PermissionError):
            self.call("customer", 7, "submit_prices", prices=[])
        for price in (True, 0, -1, "90"):
            with self.subTest(price=price), self.assertRaises(ValueError):
                self.call(
                    "seller", 10, "submit_prices", prices=[{"item_id": 1, "price_raw": price}]
                )
        with self.assertRaises(ValueError):
            self.call(
                "seller",
                10,
                "submit_prices",
                prices=[{"item_id": 1, "price_raw": 90}, {"item_id": 1, "price_raw": 95}],
            )

    def test_no_implicit_price_change_or_search_overwrite(self):
        before = self.runtime.snapshot()
        self.call("seller", 10, "submit_prices", prices=[])
        self.assertEqual(before, self.runtime.snapshot())
        self.search()
        with self.assertRaises(ValueError):
            self.search()

    def test_sft_and_runtime_initial_memory_use_identical_context_builder(self):
        from agent_simulation.agents.prompts.seller import decode_input
        from agent_simulation.agents.context.pricing import overview_context

        events = self.data.records(1) + self.data.records(3)
        target = {**events[0], "time_index": 10, "price_raw": 999}
        source_state, _ = build_pricing_context(
            events + [target],
            owner_id=10,
            cutoff=10,
            observations_per_item=2,
            public_pages=[],
            requested_item_ids=[1],
        )
        state, _ = self.runtime.inputs.seller(10, public_pages=[], detail_item_ids=[1])
        state.pop("simulation_state")
        expected = overview_context(source_state, [1])
        from agent_simulation.agents.context.market import load_market_context
        load_market_context(expected, self.data, simulation=self.runtime.snapshot())
        from agent_simulation.agents.retrieval.history import seller_history
        expected["retrieved_context"] = {"history": seller_history(
            events, seller_id=10, cutoff=10, recent_n=2, focus_ids=[1], simulation_before=1)}
        from agent_simulation.agents.context.pricing import attach_retrieved_products
        attach_retrieved_products(expected, lambda item: self.runtime.snapshot()["products"][str(item)])
        expected.pop("requested_price_item_ids")
        expected["price_scope"] = state["price_scope"]
        expected["shop_scope"] = state["shop_scope"]
        expected["simulation_cutoff_exclusive"] = state["simulation_cutoff_exclusive"]
        self.assertEqual(state, expected)
        sft = overview_context(source_state, [1])
        load_market_context(sft, self.data)
        historical_products = deepcopy(state["owned_products"])
        for product in historical_products:
            product["market_position"].pop("simulation")
            for card in product["market_position"]["competitor_examples"]["products"]:
                card.pop("simulation_price", None)
        self.assertEqual(sft["owned_products"], historical_products)
        self.assertEqual(sft["short_term_memory"], state["short_term_memory"])
        self.assertNotIn("requested_price_item_ids", state)
        actual = seller_decision_messages(state)
        decoded = decode_input(actual[1]["content"])
        from agent_simulation.tests.seller_context_fixture import decision_context
        self.assertEqual(decoded, decision_context(state))

    def test_settled_feedback_replaces_only_its_own_item_history(self):
        initial, _ = self.runtime.inputs.seller(10, public_pages=[], detail_item_ids=[1, 3])
        self.assertEqual(sum(len(e["content"]) for e in initial["short_term_memory"]), 3)
        self.search()
        pending, _ = self.runtime.inputs.seller(10, public_pages=[], detail_item_ids=[1, 3])
        self.assertEqual(initial["short_term_memory"], pending["short_term_memory"])
        self.call("customer", 7, "submit_response", clicked_item_ids=[1], purchased_item_ids=[])
        after, coverage = self.runtime.inputs.seller(10, public_pages=[], detail_item_ids=[1, 3])
        self.assertEqual([c["retained_points"] for c in coverage], [2, 2])
        self.assertEqual([c["simulated_points"] for c in coverage], [1, 1])
        self.assertEqual(
            after["short_term_memory"][-1]["generator"], "environment:simulated_business_event"
        )
        self.assertNotIn("time_index", after["short_term_memory"][-1])
        self.search(8)
        self.call("customer", 8, "submit_response", clicked_item_ids=[], purchased_item_ids=[])
        latest, coverage = self.runtime.inputs.seller(10, public_pages=[], detail_item_ids=[1, 3])
        self.assertEqual([c["observed_points"] for c in coverage], [0, 0])
        self.assertEqual([c["retained_points"] for c in coverage], [2, 2])
        self.assertEqual(len(latest["short_term_memory"]), 2)

    def test_submitted_price_visible_without_inventing_feedback(self):
        before, _ = self.runtime.inputs.seller(10, public_pages=[], detail_item_ids=[1, 3])
        self.call("seller", 10, "submit_prices", prices=[{"item_id": 1, "price_raw": 90}])
        after, _ = self.runtime.inputs.seller(10, public_pages=[], detail_item_ids=[1, 3])
        self.assertEqual(before["short_term_memory"], after["short_term_memory"])
        self.assertEqual(after["simulation_state"]["active_prices"]["1"]["price_raw"], 90)
        self.assertEqual(
            after["simulation_state"]["active_prices"]["1"]["source"], "simulation:submitted_price"
        )

    def test_rolled_out_feedback_remains_queryable_without_duplicating_memory(self):
        for _ in range(3):
            self.search()
            self.call("customer", 7, "submit_response", clicked_item_ids=[1], purchased_item_ids=[])
        context, coverage = self.runtime.inputs.seller(10, public_pages=[], detail_item_ids=[1, 3])
        self.assertEqual([c["retained_points"] for c in coverage], [2, 2])
        self.assertEqual([m["simulation_step"] for m in context["short_term_memory"]], [4, 6])
        recent = self.call(
            "seller", 10, "query_business", item_ids=[1], before_time_index=10, limit=2
        )
        self.assertTrue(recent["simulation"]["has_earlier_records"])
        earlier = self.call(
            "seller",
            10,
            "query_business",
            item_ids=[1],
            before_time_index=10,
            before_simulation_step=4,
            limit=2,
        )
        self.assertEqual([f["settled_step"] for f in earlier["simulation"]["settled_series"]], [2])
        after, _ = self.runtime.inputs.seller(10, public_pages=[], detail_item_ids=[1, 3])
        self.assertEqual(context, after)
        with self.assertRaisesRegex(ValueError, "simulation cutoff"):
            self.call(
                "seller",
                10,
                "query_business",
                item_ids=[1],
                before_time_index=10,
                before_simulation_step=8,
                limit=2,
            )


if __name__ == "__main__":
    unittest.main()
