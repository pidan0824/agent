"""Check metric denominators, DTW behavior and held-out inference isolation."""

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import duckdb

from agent_simulation.evaluation.macro.trajectories import (
    compare, customer_distribution, dtw, price_dynamics,
)
from agent_simulation.evaluation.micro.behavior_prediction import (
    choices, predict, prices, run, summarize,
)
from agent_simulation.inference.client import ModelOutputError


class BehaviorTests(unittest.TestCase):
    def test_programming_failure_propagates_and_user_message_is_not_a_prediction(self):
        tokenizer = SimpleNamespace(chat_template="fixture", apply_chat_template=lambda *a, **kw: [1])
        model = Mock(tokenizer=tokenizer, max_new_tokens=8)
        sample = {"sample_id": "query:1", "input": {}, "target": {"query": "杯子"}, "tools": [],
                  "messages": [{"role": "system", "content": "fixture"}, {"role": "user", "content": "fixture"},
                               {"role": "assistant", "content": '{"query":"杯子"}'}]}
        for error in (TypeError("broken adapter"), ValueError("invalid configuration"), PermissionError("denied")):
            model.complete.side_effect = error
            with self.assertRaises(type(error)) as caught:
                predict(sample, model)
            self.assertIs(caught.exception, error)
        model.complete.side_effect = None
        model.complete.return_value = {"role": "user", "content": '{"query":"杯子"}'}
        result = predict(sample, model)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("assistant", result["error"])
        self.assertNotIn("prediction", result)

    def test_invalid_response_keeps_raw_output_and_next_record_is_evaluated(self):
        tokenizer = SimpleNamespace(
            chat_template="test", apply_chat_template=lambda *a, **k: [1, 2],
            get_vocab=lambda: {"test": 1},
        )

        def sample(stage, identity):
            target = {"query": "杯子"} if stage == "query" else {
                "clicked_item_ids": [], "purchased_item_ids": []
            }
            return {
                "sample_id": f"{stage}:{identity}", "split": "test", "tools": [],
                "input": {"displayed_items": [{"item_id": 1}]}, "target": target,
                "messages": [{"role": "system", "content": "test"},
                             {"role": "user", "content": "test"},
                             {"role": "assistant", "content": json.dumps(target)}],
            }

        model = Mock(tokenizer=tokenizer, max_new_tokens=100)
        model.complete.side_effect = [
            {"role": "assistant", "content": '{"query":"杯子"}'},
            {"role": "assistant", "content": '{"clicked_item_ids":[999],"purchased_item_ids":[]}'},
            {"role": "assistant", "content": '{"query":"杯子"}'},
            {"role": "assistant", "content": '{"clicked_item_ids":[],"purchased_item_ids":[]}'},
        ]
        backend = model
        data = Mock(tokenizer=tokenizer)
        data.catalog.connection.execute.return_value.fetchall.return_value = [(1,), (2,)]
        data.request_samples.side_effect = lambda identity: [sample(stage, identity) for stage in ("query", "response")]
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "manifest.json").write_text("{}")
            args = SimpleNamespace(
                dataset=root, model=root, tokenizer=root, output=root / "evaluation",
                role="customer", count=2, seed=1, adapter=None, device="cpu", max_new_tokens=100,
            )
            with (
                patch("agent_simulation.evaluation.micro.behavior_prediction.SFTDataset", return_value=data),
                patch("agent_simulation.evaluation.micro.behavior_prediction.QwenLocal", return_value=backend),
                patch("agent_simulation.evaluation.micro.behavior_prediction.verify_checkpoint", side_effect=AssertionError("Default run must not hash model weights")),
            ):
                run(args)
            results = json.loads((args.output / "results.json").read_text())
            failed = json.loads((args.output / "response_1.json").read_text())
            self.assertEqual(failed["status"], "invalid")
            self.assertIn("999", failed["calls"][0]["response"]["content"])
            self.assertNotIn("prediction", failed)
            self.assertEqual(results["metrics"]["response"]["invalid_rate"], 0.5)
            self.assertTrue((args.output / "completion.json").exists())
        self.assertEqual(model.complete.call_count, 4)
        data.close.assert_called_once()

    def test_evaluation_rejects_adapter_trained_on_different_base_weights(self):
        tokenizer = SimpleNamespace(chat_template="fixture", get_vocab=lambda: {"test": 1})
        backend = Mock(tokenizer=tokenizer)
        data = Mock(tokenizer=tokenizer)
        checkpoint = {"files": {"config.json": {"sha256": "cfg"},
                                "base.safetensors": {"sha256": "actual"}}}
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "manifest.json").write_text("{}")
            (root / "run_config.json").write_text(json.dumps({"identity": {
                "role": "seller", "base_config_sha256": "cfg",
                "base_weight_sha256": {"base.safetensors": "different"}}}))
            args = SimpleNamespace(dataset=root, model=root, tokenizer=root, output=root / "evaluation",
                                   role="seller", count=1, seed=1, adapter=root / "adapter",
                                   device="cpu", max_new_tokens=100, verify=True)
            with (
                patch("agent_simulation.evaluation.micro.behavior_prediction.SFTDataset", return_value=data),
                patch("agent_simulation.evaluation.micro.behavior_prediction.select_record_ids", return_value=[]),
                patch("agent_simulation.evaluation.micro.behavior_prediction.QwenLocal", return_value=backend),
                patch("agent_simulation.evaluation.micro.behavior_prediction.verify_checkpoint", return_value=checkpoint),
                self.assertRaisesRegex(ValueError, "training base differs"),
            ):
                run(args)
            backend.complete.assert_not_called()

    def test_unfinished_model_output_is_saved_without_inventing_a_prediction(self):
        tokenizer = SimpleNamespace(chat_template="test", apply_chat_template=lambda *a, **k: [1, 2])
        model = Mock(tokenizer=tokenizer, max_new_tokens=10)
        model.complete.side_effect = ModelOutputError("unfinished", raw_output='{"query":', finished=False)
        sample = {
            "sample_id": "query:1", "input": {}, "target": {"query": "杯子"}, "tools": [],
            "messages": [{"role": "system", "content": "test"},
                         {"role": "user", "content": "test"},
                         {"role": "assistant", "content": "test"}],
        }
        result = predict(sample, model)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["calls"][0]["raw_output"], '{"query":')
        self.assertFalse(result["calls"][0]["finished"])
        self.assertNotIn("prediction", result)

    def test_trajectories_use_the_completed_index_window_and_sparse_price_labels(self):
        with TemporaryDirectory() as directory, duckdb.connect() as connection:
            root = Path(directory)
            catalog = Mock(directory=root / "catalog", connection=connection)
            from agent_simulation.data.common import sha256
            catalog.directory.mkdir()
            manifest = catalog.directory / "manifest.json"
            manifest.write_text("{}")
            initial = {
                "catalog_manifest_sha256": sha256(manifest),
                "historical_cutoff_exclusive": 100,
                "catalog": str(catalog.directory),
                "products": {"1": {"seller_id": 7}, "2": {"seller_id": 7}},
                "active_prices": {"1": {"price_raw": 100, "time_index": 95}, "2": {"price_raw": None}},
            }
            final = {
                "clock": {"start": 100, "tick": 4, "tick_size": 10},
                "price_actions": [
                    {"item_id": 1, "logical_time": 100, "price_raw": 105},
                    {"item_id": 1, "logical_time": 100, "price_raw": 110},
                    {"item_id": 1, "logical_time": 110, "price_raw": 80},
                    {"item_id": 1, "logical_time": 130, "price_raw": 85},
                    {"item_id": 1, "logical_time": 140, "price_raw": 999},
                ],
                "customers": {
                    "3": {
                        "exit": {"tick": 4, "logical_time": 140},
                        "simulation_history": [
                            {
                                "query": "杯子",
                                "clicked_items": [],
                                "purchased_items": [],
                                "search_time": 100,
                            },
                            {
                                "query": "第二次搜索",
                                "clicked_items": [],
                                "purchased_items": [{"item_id": 1}],
                                "search_time": 120,
                            },
                            {
                                "query": "坏的未完成轮",
                                "clicked_items": [],
                                "purchased_items": [],
                                "search_time": 140,
                            },
                        ]
                    },
                    "4": {"exit": {"tick": 1, "logical_time": 110}, "simulation_history": []},
                },
            }
            for name, value in [
                ("initial_state", initial),
                ("final_state", final),
                ("report", {"status": "failed"}),
            ]:
                (root / f"{name}.json").write_text(json.dumps(value))
            connection.execute(
                """CREATE TABLE seller_history AS SELECT 1 AS item_id, time_index, price_raw, price_status
                FROM (VALUES (99, 200, 'observed'), (105, 110, 'observed'), (110, 75, 'missing'),
                             (118, 100, 'observed'), (122, 900, 'observed'), (138, 90, 'observed'),
                             (140, 200, 'observed')) AS quotes(time_index, price_raw, price_status)"""
            )
            connection.execute(
                """CREATE TABLE requests AS
                SELECT time_index AS session_id, user_id, time_index, '杯子' AS query,
                       []::BIGINT[] AS clicked_item_ids, []::BIGINT[] AS purchased_item_ids
                FROM (VALUES (3, 99), (3, 101), (3, 112), (3, 139), (3, 140), (4, 135))
                AS searches(user_id, time_index)"""
            )
            with patch("agent_simulation.evaluation.macro.trajectories.Catalog", return_value=catalog):
                result = compare(root, catalog.directory, root / "comparison.json")
            self.assertEqual(result["reference_window"], {"start_inclusive": 100, "end_exclusive": 140})
            self.assertEqual(result["episode_status"], "failed")
            self.assertEqual(result["seller"][1]["status"], "missing_initial_price")
            price = result["seller"][0]
            self.assertEqual(price["initial_price"]["time_index"], 95)
            self.assertEqual([p["price_raw"] for p in price["simulated"]], [110, 80, 80, 85])
            self.assertEqual(
                [(p["time_index"], p["logical_time"]) for p in price["observed"]],
                [(105, 100), (118, 120), (138, 130)],
            )
            self.assertEqual(price["reference_observations"], 4)
            self.assertEqual(price["matched_future_points"], 3)
            self.assertAlmostEqual(price["price_mae_raw"], 25 / 3)
            self.assertAlmostEqual(price["mean_relative_absolute_error"], (20 / 100 + 5 / 90) / 3)
            self.assertEqual(price["direction_agreement"], 0.5)
            full = price["simulated_full_dynamics"]
            self.assertEqual(full["same_price_runs_observations"], [1, 2, 1])
            self.assertEqual(full["relative_changes"][0], 0.1)
            customer, exited = result["customer"]
            self.assertEqual(customer["simulated_distribution"]["requests"], 2)
            self.assertEqual(customer["observed_distribution"]["requests"], 3)
            self.assertEqual(customer["observed_search_intervals"], [11, 27])
            self.assertEqual(customer["simulated_search_intervals"], [20])
            self.assertEqual(customer["simulated_opportunities"], 4)
            self.assertEqual(customer["simulated_no_action_count"], 2)
            self.assertEqual(customer["simulated_search_rate"], 0.5)
            self.assertIsNone(customer["exit"])
            self.assertEqual(exited["simulated_distribution"]["requests"], 0)
            self.assertEqual(exited["observed_distribution"]["requests"], 1)
            self.assertEqual(exited["simulated_opportunities"], 2)
            self.assertEqual(exited["simulated_no_action_count"], 1)
            self.assertEqual(exited["exit"]["tick"], 1)
            manifest.write_text('{"version":2}')
            with patch("agent_simulation.evaluation.macro.trajectories.Catalog", return_value=catalog):
                with self.assertRaisesRegex(ValueError, "Different source catalog release"):
                    compare(root, catalog.directory, root / "invalid.json")

    def test_search_without_response_is_still_a_request(self):
        r = customer_distribution(
            [
                {"query": "杯子", "clicked_item_ids": [], "purchased_item_ids": []},
                {"query": "杯子", "clicked_item_ids": [1, 2], "purchased_item_ids": [2]},
            ]
        )
        self.assertEqual(r["requests"], 2)
        self.assertEqual(r["click_request_rate"], 0.5)
        self.assertEqual(r["no_response_rate"], 0.5)
        self.assertEqual(r["repeated_query_share"], 1)
        self.assertIsNone(customer_distribution([])["purchase_request_rate"])

    def test_set_metrics_and_empty_requests(self):
        r = choices([([1, 2], [2, 3]), ([], [])])
        self.assertEqual(r["micro_f1"], 0.5)
        self.assertEqual(r["mean_request_f1"], 0.75)
        self.assertEqual(r["exact_set_rate"], 0.5)
        self.assertIsNone(choices([([], [])])["recall"])

    def test_invalid_is_not_an_empty_prediction(self):
        records = [
            dict(sample_id="response:1", stage="response", status="invalid"),
            dict(
                stage="response",
                status="ok",
                prediction={"clicked_item_ids": [], "purchased_item_ids": []},
                target={"clicked_item_ids": [], "purchased_item_ids": []},
            ),
        ]
        r = summarize(records)["response"]
        self.assertEqual(r["invalid_rate"], 0.5)
        self.assertEqual(r["clicked_item_ids"]["exact_set_rate"], 1)
        self.assertEqual(r["clicked_item_ids"]["end_to_end_exact_set_rate"], 0.5)

    def test_price_error_and_warping(self):
        self.assertEqual(prices([(12, 10), (8, 10)])["mae"], 2)
        self.assertEqual(prices([(12, 10), (8, 10)])["rmse"], 2)
        self.assertEqual(dtw([1, 2], [1, 1, 2]), 0)
        self.assertEqual(dtw([1], [3]), 2)
        with self.assertRaises(ValueError):
            dtw([], [1])
        with self.assertRaises(ValueError):
            dtw([float("nan")], [1])

    def test_seller_change_metrics_compare_to_last_quote_and_separate_directions(self):
        def record(item, actual, prediction):
            return {
                "stage": "seller",
                "sample_id": f"seller:10:{item}:11",
                "status": "ok",
                "previous_price_raw": 100,
                "target": {"prices": [{"item_id": item, "price_raw": actual}]},
                "prediction": {"prices": [{"item_id": item, "price_raw": prediction}]},
            }

        summary = summarize([
            record(1, 120, 110),
            record(2, 80, 110),
            record(3, 100, 100),
            {"stage": "seller", "sample_id": "seller:10:4:11", "status": "invalid"},
        ])["seller"]
        changes = summary["price_changes"]
        self.assertEqual(summary["invalid"], 1)
        self.assertEqual(changes["observations"], 3)
        self.assertEqual(changes["direction_accuracy"], 2 / 3)
        self.assertAlmostEqual(changes["relative_change_mae"], 0.4 / 3)
        self.assertEqual(changes["keep_previous_price_baseline"]["direction_accuracy"], 1 / 3)
        self.assertAlmostEqual(changes["keep_previous_price_baseline"]["relative_change_mae"], 0.4 / 3)
        self.assertEqual(changes["by_actual_direction"]["increase"]["direction_accuracy"], 1)
        self.assertEqual(changes["by_actual_direction"]["decrease"]["direction_accuracy"], 0)
        self.assertEqual(changes["by_actual_direction"]["keep"]["relative_change_mae"], 0)

    def test_price_runs_are_observations(self):
        r = price_dynamics([880, 780, 780, 850])
        self.assertEqual(r["same_price_runs_observations"], [1, 2, 1])
        self.assertEqual(r["direction_shares"], dict(keep=1 / 3, increase=1 / 3, decrease=1 / 3))


    def test_target_is_not_sent_to_model(self):
        tokenizer = SimpleNamespace(
            chat_template="test", apply_chat_template=lambda *a, **k: [1, 2]
        )
        model = Mock(tokenizer=tokenizer, max_new_tokens=10)
        model.complete.return_value = {"role": "assistant", "content": '{"query":"杯子"}'}
        sample = {
            "sample_id": "query:1",
            "input": {"profile": {"user_id": 7}, "history_cutoff_exclusive": 100},
            "target": {"query": "SECRET_TARGET"},
            "tools": [],
            "messages": [
                {"role": "system", "content": "Predict query"},
                {"role": "user", "content": "History"},
                {"role": "assistant", "content": "SECRET_TARGET"},
            ],
        }
        result = predict(sample, model)
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("SECRET_TARGET", str(model.complete.call_args))
        self.assertEqual(model.complete.call_args.args[0]["messages"], sample["messages"][:-1])
        model.complete.side_effect = RuntimeError("fixture generation error")
        with self.assertRaisesRegex(RuntimeError, "fixture generation error"):
            predict(sample, model)

    def test_seller_eval_sends_and_enforces_the_supervised_item_schema(self):
        from agent_simulation.tests.fixtures import ScriptedModel, TokenizerFixture, answer
        sample = {"sample_id": "seller:10:7:11", "target": {"prices": [{"item_id": 7, "price_raw": 100}]},
                  "input": {"requested_price_item_ids": [7], "owned_products": [{
                      "item_id": 7, "last_observed_price_raw": None,
                      "last_valid_reference": {"price_raw": 90, "time_index": 9},
                  }]}, "tools": [],
                  "messages": [{"role": "system", "content": "Price the requested product."},
                               {"role": "user", "content": "{}"}, {"role": "assistant", "content": None}]}
        for prices, status in (([], "invalid"), ([{"item_id": 8, "price_raw": 100}], "invalid"),
                               ([{"item_id": 7, "price_raw": None}], "invalid"),
                               ([{"item_id": 7, "price_raw": 123}], "ok")):
            model = ScriptedModel([answer({"prices": prices})])
            model.tokenizer, model.max_new_tokens = TokenizerFixture(), 100
            result = predict(sample, model)
            self.assertEqual(result["status"], status)
            self.assertEqual(result["previous_price_raw"], 90)
            self.assertEqual(model.requests[0]["tools"], sample["tools"])

    def test_device_failure_is_not_counted_as_an_invalid_prediction(self):
        import torch

        tokenizer = SimpleNamespace(chat_template="test", apply_chat_template=lambda *a, **k: [1, 2])
        for stage in ("query", "response", "seller"):
            for error_type in (torch.OutOfMemoryError, RuntimeError):
                with self.subTest(stage=stage, error=error_type.__name__):
                    model = Mock(tokenizer=tokenizer, max_new_tokens=10)
                    model.complete.side_effect = error_type("fixture device failure")
                    sample = {
                        "sample_id": f"{stage}:1", "target": {},
                        "input": {"requested_price_item_ids": [1], "displayed_items": [{"item_id": 1}],
                                  "owned_products": [{"item_id": 1, "last_observed_price_raw": 100}]},
                        "tools": [],
                        "messages": [{"role": "system", "content": "test"},
                                     {"role": "user", "content": "test"},
                                     {"role": "assistant", "content": "test"}],
                    }
                    with self.assertRaises(error_type):
                        predict(sample, model)
                    self.assertEqual(model.complete.call_count, 1)



if __name__ == "__main__":
    unittest.main()
