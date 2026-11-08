"""Model boot failures remain visible through the assembled main/logging path."""

from agent_simulation import PROJECT_ROOT

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agent_simulation.simulation.run import parser, run
from agent_simulation.tests.fixtures import TokenizerFixture


class WorkerFixture:
    """Spawned by the production service, without the model loader."""
    def __init__(self, **options):
        self.generator_id = "fixture-shared"
        self.batches = options["log_path"].with_name("fixture_batches.jsonl")
        self.routes = options["role_adapters"]

    def complete_batch(self, jobs):
        with self.batches.open("a") as out:
            out.write(json.dumps([str(self.routes.get(req.get("agent_role"), "__base__"))
                                  for _, req, _ in jobs]) + "\n")
        return [{"role": "assistant", "tool_calls": [],
                 "content": json.dumps({"action": "no_update"}
                    if "memory_generation_method" in request else {"action": "no_action"})}
                for _, request, _ in jobs]


class MainTests(unittest.TestCase):
    def test_served_model_name_requires_the_vllm_backend(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "episode"
            args = parser().parse_args([
                "--catalog", "unused", "--model", "unused", "--output", str(output),
                "--customers", "1", "--served-model", "seller-sft",
            ])
            with self.assertRaisesRegex(ValueError, "require --inference-backend vllm"):
                run(args)
            self.assertFalse(output.exists())

    def test_close_failure_preserves_decision_error_and_writes_final_state(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        for fail_decision in (True, False):
            with self.subTest(fail_decision=fail_decision), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / "episode"
                args = parser().parse_args([
                    "--catalog", str(PROJECT_ROOT / "data/kuaisearch/lite/source/catalog"),
                    "--model", "fixture", "--output", str(output), "--sellers", "6", "85",
                    "--customers", "1", "--ticks", "1", "--history-retrieval", "structured",
                ])
                decision_error = ValueError("decision failed")
                close_error = OSError("close failed")
                def complete(request, *, call_id):
                    if fail_decision:
                        raise decision_error
                    answer = ({"action": "no_update"} if "memory_generation_method" in request
                              else {"action": "no_action"})
                    return {"role": "assistant", "content": json.dumps(answer)}
                model = SimpleNamespace(generator_id="fixture", tokenizer=TokenizerFixture(),
                                        max_new_tokens=100, complete=complete, close=Mock(side_effect=close_error))
                with redirect_stdout(io.StringIO()), self.assertRaises((ValueError, OSError)) as caught:
                    run(args, load_model=lambda _: (model, {}))
                self.assertIs(caught.exception, decision_error if fail_decision else close_error)
                model.close.assert_called_once()
                report = json.loads((output / "report.json").read_text())
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["finalization_errors"], ["close model: OSError: close failed"])
                self.assertIn("decision failed" if fail_decision else "close failed", report["error"])
                state = json.loads((output / "final_state.json").read_text())
                self.assertEqual(state["clock"]["status"], "failed" if fail_decision else "complete")

    def test_default_service_path_runs_all_roles_without_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "evaluation"
            args = parser().parse_args([
                "--catalog", str(PROJECT_ROOT / "data/kuaisearch/lite/source/catalog"),
                "--model", "fixture", "--output", str(output), "--sellers", "6", "85",
                "--customers", "1", "--ticks", "1", "--history-retrieval", "structured",
                "--customer-adapter", "fixture/customer", "--seller-adapter", "fixture/seller",
                "--max-concurrent-requests", "2", "--request-timeout", "10"])
            prefix = "agent_simulation.simulation.run."
            with (patch(prefix + "QwenLocal", WorkerFixture),
                  patch(prefix + "verify_checkpoint", side_effect=AssertionError("Default run must not hash model weights")),
                  patch(prefix + "load_tokenizer", return_value=TokenizerFixture()),
                  redirect_stdout(io.StringIO())):
                run(args)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["model_id"], "fixture-shared")
            batches = [json.loads(line) for line in (output / "fixture_batches.jsonl").read_text().splitlines()]
            self.assertGreaterEqual(sum(map(len, batches)), 3)
            self.assertLessEqual(max(map(len, batches)), 2)
            observed = {route for batch in batches for route in batch}
            self.assertTrue({"fixture/customer", "fixture/seller"} <= observed)
    def test_custom_model_runs_agents_and_episode_without_loading_qwen(self):
        class ClientFixture:
            generator_id = "fixture:other-backend"
            max_new_tokens = 100

            def __init__(self):
                self.tokenizer = self
                self.chat_template = "fixture-only"
                self.calls = []

            def apply_chat_template(self, messages, tools, **kwargs):
                return list(range(len(json.dumps([messages, tools])) // 4))

            def complete(self, request, *, call_id):
                self.calls.append(call_id)
                result = ({"action": "no_update"}
                          if "memory_generation_method" in request else {"action": "no_action"})
                return {"role": "assistant", "content": json.dumps(result), "tool_calls": []}

        root = PROJECT_ROOT
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "episode"
            args = parser().parse_args([
                "--catalog", str(root / "data/kuaisearch/lite/source/catalog"),
                "--model", "fixture-unused-path", "--output", str(output),
                "--sellers", "6", "85", "--customers", "1", "--ticks", "1",
                "--history-retrieval", "structured",
            ])
            model = ClientFixture()
            with patch("agent_simulation.simulation.run.load_default_model", side_effect=AssertionError("Qwen loaded")), \
                 redirect_stdout(io.StringIO()):
                run(args, load_model=lambda config: (model, {"fixture_backend": True}))
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["model_id"], model.generator_id)
            self.assertEqual(report["checkpoint"], {"fixture_backend": True})
            self.assertGreaterEqual(len(model.calls), 3)

    def test_model_boot_failure_is_not_masked_by_uninitialized_logs(self):
        root = PROJECT_ROOT
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "failed_run"
            args = parser().parse_args(
                [
                    "--catalog",
                    str(root / "data/kuaisearch/lite/source/catalog"),
                    "--model",
                    str(root / "models/Qwen2.5-3B-Instruct"),
                    "--output",
                    str(output),
                    "--sellers",
                    "6",
                    "85",
                    "--customers",
                    "1",
                ]
            )
            self.assertEqual(args.history_retrieval, "bge")
            with (
                patch("agent_simulation.simulation.run.verify_checkpoint", return_value={"verified": True}),
                patch(
                    "agent_simulation.simulation.run.load_tokenizer", return_value=None
                ),
                patch("agent_simulation.inference.local.service.LocalInferenceService",
                      side_effect=ValueError("model boot fixture")),
                redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(ValueError, "model boot fixture"),
            ):
                run(args)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["error"], "ValueError: model boot fixture")
            self.assertEqual(report["agent_tool_calls"], {})
            with sqlite3.connect(output / "runtime.sqlite") as db:
                self.assertEqual(db.execute("SELECT count(*) FROM agent_runs").fetchone()[0], 0)
                state = json.loads(db.execute("SELECT value FROM state WHERE id=1").fetchone()[0])
                self.assertEqual({p["seller_id"] for p in state["products"].values()}, {6, 85})
                self.assertEqual(set(state["initial_seller_memory"]), {"6", "85"})
