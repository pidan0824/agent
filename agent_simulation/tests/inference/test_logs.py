"""Logs preserve failed model output and Agent traces without executing actions."""

from contextlib import nullcontext
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

import torch

from agent_simulation.inference.client import ModelOutputError
from agent_simulation.inference.local.qwen import QwenLocal
from agent_simulation.logs import AgentRunLog, ModelLog


class LogTests(unittest.TestCase):
    def test_concurrent_reports_publish_complete_json(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        from unittest.mock import patch
        from agent_simulation.logs import write_report

        ready = Barrier(2)
        replace = Path.replace

        def publish(temporary, destination):
            ready.wait(timeout=5)
            return replace(temporary, destination)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "report.json"
            values = [{"writer": "甲"}, {"writer": "乙"}]
            with patch.object(Path, "replace", publish), ThreadPoolExecutor(2) as pool:
                results = [pool.submit(write_report, path, value) for value in values]
                for result in results:
                    result.result()
            self.assertIn(json.loads(path.read_text()), values)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_failed_report_preserves_previous_file_and_propagates_error(self):
        from unittest.mock import patch
        from agent_simulation.logs import write_report

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "report.json"
            write_report(path, {"done": 1})
            with patch.object(Path, "replace", side_effect=OSError("cannot publish")):
                with self.assertRaisesRegex(OSError, "cannot publish"):
                    write_report(path, {"done": 2})
            self.assertEqual(json.loads(path.read_text()), {"done": 1})
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_batched_transport_preserves_per_row_grammar_eos_and_audit(self):
        from transformers import BatchEncoding
        from unittest.mock import Mock
        import time
        backend = object.__new__(QwenLocal)
        backend.tokenizer = Mock(pad_token_id=0)
        backend.tokenizer.return_value = BatchEncoding({
            "input_ids": torch.tensor([[0, 3], [6, 3]]),
            "attention_mask": torch.tensor([[0, 1], [1, 1]])})
        backend.tokenizer.decode.side_effect = ['{"a":1}', '{"b":1}']
        backend.tokenizer.apply_chat_template.return_value = "prompt"
        backend.device, backend.max_new_tokens, backend.generator_id = "cpu", 10, "fixture"
        backend._torch, backend.log = torch, Mock()
        backend._decoding = lambda req: (
            {"prefix_allowed_tokens_fn": lambda idx, tokens: [req["token"]]} if "token" in req else {}
        )
        def generate(input_ids, prefix_allowed_tokens_fn, **kwargs):
            self.assertEqual(prefix_allowed_tokens_fn(0, input_ids[0]), [11])
            self.assertEqual(prefix_allowed_tokens_fn(1, input_ids[1]), list(range(16)))
            return torch.tensor([[0, 3, 11, 2, 0], [6, 3, 12, 12, 2]])
        backend.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=128, vocab_size=16),
            generation_config=SimpleNamespace(eos_token_id=[2]), generate=generate)
        jobs = [(str(i), {"token": i, "messages": [], "tools": []}, time.monotonic()+5)
                for i in (11, 12)]
        del jobs[1][1]["token"]  # A plain request can share a batch with a constrained one.
        replies = backend.complete_batch(jobs)
        self.assertEqual([r["content"] for r in replies], ['{"a":1}', '{"b":1}'])
        calls = [c.args[0] for c in backend.log.append.call_args_list]
        self.assertEqual([(c["call_id"], c["input_tokens"], c["output_tokens"]) for c in calls],
                         [("11", 1, 2), ("12", 2, 3)])
        self.assertEqual(backend.tokenizer.call_args.kwargs["padding_side"], "left")

    def test_batch_capacity_is_checked_before_prefix_prefill(self):
        from unittest.mock import Mock

        backend = object.__new__(QwenLocal)
        backend.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8))
        backend.max_new_tokens = 3
        backend._inputs = Mock(return_value=SimpleNamespace(input_ids=torch.zeros(1, 6)))
        backend._reuse_prefix = Mock(side_effect=AssertionError("Prefix prefill must wait for capacity check"))
        with self.assertRaisesRegex(ValueError, "checkpoint capacity"):
            backend.complete_batch([("overflow", {}, 1)])
        backend._reuse_prefix.assert_not_called()

    def test_agent_log_exists_before_the_first_workflow_and_saves_exact_traces(self):
        with sqlite3.connect(":memory:") as db:
            log = AgentRunLog(db)
            self.assertEqual(log.tool_counts(), {})
            run = log.start("seller", 10)
            run["tools"].append({"name": "query_business", "arguments": {"item_ids": [1]}})
            log.save(run, "failed", "explicit fixture failure")
            status, raw, error = db.execute("SELECT status,trace,error FROM agent_runs").fetchone()
            self.assertEqual(status, "failed")
            self.assertEqual(json.loads(raw), run)
            self.assertEqual(error, "explicit fixture failure")
            self.assertEqual(log.tool_counts(), {"query_business": 1})

    def test_model_api_records_raw_output_before_rejecting_invalid_json(self):
        # A token transport fixture verifies the API/log boundary, not real model quality.
        class Tokens(dict):
            def __init__(self):
                super().__init__(input_ids=torch.tensor([[1, 3]]))
                self.input_ids = self["input_ids"]

            def to(self, device):
                return self

        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return "prompt fixture"

            def __call__(self, *args, **kwargs):
                return Tokens()

            def decode(self, *args, **kwargs):
                return '{"broken":'

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.jsonl"
            model = object.__new__(QwenLocal)
            model.tokenizer, model.device = Tokenizer(), "cpu"
            model.max_new_tokens, model.generator_id = 10, "fixture-only"
            model._torch = SimpleNamespace(inference_mode=nullcontext)
            model.model = SimpleNamespace(
                config=SimpleNamespace(max_position_embeddings=100),
                generation_config=SimpleNamespace(eos_token_id=2),
                generate=lambda **kwargs: torch.tensor([[1, 3, 2]]),
            )
            model.log = ModelLog(path)
            request = {"messages": [], "tools": []}
            with self.assertRaises(ModelOutputError):
                model.complete(request, call_id="fixture")
            record = json.loads(path.read_text())
            self.assertEqual(record["raw_output"], '{"broken":')
            self.assertTrue(record["finished"])
            self.assertEqual(record["request"], request)
            with self.assertRaises(FileExistsError):
                ModelLog(path)
