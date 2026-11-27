"""Completed-run pairing and generated-CoT evaluation, using only offline fixtures."""
from argparse import Namespace
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent_simulation.data.common import sha256
from agent_simulation.evaluation.customer_cot import export_comparison, validate_training_pair
from agent_simulation.evaluation.trajectories import digest
from agent_simulation.tests.evaluation.test_customer_cot import fixture
from agent_simulation.tests.evaluation.test_sft_pilot import case
from agent_simulation.tests.ekb_fixture import rationale
from experiments.cot_sft_pilot import comparison, decision_request, dump, write_json


def training_pair(root):
    records, observed, reviews = fixture()
    release = root / "release"
    export_comparison(records, observed, reviews, release, min_rationale_scenes=1)
    paths = [release]
    options = dict(epochs=1, max_steps=None, batch_size=1, accumulation=4, learning_rate=1e-5,
                   dtype="bfloat16", rank=16, seed=42, max_length=8192, gradient_checkpointing=True,
                   click_weight=1., purchase_weight=1., rationale_weight=.3, tool_weight=1.,
                   format_weight=.25, response_decision_weight=0., limit=0)
    for name in ("action_only", "cot"):
        run = root / name; run.mkdir()
        config = {"options": deepcopy(options), "identity": {
            "mode": "sft", "role": "customer", "input_sha256": sha256(release / (name + ".jsonl")),
            "records": len(observed), "base_config_sha256": "base", "base_weight_sha256": {"weight": "fixture"},
            "tokenizer_sha256": "tokenizer", "initial_adapter_sha256": None,
            "supervision": {"objective": name, "rationale_scenes": int(name == "cot"),
                            "rationale_tokens": 50 if name == "cot" else 0}}}
        write_json(run / "config.json", config)
        write_json(run / "result.json", {"status": "completed", "mode": "sft", "steps": 1})
        paths.append(run)
    return paths


def prediction_pair(root):
    c = case(partition="dev")
    (root / "cases.jsonl").write_text(dump(c) + "\n")
    checksum = sha256(root / "cases.jsonl")
    write_json(root / "manifest.json", {"status": "complete", "cases_sha256": checksum})
    for name, style, budget in (("zero", "direct", 256), ("sft", "concise", 768)):
        path = root / name; path.mkdir()
        request = decision_request(c, style)
        r = rationale()
        r["evidence_refs"] = {"stimulus": [next(iter(c["context"]["facts"]))], "knowledge": [], "evaluation": []}
        output = c["target"] if style == "direct" else {"rationale": r, "action": c["target"]}
        row = {"case_id": c["case_id"], "status": "ok", "request_sha256": digest(request),
               "prediction": c["target"], "rationale": {} if style == "direct" else r,
               "messages": request["messages"] + [{"role": "assistant", "content": dump(output)}]}
        (path / "predictions.jsonl").write_text(dump(row) + "\n")
        write_json(path / "manifest.json", {"style": style, "cases_sha256": checksum, "ids": [c["case_id"]],
            "base_config_sha256": "base", "max_new_tokens": budget, "temperature": 0, "seed": 42,
            "partition": "dev", "model": name, "decoder_schema_sha256": "decoder"})
    return Namespace(cohorts=root, baseline=root / "zero", sft=root / "sft", output=root / "report.json",
                     response_protocol="generated_cot")


class CotProtocolTests(unittest.TestCase):
    def test_completed_pair_checks_data_initialization_steps_and_supervision(self):
        with TemporaryDirectory() as temp:
            paths = training_pair(Path(temp))
            report = validate_training_pair(*paths)
            self.assertEqual(report["steps_per_arm"], 1)
            self.assertTrue(report["comparison_id"])
            leaked = json.loads((paths[0] / "action_only.jsonl").read_text().splitlines()[0])
            with self.assertRaisesRegex(ValueError, "Evaluation parent"):
                validate_training_pair(*paths, evaluation_cases=[leaked])

    def test_pair_rejects_mismatched_or_unfinished_training(self):
        for mutation in ("data", "initialization", "steps", "options", "zero_rationale", "unfinished"):
            with self.subTest(mutation=mutation), TemporaryDirectory() as temp:
                paths = training_pair(Path(temp))
                run = paths[2]
                config = json.loads((run / "config.json").read_text())
                result = json.loads((run / "result.json").read_text())
                if mutation == "data": config["identity"]["input_sha256"] = "other"
                if mutation == "initialization": config["identity"]["initial_adapter_sha256"] = "other"
                if mutation == "steps": result["steps"] = 2
                if mutation == "options": config["options"]["learning_rate"] = .1
                if mutation == "zero_rationale": config["identity"]["supervision"]["rationale_tokens"] = 0
                if mutation == "unfinished": result["status"] = "running"
                write_json(run / "config.json", config); write_json(run / "result.json", result)
                with self.assertRaises(ValueError):
                    validate_training_pair(*paths)

    def test_generated_cot_compares_same_context_and_reports_cost_and_quality(self):
        with TemporaryDirectory() as temp:
            args = prediction_pair(Path(temp))
            with redirect_stdout(io.StringIO()): comparison(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(report["response_protocol"], "generated_cot")
            self.assertEqual(report["customer_response"]["sft_mean"], 1)
            self.assertEqual(report["output_budgets"], {"zero_shot": 256, "sft": 768})
            self.assertEqual(report["rationale_quality"]["with_cited_evidence"], 1)
            self.assertEqual(report["training_control"]["status"], "not_verified")
            args.response_protocol = "direct"
            with self.assertRaisesRegex(ValueError, "Output style"):
                comparison(args)

    def test_generated_cot_rejects_changed_context_output_and_decoder(self):
        for mutation in ("request", "messages", "action", "rationale", "decoder"):
            with self.subTest(mutation=mutation), TemporaryDirectory() as temp:
                args = prediction_pair(Path(temp))
                path = args.sft / "predictions.jsonl"
                row = json.loads(path.read_text())
                if mutation == "request": row["request_sha256"] = "different"
                if mutation == "messages": row["messages"][1]["content"] += " extra history"
                if mutation == "action": row["prediction"] = {"clicked_item_ids": [], "purchased_item_ids": []}
                if mutation == "rationale": row["rationale"] = {}
                if mutation == "decoder":
                    manifest = json.loads((args.sft / "manifest.json").read_text())
                    manifest["decoder_schema_sha256"] = "other"
                    write_json(args.sft / "manifest.json", manifest)
                path.write_text(dump(row) + "\n")
                with self.assertRaises(ValueError): comparison(args)

    def test_invalid_cot_stays_in_denominator_and_cannot_be_reviewed_as_valid(self):
        with TemporaryDirectory() as temp:
            args = prediction_pair(Path(temp))
            path = args.sft / "predictions.jsonl"
            row = json.loads(path.read_text())
            row.update(status="invalid")
            row.pop("prediction"); row.pop("messages")
            path.write_text(dump(row) + "\n")
            with redirect_stdout(io.StringIO()): comparison(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(report["invalid"]["sft"], 1)
            self.assertEqual(report["rationale_quality"]["valid"], 0)
            self.assertEqual(report["customer_response"]["sft_mean"], 0)

    def test_comparison_connects_completed_training_pair_and_rejects_partial_provenance(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            paths = training_pair(root / "training")
            evaluation = root / "evaluation"; evaluation.mkdir()
            args = prediction_pair(evaluation)
            args.comparison_release, args.action_only_training, args.cot_training = paths
            with redirect_stdout(io.StringIO()): comparison(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(report["training_control"]["steps_per_arm"], 1)
            self.assertEqual(report["arm_labels"], {"zero_shot": "action_only", "sft": "cot"})
            args.cot_training = None
            with self.assertRaisesRegex(ValueError, "both training runs"):
                comparison(args)
