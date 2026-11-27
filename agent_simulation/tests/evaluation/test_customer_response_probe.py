"""Probe correctness without loading real model weights or updating any parameters."""
from argparse import Namespace
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from experiments.customer_response_probe import without_history, prepare, encode_options, option_scores
from experiments.cot_sft_pilot import decision_request, dump, write_json
from agent_simulation.data.common import sha256
from agent_simulation.evaluation.customer_evidence import expand_sources
from agent_simulation.evaluation.customer_contract import single_item_actions
from agent_simulation.evaluation.trajectories import digest
from agent_simulation.tests.evaluation.test_sft_pilot import case
from agent_simulation.tests.training.test_trajectory_training import CharacterChatTokenizer, ContextModel


class OffsetTokenizer(CharacterChatTokenizer):
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) + 1 for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def context(request):
    return expand_sources(json.loads(request["messages"][-1]["content"]))


class ResponseProbeTests(unittest.TestCase):
    def test_ablation_removes_events_facts_and_statistics_preserves_current_input(self):
        request = decision_request(case(), "direct")
        before = deepcopy(request)
        original = context(request)
        ablated = without_history(request)
        after = context(ablated)
        self.assertEqual(request, before)
        self.assertEqual(request["messages"][:-1], ablated["messages"][:-1])
        self.assertEqual(request["final_schema"], ablated["final_schema"])
        for field in ("profile", "query", "candidate_context"):
            self.assertEqual(original[field], after[field])
        self.assertNotIn("behavior_features", after)
        self.assertNotIn("available_history_events", after)
        self.assertEqual(after["history_events"], [])
        self.assertTrue(all(f["kind"] in ("product", "query") for f in after["facts"].values()))
        self.assertLess(len(after["facts"]), len(original["facts"]))
        for item in after["candidate_context"]:
            self.assertTrue(set(item["fact_refs"].values()) <= set(after["facts"]))

    def test_unknown_context_fields_require_review_not_silent_history_leak(self):
        request = decision_request(case(), "direct")
        value = json.loads(request["messages"][-1]["content"])
        value["model_summary"] = "hidden historical preference"
        request["messages"][-1]["content"] = dump(value)
        with self.assertRaisesRegex(ValueError, "Unknown context"):
            without_history(request)

    def test_prepare_is_deterministic_dev_only_one_child_per_parent(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            cases = []
            for i, partition in enumerate(("fit", "dev", "dev", "dev", "audit")):
                c = case(partition=partition)
                c.update(case_id=str(i), source_request_id=1 if i in (1, 2) else i,
                         action_mode="single_item", candidate_item_ids=[4])
                cases.append(c)
            (root / "cases.jsonl").write_text("".join(dump(c) + "\n" for c in cases))
            write_json(root / "manifest.json", {"status": "complete", "cases_sha256": sha256(root / "cases.jsonl")})
            args = Namespace(cohorts=root, count=2, max_candidates=5, seed=17, output=root / "a.json")
            with redirect_stdout(io.StringIO()):
                prepare(args)
                args.output = root / "b.json"
                prepare(args)
            self.assertEqual((root / "a.json").read_bytes(), (root / "b.json").read_bytes())
            plan = json.loads(args.output.read_text())
            self.assertEqual({c["source_request_id"] for c in plan["cases"]}, {1, 3})
            self.assertTrue(all(c["partition"] == "dev" for c in plan["cases"]))
            self.assertEqual(plan["cases_digest"], digest(plan["cases"]))

    def test_scores_match_full_causal_logprob_without_parameter_updates(self):
        tokenizer = OffsetTokenizer()
        request = {"messages": [{"role": "user", "content": "Frozen context"}], "tools": []}
        c = {"requests": {"full": request}, "request_sha256": {"full": digest(request)},
             "actions": single_item_actions([4])}
        turns = encode_options(c, "full", tokenizer, 2000)
        torch.manual_seed(11)
        model = ContextModel().eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        scores = option_scores(model, turns, 0, "cpu")
        with torch.inference_mode():
            for turn, result in zip(turns, scores):
                ids = torch.tensor([turn.input_ids])
                outputs = model(ids, torch.ones_like(ids), logits_to_keep=torch.arange(ids.shape[1] - 1))
                logprobs = outputs.logits.log_softmax(-1)
                labels = torch.tensor(turn.labels[1:])
                pos = labels.ne(-100).nonzero().flatten()
                expected = logprobs[0, pos, labels[pos]].sum().item()
                self.assertAlmostEqual(result["logprob"], expected, places=3)
        self.assertAlmostEqual(sum(r["menu_probability"] for r in scores), 1)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        changed = deepcopy(c)
        changed["target"] = {"not": "part of model input"}
        self.assertEqual(encode_options(changed, "full", tokenizer, 2000), turns)
        changed["requests"]["full"]["messages"][0]["content"] = "tampered"
        with self.assertRaisesRegex(ValueError, "Changed frozen"):
            encode_options(changed, "full", tokenizer, 2000)

    def test_score_entry_writes_paired_complete_reports_without_training(self):
        from unittest.mock import patch
        from experiments.customer_response_probe import score
        from agent_simulation.evaluation.customer_evidence import encode_context
        from agent_simulation.training.trajectory import base_weight_digest
        ctx = {"profile": {}, "query": "cup", "candidate_context": [
            {"item_id": 4, "fact_refs": {"item_title": "P4"}}],
            "history_events": [], "facts": {
                "P4": {"kind": "product", "item_id": 4, "actor_id": None, "time_index": None,
                       "source_id": "item:4", "field": "item_title", "value": "cup"}}}
        full = {"messages": [{"role": "user", "content": encode_context(ctx)}], "tools": []}
        requests = {"full": full, "without_history": without_history(full)}
        actions = single_item_actions([4])
        cases = [{"case_id": "dev-1", "partition": "dev", "stratum": "no_click/no_purchase",
                  "target": actions[0], "actions": actions, "requests": requests,
                  "request_sha256": {k: digest(v) for k, v in requests.items()}}]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir, run = root / "model", root / "sft"
            model_dir.mkdir(); (run / "adapter").mkdir(parents=True)
            (model_dir / "config.json").write_text("{}")
            (model_dir / "model.safetensors").write_text("fixture")
            (run / "adapter/adapter_config.json").write_text("{}")
            (run / "adapter/adapter_model.safetensors").write_text("fixture")
            write_json(run / "config.json", {"identity": {
                "role": "customer", "mode": "sft",
                "base_config_sha256": sha256(model_dir / "config.json"),
                "base_weight_sha256": base_weight_digest(model_dir)}})
            plan = root / "plan.json"
            write_json(plan, {"cases": cases, "cases_digest": digest(cases)})
            args = Namespace(plan=plan, model=model_dir, sft_run=run, output=root / "scores",
                             device="cpu", dtype="float32", max_length=4096)
            model = ContextModel().eval()
            before = {k: v.clone() for k, v in model.state_dict().items()}
            with patch("experiments.customer_response_probe.load_tokenizer", return_value=OffsetTokenizer()), \
                 patch("agent_simulation.training.run.load_base_model", return_value=model), \
                 patch("peft.PeftModel.from_pretrained", side_effect=lambda m, *a, **kw: m), \
                 redirect_stdout(io.StringIO()):
                score(args)
            result = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual((result["status"], result["records"]), ("complete", 4))
            report = json.loads((args.output / "report.json").read_text())
            self.assertEqual(len(report["gold_scores_by_behavior"]), 4)
            self.assertEqual(report["metrics"]["base"], report["metrics"]["sft"])
            self.assertTrue(report["no_training"])
            self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
