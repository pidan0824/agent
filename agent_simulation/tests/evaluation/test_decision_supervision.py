"""Observed-action preservation, shared prompts and direction-sensitive curation."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
from agent_simulation.evaluation.customer_curation import observed_action_rows, fixed_response_requests, CurationTools
from agent_simulation.evaluation.customer_filtering import export_training, combine_observed_sft
from agent_simulation.evaluation.customer_ekb_ablation import generate
from agent_simulation.evaluation.trajectories import score_group, OutcomeScorer, generate_trajectory
from agent_simulation.agents.agentic import AgenticOptions
from agent_simulation.agents.context.seller_features import with_evidence_id
from agent_simulation.evaluation.seller_quality import check_seller_evidence
from agent_simulation.tests.evaluation.test_customer_agentic import request
from agent_simulation.tests.evaluation.test_customer_filtering import record, action
from agent_simulation.tests.ekb_fixture import rationale


class DecisionSupervisionTests(unittest.TestCase):
    def test_failed_teacher_preserves_full_observed_action_and_never_reuses_bad_rationale(self):
        source = request()
        source["assigned_split"] = "train"
        source["response_input"]["displayed_items"].append({"item_id": 5, "item_title": "Other"})
        source["response_target"] = action([4, 5], [5])
        observed = observed_action_rows(source, 11, "train", "release")
        row = record("wrong", "strong", action(), source["response_target"], None)
        row["trajectory"].update(source_request_id=11, dataset_manifest_sha256="release")
        original = deepcopy(row)
        with TemporaryDirectory() as d:
            report = export_training([row], Path(d), observed_actions=observed, sft_only=True)
            saved = [json.loads(line) for line in (Path(d) / "sft.jsonl").read_text().splitlines()]
        self.assertEqual(report["observed_action_rows"], 2)
        self.assertEqual(report["pairs_total"], 0)
        self.assertEqual(row, original)
        response = next(r for r in saved if r["stage"] == "response")
        self.assertEqual(json.loads(response["messages"][-1]["content"]), source["response_target"])
        self.assertFalse(any(m["role"] in ("assistant", "tool") for m in response["messages"][:-1]))
        self.assertEqual(observed_action_rows(source, 11, "test", "release"), [])

    def test_observed_pool_rejects_cross_split_or_modified_action(self):
        source = request()
        observed = observed_action_rows(source, 11, "train", "release")
        bad = deepcopy(observed)
        bad[-1]["messages"][-1]["content"] = json.dumps(action())
        with self.assertRaisesRegex(ValueError, "mismatch"):
            combine_observed_sft([], [], bad)
        row = record("heldout", "strong", action(), action(), None)
        row["trajectory"].update(source_request_id=11, dataset_manifest_sha256="release", source_split="test")
        row["source_split"] = "test"
        with self.assertRaisesRegex(ValueError, "crosses"):
            combine_observed_sft([], [row], observed)

    def test_controlled_eval_and_fixed_curation_send_identical_prompts(self):
        source = request()
        context, requests = fixed_response_requests(source["response_input"])
        self.assertEqual(requests["direct"]["messages"][1:], requests["ekb"]["messages"][1:])
        calls = []
        class Model:
            seed, model = 1, "fixture"
            async def acomplete(self, value, **kwargs):
                calls.append(deepcopy(value))
                return {"role": "assistant", "content": json.dumps({"rationale": rationale(), "action": action()}), "tool_calls": []}
        binding = CurationTools(source, "response", mode="fixed")
        asyncio.run(generate_trajectory(binding, Model(), role="customer", actor_id=7, stage="response",
            record_id=11, source_split="train", catalog_identity="release",
            options=AgenticOptions(max_model_calls=1), sample_index=0))
        asyncio.run(generate(Model(), context, "ekb", 0, request=requests["ekb"]))
        self.assertEqual(calls[0], calls[1])

    def test_sft_only_pipeline_never_connects_weak_or_predictor(self):
        from agent_simulation.evaluation.customer_curation import main
        source = request()
        source["assigned_split"] = "train"
        class Client:
            model, seed = "fixture", 42
            def __init__(self, *args, **kwargs):
                pass
            def check_models(self):
                pass
            def close(self):
                pass
            async def acomplete(self, value, **kwargs):
                return {"role": "assistant", "content": json.dumps({
                    "rationale": rationale(), "action": source["response_target"]}), "tool_calls": []}
        data = Mock()
        data.catalog.request.return_value = source
        data.catalog.connection.execute.return_value.fetchall.return_value = [(11,)]
        with TemporaryDirectory() as d:
            root = Path(d)
            (root / "manifest.json").write_text("{}")
            with patch("agent_simulation.evaluation.customer_curation.SFTDataset", return_value=data), \
                 patch("agent_simulation.evaluation.customer_curation.AgenticClient", side_effect=Client) as clients, \
                 patch("agent_simulation.evaluation.customer_curation.load_tokenizer", side_effect=AssertionError("weak tokenizer")), \
                 patch("agent_simulation.evaluation.customer_curation.OptionPredictor", side_effect=AssertionError("predictor")):
                main(["--dataset", str(root), "--output", str(root / "out"), "--candidate-policies", "full",
                      "--request-id", "11", "--count", "1", "--samples", "1", "--sft-only"])
            self.assertEqual(clients.call_count, 1)
            rows = [json.loads(v) for v in (root / "out/sft.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(sum(r["response_style"] == "direct" for r in rows), 2)
            self.assertEqual((root / "out/preferences.jsonl").read_text(), "")

    def test_seller_rejects_small_wrong_direction_and_false_changes(self):
        for actual, predicted, accepted in ((102,100,False), (100,102,False), (102,99,False),
                                             (102,102,True), (100,100,True), (98,98,True)):
            row = {"stage": "seller", "status": "ok", "previous_prices": {"1": 100},
                   "final_action": {"prices": [{"item_id": 1, "price_raw": predicted}]}}
            score_group([row], {"prices": [{"item_id": 1, "price_raw": actual}]}, OutcomeScorer())
            self.assertEqual(row["accepted"], accepted)
        row.pop("previous_prices")
        score_group([row], {"prices": [{"item_id": 1, "price_raw": 98}]}, OutcomeScorer())
        self.assertFalse(row["accepted"])
        self.assertIn("missing_previous_price", row["quality_reasons"])

    def test_seller_citations_must_resolve_to_actual_observations(self):
        facts = with_evidence_id({"last_price": 100})
        r = {"hypothesis": "keep", "evidence": ["Previous price [" + facts["evidence_id"] + "]"], "uncertainty": "unknown costs"}
        a = {"prices": [{"item_id": 1, "price_raw": 100}]}
        t = {"rationale": r, "final_action": a, "messages": [
            {"role": "user", "content": json.dumps(facts)},
            {"role": "assistant", "content": json.dumps({"rationale": r, "action": a})}]}
        self.assertTrue(check_seller_evidence(t)["accepted"])
        t["semantic_review"] = {"verdict": "fail"}
        self.assertFalse(check_seller_evidence(t)["accepted"])
        del t["semantic_review"]
        facts["last_price"] = 200
        t["messages"][0]["content"] = json.dumps(facts)
        self.assertFalse(check_seller_evidence(t)["accepted"])


if __name__ == "__main__":
    unittest.main()
