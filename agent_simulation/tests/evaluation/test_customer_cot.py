"""Offline CoT release gates and matched action-only controls."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent_simulation.agents.agentic import AgenticOptions, initial_agentic_request
from agent_simulation.datasets.trajectory import read_training_records, sft_supervision
from agent_simulation.evaluation.customer_context import CurationTools
from agent_simulation.evaluation.customer_cot import export_comparison, prepare_comparison, write_review_audit
from agent_simulation.evaluation.customer_curation import observed_action_rows, main
from agent_simulation.evaluation.trajectories import digest, trajectory_messages
from agent_simulation.tests.evaluation.test_customer_generation import job
from agent_simulation.tests.ekb_fixture import rationale
from agent_simulation.tests.training.test_trajectory_supervision import OffsetTokenizer


def fixture():
    value = job(11)
    source, candidate = value[4:6]
    bundle = {"request": source, "query_enabled": True,
              "response_scenes": [{"request": source, "candidate": candidate}]}
    observed = observed_action_rows(source, 11, "train", "release", bundle=bundle)
    binding = CurationTools(source, "response", mode="fixed", concise=True)
    _, prompt = trajectory_messages(binding, role="customer", stage="response", theory=True)
    prompt, _, _ = initial_agentic_request(prompt, [], AgenticOptions(max_model_calls=1))
    explanation = rationale()
    explanation["evidence_refs"]["knowledge"] = [next(iter(binding.initial_context()["facts"]))]
    t = {**deepcopy(observed[1]), "trajectory_id": "teacher-1", "status": "ok",
         "response_style": "ekb", "rationale": explanation, "final_action": source["response_target"],
         "messages": prompt + [{"role": "assistant", "content": json.dumps({
             "rationale": explanation, "action": source["response_target"]})}]}
    records = [{"source_split": "train", "variant": "strong", "trajectory": t}]
    reviews = {"teacher-1": {"trajectory_id": "teacher-1", "messages_sha256": digest(t["messages"]),
               "verdict": "pass", "reason": "Offline fixture only", "reviewer": "test"}}
    return records, observed, reviews


class CotComparisonTests(unittest.TestCase):
    def test_pair_keeps_scene_order_labels_inputs_weights_and_real_rationale_mask(self):
        records, observed, reviews = fixture()
        before = deepcopy((records, observed, reviews))
        with TemporaryDirectory() as d:
            out = Path(d) / "pair"
            report = export_comparison(records, observed, reviews, out, min_rationale_scenes=1)
            self.assertEqual(report["rationale_scenes"], 1)
            raw = {name: [json.loads(s) for s in (out / (name + ".jsonl")).read_text().splitlines()]
                   for name in ("action_only", "cot")}
            for direct, cot in zip(raw["action_only"], raw["cot"], strict=True):
                for field in ("target", "actor_id", "source_request_id", "scene_id"):
                    self.assertEqual(direct[field], cot[field])
                self.assertEqual(direct["messages"][1:-1], cot["messages"][1:-1])
            self.assertEqual(raw["cot"][1]["messages"], records[0]["trajectory"]["messages"])
            a, b = [read_training_records(out / (name + ".jsonl"), "sft", OffsetTokenizer())
                    for name in ("action_only", "cot")]
            self.assertEqual([r["weight"] for r in a], [r["weight"] for r in b])
            self.assertEqual(sft_supervision(a)["rationale_tokens"], 0)
            self.assertGreater(sft_supervision(b)["rationale_tokens"], 0)
            with self.assertRaisesRegex(ValueError, "rationale supervision"):
                read_training_records(out / "cot.jsonl", "sft", OffsetTokenizer(), limit=1)
            with self.assertRaisesRegex(ValueError, "rationale supervision"):
                read_training_records(out / "action_only.jsonl", "sft", OffsetTokenizer(), require_rationales=True)
        self.assertEqual((records, observed, reviews), before)

    def test_minimum_counts_scenes_not_multiple_versions_of_one_scene(self):
        records, observed, reviews = fixture()
        with TemporaryDirectory() as d:
            out = Path(d) / "pair"
            export_comparison(records, observed, reviews, out, min_rationale_scenes=1)
            rows = [json.loads(s) for s in (out / "cot.jsonl").read_text().splitlines()]
            one = rows[-1]
            duplicate = deepcopy(one)
            duplicate.update(trajectory_id="other-version", initial_state_id="other-state")
            values = [one, duplicate]
            for row in values:
                row["min_rationale_scenes"] = 2
            path = Path(d) / "duplicated-scenes.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in values) + "\n")
            with self.assertRaisesRegex(ValueError, "Insufficient rationale"):
                read_training_records(path, "sft", OffsetTokenizer())

    def test_zero_unreviewed_or_insufficient_rationales_never_export_training_files(self):
        records, observed, reviews = fixture()
        for rs, rv, minimum in (([], {}, 1), (records, {}, 1), (records, reviews, 2)):
            with self.subTest(minimum=minimum, reviewed=bool(rv)), TemporaryDirectory() as d:
                out = Path(d) / "pair"
                with self.assertRaisesRegex(ValueError, "Insufficient reviewed"):
                    export_comparison(rs, observed, rv, out, min_rationale_scenes=minimum)
                self.assertTrue((out / "report.json").exists())
                self.assertFalse((out / "cot.jsonl").exists())
                self.assertFalse((out / "action_only.jsonl").exists())

    def test_mismatched_or_stale_review_cannot_authorize_export(self):
        records, observed, reviews = fixture()
        reviews["teacher-1"]["messages_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "transcript"):
            prepare_comparison(records, observed, reviews)
        reviews["teacher-1"]["messages_sha256"] = digest(records[0]["trajectory"]["messages"])
        reviews["teacher-1"]["verdict"] = "fail"
        _, _, report, decisions, _ = prepare_comparison(records, observed, reviews)
        self.assertEqual(report["rationale_scenes"], 0)
        self.assertIn("semantic_review_failed", decisions[0]["reasons"])

    def test_missing_citations_or_wrong_action_are_rejected_without_rewriting(self):
        for change, reason in (("citation", "no_cited_evidence"), ("action", "action_mismatch")):
            records, observed, reviews = fixture()
            t = records[0]["trajectory"]
            if change == "citation":
                t["rationale"]["evidence_refs"] = {k: [] for k in t["rationale"]["evidence_refs"]}
            else:
                t["final_action"] = {"clicked_item_ids": [], "purchased_item_ids": []}
            t["messages"][-1]["content"] = json.dumps({"action": t["final_action"], "rationale": t["rationale"]})
            t["evidence_quality"] = {"accepted": True}
            reviews["teacher-1"]["messages_sha256"] = digest(t["messages"])
            _, _, report, decisions, _ = prepare_comparison(records, observed, reviews)
            self.assertEqual(report["rationale_scenes"], 0)
            self.assertIn(reason, decisions[0]["reasons"])

    def test_changed_frozen_context_is_not_reused_even_with_same_scene_id(self):
        records, observed, reviews = fixture()
        t = records[0]["trajectory"]
        t["messages"][-2]["content"] += "\nA changed observation"
        reviews["teacher-1"]["messages_sha256"] = digest(t["messages"])
        _, _, report, decisions, _ = prepare_comparison(records, observed, reviews)
        self.assertEqual(report["rationale_scenes"], 0)
        self.assertIn("frozen_input_mismatch", decisions[0]["reasons"])

    def test_one_teacher_per_scene_and_reviews_do_not_depend_on_whole_cohort_hash(self):
        records, observed, reviews = fixture()
        duplicate = deepcopy(records[0])
        duplicate["trajectory"]["trajectory_id"] = "teacher-2"
        reviews["teacher-2"] = {**reviews["teacher-1"], "trajectory_id": "teacher-2"}
        records.append(duplicate)
        extra = deepcopy(observed[0])
        extra.update(trajectory_id="query-other", source_request_id=12, scene_id="12")
        observed.append(extra)
        direct, cot, report, decisions, _ = prepare_comparison(records, observed, reviews)
        self.assertEqual((len(direct), len(cot), report["rationale_scenes"]), (3, 3, 1))
        self.assertIn("duplicate_scene", decisions[1]["reasons"])

    def test_generation_review_queue_is_untruncated_and_not_a_training_release(self):
        records, observed, _ = fixture()
        with TemporaryDirectory() as d:
            report = write_review_audit(records, observed, d)
            self.assertEqual(report["status"], "awaiting_semantic_review")
            queue = [json.loads(s) for s in (Path(d) / "cot-review.jsonl").read_text().splitlines()]
            self.assertEqual(queue[0]["messages_sha256"], digest(records[0]["trajectory"]["messages"]))
            self.assertFalse((Path(d) / "sft.jsonl").exists())

    def test_held_out_and_wrong_mode_fail_before_model_access(self):
        records, observed, reviews = fixture()
        records[0]["source_split"] = records[0]["trajectory"]["source_split"] = "test"
        with self.assertRaises(ValueError):
            prepare_comparison(records, observed, reviews)
        for flags in ([], ["--sft-only", "--teacher-budget", "0", "--cot-preset", "lightweight"],
                      ["--sft-only", "--modes", "autonomous"]):
            with self.assertRaises(ValueError):
                main(["--dataset", "unused", "--output", "unused", "--cot-sft", *flags])

    def test_zero_rationale_weight_is_rejected_before_loading_model(self):
        from unittest.mock import patch
        from agent_simulation.training.trajectory import parser, train
        records, observed, reviews = fixture()
        with TemporaryDirectory() as d:
            root = Path(d)
            export_comparison(records, observed, reviews, root / "pair", min_rationale_scenes=1)
            args = parser().parse_args(["--mode", "sft", "--input", str(root / "pair/cot.jsonl"),
                "--output", str(root / "training"), "--device", "cpu", "--rationale-weight", "0"])
            with patch("agent_simulation.training.trajectory.load_tokenizer", return_value=OffsetTokenizer()), \
                 patch("agent_simulation.training.run.load_base_model") as model:
                with self.assertRaisesRegex(ValueError, "positive rationale loss"):
                    train(args)
                model.assert_not_called()
            self.assertFalse((root / "training").exists())

    def test_internal_audit_rows_cannot_enter_training_even_inside_official_train(self):
        records, observed, reviews = fixture()
        observed[0]["partition"] = "audit"
        with self.assertRaisesRegex(ValueError, "dev/audit"):
            prepare_comparison(records, observed, reviews)

    def test_cot_marker_cannot_hide_missing_supervision_or_mix_control_arms(self):
        _, observed, _ = fixture()
        for mode in ("zero", "mixed"):
            values = deepcopy(observed)
            for row in values:
                row.update(sft_objective="cot", comparison_id="pair", min_rationale_scenes=1)
            if mode == "mixed":
                values[0]["sft_objective"] = "action_only"
            with TemporaryDirectory() as d:
                path = Path(d) / "data.jsonl"
                path.write_text("".join(json.dumps(r) + "\n" for r in values))
                with self.assertRaises(ValueError):
                    read_training_records(path, "sft", OffsetTokenizer())
