"""Offline checks for leakage, source references, action contracts and strict DPO."""
from copy import deepcopy
import json
import unittest
import numpy as np
from agent_simulation.evaluation.customer_contract import single_item_schema, validate_request_splits
from agent_simulation.evaluation.customer_evidence import attach_sources, check_evidence
from agent_simulation.evaluation.customer_filtering import select_preferences, probability_record, select_sft
from agent_simulation.agents.schemas import validate_arguments
from agent_simulation.tests.evaluation.test_customer_filtering import record, action
from agent_simulation.tests.ekb_fixture import rationale


class ProvenanceTests(unittest.TestCase):
    def trajectory(self):
        context = attach_sources({"history_cutoff_exclusive": 10, "evidence_actor_id": 3,
            "observed_history": [{"session_id": 1, "time_index": 2,
                                 "clicked_items": [7], "purchased_items": []}]})
        key = next(k for k,v in context["facts"].items() if v["field"] == "clicked_items")
        value = rationale()
        value["evidence_refs"] = {stage: [key] for stage in ("stimulus", "knowledge", "evaluation")}
        return {"stage": "response", "response_style": "ekb", "rationale": value,
                "messages": [{"role": "user", "content": json.dumps(context)}]}

    def test_code_facts_and_visibility(self):
        t = self.trajectory()
        self.assertTrue(check_evidence(t)["accepted"])
        context = json.loads(t["messages"][0]["content"])
        self.assertNotIn("observed_history", context)
        bad = deepcopy(t)
        bad["rationale"]["evidence_refs"]["stimulus"] = ["H_nonexistent"]
        self.assertIn("unknown_fact_id", check_evidence(bad)["reasons"])
        context["history_cutoff_exclusive"] = 2
        t["messages"][0]["content"] = json.dumps(context)
        self.assertIn("non_prior_evidence", check_evidence(t)["reasons"])

    def test_actor_identity_and_semantic_review(self):
        t = self.trajectory()
        context = json.loads(t["messages"][0]["content"])
        context["evidence_actor_id"] = 4
        t["messages"][0]["content"] = json.dumps(context)
        self.assertIn("wrong_actor_evidence", check_evidence(t)["reasons"])
        t = self.trajectory()
        t["semantic_review"] = {"verdict": "fail", "reason": "unsupported price preference"}
        self.assertIn("semantic_review_failed", check_evidence(t)["reasons"])
        t["rationale"].pop("evidence_refs")
        self.assertEqual(check_evidence(t)["status"], "unverified")

    def test_sibling_scenes_cannot_cross_splits_even_across_domains(self):
        rows = [{"dataset": domain, "source_split": split, "trajectory": {
            "source_request_id": 42, "scene_id": scene, "actor_id": 1,
            "source_split": split, "dataset_manifest_sha256": "release"}}
            for domain, split, scene in (("one", "train", "42:A"), ("two", "test", "42:B"))]
        with self.assertRaisesRegex(ValueError, "crosses"):
            validate_request_splits(rows)
        rows[1]["source_split"] = rows[1]["trajectory"]["source_split"] = "train"
        validate_request_splits(rows)

    def test_single_item_union_not_just_per_head_limit(self):
        schema = single_item_schema([1, 2])
        for value in (action(), action([1]), action((), [2]), action([2], [2])):
            validate_arguments(value, schema)
        for value in (action([1, 2]), action([1], [2])):
            with self.assertRaises(ValueError):
                validate_arguments(value, schema)

    def test_single_item_probabilities_are_joint_and_include_no_action(self):
        from agent_simulation.evaluation.customer_contract import single_item_actions
        actions = single_item_actions([1, 2])
        values = probability_record({"actions": actions, "probabilities": [1.] * 7},
            [1, 2], action(), rng=np.random.default_rng(0))
        self.assertAlmostEqual(values["logloss"], np.log2(7))
        self.assertEqual(values["logloss_unit"], "bits_per_decision")
        peaked = probability_record({"actions": actions, "probabilities": [1.] + [0.] * 6},
            [1, 2], action(), rng=np.random.default_rng(0))
        self.assertTrue(np.isfinite(peaked["entropy"]))
        self.assertTrue(np.isfinite(peaked["logloss"]))
        self.assertLessEqual(len(set(values["sampled_action"]["clicked_item_ids"] +
                                     values["sampled_action"]["purchased_item_ids"])), 1)

    def test_joint_probability_selection_and_report_units(self):
        from agent_simulation.evaluation.customer_contract import single_item_actions
        from agent_simulation.evaluation.customer_curation import summarize
        target = action([1])
        rows = [record("good", "strong", target, target, None),
                record("bad", "weak", action(), target, None)]
        for row, probs in zip(rows, ([.01,.94,.01,.01,.01,.01,.01],
                                     [.94,.01,.01,.01,.01,.01,.01])):
            row.update(probability_record({"actions": single_item_actions([1, 2]),
                "probabilities": probs}, [1, 2], target, rng=np.random.default_rng(0)))
            row.update(mode="fixed", policy="sampled", response_style="ekb")
            row["trajectory"]["action_mode"] = "single_item"
        sft, pairs, metrics = select_preferences(rows)
        self.assertEqual((len(sft), len(pairs), len(metrics)), (1, 1, 1))
        self.assertEqual(pairs[0]["logloss_unit"], "bits_per_decision")
        report = summarize(rows, [])
        self.assertEqual(len(report["arms"]), 2)
        self.assertTrue(all(arm["source_behavior_group"] == "single_click/no_purchase"
                            for arm in report["arms"]))

    def test_dpo_matrix_and_original_transcript_preservation(self):
        target = action([1])
        good = record("good", "strong", target, target, [[.01,.97,.01,.01],[.97,.01,.01,.01]])
        bad = record("bad", "weak", action(), target, [[.9,.08,.01,.01],[.9,.08,.01,.01]])
        # Both actions wrong -> never pair, even if chosen would pass a soft F1 gate.
        both_wrong = deepcopy([good, bad])
        both_wrong[0]["trajectory"]["final_action"] = action([2])
        self.assertFalse(select_preferences(both_wrong)[1])
        # Correct behavior with missing evidence -> not a chosen branch.
        missing = deepcopy(good)
        missing["trajectory"]["rationale"].pop("evidence_refs")
        self.assertFalse(select_preferences([missing, bad])[1])
        self.assertFalse(select_sft([missing]))
        # Equal correct behavior + objectively broken source reference can pair.
        same = deepcopy(bad)
        same["trajectory"]["final_action"] = target
        same["trajectory"]["rationale"] = self.trajectory()["rationale"]
        # Actual messages contain no cited source.
        self.assertEqual(check_evidence(same["trajectory"])["status"], "failed")
        same["trajectory"]["messages"][-1]["content"] = json.dumps({
            "action": target, "rationale": same["trajectory"]["rationale"]})
        original = deepcopy(same["trajectory"]["messages"])
        pairs = select_preferences([good, same])[1]
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["pair_reason"], "evidence_quality")
        self.assertEqual(same["trajectory"]["messages"], original)
        # A lower NLL alone cannot make a structurally unverified EKB preferable.
        missing["logloss"] = 0.
        self.assertFalse(select_preferences([missing, bad])[1])


class StageContractTests(unittest.TestCase):
    def test_query_direct_and_frozen_response_ablation(self):
        import asyncio
        from agent_simulation.agents.agentic import AgenticOptions
        from agent_simulation.evaluation.customer_curation import CurationTools, generation_jobs, parser
        from agent_simulation.evaluation.trajectories import generate_trajectory, trajectory_messages
        from agent_simulation.tests.evaluation.test_customer_agentic import request
        from agent_simulation.tests.evaluation.test_trajectories import envelope
        original = request()
        bundle = {"query_enabled": True, "response_scenes": [
            {"request": deepcopy(original), "candidate": {"scene_id": "11:4"}},
            {"request": deepcopy(original), "candidate": {"scene_id": "11:5"}}]}
        jobs = generation_jobs(original, bundle=bundle)
        self.assertEqual([job[0] for job in jobs], ["query", "response", "response"])
        self.assertTrue(all(job[2]["response_input"]["response_action_mode"] == "single_item"
                            for job in jobs if job[0] == "response"))
        binding = CurationTools(original, "query", mode="fixed")
        schema, prompt = trajectory_messages(binding, role="customer", stage="query", theory=True)
        self.assertNotIn("SECRET TARGET", json.dumps(prompt))
        self.assertNotIn("evidence_refs", json.dumps(prompt))
        self.assertNotIn("rationale", json.dumps(schema))
        response = CurationTools(jobs[1][2], "response", mode="fixed")
        _, direct = trajectory_messages(response, role="customer", stage="response", theory=False)
        _, ekb = trajectory_messages(response, role="customer", stage="response", theory=True)
        self.assertEqual(direct[1], ekb[1])
        self.assertFalse(response.schemas)
        class Model:
            seed, model = 42, "offline-fixture"
            async def acomplete(self, value, *, call_id):
                validate_arguments(action(), value["final_schema"])
                return {"role": "assistant", "content": json.dumps(action()), "tool_calls": []}
        result = asyncio.run(generate_trajectory(response, Model(), role="customer", actor_id=7,
            stage="response", record_id="11:4", source_split="test", catalog_identity="release",
            options=AgenticOptions(theory=False), sample_index=0))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["rationale"], {})
        self.assertEqual(result["evidence_quality"]["status"], "not_applicable")
        self.assertEqual(result["action_mode"], "single_item")
        args = parser().parse_args(["--dataset", "unused", "--output", "unused",
            "--response-styles", "direct", "ekb", "--evaluation-only"])
        self.assertTrue(args.evaluation_only)
