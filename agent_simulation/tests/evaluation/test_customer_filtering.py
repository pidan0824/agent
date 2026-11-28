"""Offline fixtures only: no models, catalog, HTTP, GPU or experiments."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

from agent_simulation.evaluation.customer_filtering import (
    probability_record, select_preferences, export_training, select_sft)
from agent_simulation.evaluation.customer_quality import response_quality
from agent_simulation.evaluation.trajectories import OutcomeScorer, score_group
from agent_simulation.tests.evaluation.test_trajectories import branch


def action(click=(), buy=()):
    return {"clicked_item_ids": list(click), "purchased_item_ids": list(buy)}


def record(identity, variant, predicted, target, probabilities, *, cid="same", tid=0):
    t = branch(identity, predicted, identity + " actual evidence")
    score_group([t], target, OutcomeScorer())
    row = {"stage": "response", "dataset": "KuaiSearch/domain", "cid": cid,
        "tid": tid, "variant": variant, "source_split": "train", "trajectory": t,
        "reachable": True, "choice_cnt": 2, "behavior_group": "unused", "prob": None}
    if probabilities is not None:
        row.update(probability_record(probabilities, [1, 2], target, rng=np.random.default_rng(0)))
    return row


class FilteringTests(unittest.TestCase):
    def pair_rows(self):
        target = action([1, 2], [2])
        return [
            record("good", "strong", target, target, [[.01,.97,.01,.01],[.01,.01,.01,.97]]),
            record("bad", "weak", action(), target, [[.8,.1,.05,.05],[.8,.05,.05,.1]])]

    def test_uncertainty_reports_matched_sample_count_and_single_sample_degeneracy(self):
        rows = self.pair_rows()
        one = select_preferences(rows)[2][0]
        self.assertEqual(one["matched_sample_count"], 1)
        self.assertTrue(one["low_sample"])
        self.assertAlmostEqual(one["teacher_data_uncertainty"], 0.)
        self.assertAlmostEqual(one["student_data_uncertainty"], 0.)
        target = action([1, 2], [2])
        rows.append(record("second-strong", "strong", target, target,
                           [[.97,.01,.01,.01],[.97,.01,.01,.01]], tid=1))
        # Unpaired samples must not inflate the decomposition's sample count.
        unmatched = select_preferences(rows)[2][0]
        self.assertEqual(unmatched["matched_sample_count"], 1)
        rows.append(record("second-weak", "weak", action(), target,
                           [[.8,.1,.05,.05],[.8,.05,.05,.1]], tid=1))
        two = select_preferences(rows)[2][0]
        self.assertEqual(two["matched_sample_count"], 2)
        self.assertGreater(two["teacher_data_uncertainty"], 0.)
        self.assertAlmostEqual(two["student_data_uncertainty"], 0.)

    def test_strict_export_requires_transcript_bound_review_for_sft_and_dpo(self):
        from agent_simulation.evaluation.customer_filtering import apply_reviews
        from agent_simulation.evaluation.trajectories import digest
        rows = self.pair_rows()
        with TemporaryDirectory() as directory:
            report = export_training(rows, Path(directory), require_semantic_review=True)
            self.assertEqual(report["response_sft"], 0)
            self.assertEqual((Path(directory) / "preferences.jsonl").read_text(), "")
        good = rows[0]["trajectory"]
        review = {"trajectory_id": good["trajectory_id"], "messages_sha256": digest(good["messages"]),
                  "verdict": "pass", "reason": "Checked the original facts", "reviewer": "fixture reviewer"}
        apply_reviews(rows, [review])
        with TemporaryDirectory() as directory:
            report = export_training(rows, Path(directory), require_semantic_review=True)
            self.assertEqual(report["response_sft"], 1)
            self.assertEqual(len((Path(directory) / "preferences.jsonl").read_text().splitlines()), 1)
        stale = {**review, "messages_sha256": "changed"}
        with self.assertRaisesRegex(ValueError, "Review transcript"):
            apply_reviews(rows, [stale])
        with self.assertRaisesRegex(ValueError, "Duplicate review"):
            apply_reviews(rows, [review, review])
        apply_reviews(rows, [])  # Saved annotations cannot silently count as external review.
        self.assertNotIn("semantic_review", rows[0]["trajectory"])

    def test_export_checks_each_response_once_and_rechecks_next_export(self):
        from agent_simulation.evaluation import customer_filtering as module
        for sft_only in (True, False):
            rows = self.pair_rows()
            before = deepcopy(rows)
            with TemporaryDirectory() as directory:
                with patch.object(module, "check_evidence", wraps=module.check_evidence) as check:
                    report = export_training(rows, Path(directory), sft_only=sft_only)
                    self.assertEqual(check.call_count, len(rows))
                    self.assertEqual(report["response_sft"], 1)
                    export_training(rows, Path(directory), sft_only=sft_only)
                    self.assertEqual(check.call_count, 2 * len(rows))
            np.testing.assert_equal(rows, before)

    def test_standalone_sft_rechecks_changed_evidence_and_ignores_saved_pass(self):
        from agent_simulation.evaluation import customer_filtering as module
        row = self.pair_rows()[0]
        with patch.object(module, "check_evidence", wraps=module.check_evidence) as check:
            selected = select_sft([row])
            self.assertEqual(len(selected), 1)
            self.assertEqual(check.call_count, 1)
            row["trajectory"]["evidence_quality"] = deepcopy(selected[0]["evidence_quality"])
            row["trajectory"]["messages"][-1]["content"] = "{}"
            self.assertEqual(select_sft([row]), [])
            self.assertEqual(check.call_count, 2)

    def test_semantic_review_change_is_rechecked_on_next_selection(self):
        row = self.pair_rows()[0]
        self.assertEqual(len(select_sft([row])), 1)
        row["trajectory"]["semantic_review"] = {"verdict": "fail"}
        self.assertEqual(select_sft([row]), [])

    def test_logloss_preserves_original_transcripts_even_if_predictor_sample_disagrees(self):
        rows = self.pair_rows()
        rows[0]["sampled_action"] = action()
        rows[0]["preference"] = False
        before = deepcopy(rows)
        sft, pairs, _ = select_preferences(rows)
        self.assertEqual(len(pairs), 1)
        prefix = len(rows[0]["trajectory"]["prompt"])
        self.assertEqual(pairs[0]["chosen"], rows[0]["trajectory"]["messages"][prefix:])
        self.assertEqual(pairs[0]["rejected"], rows[1]["trajectory"]["messages"][prefix:])
        self.assertEqual(json.loads(pairs[0]["chosen"][-1]["content"])["action"], action([1,2],[2]))
        self.assertEqual(sft[0]["messages"], rows[0]["trajectory"]["messages"])
        self.assertEqual(rows[0]["trajectory"], before[0]["trajectory"])

    def test_logloss_can_choose_weak_when_both_signals_favor_it(self):
        rows = self.pair_rows()
        rows[0]["variant"], rows[1]["variant"] = "weak", "strong"
        _, pairs, _ = select_preferences(rows)
        self.assertEqual(pairs[0]["chosen_variant"], "weak")

    def test_logloss_conflict_and_small_gap_do_not_create_pairs(self):
        rows = self.pair_rows()
        rows[0]["logloss"], rows[1]["logloss"] = 2., 1.
        report = []
        self.assertFalse(select_preferences(rows, diagnostics=report)[1])
        self.assertEqual(report[0]["reason"], "logloss_gap_or_direction_conflict")
        rows[0]["logloss"], rows[1]["logloss"] = 1., 1.01
        self.assertFalse(select_preferences(rows, logloss_min_gap=.05)[1])

    def test_equal_actions_do_not_become_explanation_only_agent_pairs(self):
        rows = self.pair_rows()
        rows[1] = record("same-action", "weak", action([1,2],[2]), action([1,2],[2]),
                         [[.8,.1,.05,.05],[.8,.05,.05,.1]])
        self.assertFalse(select_preferences(rows)[1])

    def test_sampled_protocol_checks_original_action_and_never_patches_it(self):
        rows = self.pair_rows()
        rows[0]["preference"], rows[1]["preference"] = True, False
        pairs = select_preferences(rows, pair_protocol="sampled_match")[1]
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["chosen"][-1], rows[0]["trajectory"]["messages"][-1])
        rows[0]["trajectory"]["final_action"] = action()
        self.assertFalse(select_preferences(rows, pair_protocol="sampled_match")[1])

    def test_separate_gates_reject_average_score_pass_and_false_empty_head_actions(self):
        # click F1 2/3, purchase F1 1 -> old mean 5/6 passed .8.
        q = response_quality(action([1]), action([1,2]))
        self.assertGreater(q["metrics"]["score"], .8)
        self.assertFalse(q["accepted"])
        self.assertIn("click_below_threshold", q["reasons"])
        self.assertTrue(response_quality(action(), action())["accepted"])
        self.assertTrue(response_quality(action((),[2]), action((),[2]))["accepted"])
        q = response_quality(action([1],[2]), action([1]))
        self.assertIn("false_purchase", q["reasons"])
        q = response_quality(action(), action((),[2]))
        self.assertIn("purchase_below_threshold", q["reasons"])

    def test_sft_independent_of_probabilities_and_preserves_multi_and_empty(self):
        rows = [record("multi", "strong", action([1,2],[2]), action([1,2],[2]), None, cid="multi"),
                record("empty", "strong", action(), action(), None, cid="empty")]
        self.assertEqual(len(select_sft(rows)), 2)
        self.assertEqual(select_preferences(rows)[1], [])
        for row in rows:
            row["source_split"] = row["trajectory"]["source_split"] = "test"
        self.assertFalse(select_sft(rows))

    def test_offline_export_counts_strata_drop_reasons_and_query(self):
        rows = self.pair_rows()
        rows.extend([
            record("empty", "strong", action(), action(), None, cid="empty"),
            record("partial", "strong", action([1]), action([1,2]), None, cid="partial"),
            record("false-buy", "strong", action([1],[2]), action([1]), None, cid="false-buy")])
        qrows = []
        for name, score in (("qgood", 1.), ("qbad", .2)):
            t = branch(name, {"query": name}, name)
            t.update(stage="query", score=score, accepted=score >= .7, target={"query":"qgood"})
            qrows.append({"stage":"query","dataset":"KuaiSearch/Query","cid":"query",
                "variant":"strong","source_split":"train","trajectory":t,"prob":None})
        rows.extend(qrows)
        with TemporaryDirectory() as directory:
            report = export_training(rows, Path(directory))
            self.assertEqual((report["query_sft"],report["response_sft"]), (1,2))
            self.assertEqual((report["query_pairs"],report["response_pairs"]), (1,1))
            self.assertEqual(report["response_probability_missing"], 3)
            self.assertIn("click_below_threshold",report["sft_exclusion_reasons"])
            self.assertIn("false_purchase",report["sft_exclusion_reasons"])
            self.assertIn("response/missing_probabilities",report["pair_reasons"])
            strata = {s["stratum"]:s for s in report["strata"] if s["variant"]=="strong"}
            self.assertEqual(strata["no_click/no_purchase"]["sft_retained"],1)
            self.assertEqual(strata["multi_click/no_purchase"]["sft_retained"],0)
            self.assertEqual(len((Path(directory)/"filter-decisions.jsonl").read_text().splitlines()),7)
            self.assertTrue((Path(directory)/"pair-decisions.jsonl").exists())

    def test_replay_thresholds_and_test_split_are_applied_to_both_exports(self):
        rows = [record("partial","strong",action([1]),action([1,2]),None)]
        with TemporaryDirectory() as directory:
            report = export_training(rows, Path(directory), click_min=.6)
            self.assertEqual(report["sft_total"],1)
        rows = self.pair_rows()
        for row in rows:
            row["source_split"] = row["trajectory"]["source_split"] = "test"
        with TemporaryDirectory() as directory:
            report = export_training(rows, Path(directory))
            self.assertEqual((report["sft_total"],report["pairs_total"]),(0,0))

    def test_pair_budget_and_group_mismatch(self):
        rows = self.pair_rows()
        more = deepcopy(rows)
        for row in more:
            row["cid"] = "other"
            row["trajectory"]["trajectory_id"] += "other"
        self.assertEqual(len(select_preferences(rows+more,max_pairs=1)[1]),1)
        rows[1]["trajectory"]["prompt"][0]["content"] = "different input"
        with self.assertRaisesRegex(ValueError,"contexts"):
            select_preferences(rows)

    def test_missing_rank_is_excluded_without_zero_fallback_and_legacy_is_available(self):
        rows = self.pair_rows()
        missing = deepcopy(rows)
        for row in missing:
            row["cid"] = "missing"
            row["variant"] = "strong"  # No matched strong/weak metric.
            row["trajectory"]["trajectory_id"] += "-missing"
        diagnostics, ranking = [], {}
        sft, pairs, metrics = select_preferences(rows + missing, max_pairs=10000,
            diagnostics=diagnostics, selection_report=ranking)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(next(d["reason"] for d in diagnostics if d["cid"] == "missing"),
                         "missing_uncertainty_metric")
        self.assertEqual((ranking["candidate_pairs"], ranking["rankable_pairs"],
                          ranking["missing_metric_pairs"], ranking["selected_pairs"]), (2, 1, 1, 1))
        self.assertFalse(ranking["sorting_truncated"])
        self.assertIn("不确定性排序未起作用", ranking["sorting_note"])
        old_sft, old_pairs, old_metrics = select_preferences(rows + missing,
                                                           selection_protocol="legacy-v1")
        self.assertEqual(sft, old_sft)
        self.assertEqual(metrics, old_metrics)
        self.assertEqual(len(old_pairs), 2)
        self.assertTrue(any(not p["selection_metric_available"] for p in old_pairs))

    def test_ranked_selection_reports_actual_truncation_with_signed_metrics(self):
        rows = []
        for cid in ("negative", "positive", "missing"):
            group = deepcopy(self.pair_rows())
            for row in group:
                row["cid"] = cid
                row["trajectory"]["trajectory_id"] += "-" + cid
            rows.extend(group)
        metrics = [
            {"dataset": rows[0]["dataset"], "cid": cid,
             "diff_model_uncertainty": rank, "low_sample": False}
            for cid, rank in (("negative", -.2), ("positive", .3))]
        with patch("agent_simulation.evaluation.customer_filtering._uncertainty_metrics", return_value=metrics):
            report, decisions = {}, []
            _, pairs, _ = select_preferences(rows, max_pairs=1, selection_report=report,
                                              diagnostics=decisions)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["selection_value"], .3)
            self.assertTrue(report["sorting_truncated"])
            self.assertEqual((report["candidate_pairs"], report["rankable_pairs"],
                              report["selected_pairs"]), (3, 2, 1))
            report = {}
            _, pairs, _ = select_preferences(rows, max_pairs=10000, selection_report=report)
            self.assertEqual({p["selection_value"] for p in pairs}, {-.2, .3})
            self.assertFalse(report["sorting_truncated"])

    def test_random_selection_does_not_require_uncertainty_metric(self):
        rows = self.pair_rows()
        for row in rows:
            row["variant"] = "strong"
        report = {}
        _, pairs, _ = select_preferences(rows, filter_column="random", selection_report=report)
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0]["selection_metric_available"])
        self.assertFalse(report["uncertainty_sorting_affected_selection"])
        self.assertIn("no uncertainty ranking", report["sorting_note"])
        with TemporaryDirectory() as directory:
            report = export_training(rows, Path(directory), sft_only=True)
            self.assertEqual(report["preference_selection"]["status"], "not_run_sft_only")

    def test_three_matched_samples_clear_low_sample_flag(self):
        rows = []
        for tid in range(3):
            group = deepcopy(self.pair_rows())
            for row in group:
                row["tid"] = tid
                row["trajectory"]["trajectory_id"] += "-" + str(tid)
            rows.extend(group)
        metrics = select_preferences(rows)[2]
        self.assertEqual(metrics[0]["matched_sample_count"], 3)
        self.assertFalse(metrics[0]["low_sample"])
        with TemporaryDirectory() as directory:
            report = export_training(rows, Path(directory))
            self.assertEqual(report["uncertainty_low_sample_scenes"], 0)
            self.assertEqual(report["preference_selection"]["selected_pairs"], 1)
            self.assertFalse(report["preference_selection"]["sorting_truncated"])

    def test_logloss_is_mean_bits_and_probability_input_is_checked(self):
        r = probability_record([[.25]*4]*2,[1,2],action([1],[2]),rng=np.random.default_rng(0))
        self.assertAlmostEqual(r["logloss"],2.)
        self.assertEqual(r["logloss_unit"],"bits_per_item")
        for values in ([[0]*4]*2, [[-1,1,1,1]]*2, [[float("nan"),1,1,1]]*2):
            with self.assertRaises(ValueError):
                probability_record(values,[1,2],action(),rng=np.random.default_rng(0))

    def test_duplicate_rows_count_once_and_bad_configuration_fails_without_models(self):
        from agent_simulation.evaluation.customer_filtering import validate_filter_options
        row = record("empty", "strong", action(), action(), None)
        with TemporaryDirectory() as directory:
            report = export_training([row, deepcopy(row)], Path(directory))
            self.assertEqual(report["sft_total"], 1)
            self.assertEqual(report["strata"][0]["sft_retained"], 1)
            self.assertEqual(report["sft_exclusion_reasons"]["duplicate_transcript"], 1)
        for values in ({"click_min": 2}, {"purchase_min": float("nan")},
                       {"action_min_gap": 0}, {"logloss_min_gap": -1}):
            with self.assertRaises(ValueError):
                validate_filter_options(**values)

    def test_split_provenance_survives_sft_and_dpo_without_entering_messages(self):
        rows=self.pair_rows()
        original=action([1,2,99],[2])
        for row in rows:
            row["trajectory"].update(source_request_id=100,scene_id="100:positive:2",
                                     source_response_target=original)
        sft,pairs,_=select_preferences(rows)
        for item in (sft[0],pairs[0]):
            self.assertEqual(item["source_request_id"],100)
            self.assertEqual(item["source_response_target"],original)
        self.assertNotIn("source_response_target",json.dumps(sft[0]["messages"]))
        self.assertNotIn("source_response_target",json.dumps(pairs[0]["chosen"]))
