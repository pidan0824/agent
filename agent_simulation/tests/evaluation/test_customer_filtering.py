"""Offline fixtures only: no models, catalog, HTTP, GPU or experiments."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
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
            row["source_split"] = "test"
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
            row["source_split"] = "test"
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
