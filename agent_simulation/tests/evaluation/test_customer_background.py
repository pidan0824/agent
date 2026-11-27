"""Restore prior item evidence without labels, double counting or broken citations."""
from copy import deepcopy
import json
import unittest

from agent_simulation.evaluation.customer_context import CurationTools, fixed_response_requests
from agent_simulation.evaluation.customer_evidence import encode_context, expand_sources, check_evidence
from agent_simulation.evaluation.customer_run import FrozenPredictionBinding
from agent_simulation.evaluation.trajectories import digest
from agent_simulation.tests.evaluation.test_customer_agentic import request
from agent_simulation.tests.evaluation.test_context_optimization import CharacterTokenizer
from agent_simulation.tests.ekb_fixture import rationale


def source():
    value = request()
    ctx = value["response_input"]
    ctx["displayed_items"] = [
        {"item_id": 4, "item_title": "真皮女士休闲板鞋", "brand_name": "品牌甲",
         "seller_id": 12, "seller_name": "鞋店", "category_level2_name": "鞋"},
        {"item_id": 5, "item_title": "帆布女士休闲板鞋", "brand_name": "品牌乙",
         "seller_id": 13, "seller_name": "另一家", "category_level2_name": "鞋"}]
    ctx["behavioral_summary"] = {"observed_time_index": 3, "statistics_30d": {"orders": 1},
        "recent_clicked_item_ids": [4, 4, 9], "recent_purchased_item_ids": [9],
        "item_details": [{"item_id": 4, "item_title": "旧款真皮板鞋", "brand_name": "品牌甲"},
                         {"item_id": 9, "item_title": "帆布低帮女士休闲板鞋", "brand_name": "品牌乙",
                          "seller_id": 13, "seller_name": "另一家", "category_level2_name": "鞋"}]}
    return value


class BackgroundContextTests(unittest.TestCase):
    def context(self, value=None, budget=None):
        return CurationTools(value or source(), "response", mode="fixed",
            history_token_budget=budget, tokenizer=CharacterTokenizer()).initial_context()

    def trajectory(self, ctx, ref):
        r = rationale()
        r["evidence_refs"] = {s: [ref] for s in ("stimulus", "knowledge", "evaluation")}
        action = {"clicked_item_ids": [4], "purchased_item_ids": []}
        return {"stage": "response", "response_style": "ekb", "rationale": r, "final_action": action,
            "messages": [{"role": "user", "content": encode_context(ctx)},
                         {"role": "assistant", "content": json.dumps({"rationale": r, "action": action})}]}

    def test_complete_sequences_details_and_source_survive_roundtrip(self):
        value = source()
        before = deepcopy(value)
        ctx = self.context(value)
        self.assertEqual(expand_sources(json.loads(encode_context(ctx))), ctx)
        self.assertEqual(value, before)
        facts = [f for f in ctx["facts"].values() if f["kind"] == "background"]
        lists = next(f for f in facts if f["field"] == "recent_item_ids")
        self.assertEqual(lists["value"]["recent_clicked_item_ids"], [4, 4, 9])
        self.assertEqual(lists["value"]["recent_purchased_item_ids"], [9])
        self.assertTrue(all(f["actor_id"] == 7 and f["time_index"] == 3 for f in facts))
        rows = {r["item_id"]: r for r in ctx["candidate_context"]}
        self.assertEqual(rows[4]["snapshot_interactions"]["click_occurrences"], 2)
        self.assertEqual(rows[5]["snapshot_interactions"]["purchase_occurrences"], 0)
        self.assertEqual(rows[5]["attributes"]["seller_name"], "另一家")

    def test_new_id_candidate_is_linked_by_attributes_not_frequency(self):
        ctx = self.context()
        row = next(r for r in ctx["candidate_context"] if r["item_id"] == 5)
        link = next(r for r in row["related_history"] if r["item_id"] == 9)
        self.assertIn("brand_name", link["matched_attributes"])
        self.assertIn("seller_id", link["matched_attributes"])
        self.assertIn("item_title", link["matched_attributes"])
        self.assertEqual(ctx["facts"][link["fact_ref"]]["value"]["purchase_occurrences"], 1)

    def test_unknown_brand_and_shop_do_not_create_preference_matches(self):
        from agent_simulation.evaluation.customer_background import product_match
        for brand in ("UNKNOWN", None, "无品牌", "其他/other"):
            fields, _ = product_match({"item_id": 1, "brand_name": brand, "seller_id": 0},
                                      {"item_id": 2, "brand_name": brand, "seller_id": 0})
            self.assertEqual(fields, [])

    def test_missing_list_is_unknown_not_zero(self):
        value = source()
        del value["response_input"]["behavioral_summary"]["recent_purchased_item_ids"]
        ctx = self.context(value)
        self.assertTrue(all("purchase_occurrences" not in row["snapshot_interactions"]
                            for row in ctx["candidate_context"]))
        for fact in ctx["facts"].values():
            if fact["kind"] == "background" and fact["field"] == "item_history":
                self.assertNotIn("purchase_occurrences", fact["value"])

    def test_shared_budget_preserves_lists_and_limits_details_without_mutation(self):
        value = source()
        value["response_input"]["behavioral_summary"]["item_details"][1]["item_title"] *= 1000
        empty = self.context(value, 0)
        selected = self.context(value, 3000)
        self.assertLessEqual(len(encode_context(selected)) - len(encode_context(empty)), 3000)
        self.assertEqual(empty["snapshot_detail_selection"]["shown"], 0)
        self.assertLess(selected["snapshot_detail_selection"]["shown"], 2)
        self.assertEqual({r["item_id"] for r in selected["candidate_context"]}, {4, 5})
        self.assertEqual(next(f["value"] for f in empty["facts"].values() if f["field"] == "recent_item_ids"),
                         next(f["value"] for f in selected["facts"].values() if f["field"] == "recent_item_ids"))

    def test_direct_cot_and_frozen_probability_context_are_identical_and_label_free(self):
        value = source()
        ctx, requests = fixed_response_requests(value["response_input"], history_token_budget=4000,
                                               tokenizer=CharacterTokenizer())
        def visible(messages):
            return [m for m in messages if m["role"] == "user" and m["content"].startswith("{")]
        self.assertEqual(visible(requests["direct"]["messages"]), visible(requests["ekb"]["messages"]))
        row = {"probability_context": ctx, "probability_context_sha256": digest(ctx),
               "trajectory": {"prompt": requests["ekb"]["messages"],
                              "candidate_item_ids": [4, 5], "action_mode": "single_item"}}
        self.assertEqual(FrozenPredictionBinding(row).initial_context(), ctx)
        changed = deepcopy(value)
        changed["response_target"] = {"clicked_item_ids": [5], "purchased_item_ids": [5]}
        self.assertEqual(self.context(changed), self.context(value))

    def test_snapshot_citations_enforce_owner_cutoff_source_and_item(self):
        ctx = self.context()
        ref = next(k for k, f in ctx["facts"].items() if f["field"] == "item_history")
        self.assertTrue(check_evidence(self.trajectory(ctx, ref))["accepted"])
        for field, val, reason in [
            ("actor_id", 999, "wrong_actor_evidence"),
            ("time_index", 10, "non_prior_evidence"),
            ("time_index", None, "non_prior_evidence"),
            ("source_id", "background:wrong", "wrong_background_source"),
            ("item_id", 999, "wrong_item_evidence")]:
            with self.subTest(field=field, val=val):
                bad = deepcopy(ctx)
                bad["facts"][ref][field] = val
                self.assertIn(reason, check_evidence(self.trajectory(bad, ref))["reasons"])

    def test_bad_snapshot_rejected_before_prompt_construction(self):
        for field, val in [("observed_time_index", 10), ("user_id", 999),
                           ("recent_clicked_item_ids", [True])]:
            value = source()
            value["response_input"]["behavioral_summary"][field] = val
            with self.assertRaises(ValueError):
                self.context(value)

    def test_candidate_wire_moves_values_once_and_rejects_bad_references(self):
        ctx = self.context()
        wire = json.loads(encode_context(ctx))
        self.assertEqual(wire["candidate_attributes_encoding"], "attribute-facts-v2")
        key = ctx["candidate_context"][0]["fact_refs"]["item_title"]
        self.assertNotIn(key, wire["facts"])
        self.assertEqual(wire["candidate_context"][0]["attributes"][key][:2],
                         ["item_title", ctx["facts"][key]["value"]])
        for change in ("version", "width", "duplicate", "source", "attributes", "metadata"):
            bad = deepcopy(wire)
            if change == "version":
                bad["candidate_attributes_encoding"] = "unknown"
            elif change == "width":
                bad["candidate_context"][0]["attributes"][key] = ["item_title"]
            elif change == "duplicate":
                bad["candidate_context"][1]["attributes"][key] = ["duplicate", "value"]
            elif change == "attributes":
                bad["candidate_context"][0]["attributes"] = []
            elif change == "metadata":
                bad["fact_sources"]["item:4"] = []
            else:
                bad["fact_sources"]["item:4"]["item_id"] = 999
            with self.subTest(change=change), self.assertRaises(ValueError):
                expand_sources(bad)

    def test_actual_probability_requests_reuse_the_same_encoded_context(self):
        from unittest.mock import Mock, patch
        from types import SimpleNamespace
        from agent_simulation.evaluation.customer_filtering import OptionPredictor
        ctx = self.context()
        client = SimpleNamespace(tokenizer=CharacterTokenizer(), model="fixture",
                                 http=Mock())
        client.http.post.side_effect = RuntimeError("captured")
        predictor = OptionPredictor(client)
        for mode in ("single_item", "full"):
            binding = SimpleNamespace(context={"response_action_mode": mode},
                displayed=[{"item_id": 4}, {"item_id": 5}], initial_context=lambda: deepcopy(ctx))
            with patch("agent_simulation.evaluation.customer_filtering.prompt_tokens", return_value=[1, 2]) as capture:
                with self.assertRaisesRegex(RuntimeError, "captured"):
                    predictor.score(binding, {"rationale": {}, "trace": {"tools": []}})
                messages = capture.call_args.args[0]
                evidence = json.loads(messages[1]["content"])
                self.assertEqual(expand_sources(evidence["context"]), ctx)
                self.assertNotIn("target", evidence)
                self.assertNotIn("final_action", evidence)

    def test_scene_evidence_uses_restored_lists_even_without_detail_budget(self):
        from agent_simulation.evaluation.customer_generation import scene_evidence
        value = source()
        value["response_input"]["short_term_memory"]["search_history"] = []
        binding = CurationTools(value, "response", mode="fixed", history_token_budget=0)
        self.assertTrue(scene_evidence(binding)["same_item_history"])

    def test_single_item_scoring_uses_one_token_labels_with_eight_candidates(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from agent_simulation.evaluation.customer_filtering import OptionPredictor
        client = SimpleNamespace(tokenizer=CharacterTokenizer(), model="fixture", http=Mock())
        client.http.post.side_effect = RuntimeError("captured")
        ctx = self.context()
        binding = SimpleNamespace(context={"response_action_mode": "single_item"},
            displayed=[{"item_id": i} for i in range(8)], initial_context=lambda: ctx)
        with patch("agent_simulation.evaluation.customer_filtering.prompt_tokens", return_value=[1]) as capture:
            with self.assertRaisesRegex(RuntimeError, "captured"):
                OptionPredictor(client).score(binding, {"rationale": {}, "trace": {"tools": []}})
            evidence = json.loads(capture.call_args.args[0][1]["content"])
        labels = list(evidence["options"])
        self.assertEqual(len(labels), 25)
        self.assertEqual(labels[:10], list("0123456789"))
        self.assertTrue(all(len(client.tokenizer.encode(label)) == 1 for label in labels))
        allowed = client.http.post.call_args.kwargs["json"]["allowed_token_ids"]
        self.assertEqual(len(set(allowed)), 25)
        self.assertEqual(evidence["options"][labels[-1]],
                         {"clicked_item_ids": [7], "purchased_item_ids": [7]})

    def test_first_candidate_wire_version_remains_readable(self):
        ctx = self.context()
        wire = json.loads(encode_context(ctx))
        wire["candidate_attributes_encoding"] = "attribute-facts-v1"
        for row in wire["candidate_context"]:
            row["attributes"] = {entry[0]: [ref, *entry[1:]]
                                 for ref, entry in row["attributes"].items()}
        self.assertEqual(expand_sources(wire), ctx)
