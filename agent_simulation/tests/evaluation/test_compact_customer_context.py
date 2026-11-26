"""Lossless context compression must not weaken evidence or action checks."""
from copy import deepcopy
import json
import unittest

from agent_simulation.evaluation.customer_evidence import (
    attach_sources, compact_sources, expand_sources, encode_context, check_evidence,
    FACT_REFS_ENCODING, FACT_REFS_NOTE)
from agent_simulation.tests.ekb_fixture import rationale


class CompactContextTests(unittest.TestCase):
    def context(self):
        return attach_sources({"history_cutoff_exclusive": 10, "evidence_actor_id": 7,
            "observed_history": [{"session_id": 1, "time_index": 2, "user_id": 7,
                "query": "水杯", "clicked_items": [{"item_id": 8, "price_raw": 30}], "purchased_items": []}],
            "query": "水杯", "displayed_items": [
                {"item_id": 8, "item_title": "水杯", "brand_name": "品牌", "category_level3_name": "杯子",
                 "price_raw": 30, "price_observed_time_index": 3},
                {"item_id": 9, "item_title": "未知价格杯", "price_raw": None}]})

    def trajectory(self, context):
        r = rationale()
        history = next(k for k,v in self.context()["facts"].items() if v["kind"] == "history")
        r["evidence_refs"] = {stage: [history] for stage in ("stimulus","knowledge","evaluation")}
        action = {"clicked_item_ids": [8], "purchased_item_ids": []}
        return {"stage": "response", "response_style": "ekb", "rationale": r, "final_action": action,
                "messages": [{"role": "user", "content": json.dumps(context)}, {"role": "assistant",
                    "content": json.dumps({"rationale": r, "action": action})}]}

    def test_roundtrip_keeps_all_values_ids_times_and_unknown_price(self):
        original = self.context()
        before = deepcopy(original)
        wire = json.loads(encode_context(original))
        self.assertEqual(expand_sources(wire), original)
        self.assertEqual(original, before)
        self.assertEqual(set(wire["facts"]), set(original["facts"]))
        self.assertEqual(wire["fact_refs_encoding"], FACT_REFS_ENCODING)
        for name in ("history_events", "candidate_context"):
            self.assertEqual(wire[name], [{k: v for k, v in row.items() if k != "fact_refs"}
                                          for row in original[name]])
        self.assertLess(len(encode_context(original)), len(json.dumps(original, ensure_ascii=False)))
        self.assertEqual(check_evidence(self.trajectory(original)),
                         check_evidence(self.trajectory(wire)))

    def test_sparse_context_keeps_references_when_encoding_overhead_is_larger(self):
        context = attach_sources({"displayed_items": [{"item_id": 9, "item_title": "水杯"}]})
        wire = compact_sources(context)
        self.assertNotIn("fact_refs_encoding", wire)
        self.assertNotIn("fact_refs_note", wire)
        self.assertEqual(wire["candidate_context"], context["candidate_context"])
        self.assertEqual(expand_sources(wire), context)

    def test_legacy_raw_and_compact_contexts_remain_readable(self):
        empty = attach_sources({"query": "水杯"})
        self.assertNotIn("fact_refs_encoding", compact_sources(empty))
        self.assertEqual(expand_sources(compact_sources(empty)), empty)
        original = self.context()
        self.assertEqual(expand_sources(original), original)
        legacy = compact_sources(original)
        legacy.pop("fact_refs_encoding")
        legacy.pop("fact_refs_note")
        for name in ("history_events", "candidate_context"):
            legacy[name] = deepcopy(original[name])
        self.assertEqual(expand_sources(legacy), original)
        self.assertEqual(check_evidence(self.trajectory(legacy)), check_evidence(self.trajectory(original)))

    def test_repeated_items_keep_event_identity_action_order_and_changed_prices(self):
        first = {"item_id": 8, "item_title": "水杯", "price_raw": 30, "unknown": None}
        second = {**first, "price_raw": 45}
        context = attach_sources({"history_cutoff_exclusive": 10, "evidence_actor_id": 7,
            "observed_history": [
                {"session_id": 1, "time_index": 2, "user_id": 7, "query": "水杯",
                 "clicked_items": [first, first], "purchased_items": []},
                {"session_id": 2, "time_index": 4, "user_id": 7, "query": "水杯",
                 "clicked_items": [], "purchased_items": [second]}],
            "displayed_items": [{"item_id": 9, "item_title": "水杯"},
                                {"item_id": 8, "item_title": "水杯"}]})
        context["candidate_context"][0]["extra"] = "preserved"
        wire = compact_sources(context)
        self.assertEqual(expand_sources(wire), context)
        self.assertEqual([p["item_id"] for p in wire["candidate_context"]], [9, 8])
        self.assertEqual(wire["candidate_context"][0]["extra"], "preserved")

    def test_nonstandard_or_ambiguous_indexes_keep_legacy_references(self):
        for change in ("wrong_ref", "missing_refs", "missing_row", "duplicate_row", "ambiguous_fact"):
            with self.subTest(change=change):
                context = self.context()
                if change == "wrong_ref":
                    context["history_events"][0]["fact_refs"]["query"] = "unrelated"
                elif change == "missing_refs":
                    context["candidate_context"][0].pop("fact_refs")
                elif change == "missing_row":
                    context["candidate_context"].pop()
                elif change == "duplicate_row":
                    context["history_events"].append(deepcopy(context["history_events"][0]))
                else:
                    fact = next(v for v in context["facts"].values() if v["kind"] == "history")
                    context["facts"]["duplicate_fact"] = deepcopy(fact)
                wire = compact_sources(context)
                self.assertNotIn("fact_refs_encoding", wire)
                self.assertNotIn("fact_refs_note", wire)
                self.assertEqual(wire["history_events"], context["history_events"])
                self.assertEqual(wire["candidate_context"], context["candidate_context"])
                self.assertEqual(expand_sources(wire), context)

    def test_product_identity_check_survives_derived_references(self):
        context = self.context()
        key = next(k for k, v in context["facts"].items() if v["kind"] == "product")
        context["facts"][key]["source_id"] = "item:wrong"
        trajectory = self.trajectory(compact_sources(context))
        trajectory["rationale"]["evidence_refs"] = {stage: [key] for stage in ("stimulus", "knowledge", "evaluation")}
        trajectory["messages"][-1]["content"] = json.dumps({
            "rationale": trajectory["rationale"], "action": trajectory["final_action"]})
        self.assertIn("wrong_item_evidence", check_evidence(trajectory)["reasons"])

    def test_invalid_derived_markers_and_indexes_are_rejected(self):
        for corrupt in ("version", "note", "orphan_note", "no_layout", "explicit_refs", "unknown_item",
                        "missing_history", "duplicate_row", "ambiguous_fact"):
            with self.subTest(corrupt=corrupt):
                wire = compact_sources(self.context())
                if corrupt == "version":
                    wire["fact_refs_encoding"] = "future-version"
                elif corrupt == "note":
                    wire.pop("fact_refs_note")
                elif corrupt == "orphan_note":
                    wire.pop("fact_refs_encoding")
                elif corrupt == "no_layout":
                    wire.pop("fact_layout")
                elif corrupt == "explicit_refs":
                    wire["candidate_context"][0]["fact_refs"] = {}
                elif corrupt == "unknown_item":
                    wire["candidate_context"][0]["item_id"] = 999
                elif corrupt == "missing_history":
                    wire.pop("history_events")
                elif corrupt == "duplicate_row":
                    wire["history_events"].append(deepcopy(wire["history_events"][0]))
                else:
                    key = next(k for k, row in wire["facts"].items() if row[0].startswith("history:"))
                    wire["facts"]["duplicate_fact"] = deepcopy(wire["facts"][key])
                with self.assertRaises(ValueError):
                    expand_sources(wire)
                self.assertIn("invalid_fact_encoding", check_evidence(self.trajectory(wire))["reasons"])
        for field, value in (("fact_refs_encoding", FACT_REFS_ENCODING), ("fact_refs_note", FACT_REFS_NOTE)):
            context = self.context()
            context[field] = value
            with self.assertRaises(ValueError):
                compact_sources(context)

    def test_encoded_time_owner_and_price_checks_still_apply(self):
        for mutate, reason in (
            (lambda c: c.update(history_cutoff_exclusive=2), "non_prior_evidence"),
            (lambda c: c.update(evidence_actor_id=8), "wrong_actor_evidence"),
        ):
            context = self.context()
            mutate(context)
            self.assertIn(reason, check_evidence(self.trajectory(compact_sources(context)))["reasons"])
        trajectory = self.trajectory(compact_sources(self.context()))
        trajectory["rationale"]["knowledge"] = "Item 8 has a price of 999."
        trajectory["messages"][-1]["content"] = json.dumps({
            "rationale": trajectory["rationale"], "action": trajectory["final_action"]})
        self.assertIn("item_price_value_mismatch", check_evidence(trajectory)["reasons"])

    def test_malformed_or_conflicting_wire_metadata_is_rejected(self):
        for corrupt in ("layout", "source", "override"):
            context = compact_sources(self.context())
            key = next(iter(context["facts"]))
            if corrupt == "layout":
                context["fact_layout"] = ["unknown"]
            elif corrupt == "source":
                context["facts"][key][0] = "missing"
            else:
                context["facts"][key] = context["facts"][key][:3] + [{"kind": "product"}]
            with self.assertRaises(ValueError):
                expand_sources(context)
            self.assertIn("invalid_fact_encoding", check_evidence(self.trajectory(context))["reasons"])


if __name__ == "__main__":
    unittest.main()
