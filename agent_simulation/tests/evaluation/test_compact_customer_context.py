"""Lossless context compression must not weaken evidence or action checks."""
from copy import deepcopy
import json
import unittest

from agent_simulation.evaluation.customer_evidence import (
    attach_sources, compact_sources, expand_sources, encode_context, check_evidence)
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
        self.assertLess(len(encode_context(original)), len(json.dumps(original, ensure_ascii=False)))
        self.assertEqual(check_evidence(self.trajectory(original)),
                         check_evidence(self.trajectory(wire)))

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
