from copy import deepcopy
import unittest

from agent_simulation.agents.schemas import validate_arguments
from agent_simulation.evaluation.customer_ekb_ablation import (
    frozen_context, generation_request, select_blind, evaluate_selected,
)

def action(click=(), purchase=()):
    return {"clicked_item_ids": list(click), "purchased_item_ids": list(purchase)}

def attempt(index, value=None):
    return {"sample_index": index, "status": "ok" if value is not None else "invalid", "action": value}

def context_input():
    return {"query": "真实 query", "history_cutoff_exclusive": 10,
            "short_term_memory": {"search_history": [
                {"session_id": 1, "time_index": 2, "query": "旧 query",
                 "clicked_items": [{"item_id": 9, "item_title": "旧商品", "price_raw": 5}],
                 "purchased_items": []}]},
            "displayed_items": [{"item_id": 3, "item_title": "商品3"},
                                {"item_id": 2, "item_title": "商品2"}]}

class AblationTests(unittest.TestCase):
    def test_labels_and_scene_metadata_never_enter_context(self):
        source = context_input()
        expected = frozen_context(source, 10)
        source.update(response_target=action([2]), target_item_id=2, positive_position=0)
        source["displayed_items"][0]["clicked"] = True
        assert frozen_context(source, 10) == expected
        a = generation_request(expected, "direct")
        b = generation_request(expected, "ekb")
        assert a["messages"][1] == b["messages"][1]
        assert [p["item_id"] for p in expected["candidates"]] == [2, 3]
        assert expected["history"][0]["purchased_items"] == []

    def test_future_history_rejected(self):
        source = context_input()
        source["short_term_memory"]["search_history"][0]["time_index"] = 10
        with self.assertRaises(ValueError):
            frozen_context(source, 10)

    def test_plurality_uses_joint_sets_not_headwise_votes_or_labels(self):
        rows = [attempt(0, action([2], [3])), attempt(1, action([3], [2])),
                attempt(2, action([3], [2])), attempt(3, action())]
        before = select_blind(rows)
        assert before["sample_index"] == 1
        assert before["action"] == action([3], [2])
        for row in rows:
            row.update(target=action([2], [3]), score=100 if row["sample_index"] == 0 else 0)
        assert select_blind(list(reversed(rows))) == before
        assert select_blind(rows, first=True)["sample_index"] == 0

    def test_failure_is_not_abstention_or_retry_for_single_sample_arm(self):
        rows = [attempt(0), attempt(1, action())]
        assert select_blind(rows, first=True)["status"] == "invalid"
        assert select_blind(rows)["action"] == action()
        assert select_blind([attempt(0)])["action"] is None
        with self.assertRaises(ValueError):
            select_blind([attempt(0), attempt(0)])

    def test_order_invariance_and_tie_policy(self):
        rows = [attempt(2, action([3, 2])), attempt(0, action([2, 3])), attempt(1, action([5]))]
        result = select_blind(rows)
        assert result["distinct_actions"] == 2
        assert result["sample_index"] == 0
        assert select_blind([attempt(1, action([5])), attempt(0, action())])["sample_index"] == 0

    def test_schemas_preserve_multi_purchase_only_and_no_action(self):
        context = frozen_context(context_input(), 10)
        schema = generation_request(context, "direct")["final_schema"]
        for value in [action(), action([2, 3]), action(purchase=[3])]:
            validate_arguments(value, schema)
        with self.assertRaises(ValueError):
            validate_arguments(action([100]), schema)

    def test_evaluation_cannot_change_selections_and_invalid_empty_is_not_correct(self):
        invalid = select_blind([attempt(0)], first=True)
        selections = [{"scene_id": "1", "source_request_id": 1, "arms": {
            arm: deepcopy(invalid) for arm in ("A", "B", "C", "A10")}}]
        original = deepcopy(selections)
        report = evaluate_selected(selections, {"1": {"target": action(), "source_target": action()}}, "full")
        assert selections == original
        assert report["macro_all_requests"]["A"]["clicked_item_ids_exact_match"] == 0
        assert report["macro_all_requests"]["A"]["clicked_item_ids_f1"] == 0


if __name__ == "__main__":
    unittest.main()
