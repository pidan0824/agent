"""Token budgets, lossless history tables, equivalent action schemas and Seller evidence."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent_simulation.agents.schemas import StrictSchema
from agent_simulation.agents.context.seller_features import seller_behavior_features
from agent_simulation.evaluation.customer_contract import single_item_actions, single_item_schema
from agent_simulation.evaluation.customer_curation import (
    CurationTools, budget_history, fixed_response_requests, observed_action_rows,
)
from agent_simulation.evaluation.customer_evidence import (
    attach_sources, compact_sources, encode_context, expand_sources,
)
from agent_simulation.evaluation.trajectories import FrozenSellerSampleTools
from agent_simulation.tests.evaluation.test_customer_agentic import request


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text)


class ContextOptimizationTests(unittest.TestCase):
    def test_history_budget_preserves_recent_related_and_same_query_no_action(self):
        context = {"evidence_actor_id": 7, "history_cutoff_exclusive": 20,
                   "query": "cup", "displayed_items": [{"item_id": 4, "item_title": "cup"}]}
        history = [
            {"session_id": 1, "time_index": 1, "query": "old", "clicked_items": [{"item_id": 4}], "purchased_items": []},
            {"session_id": 2, "time_index": 2, "query": " ＣＵＰ ", "clicked_items": [], "purchased_items": []},
            {"session_id": 3, "time_index": 3, "query": "irrelevant " * 2000, "clicked_items": [], "purchased_items": []},
            {"session_id": 4, "time_index": 4, "query": "recent", "clicked_items": [], "purchased_items": []},
        ]
        original = deepcopy((context, history))
        def render(events, budget):
            return budget_history(context, events, stage="response", tokenizer=CharacterTokenizer(), budget=budget)
        wanted = [history[i] for i in (0, 1, 3)]
        budget = len(encode_context(render(wanted, 100000))) - len(encode_context(render([], 100000)))
        result = render(history, budget)
        self.assertEqual([e["event_id"] for e in result["history_events"]], ["history:1", "history:2", "history:4"])
        self.assertLessEqual(len(encode_context(result)) - len(encode_context(render([], budget))), budget)
        no_action = next(e for e in result["history_events"] if e["event_id"] == "history:2")
        for field in ("clicked_items", "purchased_items"):
            self.assertEqual(result["facts"][no_action["fact_refs"][field]]["value"], [])
        self.assertEqual((context, history), original)

    def test_oversized_recent_event_is_not_truncated_and_other_evidence_can_fit(self):
        context = {"evidence_actor_id": 7, "history_cutoff_exclusive": 20, "query": "cup",
                   "displayed_items": [{"item_id": 4, "item_title": "cup"}]}
        history = [{"session_id": i, "time_index": i, "query": "cup" if i == 1 else "long" * 10000,
                    "clicked_items": [], "purchased_items": []} for i in (1, 2)]
        result = budget_history(context, history, stage="response", tokenizer=CharacterTokenizer(), budget=1000)
        self.assertEqual([e["event_id"] for e in result["history_events"]], ["history:1"])
        self.assertFalse(any("long" in str(f["value"]) for f in result["facts"].values()))

    def test_budgeted_query_remains_hidden_cache_is_read_only_and_labels_do_not_select_history(self):
        source = request()
        binding = CurationTools({"query_input": source["query_input"]}, "query", mode="fixed", history_token_budget=0)
        context = binding.initial_context()
        self.assertNotIn("SECRET TARGET", json.dumps(context))
        self.assertEqual(context["history_events"], [])
        first = CurationTools(source, "response", mode="fixed", history_token_budget=0)
        original = first.initial_context()
        mutated = first.initial_context()
        mutated["facts"].clear()
        self.assertEqual(first.initial_context(), original)
        changed = deepcopy(source)
        changed["response_target"] = {"clicked_item_ids": [], "purchased_item_ids": []}
        self.assertEqual(original, CurationTools(changed, "response", mode="fixed", history_token_budget=0).initial_context())
        first.context["query"] = "changed"
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            first.initial_context()
        for kwargs in ({"history_token_budget": -1}, {"history_token_budget": 1},
                       {"history_token_budget": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CurationTools(source, "response", mode="fixed", **kwargs)

    def test_teacher_direct_and_observed_actions_share_budgeted_facts(self):
        source = request()
        context, requests = fixed_response_requests(source["response_input"], history_token_budget=0)
        observed = observed_action_rows(source, 11, "train", "release", policies=("full",), history_token_budget=0)
        row = next(row for row in observed if row["stage"] == "response")
        def facts(messages):
            return next(json.loads(m["content"]) for m in messages if m["role"] == "user"
                        and m["content"].startswith("{") and "facts" in json.loads(m["content"]))
        self.assertEqual(facts(requests["direct"]["messages"]), facts(requests["ekb"]["messages"]))
        self.assertEqual(facts(row["messages"]), facts(requests["direct"]["messages"]))
        self.assertEqual(expand_sources(facts(row["messages"])), context)
        self.assertEqual(row["target"], source["response_target"])

    def item_context(self):
        items = [{"item_id": i, "item_title": "same long descriptive product title",
                  "seller_id": 7, "seller_name": "same seller", "brand_id": 8, "brand_name": "same brand",
                  "category_level1_name": "food", "category_level2_name": "snacks",
                  "category_level3_name": "plums", "price_raw": None if i == 3 else i * 10}
                 for i in (1, 2, 3, 1)]
        return attach_sources({"evidence_actor_id": 7, "history_cutoff_exclusive": 10,
            "observed_history": [{"session_id": 1, "time_index": 2, "query": "plums",
                                 "clicked_items": items, "purchased_items": [items[1]]}]})

    def test_shared_history_tables_preserve_action_order_ids_null_prices_and_fact_ids(self):
        context = self.item_context()
        before = deepcopy(context)
        wire = compact_sources(context)
        self.assertIn("history_item_tables", wire)
        self.assertEqual(expand_sources(wire), context)
        self.assertEqual(context, before)
        self.assertEqual(set(wire["facts"]), set(context["facts"]))
        key = next(k for k, f in context["facts"].items() if f["field"] == "clicked_items")
        self.assertIn("$table", wire["facts"][key][2])
        values = expand_sources(wire)["facts"][key]["value"]
        self.assertEqual([v["item_id"] for v in values], [1, 2, 3, 1])
        self.assertEqual([v["price_raw"] for v in values], [10, 20, None, 10])
        legacy = deepcopy(wire)
        legacy.pop("history_item_tables")
        legacy["facts"][key][2] = deepcopy(context["facts"][key]["value"])
        self.assertLess(len(json.dumps(wire)), len(json.dumps(legacy)))
        self.assertEqual(expand_sources(legacy), context)

    def test_corrupt_history_tables_are_rejected(self):
        for corrupt in ("marker", "missing_marker", "width", "duplicate_column", "shared_collision", "extra"):
            with self.subTest(corrupt=corrupt):
                wire = compact_sources(self.item_context())
                key = next(k for k, r in wire["facts"].items() if r[1] == "clicked_items")
                table = wire["facts"][key][2]["$table"]
                if corrupt == "marker":
                    wire["history_item_tables"] = "unsupported"
                elif corrupt == "missing_marker":
                    wire.pop("history_item_tables")
                elif corrupt == "width":
                    table["rows"][0].append(99)
                elif corrupt == "duplicate_column":
                    table["columns"][1] = table["columns"][0]
                elif corrupt == "shared_collision":
                    table["shared"][table["columns"][0]] = 99
                else:
                    table["extra"] = "invalid"
                with self.assertRaises(ValueError):
                    expand_sources(wire)

    def test_action_enum_preserves_the_entire_joint_action_space(self):
        for ids in ([], [1], [1, 2], list(range(100, 112))):
            actions = single_item_actions(ids)
            old = {"anyOf": [{"const": action} for action in actions]}
            new = single_item_schema(ids, compact=True)
            self.assertEqual(new["enum"], actions)
            self.assertEqual(len(actions), 1 + 3 * len(ids))
            self.assertLess(len(json.dumps(new)), len(json.dumps(old)))
            choices = [[], [1], [2], [99], [1, 2], [1, 1], [1.0], ["1"], [True]]
            invalid_or_valid = [{"clicked_item_ids": c, "purchased_item_ids": p} for c in choices for p in choices]
            invalid_or_valid += [{}, {"clicked_item_ids": []}, {"clicked_item_ids": [], "purchased_item_ids": [], "extra": 1}]
            for action in actions + invalid_or_valid:
                self.assertEqual(StrictSchema(old).is_valid(action), StrictSchema(new).is_valid(action), action)

    def test_seller_recent_changes_survive_long_stable_tail_and_skip_unknown_quotes(self):
        observations = [(1, 100), (2, 110), (3, None), (4, 110), (5, 90)]
        observations += [(t, 90) for t in range(6, 19)] + [(19, None), (20, 999)]
        records = [{"seller_id": 7, "item_id": 4, "time_index": t, "price_raw": price,
                    "price_status": "observed" if price is not None else "unknown"} for t, price in observations]
        before = deepcopy(records)
        feature = seller_behavior_features(records, seller_id=7, cutoff=20)["products"][0]
        self.assertEqual(feature["last_observed_index"], 19)
        self.assertEqual(feature["last_valid_price_raw"], 90)
        self.assertEqual(feature["last_valid_quote_index"], 18)
        self.assertEqual(feature["recent_quote_changes"], [
            {"before": {"time_index": 1, "price_raw": 100}, "after": {"time_index": 2, "price_raw": 110}},
            {"before": {"time_index": 4, "price_raw": 110}, "after": {"time_index": 5, "price_raw": 90}}])
        self.assertEqual(records, before)
        unknown = seller_behavior_features([records[2]], seller_id=7, cutoff=20)["products"][0]
        self.assertIsNone(unknown["last_valid_quote_index"])
        self.assertEqual(unknown["recent_quote_changes"], [])

    def test_seller_initial_context_reuses_scoped_market_without_another_query(self):
        sample = {"input": {"profile": {"seller_id": 7}, "history_cutoff_exclusive": 10,
            "requested_price_item_ids": [4], "market_semantics": "public quote snapshot",
            "owned_products": [{"item_id": 4, "last_observed_price_raw": 12,
                "market_position": {"category_ids": [1, 2, 3], "window": {"start_inclusive": 0, "end_exclusive": 10},
                                    "recent": {"median_price": 15, "products": 20},
                                    "analysis": "duplicate prose", "competitor_examples": {"unused": True}}}]},
            "target": {"prices": [{"item_id": 4, "price_raw": 99999}]}}
        history = [{"seller_id": 7, "item_id": 4, "time_index": 9, "price_status": "observed", "price_raw": 12}]
        catalog = SimpleNamespace(connection=Mock(), config=object())
        before = deepcopy(sample)
        with patch("agent_simulation.evaluation.trajectories.DataRepository") as repository, \
             patch("agent_simulation.evaluation.trajectories.rows", return_value=history):
            binding = FrozenSellerSampleTools(catalog, sample)
            context = binding.initial_context()
            market = context["owned_products"][0]["market_position"]
            self.assertEqual(market["recent"], sample["input"]["owned_products"][0]["market_position"]["recent"])
            self.assertNotIn("analysis", market)
            self.assertNotIn("competitor_examples", market)
            self.assertNotIn("99999", json.dumps(context))
            self.assertEqual(context["market_semantics"], "public quote snapshot")
            self.assertEqual(repository.return_value.mock_calls, [])
            bad = deepcopy(sample)
            bad["input"]["owned_products"][0]["market_position"]["window"]["end_exclusive"] = 11
            with self.assertRaisesRegex(ValueError, "Future Seller market window"):
                FrozenSellerSampleTools(catalog, bad).initial_context()
        self.assertEqual(sample, before)

    def test_budgeted_recall_keeps_same_query_no_action_beside_same_item_history(self):
        from agent_simulation.tests.evaluation.test_customer_curation import CurationTests
        source = CurationTests.history_request()
        history = source["response_input"]["short_term_memory"]["search_history"]
        history[0]["clicked_items"] = [{"item_id": 4}]
        history[1]["purchased_items"] = [{"item_id": 4}]
        history[2]["query"] = source["response_input"]["query"]
        selected = CurationTools(source, "response", mode="fixed", history_token_budget=100000,
                                 tokenizer=CharacterTokenizer()).initial_context()
        ids = {e["event_id"] for e in selected["history_events"]}
        self.assertTrue({"history:1", "history:2", "history:3", "history:13"} <= ids)
        self.assertLessEqual(len(ids), 10)

    def test_history_table_evidence_checks_match_original(self):
        from agent_simulation.evaluation.customer_evidence import check_evidence
        from agent_simulation.tests.ekb_fixture import rationale
        source = self.item_context()
        key = next(k for k, f in source["facts"].items() if f["field"] == "clicked_items")
        explanation = rationale()
        explanation["evidence_refs"] = {stage: [key] for stage in ("stimulus", "knowledge", "evaluation")}
        action = {"clicked_item_ids": [], "purchased_item_ids": []}
        def trajectory(context):
            return {"stage": "response", "response_style": "ekb", "rationale": explanation,
                    "final_action": action, "messages": [
                        {"role": "user", "content": json.dumps(context)},
                        {"role": "assistant", "content": json.dumps({"rationale": explanation, "action": action})}]}
        expected = check_evidence(trajectory(source))
        self.assertTrue(expected["accepted"])
        self.assertEqual(check_evidence(trajectory(compact_sources(source))), expected)

    def test_compact_prompt_keeps_original_decoder_schema(self):
        from agent_simulation.evaluation.trajectories import trajectory_messages
        source = request()
        source["response_input"]["response_action_mode"] = "single_item"
        for theory in (False, True):
            binding = CurationTools(source, "response", mode="fixed", theory=theory)
            schemas, messages = trajectory_messages(binding, role="customer", stage="response", theory=theory)
            self.assertEqual(schemas, [single_item_schema([4])])
            output = json.loads(messages[0]["content"].split("Schema: ")[-1])
            compact = output if not theory else output["properties"]["action"]
            self.assertEqual(compact, single_item_schema([4], compact=True))


    def test_teacher_budget_priority_uses_the_budgeted_history(self):
        from agent_simulation.evaluation.customer_generation import budget_scenes
        from agent_simulation.tests.evaluation.test_customer_generation import job
        import io
        value = job(1)
        value[4]["response_input"]["short_term_memory"]["search_history"][0]["clicked_items"] = [{"item_id": 4}]
        audit = io.StringIO()
        budget_scenes([value], 1, 42, audit, history_token_budget=0)
        self.assertFalse(json.loads(audit.getvalue())["same_item_history"])
