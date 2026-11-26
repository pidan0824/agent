import json
from copy import deepcopy
import unittest
import numpy as np

from agent_simulation.datasets.customer_candidates import construct_candidates, candidate_coverage
from agent_simulation.evaluation.customer_filtering import probability_record, select_preferences, decode_states
from agent_simulation.evaluation.customer_curation import CurationTools
from agent_simulation.tests.evaluation.test_customer_agentic import request
from agent_simulation.tests.evaluation.test_trajectories import branch


class CurationTests(unittest.TestCase):
    def test_shortlist_does_not_insert_true_item_and_missing_positive_is_not_no_click(self):
        context = request()["response_input"]
        context["displayed_items"] = [{"item_id": i, "item_title": str(i)} for i in (1, 2, 3)]
        class Encoder:
            def encode_many(self, texts, **kwargs):
                return [np.array([float(text.strip())]) if text.strip().isdigit() else np.array([1.]) for text in texts]
        short, info = construct_candidates(context, policy="retrieved", limit=1, encoder=Encoder())
        self.assertEqual([p["item_id"] for p in short["displayed_items"]], [3])
        target = {"clicked_item_ids": [1, 2], "purchased_item_ids": []}
        status = candidate_coverage(info, target)
        self.assertFalse(status["reachable"])
        self.assertEqual(status["behavior_group"], "multi_click")
        self.assertEqual(target["clicked_item_ids"], [1, 2])
        self.assertEqual(len(context["displayed_items"]), 3)

    def test_four_states_preserve_multi_no_click_and_purchase_without_click(self):
        self.assertEqual(decode_states([1,2,3,4], [0,1,2,3]),
            {"clicked_item_ids": [2,4], "purchased_item_ids": [3,4]})
        target = {"clicked_item_ids": [], "purchased_item_ids": []}
        values = probability_record([[1.,0,0,0]]*2, [1,2], target, rng=np.random.default_rng(0))
        self.assertTrue(values["preference"])
        self.assertEqual(values["sampled_action"], target)
        self.assertAlmostEqual(values["entropy"] - values["mean_item_entropy_bits"], 1.)

    def test_original_filtering_selects_same_context_pair_and_excludes_unreachable_test(self):
        target = {"clicked_item_ids": [4], "purchased_item_ids": [4]}
        rows=[]
        for variant, probability in [("strong", [0,0,0,1]), ("weak", [1,0,0,0])]:
            row = probability_record([probability], [4], target, rng=np.random.default_rng(0))
            row.update(dataset="KuaiSearch/test-domain", cid="same", tid=0, variant=variant,
                       source_split="train", reachable=True, choice_cnt=1, behavior_group="single_click",
                       trajectory=branch(variant, row["sampled_action"], variant))
            row["trajectory"].update(target=target, accepted=variant=="strong", score=1. if variant=="strong" else 0.)
            row["trajectory"]["rationale"] = json.loads(row["trajectory"]["messages"][-1]["content"])["rationale"]
            rows.append(row)
        sft, pairs, metrics = select_preferences(rows, pair_protocol="sampled_match")
        self.assertEqual((len(sft), len(pairs), len(metrics)), (1,1,1))
        self.assertEqual(pairs[0]["chosen_trajectory_id"], "strong")
        self.assertEqual(pairs[0]["rejected_trajectory_id"], "weak")
        self.assertEqual(metrics[0]["teacher_hit"], 1.)
        self.assertEqual(metrics[0]["student_hit"], 0.)
        for name, value in [("reachable",False), ("source_split","test")]:
            changed=deepcopy(rows)
            for row in changed:
                row[name]=value
                if name == "source_split":
                    row["trajectory"]["source_split"] = value
            self.assertEqual(select_preferences(changed), ([],[],[]))

    def test_fixed_and_autonomous_keep_query_hidden_and_exposure_read_only(self):
        observed=request()
        fixed=CurationTools(observed,"query",mode="fixed")
        autonomous=CurationTools(observed,"query",mode="autonomous")
        self.assertFalse(fixed.schemas)
        self.assertTrue(autonomous.schemas)
        self.assertNotIn("SECRET TARGET",json.dumps(fixed.initial_context()))
        self.assertNotIn("SECRET TARGET",json.dumps(autonomous.initial_context()))
        self.assertNotIn("displayed_items",fixed.initial_context())
        self.assertEqual(fixed.cutoff,autonomous.cutoff)

    @staticmethod
    def history_request():
        observed = request()
        history = [{"session_id": i, "time_index": i, "query": f"unrelated {i}",
                    "clicked_items": [], "purchased_items": []} for i in range(1, 14)]
        for stage in ("query", "response"):
            observed[stage + "_input"]["history_cutoff_exclusive"] = 20
            observed[stage + "_input"]["short_term_memory"]["search_history"] = deepcopy(history)
        return observed

    def test_response_recalls_same_id_actions_before_newer_exact_query(self):
        observed = self.history_request()
        history = observed["response_input"]["short_term_memory"]["search_history"]
        history[0]["clicked_items"] = [{"item_id": 4}]
        history[1]["purchased_items"] = [{"item_id": 4}]
        history[2]["query"] = observed["response_input"]["query"]
        binding = CurationTools(observed, "response", mode="fixed")
        context = binding.initial_context()
        self.assertEqual([e["event_id"] for e in context["history_events"]],
                         [f"history:{i}" for i in [1, 2, *range(6, 14)]])
        self.assertEqual((context["selected_history_events"], context["available_history_events"]), (10, 13))
        binding.guard()
        self.assertEqual(binding.history, history)
        changed = deepcopy(observed)
        changed["response_target"] = {"clicked_item_ids": [], "purchased_item_ids": []}
        self.assertEqual(context, CurationTools(changed, "response", mode="fixed").initial_context())
        history[2]["clicked_items"] = [{"item_id": 4}]
        newer = CurationTools(observed, "response", mode="fixed").initial_context()
        self.assertEqual([e["event_id"] for e in newer["history_events"]],
                         [f"history:{i}" for i in [2, 3, *range(6, 14)]])

    def test_response_exact_query_normalization_retains_no_action_event(self):
        observed = self.history_request()
        observed["response_input"]["query"] = "cup STRASSE"
        history = observed["response_input"]["short_term_memory"]["search_history"]
        history[0]["query"] = "  ＣＵＰ　Straße\t "
        context = CurationTools(observed, "response", mode="fixed").initial_context()
        self.assertEqual([e["event_id"] for e in context["history_events"]],
                         [f"history:{i}" for i in [1, *range(5, 14)]])
        event = context["history_events"][0]
        for field in ("clicked_items", "purchased_items"):
            self.assertEqual(context["facts"][event["fact_refs"][field]]["value"], [])

    def test_response_does_not_recall_same_title_or_partial_query(self):
        observed = self.history_request()
        history = observed["response_input"]["short_term_memory"]["search_history"]
        history[0]["clicked_items"] = [{"item_id": 99, "item_title": "Cup"}]
        history[0]["query"] = "SECRET"
        context = CurationTools(observed, "response", mode="fixed").initial_context()
        self.assertEqual([e["event_id"] for e in context["history_events"]],
                         [f"history:{i}" for i in range(4, 14)])

    def test_history_recall_preserves_foreign_actor_rejection(self):
        observed = self.history_request()
        event = observed["response_input"]["short_term_memory"]["search_history"][0]
        event.update(user_id=8, clicked_items=[{"item_id": 4}])
        with self.assertRaisesRegex(ValueError, "Foreign Customer history"):
            CurationTools(observed, "response", mode="fixed").initial_context()

    def test_history_budget_keeps_a_recent_event_and_disables_recall_below_two(self):
        observed = self.history_request()
        history = observed["response_input"]["short_term_memory"]["search_history"]
        for event in history[:3]:
            event["clicked_items"] = [{"item_id": 4}]
        expected = {0: [], 1: [13], 2: [3, 13], 3: [2, 3, 13],
                    10: [2, 3, *range(6, 14)], 20: list(range(1, 14))}
        for limit, ids in expected.items():
            with self.subTest(history_limit=limit):
                context = CurationTools(observed, "response", mode="fixed", history_limit=limit).initial_context()
                self.assertEqual([e["event_id"] for e in context["history_events"]],
                                 [f"history:{i}" for i in ids])
                self.assertEqual(context["selected_history_events"], len(ids))
                self.assertLessEqual(len(ids), limit)

    def test_query_keeps_recent_history_without_response_or_target_access(self):
        observed = self.history_request()
        observed["query_input"]["short_term_memory"]["search_history"][0]["clicked_items"] = [{"item_id": 4}]
        query_only = {"query_input": deepcopy(observed["query_input"])}
        context = CurationTools(query_only, "query", mode="fixed").initial_context()
        self.assertEqual(context, CurationTools(observed, "query", mode="fixed").initial_context())
        self.assertEqual([e["event_id"] for e in context["history_events"]],
                         [f"history:{i}" for i in range(4, 14)])
        self.assertNotIn("SECRET TARGET", json.dumps(context))
        self.assertNotIn("query", context)
        self.assertEqual(context["candidate_context"], [])

    def test_history_recall_rejects_current_and_future_events(self):
        for when in (20, 21):
            with self.subTest(time_index=when):
                observed = self.history_request()
                event = observed["response_input"]["short_term_memory"]["search_history"][0]
                event.update(time_index=when, clicked_items=[{"item_id": 4}])
                with self.assertRaisesRegex(ValueError, "Non-prior history"):
                    CurationTools(observed, "response", mode="fixed")

    def test_information_modes_have_distinct_trajectory_ids(self):
        import asyncio
        from agent_simulation.agents.agentic import AgenticOptions
        from agent_simulation.evaluation.trajectories import generate_trajectory
        from agent_simulation.tests.evaluation.test_trajectories import envelope
        class Model:
            model, seed = "same-model", 42
            async def acomplete(self, value, *, call_id):
                return {"role": "assistant", "content": json.dumps(envelope({"query": "杯子"})), "tool_calls": []}
        results=[]
        for mode in ("fixed", "autonomous"):
            results.append(asyncio.run(generate_trajectory(
                CurationTools(request(), "query", mode=mode), Model(), role="customer", actor_id=7,
                stage="query", record_id=11, source_split="train", catalog_identity="catalog",
                options=AgenticOptions(), sample_index=0)))
        self.assertTrue(all(row["status"]=="ok" for row in results))
        self.assertEqual(results[0]["initial_state_id"],results[1]["initial_state_id"])
        self.assertNotEqual(results[0]["trajectory_id"],results[1]["trajectory_id"])

    def test_sft_does_not_require_predictor_success_or_a_rejected_branch(self):
        from agent_simulation.evaluation.customer_filtering import select_preferences
        t=branch("good",{"clicked_item_ids":[],"purchased_item_ids":[]},"valid original evidence")
        t.update(target={"clicked_item_ids":[],"purchased_item_ids":[]},accepted=True,score=1.)
        row={"stage":"response","variant":"strong","source_split":"train","trajectory":t,"prob":None,"reachable":True}
        sft,pairs,metrics=select_preferences([row])
        self.assertEqual(len(sft),1)
        self.assertEqual(sft[0]["messages"],t["messages"])
        self.assertEqual((pairs,metrics),([],[]))

    def test_generation_defaults_to_prepared_sampled_scenes(self):
        from agent_simulation.evaluation.customer_curation import parser
        args=parser().parse_args(["--dataset","unused","--output","unused"])
        self.assertEqual(args.candidate_policies,["sampled"])
        self.assertEqual(args.modes,["fixed"])
        self.assertEqual(args.weak_model,"Qwen3.5-9B")
        self.assertEqual(str(args.weak_tokenizer),"models/Qwen3.5-9B")
        self.assertEqual(args.count,0)

    def test_offline_export_keeps_sft_without_probabilities(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from agent_simulation.evaluation.customer_filtering import export_training
        t=branch("accepted",{"clicked_item_ids":[],"purchased_item_ids":[]},"facts")
        t.update(target={"clicked_item_ids":[],"purchased_item_ids":[]},accepted=True,score=1.)
        row={"stage":"response","variant":"strong","source_split":"train","trajectory":t,
             "prob":None,"reachable":True}
        with TemporaryDirectory() as directory:
            result=export_training([row],Path(directory))
            self.assertEqual((result["sft_total"],result["pairs_total"]),(1,0))
            self.assertEqual(json.loads((Path(directory)/"sft.jsonl").read_text())["messages"],t["messages"])

    def test_ekb_preserves_original_detail_without_extra_length_caps(self):
        from agent_simulation.agents.prompts.customer_ekb import EKB_SCHEMA, ekb_instructions
        from agent_simulation.vendor.usermirrorer.template import default_system
        from agent_simulation.agents.schemas import validate_arguments
        from agent_simulation.tests.ekb_fixture import rationale
        prompt=ekb_instructions()
        self.assertTrue(prompt.startswith(default_system))
        self.assertNotIn("100 Chinese characters",prompt)
        self.assertNotIn("one short sentence",prompt)
        value=rationale()
        for name in ("stimulus","knowledge","evaluation"):
            value[name]="详细依据"*500
            self.assertNotIn("maxLength",EKB_SCHEMA["properties"][name])
        validate_arguments(value,EKB_SCHEMA)
