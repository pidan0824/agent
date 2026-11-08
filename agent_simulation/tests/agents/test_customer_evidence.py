"""Customer prompts preserve request boundaries, clocks and citation positions."""
from copy import deepcopy
import json
import unittest
from agent_simulation.agents.context.shopping import project_customer
from agent_simulation.agents.prompts.customer import decode_input, encode_input, messages, prepare_history
from agent_simulation.agents.prompts.customer import customer_decision_messages


def state():
    recent=[dict(session_id=i,time_index=i,query='同一搜索',clicked_items=[{'item_id':1,'price_raw':p}],purchased_items=[])
            for i,p in [(7,100),(8,90),(9,110)]]
    return dict(profile={'user_id':7},history_cutoff_exclusive=10,
        short_term_memory={'search_history':recent},long_term_memory=[],
        retrieved_context={'history':dict(kind='retrieved_factual_history',eligible_segments=6,returned_segments=1,
            segments=[dict(owner_id=7,clock='observed_time_index',start_inclusive=2,end_inclusive=2,
                evidence_kind='observed_fact',records=[dict(recent[0],session_id=2,time_index=2)])])})


class CustomerEvidenceTests(unittest.TestCase):
    def test_projection_preserves_order_and_does_not_mutate_source(self):
        original = state()
        simulated = {
            "request_id": "sim-3", "settled_step": 3, "query": "同一搜索",
            "clicked_items": [], "purchased_items": [],
        }
        original["retrieved_context"]["history"]["segments"].append({
            "owner_id": 7, "clock": "simulation_settled_step",
            "start_inclusive": 3, "end_inclusive": 3, "evidence_kind": "simulated_fact",
            "records": [simulated],
        })
        before = deepcopy(original)
        visible = project_customer(original)
        self.assertEqual(original, before)
        recent = visible["short_term_memory"]["search_history"]
        self.assertEqual(
            [(e["time_index"], e["clicked_items"][0]["price_raw"]) for e in recent],
            [(7, 100), (8, 90), (9, 110)],
        )
        self.assertTrue(all("session_id" not in e for e in recent))
        history = visible["retrieved_context"]["history"]
        self.assertNotIn("eligible_segments", history)
        self.assertNotIn("returned_segments", history)
        observed, simulated_segment = history["segments"]
        self.assertEqual(observed["clock"], "observed_time_index")
        self.assertEqual(observed["records"], [{**recent[0], "time_index": 2}])
        self.assertEqual(simulated_segment["clock"], "simulation_settled_step")
        self.assertEqual(simulated_segment["records"], [simulated])
        for segment in history["segments"]:
            self.assertNotIn("start_inclusive", segment)
            self.assertNotIn("end_inclusive", segment)

    def test_sft_runtime_and_prepared_response_share_projection_and_citation_positions(self):
        original=state()
        original['long_term_memory']=[{'content':'A recorded experience','memory_id':'customer:7:10',
            'owner':'customer:7','source_ids':['internal-source-id'],'evidence_times':[
            {'same_observations_in_retrieved_history':[{'segment_number':1,'record_number':1}]}]}]
        sft = messages(original, {'query': '答案'})
        runtime=customer_decision_messages(original)
        self.assertEqual(sft[1],runtime[1])
        self.assertIsInstance(json.loads(sft[1]['content']), dict)
        self.assertNotIn('答案',sft[1]['content'])
        self.assertNotIn('internal-source-id', sft[1]['content'])
        self.assertNotIn('customer:7:10', sft[1]['content'])
        self.assertIn('same_observations_in_retrieved_history', sft[1]['content'])
        self.assertEqual(original['long_term_memory'][0]['source_ids'], ['internal-source-id'])
        self.assertEqual(decode_input(sft[1]['content']), project_customer(original))
        response={**original,'query':'杯子','displayed_items':[{'item_id':1}]}
        prepared=prepare_history(project_customer(response))
        self.assertEqual(messages(response,response=True),messages(response,response=True,prepared_history=prepared))

    def test_compact_background_preserves_batches_labels_and_other_context(self):
        from agent_simulation.datasets.labels import expand_record

        original = state()
        original["behavioral_summary"] = {
            "observed_time_index": 6,
            "statistics_30d": {"exposures": 100, "clicks": 4, "orders": 1, "gmv_raw": 90},
            "recent_clicked_item_ids": [1, 1, 2],
            "recent_purchased_item_ids": [1],
            "item_details": [dict(item_id=i, item_title="杯子", brand_name="品牌",
                                  category_level3_name="水杯", category_level3_id=8,
                                  seller_id=20, category_level1_name="日用") for i in (1, 2)],
        }
        original.update(query="杯子", displayed_items=[
            {"item_id": i, "price_raw": 100 + i} for i in range(1, 69)
        ])
        query = dict(sample_id="query:1", input=original, target={"query": "杯子"})
        query["messages"] = messages(original, query["target"])
        response = dict(sample_id="response:1", input=original,
                        target={"clicked_item_ids": [1, 68], "purchased_item_ids": [68]})
        before = deepcopy((query, response))
        full = list(expand_record("customer", (query, response)))
        compact = list(expand_record("customer", (query, response), compact_background=True))
        self.assertEqual(len(compact), 3)
        for left, right in zip(full, compact, strict=True):
            self.assertEqual(left["sample_id"], right["sample_id"])
            self.assertEqual(left["target"], right["target"])
            self.assertEqual(left.get("batching"), right.get("batching"))
            a, b = (decode_input(row["messages"][1]["content"]) for row in (left, right))
            details = a["behavioral_summary"]["item_details"]
            a["behavioral_summary"]["item_details"] = [
                {k: v for k, v in item.items()
                 if k in ("item_id", "item_title", "brand_name", "category_level3_name")}
                for item in details
            ]
            self.assertEqual(a, b)
            self.assertEqual(right["messages"][1], customer_decision_messages(
                right["input"], response=right["sample_id"].startswith("response"),
                compact_background=True)[1])
        self.assertEqual((query, response), before)

    def test_customer_encoder_rejects_a_projection_with_an_audit_sidecar(self):
        visible = project_customer(state())
        with self.assertRaisesRegex(ValueError, "Customer context must be an object"):
            encode_input((visible, {"session_ids": [7, 8, 9]}))

    def test_projection_rejects_explicit_foreign_ownership(self):
        for target in ('recent','retrieved'):
            s=state()
            if target=='recent':s['short_term_memory']['search_history'][0]['user_id']=8
            else:s['retrieved_context']['history']['segments'][0]['owner_id']=8
            with self.assertRaises(ValueError):project_customer(s)

    def test_projection_reports_invalid_history_structure(self):
        with self.assertRaisesRegex(ValueError, "Customer retrieved context must be an object"):
            project_customer({"retrieved_context": None})
        with self.assertRaisesRegex(ValueError, "Customer history records must be a list"):
            project_customer({"short_term_memory": {"search_history": None}})
