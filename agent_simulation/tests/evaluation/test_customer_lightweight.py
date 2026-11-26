import json
import unittest
import httpx
from agent_simulation.inference.agentic import AgenticClient
from agent_simulation.evaluation.customer_request_metrics import request_metrics

class LightweightTests(unittest.TestCase):
    def test_stream_timing_and_assembly(self):
        output = json.dumps({"query": "杯子"}, ensure_ascii=False)
        events = [
            {"choices":[{"delta":{"role":"assistant"},"finish_reason":None}]},
            {"choices":[{"delta":{"content":output[:4]},"finish_reason":None}]},
            {"choices":[{"delta":{"content":output[4:]},"finish_reason":"stop"}]},
            {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":8}}]
        body = "".join("data: "+json.dumps(e)+"\n\n" for e in events)+"data: [DONE]\n\n"
        client = AgenticClient("http://fixture/v1", "fixture", None,
            transport=httpx.MockTransport(lambda req: httpx.Response(200, text=body)), stream=True)
        record = {}
        data = client._stream_response({}, record)
        self.assertEqual(json.loads(data["choices"][0]["message"]["content"]), {"query":"杯子"})
        self.assertIsNotNone(record["ttft_seconds"])
        self.assertEqual(data["usage"]["completion_tokens"], 8)
        client.close()

    def test_reconstruction_deduplicates_and_rejects_missing_children(self):
        target = {"clicked_item_ids":[1,2], "purchased_item_ids":[2]}
        rows = []
        for item in (1,2):
            rows.append({"stage":"response","variant":"strong","policy":"sampled","tid":0,
              "trajectory":{"source_request_id":7,"scene_id":str(item),"status":"ok",
              "source_response_target":target,"final_action":{"clicked_item_ids":[item],
                "purchased_item_ids":[2] if item==2 else []}}})
        report = request_metrics(rows)[0]
        self.assertTrue(report["complete"])
        self.assertEqual(report["protocol"], "split_reconstructed")
        self.assertTrue(report["metrics"]["clicked_item_ids"]["exact_match"])
        self.assertFalse(request_metrics(rows[:1])[0]["complete"])

    def test_history_grouping_and_historical_price(self):
        from agent_simulation.evaluation.customer_evidence import attach_sources, price_claim_flags
        context = attach_sources({"evidence_actor_id":1, "observed_history":[
            {"session_id":2,"time_index":5,"query":"q","clicked_items":[],"purchased_items":[]},
            {"session_id":1,"time_index":2,"query":"p","clicked_items":[],"purchased_items":[]}],
            "displayed_items":[{"item_id":42,"item_title":"a","price_raw":39,"price_observed_time_index":3},
                               {"item_id":43,"item_title":"b","price_raw":None}]})
        self.assertEqual([e["time_index"] for e in context["history_events"]],[2,5])
        self.assertEqual(set(context["history_events"][0]["fact_refs"]),{"query","clicked_items","purchased_items"})
        self.assertIn("price_claim_for_unknown_item_price", price_claim_flags(
            {"knowledge":"Item 43 has a price of 1096."},context["facts"]))
        self.assertIn("historical_price_presented_without_time_qualification", price_claim_flags(
            {"knowledge":"Item 42 has a price of 39."},context["facts"]))
        self.assertEqual(price_claim_flags({"knowledge":"Item 42 had a historical observed price of 39."},
                                           context["facts"]),[])
        self.assertIn("item_price_value_mismatch",price_claim_flags(
            {"knowledge":"Item 42 had a historical price of 20."},context["facts"]))

    def test_fixed_prompt_preserves_author_template(self):
        from agent_simulation.agents.prompts.customer_agentic import fixed_customer_instructions
        from agent_simulation.vendor.usermirrorer.template import default_system
        prompt=fixed_customer_instructions()
        self.assertEqual(prompt.count(default_system),1)
        self.assertNotIn("Decision hypotheses available in this run",prompt)
