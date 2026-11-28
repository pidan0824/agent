"""Readable protocol contracts, provenance and offline export; no model services."""
from copy import deepcopy
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
import httpx

from agent_simulation.agents.agentic import AgenticOptions, initial_agentic_request
from agent_simulation.agents.schemas import validate_arguments
from agent_simulation.evaluation.customer_context import CurationTools, ContextCache
from agent_simulation.evaluation.customer_evidence import encode_context, check_evidence
from agent_simulation.evaluation.customer_readable import (
    mark_readable, readable_view, semantic_values, view_values, budget_readable)
from agent_simulation.evaluation.trajectories import (
    trajectory_messages, generate_trajectory, OutcomeScorer, score_group, digest)
from agent_simulation.evaluation.customer_filtering import export_training, OptionPredictor
from agent_simulation.evaluation.customer_curation import observed_action_rows
from agent_simulation.tests.evaluation.test_customer_background import source
from agent_simulation.tests.evaluation.test_context_optimization import CharacterTokenizer
from agent_simulation.tests.ekb_fixture import rationale

TOKENIZER = CharacterTokenizer()


def binding(stage="A", policy="frozen", raw=None):
    return CurationTools(raw or source(), "response", mode="fixed", concise=True,
        history_token_budget=10000, tokenizer=TOKENIZER,
        response_context_protocol="readable-v1", readable_stage=stage, readable_history_policy=policy)


def trajectory(ctx):
    r = rationale()
    r["evidence_refs"] = {"stimulus": ["query"], "knowledge": ["item:9"], "evaluation": ["item:4"]}
    action = {"clicked_item_ids": [4], "purchased_item_ids": []}
    return {"stage": "response", "role": "customer", "response_style": "ekb", "context_protocol": "readable-v1",
            "evidence_context": ctx, "rationale": r, "final_action": action,
            "messages": [{"role": "user", "content": encode_context(ctx)},
                         {"role": "assistant", "content": json.dumps({"rationale": r, "action": action})}]}


class ReadableTests(unittest.TestCase):
    def test_a_preserves_all_fact_values_and_has_no_transport_hashes(self):
        old = CurationTools(source(), "response", mode="fixed", history_token_budget=10000,
                            tokenizer=TOKENIZER).initial_context()
        new = binding().initial_context()
        canonical = {k: v for k, v in new.items() if k not in (
            "context_protocol", "readable_stage", "readable_history_policy")}
        self.assertEqual(old, canonical)
        view = readable_view(new)
        self.assertEqual(semantic_values(old), view_values(view))
        text = encode_context(new)
        for ref in old["facts"]:
            self.assertNotIn(ref, text)
        self.assertNotIn("fact_layout", text)
        self.assertNotIn("fact_ref", text)
        self.assertIn("clicked_items", text)
        self.assertIn("related_history", text)

    def test_stages_change_one_factor_at_a_time_and_cache_does_not_mix(self):
        cache = ContextCache()
        values, systems = [], []
        for stage in ("A", "B", "C"):
            b = CurationTools(source(), "response", mode="fixed", concise=True,
                history_token_budget=10000, tokenizer=TOKENIZER, context_cache=cache,
                response_context_protocol="readable-v1", readable_stage=stage)
            ctx = b.initial_context()
            values.append(readable_view(ctx))
            systems.append(trajectory_messages(b, role="customer", stage="response", theory=True)[1][0])
        a, b, c = deepcopy(values)
        for p in a["candidate_context"]:
            p.pop("related_history", None); p.pop("snapshot_interactions", None)
        a["readable_stage"] = b["readable_stage"]
        self.assertEqual(a, b)
        b["readable_stage"] = c["readable_stage"]
        self.assertEqual(b, c)
        self.assertEqual(systems[0], systems[1])
        self.assertNotEqual(systems[1], systems[2])
        self.assertNotIn("Past interaction counts indicate familiarity", systems[2]["content"])

    def test_readable_citations_pass_and_transcript_or_sidecar_mutations_fail(self):
        ctx = binding().initial_context()
        t = trajectory(ctx)
        self.assertTrue(check_evidence(t)["accepted"])
        mutations = ("view", "actor", "time", "reference", "final")
        for name in mutations:
            bad = deepcopy(t)
            if name == "view":
                bad["messages"][0]["content"] = bad["messages"][0]["content"].replace("品牌甲", "篡改品牌")
            elif name == "actor":
                f = next(f for f in bad["evidence_context"]["facts"].values() if f["kind"] == "background")
                f["actor_id"] = 999
            elif name == "time":
                f = next(f for f in bad["evidence_context"]["facts"].values() if f["kind"] == "background")
                f["time_index"] = 999
            elif name == "reference":
                bad["rationale"]["evidence_refs"]["knowledge"] = ["item:999"]
            else:
                bad["final_action"]["clicked_item_ids"] = []
            with self.subTest(name=name):
                self.assertFalse(check_evidence(bad)["accepted"])

    def test_prior_prices_validated_even_if_uncited(self):
        ctx = binding().initial_context()
        key = "Pexample"
        ctx["facts"][key] = {"kind": "product", "source_id": "item:4", "item_id": 4,
                            "field": "price_raw", "value": 20, "time_index": 10}
        with self.assertRaisesRegex(ValueError, "strictly-prior"):
            mark_readable(ctx)

    def test_hidden_snapshot_counts_are_not_mapped_through_candidate_id(self):
        from agent_simulation.evaluation.customer_readable import reference_map
        ctx = binding("B").initial_context()
        ctx["readable_history_policy"] = "budgeted"
        ctx["facts"] = {k: f for k, f in ctx["facts"].items() if f["field"] != "item_history"}
        refs = reference_map(ctx)
        self.assertNotIn("snapshot", refs)
        self.assertTrue(all(ctx["facts"][ref]["kind"] == "product" for ref in refs["item:4"]))

    def test_frozen_sidecar_cannot_be_missing(self):
        t = trajectory(binding().initial_context());del t["evidence_context"]
        self.assertFalse(check_evidence(t)["accepted"])

    def test_query_unchanged_and_autonomous_response_rejected(self):
        raw = source()
        legacy = CurationTools(raw, "query", mode="fixed").initial_context()
        readable = CurationTools(raw, "query", mode="fixed", response_context_protocol="readable-v1",
                                 readable_stage="C").initial_context()
        self.assertEqual(legacy, readable)
        with self.assertRaisesRegex(ValueError, "fixed"):
            CurationTools(raw, "response", mode="autonomous", response_context_protocol="readable-v1")

    def test_no_budget_message_readable_fixed_legacy_retains_it(self):
        for protocol, count in (("readable-v1", 2), ("legacy", 3)):
            b = CurationTools(source(), "response", mode="fixed", concise=True,
                history_token_budget=10000, tokenizer=TOKENIZER, response_context_protocol=protocol)
            _, messages = trajectory_messages(b, role="customer", stage="response", theory=True)
            request, _, index = initial_agentic_request(messages, [], AgenticOptions(max_model_calls=1))
            self.assertEqual(len(request), count)
            self.assertEqual(index is None, protocol == "readable-v1")
            if protocol == "readable-v1":
                self.assertNotIn('"enum"', request[0]["content"])

    def test_budget_uses_complete_records_and_labels_do_not_affect_selection(self):
        ctx = binding("B").initial_context()
        new = budget_readable(ctx, tokenizer=TOKENIZER, budget=3000)
        view = readable_view(new)
        fields = {key: view[key] for key in ("history_events", "prior_interacted_products") if key in view}
        cost = len(TOKENIZER.encode(json.dumps(fields, ensure_ascii=False, separators=(",", ":")),
                                    add_special_tokens=False))
        self.assertEqual(cost, new["readable_history_tokens"])
        self.assertLessEqual(cost, 3000)
        for p in view["prior_interacted_products"]["products"]:
            expected = next(f["value"] for f in ctx["facts"].values()
                            if f["field"] == "item_history" and f["item_id"] == p["item_id"])
            self.assertEqual(p, expected)
        changed = source();changed["response_target"] = {"clicked_item_ids": [999], "purchased_item_ids": [999]}
        self.assertEqual(binding("B", "budgeted").initial_context(),
                         binding("B", "budgeted", changed).initial_context())

    def test_generation_checker_and_export_keep_readable_rationale(self):
        b = binding("C")
        t = trajectory(b.initial_context())
        response = t["messages"][-1]
        async def complete(request, call_id):
            return deepcopy(response)
        model = SimpleNamespace(model="fixture", seed=1, acomplete=complete)
        generated = asyncio.run(generate_trajectory(b, model, role="customer", actor_id=7, stage="response",
            record_id=11, source_split="train", catalog_identity="fixture",
            options=AgenticOptions(max_model_calls=1), sample_index=0))
        self.assertEqual(generated["status"], "ok")
        self.assertTrue(generated["evidence_quality"]["accepted"])
        self.assertEqual(len(generated["prompt"]), 2)
        target = t["final_action"]
        score_group([generated], target, OutcomeScorer())
        row = {"stage": "response", "dataset": "fixture", "cid": "one", "tid": 0, "variant": "strong",
               "source_split": "train", "trajectory": generated, "reachable": True,
               "choice_cnt": 2, "prob": None}
        with TemporaryDirectory() as path:
            report = export_training([row], Path(path), sft_only=True)
            self.assertEqual(report["response_sft"], 1)
            saved = json.loads((Path(path) / "sft.jsonl").read_text())
            self.assertEqual(saved["context_protocol"], "readable-v1")
            self.assertEqual(saved["messages"], generated["messages"])

    def test_readable_cot_comparison_preserves_reviewed_explanation(self):
        from agent_simulation.evaluation.customer_cot import prepare_comparison
        from agent_simulation.evaluation.customer_filtering import combine_observed_sft
        raw = source()
        action = {"clicked_item_ids": [4], "purchased_item_ids": []}
        raw["response_target"] = action
        b = binding("C", raw=raw)
        t = trajectory(b.initial_context())
        _, prompt = trajectory_messages(b, role="customer", stage="response", theory=True)
        prompt, _, _ = initial_agentic_request(prompt, [], AgenticOptions(max_model_calls=1))
        t.update(trajectory_id="readable-teacher", source_split="train", status="ok", actor_id=7,
                 source_request_id=11, scene_id="11", dataset_manifest_sha256="fixture",
                 initial_state_id="readable-state", candidate_item_ids=[4, 5], tools=[],
                 action_mode="multi_item", target=action, prompt=prompt,
                 messages=prompt + [t["messages"][-1]])
        observed = observed_action_rows(raw, 11, "train", "fixture", policies=("full",),
                    history_token_budget=10000, tokenizer=TOKENIZER,
                    response_context_protocol="readable-v1", readable_stage="C")
        self.assertEqual(len(combine_observed_sft([], [], observed)), 2)
        reviews = {t["trajectory_id"]: {"trajectory_id": t["trajectory_id"],
            "messages_sha256": digest(t["messages"]), "verdict": "pass",
            "reason": "Offline fixture only", "reviewer": "test"}}
        records = [{"source_split": "train", "variant": "strong", "trajectory": t}]
        direct, cot, report, _, _ = prepare_comparison(records, observed, reviews)
        self.assertEqual(report["rationale_scenes"], 1)
        self.assertEqual(cot[-1]["messages"], t["messages"])
        self.assertEqual(direct[-1]["messages"][1:-1], cot[-1]["messages"][1:-1])

    def test_probability_inputs_use_the_same_readable_view(self):
        b = binding("B")
        client = SimpleNamespace(tokenizer=TOKENIZER, model="fixture", http=Mock())
        client.http.post.side_effect = RuntimeError("captured")
        with patch("agent_simulation.evaluation.customer_filtering.prompt_tokens", return_value=[1]) as capture:
            with self.assertRaisesRegex(RuntimeError, "captured"):
                OptionPredictor(client).score(b, {"rationale": {}, "trace": {"tools": []}})
            value = json.loads(capture.call_args_list[0].args[0][1]["content"])
        self.assertEqual(value["context"], readable_view(b.initial_context()))
        self.assertNotIn("facts", value["context"])

    def test_decoder_receives_strict_schema_with_short_prompt(self):
        from agent_simulation.inference.agentic import AgenticClient
        b = binding()
        schemas, messages = trajectory_messages(b, role="customer", stage="response", theory=True)
        from agent_simulation.agents.agentic import teacher_output
        expected = teacher_output(schemas, customer_ekb=True)
        seen = []
        def handler(request):
            body = json.loads(request.content);seen.append(body)
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                "message": {"role": "assistant", "content": "{}"}}]})
        from contextlib import closing
        with patch("agent_simulation.inference.requests.prompt_tokens", return_value=[1]), closing(
                AgenticClient("http://fixture/v1", "fixture", TOKENIZER,
                              transport=httpx.MockTransport(handler))) as client:
            client.complete({"messages": messages, "tools": [], "final_schema": expected}, call_id="offline")
        payload = seen[0]
        self.assertEqual(payload["response_format"]["json_schema"]["strict"], True)
        self.assertIn("action", payload["response_format"]["json_schema"]["schema"]["properties"])
