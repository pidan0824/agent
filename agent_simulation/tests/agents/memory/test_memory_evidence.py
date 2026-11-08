"""Test evidence lineage and memory revisions."""

from agent_simulation.agents.prompts.seller import seller_decision_messages

from agent_simulation.agents.memory.evidence import (
    eligible_evidence, experiences, reflection_profile, with_settled_offers,
)

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from agent_simulation.tests.fixtures import MarketFixture
from agent_simulation.tests.fixtures import ProfileCatalogFixture
import agent_simulation.agents.memory.reflection as manager
from agent_simulation.agents.memory.versions import active, memory_id, retire
from agent_simulation.agents.retrieval.similarity import related_memories


class MemoryEvidenceTests(MarketFixture, unittest.TestCase):
    def seed(self, count=30):
        state = self.runtime.snapshot()
        state["historical_cutoff_exclusive"] = self.data.cutoff = 100
        initial = state["initial_seller_memory"]["10"][0]
        state["initial_seller_memory"]["10"] = [
            {"time_index": i, "content": [{**deepcopy(initial["content"][0]), "time_index": i}]}
            for i in range(1, count+1)]
        with self.runtime.db:
            self.runtime.db.execute("UPDATE state SET value=?", (json.dumps(state),))
        return state

    def update(self, callback):
        return self.runtime.long_term_memory.update("seller", 10, generate=callback, model_id="test-only",
                                          embedder=lambda text: [1., 0.], encoder_id="test-only")

    def test_customer_observed_snapshots_are_filtered_at_all_memory_entry_points(self):
        state = self.runtime.snapshot()
        event = state["customers"]["7"]["observed_history"][0]
        offer = {"item_id": 1, "price_raw": 100, "is_clicked": True,
                 "price_provenance": {"internal": "source"}, "private_metric": 99}
        event["clicked_items"] = [offer]
        before = deepcopy(state)
        expected = [{"item_id": 1, "price_raw": 100}]

        experience = experiences(state, "customer", 7)[0]
        self.assertEqual(experience["content"]["clicked_items"], expected)
        self.assertTrue(experience["source_id"].endswith(f':session:{event["session_id"]}'))

        metadata = {"owner": "customer:7", "event_count": 1, "available_step": 0,
                    "historical_cutoff_exclusive": state["historical_cutoff_exclusive"]}
        sources = [
            {**metadata, "kind": "historical_segment",
             "content": {"clock": "observed_time_index", "records": [event]}},
            {**metadata, "kind": "query_customer_history",
             "content": {"result": {"observed_history": [event], "simulation_history": []}}},
        ]
        segment, query = eligible_evidence(sources, "customer:7", 1, state)
        self.assertEqual(segment["content"]["records"][0]["clicked_items"], expected)
        self.assertEqual(query["content"]["result"]["observed_history"][0]["clicked_items"], expected)
        self.assertEqual(state, before)

    def test_customer_settled_offers_use_only_their_own_request(self):
        displays = [self.runtime.execute("customer", actor, "search_products", {"query": "杯子"})
                    for actor in (7, 8)]
        self.assertNotEqual(displays[0]["request_id"], displays[1]["request_id"])
        for actor, display in zip((7, 8), displays, strict=True):
            clicked = [display["displayed_items"][0]["item_id"]] if actor == 7 else []
            self.runtime.execute("customer", actor, "submit_response",
                                 {"clicked_item_ids": clicked, "purchased_item_ids": []})
        state = self.runtime.snapshot()
        for actor, display in zip((7, 8), displays, strict=True):
            event = experiences(state, "customer", actor)[-1]
            enriched, = with_settled_offers([event], state["feedback"])
            self.assertEqual([offer["item_id"] for offer in enriched["displayed_offers"]],
                             [item["item_id"] for item in display["displayed_items"]])
            self.assertEqual(sum(offer["click_records"] for offer in enriched["displayed_offers"]),
                             int(actor == 7))

    def test_relevance_survives_age_and_negative_similarity_is_not_promoted(self):
        ranked = related_memories([1, 0], ["old", "new"], [[1, 0], [.1, .99]], [0, 100], 100)
        self.assertEqual(ranked, ["old", "new"])
        ranked = related_memories([1, 0], ["old_bad", "new_less_bad"],
                                  [[-1, 0], [-.1, .99]], [0, 100], 100)
        self.assertEqual(ranked, ["new_less_bad", "old_bad"])
        contents = [{"id": i} for i in range(7)]
        newest = related_memories([1, 0], contents, [[1, 0]] * 7, list(range(7)), 7)
        self.assertEqual([entry["id"] for entry in newest], [6, 5, 4, 3, 2])
        newest[0]["id"] = 99
        self.assertEqual(contents[6]["id"], 6)
        with self.assertRaisesRegex(ValueError, "future"):
            related_memories([1, 0], [{}], [[1, 0]], [8], 7)

    def test_changed_market_during_generation_aborts_memory_commit(self):
        self.seed(10)
        before = self.runtime.snapshot()
        def concurrent_update(_):
            changed = deepcopy(before)
            changed["simulation_step"] += 1
            with self.runtime.db:
                self.runtime.db.execute("UPDATE state SET value=? WHERE id=1", (json.dumps(changed),))
                self.runtime.db.execute("UPDATE state SET value=? WHERE id=1", (json.dumps(before),))
            return {"action": "no_update", "content": "", "source_ids": []}
        with self.assertRaisesRegex(ValueError, "State changed during generation") as caught:
            self.update(concurrent_update)
        from agent_simulation.agents.memory.validation import ReflectionDraftError
        self.assertNotIsInstance(caught.exception, ReflectionDraftError)
        self.assertEqual(self.runtime.snapshot(), before)

    def test_reflection_profile_does_not_leak_latest_reference_into_earlier_batch(self):
        profile = {"products": [{"item_id": 1, "item_title": "保温杯",
                                 "reference_price_raw": 999, "reference_price_time_index": 99}]}
        selected = [{"content": [{"item_id": 1, "time_index": 2, "price_raw": 100}]}]
        result = reflection_profile("seller", profile, selected)
        self.assertEqual(result["products"], [{"item_id": 1, "item_title": "保温杯"}])
        self.assertEqual(profile["products"][0]["reference_price_raw"], 999)

    def test_authorized_inputs_and_queries_are_citable_without_creating_events(self):
        self.seed(10)
        initial = self.runtime.snapshot()
        context, _ = self.runtime.inputs.seller(10, detail_item_ids=[1])
        args = {"item_ids": [2], "before_time_index": 5, "limit": 10}
        public = self.runtime.execute("seller", 10, "query_public_products", args)
        self.runtime.execute("seller", 10, "query_business",
                             {"item_ids": [1], "before_time_index": 4, "limit": 10})
        self.runtime.execute("seller", 10, "query_market_position",
                             {"item_id": 1, "before_time_index": 5})
        self.assertEqual(initial, self.runtime.snapshot())
        entries = self.runtime.long_term_memory.evidence.eligible("seller:10", 10, 0)
        kinds = {e["kind"] for e in entries}
        self.assertTrue({"market_position", "query_public_products", "query_business",
                         "query_market_position"} <= kinds)
        saved_public = next(e for e in entries if e["kind"] == "query_public_products")
        self.assertEqual(saved_public["content"]["result"], public)
        self.assertNotIn("purchase_records", json.dumps(public))
        self.assertEqual(self.runtime.long_term_memory.evidence.eligible("seller:20", 10, 0), [])
        before_count = len(entries)
        with self.assertRaises(PermissionError):
            self.runtime.execute("seller", 10, "query_business",
                                 {"item_ids": [2], "before_time_index": 5, "limit": 10})
        self.assertEqual(len(self.runtime.long_term_memory.evidence.eligible("seller:10", 10, 0)), before_count)
        captured = []
        def summary(request):
            payload = json.loads(request[1]["content"])
            captured.append(payload)
            return {"action": "add", "content": "TEST ONLY: authorized context connected.",
                    "source_ids": payload["new_evidence_ids"][:1] +
                                  [e["source_id"] for e in payload["supplemental_evidence"][:1]]}
        result = self.update(summary)
        self.assertEqual(result["status"], "saved")
        self.assertTrue(captured[0]["supplemental_evidence"])
        visible = self.runtime.long_term_memory.context("seller", 10, "pricing")
        self.assertTrue(any(e["origin"] == "context" for e in visible[0]["evidence_times"]))
        self.assertEqual(initial["feedback"], self.runtime.snapshot()["feedback"])

    def test_later_context_cannot_leak_into_earlier_initialization_batch(self):
        self.seed(20)
        self.runtime.inputs.seller(10)
        captured = []
        def empty(request):
            captured.append(json.loads(request[1]["content"]))
            return {"action": "no_update", "content": "", "source_ids": []}
        self.update(empty)
        self.assertEqual(captured[0]["supplemental_evidence"], [])
        self.update(empty)
        self.assertTrue(captured[1]["supplemental_evidence"])
        self.assertTrue(all(e["event_count"] <= 20 for e in captured[1]["supplemental_evidence"]))

    def test_idle_ticks_do_not_make_later_evidence_visible_in_an_earlier_snapshot(self):
        state = self.runtime.snapshot()
        state["clock"] = {"time": 100}
        with self.runtime.db:
            self.runtime.long_term_memory.evidence.record("seller:10", state, 3, "market_position", {"price_raw": 100})
        self.assertEqual(self.runtime.long_term_memory.evidence.eligible("seller:10", 3, 0, logical_time=99), [])
        entries = self.runtime.long_term_memory.evidence.eligible("seller:10", 3, 0, logical_time=100)
        self.assertEqual(len(entries), 1)
        state["clock"]["time"] = 99
        with self.assertRaisesRegex(ValueError, "logical-time"):
            eligible_evidence(entries, "seller:10", 3, state)

    def test_old_public_quotes_exclude_private_submission_metadata(self):
        state = self.runtime.snapshot()
        old = {"price_raw": 85, "simulation_step": 42, "source": "simulation:submitted_price"}
        with self.runtime.db:
            self.runtime.long_term_memory.evidence.record("seller:10", state, 0, "query_public_products",
                {"result": {"products": [{"item_id": 2, "simulation_current_price": old}]}})
        stored = self.runtime.long_term_memory.evidence.eligible("seller:10", 0, 0)
        visible = eligible_evidence(stored, "seller:10", 0, state)
        self.assertEqual(visible[0]["content"]["result"]["products"][0]["simulation_current_price"],
                         {"price_raw": 85})
        self.assertEqual(stored[0]["content"]["result"]["products"][0]["simulation_current_price"], old)

    def test_observed_context_rolls_back_with_the_market_transaction(self):
        before = self.runtime.freeze()
        context, _ = self.runtime.inputs.seller(10, observe=False)
        memory = self.runtime.long_term_memory
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            with self.runtime.store.transaction():
                state = self.runtime.snapshot()
                state["simulation_step"] += 1
                self.runtime.store.write(state)
                memory.observe_context("seller", 10, context, state)
                self.assertTrue(memory.evidence.eligible("seller:10", 3, 1))
                raise RuntimeError("rollback")

        self.assertEqual(self.runtime.freeze(), before)
        self.assertEqual(memory.evidence.eligible("seller:10", 3, 1), [])

    def test_identical_evidence_keeps_first_availability_across_repeated_reads(self):
        state = self.runtime.snapshot()
        store = self.runtime.long_term_memory.evidence
        with self.runtime.db:
            first = store.record("seller:10", state, 3, "query_business", {"price_raw": 100})
            state["simulation_step"] = 10
            second = store.record("seller:10", state, 13, "query_business", {"price_raw": 100})
        self.assertEqual(first, second)
        rows = store.eligible("seller:10", 3, 0, source_ids=[first])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_count"], 3)

    def test_direct_summary_rejects_inconsistent_operations_without_consuming_events(self):
        self.seed(10)
        before = self.runtime.snapshot()

        for fields, error in (({"supersedes": []}, "empty supersedes"),
                              ({"action": "no_update"}, "does not match content"),
                              ({"action": "revise"}, "does not match supersedes")):
            def generate(request):
                payload = json.loads(request[1]["content"])
                return {"action": "add", "content": "TEST ONLY: a proposed memory.",
                        "source_ids": payload["new_evidence_ids"][:1], **fields}

            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, error):
                self.update(generate)
            self.assertEqual(self.runtime.snapshot(), before)

    def test_invalid_summary_embedding_preserves_prior_state(self):
        self.seed(10)
        before = self.runtime.snapshot()

        def generate(request):
            payload = json.loads(request[1]["content"])
            return {"action": "add", "content": "TEST ONLY: grounded observation.",
                    "source_ids": payload["new_evidence_ids"][:1]}

        with self.assertRaisesRegex(ValueError, "summary embedding"):
            self.runtime.long_term_memory.update(
                "seller", 10, generate=generate, model_id="test-only",
                embedder=lambda text: ([float("nan")] if text == "TEST ONLY: grounded observation." else [1., 0.]),
                encoder_id="test-only")
        self.assertEqual(self.runtime.snapshot(), before)

    def test_replace_then_retire_preserves_versions_and_original_facts(self):
        self.seed(30)
        original = experiences(self.runtime.snapshot(), "seller", 10)
        def draft(content, supersedes=()):
            def generate(request):
                p = json.loads(request[1]["content"])
                return {"action": "retract" if not content else "revise" if supersedes else "add",
                        "content": content, "source_ids": p["new_evidence_ids"][:1] if content else [],
                        **({"supersedes": list(supersedes)} if supersedes else {})}
            return generate
        added = self.update(draft("TEST ONLY: initial hypothesis."))
        self.assertEqual(added["operation"], "add")
        first = added["saved"]
        self.assertNotIn("supersedes", first)
        self.assertNotIn("supersedes", added["raw_output"])
        revised = self.update(draft("TEST ONLY: corrected hypothesis.", [memory_id(first)]))
        self.assertEqual(revised["operation"], "revise")
        second = revised["saved"]
        visible = self.runtime.long_term_memory.context("seller", 10, "pricing")
        self.assertEqual([e["memory_id"] for e in visible], [memory_id(second)])
        self.assertEqual(len(self.runtime.snapshot()["long_term"]["seller:10"]), 2)
        result = self.update(draft("", [memory_id(second)]))
        self.assertEqual(result["status"], "retired")
        self.assertEqual(result["operation"], "retract")
        self.assertEqual(self.runtime.long_term_memory.context("seller", 10, "pricing"), [])
        versions = self.runtime.snapshot()["long_term"]["seller:10"]
        self.assertEqual(versions[0]["content"], first["content"])
        self.assertEqual(versions[0]["replaced_by"], memory_id(second))
        self.assertIsNone(versions[1]["replaced_by"])
        self.assertEqual(experiences(self.runtime.snapshot(), "seller", 10), original)

    def test_existing_version_prevents_replaying_its_batch(self):
        self.seed(10)
        self.update(lambda request: {"action": "add",
            "content": "Initial hypothesis.",
            "source_ids": json.loads(request[1]["content"])["new_evidence_ids"][:1],
        })
        state = self.runtime.snapshot()
        state["memory_processed"]["seller:10"] = 0
        with self.runtime.db:
            self.runtime.db.execute("UPDATE state SET value=?", (json.dumps(state),))

        with self.assertRaisesRegex(ValueError, "Duplicate long-term memory ID"):
            self.update(lambda _: self.fail("Duplicate batch reached the generator"))
        self.assertEqual(self.runtime.snapshot(), state)

    def test_retirement_failure_does_not_publish_a_replacement_or_consume_events(self):
        state = self.seed(20)
        profile = {"seller_id": 10, "products": [p for p in state["products"].values()
                                                 if p["seller_id"] == 10]}

        def draft(request, content, **fields):
            new_ids = json.loads(request[1]["content"])["new_evidence_ids"]
            return {"action": "revise" if "supersedes" in fields else "add",
                    "content": content, "source_ids": new_ids[:1], **fields}

        args = (lambda text: [1., 0.], "fixture", "fixture")
        first = manager.update(state, "seller", 10, profile,
                               lambda request: draft(request, "Initial observation."), *args)
        before = deepcopy(state)
        retire = manager.retire

        def fail_after_staging(*args, **kwargs):
            retire(*args, **kwargs)
            raise RuntimeError("retirement failed")

        with patch.object(manager, "retire", side_effect=fail_after_staging):
            with self.assertRaisesRegex(RuntimeError, "retirement failed"):
                manager.update(state, "seller", 10, profile,
                               lambda request: draft(request, "Corrected observation.",
                                                     supersedes=[memory_id(first["saved"])]), *args)
        self.assertEqual(state, before)

    def test_retraction_checks_target_and_records_review_context_without_fake_citations(self):
        state = self.seed(10)
        selected = experiences(state, "seller", 10)
        with self.assertRaisesRegex(ValueError, "supplied active"):
            manager.read_draft({"action": "retract", "content": "", "source_ids": [selected[0]["source_id"]],
                                     "supersedes": ["seller:20:10"]}, selected, [])
        first = self.update(lambda request: {"action": "add", "content": "TEST ONLY: keep this memory.",
                            "source_ids": json.loads(request[1]["content"])["new_evidence_ids"][:1]})["saved"]
        self.seed(20)
        result = self.update(lambda request: {"action": "retract", "content": "", "source_ids": [],
                                             "supersedes": [memory_id(first)]})
        self.assertEqual(result['operation'], 'retract')
        retired = self.runtime.snapshot()['long_term']['seller:10'][0]
        self.assertEqual(retired['retirement_source_ids'], [])
        self.assertEqual(len(retired['retirement_context_source_ids']), 10)
        self.assertEqual(self.runtime.long_term_memory.context('seller', 10, 'pricing'), [])

    def test_customer_history_read_is_private_and_survives_restart(self):
        self.runtime.catalog = ProfileCatalogFixture(self.runtime)
        state = self.runtime.snapshot()
        result = self.runtime.execute("customer", 7, "query_customer_history",
                                      {"before_time_index": 10, "limit": 10})
        sources = self.runtime.long_term_memory.evidence.eligible("customer:7", 1, 0)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["content"]["result"], result)
        self.assertEqual(self.runtime.long_term_memory.evidence.eligible("customer:8", 1, 0), [])
        self.assertEqual(self.runtime.snapshot(), state)
        from agent_simulation.environment.memory import EvidenceStore
        self.assertEqual(EvidenceStore(self.runtime.db).eligible("customer:7", 1, 0), sources)

    def test_old_factual_segment_is_preserved_and_repeated_reads_are_deduplicated(self):
        state = self.runtime.snapshot()
        state["history_per_item"] = 1
        with self.runtime.db:
            self.runtime.db.execute("UPDATE state SET value=?", (json.dumps(state),))
        context, _ = self.runtime.inputs.seller(10, detail_item_ids=[1])
        first = self.runtime.long_term_memory.evidence.eligible("seller:10", 3, 0)
        retrieved = context["retrieved_context"]["history"]["segments"]
        saved = [e["content"] for e in first if e["kind"] == "historical_segment"]
        self.assertCountEqual(saved, retrieved)
        segment = next(e for e in first if e["kind"] == "historical_segment" and e["content"]["item_id"] == 1)
        self.assertEqual(segment["content"]["records"][0]["index"], 2)
        self.runtime.inputs.seller(10, detail_item_ids=[1])
        self.assertEqual(self.runtime.long_term_memory.evidence.eligible("seller:10", 3, 0), first)
        self.assertEqual(self.runtime.snapshot(), state)

    def test_seller_messages_link_cited_row_to_retrieved_history(self):
        state = self.seed(10)
        state["history_per_item"] = 1
        with self.runtime.db:
            self.runtime.db.execute("UPDATE state SET value=?", (json.dumps(state),))
        self.runtime.inputs.seller(10, detail_item_ids=[1])

        cited = {}
        def summarize(request):
            payload = json.loads(request[1]["content"])
            segment = next(e for e in payload["supplemental_evidence"] if e["kind"] == "historical_segment")
            cited.update(segment['content'])
            return {"action": "add", "content": "Test summary based on a retrieved fact.",
                    "source_ids": [payload["new_evidence_ids"][0], segment["source_id"]]}

        self.update(summarize)
        ctx, _ = self.runtime.inputs.seller(10, detail_item_ids=[1])
        memory = ctx["long_term_memory"][0]
        position = next(e for e in memory["evidence_times"] if e.get("kind") == "historical_segment")
        refs = position["same_observations_in_retrieved_history"]
        actual = set()
        for ref in refs:
            segment = ctx["retrieved_context"]["history"]["segments"][ref["segment_number"] - 1]
            row = segment['records'][ref['record_number'] - 1]
            actual.add((segment['clock'], segment['item_id'], row['index']))
        expected = {(cited['clock'], cited['item_id'], row['index']) for row in cited['records']}
        self.assertTrue(actual)
        self.assertEqual(actual, expected)
        self.assertIn("supporting_evidence_refs",
                      seller_decision_messages(self.runtime.inputs.seller(10, detail_item_ids=[1])[0])[1]["content"])

    def test_customer_same_index_different_sessions_and_clocks(self):
        from agent_simulation.agents.retrieval.long_term import retrieve_long_term_memory
        from agent_simulation.agents.prompts.customer import customer_decision_messages
        state = self.runtime.snapshot()
        event = state["customers"]["7"]["observed_history"][0]
        second = dict(event, session_id=6)
        simulated = dict(query="水杯", request_id=1, settled_step=2, clicked_items=[], purchased_items=[])
        state["customers"]["7"]["observed_history"].append(second)
        state["customers"]["7"]["simulation_history"].append(simulated)
        state["simulation_step"] = 2
        sources = experiences(state, "customer", 7)
        state["long_term"]["customer:7"] = [{
            "owner": "customer:7", "generated_step": 2, "event_count": 3,
            "embedding_model": "test", "embedding": [1., 0.], "content": "Test shopping habit.",
            "source_ids": [sources[0]["source_id"], sources[2]["source_id"]]}]
        records = [("observed_time_index", second), ("simulation_settled_step", simulated),
                   ("observed_time_index", event)]
        ctx = {"query": "水杯", "displayed_items": [],
               "retrieved_context": {"history": {"segments": [
                   {"clock": clock, "records": [row]} for clock, row in records]}}}
        ctx["long_term_memory"] = retrieve_long_term_memory(state, "customer", 7, ctx, lambda _: [1., 0.], "test")
        refs = [e["same_observations_in_retrieved_history"] for e in ctx["long_term_memory"][0]["evidence_times"]]
        self.assertEqual(refs, [[{"segment_number": 3, "record_number": 1}],
                                [{"segment_number": 2, "record_number": 1}]])
        self.assertIn("same_observations_in_retrieved_history",
                      customer_decision_messages(ctx, response=True)[1]["content"])

    def test_seller_exposure_match_uses_cited_counts_and_keeps_uncounted_slots(self):
        from agent_simulation.agents.retrieval.long_term import rank_by_seller_evidence
        rows = [{"item_id": 1, "time_index": i, "price_raw": 880, "exposure_records": n,
                 "purchase_records": 0} for i, n in enumerate((1, 100))]
        sources = {str(i): {"origin": "observed", "content": [row]} for i, row in enumerate(rows)}
        sources["shop"] = {"kind": "market_position", "content": {}}
        # A better count match can break a tie between adjacent BGE ranks.
        entries = [{"source_ids": [s]} for s in ("1", "0", "shop")]
        context = {"short_term_memory": [{"content": [rows[0]]}]}
        self.assertEqual(rank_by_seller_evidence(entries, context, sources),
                         [entries[1], entries[0], entries[2]])

        # The intervening unscored memory still occupies BGE rank 2.
        separated = [entries[0], entries[2], entries[1]]
        self.assertEqual(rank_by_seller_evidence(separated, context, sources), separated)

    def test_private_peer_traffic_in_stored_market_evidence_fails_on_read(self):
        state = self.runtime.snapshot()
        old = {"simulation": {"cumulative": {"click_records": {"own_value": 0, "median": 2}}}}
        for kind, content in (
            ("market_position", {"market_position": old}),
            ("query_market_position", {"result": old}),
        ):
            entry = {"kind": kind, "content": content, "owner": "seller:10",
                     "historical_cutoff_exclusive": state["historical_cutoff_exclusive"],
                     "available_step": 0, "event_count": 0}
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "legacy peer traffic"):
                eligible_evidence([entry], "seller:10", 0, state)
        with self.assertRaisesRegex(ValueError, "Unsupported memory evidence kind"):
            eligible_evidence([{**entry, "kind": "retired_tool"}], "seller:10", 0, state)

    def test_customer_can_revise_habits_using_new_evidence(self):
        state = self.runtime.snapshot()
        state["historical_cutoff_exclusive"] = self.data.cutoff = 100
        first = state["customers"]["7"]["observed_history"][0]
        state["customers"]["7"]["observed_history"] = [
            {**deepcopy(first), "session_id": i, "time_index": i} for i in range(1, 21)]
        with self.runtime.db:
            self.runtime.db.execute("UPDATE state SET value=?", (json.dumps(state),))
        self.runtime.catalog = ProfileCatalogFixture(self.runtime)
        def generate(request):
            payload = json.loads(request[1]["content"])
            previous = payload["previous_long_term_memory"]
            return {"action": "revise" if previous else "add", "content": "TEST ONLY: revised habit." if previous else "TEST ONLY: initial habit.",
                    "source_ids": payload["new_evidence_ids"][:1],
                    **({"supersedes": [p["memory_id"] for p in previous]} if previous else {})}
        for _ in range(2):
            self.runtime.long_term_memory.update("customer", 7, generate=generate, model_id="test-only",
                                       embedder=lambda text: [1., 0.], encoder_id="test-only")
        visible = self.runtime.long_term_memory.context("customer", 7, "habits")
        self.assertEqual([e["content"] for e in visible], ["TEST ONLY: revised habit."])
        self.assertEqual(state["customers"], self.runtime.snapshot()["customers"])


class MemoryVersionTests(unittest.TestCase):
    def test_retirement_keeps_the_old_version_visible_until_all_boundaries(self):
        old = {"owner": "seller:10", "event_count": 10, "generated_step": 3}
        new = {**old, "event_count": 20, "generated_step": 7, "generated_logical_time": 25}
        self.assertEqual(active([old], 10, 3, logical_time=0), [old])

        updated = retire([old, new], [memory_id(old)], count=20, step=7, logical_time=25,
                         replacement=memory_id(new), source_ids=["source"])
        for boundary in ((19, 7, 25), (20, 6, 25), (20, 7, 24)):
            with self.subTest(boundary=boundary):
                self.assertEqual(active(updated, *boundary), [updated[0]])
        self.assertEqual(active(updated, 20, 7, 25), [new])
        self.assertEqual(active(updated, 20, 7), [new])
        self.assertNotIn("retired_at_event_count", old)

    def test_retirement_rejects_inconsistent_targets_and_replacements(self):
        old = {"owner": "seller:10", "event_count": 10, "generated_step": 3}
        new = {**old, "event_count": 20, "generated_step": 7, "generated_logical_time": 25}
        old_id, new_id = memory_id(old), memory_id(new)
        retired = retire([old], [old_id], count=15, step=5, replacement=None, source_ids=[])[0]
        future = {**old, "generated_logical_time": 26}
        cases = (
            ([old], [old_id, "seller:10:5"], None, "Retirement target does not exist"),
            ([old, dict(old)], [old_id], None, "Duplicate long-term memory ID"),
            ([retired], [old_id], None, "already retired"),
            ([future], [old_id], None, "not an earlier visible memory"),
            ([old, new], [new_id], new_id, "not an earlier visible memory"),
            ([old], [old_id], new_id, "Replacement memory does not exist"),
            ([old, {**new, "generated_step": 6}], [old_id], new_id, "must be new in this batch"),
        )
        for entries, targets, replacement, message in cases:
            before = deepcopy(entries)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                retire(entries, targets, count=20, step=7, logical_time=25,
                       replacement=replacement, source_ids=[])
            self.assertEqual(entries, before)


if __name__ == "__main__":
    unittest.main()
