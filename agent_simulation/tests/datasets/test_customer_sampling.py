import json
import random
import re
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
import duckdb

from agent_simulation.datasets.customer_candidate_sampling import SampledCandidateBuilder
from agent_simulation.datasets.customer_scenes import prepare_scenes


class Encoder:
    identity="fixture-encoder"
    def encode_many(self,texts,**kwargs):
        return [np.array([float((int(re.search(r"\d+",text)[0]) if re.search(r"\d+",text) else 1)%7),1.,.5],dtype=np.float32) for text in texts]


def request(target=None):
    target=target or {"clicked_item_ids":[10,11],"purchased_item_ids":[11]}
    query={"profile":{"user_id":1},"history_cutoff_exclusive":20,
           "short_term_memory":{"search_history":[]},"long_term_memory":[]}
    return {"session_id":100,"assigned_split":"train","query_input":query,
            "query_target":{"query":"target hidden"},"response_input":{**deepcopy(query),"query":"item10",
            "displayed_items":[{"item_id":i,"item_title":f"item{i}","price_raw":999} for i in range(10,22)]},
            "response_target":target}


def pool():
    products=[{"item_id":i,"item_title":f"item{i}","first_seen_time_index":0} for i in range(1,41)]
    interactions=[{"user_id":u,"item_id":i,"time_index":i} for u in range(1,4) for i in range(1,4)]
    return products,interactions


class AuthorCandidateTests(unittest.TestCase):
    def builder(self,**kwargs):
        products,interactions=pool()
        return SampledCandidateBuilder(products,interactions,Encoder(),seed=8,**kwargs)

    def test_multi_targets_history_filter_and_original_functions(self):
        original=request();before=deepcopy(original)
        result=self.builder(min_items=5,max_items=8).build(original,target_item_id=10)
        ids=result["candidate"]["candidate_item_ids"]
        self.assertIn(10,ids)
        self.assertNotIn(11,ids)
        self.assertFalse({1,2,3}&set(ids))
        self.assertEqual(len(ids),len(set(ids)))
        self.assertTrue(result["candidate"]["outcome_conditioned"])
        self.assertEqual(result["request"]["response_target"],{"clicked_item_ids":[10],"purchased_item_ids":[]})
        self.assertEqual(result["request"]["query_input"],original["query_input"])
        self.assertEqual(original,before)
        self.assertEqual(json.loads(json.dumps(result,allow_nan=False))["candidate"]["candidate_item_ids"],ids)

    def test_collaborative_shortlist_scores_match_global_formula_with_repeated_outside_history(self):
        from unittest.mock import patch
        import torch
        from agent_simulation.vendor.usermirrorer import sampler as original
        products, interactions = pool()
        interactions.extend([
            {"user_id": 1, "item_id": 2, "time_index": 4},
            {"user_id": 1, "item_id": 12, "time_index": 5},
            {"user_id": 1, "item_id": 4, "time_index": 30},
        ])
        builder = SampledCandidateBuilder(products, interactions, None,
            methods=("collaborative",), min_items=5, max_items=8, fit_cutoff=10)
        global_map = {p["item_id"]: i for i, p in enumerate(products)}
        positions = torch.arange(len(products), dtype=torch.float32)
        cf = torch.stack((positions / 100, torch.sin(positions / 3) / 5,
                          torch.cos(positions / 5) / 3), dim=1)
        builder._cf_cache[10] = (global_map, cf)
        # Item 2 occurs twice; 1, 2, 3 are outside this exposure; future item 4 is absent.
        history = torch.zeros((1, len(products)))
        history[0, [global_map[i] for i in (1, 2, 3, 12)]] = 1
        history[0, global_map[2]] = 2
        candidate_indices = [global_map[i] for i in range(10, 22) if i != 11]
        expected = torch.softmax(original.collaborative_predict(cf, history)[:, candidate_indices], dim=1)
        with patch("agent_simulation.datasets.customer_candidate_sampling.original.topk_sampling",
                   wraps=original.topk_sampling) as topk:
            builder.build(request(), target_item_id=10)
        topk.assert_called_once()
        torch.testing.assert_close(topk.call_args.args[0], expected, rtol=1e-6, atol=1e-7)
        deduplicated = history.clamp(max=1)
        exposure_only = history.clone()
        exposure_only[0, [global_map[i] for i in (1, 2, 3)]] = 0
        for wrong in (deduplicated, exposure_only):
            self.assertFalse(torch.allclose(expected,
                torch.softmax(original.collaborative_predict(cf, wrong)[:, candidate_indices], dim=1)))

    def test_active_cf_columns_match_global_nonzero_spectrum_and_unseen_scores_are_zero(self):
        from unittest.mock import patch
        import torch
        from agent_simulation.vendor.usermirrorer import sampler as original
        products, _ = pool()
        # Positive dense support plus distinct diagonal counts gives a separated,
        # nonzero retained spectrum; the remaining catalog columns are all zero.
        fit = [{"user_id": user, "item_id": item, "time_index": repeat + 1}
               for user in range(1, 13) for item in range(1, 13)
               for repeat in range(1 + (user if user == item else 0))]
        interactions = fit + [{"user_id": 1, "item_id": 1, "time_index": time} for time in (150, 151)]
        source = request({"clicked_item_ids": [10], "purchased_item_ids": []})
        source["query_input"]["history_cutoff_exclusive"] = 200
        source["response_input"]["history_cutoff_exclusive"] = 200
        indices = torch.tensor([[r["user_id"] - 1 for r in fit], [r["item_id"] - 1 for r in fit]])
        full_matrix = torch.sparse_coo_tensor(indices, torch.ones(len(fit)), (32, 40)).coalesce()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            full_cf = original.get_spectral_info(full_matrix, 8)
        history = torch.zeros((1, 40))
        for row in interactions:
            if row["user_id"] == 1:
                history[0, row["item_id"] - 1] += 1
        expected = original.collaborative_predict(full_cf, history)[:, 9:21]
        builder = SampledCandidateBuilder(products, interactions, None, methods=("collaborative",),
                                           fit_cutoff=100, min_items=5, max_items=8)
        with patch("agent_simulation.datasets.customer_candidate_sampling.torch.softmax",
                   wraps=torch.softmax) as softmax:
            builder.build(source, target_item_id=10)
        softmax.assert_called_once()
        scores = softmax.call_args.args[0]
        self.assertEqual(set(builder._cf_cache[100][0]), set(range(1, 13)))
        torch.testing.assert_close(scores, expected, rtol=1e-4, atol=2e-4)
        self.assertTrue(torch.equal(scores[:, 3:], torch.zeros_like(scores[:, 3:])))
        exposure_only = history.clone()
        exposure_only[:, :9] = 0
        self.assertFalse(torch.allclose(expected,
            original.collaborative_predict(full_cf, exposure_only)[:, 9:21], rtol=1e-4, atol=2e-4))

    def test_empty_cf_fit_skips_svd_and_uses_uniform_probabilities(self):
        from unittest.mock import patch
        import torch
        from agent_simulation.vendor.usermirrorer import sampler as original
        builder = self.builder(methods=("collaborative",), fit_cutoff=1)
        with patch("agent_simulation.datasets.customer_candidate_sampling.original.get_spectral_info",
                   side_effect=AssertionError("Empty fit must not run SVD")), \
             patch("agent_simulation.datasets.customer_candidate_sampling.original.topk_sampling",
                   wraps=original.topk_sampling) as topk:
            builder.build(request(), target_item_id=10)
        topk.assert_called_once()
        torch.testing.assert_close(topk.call_args.args[0], torch.full((1, 11), 1 / 11))
        self.assertEqual(builder._cf_cache[1][0], {})

    def test_no_click_keeps_observed_negatives_and_inserts_nothing(self):
        result=self.builder(min_items=5,max_items=8).build(request({"clicked_item_ids":[],"purchased_item_ids":[]}))
        self.assertEqual(result["candidate"]["positive_count"],0)
        self.assertTrue(set(result["candidate"]["candidate_item_ids"])<=set(range(10,22)))
        self.assertEqual(result["request"]["response_target"],{"clicked_item_ids":[],"purchased_item_ids":[]})

    def test_seed_is_repeatable_and_does_not_change_caller_rng(self):
        builder=self.builder();state=random.getstate()
        a=builder.build(request(),target_item_id=10);b=builder.build(request(),target_item_id=10)
        self.assertEqual(a["candidate"]["candidate_item_ids"],b["candidate"]["candidate_item_ids"])
        self.assertEqual(random.getstate(),state)

    def test_many_positives_split_and_never_become_distractors(self):
        source=request({"clicked_item_ids":[10,11,12],"purchased_item_ids":[12,13]})
        scenes=self.builder(min_items=5,max_items=5).build_all(source)
        self.assertEqual(len(scenes),4)
        self.assertEqual(len({s["candidate"]["scene_id"] for s in scenes}),4)
        for scene in scenes:
            meta=scene["candidate"]; item=meta["target_item_id"]
            self.assertEqual(set(meta["candidate_item_ids"]) & {10,11,12,13},{item})
            self.assertEqual(meta["positive_count"],1)
            self.assertEqual(meta["distractor_count"],4)
            self.assertEqual(scene["request"]["query_input"],source["query_input"])
        self.assertEqual(scenes[-1]["request"]["response_target"],
                         {"clicked_item_ids":[],"purchased_item_ids":[13]})

    def test_short_exposure_pool_preserves_supervision_without_fabricating_candidates(self):
        scene = self.builder(min_items=40,max_items=40).build(request(),target_item_id=10)
        self.assertTrue(scene["candidate"]["short_pool"])
        self.assertEqual(set(scene["candidate"]["candidate_item_ids"]), set(range(10,22)) - {11})
        self.assertEqual(scene["request"]["response_target"], {"clicked_item_ids":[10],"purchased_item_ids":[]})

    def test_catalog_preparation_uses_shared_pool_and_uniform_prior_prices(self):
        c=duckdb.connect()
        self.addCleanup(c.close)
        c.execute("CREATE TABLE products(item_id BIGINT,item_title VARCHAR)")
        c.executemany("INSERT INTO products VALUES (?,?)",[(i,f"item{i}") for i in range(1,41)])
        c.execute("CREATE TABLE requests(session_id BIGINT,user_id BIGINT,time_index BIGINT,split VARCHAR,impressed_item_ids BIGINT[],clicked_item_ids BIGINT[],purchased_item_ids BIGINT[])")
        c.executemany("INSERT INTO requests VALUES (?,?,?,?,?,?,?)",[(i,1,i,"train",list(range(1,41)),[i],[]) for i in range(1,4)])
        c.execute("CREATE TABLE ranking_observations(session_id BIGINT,item_id BIGINT,price_raw DOUBLE)")
        c.executemany("INSERT INTO ranking_observations VALUES (?,?,?)",[(3,i,50.) for i in range(1,41)])
        from unittest.mock import patch
        with patch("agent_simulation.datasets.customer_scenes.SampledCandidateBuilder",
                   wraps=SampledCandidateBuilder) as builder:
            scenes,skipped=prepare_scenes(SimpleNamespace(connection=c),[request()],Encoder(),min_items=8,max_items=8)
        self.assertEqual({p["item_id"] for p in builder.call_args.args[0]},
                         {1, 2, 3} | set(range(10, 22)))
        self.assertFalse(skipped)
        self.assertEqual(len(scenes),1)
        self.assertEqual(len(scenes[0]["response_scenes"]),2)
        self.assertEqual(scenes[0]["response_scenes"][0]["candidate"]["pool_size"],11)
        self.assertTrue(all(p["price_raw"]==50. for child in scenes[0]["response_scenes"] for p in child["request"]["response_input"]["displayed_items"]))
        failed, skipped = prepare_scenes(SimpleNamespace(connection=c), [request()], Encoder(), min_history=99)
        self.assertEqual(len(failed), 1)
        self.assertTrue(failed[0]["query_enabled"])
        self.assertEqual(failed[0]["response_scenes"], [])
        self.assertEqual(len(skipped), 2)

    def test_length_filter_uses_generation_messages_without_inference(self):
        from agent_simulation.datasets.customer_scenes import filter_lengths
        from agent_simulation.tests.inference.test_agentic import Tokenizer
        scene={"request":request(),"query_enabled":True,"response_scenes":self.builder().build_all(request())}
        kept,skipped=filter_lengths([scene],Tokenizer(),100000)
        self.assertEqual((len(kept),len(skipped)),(1,0))
        kept,skipped=filter_lengths([scene],Tokenizer(),1)
        self.assertEqual((len(kept),len(skipped)),(0,3))
        self.assertEqual(skipped[0]["reason"],"context_length")


    def test_query_once_and_no_click_once_in_generation_jobs(self):
        from agent_simulation.evaluation.customer_curation import generation_jobs, CurationTools
        source=request()
        bundle={"request":source,"query_enabled":True,"response_scenes":self.builder().build_all(source)}
        jobs=generation_jobs(source,bundle=bundle)
        self.assertEqual([j[0] for j in jobs],["query","response","response"])
        self.assertEqual([j[2]["response_target"] for j in jobs[1:]],
            [{"clicked_item_ids":[10],"purchased_item_ids":[]},
             {"clicked_item_ids":[11],"purchased_item_ids":[11]}])
        # Original targets and excluded-positive metadata must never enter model input.
        for stage,policy,adapted,candidate in jobs:
            visible=json.dumps(CurationTools(adapted,stage,mode="fixed").initial_context())
            self.assertNotIn("source_response_target",visible)
            self.assertNotIn("excluded_positive_ids",visible)
        bundle["response_scenes"]=[]
        self.assertEqual(len(generation_jobs(source,bundle=bundle)),1)
        empty=request({"clicked_item_ids":[],"purchased_item_ids":[]})
        bundle={"request":empty,"query_enabled":True,"response_scenes":self.builder().build_all(empty)}
        jobs=generation_jobs(empty,bundle=bundle)
        self.assertEqual([j[0] for j in jobs],["query","response"])
        self.assertEqual(jobs[1][2]["response_target"],empty["response_target"])


    def test_single_positive_scene_file_roundtrip(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from agent_simulation.data.common import sha256
        from agent_simulation.datasets.customer_scenes import read_scenes
        source=request()
        bundle={"request":source,"query_enabled":True,"response_scenes":self.builder().build_all(source)}
        with TemporaryDirectory() as temporary:
            directory=Path(temporary)
            (directory/"scenes.jsonl").write_text(json.dumps(bundle)+"\n")
            (directory/"manifest.json").write_text(json.dumps({
                "schema":"customer-single-positive-scenes-v2","split":"train",
                "scenes_sha256":sha256(directory/"scenes.jsonl")}))
            loaded, _ = read_scenes(directory,split="train")
            self.assertEqual(len(loaded),1)
            self.assertEqual(len(loaded[100]["response_scenes"]),2)
            self.assertEqual(loaded[100]["request"]["response_target"],source["response_target"])


class ExposurePoolTests(unittest.TestCase):
    def test_all_routes_keep_exposures_without_query_or_category_gates(self):
        products, interactions = pool()
        for p in products:
            p["category_level3_name"] = "sleepwear"
        products[12]["category_level3_name"] = "books"
        products[14].pop("category_level3_name")
        class MixedEncoder:
            identity = "mixed-relevance-fixture"
            def encode_many(self, texts, query=False):
                if query:
                    return [np.array([1., 0.])] * len(texts)
                return [np.array([1., 0.]) if int(re.search(r"\d+", text)[0]) in {10,11,12,13,15,40}
                        else np.array([0., 1.]) for text in texts]
        source = request()
        expected = set(range(10,22)) - {11}
        for method in ("random", "content", "collaborative"):
            encoder = MixedEncoder() if method == "content" else None
            builder = SampledCandidateBuilder(products, interactions, encoder,
                methods=(method,), min_items=11, max_items=11)
            child = builder.build(source, target_item_id=10)
            # Includes off-category, unknown-category and zero-similarity exposures.
            # Excludes both out-of-exposure items and the other actual positive.
            self.assertEqual(set(child["candidate"]["candidate_item_ids"]), expected)
            self.assertFalse(child["candidate"]["scope"]["query_category_filter"])
            self.assertEqual(child["candidate"]["scope"]["exposure_pool_size"], 12)
            self.assertEqual(child["request"]["query_input"], source["query_input"])

    def test_no_click_can_keep_all_exposures_without_an_encoder(self):
        products, interactions = pool()
        source = request({"clicked_item_ids": [], "purchased_item_ids": []})
        child = SampledCandidateBuilder(products, interactions, None, methods=("random",),
            min_items=12, max_items=12).build(source)
        self.assertEqual(set(child["candidate"]["candidate_item_ids"]), set(range(10,22)))
        self.assertEqual(child["request"]["response_target"], source["response_target"])

    def test_positive_insertion_can_reach_every_slot_including_last(self):
        products, interactions = pool()
        for size in (2, 3, 5):
            positions = set()
            for seed in range(64):
                child = SampledCandidateBuilder(products, interactions, None,
                    methods=("random",), min_items=size, max_items=size, seed=seed).build(
                    request({"clicked_item_ids": [10], "purchased_item_ids": [10]}))
                ids = child["candidate"]["candidate_item_ids"]
                positions.add(ids.index(10))
                self.assertEqual(len(ids), size)
                self.assertEqual(child["candidate"]["target_position_policy"], "scene-seeded-uniform-v1")
            self.assertEqual(positions, set(range(size)))

    def test_indexed_history_is_strictly_prior_and_does_not_scan_global_rows(self):
        products, interactions = pool()
        interactions += [{"user_id": user, "item_id": item, "time_index": time}
                         for user, item, time in ((1, 12, 19), (1, 13, 20), (1, 14, 21), (2, 15, 19))]
        builder = SampledCandidateBuilder(products, list(reversed(interactions)), None,
                                         methods=("random",), min_items=12, max_items=12)
        class NoGlobalScan(list):
            def __iter__(self):
                raise AssertionError("Per-scene history must use the user index")
        builder.interactions = NoGlobalScan(builder.interactions)
        child = builder.build(request({"clicked_item_ids": [10], "purchased_item_ids": []}))
        self.assertEqual(set(child["candidate"]["candidate_item_ids"]), set(range(10, 22)) - {12})
        self.assertEqual(child["candidate"]["prior_interactions"], 4)

    def test_lightweight_length_filter_matches_formal_prompt_and_exact_boundary(self):
        from agent_simulation.datasets.customer_scenes import filter_lengths, freeze_prices
        from agent_simulation.evaluation.customer_context import CurationTools
        from agent_simulation.evaluation.trajectories import trajectory_messages
        from agent_simulation.agents.agentic import initial_agentic_request, AgenticOptions
        from agent_simulation.inference.requests import prompt_tokens
        from agent_simulation.tests.inference.test_agentic import Tokenizer
        class CountingTokenizer(Tokenizer):
            def encode(self, text, **kwargs):
                return list(text)
        tokenizer = CountingTokenizer()
        source = request({"clicked_item_ids": [10], "purchased_item_ids": []})
        source["response_input"]["short_term_memory"]["search_history"] = [
            {"session_id": i, "time_index": i, "query": "unrelated history " * 40,
             "clicked_items": [], "purchased_items": []} for i in range(1, 14)]
        products, interactions = pool()
        child = freeze_prices(None, SampledCandidateBuilder(products, interactions, Encoder()).build(source), prices={})
        bundle = {"request": source, "query_enabled": True, "response_scenes": [child]}
        before = deepcopy(bundle)
        for budget in (0, 2048):
            binding = CurationTools(child["request"], "response", mode="fixed", concise=True,
                                   history_token_budget=budget, tokenizer=tokenizer)
            _, messages = trajectory_messages(binding, role="customer", stage="response", theory=True)
            messages, tools, _ = initial_agentic_request(messages, [], AgenticOptions(max_model_calls=1, theory=True))
            count = len(prompt_tokens(messages, tools, tokenizer))
            kept, skipped = filter_lengths([bundle], tokenizer, count, cot_preset="lightweight",
                                           history_token_budget=budget)
            self.assertEqual(kept[0]["response_scenes"][0]["candidate"]["input_tokens"], {"fixed": count})
            _, too_long = filter_lengths([bundle], tokenizer, count - 1, cot_preset="lightweight",
                                        history_token_budget=budget)
            self.assertTrue(any(s["stage"] == "response" for s in too_long))
            _, old = filter_lengths([bundle], tokenizer, count, modes=("fixed",))
            self.assertTrue(any(s["stage"] == "response" for s in old))
        self.assertEqual(bundle, before)

    def test_length_filter_rejects_incompatible_context_settings(self):
        from agent_simulation.datasets.customer_scenes import filter_lengths
        for options in ({"cot_preset": "lightweight", "modes": ("autonomous",)},
                        {"history_token_budget": 100, "modes": ("fixed", "autonomous")},
                        {"modes": ()}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                filter_lengths([], None, 100, **options)
