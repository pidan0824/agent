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
        result=self.builder(min_items=5,max_items=8).build(original)
        ids=result["candidate"]["candidate_item_ids"]
        self.assertTrue({10,11}<=set(ids))
        self.assertFalse({1,2,3}&set(ids))
        self.assertEqual(len(ids),len(set(ids)))
        self.assertTrue(result["candidate"]["outcome_conditioned"])
        self.assertEqual(result["request"]["response_target"],original["response_target"])
        self.assertEqual(result["request"]["query_input"],original["query_input"])
        self.assertEqual(original,before)
        self.assertEqual(json.loads(json.dumps(result,allow_nan=False))["candidate"]["candidate_item_ids"],ids)

    def test_no_click_keeps_observed_negatives_and_inserts_nothing(self):
        result=self.builder(min_items=5,max_items=8).build(request({"clicked_item_ids":[],"purchased_item_ids":[]}))
        self.assertEqual(result["candidate"]["positive_count"],0)
        self.assertTrue(set(result["candidate"]["candidate_item_ids"])<=set(range(10,22)))
        self.assertEqual(result["request"]["response_target"],{"clicked_item_ids":[],"purchased_item_ids":[]})

    def test_seed_is_repeatable_and_does_not_change_caller_rng(self):
        builder=self.builder();state=random.getstate()
        a=builder.build(request());b=builder.build(request())
        self.assertEqual(a["candidate"]["candidate_item_ids"],b["candidate"]["candidate_item_ids"])
        self.assertEqual(random.getstate(),state)

    def test_many_positives_are_preserved_above_maximum(self):
        result=self.builder(min_items=2,max_items=2).build(request({"clicked_item_ids":[10,11,12],"purchased_item_ids":[13]}))
        self.assertEqual(set(result["candidate"]["candidate_item_ids"]),{10,11,12,13})
        self.assertTrue(result["candidate"]["over_max_to_preserve_positives"])

    def test_catalog_preparation_uses_shared_pool_and_uniform_prior_prices(self):
        c=duckdb.connect()
        self.addCleanup(c.close)
        c.execute("CREATE TABLE products(item_id BIGINT,item_title VARCHAR)")
        c.executemany("INSERT INTO products VALUES (?,?)",[(i,f"item{i}") for i in range(1,41)])
        c.execute("CREATE TABLE requests(session_id BIGINT,user_id BIGINT,time_index BIGINT,split VARCHAR,impressed_item_ids BIGINT[],clicked_item_ids BIGINT[],purchased_item_ids BIGINT[])")
        c.executemany("INSERT INTO requests VALUES (?,?,?,?,?,?,?)",[(i,1,i,"train",list(range(1,41)),[i],[]) for i in range(1,4)])
        c.execute("CREATE TABLE ranking_observations(session_id BIGINT,item_id BIGINT,price_raw DOUBLE)")
        c.executemany("INSERT INTO ranking_observations VALUES (?,?,?)",[(3,i,50.) for i in range(1,41)])
        scenes,skipped=prepare_scenes(SimpleNamespace(connection=c),[request()],Encoder(),min_items=8,max_items=8)
        self.assertFalse(skipped)
        self.assertEqual(len(scenes),1)
        self.assertEqual(scenes[0]["candidate"]["pool_size"],40)
        self.assertTrue(all(p["price_raw"]==50. for p in scenes[0]["request"]["response_input"]["displayed_items"]))

    def test_length_filter_uses_generation_messages_without_inference(self):
        from agent_simulation.datasets.customer_scenes import filter_lengths
        from agent_simulation.tests.inference.test_agentic import Tokenizer
        scene=self.builder().build(request())
        kept,skipped=filter_lengths([scene],Tokenizer(),100000)
        self.assertEqual((len(kept),len(skipped)),(1,0))
        kept,skipped=filter_lengths([scene],Tokenizer(),1)
        self.assertEqual((len(kept),len(skipped)),(0,1))
        self.assertEqual(skipped[0]["reason"],"context_length")
