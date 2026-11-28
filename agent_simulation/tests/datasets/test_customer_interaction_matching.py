"""Interaction-matched candidates: CPU fixtures only, no embeddings/model calls."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
import duckdb

from agent_simulation.datasets.customer_candidate_sampling import (
    SampledCandidateBuilder, match_interacted_distractors, INTERACTION_MATCHED)
from agent_simulation.datasets.customer_scenes import estimate_interaction_calibration
from agent_simulation.tests.datasets.test_customer_sampling import request, pool


def calibration(probability=1.):
    return {"scope": "frozen_train_fit_full_exposures", "source_split": "train",
            "nonpositive_exposure_items": 10, "interacted_nonpositive_items": 1,
            "rate": .1, "seen_acceptance_probability": probability, "request_ids_sha256": "fixture"}


class InteractionMatchedTests(unittest.TestCase):
    def builders(self, probability=1.):
        products, interactions = pool()
        interactions += [{"user_id": 1, "item_id": 12, "time_index": 4},
                         {"user_id": 1, "item_id": 13, "time_index": 5},
                         {"user_id": 1, "item_id": 14, "time_index": 30}]
        opts = dict(methods=("random",), seed=8, min_items=6, max_items=6, min_history=0)
        old = SampledCandidateBuilder(products, interactions, None, **opts)
        new = SampledCandidateBuilder(products, interactions, None, **opts,
                candidate_protocol=INTERACTION_MATCHED, interaction_calibration=calibration(probability))
        return old, new

    def test_old_default_unchanged_and_zero_admission_keeps_candidates(self):
        old, new = self.builders(0.)
        raw = request()
        a = old.build(raw, target_item_id=10)
        b = new.build(raw, target_item_id=10)
        self.assertEqual(a["candidate"]["candidate_item_ids"], b["candidate"]["candidate_item_ids"])
        self.assertNotIn("candidate_protocol_version", a["candidate"])

    def test_seen_distractors_admitted_without_moving_target_or_changing_length(self):
        old, new = self.builders()
        raw = request(); before = deepcopy(raw)
        prices = {i:(100., 1) for i in range(10,22)}
        a = old.build(raw, target_item_id=10, price_observations=prices)
        b = new.build(raw, target_item_id=10, price_observations=prices)
        left,right=(r["candidate"]["candidate_item_ids"] for r in (a,b))
        self.assertEqual(len(left), len(right))
        self.assertEqual(left.index(10), right.index(10))
        self.assertEqual(a["candidate"]["target_positions"], b["candidate"]["target_positions"])
        self.assertTrue({12,13} <= set(right))
        self.assertNotIn(11, right)  # Another actual positive remains excluded.
        self.assertNotIn(14, set(right)-set(left))  # Future-only interaction is not a seen insertion.
        self.assertTrue(set(right) <= set(range(10,22)))
        self.assertEqual(a["request"]["response_target"], b["request"]["response_target"])
        self.assertEqual(raw, before)
        self.assertEqual(a["candidate"]["scope"]["deduplication"],
                         b["candidate"]["scope"]["deduplication"])
        self.assertEqual(new.build(raw,target_item_id=10)["candidate"]["candidate_item_ids"],right)

    def test_no_action_and_insufficient_seen_do_not_change_length(self):
        old,new=self.builders()
        raw=request({"clicked_item_ids":[],"purchased_item_ids":[]})
        a,b=(builder.build(raw) for builder in (old,new))
        self.assertEqual(len(a["candidate"]["candidate_item_ids"]),len(b["candidate"]["candidate_item_ids"]))
        self.assertEqual(b["request"]["response_target"],raw["response_target"])
        result,info=match_interacted_distractors([1,2],[1,2],[99],[],rate=.5,
            acceptance_probability=1.,seed=1,scene_id="none")
        self.assertEqual(result,[1,2])
        self.assertEqual(info["eligible_seen_count"],0)

    def test_sampling_rate_is_stochastic_not_forced_in_every_scene(self):
        counts=[]
        for sid in range(100):
            out,info=match_interacted_distractors([1,2,3],[1,2,3,4],[4],[],
                rate=.1,acceptance_probability=.3,seed=4,scene_id=str(sid))
            counts.append(info["selected_seen_count"])
            self.assertEqual(len(out),3)
            self.assertEqual(len(out),len(set(out)))
        self.assertTrue(0 < sum(counts) < 100)
        self.assertIn(0,counts)

    def test_invalid_calibration_rejected_and_future_not_used(self):
        products,interactions=pool()
        for field,value in (("scope","dev"),("rate",.5),("seen_acceptance_probability",None)):
            bad=calibration();bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                SampledCandidateBuilder(products,interactions,None,candidate_protocol=INTERACTION_MATCHED,
                                        interaction_calibration=bad)

    def test_calibration_only_fit_train_and_strict_prior(self):
        db=duckdb.connect(":memory:")
        try:
            db.execute("CREATE TABLE customer_history(source_split VARCHAR,user_id BIGINT,time_index BIGINT,clicked_item_ids BIGINT[],purchased_item_ids BIGINT[])")
            db.execute("INSERT INTO customer_history VALUES ('train',1,1,[12],[]),('train',1,20,[13],[]),('test',1,2,[14],[])")
            fit={"role":"customer","stage":"response","partition":"fit","source_split":"train",
                 "source_request_id":100,"target":{"clicked_item_ids":[10],"purchased_item_ids":[]},
                 "context":{"evidence_actor_id":1,"history_cutoff_exclusive":20,
                            "candidate_context":[{"item_id":i} for i in range(10,22)],"facts":{}}}
            dev=deepcopy(fit);dev.update(partition="dev",source_request_id=200)
            dev["target"]={"clicked_item_ids":[],"purchased_item_ids":[]}
            with TemporaryDirectory() as directory:
                path=Path(directory)/"cases.jsonl"
                path.write_text("\n".join(json.dumps(r) for r in (fit,dev))+"\n")
                result=estimate_interaction_calibration(SimpleNamespace(connection=db),str(path),
                                                       seed=8,min_items=6,max_items=6,min_history=0)
                self.assertEqual(result["interacted_nonpositive_items"],1)
                self.assertEqual(result["nonpositive_exposure_items"],11)
                self.assertEqual(result["request_ids"],[100])
                self.assertAlmostEqual(result["allocation"]["expected_seen_count"],
                                       result["allocation"]["target_seen_count"],places=8)
        finally:
            db.close()
