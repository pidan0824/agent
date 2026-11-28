"""Seller change protocol tests; no catalog, model service, GPU or training."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent_simulation.evaluation.seller_signals import (
    SELLER_CHANGE_PROTOCOL, fit_signal_policy, change_signals, upgrade_seller_context, seller_coverage)
from agent_simulation.evaluation.seller_context import observed_seller_context
from agent_simulation.evaluation.seller_quality import check_seller_evidence
from agent_simulation.tests.evaluation.test_seller_context_projection import fixture
from experiments.cot_sft_pilot import export, decision_request


def policy():
    return {"protocol": SELLER_CHANGE_PROTOCOL, "window_size": 5, "exposure_min": 3,
            "fit_case_ids_sha256": "fixture-fit-only"}


def history():
    def row(item, when, price, exposures=1, status="observed"):
        return dict(seller_id=7, item_id=item, time_index=when, price_raw=price,
                    price_status=status, exposure_records=exposures, click_records=0, purchase_records=0)
    return [row(4,1,20),row(4,3,22),row(4,4,0,status="unknown"),row(4,5,22,2),
            row(5,2,10),row(5,6,12),row(6,7,99),row(7,2,10),row(7,8,9)]


def case(context, label="keep"):
    return {"case_id":"seller:7:4:10:"+label, "role":"seller","stage":"seller","partition":"fit",
        "actor_id":7,"source_split":"train","source_request_id":"fixture:"+label,
        "dataset_manifest_sha256":"fixture","stratum":label,"context":context,
        "target":{"prices":[{"item_id":4,"price_raw":22}]}, "previous_prices":{"4":22}}


class SellerChangeTests(unittest.TestCase):
    def test_own_change_interval_length_and_magnitude(self):
        own, shop = change_signals(history(),seller_id=7,item_id=4,cutoff=10,policy=policy())
        self.assertEqual((own["history_observation_count"],own["valid_quote_observation_count"]),(4,3))
        self.assertEqual(own["observations_since_last_change"],1)
        self.assertEqual(own["last_change"]["direction"],"up")
        self.assertAlmostEqual(own["last_change"]["relative_change"],.1)
        self.assertEqual((own["last_change"]["before_time_index"],own["last_change"]["after_time_index"]),(1,3))
        self.assertEqual(own["traffic_evidence"]["exposures"],2)
        self.assertEqual(own["traffic_evidence"]["interpretation"],"流量数据不足，不支持推断")

    def test_shop_known_predecessor_denominator_and_zero_data(self):
        own, shop = change_signals(history(),seller_id=7,item_id=4,cutoff=10,policy=policy())
        self.assertEqual((shop["denominator_known_previous_quote"],shop["changed_products"],
                          shop["up_products"],shop["down_products"]),(2,2,1,1))
        self.assertEqual(shop["missing_previous_quote_products"],1)
        self.assertEqual(shop["change_fraction"],1)
        self.assertEqual(shop["earliest_previous_time_index"],2)
        _, empty = change_signals([r for r in history() if r["item_id"]==4],
                                 seller_id=7,item_id=4,cutoff=10,policy=policy())
        self.assertEqual(empty["status"],"无数据")
        self.assertIsNone(empty["change_fraction"])

    def test_no_observed_change_does_not_invent_age_or_direction(self):
        hs=[r for r in history() if r["item_id"]==4]
        for r in hs:
            if r["price_status"]=="observed":r["price_raw"]=20
        own,_=change_signals(hs,seller_id=7,item_id=4,cutoff=10,policy=policy())
        self.assertFalse(own["has_observed_change"])
        self.assertIsNone(own["last_change"])
        self.assertIsNone(own["observations_since_last_change"])

    def test_nonprior_foreign_duplicate_and_unfitted_policy_rejected(self):
        for kind in ("future","foreign","duplicate","policy"):
            hs=history();p=policy()
            if kind=="future":hs[-1]["time_index"]=10
            elif kind=="foreign":hs[-1]["seller_id"]=8
            elif kind=="duplicate":hs.append(deepcopy(hs[0]))
            else:p.pop("fit_case_ids_sha256")
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                change_signals(hs,seller_id=7,item_id=4,cutoff=10,policy=p)

    def test_fit_calibration_excludes_dev_and_does_not_use_labels(self):
        data=[{"case_id":str(i),"partition":"fit","cutoff":i+20,"exposures":x,"direction":"keep"}
              for i,x in enumerate((0,1,4,10,20))]
        a=fit_signal_policy(data,window_size=5)
        self.assertEqual(a["exposure_min"],1)
        changed=deepcopy(data)
        for r in changed:r["direction"]="up"
        self.assertEqual(a,fit_signal_policy(changed,window_size=5))
        changed[0]["partition"]="dev"
        with self.assertRaises(ValueError):fit_signal_policy(changed)

    def test_legacy_is_default_and_new_sources_are_separately_citable(self):
        state,hs,features=fixture()
        old=observed_seller_context(state,features,hs)
        self.assertEqual(old,observed_seller_context(state,features,hs,protocol="legacy"))
        new=observed_seller_context(state,features,hs,protocol=SELLER_CHANGE_PROTOCOL,signal_policy=policy())
        self.assertEqual(old,observed_seller_context(state,features,hs))
        own=new["owned_products"][0]
        refs=[own[key]["evidence_id"] for key in ("quote_evidence","same_shop_activity","market_position")]
        self.assertEqual(len(set(refs)),3)
        rationale={"hypothesis":"Fixture", "evidence":["["+ref+"] Supplied source" for ref in refs],
                   "uncertainty":"No causal claim"}
        action={"prices":[{"item_id":4,"price_raw":22}]}
        t={"rationale":rationale,"final_action":action,"messages":[
            {"role":"user","content":json.dumps(new)},
            {"role":"assistant","content":json.dumps({"rationale":rationale,"action":action})}]}
        self.assertTrue(check_seller_evidence(t)["accepted"])
        bad=deepcopy(t);view=json.loads(bad["messages"][0]["content"])
        view["owned_products"][0]["same_shop_activity"]["changed_products"]=99
        bad["messages"][0]["content"]=json.dumps(view)
        self.assertIn("unknown_evidence_reference",check_seller_evidence(bad)["reasons"])

    def test_direct_cot_share_new_facts_and_target_does_not_change_inputs(self):
        state,hs,features=fixture()
        ctx=observed_seller_context(state,features,hs,protocol=SELLER_CHANGE_PROTOCOL,signal_policy=policy())
        a=case(ctx);b=deepcopy(a);b["target"]["prices"][0]["price_raw"]=999999
        self.assertEqual(decision_request(a,"concise"),decision_request(b,"concise"))
        self.assertEqual(decision_request(a,"direct")["messages"][1:],
                         decision_request(a,"concise")["messages"][1:])

    def test_coverage_requires_passing_each_stratum(self):
        rows=[dict(role="seller",stratum=s,status="ok",quality={"accepted":s=="keep"})
              for s in ("keep","up","down")]
        report=seller_coverage(rows)
        self.assertFalse(report["release_allowed"])
        self.assertEqual(report["missing_pass_strata"],["up","down"])
        for r in rows:r["quality"]["accepted"]=True
        self.assertTrue(seller_coverage(rows)["release_allowed"])
        for row in rows:row["style"]="direct"
        self.assertFalse(seller_coverage(rows)["release_allowed"])

    def test_zero_stratum_export_writes_failure_without_training_files(self):
        state,hs,features=fixture()
        ctx=observed_seller_context(state,features,hs,protocol=SELLER_CHANGE_PROTOCOL,signal_policy=policy())
        cases=[case(ctx,s) for s in ("keep","up","down")]
        with TemporaryDirectory() as directory:
            root=Path(directory);cohorts=root/"cohorts";cohorts.mkdir()
            (cohorts/"cases.jsonl").write_text("fixture")
            out=root/"release"
            with patch("experiments.cot_sft_pilot.load_cases",return_value=cases):
                report=export(SimpleNamespace(cohorts=cohorts,teachers=[],output=out))
            self.assertEqual(report["status"],"failed")
            self.assertTrue((out/"seller-export-failure.json").exists())
            self.assertFalse((out/"seller.jsonl").exists())
            self.assertEqual(list(out.glob("*.jsonl")),[])

    def test_old_export_remains_available_without_new_gate(self):
        state,hs,features=fixture();ctx=observed_seller_context(state,features,hs)
        with TemporaryDirectory() as directory:
            root=Path(directory);cohorts=root/"cohorts";cohorts.mkdir()
            (cohorts/"cases.jsonl").write_text("fixture")
            out=root/"legacy-release"
            with patch("experiments.cot_sft_pilot.load_cases",return_value=[case(ctx)]):
                export(SimpleNamespace(cohorts=cohorts,teachers=[],output=out))
            self.assertTrue((out/"seller.jsonl").exists())
            self.assertFalse((out/"seller-export-failure.json").exists())
