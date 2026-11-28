"""Offline, versioned Seller context rebuild; never starts a model service."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from agent_simulation.data.catalog import Catalog
from agent_simulation.data.common import require, rows, sha256
from agent_simulation.evaluation.seller_signals import (
    SELLER_CHANGE_PROTOCOL, fit_signal_policy, window_exposures, upgrade_seller_context)
from agent_simulation.evaluation.seller_quality import check_seller_evidence
from agent_simulation.inference.requests import load_tokenizer, prompt_tokens
from experiments.cot_sft_pilot import load_cases, decision_request, dump


def stats(values):
    return {"n": len(values), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "max": max(values)}


def rebuild(args):
    require(not args.output.exists(), "Use a new Seller rebuild directory")
    source = [r for r in load_cases(args.cohorts) if r["role"] == "seller"]
    require(source and all(r["source_split"] == "train" for r in source), "Train Seller cohorts required")
    require(args.context_protocol in ("legacy", SELLER_CHANGE_PROTOCOL), "Unknown Seller protocol")
    require(all(r.get("context", {}).get("context_protocol", "legacy") != SELLER_CHANGE_PROTOCOL for r in source),
            "Rebuild starts from frozen legacy contexts")
    args.output.mkdir(parents=True)
    targets = [{"case_id": r["case_id"], "seller_id": int(r["actor_id"]),
                "item_id": r["context"]["requested_price_item_ids"][0],
                "cutoff": r["context"]["history_cutoff_exclusive"], "partition": r["partition"]}
               for r in source]
    (args.output/"targets.jsonl").write_text("".join(dump(r)+"\n" for r in targets))
    started = time.monotonic()
    catalog = Catalog(args.catalog)
    try:
        db = catalog.connection
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GB'")
        db.execute("SET temp_directory='"+str((args.output/"spill").resolve())+"'")
        db.execute("CREATE TEMP TABLE seller_change_targets AS SELECT * FROM read_json_auto(?)",
                   [str(args.output/"targets.jsonl")])
        history = rows(db, """SELECT h.seller_id,h.item_id,h.time_index,h.price_raw,h.price_status,
                    h.exposure_records,h.click_records,h.purchase_records
            FROM seller_history h JOIN (
                SELECT seller_id,max(cutoff) cutoff FROM seller_change_targets GROUP BY seller_id
            ) t USING(seller_id) WHERE h.time_index<t.cutoff ORDER BY seller_id,time_index,item_id""")
    finally:
        catalog.close()
    by_seller = defaultdict(list)
    for row in history:
        by_seller[row["seller_id"]].append(row)
    histories = {}
    fit = []
    for t in targets:
        hs = [r for r in by_seller[t["seller_id"]] if r["time_index"] < t["cutoff"]]
        histories[t["case_id"]] = hs
        if t["partition"] == "fit":
            traffic = window_exposures(hs, t["item_id"], t["cutoff"], args.window_size)
            fit.append({**t, "exposures": traffic["exposures"]})
    policy = fit_signal_policy(fit, window_size=args.window_size)
    policy.update(cohorts_cases_sha256=sha256(args.cohorts/"cases.jsonl"),
                  fit_case_ids=sorted(r["case_id"] for r in fit))
    (args.output/"signal-policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2)+"\n")
    tokenizer = load_tokenizer(args.tokenizer, local_files_only=True)
    result, details = [], []
    for old in source:
        hs = histories[old["case_id"]]
        fresh = deepcopy(old)
        context = (upgrade_seller_context(old["context"], hs, policy)
                   if args.context_protocol == SELLER_CHANGE_PROTOCOL else deepcopy(old["context"]))
        fresh["context"] = context
        changed_target = deepcopy(old)
        changed_target["target"]["prices"][0]["price_raw"] = 987654321
        assert (upgrade_seller_context(changed_target["context"], hs, policy) == context
                if args.context_protocol == SELLER_CHANGE_PROTOCOL else changed_target["context"] == context)
        old_request = decision_request(old, "concise")
        new_request = decision_request(fresh, "concise")
        direct = decision_request(fresh, "direct")
        assert direct["messages"][1:] == new_request["messages"][1:]
        nt = len(prompt_tokens(new_request["messages"], [], tokenizer))
        ot = len(prompt_tokens(old_request["messages"], [], tokenizer))
        refs = []
        def collect(node):
            if isinstance(node, dict):
                if "evidence_id" in node: refs.append(node["evidence_id"])
                for value in node.values(): collect(value)
            elif isinstance(node, list):
                for value in node: collect(value)
        collect(context)
        if args.context_protocol == SELLER_CHANGE_PROTOCOL:
            for product in context["owned_products"]:
                expected = {"quote_evidence", "same_shop_activity", "market_position"}
                available = expected & set(product)
                assert len({product[key]["evidence_id"] for key in available}) == len(available)
                explanation = {"hypothesis": "Offline citation fixture",
                               "evidence": ["["+product[key]["evidence_id"]+"] Supplied source block"
                                            for key in sorted(available)],
                               "uncertainty": "No teacher generated"}
                action = old["target"]
                fixture = {"rationale": explanation, "final_action": action,
                    "messages": new_request["messages"]+[{"role": "assistant", "content": dump({
                        "rationale": explanation, "action": action})}]}
                assert check_seller_evidence(fixture)["accepted"]
        result.append(fresh)
        details.append({"case_id": old["case_id"], "partition": old["partition"], "stratum": old["stratum"],
                        "old_input_tokens": ot, "new_input_tokens": nt, "delta": nt-ot,
                        "fits8192_with768": nt+768<=8192, "evidence_ids": refs,
                        "own_observation_count": context["owned_products"][0].get("quote_evidence",{}).get("history_observation_count"),
                        "traffic_status": context["owned_products"][0].get("quote_evidence",{}).get("traffic_evidence",{}).get("status")})
    cases_path = args.output/"cases.jsonl"
    cases_path.write_text("".join(dump(r)+"\n" for r in result))
    (args.output/"details.jsonl").write_text("".join(dump(r)+"\n" for r in details))
    report = {"context_protocol": args.context_protocol, "scenes": len(result),
        "strata": dict(Counter(r["stratum"] for r in result)),
        "partitions": dict(Counter(r["partition"] for r in result)),
        "policy": policy, "all_original_cases_preserved": True,
        "all_strictly_prior_actor_and_predecessor_checked": True, "target_mutation_invariant": True,
        "direct_cot_facts_equal": True, "nested_sources_pass_fixture_check": True,
        "tokens": {"old": stats([r["old_input_tokens"] for r in details]),
                   "new": stats([r["new_input_tokens"] for r in details]),
                   "paired_delta": stats([r["delta"] for r in details])},
        "input_capacity_failures": [r["case_id"] for r in details if not r["fits8192_with768"]],
        "reference_count": dict(Counter(len(r["evidence_ids"]) for r in details)),
        "traffic_status": dict(Counter(r["traffic_status"] for r in details)),
        "model_calls": 0, "training": False, "seconds": time.monotonic()-started,
        "scope": "Only input reconstruction, fixture citations and lengths; not teacher quality or predictive gain"}
    (args.output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    manifest = {"status": "complete", "schema": "seller-change-context-release-v1",
        "context_protocol": args.context_protocol, "cases_sha256": sha256(cases_path),
        "source_cases_sha256": sha256(args.cohorts/"cases.jsonl"),
        "signal_policy_sha256": sha256(args.output/"signal-policy.json"),
        "tokenizer": str(args.tokenizer), "model_calls": 0,
        "source_sha256": {str(p):sha256(p) for p in (
            Path(__file__), Path("agent_simulation/evaluation/seller_signals.py"),
            Path("agent_simulation/evaluation/seller_context.py"))}}
    (args.output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohorts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--catalog", type=Path, default=Path("data/kuaisearch/lite/source/catalog"))
    p.add_argument("--tokenizer", type=Path, default=Path("models/Qwen3-32B"))
    p.add_argument("--context-protocol", choices=("legacy", SELLER_CHANGE_PROTOCOL), default="legacy")
    p.add_argument("--window-size", type=int, default=10000)
    return p


if __name__ == "__main__":
    rebuild(parser().parse_args())
