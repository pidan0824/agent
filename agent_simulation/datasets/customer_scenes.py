"""Prepare frozen UserMirrorer Customer training scenes, without LLM inference."""
import argparse
from collections import Counter
from contextlib import closing
from copy import deepcopy
import json
import hashlib
import math
import random
from pathlib import Path

from agent_simulation.data.catalog import Catalog
from agent_simulation.data.common import require, rows, sha256
from agent_simulation.datasets.customer_candidate_sampling import SampledCandidateBuilder, CANDIDATE_DEDUPLICATION, INTERACTION_MATCHED, LEGACY_CANDIDATES
from agent_simulation.datasets.labels import select_record_ids


def prior_prices(catalog, ids, cutoff):
    """One frozen price lookup per request, shared by deduplication and all children."""
    observations = catalog.connection.execute("""
        SELECT o.item_id, arg_max(o.price_raw,r.time_index) AS price_raw, max(r.time_index) AS observed_time_index
        FROM ranking_observations o JOIN requests r USING(session_id)
        WHERE r.time_index < ? AND o.item_id IN (SELECT unnest(?::BIGINT[]))
          AND isfinite(o.price_raw) AND o.price_raw > 0 GROUP BY o.item_id
    """, [cutoff,ids]).fetchall()
    return {item:(price,time) for item,price,time in observations}


def freeze_prices(catalog, scene, *, prices=None):
    """Use the same strictly-prior price lookup for positives and distractors."""
    request = scene["request"]
    request["response_input"]["response_action_mode"] = "single_item"
    cutoff = request["response_input"]["history_cutoff_exclusive"]
    ids = scene["candidate"]["candidate_item_ids"]
    if prices is None:
        prices = prior_prices(catalog, ids, cutoff)
    for product in request["response_input"]["displayed_items"]:
        price,time = prices.get(product["item_id"],(None,None))
        product["price_raw"] = price
        product["price_status"] = "observed" if price is not None else "unknown"
        product["price_observed_time_index"] = time
    request["response_input"]["candidate_protocol"] = "Synthetic choice list; prices are strictly-prior observations, not current quotes."
    scene["candidate"]["price_policy"] = "Latest valid strictly-prior observation for every candidate; never target-only current prices"
    return scene



def estimate_interaction_calibration(catalog, fit_cases, *, seed=42, min_items=2, max_items=12, min_history=3):
    """Use only frozen Train fit full-exposure Response cases; no dev/audit estimates."""
    from agent_simulation.evaluation.customer_evidence import expand_sources
    fit_cases = Path(fit_cases)
    cases = [row for row in map(json.loads, Path(fit_cases).read_text().splitlines())
             if row.get("role") == "customer" and row.get("stage") == "response"
             and row.get("partition") == "fit"]
    require(cases and all(row["source_split"] == "train" for row in cases),
            "Calibration requires frozen Train fit Response cases")
    require(all(row.get("action_mode") != "single_item" and
                row.get("evaluation_protocol") != "outcome_conditioned_single_positive" for row in cases),
            "Interaction calibration requires original full-exposure cases")
    contexts = [expand_sources(row["context"]) for row in cases]
    cohort_protocol_path = Path(fit_cases).with_name("protocol.json")
    fit_interval = None
    if cohort_protocol_path.exists():
        fit_interval = json.loads(cohort_protocol_path.read_text()).get("time_intervals_inclusive", {}).get("fit")
        if fit_interval:
            require(all(fit_interval[0] <= c["history_cutoff_exclusive"] <= fit_interval[1] for c in contexts),
                    "Fit calibration case outside frozen fit interval")
    users = sorted({c["evidence_actor_id"] for c in contexts})
    histories = rows(catalog.connection, """
        SELECT user_id,time_index,clicked_item_ids,purchased_item_ids FROM customer_history
        WHERE source_split='train' AND user_id IN (SELECT unnest(?::BIGINT[]))
          AND time_index < ? ORDER BY user_id,time_index
    """, [users, max(c["history_cutoff_exclusive"] for c in contexts)])
    by_user = {}
    for event in histories:
        by_user.setdefault(event["user_id"], []).append(event)
    numerator = denominator = 0
    ids, opportunities = [], []
    for row, context in zip(cases, contexts, strict=True):
        require(context.get("candidate_protocol", "full") !=
                "Synthetic choice list; prices are strictly-prior observations, not current quotes.",
                "Calibration requires full exposures, not synthetic shortlists")
        seen = {item for event in by_user.get(context["evidence_actor_id"], [])
                if event["time_index"] < context["history_cutoff_exclusive"]
                for field in ("clicked_item_ids", "purchased_item_ids") for item in event[field]}
        positives = set(row["target"]["clicked_item_ids"]) | set(row["target"]["purchased_item_ids"])
        exposed = {p["item_id"] for p in context["candidate_context"]}
        require(positives <= exposed, "Calibration positives absent from full exposure")
        negatives = exposed - positives
        denominator += len(negatives)
        numerator += len(negatives & seen)
        ids.append(row["source_request_id"])
        prior_count = sum(len(set(event["clicked_item_ids"]) | set(event["purchased_item_ids"]))
                          for event in by_user.get(context["evidence_actor_id"], [])
                          if event["time_index"] < context["history_cutoff_exclusive"])
        if prior_count >= min_history:
            for positive in sorted(positives) or [None]:
                sid = (f'{row["source_request_id"]}:positive:{positive}' if positive is not None else
                       f'{row["source_request_id"]}:no_action')
                size = random.Random(f"{seed}:{sid}:target-size-v1").randint(min_items, max_items)
                slots = min(size - int(positive is not None), len(negatives - seen))
                opportunities.append((slots, len(negatives & seen)))
    require(denominator > 0 and len(ids) == len(set(ids)), "Empty or duplicate fit calibration")
    total_slots = sum(slots for slots, count in opportunities)
    require(total_slots > 0, "Fit calibration has no eligible negative slots")
    target_count = total_slots * numerator / denominator
    def expected(probability):
        total = 0.
        for slots, count in opportunities:
            # Exact expectation of a capped binomial; no Monte Carlo fit.
            total += sum(min(slots, k) * math.comb(count, k) * probability**k *
                         (1 - probability)**(count-k) for k in range(count+1))
        return total
    low, high = 0., 1.
    for _ in range(50):
        middle = (low + high) / 2
        if expected(middle) < target_count:
            low = middle
        else:
            high = middle
    probability = (low + high) / 2
    return {"schema": "fit-interaction-calibration-v1", "scope": "frozen_train_fit_full_exposures",
            "source_split": "train", "source_path": str(Path(fit_cases)),
            "source_sha256": sha256(fit_cases), "request_ids": sorted(ids),
            "request_ids_sha256": hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest(),
            "fit_time_interval_inclusive": fit_interval,
            "fit_cutoff_range": [min(c["history_cutoff_exclusive"] for c in contexts),
                                 max(c["history_cutoff_exclusive"] for c in contexts)],
            "request_count": len(ids), "nonpositive_exposure_items": denominator,
            "seen_acceptance_probability": probability,
            "allocation": {"seed": seed, "min_items": min_items, "max_items": max_items, "min_history": min_history,
                           "eligible_scene_count": len(opportunities), "legacy_negative_slot_estimate": total_slots,
                           "target_seen_count": target_count, "expected_seen_count": expected(probability),
                           "maximum_seen_count": sum(min(slots, count) for slots, count in opportunities),
                           "opportunity_scope": "Fit full-exposure child lists before content deduplication; realized rates reported separately"},
            "interacted_nonpositive_items": numerator, "rate": numerator / denominator,
            "estimator": "Frozen full-exposure fit sample estimate, not whole-Train population rate",
            "history_source": "customer_history Train, strictly before each request cutoff; no snapshots",
            "excluded_partitions": ["dev", "audit"]}

def prepare_scenes(catalog, requests, encoder, *, min_items=2, max_items=12, seed=42,
                   min_history=3, methods=("random", "content", "collaborative"),
                   candidate_protocol=LEGACY_CANDIDATES, interaction_calibration=None):
    require(bool(requests), "Empty request selection")
    cutoffs = [r["response_input"]["history_cutoff_exclusive"] for r in requests]
    interactions = rows(catalog.connection, """
        SELECT user_id,unnest(list_distinct(list_concat(clicked_item_ids,purchased_item_ids))) AS item_id,
               time_index FROM requests WHERE split='train' AND time_index < ?
    """, [max(cutoffs)])
    fit_cutoff = min(cutoffs)
    # Global fit history remains available, including items outside these exposures.
    # Exposure eligibility is unchanged; unrelated historical impressions need no facts.
    needed_ids = {r["item_id"] for r in interactions if r["time_index"] < fit_cutoff}
    needed_ids.update(p["item_id"] for request in requests
                      for p in request["response_input"]["displayed_items"])
    products = rows(catalog.connection, """
        SELECT * FROM products WHERE item_id IN (SELECT unnest(?::BIGINT[]))
    """, [sorted(needed_ids)])
    by_id = {p["item_id"]: p for p in products}
    for request in requests:
        for product in request["response_input"]["displayed_items"]:
            if product["item_id"] not in by_id:
                by_id[product["item_id"]] = {**deepcopy(product),
                    "first_seen_time_index": request["response_input"]["history_cutoff_exclusive"]}
    builder = SampledCandidateBuilder(list(by_id.values()), interactions, encoder,
        min_items=min_items,max_items=max_items,seed=seed,min_history=min_history,methods=methods,
        fit_cutoff=fit_cutoff, candidate_protocol=candidate_protocol, interaction_calibration=interaction_calibration)
    scenes, skipped = [], []
    expected = {"insufficient_prior_interactions", "empty_available_item_pool",
                "empty_constructed_list", "insufficient_distractors"}
    for request in requests:
        bundle = {"request": deepcopy(request), "query_enabled": True, "response_scenes": []}
        target = request["response_target"]
        positives = sorted(set(target["clicked_item_ids"]) | set(target["purchased_item_ids"]))
        prices = prior_prices(catalog,
            [p["item_id"] for p in request["response_input"]["displayed_items"]],
            request["response_input"]["history_cutoff_exclusive"])
        for item in positives or [None]:
            try:
                scene = builder.build(request, target_item_id=item, price_observations=prices)
                bundle["response_scenes"].append(freeze_prices(catalog, scene, prices=prices))
            except ValueError as error:
                if str(error) not in expected:
                    raise
                skipped.append({"session_id": request["session_id"], "stage": "response",
                                "target_item_id": item, "reason": str(error)})
        # A Response construction failure never removes its independent Query.
        scenes.append(bundle)
    return scenes, skipped


def filter_lengths(scenes, tokenizer, limit, *, modes=None, cot_preset="original", history_token_budget=None):
    """Count the actual generation prompt/schema/tools, without model inference."""
    from agent_simulation.agents.agentic import initial_agentic_request, AgenticOptions
    from agent_simulation.evaluation.customer_context import CurationTools, resolve_context_options
    from agent_simulation.evaluation.trajectories import trajectory_messages
    from agent_simulation.inference.requests import prompt_tokens
    require(limit>0, "Length limit must be positive")
    if modes is None:
        modes = ("fixed",) if cot_preset == "lightweight" else ("fixed", "autonomous")
    context_options = resolve_context_options(cot_preset, modes, history_token_budget)
    kept,skipped=[],[]
    for bundle in deepcopy(scenes):
        def counts_for(request, stage):
            counts = {}
            for mode in modes:
                binding = CurationTools(request, stage, mode=mode, tokenizer=tokenizer, **context_options)
                _, messages = trajectory_messages(binding, role="customer", stage=stage, theory=True)
                options = AgenticOptions(max_model_calls=1 if mode=="fixed" else 3)
                initial, tools, _ = initial_agentic_request(messages, binding.schemas, options)
                counts[mode] = len(prompt_tokens(initial, tools, tokenizer))
            return counts
        counts = counts_for(bundle["request"], "query")
        bundle["query_enabled"] = max(counts.values()) <= limit
        if not bundle["query_enabled"]:
            skipped.append({"session_id": bundle["request"]["session_id"], "stage": "query",
                            "reason": "context_length", "input_tokens": counts})
        children = []
        for scene in bundle["response_scenes"]:
            counts = counts_for(scene["request"], "response")
            if max(counts.values()) > limit:
                skipped.append({"session_id": bundle["request"]["session_id"], "stage": "response",
                                "scene_id": scene["candidate"]["scene_id"],
                                "reason": "context_length", "input_tokens": counts})
            else:
                scene["candidate"]["input_tokens"] = counts
                children.append(scene)
        bundle["response_scenes"] = children
        if bundle["query_enabled"] or children:
            kept.append(bundle)
    return kept,skipped


def read_scenes(directory, *, dataset=None, split=None):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text())
    require(manifest["schema"]=="customer-single-positive-scenes-v2", "Prepare scenes again: single-positive v2 required")
    protocol = manifest.get("candidate_protocol_version", LEGACY_CANDIDATES)
    require(protocol in (LEGACY_CANDIDATES, INTERACTION_MATCHED), "Unknown scene candidate protocol")
    deduplication = manifest.get("candidate_deduplication")
    require(deduplication in (None, CANDIDATE_DEDUPLICATION), "Unknown candidate deduplication policy")
    if dataset is not None:
        require(manifest["dataset_manifest_sha256"]==sha256(Path(dataset)/"manifest.json"), "Scene label release differs")
    if split is not None:
        require(manifest["split"]==split, "Scene source split differs")
    require(sha256(directory/"scenes.jsonl")==manifest["scenes_sha256"], "Prepared scenes changed")
    scenes=[json.loads(s) for s in (directory/"scenes.jsonl").read_text().splitlines() if s.strip()]
    result={s["request"]["session_id"]:s for s in scenes}
    require(len(result)==len(scenes), "Duplicate prepared request")
    scene_ids = [child["candidate"]["scene_id"] for bundle in scenes for child in bundle["response_scenes"]]
    require(len(scene_ids) == len(set(scene_ids)), "Duplicate split scene IDs")
    for bundle in scenes:
        source = bundle["request"]
        require(source["assigned_split"] == manifest["split"], "Source request split differs")
        positives = set(source["response_target"]["clicked_item_ids"]) | set(source["response_target"]["purchased_item_ids"])
        for child in bundle["response_scenes"]:
            if protocol == INTERACTION_MATCHED:
                require(child["candidate"].get("candidate_protocol_version") == protocol,
                        "Child interaction protocol differs from scene manifest")
            if deduplication is not None:
                require(child["candidate"].get("scope", {}).get("deduplication", {}).get("protocol") == deduplication,
                        "Child deduplication policy differs from scene manifest")
            require(child["request"]["session_id"] == source["session_id"], "Mixed source request")
            require(child["request"]["assigned_split"] == source["assigned_split"], "Child request crosses dataset splits")
            require(child["candidate"]["source_request_id"] == source["session_id"], "Mixed candidate source request")
            child["request"]["response_input"]["response_action_mode"] = "single_item"
            chosen = child["candidate"]["target_item_id"]
            require(chosen in positives if positives else chosen is None, "Invalid split target")
            selected = set(child["candidate"]["candidate_item_ids"])
            require(not (selected & (positives - {chosen})), "Other true positives entered a split list")
            projected = {f: [i for i in source["response_target"][f] if i == chosen]
                         for f in ("clicked_item_ids", "purchased_item_ids")}
            require(child["request"]["response_target"] == projected, "Invalid split labels")
    return result,manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--split",choices=["train","test"],default="train")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--request-id",type=int,action="append")
    selection.add_argument("--selection",type=Path,help="Complete-Train scene selection directory; uses all selected IDs")
    selection.add_argument("--count",type=int,default=100)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--min-items",type=int,default=2)
    p.add_argument("--max-items",type=int,default=12)
    p.add_argument("--min-history",type=int,default=3)
    p.add_argument("--candidate-protocol", choices=[LEGACY_CANDIDATES, INTERACTION_MATCHED], default=LEGACY_CANDIDATES)
    p.add_argument("--interaction-fit-cases", type=Path,
                   help="Frozen full-exposure cases.jsonl; calibrates from Train fit rows only")
    p.add_argument("--methods",nargs="+",choices=["random","content","collaborative"],default=["random","content","collaborative"])
    p.add_argument("--bge-path",type=Path,default=Path("models/bge-small-zh-v1.5"))
    p.add_argument("--bge-device",default="cuda",help="Embedding device, e.g. cuda, cuda:1 or cpu; honor CUDA_VISIBLE_DEVICES")
    p.add_argument("--max-input-tokens",type=int,default=0,help="Optional actual-prompt length filter; 0 disables")
    p.add_argument("--tokenizer",type=Path,default=Path("models/Qwen3-32B"))
    p.add_argument("--cot-preset", choices=["original", "lightweight"], default="original",
                   help="Match the generation preset when filtering input lengths")
    p.add_argument("--modes", nargs="+", choices=["fixed", "autonomous"],
                   help="Length-filter modes: lightweight defaults to fixed; original to both")
    p.add_argument("--history-token-budget", type=int, help="Match the generation history budget")
    args=p.parse_args(argv)
    from agent_simulation.evaluation.customer_context import resolve_context_options
    if args.modes is None:
        args.modes = ["fixed"] if args.cot_preset == "lightweight" else ["fixed", "autonomous"]
    args.history_token_budget = resolve_context_options(
        args.cot_preset, args.modes, args.history_token_budget)["history_token_budget"]
    require(args.count>0 and not args.output.exists(), "Use a positive count and new output directory")
    release=json.loads((args.dataset/"manifest.json").read_text())
    with closing(Catalog(release["catalog"])) as catalog:
        require(sha256(catalog.directory/"manifest.json")==release["catalog_manifest_sha256"], "Catalog changed")
        selection_manifest = None
        if args.selection:
            from agent_simulation.datasets.customer_selection import read_selection
            ids, selection_manifest = read_selection(args.selection, dataset=args.dataset, split=args.split)
        else:
            ids=args.request_id or select_record_ids(catalog.connection,args.dataset,"customer",args.split,args.count,args.seed)
        members={r[0] for r in catalog.connection.execute(
            "SELECT session_id FROM read_parquet(?) WHERE session_id IN (SELECT unnest(?::BIGINT[]))",
            [str(args.dataset/"customer"/args.split/"labels.parquet"),ids]).fetchall()}
        require(ids and len(ids)==len(set(ids)) and members==set(ids), "Invalid scene request selection")
        requests=[catalog.request(identity,all_prior=True) for identity in ids]
        encoder=None
        if "content" in args.methods:
            from agent_simulation.agents.retrieval.bge_encoder import bge_encoder
            encoder,_=bge_encoder(args.bge_path,device=args.bge_device)
        calibration = None
        if args.candidate_protocol == INTERACTION_MATCHED:
            require(args.interaction_fit_cases is not None, "Specify --interaction-fit-cases for the new protocol")
            calibration = estimate_interaction_calibration(catalog, args.interaction_fit_cases, seed=args.seed,
                min_items=args.min_items, max_items=args.max_items, min_history=args.min_history)
        scenes,skipped=prepare_scenes(catalog,requests,encoder,min_items=args.min_items,max_items=args.max_items,
            seed=args.seed,min_history=args.min_history,methods=args.methods,
            candidate_protocol=args.candidate_protocol, interaction_calibration=calibration)
    if args.max_input_tokens:
        from agent_simulation.inference.requests import load_tokenizer
        scenes,length_skipped=filter_lengths(scenes,load_tokenizer(args.tokenizer,local_files_only=True),
            args.max_input_tokens, modes=args.modes, cot_preset=args.cot_preset,
            history_token_budget=args.history_token_budget)
        skipped.extend(length_skipped)
    args.output.mkdir(parents=True)
    (args.output/"scenes.jsonl").write_text("".join(json.dumps(s,ensure_ascii=False,allow_nan=False)+"\n" for s in scenes))
    (args.output/"skipped.jsonl").write_text("".join(json.dumps(s,ensure_ascii=False)+"\n" for s in skipped))
    manifest={"schema":"customer-single-positive-scenes-v2","dataset":str(args.dataset.resolve()),
        "dataset_manifest_sha256":sha256(args.dataset/"manifest.json"),"split":args.split,
        "args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "selection_manifest_sha256":sha256(args.selection/"manifest.json") if args.selection else None,
        "selection_population":selection_manifest["population"] if selection_manifest else None,
        "scenes_sha256":sha256(args.output/"scenes.jsonl"),"prepared":len(scenes),"skipped":len(skipped),
        "query_scenes":sum(s["query_enabled"] for s in scenes),
        "response_scenes":sum(len(s["response_scenes"]) for s in scenes),
        "skip_reasons":dict(Counter(s["reason"] for s in skipped)),
        "protocol":"Outcome-conditioned shortlists sampled within recorded exposures, without Query/category gates; not full-exposure evaluation.",
        "candidate_pool":"request_exposures","query_category_filter":False,
        "candidate_deduplication":CANDIDATE_DEDUPLICATION,
        "candidate_protocol_version":args.candidate_protocol,
        "interaction_calibration":calibration,
        "query":"Current Query target/exposures remain absent from query_input.",
        "collaborative_fit_cutoff_exclusive":min(r["response_input"]["history_cutoff_exclusive"] for r in requests),
        "author_provenance":json.loads((Path(__file__).parents[1]/"vendor/usermirrorer/PROVENANCE.json").read_text())}
    (args.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"prepared":len(scenes),"skipped":len(skipped)},ensure_ascii=False))


if __name__=="__main__":
    main()
