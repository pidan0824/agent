"""Prepare frozen UserMirrorer Customer training scenes, without LLM inference."""
import argparse
from collections import Counter
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path

from agent_simulation.data.catalog import Catalog
from agent_simulation.data.common import require, rows, sha256
from agent_simulation.datasets.customer_candidate_sampling import SampledCandidateBuilder
from agent_simulation.datasets.labels import select_record_ids


def freeze_prices(catalog, scene):
    """Use the same strictly-prior price lookup for positives and distractors."""
    request = scene["request"]
    cutoff = request["response_input"]["history_cutoff_exclusive"]
    ids = scene["candidate"]["candidate_item_ids"]
    observations = catalog.connection.execute("""
        SELECT o.item_id, arg_max(o.price_raw,r.time_index) AS price_raw, max(r.time_index) AS observed_time_index
        FROM ranking_observations o JOIN requests r USING(session_id)
        WHERE r.time_index < ? AND o.item_id IN (SELECT unnest(?::BIGINT[]))
          AND isfinite(o.price_raw) AND o.price_raw > 0 GROUP BY o.item_id
    """, [cutoff,ids]).fetchall()
    prices = {item:(price,time) for item,price,time in observations}
    for product in request["response_input"]["displayed_items"]:
        price,time = prices.get(product["item_id"],(None,None))
        product["price_raw"] = price
        product["price_status"] = "observed" if price is not None else "unknown"
        product["price_observed_time_index"] = time
    request["response_input"]["candidate_protocol"] = "Synthetic choice list; prices are strictly-prior observations, not current quotes."
    scene["candidate"]["price_policy"] = "Latest valid strictly-prior observation for every candidate; never target-only current prices"
    return scene


def prepare_scenes(catalog, requests, encoder, *, min_items=2, max_items=12, seed=42,
                   min_history=3, methods=("random", "content", "collaborative")):
    require(bool(requests), "Empty request selection")
    cutoffs = [r["response_input"]["history_cutoff_exclusive"] for r in requests]
    # Shared item facts support CF history; candidate eligibility is request-exposure-only.
    # No per-user LLM summaries or hidden current action labels enter features.
    products = rows(catalog.connection, """
        WITH available AS (
          SELECT item_id,min(time_index) AS first_seen_time_index FROM
          (SELECT unnest(impressed_item_ids) AS item_id,time_index FROM requests WHERE time_index < ?)
          GROUP BY item_id)
        SELECT p.*,a.first_seen_time_index FROM products p JOIN available a USING(item_id)
    """, [max(cutoffs)])
    by_id = {p["item_id"]: p for p in products}
    for request in requests:
        target_ids = {p["item_id"] for p in request["response_input"]["displayed_items"]}
        for product in request["response_input"]["displayed_items"]:
            if product["item_id"] in target_ids and product["item_id"] not in by_id:
                by_id[product["item_id"]] = {**deepcopy(product),
                    "first_seen_time_index": request["response_input"]["history_cutoff_exclusive"]}
    interactions = rows(catalog.connection, """
        SELECT user_id,unnest(list_distinct(list_concat(clicked_item_ids,purchased_item_ids))) AS item_id,
               time_index FROM requests WHERE split='train' AND time_index < ?
    """, [max(cutoffs)])
    builder = SampledCandidateBuilder(list(by_id.values()), interactions, encoder,
        min_items=min_items,max_items=max_items,seed=seed,min_history=min_history,methods=methods,
        fit_cutoff=min(cutoffs))
    scenes, skipped = [], []
    expected = {"insufficient_prior_interactions", "empty_available_item_pool",
                "empty_constructed_list", "insufficient_distractors"}
    for request in requests:
        bundle = {"request": deepcopy(request), "query_enabled": True, "response_scenes": []}
        target = request["response_target"]
        positives = sorted(set(target["clicked_item_ids"]) | set(target["purchased_item_ids"]))
        for item in positives or [None]:
            try:
                scene = builder.build(request, target_item_id=item)
                bundle["response_scenes"].append(freeze_prices(catalog, scene))
            except ValueError as error:
                if str(error) not in expected:
                    raise
                skipped.append({"session_id": request["session_id"], "stage": "response",
                                "target_item_id": item, "reason": str(error)})
        # A Response construction failure never removes its independent Query.
        scenes.append(bundle)
    return scenes, skipped


def filter_lengths(scenes, tokenizer, limit, *, modes=("fixed","autonomous")):
    """Count the actual generation prompt/schema/tools, without model inference."""
    from agent_simulation.agents.agentic import initial_agentic_request, AgenticOptions
    from agent_simulation.evaluation.customer_curation import CurationTools
    from agent_simulation.evaluation.trajectories import trajectory_messages
    from agent_simulation.inference.requests import prompt_tokens
    require(limit>0, "Length limit must be positive")
    kept,skipped=[],[]
    for bundle in deepcopy(scenes):
        def counts_for(request, stage):
            counts = {}
            for mode in modes:
                binding = CurationTools(request, stage, mode=mode)
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
        positives = set(source["response_target"]["clicked_item_ids"]) | set(source["response_target"]["purchased_item_ids"])
        for child in bundle["response_scenes"]:
            require(child["request"]["session_id"] == source["session_id"], "Mixed source request")
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
    p.add_argument("--request-id",type=int,action="append")
    p.add_argument("--count",type=int,default=100)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--min-items",type=int,default=2)
    p.add_argument("--max-items",type=int,default=12)
    p.add_argument("--min-history",type=int,default=3)
    p.add_argument("--methods",nargs="+",choices=["random","content","collaborative"],default=["random","content","collaborative"])
    p.add_argument("--bge-path",type=Path,default=Path("models/bge-small-zh-v1.5"))
    p.add_argument("--bge-device",default="cuda",help="Embedding device, e.g. cuda, cuda:1 or cpu; honor CUDA_VISIBLE_DEVICES")
    p.add_argument("--max-input-tokens",type=int,default=0,help="Optional actual-prompt length filter; 0 disables")
    p.add_argument("--tokenizer",type=Path,default=Path("models/Qwen3-32B"))
    args=p.parse_args(argv)
    require(args.count>0 and not args.output.exists(), "Use a positive count and new output directory")
    release=json.loads((args.dataset/"manifest.json").read_text())
    with closing(Catalog(release["catalog"])) as catalog:
        require(sha256(catalog.directory/"manifest.json")==release["catalog_manifest_sha256"], "Catalog changed")
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
        scenes,skipped=prepare_scenes(catalog,requests,encoder,min_items=args.min_items,max_items=args.max_items,
            seed=args.seed,min_history=args.min_history,methods=args.methods)
    if args.max_input_tokens:
        from agent_simulation.inference.requests import load_tokenizer
        scenes,length_skipped=filter_lengths(scenes,load_tokenizer(args.tokenizer,local_files_only=True),args.max_input_tokens)
        skipped.extend(length_skipped)
    args.output.mkdir(parents=True)
    (args.output/"scenes.jsonl").write_text("".join(json.dumps(s,ensure_ascii=False,allow_nan=False)+"\n" for s in scenes))
    (args.output/"skipped.jsonl").write_text("".join(json.dumps(s,ensure_ascii=False)+"\n" for s in skipped))
    manifest={"schema":"customer-single-positive-scenes-v2","dataset":str(args.dataset.resolve()),
        "dataset_manifest_sha256":sha256(args.dataset/"manifest.json"),"split":args.split,
        "args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "scenes_sha256":sha256(args.output/"scenes.jsonl"),"prepared":len(scenes),"skipped":len(skipped),
        "query_scenes":sum(s["query_enabled"] for s in scenes),
        "response_scenes":sum(len(s["response_scenes"]) for s in scenes),
        "skip_reasons":dict(Counter(s["reason"] for s in skipped)),
        "protocol":"Outcome-conditioned shortlists sampled within recorded exposures, without Query/category gates; not full-exposure evaluation.",
        "candidate_pool":"request_exposures","query_category_filter":False,
        "query":"Current Query target/exposures remain absent from query_input.",
        "collaborative_fit_cutoff_exclusive":min(r["response_input"]["history_cutoff_exclusive"] for r in requests),
        "author_provenance":json.loads((Path(__file__).parents[1]/"vendor/usermirrorer/PROVENANCE.json").read_text())}
    (args.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"prepared":len(scenes),"skipped":len(skipped)},ensure_ascii=False))


if __name__=="__main__":
    main()
