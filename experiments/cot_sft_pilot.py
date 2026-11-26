"""Budgeted, temporal, same-base Customer/Seller SFT pilot. No service/process management."""
import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import time

from agent_simulation.data.catalog import Catalog
from agent_simulation.data.common import literal, require, rows, sha256
from agent_simulation.evaluation.customer_curation import CurationTools
from agent_simulation.evaluation.customer_evidence import encode_context, check_evidence
from agent_simulation.evaluation.seller_quality import check_seller_evidence, seller_action_quality
from agent_simulation.agents.context.seller_features import seller_behavior_features, with_evidence_id
from agent_simulation.agents.decisions import action_output
from agent_simulation.agents.agentic import initial_agentic_request, AgenticOptions, teacher_output
from agent_simulation.agents.schemas import strict_json, validate_arguments
from agent_simulation.agents.prompts.instructions import FACT_RULES
from agent_simulation.evaluation.trajectories import digest
from agent_simulation.inference.requests import load_tokenizer, prompt_tokens
from agent_simulation.inference.agentic import AgenticClient


def dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def decision_request(case, style):
    require(style in ("direct", "concise", "original"), "Unknown rationale style")
    stage, role = case["stage"], case["role"]
    frozen = case.get("direct_request")
    direct = stage == "query" or style == "direct"
    if frozen and direct:
        return deepcopy(frozen)
    context = case.get("context", {})
    displayed = [p["item_id"] for p in context.get("candidate_context", [])]
    schemas = ([frozen["final_schema"]] if frozen else
        action_output(stage, displayed_ids=displayed if stage == "response" else None,
        price_ids=context.get("requested_price_item_ids") if stage == "seller" else None)["final_schemas"])
    if direct:
        instructions = FACT_RULES + "\nPredict the next observed action from the supplied strictly prior facts. "
        instructions += ("Generate the next Chinese query; the current query and exposure are hidden." if stage == "query" else
            "Predict clicks and purchases separately; no action is valid. Relevance does not prove a click." if stage == "response" else
            "Predict the next observed quote, not an unobserved optimal price. Preserve raw price units.")
        output = schemas[0]
    else:
        if role == "customer":
            from agent_simulation.agents.prompts.customer_agentic import fixed_customer_instructions
            from agent_simulation.agents.prompts.customer_ekb import concise_ekb_instructions
            instructions = (FACT_RULES + "\n" + concise_ekb_instructions()
                            if style == "concise" else fixed_customer_instructions())
        else:
            from agent_simulation.agents.prompts.seller_agentic import (
                concise_seller_instructions, agentic_seller_instructions)
            instructions = concise_seller_instructions() if style == "concise" else agentic_seller_instructions()
        output = teacher_output(schemas, customer_ekb=role == "customer")
    if frozen:
        messages = deepcopy(frozen["messages"])
        messages[0] = {"role": "system", "content": instructions + "\nReturn only JSON. Schema: " + dump(output)}
        return {"messages": messages, "tools": [], "final_schema": output}
    messages = [{"role": "system", "content": instructions + "\nReturn only JSON. Schema: " + dump(output)},
                {"role": "user", "content": encode_context(context) if role == "customer" else dump(context)}]
    messages, tools, _ = initial_agentic_request(messages, [], AgenticOptions(max_model_calls=1, theory=not direct))
    return {"messages": messages, "tools": tools, "final_schema": output}


def sampled_customer_cases(bundle, partition, stratum, dataset_identity):
    """Reuse production observed-action prompts and freeze the same child action space."""
    from agent_simulation.evaluation.customer_curation import observed_action_rows
    from agent_simulation.evaluation.customer_contract import single_item_schema
    source = bundle["request"]
    observed = observed_action_rows(source, source["session_id"], "train", dataset_identity, bundle=bundle)
    cases = []
    for row in observed:
        schema = (single_item_schema(row["candidate_item_ids"]) if row["stage"] == "response"
                  else action_output("query")["final_schemas"][0])
        case = {k: deepcopy(v) for k, v in row.items() if k not in ("messages", "tools", "trajectory_id")}
        case.update(case_id=row["stage"] + ":" + row["scene_id"], partition=partition, stratum=stratum,
                    direct_request={"messages": deepcopy(row["messages"][:-1]), "tools": [], "final_schema": schema})
        cases.append(case)
    return cases


def customer_case(catalog, identity, partition, stratum, dataset_identity):
    original = catalog.request(int(identity), all_prior=True)
    result = []
    for stage in ("query", "response"):
        context = CurationTools(original, stage, mode="fixed", theory=stage=="response").initial_context()
        result.append(dict(case_id=f"{stage}:{identity}", role="customer", stage=stage,
            actor_id=original[stage+"_input"]["profile"]["user_id"], source_request_id=int(identity),
            source_split="train", partition=partition, stratum=stratum, context=context,
            target=deepcopy(original[stage+"_target"]), dataset_manifest_sha256=dataset_identity))
    return result


def seller_case(catalog, label, partition, dataset_identity):
    seller, item, cutoff = label["seller_id"], label["item_id"], label["history_cutoff_exclusive"]
    history = rows(catalog.connection, """SELECT * FROM seller_history
        WHERE seller_id=? AND item_id=? AND time_index<? ORDER BY time_index""", [seller,item,cutoff])
    require(history and all(r["time_index"]<cutoff for r in history), "Missing past Seller evidence")
    features = seller_behavior_features(history, seller_id=seller, cutoff=cutoff, item_ids=[item])
    last = history[-1]
    previous = features["products"][0]["last_valid_price_raw"]
    require(previous == label["previous_price_raw"], "Seller reference differs from frozen cohort")
    from agent_simulation.data.catalog import DataRepository
    from agent_simulation.agents.context.market import context_view, SELLER_MARKET_NOTE
    repo = DataRepository(catalog.connection,catalog.config,cutoff)
    market = context_view(repo.market_position(seller,item_id=item,before_time_index=cutoff))
    context = with_evidence_id({
        "profile":{"seller_id":seller}, "history_cutoff_exclusive":cutoff,
        "requested_price_item_ids":[item],
        "owned_products":[{k:last.get(k) for k in ("item_id","item_title","brand_name","category_level3_name")}],
        "behavior_features":features,
        "recent_quote_observations":[{k:r.get(k) for k in (
            "time_index","price_raw","price_status","exposure_records","click_records","purchase_records")}
            for r in history[-10:]],
        "market_position":market, "market_semantics":SELLER_MARKET_NOTE})
    return dict(case_id=label["sample_id"],role="seller",stage="seller",actor_id=seller,
        source_request_id=label["sample_id"],source_split="train",partition=partition,
        stratum=label["direction"],context=context,
        target={"prices":[{"item_id":item,"price_raw":label["target_price_raw"]}]},
        previous_prices={str(item):previous}, dataset_manifest_sha256=dataset_identity)


def prepare(args):
    require(not args.output.exists() or args.resume, "Use a new cohort directory or explicit --resume")
    source = json.loads((args.dataset/"manifest.json").read_text())
    require(not source.get("time_window_experiment"), "Use full Lite Train")
    require(source["counts"]["customer"]["train"] == 539012, "Unexpected full Lite population; review protocol")
    identity = sha256(args.dataset/"manifest.json")
    tokenizer = load_tokenizer(args.tokenizer,local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    partial = args.output/"prepared-cases.partial.jsonl"
    cached = {r["case_id"]: r for r in (json.loads(line) for line in partial.read_text().splitlines())} if partial.exists() else {}
    require(not cached or args.resume, "Existing partial cohort needs --resume")
    require(all(r["dataset_manifest_sha256"] == identity for r in cached.values()), "Partial cohort release mismatch")
    skipped, cases, cohorts = [], [], []
    def persist(group):
        new = [r for r in group if r["case_id"] not in cached]
        with partial.open("a") as stream:
            for r in new:
                stream.write(dump(r)+"\n")
                cached[r["case_id"]] = r
    with closing(Catalog(source["catalog"])) as catalog:
        c=catalog.connection
        c.execute(f"SET temp_directory={literal(args.output/'spill')}")
        lo,hi=c.execute("SELECT min(history_cutoff_exclusive),max(history_cutoff_exclusive) FROM read_parquet(?)",
                       [str(args.dataset/"customer/train/labels.parquet")]).fetchone()
        cut1,cut2=lo+int((hi-lo)*.70),lo+int((hi-lo)*.85)
        interval={"fit":(lo,cut1),"dev":(cut1+1,cut2),"audit":(cut2+1,hi)}
        protocol = {
            "schema":"dual-agent-sft-pilot-v1","dataset_manifest_sha256":identity,
            "time_intervals_inclusive":interval,"seed":args.seed,
            "fit_counts":{"customer":128,"seller":128},"dev_counts":{"customer":64,"seller":64},
            "audit_counts":{"customer":128,"seller":128},
            "scope":"Temporal, stratified, full-exposure pilot with <=32 Customer candidates; not full-population or end-to-end simulation",
            "primary":{"customer":"Response mean of click/purchase request F1; report Query cosine separately",
                       "seller":"Lower mean relative absolute price error and nondecreasing direction macro-F1"},
            "selection":"Fixed before model generation; input capacity filtering only; Test untouched",
            "audit_policy":"Audit reserved until all model choices are frozen",
            "limits":{"input_tokens":args.input_limit,"total_context":8192,"max_output":768}}
        if (args.output/"protocol.json").exists():
            require(json.loads((args.output/"protocol.json").read_text()) == json.loads(dump(protocol)),
                    "Partial cohort protocol changed")
        else:
            write_json(args.output/"protocol.json",protocol)
        # Population-wide inputs already computed by the complete-Train selector.
        population=literal(args.population/"candidates.parquet")
        require(json.loads((args.population/"manifest.json").read_text())["dataset_manifest_sha256"]==identity,
                "Selection population release mismatch")
        for partition,(lower,upper) in interval.items():
            quotas={"no_action":48,"click_only":48,"click_purchase":28,"purchase_only":4}
            if partition=="dev":quotas={k:v//2 for k,v in quotas.items()}
            for behavior,quota in quotas.items():
                pool=rows(c,f"""WITH ranked AS (
                    SELECT *,row_number() OVER(PARTITION BY actor_id ORDER BY selection_hash) user_rank
                    FROM read_parquet({population}) WHERE exclusion_reason IS NULL AND exposure_count<=32
                      AND history_cutoff_exclusive BETWEEN ? AND ? AND behavior=?
                ) SELECT session_id,stratum FROM ranked
                ORDER BY user_rank,selection_hash LIMIT ?""",[lower,upper,behavior,quota*5])
                kept=0
                for r in pool:
                    keys=[f"{stage}:{r['session_id']}" for stage in ("query","response")]
                    group=([cached[k] for k in keys] if all(k in cached for k in keys) else
                           customer_case(catalog,r["session_id"],partition,r["stratum"],identity))
                    lengths=[len(prompt_tokens(decision_request(v,"concise")["messages"],[],tokenizer)) for v in group]
                    if max(lengths)>args.input_limit:
                        skipped.append(dict(role="customer",id=r["session_id"],partition=partition,reason="input_tokens",tokens=lengths))
                        continue
                    persist(group)
                    cases.extend(group)
                    cohorts.append(dict(role="customer",id=r["session_id"],partition=partition,stratum=r["stratum"]))
                    kept+=1
                    if kept%8==0:print(dump(dict(prepared="customer",partition=partition,behavior=behavior,kept=kept,quota=quota)),flush=True)
                    if kept==quota:break
                require(kept==quota,f"Insufficient {partition}/{behavior} capacity-checked requests")
        # Seeded broad price sample, then match its recorded strictly earlier quote.
        labels=literal(args.dataset/"seller/train/labels.parquet")
        c.execute(f"""CREATE TEMP TABLE pilot_seller_labels AS
            SELECT * FROM read_parquet({labels})
            ORDER BY sha256(sample_id||':{args.seed}') LIMIT 100000""")
        c.execute("""CREATE TEMP TABLE pilot_seller_previous AS
            SELECT l.*,t.previous_time FROM pilot_seller_labels l JOIN seller_targets t
            ON l.seller_id=t.seller_id AND l.item_id=t.item_id AND l.history_cutoff_exclusive=t.time_index""")
        c.execute("""CREATE TEMP TABLE pilot_seller AS
            SELECT l.*,h.price_raw AS previous_price_raw,
              CASE WHEN l.target_price_raw=h.price_raw THEN 'keep'
                   WHEN l.target_price_raw>h.price_raw THEN 'up' ELSE 'down' END AS direction
            FROM pilot_seller_previous l JOIN seller_history h
            ON l.seller_id=h.seller_id AND l.item_id=h.item_id AND l.previous_time=h.time_index
            WHERE h.price_status='observed' AND isfinite(h.price_raw) AND h.price_raw>0""")
        for partition,(lower,upper) in interval.items():
            for direction,quota in (("keep",64),("up",32),("down",32)):
                if partition=="dev":quota//=2
                pool=rows(c,f"""WITH ranked AS (
                    SELECT *,row_number() OVER(PARTITION BY seller_id ORDER BY sha256(sample_id||':{args.seed}')) seller_rank
                    FROM pilot_seller WHERE history_cutoff_exclusive BETWEEN ? AND ? AND direction=?
                ) SELECT * FROM ranked ORDER BY seller_rank,sha256(sample_id||':{args.seed}') LIMIT ?""",
                    [lower,upper,direction,quota*5])
                kept=0
                for label in pool:
                    case=cached.get(label["sample_id"])
                    if case is None:
                        case=seller_case(catalog,label,partition,identity)
                    length=len(prompt_tokens(decision_request(case,"concise")["messages"],[],tokenizer))
                    if length>args.input_limit:
                        skipped.append(dict(role="seller",id=label["sample_id"],partition=partition,reason="input_tokens",tokens=length))
                        continue
                    persist([case])
                    cases.append(case)
                    cohorts.append(dict(role="seller",id=label["sample_id"],partition=partition,stratum=direction))
                    kept+=1
                    if kept%8==0:print(dump(dict(prepared="seller",partition=partition,direction=direction,kept=kept,quota=quota)),flush=True)
                    if kept==quota:break
                require(kept==quota,f"Insufficient {partition}/{direction} Seller samples")
    require(len({r["case_id"] for r in cases})==len(cases),"Overlapping cohorts")
    for name,records in (("cases.jsonl",cases),("cohorts.jsonl",cohorts),("skipped.jsonl",skipped)):
        (args.output/name).write_text("".join(dump(r)+"\n" for r in records))
    write_json(args.output/"manifest.json",{"status":"complete","schema":"dual-agent-pilot-cohorts-v1",
        "dataset_manifest_sha256":identity,"cases_sha256":sha256(args.output/"cases.jsonl"),
        "protocol_sha256":sha256(args.output/"protocol.json"),"source_sha256":sha256(Path(__file__)),
        "counts":dict(Counter(r["role"]+"/"+r["partition"]+"/"+r["stage"] for r in cases)),
        "capacity_skips":len(skipped)})
    print("PREPARED",len(cases),flush=True)


def load_cases(path):
    manifest=json.loads((path/"manifest.json").read_text())
    require(manifest.get("status")=="complete" and sha256(path/"cases.jsonl")==manifest["cases_sha256"],
            "Incomplete or changed cohorts")
    return [json.loads(line) for line in (path/"cases.jsonl").read_text().splitlines()]


def quality(case, action, rationale, messages, style, semantic_review=None):
    trajectory={"role":case["role"],"stage":case["stage"],"response_style":"ekb" if style!="direct" else "direct",
                "messages":messages,"rationale":rationale,"final_action":action,"semantic_review":semantic_review or {}}
    if case["role"]=="customer":
        if case["stage"]=="query":
            return {"accepted":False,"reason":"queries_use_observed_actions"}
        exact=all(set(action[k])==set(case["target"][k]) for k in ("clicked_item_ids","purchased_item_ids"))
        evidence=check_evidence(trajectory)
        references = rationale.get("evidence_refs", {})
        cited = any(references.get(stage) for stage in ("stimulus","knowledge","evaluation"))
        return {"accepted":exact and evidence["accepted"] and (style=="direct" or cited),
                "exact_action":exact,"has_cited_evidence":cited,"evidence":evidence}
    evidence=check_seller_evidence(trajectory) if style!="direct" else {"accepted":True}
    directions=seller_action_quality(action,case["target"],case["previous_prices"])
    predicted={p["item_id"]:p["price_raw"] for p in action["prices"]}
    error=sum(abs(predicted[p["item_id"]]-p["price_raw"])/p["price_raw"] for p in case["target"]["prices"])/len(predicted)
    return {"accepted":error<=.05 and directions["accepted"] and evidence["accepted"],
            "relative_price_error":error,"direction":directions,"evidence":evidence}


def generate(args):
    from agent_simulation.inference.client import ModelOutputError
    cases=[r for r in load_cases(args.cohorts) if r["partition"]==args.partition and
           (args.role=="both" or r["role"]==args.role) and (args.stage=="all" or r["stage"]==args.stage)]
    if args.limit:
        require(args.partition=="fit","Evaluate the complete frozen held-out cohort")
        groups=defaultdict(list)
        for row in cases:
            groups[row["role"]+"/"+row["stratum"].split("/")[0]].append(row)
        for values in groups.values():
            values.sort(key=lambda row:digest([args.seed,row["case_id"]]))
        cases=[values[depth] for depth in range(max(map(len,groups.values())))
               for _,values in sorted(groups.items()) if depth<len(values)][:args.limit]
    require(cases,"Empty generation selection")
    requests={case["case_id"]:decision_request(case,args.style) for case in cases}
    identity={"requests_sha256":digest(requests),"cases_sha256":sha256(args.cohorts/"cases.jsonl"),"ids":[r["case_id"] for r in cases],
              "base_config_sha256":sha256(args.tokenizer/"config.json"),
              "model":args.model,"tokenizer":str(args.tokenizer.resolve()),"style":args.style,
              "partition":args.partition,"max_new_tokens":args.max_new_tokens,"temperature":args.temperature,
              "seed":args.seed,"concurrency":args.concurrency}
    if args.output.exists():
        require(json.loads((args.output/"manifest.json").read_text())==identity,"Resume identity mismatch")
    else:
        args.output.mkdir(parents=True)
        write_json(args.output/"manifest.json",identity)
    path=args.output/"predictions.jsonl"
    prior=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    require(len({r["case_id"] for r in prior})==len(prior),"Duplicate completed predictions")
    completed={r["case_id"] for r in prior}
    require(completed<=set(identity["ids"]),"Unexpected resumed prediction")
    if len(completed)==len(cases):
        if not (args.output/"completed.json").exists():
            write_json(args.output/"completed.json",{"status":"complete","records":len(cases),
                       "wall_seconds_this_run":0,"resumed_records":len(prior)})
        print("Already complete",len(cases),flush=True)
        return
    tokenizer=load_tokenizer(args.tokenizer,local_files_only=True)
    with closing(AgenticClient(args.url,args.model,tokenizer,max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,seed=args.seed,timeout=300,context_limit=8192,stream=True,
        log_path=args.output/f"calls-{len(prior)}-{time.time_ns()}.jsonl")) as client, path.open("a") as stream:
        client.check_models()
        async def run():
            semaphore=asyncio.Semaphore(args.concurrency)
            async def one(case):
                async with semaphore:
                    request=requests[case["case_id"]]
                    row={k:case[k] for k in ("case_id","role","stage","partition","stratum","target")}
                    row.update(status="invalid",style=args.style,request_sha256=digest(request))
                    started=time.monotonic()
                    try:
                        reply=await client.acomplete(request,call_id=case["case_id"])
                        value=strict_json(reply["content"])
                        validate_arguments(value,request["final_schema"])
                        direct=args.style=="direct" or case["stage"]=="query"
                        action=value if direct else value["action"]
                        rationale={} if direct else value["rationale"]
                        messages=request["messages"]+[reply]
                        row.update(status="ok",prediction=action,rationale=rationale,messages=messages,
                                   quality=quality(case,action,rationale,messages,args.style))
                    except (ModelOutputError,ValueError) as error:
                        row.update(error=str(error))
                    row["seconds"]=time.monotonic()-started
                    stream.write(dump(row)+"\n");stream.flush()
                    print(dump({k:row[k] for k in ("case_id","status","seconds")}),flush=True)
            await asyncio.gather(*(one(case) for case in cases if case["case_id"] not in completed))
        started=time.monotonic()
        asyncio.run(run())
        write_json(args.output/"completed.json",{"status":"complete","records":len(cases),
                   "wall_seconds_this_run":time.monotonic()-started,"resumed_records":len(prior)})


def export(args):
    cases=load_cases(args.cohorts)
    by_id={r["case_id"]:r for r in cases}
    require(not args.output.exists(),"New SFT export directory required")
    args.output.mkdir(parents=True)
    rows_by_role=defaultdict(list)
    for case in cases:
        if case["partition"]!="fit":continue
        req=decision_request(case,"direct")
        meta={k:case[k] for k in ("role","actor_id","stage","source_split","source_request_id","dataset_manifest_sha256")}
        meta.update(initial_state_id=case.get("initial_state_id") or digest([case["case_id"],case.get("context",case.get("direct_request"))]),
            trajectory_id=digest([case["case_id"],"observed",case["target"]]),
            scene_id=case.get("scene_id",case["case_id"]),tools=[],
            messages=req["messages"]+[{"role":"assistant","content":dump(case["target"])}])
        meta.update({k:deepcopy(case[k]) for k in ("target","action_mode","evaluation_protocol","source_response_target","candidate_item_ids") if k in case})
        rows_by_role[case["role"]].append(meta)
    counts={role:{"observed":len(values),"rationale":0} for role,values in rows_by_role.items()}
    for directory in args.teachers:
        identity=json.loads((directory/"manifest.json").read_text())
        require(identity["partition"]=="fit","Never export held-out generated rows")
        require(identity["cases_sha256"]==sha256(args.cohorts/"cases.jsonl"),"Teacher cohort mismatch")
        review_path=directory/"semantic-reviews.json"
        reviews=json.loads(review_path.read_text()) if review_path.exists() else {}
        for line in (directory/"predictions.jsonl").read_text().splitlines():
            row=json.loads(line)
            case=by_id[row["case_id"]]
            require(case["partition"]=="fit","Held-out teacher target")
            if row["status"]!="ok" or row["style"]=="direct":continue
            request=decision_request(case,row["style"])
            require(row["style"]==identity["style"] and row["request_sha256"]==digest(request)
                    and row["messages"][:-1]==request["messages"],"Teacher input differs from frozen case")
            require(row["messages"][-1]["role"]=="assistant" and not row["messages"][-1].get("tool_calls"),
                    "Teacher export requires its original final response")
            validate_arguments(strict_json(row["messages"][-1]["content"]),request["final_schema"])
            review=reviews.get(row["case_id"])
            if review is not None:
                require(review["messages_sha256"]==digest(row["messages"]),"Review transcript mismatch")
                require(review["verdict"] in ("pass","fail"),"Unknown semantic review verdict")
            verdict=quality(case,row["prediction"],row["rationale"],row["messages"],row["style"],review)
            if not verdict["accepted"]:continue
            meta={k:case[k] for k in ("role","actor_id","stage","source_split","source_request_id","dataset_manifest_sha256")}
            meta.update(initial_state_id=case.get("initial_state_id") or digest([case["case_id"],case.get("context",case.get("direct_request"))]),
                trajectory_id=digest([case["case_id"],identity["model"],row["messages"]]),
                scene_id=case.get("scene_id",case["case_id"]),tools=[],messages=row["messages"],evidence_quality=verdict,semantic_review=review)
            meta.update({k:deepcopy(case[k]) for k in ("target","action_mode","evaluation_protocol","source_response_target","candidate_item_ids") if k in case})
            rows_by_role[case["role"]].append(meta)
            counts[case["role"]]["rationale"]+=1
    for role,records in rows_by_role.items():
        require(len({r["trajectory_id"] for r in records})==len(records),"Duplicate SFT rows")
        (args.output/(role+".jsonl")).write_text("".join(dump(r)+"\n" for r in records))
    write_json(args.output/"manifest.json",{"cohorts_sha256":sha256(args.cohorts/"cases.jsonl"),
        "counts":counts,"held_out_exported":False})
    print(dump(counts),flush=True)



def comparison(args):
    import numpy as np
    from sklearn.metrics import f1_score
    cases={r["case_id"]:r for r in load_cases(args.cohorts)}
    predictions={}
    manifests={}
    for name,path in (("zero_shot",args.baseline),("sft",args.sft)):
        manifests[name]=json.loads((path/"manifest.json").read_text())
        require(manifests[name]["style"]=="direct","Compare the same direct decision protocol")
        require(manifests[name]["cases_sha256"]==sha256(args.cohorts/"cases.jsonl"),"Prediction cohort differs")
        records=[json.loads(line) for line in (path/"predictions.jsonl").read_text().splitlines()]
        predictions[name]={r["case_id"]:r for r in records}
        require(len(records)==len(predictions[name]) and set(predictions[name])==set(manifests[name]["ids"]),
                "Incomplete evaluation; invalid outputs must stay in the denominator")
    require(set(predictions["zero_shot"])==set(predictions["sft"]),"Unpaired evaluations")
    require(manifests["zero_shot"]["base_config_sha256"]==manifests["sft"]["base_config_sha256"],
            "Baseline and SFT base architectures differ")
    require(all(manifests["zero_shot"][key]==manifests["sft"][key]
                for key in ("max_new_tokens","temperature","seed","partition")),
            "Evaluation decoding or partition differs")
    for key in predictions["zero_shot"]:
        require(predictions["zero_shot"][key]["request_sha256"]==predictions["sft"][key]["request_sha256"],
                "Evaluation input or schema changed")
        require(cases[key]["partition"] in ("dev","audit"),"No training-set success claims")
    rng=np.random.default_rng(20261126)
    def paired(left,right,larger=True,groups=None):
        detail = {}
        if groups is not None:
            require(len(groups)==len(left)==len(right), "Parent grouping differs from scores")
            indices = defaultdict(list)
            for index, group in enumerate(groups):
                indices[group].append(index)
            detail = {"scene_n":len(left), "aggregation":"Mean child score within parent, then equal parent weights; bootstrap parents"}
            left = [float(np.mean([left[i] for i in members])) for members in indices.values()]
            right = [float(np.mean([right[i] for i in members])) for members in indices.values()]
        a=np.asarray(left,float)
        b=np.asarray(right,float)
        delta=b-a if larger else a-b
        samples=[float(delta[rng.integers(0,len(delta),len(delta))].mean()) for _ in range(2000)]
        return {**detail,"n":len(delta),"baseline_mean":float(a.mean()),"sft_mean":float(b.mean()),
                "improvement":float(delta.mean()),"paired_bootstrap_95pct":np.quantile(samples,[.025,.975]).tolist()}
    report={"protocol":"paired fixed inputs, same base family, same output budget; conditional behavior pilot",
            "partition":manifests["sft"]["partition"],"models":{n:m["model"] for n,m in manifests.items()},
            "invalid":{n:sum(r["status"]!="ok" for r in rs.values()) for n,rs in predictions.items()}}
    response=[k for k in predictions["sft"] if cases[k]["stage"]=="response"]
    def response_groups(keys):
        return ([cases[k]["source_request_id"] for k in keys]
                if all(cases[k].get("action_mode")=="single_item" for k in keys) else None)
    if response:
        values={name:{} for name in predictions}
        head_values={name:defaultdict(dict) for name in predictions}
        for name,records in predictions.items():
            for key in response:
                r=records[key]
                heads=[]
                for field in ("clicked_item_ids","purchased_item_ids"):
                    p=set(r["prediction"][field]) if r["status"]=="ok" else set()
                    t=set(cases[key]["target"][field])
                    score=(2*len(p&t)/(len(p)+len(t)) if p or t else 1.) if r["status"]=="ok" else 0.
                    heads.append(score)
                    head_values[name][field][key]=score
                values[name][key]=sum(heads)/2
        report["customer_response"]=paired([values["zero_shot"][k] for k in response],[values["sft"][k] for k in response],groups=response_groups(response))
        report["customer_response"]["by_head"]={}
        for field in ("clicked_item_ids","purchased_item_ids"):
            head=paired([head_values["zero_shot"][field][k] for k in response],
                        [head_values["sft"][field][k] for k in response],groups=response_groups(response))
            for label,positive in (("observed_positive",True),("observed_empty",False)):
                keys=[k for k in response if bool(cases[k]["target"][field])==positive]
                head[label]=paired([head_values["zero_shot"][field][k] for k in keys],
                                   [head_values["sft"][field][k] for k in keys],groups=response_groups(keys)) if keys else None
            report["customer_response"]["by_head"][field]=head
        report["customer_response"]["by_stratum"]={stratum:paired(
            [values["zero_shot"][k] for k in response if cases[k]["stratum"]==stratum],
            [values["sft"][k] for k in response if cases[k]["stratum"]==stratum],
            groups=response_groups([k for k in response if cases[k]["stratum"]==stratum]))
            for stratum in sorted({cases[k]["stratum"] for k in response})}
    queries=[k for k in predictions["sft"] if cases[k]["stage"]=="query"]
    if queries:
        from agent_simulation.agents.retrieval.bge_encoder import bge_encoder
        encoder,_=bge_encoder(Path("models/bge-small-zh-v1.5"),device="cpu")
        targets=np.asarray(encoder.encode_many([cases[k]["target"]["query"] for k in queries]),float)
        qvalues={}
        for name,records in predictions.items():
            valid=[records[k]["status"]=="ok" and bool(records[k]["prediction"]["query"].strip()) for k in queries]
            report["invalid"][name]+=sum(records[k]["status"]=="ok" and not ok for k,ok in zip(queries,valid))
            # The encoder rejects empty strings. Placeholder embeddings are always masked to zero.
            texts=[records[k]["prediction"]["query"] if ok else "invalid query output" for k,ok in zip(queries,valid)]
            embeddings=np.asarray(encoder.encode_many(texts),float)
            norms=np.linalg.norm(targets,axis=1)*np.linalg.norm(embeddings,axis=1)
            scores=np.sum(targets*embeddings,axis=1)/np.maximum(norms,1e-12)
            qvalues[name]=[float(v) if ok else 0. for ok,v in zip(valid,scores)]
        report["customer_query_cosine"]=paired(qvalues["zero_shot"],qvalues["sft"])
    seller=[k for k in predictions["sft"] if cases[k]["stage"]=="seller"]
    if seller:
        errors,macro,directions={},{},{}
        actual=[]
        for key in seller:
            case=cases[key]
            item=case["target"]["prices"][0]
            previous=case["previous_prices"][str(item["item_id"])]
            actual.append(int(np.sign(item["price_raw"]-previous)))
        for name,records in {**predictions,"persistence":None}.items():
            losses,predicted=[],[]
            for key in seller:
                case=cases[key]
                item=case["target"]["prices"][0]
                previous=case["previous_prices"][str(item["item_id"])]
                valid=records is None or records[key]["status"]=="ok"
                price=previous if records is None else (records[key]["prediction"]["prices"][0]["price_raw"] if valid else None)
                losses.append(abs(price-item["price_raw"])/item["price_raw"] if valid else 1.)
                predicted.append(int(np.sign(price-previous)) if valid else 99)
            errors[name]=losses
            directions[name]=predicted
            macro[name]=float(f1_score(actual,predicted,labels=[-1,0,1],average="macro",zero_division=0))
        report["seller_price"]=paired(errors["zero_shot"],errors["sft"],larger=False)
        report["seller_price"]["direction_macro_f1"]=macro
        report["seller_price"]["persistence_relative_error"]=float(np.mean(errors["persistence"]))
        report["seller_price"]["false_change_rate_on_keep"]={
            name:float(np.mean([p!=0 for a,p in zip(actual,ds) if a==0]))
            for name,ds in directions.items()}
        report["seller_price"]["invalid_penalty"]="Invalid output has relative error 1 and incorrect direction; invalid count also reported"
    report["pilot_gate"]={}
    if response:
        response_report=report["customer_response"]
        checks={"response_mean_improved":response_report["improvement"]>0}
        for field,name,strict in (("clicked_item_ids","positive_click_f1_not_lower",False),
                                  ("purchased_item_ids","positive_purchase_f1_improved",True)):
            positive=response_report["by_head"][field]["observed_positive"]
            if positive:
                checks[name]=positive["improvement"]>0 if strict else positive["improvement"]>=0
        if queries:
            checks["query_cosine_not_lower"]=report["customer_query_cosine"]["improvement"]>=0
        report["customer_safeguards"]=checks
        report["pilot_gate"]["customer"]=all(checks.values())
    if seller:
        report["pilot_gate"]["seller"]=(report["seller_price"]["improvement"]>0 and
             report["seller_price"]["direction_macro_f1"]["sft"]>=report["seller_price"]["direction_macro_f1"]["zero_shot"])
    report["interpretation"]="Development results guide experiments; only frozen audit results can support the final pilot conclusion"
    write_json(args.output,report)
    print(dump(report),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest="command",required=True)
    prep=subs.add_parser("prepare")
    prep.add_argument("--dataset",type=Path,required=True)
    prep.add_argument("--population",type=Path,required=True)
    prep.add_argument("--output",type=Path,required=True)
    prep.add_argument("--tokenizer",type=Path,default=Path("models/Qwen3.5-9B"))
    prep.add_argument("--input-limit",type=int,default=6656)
    prep.add_argument("--seed",type=int,default=20261126)
    prep.add_argument("--resume",action="store_true")
    gen=subs.add_parser("generate")
    gen.add_argument("--cohorts",type=Path,required=True)
    gen.add_argument("--output",type=Path,required=True)
    gen.add_argument("--partition",choices=("fit","dev","audit"),required=True)
    gen.add_argument("--role",choices=("customer","seller","both"),default="both")
    gen.add_argument("--stage",choices=("all","query","response","seller"),default="all")
    gen.add_argument("--style",choices=("direct","concise","original"),default="concise")
    gen.add_argument("--url",default="http://127.0.0.1:8026/v1")
    gen.add_argument("--model",default="Qwen3-32B")
    gen.add_argument("--tokenizer",type=Path,default=Path("models/Qwen3-32B"))
    gen.add_argument("--max-new-tokens",type=int,default=768)
    gen.add_argument("--temperature",type=float,default=0)
    gen.add_argument("--seed",type=int,default=20261126)
    gen.add_argument("--concurrency",type=int,default=4)
    gen.add_argument("--limit",type=int,default=0)
    exp=subs.add_parser("export")
    exp.add_argument("--cohorts",type=Path,required=True)
    exp.add_argument("--teachers",type=Path,nargs="*",default=[])
    exp.add_argument("--output",type=Path,required=True)
    score=subs.add_parser("compare")
    score.add_argument("--cohorts",type=Path,required=True)
    score.add_argument("--baseline",type=Path,required=True)
    score.add_argument("--sft",type=Path,required=True)
    score.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    {"prepare":prepare,"generate":generate,"export":export,"compare":comparison}[args.command](args)


if __name__=="__main__":
    main()
