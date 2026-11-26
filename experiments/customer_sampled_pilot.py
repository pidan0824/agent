"""Freeze a small Customer shortlist pilot selected from complete official Train."""
import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path

from agent_simulation.data.catalog import Catalog
from agent_simulation.data.common import require, rows, sha256
from agent_simulation.datasets.customer_scenes import prepare_scenes
from agent_simulation.agents.retrieval.bge_encoder import bge_encoder
from agent_simulation.inference.requests import load_tokenizer, prompt_tokens
from experiments.cot_sft_pilot import sampled_customer_cases, decision_request, dump, write_json


def filter_sampled_cases(scenes, parent, identity, tokenizer, input_limit):
    """Filter only the actual frozen direct/concise prompts used by this pilot."""
    kept, cases, skipped = [], [], []
    for bundle in scenes:
        selected = parent[bundle["request"]["session_id"]]
        group = sampled_customer_cases(bundle, selected["partition"], selected["stratum"], identity)
        retained = []
        for case in group:
            lengths = {style: len(prompt_tokens(decision_request(case, style)["messages"], [], tokenizer))
                       for style in ("direct", "concise")}
            if max(lengths.values()) > input_limit:
                skipped.append({"source_request_id":case["source_request_id"], "scene_id":case["scene_id"],
                                "stage":case["stage"], "reason":"context_length", "input_tokens":lengths})
            else:
                retained.append(case)
        enabled = {c["scene_id"] for c in retained if c["stage"] == "response"}
        bundle["query_enabled"] = any(c["stage"] == "query" for c in retained)
        bundle["response_scenes"] = [c for c in bundle["response_scenes"] if c["candidate"]["scene_id"] in enabled]
        if retained:
            kept.append(bundle)
            cases.extend(retained)
    return kept, cases, skipped


def prepare(args):
    release = json.loads((args.dataset / "manifest.json").read_text())
    population = json.loads((args.population / "manifest.json").read_text())
    reference = json.loads((args.reference / "protocol.json").read_text())
    identity = sha256(args.dataset / "manifest.json")
    candidates = args.population / "candidates.parquet"
    require(not release.get("time_window_experiment") and population["population_scope"] == "complete_official_train"
            and population["dataset_manifest_sha256"] == identity and reference["dataset_manifest_sha256"] == identity,
            "Complete Train provenance mismatch")
    require(sha256(candidates) == population["candidates_sha256"], "Population statistics changed")
    require(args.fit_parents > 0 and args.fit_parents % 128 == 0, "Fit budget must be a positive multiple of 128")
    require(not args.output.exists() or args.resume, "Use new output or explicit resume")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    intervals = reference["time_intervals_inclusive"]
    counts = {"fit": args.fit_parents, "dev": 64, "audit": 128}
    plan = []
    with closing(Catalog(release["catalog"])) as catalog:
        require(sha256(catalog.directory / "manifest.json") == release["catalog_manifest_sha256"], "Catalog changed")
        c = catalog.connection
        c.execute("SET memory_limit='8GB'")
        c.execute("SET temp_directory=?", [str(args.output / "spill")])
        for partition, (lower, upper) in intervals.items():
            for behavior, base in {"no_action": 48, "click_only": 48, "click_purchase": 28, "purchase_only": 4}.items():
                quota = base * counts[partition] // 128
                selected = rows(c, """WITH actors AS (
                    SELECT *,row_number() OVER(PARTITION BY stratum,actor_id
                        ORDER BY complete_titles DESC,title_coverage DESC,selection_hash,session_id) actor_rank
                    FROM read_parquet(?) WHERE exclusion_reason IS NULL AND prior_interactions>=3 AND exposure_count>=2
                      AND history_cutoff_exclusive BETWEEN ? AND ? AND behavior=?),
                    scenes AS (SELECT *,row_number() OVER(PARTITION BY stratum
                        ORDER BY complete_titles DESC,actor_rank,title_coverage DESC,selection_hash,session_id) sample_rank
                        FROM actors)
                    SELECT session_id,actor_id,stratum,prior_interactions,exposure_count,history_cutoff_exclusive
                    FROM scenes ORDER BY sample_rank,stratum,selection_hash,session_id LIMIT ?""",
                    [str(candidates), lower, upper, behavior, quota])
                plan.extend({**r, "partition": partition} for r in selected)
        require(len({r["session_id"] for r in plan}) == len(plan), "A parent crosses partitions")
        protocol = dict(schema="customer-sampled-sft-pilot-v1", dataset_manifest_sha256=identity,
            source_sha256=sha256(Path(__file__)), population_sha256=sha256(candidates),
            time_intervals_inclusive=intervals, requested_parent_counts=counts, seed=args.seed,
            input_limit=args.input_limit, methods=["random","content","collaborative"],
            min_items=2, max_items=12, min_prior_interactions=3,
            scope="Outcome-conditioned shortlists from complete Train; existing sampler requires prior interactions; not cold-start or full-exposure performance",
            selection="Input completeness, user diversity and scene coverage; no model predictions; no cap on original exposure",
            audit_policy="No audit generation until all model choices are frozen", parent_plan=plan)
        path = args.output / "protocol.json"
        if path.exists():
            require(json.loads(path.read_text()) == protocol, "Resume protocol changed")
        else:
            write_json(path, protocol)
        partial = args.output / "source-requests.partial.jsonl"
        cached = {r["session_id"]: r for r in map(json.loads, partial.read_text().splitlines())} if partial.exists() else {}
        require(set(cached) <= {r["session_id"] for r in plan}, "Unexpected cached request")
        if len(cached) < len(plan):
            # Process-local caches accelerate repeated joins; shared source files remain read-only.
            probe = next(r["session_id"] for r in plan if r["session_id"] not in cached)
            before = catalog.request(probe, all_prior=True)
            database = c.execute("SELECT current_database()").fetchone()[0]
            original = '"' + database.replace('"', '""') + '".main.'
            actors = sorted({r["actor_id"] for r in plan})
            c.execute(f"""CREATE TEMP TABLE requests AS SELECT * FROM {original}requests
                WHERE split='train' AND time_index<=? AND user_id IN (SELECT unnest(?::BIGINT[]))""",
                [intervals["audit"][1], actors])
            c.execute(f"""CREATE TEMP TABLE ranking_observations AS SELECT o.* FROM {original}ranking_observations o
                SEMI JOIN requests r USING(session_id)""")
            c.execute("""CREATE TEMP TABLE pilot_background AS SELECT * FROM customer_background
                WHERE user_id IN (SELECT unnest(?::BIGINT[]))""", [actors])
            c.execute("CREATE OR REPLACE TEMP VIEW customer_background AS SELECT * FROM pilot_background")
            c.execute(f"""CREATE TEMP TABLE products AS SELECT p.* FROM {original}products p SEMI JOIN (
                SELECT unnest(impressed_item_ids) item_id FROM requests UNION
                SELECT unnest(recently_clicked_item_ids) FROM pilot_background UNION
                SELECT unnest(recently_purchased_item_ids) FROM pilot_background) i USING(item_id)""")
            require(catalog.request(probe, all_prior=True) == before, "Local cache changed source facts")
            with partial.open("a") as stream:
                for selected in plan:
                    sid = selected["session_id"]
                    if sid in cached:
                        continue
                    cached[sid] = catalog.request(sid, all_prior=True)
                    stream.write(dump(cached[sid]) + "\n"); stream.flush()
                    if len(cached) % 16 == 0:
                        print(dump({"source_requests": len(cached), "planned": len(plan)}), flush=True)
    source_requests = [cached[r["session_id"]] for r in plan]
    with closing(Catalog(release["catalog"])) as catalog:
        c = catalog.connection
        c.execute("SET memory_limit='8GB'")
        c.execute("SET temp_directory=?", [str(args.output / "spill")])
        database = c.execute("SELECT current_database()").fetchone()[0]
        original = '"' + database.replace('"', '""') + '".main.'
        cutoff = max(r["response_input"]["history_cutoff_exclusive"] for r in source_requests)
        ids = sorted({p["item_id"] for r in source_requests for p in r["response_input"]["displayed_items"]})
        c.execute(f"CREATE TEMP TABLE requests AS SELECT * FROM {original}requests WHERE time_index<=?", [cutoff])
        # Preserve global prior quotes, including other users, for every exposure item.
        c.execute(f"""CREATE TEMP TABLE ranking_observations AS SELECT o.* FROM {original}ranking_observations o
            SEMI JOIN requests r USING(session_id) WHERE o.item_id IN (SELECT unnest(?::BIGINT[]))""", [ids])
        print(dump({"phase": "sampling", "parents": len(source_requests), "exposure_items": len(ids)}), flush=True)
        encoder, _ = bge_encoder(args.bge_path, device=args.bge_device)
        scenes, skipped = prepare_scenes(catalog, source_requests, encoder, seed=args.seed)
    tokenizer = load_tokenizer(args.tokenizer, local_files_only=True)
    parent = {r["session_id"]: r for r in plan}
    scenes, cases, length_skipped = filter_sampled_cases(scenes, parent, identity, tokenizer, args.input_limit)
    skipped.extend(length_skipped)
    require(cases and len({r["case_id"] for r in cases}) == len(cases), "Empty or duplicate cases")
    for name, values in (("cases.jsonl", cases), ("scenes.jsonl", scenes), ("skipped.jsonl", skipped)):
        (args.output / name).write_text("".join(dump(r) + "\n" for r in values))
    write_json(args.output / "manifest.json", dict(status="complete", schema=protocol["schema"],
        dataset_manifest_sha256=identity, cases_sha256=sha256(args.output / "cases.jsonl"),
        scenes_sha256=sha256(args.output / "scenes.jsonl"), protocol_sha256=sha256(args.output / "protocol.json"),
        source_requests_sha256=sha256(partial), source_sha256=protocol["source_sha256"],
        counts=dict(Counter(r["partition"]+"/"+r["stage"] for r in cases)),
        retained_parents=len({r["source_request_id"] for r in cases}),
        original_exposure_over_32=sum(r["exposure_count"]>32 for r in plan),
        skip_reasons=dict(Counter(r["reason"] for r in skipped)),
        collaborative_fit_cutoff_exclusive=min(r["response_input"]["history_cutoff_exclusive"] for r in source_requests)))
    print(dump(json.loads((args.output / "manifest.json").read_text())), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "population", "reference", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--fit-parents", type=int, default=256)
    p.add_argument("--input-limit", type=int, default=6656)
    p.add_argument("--seed", type=int, default=20261126)
    p.add_argument("--tokenizer", type=Path, default=Path("models/Qwen3.5-9B"))
    p.add_argument("--bge-path", type=Path, default=Path("models/bge-small-zh-v1.5"))
    p.add_argument("--bge-device", default="cpu")
    p.add_argument("--resume", action="store_true")
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
