"""Select a larger, scene-covered Customer fit set from full Train without new CoT."""
import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path

from agent_simulation.data.catalog import Catalog
from agent_simulation.data.common import require, rows, sha256
from agent_simulation.inference.requests import load_tokenizer, prompt_tokens, tokenizer_identity
from experiments.cot_sft_pilot import customer_case, decision_request, dump, load_cases, write_json


def prepare(args):
    require(args.requests >= 256 and args.requests % 128 == 0, "Use a multiple of 128, at least 256 to cover scenes")
    require(0 < args.input_limit <= 7936, "Reserve at least 256 output tokens in the 8192 window")
    release = json.loads((args.dataset / "manifest.json").read_text())
    population = json.loads((args.population / "manifest.json").read_text())
    identity = sha256(args.dataset / "manifest.json")
    require(not release.get("time_window_experiment") and
            population.get("population_scope") == "complete_official_train" and
            population["dataset_manifest_sha256"] == identity, "Full Train provenance required")
    candidates = args.population / "candidates.parquet"
    require(sha256(candidates) == population["candidates_sha256"], "Population statistics changed")
    reference = load_cases(args.cohorts)
    protocol = json.loads((args.cohorts / "protocol.json").read_text())
    reference_manifest = json.loads((args.cohorts / "manifest.json").read_text())
    require(sha256(args.cohorts / "protocol.json") == reference_manifest["protocol_sha256"],
            "Frozen reference protocol changed")
    require(protocol["dataset_manifest_sha256"] == identity, "Reference cohort release mismatch")
    lower, upper = protocol["time_intervals_inclusive"]["fit"]
    heldout = {r["source_request_id"] for r in reference
               if r["role"] == "customer" and r["partition"] != "fit"}
    quotas = {k: v * args.requests // 128 for k, v in
              {"no_action": 48, "click_only": 48, "click_purchase": 28, "purchase_only": 4}.items()}
    require(not args.output.exists() or args.resume, "Use a new output or explicit --resume")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    tokenizer = load_tokenizer(args.tokenizer, local_files_only=True)
    with closing(Catalog(release["catalog"])) as catalog:
        require(sha256(catalog.directory / "manifest.json") == release["catalog_manifest_sha256"],
                "Source catalog identity changed")
        c = catalog.connection
        c.execute("SET temp_directory=?", [str(args.output / "spill")])
        c.execute("""CREATE TEMP TABLE fit_population AS
            SELECT * FROM read_parquet(?) WHERE exclusion_reason IS NULL AND exposure_count<=32
              AND history_cutoff_exclusive BETWEEN ? AND ?""", [str(candidates), lower, upper])
        c.execute("""CREATE TEMP TABLE fit_ranked AS WITH actors AS (
            SELECT *,row_number() OVER(PARTITION BY stratum,actor_id
                ORDER BY title_coverage DESC,(prior_interactions>0) DESC,selection_hash,session_id) actor_rank
            FROM fit_population)
            SELECT *,row_number() OVER(PARTITION BY stratum
                ORDER BY complete_titles DESC,actor_rank,title_coverage DESC,
                         (prior_interactions>0) DESC,selection_hash,session_id) fit_rank FROM actors""")
        plan = []
        for behavior, budget in quotas.items():
            strata = [r[0] for r in c.execute(
                "SELECT DISTINCT stratum FROM fit_population WHERE behavior=? ORDER BY stratum", [behavior]).fetchall()]
            require(strata and budget >= len(strata), "Budget does not cover each nonempty scene")
            for index, stratum in enumerate(strata):
                quota = budget // len(strata) + (index < budget % len(strata))
                pool = rows(c, """SELECT session_id,actor_id,prior_requests,prior_interactions,fit_rank
                    FROM fit_ranked WHERE stratum=? ORDER BY fit_rank LIMIT ?""", [stratum, quota * 8])
                plan.append({"stratum": stratum, "quota": quota, "pool": pool})
        settings = {"schema": "customer-fit-quality-selection-v1", "source_sha256": sha256(Path(__file__)),
                    "dataset_manifest_sha256": identity, "population_sha256": sha256(candidates),
                    "reference_cases_sha256": sha256(args.cohorts / "cases.jsonl"),
                    "fit_interval": [lower, upper], "request_budget": args.requests,
                    "input_limit": args.input_limit, "tokenizer": str(args.tokenizer.resolve()),
                    "tokenizer_sha256": tokenizer_identity(tokenizer, tokenizer.chat_template),
                    "quota_policy": "Keep four behavior quotas; distribute each across every eligible history/exposure scene",
                    "ranking": population["ranking"], "no_new_cot": True,
                    "scope": "Fit only; full original exposure <=32; original dev/audit unchanged"}
        plan_path = args.output / "plan.json"
        frozen = {"settings": settings, "scenes": plan}
        if plan_path.exists():
            require(json.loads(plan_path.read_text()) == frozen, "Resume plan identity changed")
        else:
            write_json(plan_path, frozen)
        ids = {r["session_id"] for scene in plan for r in scene["pool"]}
        require(not ids.intersection(heldout), "Fit selection overlaps held-out requests")
        # Process-local caches retain all earlier Train events for these users.
        # The source catalog stays read-only; no shared table is changed.
        probes = [next(s["pool"][0]["session_id"] for s in plan
                       if s["pool"] and f"/{band}/" in s["stratum"]) for band in ("cold", "established")]
        before = {i: catalog.request(i, all_prior=True) for i in probes}
        actors = sorted({r["actor_id"] for s in plan for r in s["pool"]})
        database = c.execute("SELECT current_database()").fetchone()[0]
        source = '"' + database.replace('"', '""') + '".main.'
        c.execute(f"""CREATE TEMP TABLE requests AS SELECT * FROM {source}requests
            WHERE split='train' AND time_index<=? AND user_id IN (SELECT unnest(?::BIGINT[]))""", [upper, actors])
        c.execute(f"""CREATE TEMP TABLE ranking_observations AS
            SELECT o.* FROM {source}ranking_observations o SEMI JOIN requests r USING(session_id)""")
        require(all(catalog.request(i, all_prior=True) == before[i] for i in probes),
                "Process-local cache changed Customer facts")
        print(dump({"cache_requests": c.execute("SELECT count(*) FROM requests").fetchone()[0],
                    "cache_exposures": c.execute("SELECT count(*) FROM ranking_observations").fetchone()[0]}), flush=True)
        partial = args.output / "prepared-cases.partial.jsonl"
        cached = {r["case_id"]: r for r in map(json.loads, partial.read_text().splitlines())} if partial.exists() else {}
        require(all(r["source_request_id"] in ids and r["partition"] == "fit" and
                    r["dataset_manifest_sha256"] == identity for r in cached.values()), "Invalid resumed fit cases")
        cases, skipped, selected, coverage = [], [], [], []
        for scene in plan:
            kept = 0
            for item in scene["pool"]:
                sid = item["session_id"]
                keys = [f"{stage}:{sid}" for stage in ("query", "response")]
                group = ([cached[k] for k in keys] if all(k in cached for k in keys) else
                         customer_case(catalog, sid, "fit", scene["stratum"], identity))
                lengths = [len(prompt_tokens(decision_request(r, "direct")["messages"], [], tokenizer)) for r in group]
                if max(lengths) > args.input_limit:
                    skipped.append({"id": sid, "stratum": scene["stratum"], "reason": "input_tokens", "tokens": lengths})
                    continue
                if not all(k in cached for k in keys):
                    with partial.open("a") as stream:
                        stream.write("".join(dump(r) + "\n" for r in group))
                    cached.update({r["case_id"]: r for r in group})
                cases.extend(group)
                selected.append({**item, "stratum": scene["stratum"], "input_tokens": lengths})
                kept += 1
                if kept == scene["quota"]:
                    break
            coverage.append({"stratum": scene["stratum"], "requested": scene["quota"], "selected": kept})
            print(dump(coverage[-1]), flush=True)
        require(cases and len({r["case_id"] for r in cases}) == len(cases), "Empty or duplicate fit cases")
        for name, values in (("cases.jsonl", cases), ("selected.jsonl", selected), ("skipped.jsonl", skipped)):
            (args.output / name).write_text("".join(dump(r) + "\n" for r in values))
        write_json(args.output / "manifest.json", {
            "status": "complete", **settings, "cases_sha256": sha256(args.output / "cases.jsonl"),
            "plan_sha256": sha256(plan_path), "selected_requests": len(selected),
            "capacity_skips": len(skipped), "scene_coverage": coverage,
            "with_prior_actions": sum(r["prior_interactions"] > 0 for r in selected),
            "counts": dict(Counter(r["stage"] for r in cases)),
            "quality_limit": "Input completeness and prior evidence proxies; not semantic or behavioral noise certification"})
        print("PREPARED", len(selected), "requests", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "population", "cohorts", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=Path("models/Qwen3.5-9B"))
    p.add_argument("--requests", type=int, default=512)
    p.add_argument("--input-limit", type=int, default=6656)
    p.add_argument("--resume", action="store_true")
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
