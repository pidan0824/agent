"""Customer shortlists, structured EKB, and UserMirrorer preference selection.

Both fixed information and autonomous read-only information acquisition are run
on frozen records. Query is always generated separately, with current query and
exposure hidden. Only Response has categorical option entropy. No test export.
"""
import argparse
import asyncio
from collections import defaultdict
from contextlib import ExitStack, closing
from copy import copy, deepcopy
import json
from pathlib import Path
import numpy as np

from agent_simulation.agents.agentic import AgenticOptions
from agent_simulation.data.common import require, sha256
from agent_simulation.datasets.customer_candidates import construct_candidates, candidate_coverage
from agent_simulation.datasets.labels import SFTDataset, select_record_ids
from agent_simulation.evaluation.customer_filtering import OptionPredictor, probability_record, export_training, add_filter_arguments, validate_filter_options
from agent_simulation.evaluation.micro.customer_agentic import FrozenSampleTools
from agent_simulation.evaluation.trajectories import generate_trajectory, OutcomeScorer, score_group, digest
from agent_simulation.inference.agentic import AgenticClient
from agent_simulation.inference.requests import load_tokenizer, ContextCapacityError


class CurationTools(FrozenSampleTools):
    def __init__(self, request, stage, *, mode, theory=True, history_limit=10):
        super().__init__(request, stage, theory=theory)
        require(mode in ("fixed", "autonomous"), "Unknown information mode")
        self.mode, self.history_limit = mode, history_limit
        if mode == "fixed":
            self.schemas = []
            self.parameters = {}

    def initial_context(self):
        context = super().initial_context()
        context["information_mode"] = self.mode
        if "candidate_protocol" in self.context:
            context["candidate_protocol"] = self.context["candidate_protocol"]
        if self.mode == "fixed":
            context["observed_history"] = deepcopy(self.history[-self.history_limit:])
            if self.stage == "response":
                context["displayed_items"] = deepcopy(self.displayed)
        return context


def encoded(value):
    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=default)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--request-id", type=int, action="append")
    p.add_argument("--count", type=int, default=0, help="Prepared scenes: 0 uses all; raw alternatives require an explicit positive count")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--sample-concurrency", type=int, default=2)
    p.add_argument("--scenes", type=Path, help="Prepared sampled scenes from datasets.customer_scenes")
    p.add_argument("--candidate-policies", nargs="+", choices=["sampled", "full", "retrieved"], default=["sampled"])
    p.add_argument("--candidate-count", type=int, default=10)
    p.add_argument("--modes", nargs="+", choices=["fixed", "autonomous"], default=["fixed"])
    p.add_argument("--strong-url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--strong-model", default="Qwen3-32B")
    p.add_argument("--strong-tokenizer", type=Path, default=Path("models/Qwen3-32B"))
    p.add_argument("--weak-url", default="http://127.0.0.1:8002/v1")
    p.add_argument("--weak-model", default="Qwen3.5-9B")
    p.add_argument("--weak-tokenizer", type=Path, default=Path("models/Qwen3.5-9B"))
    p.add_argument("--predictor-url", help="Optional independent trained behavior predictor endpoint")
    p.add_argument("--predictor-model")
    p.add_argument("--predictor-tokenizer", type=Path)
    p.add_argument("--bge-path", type=Path, default=Path("models/bge-small-zh-v1.5"))
    p.add_argument("--temperature", type=float, default=.7)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-model-calls", type=int, default=6)
    p.add_argument("--max-tool-calls", type=int, default=8)
    p.add_argument("--context-limit", type=int, default=16384)
    p.add_argument("--query-min", type=float, default=.7)
    p.add_argument("--response-min", type=float, default=.8)
    p.add_argument("--max-pairs", type=int, default=10000)
    p.add_argument("--filter-column", choices=["diff_model_uncertainty", "diff_all_uncertainty", "diff_data_uncertainty", "teacher_entropy", "teacher_logloss"], default="diff_model_uncertainty")
    p.add_argument("--sampling-column", choices=["entropy", "logloss"], default="entropy")
    p.add_argument("--ascending", action="store_true")
    p.add_argument("--theory", action=argparse.BooleanOptionalAction, default=True)
    add_filter_arguments(p)
    return p


def summarize(records, candidate_records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["stage"], row["mode"], row["policy"], row["variant"])].append(row)
    result = []
    for key, rows in groups.items():
        valid = [r for r in rows if r["trajectory"]["status"] == "ok"]
        item = dict(zip(["stage", "mode", "policy", "variant"], key))
        item.update(total=len(rows), valid=len(valid), invalid=len(rows)-len(valid),
                    tool_calls=sum(len(r["trajectory"]["trace"].get("tools", [])) for r in rows))
        for field in ["direct_score", "argmax_score", "sampled_score", "mean_item_entropy_bits", "direct_click_f1", "direct_purchase_f1", "direct_exact_match"]:
            values = [r[field] for r in valid if r.get(field) is not None]
            item[field] = float(np.mean(values)) if values else None
            if field.startswith("direct_") and values:
                item[field + "_all_attempts"] = float(sum(values) / len(rows))
        result.append(item)
    return {"arms": result, "candidate_coverage": candidate_records,
            "interpretation": "Frozen-scene generation and retention diagnostics; this entry point constructs training trajectories.",
            "entropy_protocol": "One common weak predictor for both generators; factorized 4-state item distributions. Query uses semantic matching, no fabricated categorical entropy."}


def main(argv=None):
    args = parser().parse_args(argv)
    validate_filter_options(query_min=args.query_min, response_min=args.response_min,
        click_min=args.click_min, purchase_min=args.purchase_min, max_pairs=args.max_pairs,
        logloss_min_gap=args.logloss_min_gap, action_min_gap=args.action_min_gap)
    require(args.samples > 0 and args.sample_concurrency > 0 and args.count >= 0 and args.candidate_count > 0, "Positive sample/list counts required")
    require(args.samples == 1 or args.temperature > 0, "Repeated sampling requires nonzero temperature")
    require(len(set(args.modes)) == len(args.modes) and len(set(args.candidate_policies)) == len(args.candidate_policies), "Duplicate comparison arms")
    require(not args.output.exists(), "Use a new output directory")
    prepared = {}
    if "sampled" in args.candidate_policies:
        require(args.scenes is not None, "Prepare sampled scenes first: python -m agent_simulation.datasets.customer_scenes")
        from agent_simulation.datasets.customer_scenes import read_scenes
        prepared, _ = read_scenes(args.scenes, dataset=args.dataset, split=args.split)
        require(prepared, "No prepared scenes survived; inspect skipped.jsonl")
    require(prepared or args.count > 0, "Raw protocols require an explicit positive --count")
    require(all(v is None for v in (args.predictor_url,args.predictor_model,args.predictor_tokenizer))
            or all(v is not None for v in (args.predictor_url,args.predictor_model,args.predictor_tokenizer)),
            "Supply predictor URL, model and tokenizer together")
    args.output.mkdir(parents=True)
    require(0 <= args.query_min <= 1 and 0 <= args.response_min <= 1, "Quality thresholds must be in [0,1]")
    scorer = OutcomeScorer(args.bge_path)
    rng = np.random.default_rng(args.seed)
    records, coverage = [], []
    with ExitStack() as resources:
        data = resources.enter_context(closing(SFTDataset(args.dataset, args.strong_tokenizer)))
        ids = args.request_id or ((sorted(prepared)[:args.count] if args.count else sorted(prepared)) if prepared else select_record_ids(
            data.catalog.connection, args.dataset, "customer", args.split, args.count, args.seed))
        require(not prepared or set(ids) <= set(prepared), "Request missing from prepared scenes")
        require(ids and len(ids) == len(set(ids)), "Empty or duplicate request IDs")
        members = {r[0] for r in data.catalog.connection.execute(
            "SELECT session_id FROM read_parquet(?) WHERE session_id IN (SELECT unnest(?::BIGINT[]))",
            [str(args.dataset / "customer" / args.split / "labels.parquet"), ids]).fetchall()}
        require(members == set(ids), "Request not in selected release split")
        models = {}
        for variant, url, name, path in [("strong", args.strong_url, args.strong_model, args.strong_tokenizer),
                                         ("weak", args.weak_url, args.weak_model, args.weak_tokenizer)]:
            models[variant] = resources.enter_context(closing(AgenticClient(url, name,
                data.tokenizer if variant == "strong" else load_tokenizer(path, local_files_only=True),
                temperature=args.temperature, max_new_tokens=args.max_new_tokens, seed=args.seed,
                context_limit=args.context_limit,
                log_path=args.output / (variant + "-calls.jsonl"))))
            models[variant].check_models()
        predictor_client = models["weak"]
        if args.predictor_model is not None:
            predictor_client = resources.enter_context(closing(AgenticClient(args.predictor_url,args.predictor_model,
                load_tokenizer(args.predictor_tokenizer,local_files_only=True), context_limit=args.context_limit,
                max_new_tokens=1)))
            predictor_client.check_models()
        predictor = OptionPredictor(predictor_client, context_limit=args.context_limit)
        encoder = None
        if "retrieved" in args.candidate_policies:
            from agent_simulation.agents.retrieval.bge_encoder import bge_encoder
            encoder, _ = bge_encoder(args.bge_path, device="cpu")
        manifest = {"schema": "customer-usermirrorer-curation-v1", "args": vars(args), "request_ids": ids,
                    "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
                    "scenes_manifest_sha256": sha256(args.scenes / "manifest.json") if prepared else None,
                    "source_sha256": {name: sha256(Path(__file__).parents[1] / name) for name in (
                        "evaluation/customer_curation.py", "evaluation/customer_filtering.py", "evaluation/customer_quality.py",
                        "datasets/customer_candidates.py", "agents/prompts/customer_ekb.py", "inference/agentic.py")},
                    "author_provenance": json.loads((Path(__file__).parents[1] / "vendor/usermirrorer/PROVENANCE.json").read_text()),
                    "target_isolation": "Author scene construction uses target insertion offline. Generator/predictor receive frozen inputs, never current action labels or positive positions.",
                    "candidate_protocol": "Author outcome-conditioned scenes by default. full/retrieved remain explicit alternative protocols.",
                    "historical_test_use": "Existing project checkpoints may have used test for model selection; this is not a fresh blind test."}
        def path_json(obj):
            if isinstance(obj, Path): return str(obj)
            raise TypeError(type(obj).__name__)
        (args.output / "manifest.json").write_text(json.dumps(manifest, default=path_json, ensure_ascii=False, indent=2)+"\n")
        with (args.output / "records.jsonl").open("w") as output:
            for identity in ids:
                request = (prepared[identity]["request"] if prepared else data.catalog.request(identity, all_prior=True))
                actor = request["query_input"]["profile"]["user_id"]
                for stage in ("query", "response"):
                    policies = ["hidden"] if stage == "query" else args.candidate_policies
                    for policy in policies:
                        adapted = deepcopy(request)
                        candidate = None
                        if stage == "response" and policy == "sampled":
                            adapted = deepcopy(prepared[identity]["request"])
                            candidate = deepcopy(prepared[identity]["candidate"])
                        elif stage == "response":
                            raw = data.catalog.request(identity, all_prior=True) if prepared else request
                            adapted["response_input"], candidate = construct_candidates(
                                raw["response_input"], policy=policy, limit=args.candidate_count, encoder=encoder)
                        for mode in args.modes:
                            binding = CurationTools(adapted, stage, mode=mode, theory=args.theory)
                            options = AgenticOptions(max_model_calls=1 if mode == "fixed" else args.max_model_calls,
                                                      max_tool_calls=args.max_tool_calls, theory=args.theory)
                            async def generate_samples():
                                semaphore = asyncio.Semaphore(args.sample_concurrency)
                                async def one(variant, model, index):
                                    async with semaphore:
                                        # Immutable per-sample seed; HTTP transport/tokenizer may
                                        # be shared, but the mutable seed must never be shared.
                                        sampled_model = copy(model)
                                        sampled_model.seed = args.seed + index
                                        trajectory = await generate_trajectory(binding, sampled_model, role="customer",
                                            actor_id=actor, stage=stage, record_id=identity, source_split=args.split,
                                            catalog_identity=manifest["dataset_manifest_sha256"], options=options, sample_index=index)
                                        row = {"stage": stage, "mode": mode, "policy": policy, "variant": variant, "tid": index,
                                            "cid": digest([identity, mode, policy, args.theory, binding.fingerprint]),
                                            "dataset": "KuaiSearch/" + (candidate["domain"] if candidate else "Query"),
                                            "source_split": args.split, "trajectory": trajectory, "prob": None}
                                        if stage == "response" and trajectory["status"] == "ok":
                                            try:
                                                row["raw_item_probabilities"] = await asyncio.to_thread(predictor.score, binding, trajectory)
                                            except ContextCapacityError as error:
                                                row["probability_error"] = str(error)
                                        print(identity, stage, policy, mode, variant, index, trajectory["status"], flush=True)
                                        return row
                                return await asyncio.gather(*(one(variant, model, index)
                                    for variant, model in models.items() for index in range(args.samples)))
                            pending = asyncio.run(generate_samples())
                            # Neither generation nor probability scoring below can access targets.
                            target = request[stage + "_target"]
                            score_group([r["trajectory"] for r in pending], target, scorer,
                                        query_min=args.query_min,response_min=args.response_min,
                                        click_min=args.click_min,purchase_min=args.purchase_min)
                            if stage == "response":
                                status = candidate_coverage(candidate, target)
                                coverage.append({"request_id": identity, "mode": mode, **candidate, **status})
                            for row in pending:
                                t = row["trajectory"]
                                if t["status"] == "ok":
                                    row["direct_score"] = t["score"]
                                    if stage == "response":
                                        row["direct_click_f1"] = t["outcome_metrics"]["clicked_item_ids_f1"]
                                        row["direct_purchase_f1"] = t["outcome_metrics"]["purchased_item_ids_f1"]
                                        row["direct_exact_match"] = all(set(t["final_action"][field]) == set(target[field])
                                            for field in ("clicked_item_ids", "purchased_item_ids"))
                                if stage == "response":
                                    row.update(reachable=status["reachable"], behavior_group=status["behavior_group"],
                                               choice_cnt=candidate["candidate_count"])
                                    if row.get("raw_item_probabilities"):
                                        values = probability_record(row.pop("raw_item_probabilities"), candidate["candidate_item_ids"], target, rng=rng)
                                        row.update(values)
                                        for name in ("argmax", "sampled"):
                                            row[name+"_score"] = scorer.score(stage, row[name+"_action"], target)["score"]
                                output.write(encoded(row) + "\n"); output.flush()
                                records.append(row)
        exported = export_training(records,args.output,max_pairs=args.max_pairs,filter_column=args.filter_column,
            sampling_column=args.sampling_column,ascending=args.ascending,
            query_min=args.query_min,response_min=args.response_min,click_min=args.click_min,purchase_min=args.purchase_min,
            pair_protocol=args.pair_protocol,logloss_min_gap=args.logloss_min_gap,action_min_gap=args.action_min_gap)
        report = summarize(records, coverage)
        report.update(exported)
        report.update(split=args.split, filtering={"metric":args.filter_column,"sampling":args.sampling_column,
            "groups":["domain","candidate_count","observed_behavior_group"],"all_raw_rows_retained":True,
            "test_training_export":False})
        (args.output / "report.json").write_text(encoded(report)+"\n")
        print(encoded({k:v for k,v in report.items() if k != "candidate_coverage"}), flush=True)


if __name__ == "__main__":
    main()
