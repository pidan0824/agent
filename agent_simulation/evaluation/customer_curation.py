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
from agent_simulation.evaluation.trajectories import generate_trajectory, trajectory_messages, OutcomeScorer, score_group, digest
from agent_simulation.evaluation.customer_quality import behavior_stratum
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
        from agent_simulation.evaluation.customer_evidence import attach_sources
        return attach_sources(context)


def fixed_response_requests(response_input, *, history_limit=10):
    """Use the exact fixed curation prompts and schemas in controlled evaluation."""
    from agent_simulation.agents.agentic import initial_agentic_request, teacher_output
    result = {}
    for style in ("direct", "ekb"):
        theory = style == "ekb"
        binding = CurationTools({"response_input": response_input}, "response",
                               mode="fixed", theory=theory, history_limit=history_limit)
        schemas, messages = trajectory_messages(binding, role="customer", stage="response", theory=theory)
        messages, tools, _ = initial_agentic_request(messages, [], AgenticOptions(max_model_calls=1, theory=theory))
        result[style] = {"messages": messages, "tools": tools,
                        "final_schema": teacher_output(schemas, customer_ekb=True) if theory else schemas[0]}
    return binding.initial_context(), result


def observed_action_rows(request, identity, split, dataset_identity):
    """Full-exposure ground truth, independent of teacher success or shortlist labels."""
    if split != "train":
        return []
    from agent_simulation.agents.schemas import validate_arguments
    from agent_simulation.agents.agentic import initial_agentic_request
    result = []
    for stage in ("query", "response"):
        binding = CurationTools(request, stage, mode="fixed", theory=False)
        schemas, prompt = trajectory_messages(binding, role="customer", stage=stage, theory=False)
        prompt, _, _ = initial_agentic_request(prompt, [], AgenticOptions(max_model_calls=1, theory=False))
        target = deepcopy(request[stage + "_target"])
        validate_arguments(target, schemas[0])
        state_id = digest([dataset_identity, identity, stage, prompt])
        result.append({"role": "customer", "actor_id": request[stage + "_input"]["profile"]["user_id"],
            "stage": stage, "source_split": split, "source_request_id": identity,
            "dataset_manifest_sha256": dataset_identity, "initial_state_id": state_id,
            "scene_id": str(identity) + ":full", "trajectory_id": digest([state_id, target, "observed"]),
            "tools": [], "messages": prompt + [{"role": "assistant", "content": encoded(target)}],
            "target": target, "response_style": "direct",
            "action_mode": "multi_item" if stage == "response" else "query",
            "evaluation_protocol": "independent_behavior_prediction",
            "behavior_source": "observed_action; full exposure; no generated rationale"})
    return result


def generation_jobs(request, *, bundle=None, policies=("sampled",), limit=10, encoder=None):
    """One Query per source request; one Response per positive (or empty outcome)."""
    jobs = []
    if bundle is None or bundle.get("query_enabled", True):
        jobs.append(("query", "hidden", deepcopy(request), None))
    for policy in policies:
        if policy == "sampled":
            require(bundle is not None, "Sampled generation requires prepared scenes")
            for scene in bundle["response_scenes"]:
                adapted = deepcopy(scene["request"])
                adapted["response_input"]["response_action_mode"] = "single_item"
                jobs.append(("response", policy, adapted, deepcopy(scene["candidate"])))
        else:
            adapted = deepcopy(request)
            adapted["response_input"], candidate = construct_candidates(
                request["response_input"], policy=policy, limit=limit, encoder=encoder)
            jobs.append(("response", policy, adapted, candidate))
    return jobs


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
    p.add_argument("--selection", type=Path, help="Complete-Train scene selection; all selected IDs, no count truncation")
    p.add_argument("--count", type=int, default=0, help="Prepared scenes: 0 uses all; raw alternatives require an explicit positive count")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sft-only", action="store_true", help="Generate strong teacher only; skip weak model, option predictor and DPO")
    p.add_argument("--samples", type=int, default=10)
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
    p.add_argument("--bge-device", default="cuda", help="BGE device for retrieval and Query scoring; honor CUDA_VISIBLE_DEVICES")
    p.add_argument("--stream-timing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--temperature", type=float, default=.7)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-model-calls", type=int, default=6)
    p.add_argument("--max-tool-calls", type=int, default=8)
    p.add_argument("--context-limit", type=int, default=16384)
    p.add_argument("--query-min", type=float, default=.7)
    p.add_argument("--response-min", type=float, default=.8)
    p.add_argument("--max-pairs", type=int, default=10000)
    p.add_argument("--filter-column", choices=["diff_model_uncertainty", "diff_all_uncertainty", "diff_data_uncertainty", "teacher_entropy", "teacher_logloss"], default="diff_model_uncertainty")
    p.add_argument("--sampling-column", choices=["entropy", "logloss"], default="entropy")
    p.add_argument("--ascending", action="store_true")
    p.add_argument("--response-styles", nargs="+", choices=["ekb", "direct"], help="Matched frozen-scene comparison; defaults to ekb, or direct with --no-theory")
    p.add_argument("--evaluation-only", action="store_true", help="Report all attempts without exporting training rows")
    p.add_argument("--theory", action=argparse.BooleanOptionalAction, default=True)
    add_filter_arguments(p)
    return p


def summarize(records, candidate_records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["stage"], row["mode"], row["policy"], row["variant"],
                row.get("response_style", "ekb"), row.get("dataset", "unknown"),
                row.get("choice_cnt"),
                behavior_stratum(row["trajectory"]["target"]) if row["stage"] == "response" else "query",
                behavior_stratum(row["trajectory"].get("source_response_target", row["trajectory"]["target"]))
                    if row["stage"] == "response" else "query")].append(row)
    result = []
    for key, rows in groups.items():
        valid = [r for r in rows if r["trajectory"]["status"] == "ok"]
        item = dict(zip(["stage", "mode", "policy", "variant", "response_style", "domain", "candidate_count", "behavior_group", "source_behavior_group"], key))
        item.update(total=len(rows), valid=len(valid), invalid=len(rows)-len(valid),
                    tool_calls=sum(len(r["trajectory"]["trace"].get("tools", [])) for r in rows))
        for field in ["direct_score", "argmax_score", "sampled_score", "mean_item_entropy_bits", "direct_click_f1", "direct_purchase_f1", "direct_click_recall", "direct_purchase_recall", "direct_exact_match", "logloss"]:
            values = [r[field] for r in valid if r.get(field) is not None]
            item[field] = float(np.mean(values)) if values else None
            item[field + "_count"] = len(values)
            if field.startswith("direct_") and not field.endswith("_recall") and values:
                item[field + "_all_attempts"] = float(sum(values) / len(rows))
        empty = [r for r in rows if r["stage"] == "response" and
                 not r["trajectory"].get("target", {}).get("clicked_item_ids")]
        item["no_click_count"] = len(empty)
        item["no_click_accuracy_all_attempts"] = (sum(r["trajectory"]["status"] == "ok" and
            not r["trajectory"]["final_action"]["clicked_item_ids"] for r in empty) / len(empty)) if empty else None
        item["evidence_passed"] = sum(r["trajectory"].get("evidence_quality", {}).get("accepted", False) for r in valid)
        result.append(item)
    return {"arms": result, "candidate_coverage": candidate_records,
            "interpretation": "Frozen-scene generation and retention diagnostics; this entry point constructs training trajectories.",
            "entropy_protocol": "One common predictor: sampled single-item lists use joint 1+3N actions "
                "(bits/decision); full/retrieved use factorized four-state item marginals (bits/item). "
                "Never aggregate those losses. Query has no categorical entropy.",
            "recall_protocol": "Recall is null when the true head is empty; report no-click exact match separately. "
                "Metrics cover all generated attempts, not outcome-selected samples."}


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
    require(not (args.selection and (args.request_id or args.scenes)),
            "Use selection, prepared scenes, or explicit request IDs separately")
    selected_ids, selection_manifest = None, None
    if args.selection:
        from agent_simulation.datasets.customer_selection import read_selection
        selected_ids, selection_manifest = read_selection(args.selection, dataset=args.dataset, split=args.split)
        require(args.count == 0, "Selection already fixes the budget; omit --count")
    require(prepared or selected_ids or args.count > 0, "Raw protocols require --selection or a positive --count")
    require(all(v is None for v in (args.predictor_url,args.predictor_model,args.predictor_tokenizer))
            or all(v is not None for v in (args.predictor_url,args.predictor_model,args.predictor_tokenizer)),
            "Supply predictor URL, model and tokenizer together")
    args.output.mkdir(parents=True)
    require(0 <= args.query_min <= 1 and 0 <= args.response_min <= 1, "Quality thresholds must be in [0,1]")
    scorer = OutcomeScorer(args.bge_path, bge_device=args.bge_device)
    rng = np.random.default_rng(args.seed)
    styles = args.response_styles or (["ekb"] if args.theory else ["direct"])
    require(len(set(styles)) == len(styles), "Duplicate response styles")
    records, coverage, observed = [], [], []
    with ExitStack() as resources:
        data = resources.enter_context(closing(SFTDataset(args.dataset, args.strong_tokenizer)))
        ids = selected_ids or args.request_id or ((sorted(prepared)[:args.count] if args.count else sorted(prepared)) if prepared else select_record_ids(
            data.catalog.connection, args.dataset, "customer", args.split, args.count, args.seed))
        require(not prepared or set(ids) <= set(prepared), "Request missing from prepared scenes")
        require(ids and len(ids) == len(set(ids)), "Empty or duplicate request IDs")
        members = {r[0] for r in data.catalog.connection.execute(
            "SELECT session_id FROM read_parquet(?) WHERE session_id IN (SELECT unnest(?::BIGINT[]))",
            [str(args.dataset / "customer" / args.split / "labels.parquet"), ids]).fetchall()}
        require(members == set(ids), "Request not in selected release split")
        models = {}
        model_specs = [("strong", args.strong_url, args.strong_model, args.strong_tokenizer),
                       ("weak", args.weak_url, args.weak_model, args.weak_tokenizer)]
        for variant, url, name, path in (model_specs[:1] if args.sft_only else model_specs):
            models[variant] = resources.enter_context(closing(AgenticClient(url, name,
                data.tokenizer if variant == "strong" else load_tokenizer(path, local_files_only=True),
                temperature=args.temperature, max_new_tokens=args.max_new_tokens, seed=args.seed,
                context_limit=args.context_limit, stream=args.stream_timing,
                log_path=args.output / (variant + "-calls.jsonl"))))
            models[variant].check_models()
        predictor_client = models.get("weak")
        if not args.sft_only and args.predictor_model is not None:
            predictor_client = resources.enter_context(closing(AgenticClient(args.predictor_url,args.predictor_model,
                load_tokenizer(args.predictor_tokenizer,local_files_only=True), context_limit=args.context_limit,
                max_new_tokens=1)))
            predictor_client.check_models()
        predictor = OptionPredictor(predictor_client, context_limit=args.context_limit) if not args.sft_only else None
        encoder = None
        if "retrieved" in args.candidate_policies:
            from agent_simulation.agents.retrieval.bge_encoder import bge_encoder
            encoder, _ = bge_encoder(args.bge_path, device=args.bge_device)
        manifest = {"schema": "customer-usermirrorer-curation-v2", "args": vars(args), "request_ids": ids,
                    "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
                    "scenes_manifest_sha256": sha256(args.scenes / "manifest.json") if prepared else None,
                    "source_sha256": {name: sha256(Path(__file__).parents[1] / name) for name in (
                        "evaluation/customer_curation.py", "evaluation/customer_filtering.py", "evaluation/customer_quality.py",
                        "evaluation/customer_evidence.py", "evaluation/customer_contract.py", "evaluation/trajectories.py",
                        "datasets/customer_candidates.py", "agents/prompts/customer_ekb.py", "inference/agentic.py")},
                    "selection_manifest_sha256": sha256(args.selection / "manifest.json") if args.selection else None,
                    "selection_population": selection_manifest["population"] if selection_manifest else None,
                    "evaluation_protocol": "A: independent Query; Response conditioned on recorded Query and frozen candidates. Not end-to-end simulation.",
                    "response_styles": styles, "evaluation_only": args.evaluation_only,
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
                jobs = generation_jobs(request, bundle=prepared.get(identity),
                    policies=args.candidate_policies, limit=args.candidate_count, encoder=encoder)
                for stage, policy, adapted, candidate in jobs:
                    if args.sft_only and args.split == "train" and not args.evaluation_only and stage == "query":
                        continue  # The original Query is exported without teacher generation.
                    scene_identity = candidate.get("scene_id", identity) if candidate else identity
                    arms = [(mode, style) for mode in args.modes
                            for style in (styles if stage == "response" else ["direct"])]
                    for mode, style in arms:
                        use_theory = stage == "response" and style == "ekb"
                        binding = CurationTools(adapted, stage, mode=mode, theory=use_theory)
                        options = AgenticOptions(max_model_calls=1 if mode == "fixed" else args.max_model_calls,
                                                  max_tool_calls=args.max_tool_calls, theory=use_theory)
                        async def generate_samples():
                            semaphore = asyncio.Semaphore(args.sample_concurrency)
                            async def one(variant, model, index):
                                async with semaphore:
                                    # Immutable per-sample seed; HTTP transport/tokenizer may
                                    # be shared, but the mutable seed must never be shared.
                                    sampled_model = copy(model)
                                    sampled_model.seed = args.seed + index
                                    trajectory = await generate_trajectory(binding, sampled_model, role="customer",
                                        actor_id=actor, stage=stage, record_id=scene_identity, source_split=args.split,
                                        catalog_identity=manifest["dataset_manifest_sha256"], options=options, sample_index=index)
                                    trajectory["source_request_id"] = identity
                                    trajectory["dataset_manifest_sha256"] = manifest["dataset_manifest_sha256"]
                                    trajectory["scene_id"] = str(scene_identity)
                                    if candidate and "source_response_target" in candidate:
                                        trajectory["source_response_target"] = deepcopy(candidate["source_response_target"])
                                    row = {"stage": stage, "mode": mode, "policy": policy, "variant": variant, "tid": index,
                                        "cid": digest([scene_identity, mode, policy, style, binding.fingerprint]),
                                        "response_style": style, "evaluation_protocol": "independent_behavior_prediction",
                                        "dataset": "KuaiSearch/" + (candidate["domain"] if candidate else "Query"),
                                        "source_split": args.split, "trajectory": trajectory, "prob": None}
                                    if predictor is not None and stage == "response" and trajectory["status"] == "ok":
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
                        target = adapted[stage + "_target"]
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
                                    row["direct_click_recall"] = t["outcome_metrics"]["clicked_item_ids_recall"]
                                    row["direct_purchase_recall"] = t["outcome_metrics"]["purchased_item_ids_recall"]
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
                if not args.evaluation_only:
                    observed.extend(observed_action_rows(request, identity, args.split, manifest["dataset_manifest_sha256"]))
        if not args.evaluation_only:
            (args.output / "observed-actions.jsonl").write_text("".join(encoded(row) + "\n" for row in observed))
        exported = {} if args.evaluation_only else export_training(records,args.output,
            observed_actions=observed, sft_only=args.sft_only, max_pairs=args.max_pairs,filter_column=args.filter_column,
            sampling_column=args.sampling_column,ascending=args.ascending,
            query_min=args.query_min,response_min=args.response_min,click_min=args.click_min,purchase_min=args.purchase_min,
            pair_protocol=args.pair_protocol,logloss_min_gap=args.logloss_min_gap,action_min_gap=args.action_min_gap)
        review_rows, reviewed_keys = [], set()
        for row in records:
            t = row["trajectory"]
            key = (row["variant"], t.get("source_request_id"))
            if t["stage"] == "response" and t.get("response_style") == "ekb" and t["status"] == "ok" and key not in reviewed_keys and len(review_rows) < 20:
                reviewed_keys.add(key)
                review_rows.append({"trajectory_id": t["trajectory_id"], "source_request_id": t.get("source_request_id"),
                    "variant": row["variant"], "rationale": t["rationale"], "action": t["final_action"],
                    "messages": t["messages"], "verdict": "pending",
                    "review_focus": ["unsupported preference", "click vs purchase", "missing facts", "action consistency"]})
        (args.output / "semantic-review.jsonl").write_text("".join(encoded(r) + "\n" for r in review_rows))
        report = summarize(records, coverage)
        from agent_simulation.evaluation.customer_request_metrics import request_metrics
        request_report = request_metrics(records)
        (args.output / "request-metrics.json").write_text(encoded(request_report) + "\n")
        report["request_level_protocols"] = sorted({r["protocol"] for r in request_report})
        report.update(exported)
        report.update(split=args.split, filtering={"metric":args.filter_column,"sampling":args.sampling_column,
            "groups":["domain","candidate_count","observed_behavior_group"],"all_raw_rows_retained":True,
            "test_training_export":False})
        (args.output / "report.json").write_text(encoded(report)+"\n")
        print(encoded({k:v for k,v in report.items() if k != "candidate_coverage"}), flush=True)


if __name__ == "__main__":
    main()
