"""Generate frozen teacher trajectories and outcome-supervised SFT/DPO artifacts.

Targets are consulted only after every trajectory for a decision is generated.
Test runs retain every sample and never export outcome-selected training rows.
"""

import argparse
import asyncio
from collections import Counter
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from agent_simulation.agents.agentic import (
    AgenticOptions, run_agentic_workflow, seller_auxiliary_tools,
    execute_seller_auxiliary, teacher_output, validate_teacher_final,
)
from agent_simulation.agents.context.market import context_view, SELLER_MARKET_NOTE
from agent_simulation.agents.context.seller_features import seller_behavior_features, with_evidence_id
from agent_simulation.agents.decisions import action_output
from agent_simulation.agents.prompts.customer_agentic import agentic_customer_instructions
from agent_simulation.agents.prompts.seller_agentic import agentic_seller_instructions
from agent_simulation.agents.schemas import validate_arguments
from agent_simulation.data.catalog import DataRepository
from agent_simulation.data.common import require, rows, sha256
from agent_simulation.datasets.labels import SFTDataset, select_record_ids
from agent_simulation.datasets.samples.seller import seller_sample
from agent_simulation.environment.operations.api import SELLER_QUERY_TOOLS
from agent_simulation.environment.operations.queries import validate_observed_window
from agent_simulation.evaluation.micro.customer_agentic import FrozenSampleTools
from agent_simulation.inference.agentic import AgenticClient
from agent_simulation.inference.client import ModelOutputError
from agent_simulation.inference.requests import ContextCapacityError


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


class FrozenSellerSampleTools:
    """Read only historical, owner-authorized facts; never retain the price label."""

    def __init__(self, catalog, sample, *, theory=True):
        self.context = deepcopy(sample["input"])
        self.cutoff = self.context["history_cutoff_exclusive"]
        self.actor_id = self.context["profile"]["seller_id"]
        self.owned = {product["item_id"] for product in self.context["owned_products"]}
        self.theory, self.catalog = theory, catalog
        self.data = DataRepository(catalog.connection, catalog.config, self.cutoff)
        history = rows(catalog.connection,
                       "SELECT * FROM seller_history WHERE seller_id=? AND time_index<? ORDER BY time_index,item_id",
                       [self.actor_id, self.cutoff])
        self.features = seller_behavior_features(history, seller_id=self.actor_id,
                                                 cutoff=self.cutoff, item_ids=sorted(self.owned))
        allowed = {"query_business", "query_public_products", "discover_public_products",
                   "query_market_position", "query_long_term_memory"}
        self.schemas = [deepcopy(tool) for tool in SELLER_QUERY_TOOLS
                        if tool["function"]["name"] in allowed]
        self.schemas += seller_auxiliary_tools(theory=theory)
        self.parameters = {tool["function"]["name"]: tool["function"]["parameters"] for tool in self.schemas}
        self.fingerprint = digest(self.context)

    def initial_context(self):
        products = []
        for product in self.context["owned_products"]:
            products.append({key: deepcopy(product[key]) for key in (
                "item_id", "item_title", "brand_name", "category_level3_name",
                "last_observed_price_raw", "last_observed_time_index", "last_valid_reference", "price_status",
            ) if key in product})
        return with_evidence_id({"profile": {key: value for key, value in self.context["profile"].items() if key != "seller_id"},
                "history_cutoff_exclusive": self.cutoff, "owned_products": products,
                "requested_price_item_ids": self.context["requested_price_item_ids"],
                "behavior_features": self.features,
                "memory_status": "No generated memories in the observed-data snapshot."})

    def guard(self):
        require(digest(self.context) == self.fingerprint, "Frozen Seller evaluation snapshot changed")

    def execute(self, name, args):
        require(name in self.parameters, "Tool unavailable in this Seller stage")
        validate_arguments(args, self.parameters[name])
        if name == "query_long_term_memory":
            return {"experiences": [], "status": "No generated memories in this observed-data snapshot"}
        if name in ("query_business", "query_public_products", "discover_public_products", "query_market_position"):
            validate_observed_window(args, self.cutoff)
            require(not {"before_simulation_step", "start_simulation_step"}.intersection(args),
                    "Observed-data evaluation has no simulation clock")
        if name == "query_business":
            return with_evidence_id(self.data.business(self.actor_id, **args, ranking_connection=self.catalog.connection))
        if name == "query_public_products":
            return with_evidence_id(self.data.public(self.actor_id, self.owned, **args))
        if name == "discover_public_products":
            return with_evidence_id(self.data.discover(self.actor_id, self.owned, **args))
        if name == "query_market_position":
            require(args["item_id"] in self.owned, "Market query must refer to an owned product")
            return with_evidence_id({**context_view(self.data.market_position(self.actor_id, **args)), "semantics": SELLER_MARKET_NOTE})
        return execute_seller_auxiliary(name, args, theory=self.theory)


def trajectory_messages(binding, *, role, stage, theory):
    """Shared label-free initial messages for scene length checks and generation."""
    schemas = action_output(
        stage, displayed_ids=[product["item_id"] for product in binding.displayed] if stage == "response" else None,
        price_ids=binding.context["requested_price_item_ids"] if stage == "seller" else None,
    )["final_schemas"]
    single = stage == "response" and binding.context.get("response_action_mode") == "single_item"
    if single:
        from agent_simulation.evaluation.customer_contract import single_item_schema
        schemas = [single_item_schema([p["item_id"] for p in binding.displayed])]
    direct = role == "customer" and (stage == "query" or not theory)
    if direct:
        from agent_simulation.agents.prompts.instructions import FACT_RULES
        instructions = FACT_RULES + "\nPredict this user's observed behavior from available facts. " + (
            "Generate the next Chinese query. Current query and exposures are hidden."
            if stage == "query" else
            "Use the recorded query and frozen candidates. Relevance does not imply a click; "
            "no-click is normal. Do not infer purchases from clicks.")
        output = schemas[0]
        output_note = "Return only the action JSON, without an explanation. Schema: "
    else:
        instructions = (agentic_customer_instructions(theory=theory) if role == "customer"
                        else agentic_seller_instructions(theory=theory))
        if role == "customer" and theory and getattr(binding, "mode", None) == "fixed":
            from agent_simulation.agents.prompts.customer_agentic import fixed_customer_instructions
            instructions = fixed_customer_instructions(concise=getattr(binding, "concise", False))
        if role == "seller":
            instructions += "\nPredict the next observed quote, rather than an unobserved profit-optimal price."
        output = teacher_output(schemas, customer_ekb=role == "customer" and theory)
        output_note = "Return the original explanation and action together. Schema: "
    from agent_simulation.evaluation.customer_evidence import encode_context
    context = binding.initial_context()
    messages = [{"role": "system", "content": instructions + "\n" + output_note + encoded(output)},
                {"role": "user", "content": encode_context(context) if role == "customer" else encoded(context)}]
    return schemas, messages


async def generate_trajectory(binding, model, *, role, actor_id, stage, record_id,
                              source_split, catalog_identity, options, sample_index):
    """No target argument: the generator receives only the frozen input binding."""
    schemas, messages = trajectory_messages(binding, role=role, stage=stage, theory=options.theory)
    direct = role == "customer" and (stage == "query" or not options.theory)
    final_schema = schemas[0] if direct else teacher_output(
        schemas, customer_ekb=role == "customer" and options.theory)
    initial_state_id = digest([catalog_identity, role, actor_id, stage, record_id, binding.fingerprint])
    trajectory_id = digest([initial_state_id, options.theory, getattr(model, "model", "teacher"),
                            getattr(model, "seed", None), sample_index])
    # Information modes can share a physical snapshot but have different prompts.
    trajectory_id = digest([trajectory_id, messages, binding.schemas])
    trace = {}
    record = {"schema": "teacher-trajectory-v1", "role": role, "actor_id": actor_id, "stage": stage,
              "record_id": record_id, "source_split": source_split, "initial_state_id": initial_state_id,
              "trajectory_id": trajectory_id, "sample_index": sample_index,
              "seed": getattr(model, "seed", None), "theory": options.theory and stage != "query",
              "response_style": "direct" if direct else "ekb" if role == "customer" else "rationale",
              "evaluation_protocol": ("outcome_conditioned_single_positive"
                  if stage == "response" and binding.context.get("response_action_mode") == "single_item"
                  else "independent_behavior_prediction"),
              "action_mode": binding.context.get("response_action_mode", "multi_item") if stage == "response" else stage,
              "status": "running", "trace": trace}

    if role == "customer" and stage == "response":
        record["candidate_item_ids"] = [p["item_id"] for p in binding.displayed]
    if role == "seller":
        record["previous_prices"] = {str(p["item_id"]): p["last_valid_price_raw"]
                                      for p in binding.features["products"]}
    async def complete(conversation, tools):
        return await model.acomplete({"messages": conversation, "tools": tools,
                                      "final_schema": final_schema if role == "customer" else None},
                                    call_id=f'{trajectory_id}:{len(trace["model_calls"])}')

    def validate(value):
        if direct:
            validate_arguments(value, schemas[0])
            return deepcopy(value)
        return validate_teacher_final(value, schemas, trace=trace,
                                      customer_ekb=role == "customer" and options.theory)

    try:
        record["final_action"] = await run_agentic_workflow(
            messages=messages, tools=binding.schemas, complete=complete, execute_tool=binding.execute,
            validate_final=validate,
            options=options, trace=trace, guard=binding.guard)
        record["status"] = "ok"
        record["rationale"] = deepcopy(trace.get("rationale", {}))
    except (ModelOutputError, ContextCapacityError) as error:
        record.update(status="invalid", error=f"{type(error).__name__}: {error}")
    record["messages"] = deepcopy(trace.get("messages", messages))
    record["prompt"] = deepcopy(trace["model_calls"][0]["messages"] if trace.get("model_calls") else messages)
    record["tools"] = deepcopy(binding.schemas)
    require(record["messages"][:len(record["prompt"])] == record["prompt"],
            "Trajectory prefix changed; the training transcript must preserve actual context")
    if role == "customer":
        from agent_simulation.evaluation.customer_evidence import check_evidence
        record["evidence_quality"] = check_evidence(record)
    if role == "seller":
        from agent_simulation.evaluation.seller_quality import check_seller_evidence
        record["evidence_quality"] = check_seller_evidence(record)
    return record


class OutcomeScorer:
    def __init__(self, bge_path=None, *, bge_device="cpu"):
        self.bge_path, self.encoder, self.encoder_id = bge_path, None, None
        self.bge_device = bge_device
        self.vectors = {}

    def query_similarity(self, prediction, target):
        prediction, target = prediction.strip(), target.strip()
        if prediction == target:
            return 1.0
        if self.encoder is None:
            import os
            from agent_simulation import PROJECT_ROOT
            from agent_simulation.agents.retrieval.bge_encoder import bge_encoder
            path = self.bge_path or os.environ.get("MEMORY_BGE_PATH", str(PROJECT_ROOT / "models/bge-small-zh-v1.5"))
            self.encoder, self.encoder_id = bge_encoder(path, device=self.bge_device)
        missing = list(dict.fromkeys(text for text in (prediction, target) if text not in self.vectors))
        if missing:
            self.vectors.update(zip(missing, self.encoder.encode_many(missing)))
        return max(0.0, min(1.0, float(self.vectors[prediction] @ self.vectors[target])))

    def score(self, stage, action, target):
        if stage == "query":
            similarity = self.query_similarity(action["query"], target["query"])
            return {"score": similarity, "query_similarity": similarity,
                    "query_exact_match": action["query"] == target["query"]}
        if stage == "response":
            scores = {}
            for field in ("clicked_item_ids", "purchased_item_ids"):
                prediction, observed = set(action[field]), set(target[field])
                scores[field + "_f1"] = (2 * len(prediction & observed) / (len(prediction) + len(observed))
                                           if prediction or observed else 1.0)
            mean_f1 = sum(scores.values()) / 2
            for field in ("clicked_item_ids", "purchased_item_ids"):
                observed, prediction = set(target[field]), set(action[field])
                scores[field + "_recall"] = len(observed & prediction) / len(observed) if observed else None
            return {"score": mean_f1, **scores}
        predicted = {row["item_id"]: row["price_raw"] for row in action["prices"]}
        errors = [abs(predicted[row["item_id"]] - row["price_raw"]) / row["price_raw"] for row in target["prices"]]
        relative_error = sum(errors) / len(errors)
        return {"score": -relative_error, "relative_price_error": relative_error}


def score_group(trajectories, target, scorer, *, query_min=0.7, response_min=0.8, seller_max_error=0.05, click_min=None, purchase_min=None, seller_change_tolerance=0.0):
    """The complete group's generation has ended before the outcome is attached."""
    for trajectory in trajectories:
        trajectory["target"] = deepcopy(target)
        trajectory["accepted"] = False
        trajectory["score"] = None
        if trajectory["status"] != "ok":
            continue
        metrics = scorer.score(trajectory["stage"], trajectory["final_action"], target)
        trajectory["outcome_metrics"] = metrics
        trajectory["score"] = metrics["score"]
        threshold = {"query": query_min, "response": response_min, "seller": -seller_max_error}[trajectory["stage"]]
        trajectory["accepted"] = metrics["score"] >= threshold
        if trajectory["stage"] == "response":
            from agent_simulation.evaluation.customer_quality import response_quality
            quality = response_quality(trajectory["final_action"], target,
                click_min=response_min if click_min is None else click_min,
                purchase_min=response_min if purchase_min is None else purchase_min)
            trajectory["accepted"] = quality["accepted"]
            trajectory["quality_thresholds"] = quality["thresholds"]
            trajectory["quality_reasons"] = quality["reasons"]
        elif trajectory["stage"] == "seller":
            from agent_simulation.evaluation.seller_quality import seller_action_quality
            quality = seller_action_quality(trajectory["final_action"], target,
                trajectory.get("previous_prices", {}), tolerance=seller_change_tolerance)
            trajectory["accepted"] = trajectory["accepted"] and quality["accepted"]
            trajectory["quality_reasons"] = quality["reasons"]
            trajectory["outcome_metrics"]["directions"] = quality["directions"]
            trajectory["quality_thresholds"] = {"seller_max_error": seller_max_error,
                                                "seller_change_tolerance": seller_change_tolerance}
        elif trajectory["stage"] == "query":
            trajectory["quality_thresholds"] = {"query_min": query_min}


IDENTITY_FIELDS = ("role", "actor_id", "stage", "initial_state_id", "source_split")


def training_rows(trajectories, *, min_gap=0.05):
    """Export accepted SFT and at most one well-separated valid pair per decision."""
    require(bool(trajectories), "Empty trajectory group")
    first = trajectories[0]
    for trajectory in trajectories:
        require(all(trajectory[key] == first[key] for key in IDENTITY_FIELDS), "Mixed trajectory decision identity")
        require(trajectory["prompt"] == first["prompt"] and trajectory["tools"] == first["tools"],
                "Preference branches require the same initial prompt and tools")
        require(trajectory["messages"][:len(first["prompt"])] == first["prompt"], "Invalid trajectory prefix")
    if first["source_split"] != "train":
        return [], [], "held_out_no_training_export"
    metadata = {key: first[key] for key in IDENTITY_FIELDS}
    metadata.update({key: deepcopy(first[key]) for key in ("source_request_id", "scene_id", "source_response_target", "dataset_manifest_sha256", "response_style", "action_mode", "evaluation_protocol") if key in first})
    from agent_simulation.evaluation.customer_evidence import check_evidence
    accepted = [trajectory for trajectory in trajectories if trajectory["status"] == "ok" and trajectory["accepted"]
                and (trajectory["role"] != "customer" or check_evidence(trajectory)["accepted"])]
    if first["role"] == "seller":
        from agent_simulation.evaluation.seller_quality import check_seller_evidence
        accepted = [t for t in accepted if check_seller_evidence(t)["accepted"]]
    unique = {}
    for trajectory in accepted:
        key = digest([trajectory["messages"], trajectory["tools"]])
        unique.setdefault(key, trajectory)
    sft = [{**metadata, "trajectory_id": row["trajectory_id"], "messages": row["messages"],
            "tools": row["tools"], "score": row["score"]} for row in unique.values()]
    if not accepted:
        return sft, [], "no_accepted_trajectory"
    if first["role"] == "customer" and first["stage"] == "response":
        return sft, [], "customer_response_requires_probability_curation"
    chosen = max(accepted, key=lambda row: (row["score"], row["trajectory_id"]))
    rejected = min((row for row in trajectories if row["status"] == "ok"),
                   key=lambda row: (row["score"], row["trajectory_id"]))
    if chosen["score"] - rejected["score"] < min_gap or chosen["score"] == rejected["score"]:
        return sft, [], "tie_or_insufficient_score_gap"
    prefix = len(first["prompt"])
    for branch in (chosen, rejected):
        require(len(branch["messages"]) > prefix and branch["messages"][prefix]["role"] == "assistant",
                "Preference continuation must start with assistant")
    pair = {**metadata, "prompt": first["prompt"], "tools": first["tools"],
            "chosen": chosen["messages"][prefix:], "rejected": rejected["messages"][prefix:],
            "chosen_trajectory_id": chosen["trajectory_id"], "rejected_trajectory_id": rejected["trajectory_id"],
            "chosen_score": chosen["score"], "rejected_score": rejected["score"],
            "preference_source": "observed_outcome_match; not direct human preference labels"}
    return sft, [pair], "paired"


def summarize_group(trajectories, sft, pairs, reason):
    first = trajectories[0]
    valid = [row for row in trajectories if row["status"] == "ok"]
    return {**{key: first[key] for key in IDENTITY_FIELDS}, "record_id": first["record_id"],
            "trajectories": len(trajectories), "valid": len(valid),
            "accepted": sum(row["accepted"] for row in trajectories), "sft_rows": len(sft), "preference_pairs": len(pairs),
            "pair_status": reason, "all_failed": not valid,
            "no_accepted_trajectory": not any(row["accepted"] for row in trajectories),
            "mean_score_valid_only": sum(row["score"] for row in valid) / len(valid) if valid else None,
            "tool_sequences": dict(Counter(
                " > ".join(call["name"] for call in row["trace"].get("tools", [])) or "(no tools)"
                for row in trajectories)),
            "accepted_tool_sequences": dict(Counter(
                " > ".join(call["name"] for call in row["trace"].get("tools", [])) or "(no tools)"
                for row in trajectories if row["accepted"])),
            "rationale_hypotheses": dict(Counter(row.get("rationale", {}).get("hypothesis", row.get("rationale", {}).get("evaluation_style", "(unavailable)"))
                                                 for row in trajectories))}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--role", choices=("customer", "seller"), default="customer")
    p.add_argument("--stage", choices=("query", "response", "both"), default="both", help="Customer stages; Seller always uses seller")
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--record-id", action="append", help="Customer session ID or seller:SELLER:ITEM:INDEX")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--samples-per-decision", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--model", default="Qwen3-32B")
    p.add_argument("--tokenizer", type=Path, default=Path("models/Qwen3-32B"))
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-model-calls", type=int, default=6)
    p.add_argument("--max-tool-calls", type=int, default=8)
    p.add_argument("--theory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--query-min", type=float, default=0.7)
    p.add_argument("--response-min", type=float, default=0.8)
    p.add_argument("--seller-max-error", type=float, default=0.05)
    p.add_argument("--seller-change-tolerance", type=float, default=0.0, help="Relative change deadband; zero means exact observed changes")
    p.add_argument("--preference-min-gap", type=float, default=0.05)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    require(args.count > 0 and args.samples_per_decision > 0, "Positive sample counts required")
    require(args.samples_per_decision == 1 or args.temperature > 0, "Multiple trajectories require positive temperature")
    require(all(math.isfinite(value) and 0 <= value <= 1 for value in (args.query_min, args.response_min)), "Invalid acceptance threshold")
    require(math.isfinite(args.seller_change_tolerance) and args.seller_change_tolerance >= 0, "Invalid Seller change tolerance")
    require(math.isfinite(args.seller_max_error) and args.seller_max_error >= 0, "Invalid Seller error threshold")
    require(math.isfinite(args.preference_min_gap) and args.preference_min_gap > 0, "Positive preference score gap required")
    require(not args.output.exists(), "Use a new trajectory output directory")
    options = AgenticOptions(max_model_calls=args.max_model_calls, max_tool_calls=args.max_tool_calls, theory=args.theory)
    with closing(SFTDataset(args.dataset, args.tokenizer)) as data:
        ids = ([int(value) for value in args.record_id] if args.role == "customer" else args.record_id) if args.record_id else select_record_ids(
            data.catalog.connection, args.dataset, args.role, args.split, args.count, args.seed)
        require(ids and len(ids) == len(set(ids)), "Empty or duplicate record selection")
        column = "session_id" if args.role == "customer" else "sample_id"
        members = {str(row[0]) for row in data.catalog.connection.execute(
            f"SELECT {column} FROM read_parquet(?) WHERE CAST({column} AS VARCHAR) IN (SELECT unnest(?::VARCHAR[]))",
            [str(args.dataset / args.role / args.split / "labels.parquet"), list(map(str, ids))]).fetchall()}
        require(members == set(map(str, ids)), "Records do not belong to the requested official split")
        catalog_identity = sha256(data.catalog.directory / "manifest.json")
        manifest = {"schema": "teacher-trajectories-v1", "role": args.role, "source_split": args.split,
                    "record_ids": ids, "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
                    "catalog_manifest_sha256": catalog_identity, "options": asdict(options),
                    "model": args.model, "temperature": args.temperature, "seed": args.seed,
                    "samples_per_decision": args.samples_per_decision,
                    "thresholds": {key: getattr(args, key) for key in ("query_min", "response_min", "seller_max_error", "seller_change_tolerance", "preference_min_gap")},
                    "training_export": args.split == "train",
                    "target_policy": "Generate the entire decision group without the current label; attach labels and score afterward.",
                    "customer_protocol": "Query separately; Response conditional on recorded Query/display.",
                    "preference_policy": "One best accepted versus worst valid branch if scores differ sufficiently; ties and invalid branches excluded.",
                    "rationale_scope": "Brief generated evidence-based hypotheses; not recovered human thoughts or direct preference feedback."}
        args.output.mkdir(parents=True)
        (args.output / "manifest.json").write_text(encoded(manifest) + "\n")
        scorer, reports = OutcomeScorer(), []
        with closing(AgenticClient(args.url, args.model, data.tokenizer, max_new_tokens=args.max_new_tokens,
                                   temperature=args.temperature, seed=args.seed,
                                   log_path=args.output / "model-calls.jsonl")) as model, \
             (args.output / "trajectories.jsonl").open("w") as all_output, \
             (args.output / "sft.jsonl").open("w") as sft_output, \
             (args.output / "preferences.jsonl").open("w") as preference_output:
            model.check_models()
            for identity in ids:
                if args.role == "customer":
                    request = data.catalog.request(identity, all_prior=True)
                    actor = request["query_input"]["profile"]["user_id"]
                    decisions = [(stage, FrozenSampleTools(request, stage, theory=args.theory), request[stage + "_target"])
                                 for stage in (("query", "response") if args.stage == "both" else (args.stage,))]
                else:
                    prefix, actor, item, cutoff = identity.split(":")
                    require(prefix == "seller", "Invalid Seller record identity")
                    actor, item, cutoff = int(actor), int(item), int(cutoff)
                    sample = seller_sample(data.catalog, actor, item, cutoff)
                    decisions = [("seller", FrozenSellerSampleTools(data.catalog, sample, theory=args.theory), sample["target"])]
                for stage, binding, target in decisions:
                    trajectories = []
                    for index in range(args.samples_per_decision):
                        model.seed = args.seed + index
                        trajectory = asyncio.run(generate_trajectory(
                            binding, model, role=args.role, actor_id=actor, stage=stage, record_id=identity,
                            source_split=args.split, catalog_identity=catalog_identity, options=options, sample_index=index))
                        trajectories.append(trajectory)
                    score_group(trajectories, target, scorer, query_min=args.query_min,
                                response_min=args.response_min, seller_max_error=args.seller_max_error,
                                seller_change_tolerance=args.seller_change_tolerance)
                    sft, preferences, reason = training_rows(trajectories, min_gap=args.preference_min_gap)
                    for destination, records in ((all_output, trajectories), (sft_output, sft), (preference_output, preferences)):
                        for record in records:
                            destination.write(encoded(record) + "\n")
                        destination.flush()
                    reports.append(summarize_group(trajectories, sft, preferences, reason))
                    print(identity, stage, "valid", reports[-1]["valid"], "accepted", reports[-1]["accepted"], flush=True)
        total = sum(row["trajectories"] for row in reports)
        accepted = sum(row["accepted"] for row in reports)
        report = {"complete": True, "groups": reports, "decisions": len(reports), "trajectories": total,
                  "valid": sum(row["valid"] for row in reports), "accepted": accepted,
                  "acceptance_rate": accepted / total if total else None,
                  "sft_rows": sum(row["sft_rows"] for row in reports), "preference_pairs": sum(row["preference_pairs"] for row in reports),
                  "all_failed_decisions": sum(row["all_failed"] for row in reports),
                  "no_accepted_decisions": sum(row["no_accepted_trajectory"] for row in reports),
                  "query_similarity_encoder": scorer.encoder_id,
                  "interpretation": "All samples and failed decisions are retained. No accepted trajectory is not evidence of data noise. "
                                    "Held-out scores describe all samples, not oracle-selected best-of-N accuracy."}
        (args.output / "report.json").write_text(encoded(report) + "\n")
        print(encoded({key: value for key, value in report.items() if key != "groups"}), flush=True)


if __name__ == "__main__":
    main()
