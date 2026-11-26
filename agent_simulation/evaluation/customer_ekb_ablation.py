"""Label-blind Response EKB ablation. Evaluation only; never exports training rows."""
import argparse
import asyncio
from collections import Counter
from contextlib import closing
from copy import copy, deepcopy
import hashlib
import json
from pathlib import Path

from agent_simulation.agents.schemas import strict_json, validate_arguments
from agent_simulation.data.common import require, sha256

HEADS = ("clicked_item_ids", "purchased_item_ids")
COMMON = """Predict the user's recorded clicks and purchases from the supplied observations.
The current Query is recorded, not generated. Candidate order is canonical serialization,
not observed exposure position. Historical clicks do not imply purchases or preferences.
Historical prices are not current quotations. Missing evidence remains unknown.
Both action lists may be empty; purchase does not require a recorded click.
Use only supplied candidate IDs and return JSON matching the supplied schema."""
SIMPLE_EKB = """
Before the action, write stimulus, knowledge and evaluation:
stimulus: what the recorded Query and visible observations suggest about this request;
knowledge: which prior events are relevant, distinguishing facts from tentative inference;
evaluation: compare candidates against those observations, explain uncertainty, and the
possibility of clicking, purchasing or taking no action. Do not invent motives, demographic
traits, ratings, current prices or internal states. Say when evidence is insufficient.
These are simulated hypotheses, not observed thoughts. No fact-ID annotations are required."""


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(dump(value).encode()).hexdigest()


def product_view(item):
    result = {key: deepcopy(item[key]) for key in
              ("item_id", "item_title", "brand_name", "category_level3_name") if key in item}
    if "price_raw" in item:
        result["historical_observed_price_raw"] = item["price_raw"]
        result["price_observed_time_index"] = item.get("price_observed_time_index")
    return result


def frozen_context(response_input, history_limit):
    """Allowlist only input fields; no scene metadata, targets, or positive positions."""
    require(history_limit > 0, "Positive history limit required")
    cutoff = response_input["history_cutoff_exclusive"]
    events = response_input["short_term_memory"]["search_history"]
    require(all(e["time_index"] < cutoff for e in events), "Non-prior history")
    history = []
    for event in sorted(events, key=lambda e: (e["time_index"], e["session_id"]))[-history_limit:]:
        entry = {key: deepcopy(event[key]) for key in ("session_id", "time_index", "query") if key in event}
        for field in ("clicked_items", "purchased_items"):
            entry[field] = [product_view(item) for item in event.get(field, [])]
        history.append(entry)
    candidates = sorted([product_view(item) for item in response_input["displayed_items"]],
                        key=lambda item: item["item_id"])
    ids = [item["item_id"] for item in candidates]
    require(len(ids) == len(set(ids)), "Duplicate candidate ID")
    return {"query": response_input["query"], "history_cutoff_exclusive": cutoff,
            "history": history, "available_history_events": len(events), "candidates": candidates,
            "candidate_order": "item_id; exposure positions unavailable",
            "current_prices": "unknown; supplied prices are historical observations",
            "action_mode": response_input.get("response_action_mode", "multi_item")}


def schema_for(context, style):
    ids = [item["item_id"] for item in context["candidates"]]
    value = {"type": "array", "items": {"type": "integer", "enum": ids},
             "uniqueItems": True} if ids else {"type": "array", "maxItems": 0}
    if context["action_mode"] == "single_item":
        value["maxItems"] = 1
    action = {"type": "object", "properties": {field: deepcopy(value) for field in HEADS},
              "required": list(HEADS), "additionalProperties": False}
    if style == "direct":
        return action
    rationale = {"type": "object", "properties": {stage: {"type": "string", "minLength": 1}
                 for stage in ("stimulus", "knowledge", "evaluation")},
                 "required": ["stimulus", "knowledge", "evaluation"], "additionalProperties": False}
    return {"type": "object", "properties": {"rationale": rationale, "action": action},
            "required": ["rationale", "action"], "additionalProperties": False}


def generation_request(context, style):
    require(style in ("direct", "ekb"), "Unknown style")
    schema = schema_for(context, style)
    system = COMMON + (SIMPLE_EKB if style == "ekb" else "\nReturn only the action.")
    # Schema in prompt as well as transport; all groups receive identical evidence.
    return {"messages": [{"role": "system", "content": system + "\nOutput schema:\n" + dump(schema)},
                         {"role": "user", "content": dump(context)}], "final_schema": schema}


def action_key(action):
    return tuple(tuple(sorted(action[field])) for field in HEADS)


def select_blind(attempts, *, first=False):
    """Only receives generated outputs, never targets, scores, or evidence verdicts."""
    require(len({r["sample_index"] for r in attempts}) == len(attempts), "Duplicate sample index")
    attempts = sorted(attempts, key=lambda row: row["sample_index"])
    valid = [row for row in attempts if row["status"] == "ok"]
    if first:
        chosen = next((row for row in attempts if row["sample_index"] == 0), None)
        chosen = chosen if chosen and chosen["status"] == "ok" else None
    else:
        counts = Counter(action_key(row["action"]) for row in valid)
        chosen = min(valid, key=lambda row: (-counts[action_key(row["action"])],
                                             row["sample_index"])) if valid else None
    counts = Counter(action_key(row["action"]) for row in valid)
    return {"status": "ok" if chosen else "invalid",
            "sample_index": chosen["sample_index"] if chosen else None,
            "action": deepcopy(chosen["action"]) if chosen else None,
            "attempts": len(attempts), "valid_attempts": len(valid),
            "distinct_actions": len(counts),
            "agreement": max(counts.values()) / len(valid) if valid else None,
            "selection_rule": "prespecified_sample_zero" if first else "joint_action_plurality_earliest_index_tie"}


async def generate(model, context, style, index):
    request = generation_request(context, style)
    row = {"style": style, "sample_index": index, "status": "invalid"}
    try:
        reply = await model.acomplete(request, call_id=f"{fingerprint(context)}:{style}:{index}")
        row["raw_output"] = reply
        require(not reply.get("tool_calls"), "Tools unavailable in fixed-input ablation")
        obj = strict_json(reply["content"])
        validate_arguments(obj, request["final_schema"])
        action = obj if style == "direct" else obj["action"]
        if context["action_mode"] == "single_item":
            require(len(set(action[HEADS[0]] + action[HEADS[1]])) <= 1,
                    "Single-positive scene permits at most one distinct item")
        row.update(status="ok", action=action, rationale=None if style == "direct" else obj["rationale"])
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def evaluate_selected(selections, targets, policy):
    from agent_simulation.evaluation.customer_request_metrics import request_metrics
    rows = []
    for selected in selections:
        target = targets[selected["scene_id"]]
        for arm, decision in selected["arms"].items():
            rows.append({"stage": "response", "variant": "model", "mode": "fixed", "policy": policy,
                "response_style": arm, "tid": 0, "trajectory": {
                    "status": decision["status"], "source_request_id": selected["source_request_id"],
                    "scene_id": selected["scene_id"], "target": target["target"],
                    "source_response_target": target["source_target"], "final_action": decision["action"]}})
    details = request_metrics(rows)
    summary = {}
    for arm in ("A", "B", "C", "A10"):
        subset = [row for row in details if row["style"] == arm]
        metrics = {"requests": len(subset), "complete_requests": sum(row["complete"] for row in subset)}
        for head in HEADS:
            for metric in ("precision", "recall", "f1", "exact_match"):
                values = [r["metrics"][head][metric] if r["complete"] else 0.
                          for r in subset if metric != "recall" or r["target"][head]]
                metrics[head + "_" + metric] = sum(values) / len(values) if values else None
        summary[arm] = metrics
    return {"request_metrics": details, "macro_all_requests": summary,
            "failure_policy": "Incomplete/invalid requests count as zero; empty-target recall excluded."}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--scenes", type=Path, help="Optional outcome-conditioned shortlist diagnostic; default full exposure")
    p.add_argument("--request-id", type=int, action="append")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--history-limit", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=.7)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--context-limit", type=int, default=16384)
    p.add_argument("--url", default="http://127.0.0.1:8011/v1")
    p.add_argument("--model", default="Qwen3.5-9B")
    p.add_argument("--tokenizer", type=Path, default=Path("models/Qwen3.5-9B"))
    p.add_argument("--oracle-diagnostic", action="store_true", help="Separate label-assisted behavior hit ceiling; not test performance")
    return p


def main(argv=None):
    from agent_simulation.datasets.labels import SFTDataset, select_record_ids
    from agent_simulation.datasets.customer_scenes import read_scenes
    from agent_simulation.inference.agentic import AgenticClient
    args = parser().parse_args(argv)
    require(min(args.count, args.samples, args.concurrency, args.history_limit) > 0, "Positive counts required")
    require(args.samples == 1 or args.temperature > 0, "Repeated sampling requires nonzero temperature")
    require(not args.output.exists(), "Use a new output directory")
    prepared, scene_manifest = read_scenes(args.scenes, dataset=args.dataset, split=args.split) if args.scenes else ({}, None)
    policy = "sampled" if args.scenes else "full"
    with closing(SFTDataset(args.dataset, args.tokenizer)) as data:
        ids = args.request_id or (sorted(prepared)[:args.count] if args.scenes else
            select_record_ids(data.catalog.connection, args.dataset, "customer", args.split, args.count, args.seed))
        require(ids and len(set(ids)) == len(ids), "Empty or duplicate request selection")
        members = {row[0] for row in data.catalog.connection.execute(
            "SELECT session_id FROM read_parquet(?) WHERE session_id IN (SELECT unnest(?::BIGINT[]))",
            [str(args.dataset / "customer" / args.split / "labels.parquet"), ids]).fetchall()}
        require(members == set(ids), "Requests outside selected split")
        require(not args.scenes or set(ids) <= set(prepared), "Missing prepared request")
        scenes, targets = [], {}
        for identity in ids:
            source = prepared[identity]["request"] if args.scenes else data.catalog.request(identity, all_prior=True)
            children = [(child["candidate"]["scene_id"], child["request"])
                        for child in prepared[identity]["response_scenes"]] if args.scenes else [(identity, source)]
            for scene_id, request in children:
                scene_id = str(scene_id)
                require(scene_id not in targets, "Duplicate scene ID")
                context = frozen_context(request["response_input"], args.history_limit)
                scenes.append({"scene_id": scene_id, "source_request_id": identity, "context": context,
                               "context_sha256": fingerprint(context)})
                targets[scene_id] = {"target": deepcopy(request["response_target"]),
                                    "source_target": deepcopy(source["response_target"])}
        args.output.mkdir(parents=True)
        manifest = {"schema": "customer-ekb-ablation-v1", "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
            "request_ids": ids, "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
            "scenes_manifest_sha256": sha256(args.scenes / "manifest.json") if args.scenes else None,
            "source_sha256": sha256(Path(__file__)), "policy": policy,
            "protocol": "Fixed recorded Query; conditional Response only. Query evaluation remains separate.",
            "arms": {"A": "direct sample 0", "B": "simple EKB sample 0", "C": "EKB joint-action plurality",
                     "A10": "direct joint-action plurality; same sample count as C"},
            "selection": "No labels, outcome scores, predictor loss or semantic verdicts enter selection.",
            "limitations": ["Shortlists, when supplied, are outcome-conditioned diagnostic scenes.",
                "Existing checkpoints may have used this test split for selection; not necessarily untouched.",
                "Same sample count and max tokens are not equal realized generation cost.",
                "Vote frequency is not a calibrated probability or evidence quality measure.",
                "This evaluates a simplified EKB prompt, not an exact reproduction of the author's method."]}
        (args.output / "manifest.json").write_text(dump(manifest) + "\n")
        (args.output / "frozen-inputs.jsonl").write_text("".join(dump(s) + "\n" for s in scenes))
        selections, generated = [], {}
        with closing(AgenticClient(args.url, args.model, data.tokenizer, temperature=args.temperature,
                max_new_tokens=args.max_new_tokens, context_limit=args.context_limit, seed=args.seed,
                stream=True, log_path=args.output / "model-calls.jsonl")) as client:
            client.check_models()
            with (args.output / "attempts.jsonl").open("w") as output:
                for scene in scenes:
                    async def run_scene():
                        sem = asyncio.Semaphore(args.concurrency)
                        async def one(style, index):
                            async with sem:
                                model = copy(client)
                                model.seed = args.seed + index
                                row = await generate(model, scene["context"], style, index)
                                return {**row, "scene_id": scene["scene_id"]}
                        jobs = [(style, i) for i in range(args.samples) for style in ("direct", "ekb")]
                        # Counterbalance arm order without changing the samples or evidence.
                        if int(scene["context_sha256"][:8], 16) % 2:
                            jobs = [(style, i) for i in range(args.samples) for style in ("ekb", "direct")]
                        return await asyncio.gather(*(one(style, i) for style, i in jobs))
                    attempts = asyncio.run(run_scene())
                    generated[scene["scene_id"]] = attempts
                    for row in attempts:
                        output.write(dump(row) + "\n")
                    output.flush()
                    direct = [r for r in attempts if r["style"] == "direct"]
                    ekb = [r for r in attempts if r["style"] == "ekb"]
                    selections.append({k: scene[k] for k in ("scene_id", "source_request_id", "context_sha256")} |
                        {"arms": {"A": select_blind(direct, first=True), "B": select_blind(ekb, first=True),
                                  "C": select_blind(ekb), "A10": select_blind(direct)}})
                    print(scene["scene_id"], "completed", flush=True)
        # Persist choices BEFORE any label comparison; evaluator is the only label consumer.
        selection_path = args.output / "selections-label-blind.jsonl"
        selection_path.write_text("".join(dump(s) + "\n" for s in selections))
        report = evaluate_selected(selections, targets, policy)
        report["selection_sha256"] = sha256(selection_path)
        report["sample_count"] = args.samples
        report["logloss"] = None
        report["logloss_note"] = "Not inferred from votes or self-confidence; requires a separate calibrated action probability protocol."
        (args.output / "report.json").write_text(dump(report) + "\n")
        if args.oracle_diagnostic:
            oracle = []
            for scene in scenes:
                attempts = generated[scene["scene_id"]]
                target = targets[scene["scene_id"]]["target"]
                matches = [r["sample_index"] for r in attempts if r["style"] == "ekb"
                           and r["status"] == "ok" and action_key(r["action"]) == action_key(target)]
                oracle.append({"scene_id": scene["scene_id"], "any_behavior_match": bool(matches),
                               "matching_sample_indices": matches})
            (args.output / "oracle-D-not-test-performance.json").write_text(dump({
                "scope": "Behavior-only label-assisted ceiling; semantic quality unverified; no training export.",
                "scenes": oracle}) + "\n")


if __name__ == "__main__":
    main()
