"""Small frozen Response probability probes; preparation is offline, scoring never trains."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import time

from agent_simulation.data.common import require, sha256
from agent_simulation.datasets.trajectory import encode_trajectory
from agent_simulation.evaluation.customer_contract import single_item_actions
from agent_simulation.evaluation.customer_evidence import encode_context, expand_sources
from agent_simulation.evaluation.customer_quality import behavior_stratum
from agent_simulation.evaluation.customer_request_metrics import response_with_baseline
from agent_simulation.evaluation.trajectories import digest
from agent_simulation.inference.requests import load_tokenizer
from experiments.cot_sft_pilot import decision_request, dump, load_cases, write_json


def without_history(request):
    """Remove user events AND their derived statistics; retain current item facts."""
    result = deepcopy(request)
    require(not result["tools"], "This probe requires fixed inputs")
    indices = []
    for i, message in enumerate(result["messages"]):
        if message["role"] != "user":
            continue
        try:
            value = json.loads(message["content"])
        except ValueError:
            continue
        if isinstance(value, dict) and "facts" in value:
            indices.append((i, expand_sources(value)))
    require(len(indices) == 1, "Expected one frozen factual context")
    index, context = indices[0]
    retained = {"profile", "query", "history_cutoff_exclusive", "evidence_actor_id",
                "candidate_context", "information_mode", "candidate_protocol", "memory_status"}
    removed = {"behavior_features", "available_history_events", "history_events",
               "selected_history_events", "history_scope", "facts"}
    require(set(context) <= retained | removed, "Unknown context fields; review history ablation")
    require(set(context["profile"]) <= {"gender", "age_bucket", "fre_country", "fre_province", "fre_city"},
            "Profile contains unreviewed derived fields")
    ids = {p["item_id"] for p in context["candidate_context"]}
    facts = {k: v for k, v in context["facts"].items()
             if (v["kind"] == "product" and v.get("item_id") in ids)
             or (v["kind"] == "query" and v["source_id"] == "current_query")}
    stripped = {k: v for k, v in context.items() if k in retained}
    stripped.update(facts=facts, history_events=[], selected_history_events=0,
                    history_scope="Historical user behavior is not supplied; do not infer no prior actions.")
    result["messages"][index]["content"] = encode_context(stripped)
    return result


def prepare(args):
    require(1 <= args.count <= 16 and 1 <= args.max_candidates <= 12, "Small probe limits required")
    require(not args.output.exists(), "Use a new probe plan")
    cases = load_cases(args.cohorts)
    groups = defaultdict(list)
    for case in cases:
        if (case["role"] == "customer" and case["stage"] == "response" and case["partition"] == "dev"
                and case.get("action_mode") == "single_item"
                and 1 <= len(case["candidate_item_ids"]) <= args.max_candidates):
            groups[behavior_stratum(case["target"])].append(case)
    for values in groups.values():
        values.sort(key=lambda c: digest([args.seed, c["case_id"]]))
    selected, parents = [], set()
    for depth in range(max(map(len, groups.values()), default=0)):
        for _, values in sorted(groups.items()):
            if depth >= len(values):
                continue
            case = values[depth]
            parent = (case["actor_id"], case["source_request_id"])
            if parent in parents:
                continue
            parents.add(parent)
            original = decision_request(case, "direct")
            requests = {"full": original, "without_history": without_history(original)}
            actions = single_item_actions(case["candidate_item_ids"])
            require(case["target"] in actions, "True action is outside native menu")
            selected.append({"case_id": case["case_id"], "source_request_id": case["source_request_id"],
                "partition": "dev", "target": case["target"], "actions": actions,
                "stratum": behavior_stratum(case["target"]), "requests": requests,
                "request_sha256": {k: digest(v) for k, v in requests.items()}})
            if len(selected) == args.count:
                break
        if len(selected) == args.count:
            break
    require(len(selected) == args.count, "Insufficient distinct dev parents for requested probe")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"cases_sha256": sha256(args.cohorts / "cases.jsonl"), "seed": args.seed,
        "max_candidates": args.max_candidates, "selection": "Stratified dev subset, one child per parent; not representative accuracy",
        "cases": selected, "cases_digest": digest(selected)})
    print(dump({"prepared": len(selected), "native_sequences_per_model":
                sum(len(c["actions"]) * len(c["requests"]) for c in selected)}))


def encode_options(case, condition, tokenizer, max_length):
    request = case["requests"][condition]
    require(digest(request) == case["request_sha256"][condition], "Changed frozen request")
    return [encode_trajectory(request["messages"] + [{"role": "assistant", "content": dump(action)}],
        [], tokenizer, max_length=max_length, separate_components=True, response_decisions=True).turns[0]
        for action in case["actions"]]


def option_scores(model, turns, pad_token_id, device):
    """Native complete JSON likelihood, including end tokens; no optimizer/backward."""
    import torch
    from agent_simulation.training.trajectory import collate_trajectories, trajectory_token_losses
    results = []
    with torch.inference_mode():
        for turn in turns:
            loss, labels, components = trajectory_token_losses(
                model, collate_trajectories([turn], pad_token_id, device))
            valid = labels.ne(-100)
            action = valid & ((components == 1) | (components == 5) | (components == 6))
            decision = {}
            for name, component in (("click", 5), ("purchase", 6)):
                mask = valid & components.eq(component)
                require(int(mask.sum()) == 1, "Missing native decision token")
                decision[name + "_first_token_logprob"] = -float(loss[mask].sum())
            results.append({"logprob": -float(loss[valid].sum()), "tokens": int(valid.sum()),
                            "action_nll": float(loss[action].mean()), **decision})
    logits = torch.tensor([r["logprob"] for r in results], dtype=torch.float64)
    require(bool(torch.isfinite(logits).all()), "Nonfinite native scores")
    probabilities = logits.softmax(0).tolist()
    for row, probability in zip(results, probabilities):
        row["menu_probability"] = probability
        row["mean_token_logprob"] = row["logprob"] / row["tokens"]
    return results


def score(args):
    import torch
    from peft import PeftModel
    from agent_simulation.training.run import load_base_model
    from agent_simulation.training.trajectory import adapter_digest, base_weight_digest
    plan = json.loads(args.plan.read_text())
    cases = plan["cases"]
    require(1 <= len(cases) <= 16 and digest(cases) == plan["cases_digest"], "Invalid probe plan")
    require(all(c["partition"] == "dev" and set(c["requests"]) == {"full", "without_history"}
                and 4 <= len(c["actions"]) <= 37 and c["target"] in c["actions"]
                and c["requests"]["without_history"] == without_history(c["requests"]["full"])
                for c in cases), "Invalid dev conditions or action menu")
    require(not args.output.exists(), "Use a new score directory")
    weights = base_weight_digest(args.model)
    trained = json.loads((args.sft_run / "config.json").read_text())["identity"]
    require(trained["role"] == "customer" and trained["mode"] == "sft"
            and trained["base_config_sha256"] == sha256(args.model / "config.json")
            and trained["base_weight_sha256"] == weights, "SFT/base checkpoint mismatch")
    tokenizer = load_tokenizer(args.model, local_files_only=True)
    encoded = {(c["case_id"], condition): encode_options(c, condition, tokenizer, args.max_length)
               for c in cases for condition in c["requests"]}
    args.output.mkdir(parents=True)
    identity = {"plan_sha256": sha256(args.plan), "base_weight_sha256": weights,
                "adapter_sha256": adapter_digest(args.sft_run / "adapter"),
                "dtype": args.dtype, "max_length": args.max_length, "status": "running"}
    write_json(args.output / "manifest.json", identity)
    model = load_base_model(args).to(args.device).eval()
    records = []
    with (args.output / "scores.jsonl").open("x") as stream:
        for name in ("base", "sft"):
            if name == "sft":
                model = PeftModel.from_pretrained(model, args.sft_run / "adapter",
                                                 is_trainable=False, local_files_only=True).eval()
            for case in cases:
                for condition in case["requests"]:
                    started = time.monotonic()
                    scores = option_scores(model, encoded[(case["case_id"], condition)],
                                           tokenizer.pad_token_id, args.device)
                    best = max(range(len(scores)), key=lambda i: scores[i]["logprob"])
                    row = {"model": name, "condition": condition, "case_id": case["case_id"],
                           "request_sha256": case["request_sha256"][condition],
                           "target": case["target"], "stratum": case["stratum"],
                           "actions": case["actions"], "scores": scores,
                           "target_index": case["actions"].index(case["target"]), "prediction": case["actions"][best],
                           "status": "ok", "seconds": time.monotonic() - started}
                    records.append(row)
                    stream.write(dump(row) + "\n"); stream.flush()
                    print(dump({k: row[k] for k in ("model", "condition", "case_id", "seconds")}), flush=True)
    summary = {name: {condition: response_with_baseline(
        r for r in records if r["model"] == name and r["condition"] == condition)
        for condition in ("full", "without_history")} for name in ("base", "sft")}
    classes = defaultdict(list)
    for row in records:
        classes[(row["model"], row["condition"], row["stratum"])].append(
            row["scores"][row["target_index"]])
    classwise = []
    for key, values in classes.items():
        item = dict(zip(("model", "condition", "stratum"), key))
        item["scenes"] = len(values)
        item.update({name: sum(r[name] for r in values) / len(values) for name in
                     ("action_nll", "logprob", "menu_probability",
                      "click_first_token_logprob", "purchase_first_token_logprob")})
        classwise.append(item)
    write_json(args.output / "report.json", {"metrics": summary, "gold_scores_by_behavior": classwise, "interpretation":
        "Argmax over exact serialized JSON menu, not free decoding or calibrated behavior probabilities. "
        "First-token scores are conditional on each option's preceding tokens; purchase conditions on its click choice. "
        "Token mean is descriptive only. Small selected dev probe cannot establish generalization.",
        "no_training": True})
    write_json(args.output / "manifest.json", {**identity, "status": "complete", "records": len(records)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--cohorts", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--max-candidates", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261127)
    p = sub.add_parser("score")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--sft-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    {"prepare": prepare, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
