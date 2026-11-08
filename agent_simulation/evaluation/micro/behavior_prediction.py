"""Evaluate Customer actions and Seller prices on held-out examples."""

import argparse
import asyncio
from collections import Counter, defaultdict
import math
import statistics
from pathlib import Path
import time
from copy import deepcopy

from agent_simulation.data.common import require, sha256
from agent_simulation.inference.local.qwen import QwenLocal, DEFAULT_REVISION
from agent_simulation.inference.requests import prepare_request
from agent_simulation.agents.schemas import validate_arguments
from agent_simulation.datasets.labels import SFTDataset, record_samples, select_record_ids
from agent_simulation.evaluation import price_direction
from agent_simulation.evaluation.micro.price_response import PriceResponseStudy
from agent_simulation.logs import write_report
from agent_simulation.agents.decisions import action_output, run_workflow, UnavailableModelAction
from agent_simulation.inference.client import complete_once, ModelOutputError
from agent_simulation.inference.local.checkpoints import attach_adapter, verify_checkpoint


def predict(sample, model):
    stage = sample["sample_id"].split(":")[0]
    require(not sample["tools"], "Expected a JSON decision sample")
    calls = []
    displayed = [p["item_id"] for p in sample["input"]["displayed_items"]] if stage == "response" else None
    output = action_output(stage, displayed_ids=displayed,
                           price_ids=sample["input"]["requested_price_item_ids"] if stage == "seller" else None)

    async def complete(messages, schemas):
        request, count = prepare_request(
            messages, schemas, model.tokenizer, max_output_tokens=model.max_new_tokens
        )
        call = {"input_tokens": count}
        calls.append(call)
        request["action_output"] = output
        request["agent_role"] = "seller" if stage == "seller" else "customer"
        try:
            response = await complete_once(
                model, request, call_id=f'{sample["sample_id"]}:{len(calls)}'
            )
        except ModelOutputError as error:
            call.update(raw_output=error.raw_output, finished=error.finished, error=str(error))
            raise
        call["response"] = deepcopy(response)
        return response

    def final(result):
        try:
            validate_arguments(result, output["final_schemas"][0])
        except ValueError as error:
            raise ModelOutputError(str(error), raw_output=result) from error
        return result

    result = {
        "sample_id": sample["sample_id"],
        "stage": stage,
        "target": sample["target"],
        "status": "running",
        "calls": calls,
        "tools": [],
    }
    if stage == "seller":
        (product,) = sample["input"]["owned_products"]
        reference = product.get("last_valid_reference")
        result["previous_price_raw"] = (
            reference["price_raw"] if reference else product["last_observed_price_raw"]
        )
    started = time.monotonic()
    try:
        result["prediction"] = asyncio.run(
            run_workflow(
                messages=sample["messages"][:-1],
                complete=complete,
                validate_final=final,
            )
        )
        result["status"] = "ok"
    except (ModelOutputError, UnavailableModelAction) as error:
        result.update(status="invalid", error=f"{type(error).__name__}: {error}")
    result["seconds"] = time.monotonic() - started
    return result


def average(values):
    return statistics.mean(values) if values else None


def choices(pairs):
    """Score complete request sets; both-empty sets count as exact matches."""
    tp = predicted = actual = exact = 0
    f1s = []
    counts = {"predicted": [], "actual": []}
    for prediction, target in pairs:
        p, t = set(prediction), set(target)
        hits = len(p & t)
        tp += hits
        predicted += len(p)
        actual += len(t)
        exact += p == t
        f1s.append(2 * hits / (len(p) + len(t)) if p or t else 1.0)
        counts["predicted"].append(len(p))
        counts["actual"].append(len(t))
    return {
        "requests": len(pairs),
        "precision": tp / predicted if predicted else None,
        "recall": tp / actual if actual else None,
        "micro_f1": 2 * tp / (predicted + actual) if predicted + actual else None,
        "mean_request_f1": average(f1s),
        "exact_set_rate": exact / len(pairs) if pairs else None,
        "selected_count_histograms": {k: dict(Counter(v)) for k, v in counts.items()},
        "nonempty_request_rates": {k: average([n > 0 for n in v]) for k, v in counts.items()},
    }


def prices(pairs):
    """Score prices on the stored numerical scale."""
    errors = [p - t for p, t in pairs]
    return {
        "observations": len(pairs),
        "mae": average([abs(e) for e in errors]),
        "rmse": math.sqrt(average([e * e for e in errors])) if errors else None,
        "mean_relative_absolute_error": average([abs(p - t) / t for p, t in pairs]),
    }


def price_changes(pairs, previous_prices):
    by_direction = {direction: [] for direction in ("increase", "decrease", "keep")}
    for (predicted, target), previous in zip(pairs, previous_prices, strict=True):
        actual_direction = price_direction(target - previous)
        predicted_direction = price_direction(predicted - previous)
        by_direction[actual_direction].append((
            predicted_direction == actual_direction,
            abs(predicted - target) / previous,
            abs(previous - target) / previous,
        ))

    def score(group):
        return {
            "observations": len(group),
            "direction_accuracy": average([hit for hit, _, _ in group]),
            "relative_change_mae": average([error for _, error, _ in group]),
        }

    all_changes = [change for group in by_direction.values() for change in group]
    return {
        **score(all_changes),
        "by_actual_direction": {direction: score(group) for direction, group in by_direction.items()},
        "keep_previous_price_baseline": {
            "direction_accuracy": len(by_direction["keep"]) / len(all_changes) if all_changes else None,
            "relative_change_mae": average([error for _, _, error in all_changes]),
        },
    }


def merge_response_batches(identity, outputs):
    """Score one complete Customer request; a failed batch invalidates the request."""
    fields = ("clicked_item_ids", "purchased_item_ids")
    merged = {
        "sample_id": f"response:{identity}",
        "stage": "response",
        "batch_ids": [batch["sample_id"] for batch in outputs],
        "target": {key: [i for batch in outputs for i in batch["target"][key]] for key in fields},
    }
    failed = [batch for batch in outputs if batch["status"] != "ok"]
    if failed:
        merged.update(status="invalid", error="; ".join(
            f'{batch["sample_id"]}: {batch["error"]}' for batch in failed))
    else:
        merged.update(status="ok", prediction={
            key: [i for batch in outputs for i in batch["prediction"][key]] for key in fields})
    return merged


def summarize(records):
    result = {}
    for stage in ("query", "response", "seller"):
        rows = [r for r in records if r["stage"] == stage]
        valid = [r for r in rows if r["status"] == "ok"]
        out = {
            "total": len(rows),
            "valid": len(valid),
            "invalid": len(rows) - len(valid),
            "invalid_rate": (len(rows) - len(valid)) / len(rows) if rows else None,
        }
        out["failures"] = [
            {"sample_id": r["sample_id"], "error": r.get("error", "See batch records")}
            for r in rows
            if r["status"] != "ok"
        ]
        if stage == "query":
            out["exact_text_rate_valid_only"] = average(
                [r["prediction"]["query"] == r["target"]["query"] for r in valid]
            )
            out["intent_review"] = (
                "Review product type and requirements in paired queries; exact text is not intent accuracy."
            )
        elif stage == "response":
            for key in ("clicked_item_ids", "purchased_item_ids"):
                out[key] = choices([(r["prediction"][key], r["target"][key]) for r in valid])
                out[key]["end_to_end_exact_set_rate"] = (
                    sum(set(r["prediction"][key]) == set(r["target"][key]) for r in valid)
                    / len(rows)
                    if rows
                    else None
                )
        else:
            pairs = [
                (r["prediction"]["prices"][0]["price_raw"], r["target"]["prices"][0]["price_raw"])
                for r in valid
            ]
            out.update(prices(pairs))
            out["price_changes"] = price_changes(pairs, [r["previous_price_raw"] for r in valid])
            grouped = defaultdict(list)
            for record, pair in zip(valid, pairs):
                _, seller, item, _ = record["sample_id"].split(":")
                grouped[f"{seller}:{item}"].append(pair)
            out["per_product"] = {key: prices(values) for key, values in grouped.items()}
        result[stage] = out
    result["score_scope"] = (
        "Errors and F1 use valid complete outputs; invalid outputs remain in coverage and end-to-end exact-match denominators."
    )
    return result


def run(args):
    require(not args.output.exists(), "Use a new evaluation directory")
    require(args.count > 0, "Select at least one original record per role")
    study = PriceResponseStudy(args.seed) if getattr(args, "price_response", False) else None
    require(study is None or args.role != "seller",
            "Price response requires Customer test requests")
    require(not args.adapter or args.role != "both", "Specify the trained adapter's role")
    roles = ("customer", "seller") if args.role == "both" else (args.role,)
    args.output.mkdir(parents=True)
    verify = getattr(args, "verify", False)
    data = SFTDataset(args.dataset, args.tokenizer, verify_hashes=verify)
    records = []
    remote = None
    use_vllm = getattr(args, 'inference_backend', 'transformers') == 'vllm'
    try:
        selection = {role: select_record_ids(data.catalog.connection, args.dataset, role, "test",
                                             args.count, args.seed) for role in roles}
        config = {
            "selection": selection,
            "seed": args.seed,
            "split": "test",
            "sampling": "fixed SHA256 pseudo-random order over all test records; Customer batches stay together",
            "model": str(args.model),
            "tokenizer": str(args.tokenizer),
            "checkpoint": verify_checkpoint(args.model, DEFAULT_REVISION) if verify and not use_vllm else None,
            "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
            "dataset_hashes_verified": verify,
            "real_model_inference": True,
            "training": False,
            "prompt_mode": "Reconstruct SFT inputs from frozen catalog facts and test record IDs. Exclude the supervised answer.",
            "max_new_tokens": args.max_new_tokens,
            "protocol": {
                "correction_attempts_per_decision": 0,
                "extra_final_only_turn": 0,
                "partial_actions_executed": False,
                "generation": {"temperature": 0, "n": 1} if use_vllm else {"do_sample": False},
                "constrained_decoding": ("vllm structured_outputs; final JSON schema" if use_vllm
                                         else "lm-format-enforcer==0.11.3; final JSON schema"),
            },
            "adapter": str(args.adapter) if args.adapter else None,
            "role": args.role,
            "price_response": study is not None,
        }
        write_report(args.output / "config.json", config)
        if use_vllm:
            from agent_simulation.inference.vllm import server_client
            require(not args.adapter, 'Use served model names for vLLM adapters')
            remote = server_client(args, data.tokenizer)
            config['checkpoint'] = remote.check_models()
            write_report(args.output / "config.json", config)
            model = remote
        else:
            model = QwenLocal(str(args.model), device=args.device,
                              max_new_tokens=args.max_new_tokens,
                              log_path=args.output / "model_calls.jsonl")
        if data.tokenizer is not model.tokenizer:
            require(
                data.tokenizer.chat_template == model.tokenizer.chat_template
                and data.tokenizer.get_vocab() == model.tokenizer.get_vocab(),
                "Evaluation tokenizer differs from frozen SFT tokenizer",
            )
        if args.adapter:
            config["adapter_sha256"] = attach_adapter(
                model, args.adapter, expected_role=args.role, checkpoint=config["checkpoint"]
            )
            write_report(args.output / "config.json", config)
        for role in roles:
            for identity in selection[role]:
                samples = record_samples(data, role, identity)
                batches, outputs = [], []
                for sample in samples:
                    require(sample["split"] == "test", "Non-test example")
                    print("Evaluating", sample["sample_id"], flush=True)
                    result = predict(sample, model)
                    write_report(
                        args.output / (sample["sample_id"].replace(":", "_") + ".json"), result
                    )
                    print(result["status"], round(result["seconds"], 1), flush=True)
                    if result["stage"] != "response":
                        records.append(result)
                    else:
                        batches.append(sample)
                        outputs.append(result)
                if outputs:
                    merged = merge_response_batches(identity, outputs)
                    records.append(merged)
                    if study is not None:
                        study.add_request(batches, outputs, merged, model, predict)
        metrics = summarize(records)
        write_report(args.output / "results.json", {"records": records, "metrics": metrics})
        if study is not None:
            write_report(args.output / "price_response.json", study.report(records))
        write_report(
            args.output / "completion.json",
            {
                "complete": True,
                "original_records_per_role": args.count,
                "evaluated_records": {role: len(ids) for role, ids in selection.items()},
                "metrics": metrics,
            },
        )
    finally:
        if remote is not None:
            remote.close()
        data.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    from agent_simulation.inference.vllm import add_server_arguments
    add_server_arguments(p)
    p.add_argument("--request-timeout", type=float, default=120)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--role", choices=["both", "seller", "customer"], default="both")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--verify", action="store_true", help="Verify dataset and local model file hashes")
    p.add_argument("--price-response", action="store_true",
                   help="Evaluate paired price changes and item/category purchase distributions on Customer test requests")
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    run(p.parse_args())
