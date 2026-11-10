"""Evaluate Customer actions and Seller prices on held-out examples."""

import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
import json
import math
import statistics
from pathlib import Path
import time
from copy import deepcopy
from uuid import uuid4

from agent_simulation.data.common import require, sha256
from agent_simulation.inference.local.qwen import QwenLocal, DEFAULT_REVISION
from agent_simulation.inference.requests import prepare_request
from agent_simulation.agents.schemas import validate_arguments
from agent_simulation.datasets.labels import (
    SFTDataset, customer_records, expand_record, record_samples, select_record_ids,
)
from agent_simulation.datasets.tokens import bind_split_candidates
from agent_simulation.environment.batching import response_batch_policy
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
            "relative_change_mae": average([baseline_error for _, _, baseline_error in all_changes]),
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
            hits = sum(r["prediction"]["query"] == r["target"]["query"] for r in valid)
            out["exact_text_rate_valid_only"] = hits / len(valid) if valid else None
            out["end_to_end_exact_text_rate"] = hits / len(rows) if rows else None
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


def test_requests(data, role, ids, candidates, *, max_output_tokens=0):
    if role == "customer" and candidates is not None:
        path = candidates / "customer-test-candidates.parquet"
        bind_split_candidates(data.catalog.connection, {"test": path}, role, 0, 1)
        for record in customer_records(data.catalog, 0, 1, split="test"):
            yield record[0]["key"]["session_id"], expand_record(
                role, record, tokenizer=data.tokenizer, max_output_tokens=max_output_tokens,
            )
    else:
        for identity in ids:
            yield identity, record_samples(data, role, identity, max_output_tokens=max_output_tokens)


def load_evaluation_model(args, data, config, resources, *, use_vllm, resume):
    if use_vllm:
        from agent_simulation.inference.vllm import server_client
        require(not args.adapter, 'Use served model names for vLLM adapters')
        model = resources.enter_context(closing(server_client(args, data.tokenizer)))
        config["checkpoint"] = model.check_models()
    else:
        log_name = f"model_calls.resume-{uuid4().hex}.jsonl" if resume else "model_calls.jsonl"
        model = QwenLocal(str(args.model), device=args.device,
                          max_new_tokens=args.max_new_tokens,
                          log_path=args.output / log_name)
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
    return model


def evaluate_records(output, data, model, selection, candidates, study, *, resume):
    records = []
    for role, ids in selection.items():
        evaluated = set()
        for identity, samples in test_requests(
            data, role, ids, candidates, max_output_tokens=model.max_new_tokens,
        ):
            evaluated.add(identity)
            batches, outputs = [], []
            for sample in samples:
                require(sample["split"] == "test", "Non-test example")
                path = output / (sample["sample_id"].replace(":", "_") + ".json")
                if resume and path.is_file():
                    result = json.loads(path.read_text())
                else:
                    print("Evaluating", sample["sample_id"], flush=True)
                    result = predict(sample, model)
                    write_report(path, result)
                    print(result["status"], round(result["seconds"], 1), flush=True)
                if result["stage"] != "response":
                    records.append(result)
                else:
                    if study is not None:
                        batches.append(sample)
                    outputs.append(result)
            if outputs:
                merged = merge_response_batches(identity, outputs)
                records.append(merged)
                if study is not None:
                    study.add_request(batches, outputs, merged, model, predict)
        require(evaluated == set(ids), f"{role} test request coverage differs")
    return records


def run(args):
    candidates = getattr(args, "candidates", None)
    resume = getattr(args, "resume", False)
    require(candidates is not None or args.count > 0, "Select at least one original record per role")
    study = PriceResponseStudy(args.seed) if getattr(args, "price_response", False) else None
    require(study is None or args.role == "customer", "Price response requires Customer role")
    require(not args.adapter or args.role != "both", "Specify the trained adapter's role")
    roles = ("customer", "seller") if args.role == "both" else (args.role,)
    use_vllm = getattr(args, 'inference_backend', 'transformers') == 'vllm'
    if resume:
        require(not use_vllm and study is None, "Resume supports local behavior evaluation")
        saved_config = json.loads((args.output / "config.json").read_text())
    else:
        require(not args.output.exists(), "Use a new evaluation directory or --resume")
        args.output.mkdir(parents=True)
    verify = getattr(args, "verify", False)
    with ExitStack() as resources:
        data = resources.enter_context(closing(SFTDataset(args.dataset, args.tokenizer, verify_hashes=verify)))
        connection = data.catalog.connection
        if candidates is None:
            selection = {role: select_record_ids(connection, args.dataset, role, "test",
                                                 args.count, args.seed) for role in roles}
            sampling = "fixed SHA256 order over test records"
        else:
            selection = {}
            for role in roles:
                key = "session_id" if role == "customer" else "sample_id"
                path = candidates / f"{role}-test-candidates.parquet"
                selection[role] = [record for (record,) in connection.execute(
                    f"SELECT {key} FROM read_parquet(?) ORDER BY {key}", [str(path)]
                ).fetchall()]
                require(selection[role], f"No {role} test candidates in {path}")
            sampling = "all records in the frozen test candidate files"
        config = {
            "selection": selection,
            "seed": args.seed,
            "split": "test",
            "sampling": sampling + "; Customer batches stay together",
            "candidates": str(candidates) if candidates is not None else None,
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
            "device": args.device,
            "role": args.role,
            "price_response": study is not None,
        }
        if "customer" in roles:
            config["customer_response_batching"] = response_batch_policy()
        if resume:
            require(
                config == {k: v for k, v in saved_config.items() if k != "adapter_sha256"},
                "Evaluation inputs or settings changed",
            )
        model = load_evaluation_model(args, data, config, resources, use_vllm=use_vllm, resume=resume)
        if resume:
            require(config == saved_config, "Evaluation adapter changed")
        write_report(args.output / "config.json", config)
        records = evaluate_records(args.output, data, model, selection, candidates, study, resume=resume)
        metrics = summarize(records)
        write_report(args.output / "results.json", {"records": records, "metrics": metrics})
        if study is not None:
            write_report(args.output / "price_response.json", study.report(records))
        write_report(
            args.output / "completion.json",
            {
                "complete": True,
                "original_records_per_role": args.count if candidates is None else None,
                "evaluated_records": {role: len(ids) for role, ids in selection.items()},
            },
        )


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
    p.add_argument("--resume", action="store_true", help="Reuse saved results from this local evaluation")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--count", type=int, default=10)
    selection.add_argument("--candidates", type=Path,
                           help="Evaluate all test records in the frozen candidate directory")
    p.add_argument("--verify", action="store_true", help="Verify dataset and local model file hashes")
    p.add_argument("--price-response", action="store_true",
                   help="Evaluate paired price changes and item/category purchase distributions on Customer test requests")
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    run(p.parse_args())
