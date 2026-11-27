"""Score saved Customer generations with one predictor; never loads a generator."""
import argparse
import asyncio
from contextlib import ExitStack, closing
from copy import deepcopy
import json
from pathlib import Path
import time

from agent_simulation.data.common import require, sha256, write_json_atomic
from agent_simulation.evaluation.customer_curation import encoded, summarize
from agent_simulation.evaluation.customer_filtering import OptionPredictor, export_training
from agent_simulation.evaluation.customer_generation import bounded_map
from agent_simulation.evaluation.customer_request_metrics import request_metrics
from agent_simulation.evaluation.customer_run import (
    RowJournal, run_lock, task_key, sample_probabilities, FrozenPredictionBinding, predictor_trajectory,
)
from agent_simulation.evaluation.trajectories import OutcomeScorer
from agent_simulation.inference.agentic import AgenticClient
from agent_simulation.inference.requests import load_tokenizer, ContextCapacityError


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_runs(paths):
    """Merge disjoint generator branches only when their frozen run settings match."""
    records, sources, observed, reference = [], [], None, None
    ignored = {"output", "strong_url", "weak_url", "predictor_url", "predictor_model",
               "predictor_tokenizer", "generation_variants"}
    for path in sorted(paths):
        manifest = json.loads((path / "manifest.json").read_text())
        require(manifest["schema"] == "customer-usermirrorer-curation-v3"
                and manifest["args"]["defer_probabilities"], "Use a deferred generation run")
        completion = json.loads((path / "completed.json").read_text())
        require(completion["status"] == "generated_pending_probabilities"
                and completion["records_sha256"] == sha256(path / "records.jsonl"),
                "Generation is incomplete or changed")
        settings = {k:v for k,v in manifest.items() if k != "args"}
        settings["args"] = {k:v for k,v in manifest["args"].items() if k not in ignored}
        if reference is None:
            reference = settings
        require(settings == reference, "Generator runs have different frozen settings")
        rows = read_rows(path / "records.jsonl")
        require(len(rows) == completion["records"], "Generation row count changed")
        records.extend(rows)
        action_path = path / "observed-actions.jsonl"
        actions = read_rows(action_path) if action_path.exists() else []
        if observed is None:
            observed = actions
        require(actions == observed, "Generator runs have different observed supervision")
        sources.append({"run": str(path.resolve()), "manifest_sha256": sha256(path / "manifest.json"),
                        "records_sha256": sha256(path / "records.jsonl"),
                        "observed_sha256": sha256(action_path) if action_path.exists() else None})
    require(records and len({task_key(r) for r in records}) == len(records), "Empty or overlapping generator runs")
    records.sort(key=lambda row: (row["cid"], row["variant"], row["tid"]))
    return records, observed, reference["args"], sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictor-url", required=True)
    parser.add_argument("--predictor-model", required=True)
    parser.add_argument("--predictor-tokenizer", type=Path, required=True)
    parser.add_argument("--context-limit", type=int, default=16384)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    require(args.concurrency > 0, "Positive scoring concurrency required")
    records, observed, generation, sources = load_runs(args.run)
    identity = {"schema": "customer-offline-probabilities-v1", "sources": sources,
                "predictor": {"url": args.predictor_url, "model": args.predictor_model,
                              "tokenizer_config_sha256": sha256(args.predictor_tokenizer / "tokenizer_config.json"),
                              "context_limit": args.context_limit},
                "source_sha256": {name: sha256(Path(__file__).with_name(name))
                                  for name in ("customer_scoring.py", "customer_filtering.py", "customer_run.py")}}
    require(args.output.exists() if args.resume else not args.output.exists(),
            "Resume needs an existing scoring run; otherwise use a new directory")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with ExitStack() as resources:
        resources.enter_context(run_lock(args.output))
        manifest_path = args.output / "manifest.json"
        if args.resume:
            require(json.loads(manifest_path.read_text()) == identity, "Scoring resume identity mismatch")
        else:
            write_json_atomic(manifest_path, identity)
        (args.output / "completed.json").unlink(missing_ok=True)
        journal = resources.enter_context(closing(RowJournal(args.output / "probabilities.jsonl")))
        # Revalidate snapshots even when scores are cached.
        for row in records:
            if row["stage"] == "response" and row["trajectory"]["status"] == "ok":
                FrozenPredictionBinding(row)
        needed = any(r["stage"] == "response" and r["trajectory"]["status"] == "ok"
                     and task_key(r) not in journal.rows for r in records)
        predictor = None
        if needed:
            client = resources.enter_context(closing(AgenticClient(
                args.predictor_url, args.predictor_model,
                load_tokenizer(args.predictor_tokenizer, local_files_only=True),
                max_new_tokens=1, context_limit=args.context_limit,
                log_path=args.output / ("predictor-calls-" + str(time.time_ns()) + ".jsonl"))))
            client.check_models()
            predictor = OptionPredictor(client, context_limit=args.context_limit)
        scorer = OutcomeScorer()
        async def one(original):
            row = deepcopy(original)
            if row["stage"] != "response" or row["trajectory"]["status"] != "ok":
                return row
            key = task_key(row)
            cached = journal.get(key)
            if cached is None:
                cached = {"task_id": key}
                try:
                    cached["raw_item_probabilities"] = await asyncio.to_thread(
                        predictor.score, FrozenPredictionBinding(row), predictor_trajectory(row))
                except ContextCapacityError as error:
                    cached["probability_error"] = str(error)
                journal.append(cached)
            if "raw_item_probabilities" in cached:
                sample_probabilities(row, cached["raw_item_probabilities"], generation["seed"], scorer)
            else:
                row["probability_error"] = cached["probability_error"]
            return row
        async def all_rows():
            return [row async for row in bounded_map(one, records, args.concurrency)]
        scored = asyncio.run(all_rows())
        scored.sort(key=lambda row: (row["cid"], row["variant"], row["tid"]))
        (args.output / "records.jsonl").write_text("".join(encoded(r) + "\n" for r in scored))
        (args.output / "observed-actions.jsonl").write_text("".join(encoded(r) + "\n" for r in observed))
        exported = {} if generation["evaluation_only"] else export_training(
            scored, args.output, observed_actions=observed, sft_only=False,
            max_pairs=generation["max_pairs"], filter_column=generation["filter_column"],
            sampling_column=generation["sampling_column"], ascending=generation["ascending"],
            query_min=generation["query_min"], response_min=generation["response_min"],
            click_min=generation["click_min"], purchase_min=generation["purchase_min"],
            pair_protocol=generation["pair_protocol"], logloss_min_gap=generation["logloss_min_gap"],
            action_min_gap=generation["action_min_gap"])
        write_json_atomic(args.output / "report.json", {**summarize(scored, []), **exported})
        write_json_atomic(args.output / "request-metrics.json", request_metrics(scored))
        write_json_atomic(args.output / "completed.json", {
            "status": "complete", "records": len(scored), "records_sha256": sha256(args.output / "records.jsonl")})


if __name__ == "__main__":
    main()
