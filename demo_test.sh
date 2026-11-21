#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ../env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

exec ../environments/train/bin/python - "$@" <<'PY'
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from tempfile import mkdtemp

import duckdb

run = Path("result/lite-5pct-quality-1k-20261112")
dataset = "result/lite-5pct-quality-3k-20261112/release/sft"
model = "models/Qwen3.5-9B"
cases = {
    "customer": [
        (311048, "搜索词相似度"),
        (398608, "点击命中"),
        (382479, "购买命中"),
    ],
    "seller": [
        ("seller:5392:175195:937435", "报价精确命中"),
        ("seller:104431:811446:937716", "涨价方向命中"),
    ],
}
checkpoints = {"customer": "checkpoint-2510", "seller": "checkpoint-625"}
epochs = {"customer": 10, "seller": 5}

parser = argparse.ArgumentParser(description="Run the five selected Test cases.")
parser.add_argument("--resume", type=Path, metavar="OUTPUT_DIR")
args = parser.parse_args()
output = (args.resume or Path(
    mkdtemp(prefix=f"demo-selected-test-{datetime.now():%Y%m%d-%H%M%S}-", dir="result")
)).resolve()
if not output.is_dir():
    parser.error(f"Output directory does not exist: {output}")
candidates = output / "candidates"
print(f"Demo output: {output}", flush=True)
path = output / "demo_cases.json"
if args.resume and path.is_file():
    print(f"Demo already complete: {path}", flush=True)
    sys.exit(0)


def evaluate_customer(role_output):
    if (role_output / "completion.json").is_file():
        print("Reusing completed Customer inference.", flush=True)
        return
    candidates.mkdir(exist_ok=True)
    name = "customer-test-candidates.parquet"
    if not (candidates / name).is_file():
        with duckdb.connect() as db:
            ids = [identity for identity, _ in cases["customer"]]
            selected = db.sql(
                "SELECT * FROM read_parquet(?) WHERE session_id IN (SELECT unnest(?))",
                params=[str(run / "selection" / name), ids],
            )
            selected.write_parquet(str(candidates / name))
    adapter = run / "customer-training" / checkpoints["customer"]
    cmd = [
        sys.executable, "-m", "agent_simulation.evaluation.micro.behavior_prediction",
        "--dataset", dataset, "--model", model, "--tokenizer", model,
        "--adapter", str(adapter), "--role", "customer", "--candidates", str(candidates),
        "--device", "cuda", "--output", str(role_output),
    ]
    if (role_output / "config.json").is_file():
        cmd.append("--resume")
    subprocess.run(cmd, check=True)


def evaluate_seller(role_output):
    from agent_simulation.inference.client import ModelOutputError
    from agent_simulation.inference.local.checkpoints import attach_adapter
    from agent_simulation.inference.local.qwen import QwenLocal

    ids = {identity for identity, _ in cases["seller"]}
    archive = run / "behavior" / "seller-seller-epoch-5"
    archived_rows = json.loads((archive / "results.json").read_text())["records"]
    labels = {row["sample_id"]: row for row in archived_rows if row["sample_id"] in ids}
    requests = {}
    with (archive / "model_calls.jsonl").open() as log:
        for line in log:
            call = json.loads(line)
            sample_id, _, _ = call["call_id"].rpartition(":")
            if sample_id in ids:
                request = call["request"]
                request.pop("_prompt_tokens", None)
                requests[sample_id] = request
    if labels.keys() != ids or requests.keys() != ids:
        raise ValueError("Saved Seller labels or decision inputs are missing")

    client = QwenLocal(model, device="cuda", max_new_tokens=1024,
                       log_path=role_output / "model_calls.jsonl")
    attach_adapter(client, run / "seller-training" / checkpoints["seller"], expected_role="seller")
    rows = []
    for identity, _ in cases["seller"]:
        print(f"Evaluating {identity}", flush=True)
        label = labels[identity]
        row = {key: label[key] for key in ("sample_id", "stage", "target", "previous_price_raw")}
        try:
            reply = client.complete(requests[identity], call_id=f"{identity}:1")
            row.update(status="ok", prediction=json.loads(reply["content"]))
        except ModelOutputError as error:
            row.update(status="invalid", error=str(error))
        rows.append(row)
    (role_output / "results.json").write_text(
        json.dumps({"records": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def read_calls(role_output):
    calls = defaultdict(list)
    for path in sorted(role_output.glob("model_calls*.jsonl")):
        with path.open() as log:
            for line in log:
                call = json.loads(line)
                sample_id, _, _ = call["call_id"].rpartition(":")
                calls[sample_id].append({
                    "sample_id": sample_id,
                    "messages": call["request"]["messages"],
                    "raw_output": call["raw_output"],
                })
    return calls


report = []
for role, samples in cases.items():
    if role == "customer":
        role_output = output / role
        evaluate_customer(role_output)
    else:
        role_output = Path(mkdtemp(prefix="seller-replay-", dir=output))
        evaluate_seller(role_output)
    calls = read_calls(role_output)

    results = json.loads((role_output / "results.json").read_text())
    records = {row["sample_id"]: row for row in results["records"]}
    for identity, title in samples:
        stage_ids = [f"query:{identity}", f"response:{identity}"] if role == "customer" else [identity]
        decisions = []
        for stage_id in stage_ids:
            row = records[stage_id]
            decision = {
                "stage": row["stage"],
                "status": row["status"],
                "context": [
                    call for batch in row.get("batch_ids", [stage_id]) for call in calls[batch]
                ],
                "target": row["target"],
                "prediction": row.get("prediction"),
            }
            for key in ("previous_price_raw", "error"):
                if key in row:
                    decision[key] = row[key]
            decisions.append(decision)
        report.append({
            "title": title, "role": role, "sample_id": identity,
            "model": model, "sft_epoch": epochs[role], "decisions": decisions,
            "input_source": "test_context" if role == "customer" else "archived_demo_request",
        })

path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Decision contexts, targets, and predictions: {path.resolve()}", flush=True)
PY
