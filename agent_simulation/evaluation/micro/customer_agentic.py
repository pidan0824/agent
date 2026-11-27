"""Frozen, conditional Customer prediction with bounded read-only tool trajectories.

This evaluates Query separately and Response given the recorded Query/display.
It does not score a newly searched choice set against the old display's labels,
select trajectories using test outcomes, or export distillation training data.
"""

import argparse
import asyncio
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from agent_simulation.agents.agentic import (
    AgenticOptions, customer_auxiliary_tools, execute_customer_auxiliary,
)
from agent_simulation.agents.context.customer_features import customer_behavior_features
from agent_simulation.agents.context.shopping import displayed_details
from agent_simulation.agents.schemas import validate_arguments
from agent_simulation.data.common import require, sha256
from agent_simulation.environment.operations.api import CUSTOMER_QUERY_TOOLS


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class FrozenSampleTools:
    """A label-free snapshot adapter for the on-demand catalog's Customer records."""

    def __init__(self, request, stage, *, theory=True):
        require(stage in ("query", "response"), "Unknown Customer stage")
        theory = theory and stage == "response"
        self.stage, self.theory = stage, theory
        # Copy only the stage's input; no current target is retained by this binding.
        self.context = deepcopy(request[stage + "_input"])
        self.cutoff = self.context["history_cutoff_exclusive"]
        self.history = deepcopy(self.context["short_term_memory"]["search_history"])
        require(all(event["time_index"] < self.cutoff for event in self.history), "Non-prior history")
        self.displayed = deepcopy(self.context.get("displayed_items", [])) if stage == "response" else []
        allowed = {"query_customer_history", "query_long_term_memory"}
        if stage == "response":
            allowed.add("query_displayed_products")
        self.schemas = [deepcopy(tool) for tool in CUSTOMER_QUERY_TOOLS
                        if tool["function"]["name"] in allowed]
        self.schemas += customer_auxiliary_tools(theory=theory)
        self.parameters = {tool["function"]["name"]: tool["function"]["parameters"] for tool in self.schemas}
        self.fingerprint = hashlib.sha256(_json(self.context).encode()).hexdigest()

    def _base_context(self):
        """Unencoded facts shared by the fixed and tool-using adapters."""
        profile = self.context["profile"]
        result = {
            "profile": {key: value for key, value in profile.items() if key != "user_id"},
            "history_cutoff_exclusive": self.cutoff,
            "evidence_actor_id": self.context["profile"]["user_id"],
            "behavior_features": customer_behavior_features(
                self.history, cutoff=self.cutoff, behavioral_summary=self.context.get("behavioral_summary"),
                user_id=profile.get("user_id")),
            "available_history_events": len(self.history),
            "memory_status": "No generated long-term memory is supplied in the observed-data evaluation.",
        }
        if self.stage == "response":
            result.update(query=self.context["query"], displayed_items=[
                {key: product.get(key) for key in ("item_id", "item_title", "price_raw")}
                for product in self.displayed])
        return result

    def initial_context(self):
        from agent_simulation.evaluation.customer_evidence import attach_sources
        return attach_sources(self._base_context())

    def guard(self):
        require(hashlib.sha256(_json(self.context).encode()).hexdigest() == self.fingerprint,
                "Frozen evaluation snapshot changed")

    def execute(self, name, args):
        if name not in self.parameters:
            raise PermissionError("Tool unavailable in this Customer evaluation stage")
        validate_arguments(args, self.parameters[name])
        if name == "query_customer_history":
            before = args["before_time_index"]
            require(0 <= before <= self.cutoff, "History query exceeds frozen cutoff")
            require("before_simulation_step" not in args, "Observed-data evaluation has no simulation clock")
            visible = [event for event in self.history if event["time_index"] < before]
            from agent_simulation.evaluation.customer_evidence import attach_sources
            return attach_sources({"evidence_actor_id": self.context["profile"]["user_id"], "observed_history": deepcopy(visible[-args["limit"]:]), "simulation_history": [],
                    "end_exclusive": before, "eligible_events": len(visible)})
        if name == "query_long_term_memory":
            return {"experiences": [], "status": "No generated memories in this observed-data snapshot"}
        if name == "query_displayed_products":
            return displayed_details(self.displayed, args["item_ids"])
        return execute_customer_auxiliary(name, args, theory=self.theory)


async def predict_request(request, stage, model, *, options=None, sample_index=0):
    options = options or AgenticOptions()
    binding = FrozenSampleTools(request, stage, theory=options.theory)
    from agent_simulation.evaluation.trajectories import generate_trajectory
    started = time.monotonic()
    trajectory = await generate_trajectory(binding, model, role="customer",
        actor_id=request["query_input"]["profile"]["user_id"], stage=stage,
        record_id=request["session_id"], source_split=request["assigned_split"],
        catalog_identity="frozen-micro", options=options, sample_index=sample_index)
    result = {**trajectory, "sample_id": f'{stage}:{request["session_id"]}',
              "request_id": request["session_id"], "snapshot_sha256": binding.fingerprint,
              "split": request["assigned_split"], "sampling_seed": getattr(model, "seed", None)}
    if trajectory["status"] == "ok":
        result["prediction"] = trajectory["final_action"]
    result["target"] = deepcopy(request[stage + "_target"])
    result["seconds"] = time.monotonic() - started
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New output directory; existing results are never overwritten")
    p.add_argument("--split", choices=("train", "test"), default="test",
                   help="Official source split; this release does not define an independent validation pool")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--request-id", type=int, action="append")
    selection.add_argument("--candidates", type=Path, help="Frozen Customer candidates Parquet containing session_id")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--stage", choices=("query", "response", "both"), default="both")
    p.add_argument("--samples-per-decision", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--model", default="Qwen3-32B")
    p.add_argument("--tokenizer", type=Path, default=Path("models/Qwen3-32B"))
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-model-calls", type=int, default=6)
    p.add_argument("--max-tool-calls", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--theory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--baseline", type=Path, help="Canonical behavior_prediction JSONL; compared on the same sample IDs")
    return p


def load_baseline(path, expected):
    """Validate comparison coverage before inference; Response targets are sets.

    Canonical merged batches can order the same target IDs differently. Duplicates
    remain invalid rather than being silently repaired by set conversion.
    """
    baseline = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    baseline = [row for row in baseline if row["sample_id"] in expected]
    require(len(baseline) == len(expected)
            and len({row["sample_id"] for row in baseline}) == len(baseline),
            "Baseline must cover each evaluated sample exactly once")
    for row in baseline:
        reference = expected[row["sample_id"]]
        stage = reference["stage"]
        require(row["stage"] == stage, "Baseline stage differs")
        if stage == "response":
            fields = {"clicked_item_ids", "purchased_item_ids"}
            for target in (row["target"], reference["target"]):
                require(isinstance(target, dict) and set(target) == fields, "Invalid Response target fields")
                for field in fields:
                    ids = target[field]
                    require(isinstance(ids, list) and all(type(item) is int for item in ids),
                            "Response targets must be integer ID lists")
                    require(len(ids) == len(set(ids)), "Duplicate Response target IDs")
            require(all(set(row["target"][field]) == set(reference["target"][field]) for field in fields),
                    "Baseline targets differ")
        else:
            require(row["target"] == reference["target"], "Baseline targets differ")
    return baseline


def main(argv=None):
    from agent_simulation.datasets.labels import SFTDataset, select_record_ids
    from agent_simulation.inference.agentic import AgenticClient
    from agent_simulation.evaluation.micro.behavior_prediction import summarize

    args = parser().parse_args(argv)
    require(args.count > 0 and args.samples_per_decision > 0, "Positive sample counts required")
    require(args.samples_per_decision == 1 or args.temperature > 0,
            "Multiple trajectories require positive temperature; greedy sampling repeats one decision")
    require(not args.output.exists(), "Use a new evaluation output directory")
    options = AgenticOptions(max_model_calls=args.max_model_calls,
                             max_tool_calls=args.max_tool_calls, theory=args.theory)
    with closing(SFTDataset(args.dataset, args.tokenizer)) as data:
        connection = data.catalog.connection
        if args.request_id:
            ids = args.request_id
        elif args.candidates:
            ids = [row[0] for row in connection.execute(
                "SELECT session_id FROM read_parquet(?) ORDER BY sha256(CAST(session_id AS VARCHAR) || ':' || CAST(? AS VARCHAR)),session_id LIMIT ?",
                [str(args.candidates), args.seed, args.count]).fetchall()]
        else:
            ids = select_record_ids(connection, args.dataset, "customer", args.split, args.count, args.seed)
        require(ids and len(ids) == len(set(ids)), "Empty or duplicate request selection")
        # Membership is checked against the selected release, independently of CLI labels.
        members = {row[0] for row in connection.execute(
            "SELECT session_id FROM read_parquet(?) WHERE session_id IN (SELECT unnest(?::BIGINT[]))",
            [str(args.dataset / "customer" / args.split / "labels.parquet"), ids]).fetchall()}
        require(members == set(ids), "Selected requests do not belong to the requested release split")
        stages = ("query", "response") if args.stage == "both" else (args.stage,)
        baseline = None
        if args.baseline:
            # Evaluator-only targets are not passed to the agent or tool bindings.
            expected = {}
            for identity in ids:
                labels = data.catalog.request(identity, history_limit=1, include_displayed=False)
                for stage in stages:
                    expected[f"{stage}:{identity}"] = {"stage": stage, "target": labels[stage + "_target"]}
            baseline = load_baseline(args.baseline, expected)
        args.output.mkdir(parents=True)
        manifest = {"schema": "customer-agentic-evaluation-v1", "request_ids": ids,
                    "dataset": str(args.dataset.resolve()), "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
                    "selection_split": args.split, "stages": stages, "options": asdict(options),
                    "samples_per_decision": args.samples_per_decision, "model": args.model,
                    "temperature": args.temperature, "seed": args.seed,
                    "theory_application": "Response theory-on uses EKB with evidence annotations; theory-off and Query return direct actions.",
                    "protocol": "Query prediction and conditional Response on recorded Query/display; no search or settlement.",
                    "selection_policy": "All samples are reported; no outcome-based selection or distillation.",
                    "historical_test_use": "Existing project training may have used test for checkpoint selection; do not call it untouched."}
        if args.baseline:
            manifest["baseline"] = {"path": str(args.baseline.resolve()), "sha256": sha256(args.baseline)}
        (args.output / "manifest.json").write_text(_json(manifest) + "\n")
        records = []
        with closing(AgenticClient(args.url, args.model, data.tokenizer,
                                   max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                   seed=args.seed, log_path=args.output / "model-calls.jsonl")) as model:
            model.check_models()
            with (args.output / "predictions.jsonl").open("w") as output:
                for identity in ids:
                    request = data.catalog.request(identity, all_prior=True)
                    for sample_index in range(args.samples_per_decision):
                        model.seed = args.seed + sample_index
                        for stage in stages:
                            record = asyncio.run(predict_request(request, stage, model,
                                                 options=options, sample_index=sample_index))
                            records.append(record)
                            output.write(_json(record) + "\n")
                            output.flush()
                            print(record["sample_id"], sample_index, record["status"], flush=True)
        report = {"by_sample_index": {str(index): summarize([r for r in records if r["sample_index"] == index])
                                      for index in range(args.samples_per_decision)},
                  "planning_usage": planning_usage(records)}
        if baseline is not None:
            report["baseline_same_requests"] = summarize(baseline)
        (args.output / "report.json").write_text(_json(report) + "\n")
        print(_json(report), flush=True)


def planning_usage(records):
    """Separate available theory and tools from evidence that the model used them."""
    calls = [call for row in records for call in row["trace"].get("tools", [])]
    successful = [call for call in calls if "error" not in call]
    return {
        "decisions": len(records),
        "decisions_with_tool_calls": sum(bool(row["trace"].get("tools")) for row in records),
        "model_calls": sum(len(row["trace"].get("model_calls", [])) for row in records),
        "tool_calls": len(calls),
        "failed_tool_calls": len(calls) - len(successful),
        "theory_reference_reads": sum(
            call["name"] == "read_skill_reference"
            and call.get("arguments", {}).get("name") == "decision-types" for call in successful),
        "recorded_behavior_hypotheses": sum(
            call["name"] == "record_behavior_hypothesis" for call in successful),
        "interpretation": "Tool availability and theory-on do not establish tool use, EKB adherence or personalization.",
    }


if __name__ == "__main__":
    main()
