"""Offline, reviewed CoT SFT and matched action-only controls; never calls a model."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

from agent_simulation.agents.agentic import teacher_output
from agent_simulation.agents.decisions import action_output
from agent_simulation.agents.schemas import strict_json, validate_arguments
from agent_simulation.data.common import require, sha256
from agent_simulation.evaluation.customer_contract import single_item_schema, validate_request_splits
from agent_simulation.evaluation.customer_evidence import check_evidence
from agent_simulation.evaluation.customer_filtering import combine_observed_sft
from agent_simulation.evaluation.customer_quality import behavior_stratum
from agent_simulation.evaluation.trajectories import digest


def _key(row):
    return tuple(row.get(k) for k in
                 ("dataset_manifest_sha256", "source_request_id", "actor_id", "stage", "scene_id"))


def _write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare_comparison(records, observed, reviews=None):
    """Preserve original teacher transcripts; review decisions bind their full messages."""
    reviews = reviews or {}
    validate_request_splits([*records, *observed])
    require(all(row.get("trajectory", row).get("partition", "fit") in ("fit", "train")
                for row in [*records, *observed]), "Never export dev/audit CoT cases")
    direct = combine_observed_sft([], [], observed)  # Existing label/schema/split/actor guards.
    require(direct, "Empty observed action release")
    by_scene = {_key(row): row for row in direct}
    require(len(by_scene) == len(direct), "Duplicate observed scene")
    ids = [row["trajectory"]["trajectory_id"] for row in records]
    require(len(set(ids)) == len(ids), "Duplicate teacher trajectory IDs")
    require(set(reviews) <= set(ids), "Review has no matching trajectory")
    chosen, decisions, queue = {}, [], []
    strata = defaultdict(Counter)
    for row in direct:
        if row["stage"] == "response":
            strata[behavior_stratum(row["target"])]["observed_scenes"] += 1
    for source in sorted(records, key=lambda row: row["trajectory"]["trajectory_id"]):
        t = deepcopy(source["trajectory"])
        identity = t["trajectory_id"]
        reasons = []
        row = by_scene.get(_key(t))
        if (source["source_split"] != "train" or t["source_split"] != "train"):
            raise ValueError("CoT comparison requires Train trajectories")
        if t["stage"] != "response" or t.get("response_style") != "ekb" or source["variant"] != "strong":
            reasons.append("not_strong_response_ekb")
        elif row is None:
            reasons.append("no_matching_observed_scene")
        elif t["status"] != "ok":
            reasons.append("generation_invalid")
        else:
            # Only the system's output instructions differ; all other input messages
            # must be identical. Never transplant an old answer onto a new context.
            messages = t.get("messages", [])
            if (t.get("tools") or not messages or messages[0].get("role") != "system"
                    or messages[1:-1] != row["messages"][1:-1]
                    or messages[-1].get("role") != "assistant" or messages[-1].get("tool_calls")):
                reasons.append("frozen_input_mismatch")
            if (t.get("candidate_item_ids") != row.get("candidate_item_ids")
                    or t.get("target") != row["target"]):
                reasons.append("scene_target_or_candidates_mismatch")
            try:
                schema = (single_item_schema(row["candidate_item_ids"])
                          if row.get("action_mode") == "single_item" else
                          action_output("response", displayed_ids=row["candidate_item_ids"])["final_schemas"][0])
                final = strict_json(messages[-1]["content"])
                validate_arguments(final, teacher_output([schema], customer_ekb=True))
                require(final["action"] == t["final_action"] and final["rationale"] == t["rationale"],
                        "Original teacher output mismatch")
            except (ValueError, KeyError, TypeError, IndexError):
                reasons.append("invalid_original_output")
            if not reasons:
                stratum = strata[behavior_stratum(row["target"])]
                stratum["valid_aligned_attempts"] += 1
                if any(set(t["final_action"][k]) != set(row["target"][k]) for k in row["target"]):
                    reasons.append("action_mismatch")
                else:
                    stratum["exact_action"] += 1
                if not any(t["rationale"]["evidence_refs"].values()):
                    reasons.append("no_cited_evidence")
                # Serialized evidence verdicts never bypass recomputation.
                t.pop("semantic_review", None)
                evidence = check_evidence(t)
                reasons.extend(evidence["reasons"])
                if not reasons:
                    stratum["mechanical_pass"] += 1
                    queue.append({"trajectory_id": identity, "messages_sha256": digest(messages),
                                  "scene_id": t["scene_id"], "stratum": behavior_stratum(row["target"]),
                                  "messages": messages, "verdict": "pending", "reason": "", "reviewer": ""})
        review = reviews.get(identity)
        if review is not None:
            require(review.get("messages_sha256") == digest(t.get("messages")), "Review transcript mismatch")
            require(review.get("verdict") in ("pass", "fail", "pending"), "Unknown review verdict")
            if review["verdict"] != "pending":
                require(bool(str(review.get("reason", "")).strip()) and
                        bool(str(review.get("reviewer", "")).strip()), "Review requires reason and reviewer")
        if not reasons:
            if review is None or review["verdict"] == "pending":
                reasons.append("semantic_review_pending")
            elif review["verdict"] == "fail":
                reasons.append("semantic_review_failed")
            elif _key(t) in chosen:
                reasons.append("duplicate_scene")
            else:
                value = deepcopy(row)
                value.update(messages=deepcopy(t["messages"]), response_style="ekb",
                             trajectory_id=identity, initial_state_id=t["initial_state_id"],
                             behavior_source="reviewed original teacher transcript; exact observed action",
                             evidence_quality=evidence, semantic_review=deepcopy(review))
                chosen[_key(t)] = value
                strata[behavior_stratum(row["target"])]["retained_rationale_scenes"] += 1
        decisions.append({"trajectory_id": identity, "scene_id": t.get("scene_id"),
                          "retained": not reasons, "reasons": reasons})
    cot = [chosen.get(_key(row), deepcopy(row)) for row in direct]
    report = {"observed_rows": len(direct), "teacher_attempts": len(records),
              "rationale_scenes": len(chosen),
              "rationale_response_fraction": len(chosen) / max(1, sum(r["stage"] == "response" for r in direct)),
              "exclusion_reasons": dict(Counter(reason for d in decisions for reason in d["reasons"])),
              "strata": {key: dict(value) for key, value in sorted(strata.items())}}
    return direct, cot, report, decisions, queue


def write_review_audit(records, observed, output):
    """Generation ends with a review queue, not an apparently trainable zero-CoT file."""
    _, _, report, decisions, queue = prepare_comparison(records, observed)
    report["status"] = "awaiting_semantic_review" if queue else "no_eligible_rationales"
    output = Path(output)
    _write(output / "cot-filter-decisions.jsonl", decisions)
    _write(output / "cot-review.jsonl", queue)
    (output / "cot-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def export_comparison(records, observed, reviews, output, *, min_rationale_scenes, sources=None):
    require(type(min_rationale_scenes) is int and min_rationale_scenes > 0,
            "Choose a positive minimum rationale scene count")
    output = Path(output)
    require(not output.exists(), "Use a new comparison output directory")
    direct, cot, report, decisions, queue = prepare_comparison(records, observed, reviews)
    report.update(min_rationale_scenes=min_rationale_scenes,
                  status="ready" if report["rationale_scenes"] >= min_rationale_scenes else "insufficient_rationales")
    output.mkdir(parents=True)
    _write(output / "filter-decisions.jsonl", decisions)
    _write(output / "review-candidates.jsonl", queue)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    require(report["status"] == "ready", "Insufficient reviewed rationale scenes; no training files exported")
    pair_id = digest([[_key(r), r["messages"], r["target"]] for r in cot])
    for objective, rows in (("action_only", direct), ("cot", cot)):
        for row in rows:
            row.update(sft_objective=objective, comparison_id=pair_id,
                       min_rationale_scenes=min_rationale_scenes if objective == "cot" else 0)
        _write(output / (objective + ".jsonl"), rows)
    manifest = {**report, "schema": "customer-cot-comparison-v1", "comparison_id": pair_id,
                "sources": sources or {},
                "files": {name: sha256(output / (name + ".jsonl")) for name in ("action_only", "cot")},
                "protocol": "Same ordered scenes, frozen user inputs, targets and parent weights; one row per scene. "
                            "Only reviewed Response rows change system output instructions and assistant output. "
                            "Query and unselected Response rows remain identical.",
                "evaluation": "Same frozen dev/audit cases and direct-action prompts for both adapters. "
                              "Same base, initialization, seed, epochs, batch/accumulation and action weights. "
                              "Ensure both arms fit max_length; do not independently filter/truncate. "
                              "Report rationale coverage and extra tokens/time; token budgets are not equalized."}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Formal customer_curation output")
    parser.add_argument("--reviews", type=Path, required=True, help="Reviewed cot-review.jsonl; keyed by trajectory ID")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-rationale-scenes", type=int, required=True)
    args = parser.parse_args(argv)
    reviews = read_rows(args.reviews)
    require(len({r["trajectory_id"] for r in reviews}) == len(reviews), "Duplicate review IDs")
    sources = {name: sha256(path) for name, path in (
        ("records", args.run / "records.jsonl"), ("observed", args.run / "observed-actions.jsonl"),
        ("reviews", args.reviews), ("run_manifest", args.run / "manifest.json"))}
    manifest = export_comparison(read_rows(args.run / "records.jsonl"),
        read_rows(args.run / "observed-actions.jsonl"),
        {r["trajectory_id"]: r for r in reviews}, args.output,
        min_rationale_scenes=args.min_rationale_scenes, sources=sources)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
