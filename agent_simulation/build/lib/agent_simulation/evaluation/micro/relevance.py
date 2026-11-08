"""Evaluate saved relevance scores separately from Customer behavior and policy outcomes.

No model is trained or silently substituted. Benchmark scores must identify the
exact original row, query and product text. Scores on unlabelled platform lists
are estimates, not accuracy or ground-truth relevance."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean

from agent_simulation.data.common import require, sha256
from agent_simulation.logs import write_report


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def auc_metrics(labels, scores):
    """Tie-aware Mann–Whitney ROC-AUC and stepwise average precision."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None, None
    groups = defaultdict(list)
    for y, s in zip(labels, scores):
        groups[s].append(y)
    below_negative = concordant = 0
    for score in sorted(groups):
        p = sum(groups[score])
        n = len(groups[score]) - p
        concordant += p * (below_negative + n / 2)
        below_negative += n
    tp = seen = 0
    ap = 0.0
    for score in sorted(groups, reverse=True):
        p = sum(groups[score])
        tp += p
        seen += len(groups[score])
        ap += (p / positives) * (tp / seen)
    return concordant / (positives * negatives), ap


def evaluate_pairs(reference, predictions, threshold=None):
    require(len(reference) == len(predictions) and bool(reference), "Incomplete relevance scores")
    require(all(type(p.get("source_row")) is int for p in predictions),
            "Expected an integer source_row for each relevance score")
    by_row = {p["source_row"]: p for p in predictions}
    require(
        set(by_row) == set(range(len(reference))),
        "Missing or duplicate source rows",
    )
    labels, scores = [], []
    for i, row in enumerate(reference):
        p = by_row[i]
        require((p["query"], p["text"]) == (row["query"], row["text"]), "Relevance pair mismatch")
        require(
            type(row["label"]) is int and row["label"] in (0, 1), "Expected official binary labels"
        )
        require(type(p["score"]) in (int, float) and math.isfinite(p["score"]), "Invalid score")
        labels.append(row["label"])
        scores.append(p["score"])
    roc, ap = auc_metrics(labels, scores)
    result = {
        "pairs": len(labels),
        "positive_pairs": sum(labels),
        "roc_auc": roc,
        "pr_auc_average_precision": ap,
    }
    if threshold is not None:
        require(math.isfinite(threshold), "Invalid threshold")
        selected = [int(s >= threshold) for s in scores]
        f1s = []
        for category in (0, 1):
            tp = sum(a == b == category for a, b in zip(labels, selected))
            denominator = labels.count(category) + selected.count(category)
            f1s.append(2 * tp / denominator if denominator else 0.0)
        result.update(
            threshold=threshold,
            accuracy=mean(a == b for a, b in zip(labels, selected)),
            macro_f1=mean(f1s),
        )
    return result


def summarize_lists(records):
    """Measure score coverage and mean scores for each request."""
    grouped = defaultdict(list)
    seen = set()
    for row in records:
        key = (row["request_id"], row["item_id"])
        require(key not in seen, "Duplicate item in request")
        seen.add(key)
        score = row.get("score")
        require(
            score is None or type(score) in (int, float) and math.isfinite(score),
            "Invalid list score",
        )
        grouped[row["request_id"]].append(score)
    per_request = {}
    coverage = {}
    scored_items = 0
    for request_id, scores in grouped.items():
        scored = [score for score in scores if score is not None]
        scored_items += len(scored)
        per_request[request_id] = mean(scored) if scored else None
        coverage[request_id] = len(scored) / len(scores)

    valid = [s for s in per_request.values() if s is not None]
    return {
        "requests": len(grouped),
        "items": len(records),
        "scored_items": scored_items,
        "score_coverage": scored_items / len(records) if records else None,
        "mean_estimated_score_per_request": mean(valid) if valid else None,
        "request_estimated_scores": per_request,
        "request_score_coverage": coverage,
        "interpretation": "Request means use scored items only. Read them alongside score coverage "
        "and compare them only on the same score scale. These scores are model estimates. "
        "Historical exposure lists lack verified display positions.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--threshold", type=float)
    p.add_argument(
        "--score-model",
        required=True,
        help="Explicit checkpoint/version that produced these scores",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    require(
        a.threshold is None or a.reference is not None,
        "Threshold applies to labelled evaluation only",
    )
    require(not a.output.exists(), "Use a new report path")
    records = read_rows(a.predictions)
    result = {
        "score_model": a.score_model,
        "predictions_sha256": sha256(a.predictions),
        "mode": "labelled_pair_test" if a.reference else "platform_list_scores",
        "metrics": (
            evaluate_pairs(read_rows(a.reference), records, a.threshold)
            if a.reference
            else summarize_lists(records)
        ),
    }
    if a.reference:
        result["reference_sha256"] = sha256(a.reference)
    write_report(a.output, result)
