"""Observed-action gates; no model calls, exposure positions, or action rewriting."""
import math
from agent_simulation.data.common import require

FIELDS = ("clicked_item_ids", "purchased_item_ids")


def response_quality(action, target, *, click_min=.8, purchase_min=.8):
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in (click_min, purchase_min)),
            "Response thresholds must be in [0,1]")
    reasons, metrics = [], {}
    for field, threshold in zip(FIELDS, (click_min, purchase_min), strict=True):
        prediction, observed = set(action[field]), set(target[field])
        f1 = 2 * len(prediction & observed) / (len(prediction) + len(observed)) if prediction or observed else 1.
        metrics[field + "_f1"] = f1
        if not observed and prediction:
            reasons.append("false_click" if field == FIELDS[0] else "false_purchase")
        elif observed and f1 < threshold:
            reasons.append("click_below_threshold" if field == FIELDS[0] else "purchase_below_threshold")
    metrics["score"] = sum(metrics.values()) / 2
    metrics["exact_match"] = all(set(action[f]) == set(target[f]) for f in FIELDS)
    return {"accepted": not reasons, "reasons": reasons, "metrics": metrics,
            "thresholds": {"click_min": click_min, "purchase_min": purchase_min}}


def behavior_stratum(target):
    clicks = len(set(target[FIELDS[0]]))
    return ("no_click" if not clicks else "single_click" if clicks == 1 else "multi_click") + (
        "/purchase" if target[FIELDS[1]] else "/no_purchase")


def assess_record(row, *, query_min=None, response_min=None, click_min=None, purchase_min=None):
    t = row["trajectory"]
    if t["status"] != "ok":
        return {"accepted": False, "reasons": ["generation_invalid"], "metrics": {}}
    if t["stage"] == "query":
        threshold = query_min if query_min is not None else t.get("quality_thresholds", {}).get("query_min", .7)
        require(math.isfinite(threshold) and 0 <= threshold <= 1, "Invalid Query threshold")
        score = t.get("score")
        ok = score is not None and math.isfinite(score) and score >= threshold
        return {"accepted": ok, "reasons": [] if ok else ["query_below_threshold"],
                "metrics": {"score": score}, "thresholds": {"query_min": threshold}}
    if "target" not in t or "final_action" not in t:
        return {"accepted": False, "reasons": ["missing_action_or_target"], "metrics": {}}
    thresholds = t.get("quality_thresholds", {})
    fallback = .8 if response_min is None else response_min
    return response_quality(t["final_action"], t["target"],
        click_min=click_min if click_min is not None else (thresholds.get("click_min", fallback) if response_min is None else fallback),
        purchase_min=purchase_min if purchase_min is not None else (thresholds.get("purchase_min", fallback) if response_min is None else fallback))
