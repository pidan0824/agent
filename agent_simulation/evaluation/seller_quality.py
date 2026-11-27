"""Seller action consistency and source checks; not a prose truth judge."""
import json
import math
import re
from agent_simulation.agents.context.seller_features import with_evidence_id
from agent_simulation.data.common import require


def price_direction(price, previous, tolerance=0.0):
    difference = (price - previous) / previous
    return 0 if abs(difference) <= tolerance else (1 if difference > 0 else -1)


def seller_action_quality(action, target, previous, *, tolerance=0.0):
    require(math.isfinite(tolerance) and tolerance >= 0, "Invalid Seller change tolerance")
    predicted = {r["item_id"]: r["price_raw"] for r in action["prices"]}
    reasons, directions = [], []
    for row in target["prices"]:
        item, actual = row["item_id"], row["price_raw"]
        prior = previous.get(str(item), previous.get(item))
        if type(prior) not in (int, float) or not math.isfinite(prior) or prior <= 0:
            reasons.append("missing_previous_price")
            continue
        estimate = predicted[item]
        if type(estimate) not in (int, float) or not math.isfinite(estimate) or estimate <= 0:
            reasons.append("invalid_predicted_price")
            continue
        observed = price_direction(actual, prior, tolerance)
        forecast = price_direction(estimate, prior, tolerance)
        directions.append({"item_id": item, "observed": observed, "predicted": forecast})
        if observed != forecast:
            reasons.append("change_state_mismatch" if 0 in (observed, forecast) else "direction_mismatch")
    return {"accepted": not reasons, "reasons": sorted(set(reasons)), "directions": directions}


def check_seller_evidence(trajectory):
    available = set()
    for message in trajectory.get("messages", []):
        if message.get("role") not in ("user", "tool"):
            continue
        try:
            value = json.loads(message.get("content") or "")
        except (TypeError, ValueError):
            continue
        def collect(node):
            if isinstance(node, dict):
                if node.get("evidence_id") and node["evidence_id"] == with_evidence_id(node)["evidence_id"]:
                    available.add(node["evidence_id"])
                for child in node.values():
                    if isinstance(child, (dict, list)):
                        collect(child)
            elif isinstance(node, list):
                for child in node:
                    collect(child)
        collect(value)
    reasons = []
    rationale = trajectory.get("rationale", {})
    evidence = rationale.get("evidence", [])
    if not evidence:
        reasons.append("missing_evidence")
    for claim in evidence:
        refs = re.findall(r"\[(S[0-9a-f]{12})\]", claim)
        if not refs:
            reasons.append("missing_evidence_reference")
        elif not set(refs) <= available:
            reasons.append("unknown_evidence_reference")
    try:
        final = trajectory["messages"][-1]
        value = json.loads(final["content"])
        if (final["role"] != "assistant" or value.get("rationale") != rationale
                or value.get("action") != trajectory.get("final_action")):
            reasons.append("original_transcript_mismatch")
    except (KeyError, IndexError, TypeError, ValueError):
        reasons.append("invalid_original_final_json")
    review = trajectory.get("semantic_review", {}).get("verdict", "not_reviewed")
    if review == "fail":
        reasons.append("semantic_review_failed")
    return {"accepted": not reasons, "status": "failed" if reasons else "machine_checked",
            "reasons": sorted(set(reasons)), "semantic_status": review,
            "scope": "Visible code-owned source blocks and transcript consistency; prose entailment is not certified"}
