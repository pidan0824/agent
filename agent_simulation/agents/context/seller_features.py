"""Descriptive pricing proxies from strictly prior, owned observations."""

from statistics import median
import math
import hashlib
import json

from agent_simulation.data.common import require


def seller_behavior_features(records, *, seller_id, cutoff, item_ids=None):
    require(type(cutoff) is int and cutoff > 0, "Invalid Seller history cutoff")
    groups = {}
    for row in records:
        require(row["seller_id"] == seller_id, "Foreign Seller history")
        require(type(row["time_index"]) is int and row["time_index"] >= 0, "Invalid Seller ordering clock")
        if row["time_index"] >= cutoff or (item_ids is not None and row["item_id"] not in item_ids):
            continue
        groups.setdefault(row["item_id"], []).append(row)
    products = []
    for item, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["time_index"])
        require(len({r["time_index"] for r in rows}) == len(rows), "Duplicate Seller observation")
        quotes = [r for r in rows if r.get("price_status") == "observed"
                  and type(r.get("price_raw")) in (int, float)
                  and math.isfinite(r["price_raw"]) and r["price_raw"] > 0]
        pairs = list(zip(quotes, quotes[1:]))
        changes = [(b["price_raw"] - a["price_raw"]) / a["price_raw"] for a, b in pairs]
        prices = [r["price_raw"] for r in quotes]
        exposure = sum(r.get("exposure_records", 0) for r in rows)
        products.append({"item_id": item, "observations": len(rows), "valid_quote_observations": len(quotes),
            "last_observed_index": rows[-1]["time_index"],
            "last_valid_price_raw": prices[-1] if prices else None,
            "median_price_raw": median(prices) if prices else None,
            "observed_quote_change_fraction": sum(c != 0 for c in changes) / len(changes) if changes else None,
            "observed_upward_pairs": sum(c > 0 for c in changes),
            "observed_downward_pairs": sum(c < 0 for c in changes),
            "median_absolute_relative_change": median(abs(c) for c in changes) if changes else None,
            "exposure_records": exposure,
            "click_records": sum(r.get("click_records", 0) for r in rows),
            "purchase_records": sum(r.get("purchase_records", 0) for r in rows)})
    return {"schema": "seller-descriptive-features-v1", "historical_cutoff_exclusive": cutoff,
            "products": products, "limitations": [
                "Quote observations are not actual price-change actions or calendar durations.",
                "Quote-change pairs skip unknown prices; unobserved intervals remain unknown.",
                "Traffic and price associations do not identify demand elasticity or causal effects.",
                "Raw currency units, costs, margins and unobserved competitor performance are unknown.",
                "Stable prices are descriptive evidence, not proof of EDLP or any particular motive."]}


def with_evidence_id(value):
    """Name a code-owned observation block without copying its contents."""
    value = dict(value)
    value.pop("evidence_id", None)
    identity = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return {**value, "evidence_id": "S" + hashlib.sha256(identity.encode()).hexdigest()[:12]}
