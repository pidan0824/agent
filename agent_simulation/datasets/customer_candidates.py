"""Label-free, bounded Customer recommendation lists within observed exposure.

Observed exposure order is not assumed to be screen rank. Ranking uses only the
recorded query and strictly prior history. Labels are evaluated separately.
"""
from copy import deepcopy
import hashlib
import json
from collections import Counter

from agent_simulation.data.common import require


def construct_candidates(response_input, *, policy="retrieved", limit=10, encoder=None):
    require(policy in ("full", "retrieved"), "Unknown candidate policy")
    require(type(limit) is int and limit > 0, "Candidate limit must be positive")
    context = deepcopy(response_input)
    products = context["displayed_items"]
    ids = [p["item_id"] for p in products]
    require(len(ids) == len(set(ids)), "Duplicate exposure IDs")
    history = context["short_term_memory"]["search_history"]
    require(all(e["time_index"] < context["history_cutoff_exclusive"] for e in history), "Non-prior retrieval history")
    scores = None
    if policy == "retrieved" and len(products) > limit:
        require(encoder is not None, "Retrieved candidates require a configured encoder")
        # A common query/history representation, never the current true clicked item.
        queries = [context["query"], *[e.get("query", "") for e in history[-10:]]]
        queries = [q for q in queries if q and q.strip()]
        vectors = encoder.encode_many(queries, query=True)
        import numpy as np
        q = vectors[0] if len(vectors) == 1 else .8 * vectors[0] + .2 * np.mean(vectors[1:], axis=0)
        texts = [" ".join(str(product.get(k) or "") for k in
                          ("item_title", "brand_name", "category_level3_name")) or str(product["item_id"])
                 for product in products]
        values = encoder.encode_many(texts)
        scores = {product["item_id"]: float(q @ vector) for product, vector in zip(products, values, strict=True)}
        products = sorted(products, key=lambda item: (-scores[item["item_id"]], item["item_id"]))[:limit]
    context["displayed_items"] = products
    selected = [product["item_id"] for product in products]
    categories = [str(product.get("category_level1_name") or "unknown") for product in products]
    manifest = {"policy": policy, "limit": limit, "original_item_ids": ids, "candidate_item_ids": selected,
                "original_count": len(ids), "candidate_count": len(selected),
                "domain": Counter(categories).most_common(1)[0][0] if categories else "empty",
                "outcome_conditioned": False, "scores": scores,
                "ranker": getattr(encoder, "identity", None) if scores is not None else None,
                "input_sha256": hashlib.sha256(json.dumps(response_input, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    return context, manifest


def candidate_coverage(manifest, target):
    """Evaluator-only; never turn a missing positive into a no-click target."""
    selected = set(manifest["candidate_item_ids"])
    original = set(manifest["original_item_ids"])
    result = {}
    for field in ("clicked_item_ids", "purchased_item_ids"):
        truth = set(target[field])
        require(truth <= original, "Observed label outside original exposure")
        result[field + "_recall"] = len(truth & selected) / len(truth) if truth else 1.0
        result[field + "_missing"] = sorted(truth - selected)
    result["reachable"] = not any(result[field + "_missing"] for field in ("clicked_item_ids", "purchased_item_ids"))
    result["behavior_group"] = ("no_click" if not target["clicked_item_ids"] else
                                "multi_click" if len(target["clicked_item_ids"]) > 1 else "single_click")
    result["has_purchase"] = bool(target["purchased_item_ids"])
    return result
