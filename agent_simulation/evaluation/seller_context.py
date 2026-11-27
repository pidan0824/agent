"""Shared observed Seller inputs for direct, rationale and tool-based decisions."""
from copy import deepcopy

from agent_simulation.agents.context.seller_features import with_evidence_id
from agent_simulation.data.common import require

PRODUCT_ATTRIBUTES = ("item_title", "brand_name", "category_level1_name",
                      "category_level2_name", "category_level3_name")
QUOTE_FIELDS = ("time_index", "price_raw", "price_status",
                "exposure_records", "click_records", "purchase_records")


def observed_seller_context(state, features, history):
    """Keep per-item business, quote and comparable-product evidence together."""
    actor, cutoff = state["profile"]["seller_id"], state["history_cutoff_exclusive"]
    require(features["historical_cutoff_exclusive"] == cutoff, "Seller feature cutoff mismatch")
    details = {p["item_id"]: p for p in state.get("product_details", [])}
    by_item = {}
    for row in history:
        require(row["seller_id"] == actor and type(row["time_index"]) is int
                and 0 <= row["time_index"] < cutoff, "Foreign or non-prior Seller history")
        by_item.setdefault(row["item_id"], []).append(row)
    products = []
    for product in state["owned_products"]:
        item = product["item_id"]
        profile = {**product, **details.get(item, {})}
        require(profile.get("seller_id", actor) == actor, "Foreign Seller product details")
        own = {"item_id": item, "seller_id": actor,
               "history_cutoff_exclusive": cutoff,
               **{k: deepcopy(profile[k]) for k in PRODUCT_ATTRIBUTES if k in profile},
               **{k: deepcopy(product[k]) for k in (
                   "last_observed_price_raw", "last_observed_time_index",
                   "last_valid_reference", "price_status") if k in product}}
        if "last_observed_time_index" in own:
            when = own["last_observed_time_index"]
            require(type(when) is int and 0 <= when < cutoff, "Future own quote")
        if own.get("last_valid_reference"):
            require(0 <= own["last_valid_reference"]["time_index"] < cutoff, "Future fallback quote")
        summary = product.get("business_summary")
        if summary is not None:
            when = summary.get("observed_time_index")
            require(type(when) is int and 0 <= when < cutoff, "Future business summary")
            own["business_summary"] = deepcopy(summary)
            own["business_summary_scope"] = (
                "Source 30-day snapshot at observed_time_index; not additional recent quote "
                "events. Do not add these counts to historical exposure/click/purchase records.")
        own["recent_quote_observations"] = [
            {k: deepcopy(row[k]) for k in QUOTE_FIELDS if k in row}
            for row in sorted(by_item.get(item, []), key=lambda r: r["time_index"])[-10:]]
        market = product.get("market_position")
        if market is not None:
            window = market.get("window")
            require("recent" not in market or isinstance(window, dict), "Unqualified Seller market window")
            if window is not None:
                require(type(window.get("start_inclusive")) is int and
                        type(window.get("end_exclusive")) is int and
                        0 <= window["start_inclusive"] <= window["end_exclusive"] <= cutoff,
                        "Future Seller market window")
            panel = {key: deepcopy(market[key]) for key in
                     ("status", "category_ids", "window", "recent") if key in market}
            examples = market.get("competitor_examples", {})
            peers = examples.get("products", [])
            for peer in peers:
                require(type(peer.get("seller_id")) is int and peer["seller_id"] != actor
                        and type(peer.get("reference_price_time_index")) is int
                        and 0 <= peer["reference_price_time_index"] < cutoff,
                        "Own-shop or non-prior competitor quote")
            if "products" in examples:
                panel["competitor_examples"] = {
                    "available": len(peers), "shown": min(3, len(peers)),
                    "selection": deepcopy(examples.get("selection", {})),
                    "scope": "First three publicly retrieved comparable offers; not the entire market. "
                             "Same category or title similarity does not establish identical specifications.",
                    "products": [{k: deepcopy(peer[k]) for k in (
                        "item_id", "seller_id", "item_title", "brand_name", "reference_price_raw",
                        "reference_price_time_index", "price_status") if k in peer} for peer in peers[:3]]}
            own["market_position"] = panel
        products.append(with_evidence_id(own))
    require(set(state["requested_price_item_ids"]) <= {p["item_id"] for p in products},
            "Missing requested Seller product")
    return with_evidence_id({
        "profile": {k: deepcopy(v) for k, v in state["profile"].items() if k != "seller_id"},
        "evidence_actor_id": actor, "history_cutoff_exclusive": cutoff,
        "owned_products": products, "requested_price_item_ids": deepcopy(state["requested_price_item_ids"]),
        "behavior_features": deepcopy(features),
        **({"market_semantics": state["market_semantics"]} if "market_semantics" in state else {}),
        "memory_status": "No generated memories in the observed-data snapshot."})
