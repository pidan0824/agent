"""Strictly prior Seller change evidence and versioned teacher release guards."""
from collections import defaultdict, Counter
from copy import deepcopy
import hashlib
import json
import math
from statistics import median

from agent_simulation.data.common import require
from agent_simulation.data.history_records import has_observed_price
from agent_simulation.agents.context.seller_features import with_evidence_id

SELLER_CHANGE_PROTOCOL = "seller-change-v1"
SELLER_DIRECTIONS = ("keep", "up", "down")


def validate_signal_policy(policy):
    require(isinstance(policy, dict) and policy.get("protocol") == SELLER_CHANGE_PROTOCOL,
            "Seller change context requires an explicit fitted policy")
    require(type(policy.get("window_size")) is int and policy["window_size"] > 0,
            "Positive Seller signal window required")
    require(type(policy.get("exposure_min")) in (int, float) and
            math.isfinite(policy["exposure_min"]) and policy["exposure_min"] >= 1,
            "Positive fitted exposure threshold required")
    require(bool(policy.get("fit_case_ids_sha256")), "Missing fit calibration identity")


def window_exposures(history, item_id, cutoff, window_size):
    rows = [r for r in history if r["item_id"] == item_id and
            max(0, cutoff-window_size) <= r["time_index"] < cutoff]
    values = [r.get("exposure_records") for r in rows]
    known = [v for v in values if type(v) in (int, float) and math.isfinite(v) and v >= 0]
    return {"exposures": sum(known) if known else None,
            "observations": len(rows), "known_exposure_observations": len(known)}


def fit_signal_policy(fit_rows, *, window_size=10000):
    """Calibrate only observed prior traffic in fit, without using direction labels."""
    require(fit_rows and all(r["partition"] == "fit" for r in fit_rows),
            "Signal calibration requires only fit cases")
    ids = sorted(r["case_id"] for r in fit_rows)
    require(len(set(ids)) == len(ids), "Duplicate fit calibration identity")
    values = sorted(r["exposures"] for r in fit_rows
                    if r["exposures"] is not None and r["exposures"] > 0)
    # Nearest-rank lower quartile of positive observed fit traffic, minimum one.
    threshold = max(1, values[max(0, math.ceil(.25*len(values))-1)]) if values else 1
    return {"protocol": SELLER_CHANGE_PROTOCOL, "window_size": window_size,
            "exposure_min": threshold, "fit_case_ids_sha256": hashlib.sha256(
                json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
            "fit_cases": len(ids), "fit_positive_traffic_cases": len(values),
            "fit_cutoff_min": min(r["cutoff"] for r in fit_rows),
            "fit_cutoff_max": max(r["cutoff"] for r in fit_rows),
            "threshold_method": "nearest-rank P25 of positive strictly-prior fit window exposures; floor 1",
            "threshold_uses_target_direction": False,
            "scope": "Descriptive traffic sufficiency convention, not validated demand identifiability"}


def change_signals(history, *, seller_id, item_id, cutoff, policy):
    validate_signal_policy(policy)
    groups = defaultdict(list)
    for row in history:
        require(row["seller_id"] == seller_id and type(row["time_index"]) is int
                and 0 <= row["time_index"] < cutoff, "Foreign or non-prior Seller signal history")
        groups[row["item_id"]].append(row)
    for records in groups.values():
        records.sort(key=lambda r: r["time_index"])
        require(len({r["time_index"] for r in records}) == len(records),
                "Duplicate Seller signal observation")
    own = groups.get(item_id, [])
    quotes = [r for r in own if has_observed_price(r)]
    require(quotes, "Seller change evidence needs prior valid own quote")
    changes = [(i, a, b) for i, (a, b) in enumerate(zip(quotes, quotes[1:]), 1)
               if a["price_raw"] != b["price_raw"]]
    last = changes[-1] if changes else None
    own_evidence = {"seller_id": seller_id, "item_id": item_id,
        "history_cutoff_exclusive": cutoff, "history_observation_count": len(own),
        "valid_quote_observation_count": len(quotes),
        "first_observed_time_index": own[0]["time_index"],
        "last_valid_price_raw": quotes[-1]["price_raw"],
        "last_valid_quote_time_index": quotes[-1]["time_index"],
        "last_observed_time_index": own[-1]["time_index"],
        "prior_change_count": len(changes), "has_observed_change": bool(changes),
        "observations_since_last_change": len(quotes)-1-last[0] if last else None,
        "last_change": ({"before_time_index": last[1]["time_index"],
                        "after_time_index": last[2]["time_index"],
                        "before_price_raw": last[1]["price_raw"],
                        "after_price_raw": last[2]["price_raw"],
                        "direction": "up" if last[2]["price_raw"] > last[1]["price_raw"] else "down",
                        "relative_change": (last[2]["price_raw"]-last[1]["price_raw"])/last[1]["price_raw"]}
                       if last else None),
        "no_change_status": None if last else "未观察到变价；起点以前及未观测区间未知",
        "history_length_note": "Change evidence depends on observation count and coverage; no calendar-duration claim."}
    start = max(0, cutoff-policy["window_size"])
    traffic = window_exposures(history, item_id, cutoff, policy["window_size"])
    sufficient = (traffic["exposures"] is not None and
                  traffic["known_exposure_observations"] == traffic["observations"] and
                  traffic["exposures"] >= policy["exposure_min"])
    own_evidence["traffic_evidence"] = {
        **traffic, "window": {"start_inclusive": start, "end_exclusive": cutoff},
        "exposure_min": policy["exposure_min"], "status": "meets_fitted_exposure_threshold" if sufficient else "insufficient",
        "interpretation": ("Traffic is descriptive; does not establish causal demand or elasticity." if sufficient
                           else "流量数据不足，不支持推断"),
        "scope": "Retained quote-event exposure records; do not add source 30-day snapshots."}
    known, missing, up, down = 0, 0, 0, 0
    ending_times, previous_times = [], []
    for peer, records in groups.items():
        if peer == item_id:
            continue
        valid = [r for r in records if has_observed_price(r)]
        if not valid or valid[-1]["time_index"] < start:
            continue
        if len(valid) < 2:
            missing += 1
            continue
        before, after = valid[-2:]
        known += 1
        up += after["price_raw"] > before["price_raw"]
        down += after["price_raw"] < before["price_raw"]
        ending_times.append(after["time_index"]); previous_times.append(before["time_index"])
    activity = {"seller_id": seller_id, "excluded_item_id": item_id,
        "window": {"start_inclusive": start, "end_exclusive": cutoff},
        "denominator_known_previous_quote": known, "missing_previous_quote_products": missing,
        "changed_products": up+down, "up_products": up, "down_products": down,
        "change_fraction": (up+down)/known if known else None,
        "net_direction": (up-down)/known if known else None,
        "status": "observed" if known else "无数据",
        "latest_ending_time_index": max(ending_times) if ending_times else None,
        "earliest_previous_time_index": min(previous_times) if previous_times else None,
        "scope": "One latest valid quote per other owned product ending in window; previous quote may predate window. Missing predecessors excluded; shared activity is not a causal price instruction."}
    return own_evidence, activity


def upgrade_seller_context(context, history, policy):
    """Add separate own-quote/shop/market evidence blocks; old context remains readable."""
    result = deepcopy(context)
    cutoff = result["history_cutoff_exclusive"]
    actor = result.get("evidence_actor_id", result.get("profile", {}).get("seller_id"))
    require(type(actor) is int, "Missing Seller signal owner")
    for product in result["owned_products"]:
        item = product["item_id"]
        signals, shop = change_signals(history, seller_id=actor, item_id=item, cutoff=cutoff, policy=policy)
        quote = {key: product.pop(key) for key in (
            "last_observed_price_raw", "last_observed_time_index", "last_valid_reference",
            "price_status", "recent_quote_observations") if key in product}
        if "recent_quote_observations" not in quote and "recent_quote_observations" in result:
            require(len(result["owned_products"]) == 1, "Ambiguous legacy own quote observations")
            quote["recent_quote_observations"] = result.pop("recent_quote_observations")
        quote.update(signals)
        product["quote_evidence"] = with_evidence_id(quote)
        product["same_shop_activity"] = with_evidence_id(shop)
        market = product.get("market_position")
        if market is None and "market_position" in result:
            require(len(result["owned_products"]) == 1, "Ambiguous legacy market evidence")
            market = result.pop("market_position")
        if market is not None:
            product["market_position"] = with_evidence_id({
                **market, "item_id": item, "history_cutoff_exclusive": cutoff,
                "scope": "Category reference distribution and retrieved offers; specifications may differ."})
        product.update(with_evidence_id(product))
    result.update(context_protocol=SELLER_CHANGE_PROTOCOL,
                  signal_policy={"window_size": policy["window_size"], "exposure_min": policy["exposure_min"],
                                 "fit_case_ids_sha256": policy["fit_case_ids_sha256"]})
    return with_evidence_id(result)


def seller_coverage(rows, *, accepted=None):
    """Count every budget stratum; zero accepted in any class blocks new-protocol release."""
    accepted = accepted if accepted is not None else [r for r in rows if
        r.get("status") == "ok" and r.get("style") != "direct" and r.get("quality", {}).get("accepted", False)]
    def counts(values):
        unique = {r.get("case_id", "row:" + str(i)): r for i, r in enumerate(values)
                  if r.get("role") == "seller"}
        return Counter(r["stratum"] for r in unique.values())
    totals, passed = counts(rows), counts(accepted)
    values = {direction: {"attempts": totals[direction], "passed": passed[direction]}
              for direction in SELLER_DIRECTIONS}
    missing = [direction for direction in SELLER_DIRECTIONS if not passed[direction]]
    return {"protocol": SELLER_CHANGE_PROTOCOL, "strata": values, "release_allowed": not missing,
            "counting_unit": "unique frozen case IDs when available; direct outputs never count as rationales",
            "missing_pass_strata": missing,
            "reason": "all_change_strata_have_passing_rationale" if not missing else "zero_passing_seller_change_stratum"}
