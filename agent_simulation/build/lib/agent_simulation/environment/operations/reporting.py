"""Aggregate business feedback, price intervals and report windows."""

from agent_simulation.data.common import require
from collections import defaultdict
from bisect import bisect_right

from agent_simulation.data.market.statistics import COUNTS, QUANTITY_ASSUMPTION, price_evidence, statistics, totals


def cumulative_series(events, visible_keys):
    """Return cumulative simulation counts only at requested, settled item-step points."""
    grouped = defaultdict(list)
    for event in events:
        grouped[(event["item_id"], event["settled_step"])].append(event)
    running = defaultdict(lambda: {k: 0 for k in COUNTS})
    result = []
    for (item, step), rows in sorted(grouped.items(), key=lambda pair: (pair[0][1], pair[0][0])):
        for k in COUNTS:
            running[item][k] += sum(r[k] for r in rows)
        if (item, step) in visible_keys:
            result.append(
                {
                    "item_id": item,
                    "settled_step": step,
                    **running[item],
                    "estimated_units_sold": running[item]["purchase_records"],
                }
            )
    return result


def price_intervals(state, seller, *, before=None):
    """Count exposures at search time and responses at feedback time in price-change windows."""
    before = state["simulation_step"] + 1 if before is None else before
    require(1 <= before <= state["simulation_step"] + 1, "Invalid simulation cutoff")
    actions = defaultdict(list)
    for action in state["price_actions"]:
        if action["seller_id"] == seller:
            actions[action["item_id"]].append(action)
    result = {}
    starts = {}
    for key, product in state["products"].items():
        if product["seller_id"] != seller:
            continue
        item = int(key)
        ordered = sorted(actions[item], key=lambda a: a["simulation_step"])
        initial = (
            ordered[0]["previous_price_raw"]
            if ordered
            else state["active_prices"][key]["price_raw"]
        )
        windows = [{"start_step_inclusive": 0, "price_raw": initial}]
        for action in ordered:
            if action["simulation_step"] >= before:
                break
            # Explicitly retaining a price must not create a new price interval.
            if action["price_raw"] != windows[-1]["price_raw"]:
                windows.append(
                    {
                        "start_step_inclusive": action["simulation_step"],
                        "price_raw": action["price_raw"],
                    }
                )
        for index, window in enumerate(windows):
            window.update(
                end_step_exclusive=(
                    windows[index + 1]["start_step_inclusive"] if index + 1 < len(windows) else None
                ),
                outcomes_through_step_inclusive=before - 1,
                **totals([]),
            )
        result[item] = windows
        starts[item] = [w["start_step_inclusive"] for w in windows]
    for event in state["feedback"]:
        if event["seller_id"] != seller:
            continue
        item = event["item_id"]
        if event["search_step"] < before:
            window = result[item][bisect_right(starts[item], event["search_step"]) - 1]
            window["exposure_records"] += event["exposure_records"]
        if event["settled_step"] < before:
            window = result[item][bisect_right(starts[item], event["settled_step"]) - 1]
            for k in ("click_records", "purchase_records"):
                window[k] += event[k]
            window["estimated_units_sold"] = window["purchase_records"]
    # A displayed request is an exposure even if the Customer has not responded yet.
    for request in state["pending"].values():
        step = request["simulation_step"]
        if step >= before:
            continue
        for product in request["displayed_items"]:
            if product["seller_id"] == seller:
                item = product["item_id"]
                result[item][bisect_right(starts[item], step) - 1]["exposure_records"] += 1
    return result


def report(state, seller, after):
    owned = sorted(int(i) for i, p in state["products"].items() if p["seller_id"] == seller)
    require(bool(owned), "Unknown Seller")
    all_events = defaultdict(list)
    for event in state["feedback"]:
        if event["seller_id"] == seller and event["settled_step"] <= state["simulation_step"]:
            all_events[event["item_id"]].append(event)
    events = [e for rows in all_events.values() for e in rows if e["settled_step"] > after]
    intervals = price_intervals(state, seller)
    return {
        "after_step_exclusive": after,
        "through_step_inclusive": state["simulation_step"],
        "shop_exposed_requests": len({e["request_id"] for e in events}),
        "quantity_assumption": QUANTITY_ASSUMPTION,
        "cumulative_scope": "Settled simulation feedback from initialization through this report; observed history is separate.",
        "price_interval_scope": "All time windows between actual product price changes. Exposures use search_step (including pending requests); clicks and purchases use settled_step, their recorded feedback time. Late responses count in the window when feedback arrives, regardless of displayed price. Null end marks the current interval; initialization is step 0. These are market outcomes, not causal price effects.",
        "period_totals_scope": "Top-level counts retain newly settled-record totals since the previous report; price_intervals instead count each metric at its own event time.",
        "shop_totals": totals(events),
        "shop_cumulative_totals": totals([e for rows in all_events.values() for e in rows]),
        "products": [
            {
                "item_id": i,
                "price_intervals": intervals[i],
                **totals([e for e in all_events[i] if e["settled_step"] > after]),
                "cumulative": totals(all_events[i]),
            }
            for i in owned
        ],
    }


def read(state, seller):
    # Share one read-only shop slice across the report and timeline calculations.
    state = {
        **state,
        "products": {k: p for k, p in state["products"].items() if p["seller_id"] == seller},
        "feedback": [e for e in state["feedback"] if e["seller_id"] == seller],
        "price_actions": [a for a in state["price_actions"] if a["seller_id"] == seller],
    }
    if "clock" in state:
        previous = state["clock"]["last_decisions"].get(f"seller:{seller}", {})
        result = report(state, seller, previous.get("input_step", 0))
        result["logical_window"] = {
            "start_inclusive": previous.get("time", state["clock"]["start"]),
            "end_exclusive": state["clock"]["time"],
        }
        timelines = activity_timeline(state, seller)
        for product in result["products"]:
            product["activity_intervals"] = timelines[product["item_id"]]
        return result
    return report(state, seller, 0)


def activity_timeline(state, seller, *, item_ids=None):
    """Compress inactive logical-time spans, splitting at price changes and active Ticks.

    Only completed simulator intervals can establish zero activity. Observed-data
    gaps remain unknown. Counts are aggregated without synthesizing empty events.
    """
    clock = state["clock"]
    start, end, delta = clock["start"], clock["time"], clock["tick_size"]
    items = {int(k) for k, p in state["products"].items() if p["seller_id"] == seller}
    if item_ids is not None:
        items.intersection_update(item_ids)
    prices, events, pending = defaultdict(list), defaultdict(list), defaultdict(list)
    for action in state["price_actions"]:
        if action["item_id"] in items and action["seller_id"] == seller and action["price_raw"] != action["previous_price_raw"]:
            prices[action["item_id"]].append(action)
    for event in state["feedback"]:
        if event["item_id"] in items and event["seller_id"] == seller:
            events[event["item_id"]].append(event)
    for request in state["pending"].values():
        for product in request["displayed_items"]:
            if product["item_id"] in items:
                pending[product["item_id"]].append(request["logical_time"])
    result = {}
    for item in sorted(items):
        key = str(item)
        changes = sorted(prices[item], key=lambda a: a["simulation_step"])
        current = (
            changes[0]["previous_price_raw"]
            if changes
            else state["active_prices"][key]["price_raw"]
        )
        changed_at = {a["logical_time"]: a["price_raw"] for a in changes}
        counts = defaultdict(lambda: {k: 0 for k in COUNTS})
        boundaries = {start, end} | {t for t in changed_at if start <= t < end}
        # Pending lists are already exposed. Settlement replaces them with feedback.
        for t in pending[item]:
            if start <= t < end:
                boundaries.update((t, min(t + delta, end)))
                counts[t]["exposure_records"] += 1
        for event in events[item]:
            for field, names in (
                ("search_time", ("exposure_records",)),
                ("settled_time", ("click_records", "purchase_records")),
            ):
                t = event[field]
                if start <= t < end:
                    boundaries.update((t, min(t + delta, end)))
                    for name in names:
                        counts[t][name] += event[name]
        ordered = sorted(boundaries)
        spans = []
        for left, right in zip(ordered, ordered[1:]):
            current = changed_at.get(left, current)
            row = {
                "start_inclusive": left,
                "end_exclusive": right,
                "price_raw": current,
                **counts[left],
                "estimated_units_sold": counts[left]["purchase_records"],
            }
            if (
                spans
                and spans[-1]["price_raw"] == current
                and all(spans[-1][k] == row[k] == 0 for k in COUNTS)
            ):
                spans[-1]["end_exclusive"] = right
            else:
                spans.append(row)
        result[item] = spans
    return result


def simulation_statistics(by_item, metrics, start, before):
    items = []
    for item, records in by_item.items():
        counts = totals(records)
        items.append({"item_id": item, **counts,
                      "metrics": statistics(counts, metrics, price_evidence(records, paired=True))})
    return {
        "clock": "simulation_feedback_index",
        "start_inclusive": start,
        "end_exclusive": before,
        "scope": "All settled request-item records with feedback indices in the requested window, independent of detail limit. Pending responses are excluded.",
        "rate_basis": "Same settled request-item cohort; click-to-purchase uses joint click/purchase flags. Do not divide event-time price-interval counts to obtain these rates.",
        "amount_basis": "Each purchased item's recorded displayed price, one estimated unit; not actual paid revenue. Partial values cover only priced purchases.",
        "items": items,
    }


def windowed_activity_timeline(state, actor, item_ids, start_time, end_time, limit):
    timelines = activity_timeline(state, actor, item_ids=item_ids)
    result = []
    for item in item_ids:
        spans = [
            {
                **s,
                "start_inclusive": max(start_time, s["start_inclusive"]),
                "end_exclusive": min(end_time, s["end_exclusive"]),
            }
            for s in timelines[item]
            if max(start_time, s["start_inclusive"]) < min(end_time, s["end_exclusive"])
        ]
        selected = spans[-limit :]
        result.append(
            {
                "item_id": item,
                "intervals": selected,
                "has_more": len(spans) > len(selected),
                "next_before_logical_time": (
                    selected[0]["start_inclusive"] if len(spans) > len(selected) else None
                ),
                "start_inclusive": start_time,
                "end_exclusive": end_time,
            }
        )
    return result
