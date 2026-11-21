"""Descriptive Customer features from strictly prior, owner-scoped facts."""

from collections import Counter
from copy import deepcopy
import math
from statistics import mean, median


def _known(value):
    return value not in (None, "", "UNKNOWN", "unknown", "NULL")


def _summarize(events, clock):
    completed = [event for event in events if
                 isinstance(event.get("clicked_items"), list) and
                 isinstance(event.get("purchased_items"), list)]
    queries = [event["query"].strip() for event in events
               if isinstance(event.get("query"), str) and event["query"].strip()]
    clicks = [len(event["clicked_items"]) for event in completed]
    purchases = [len(event["purchased_items"]) for event in completed]
    bought = [item for event in completed for item in event["purchased_items"]]
    interacted = [item for event in completed
                  for item in event["clicked_items"] + event["purchased_items"]]
    brands = Counter(str(item["brand_name"]) for item in bought if _known(item.get("brand_name")))
    categories = Counter(str(item["category_level3_name"]) for item in interacted
                         if _known(item.get("category_level3_name")))
    prices = [item["price_raw"] for item in bought
              if type(item.get("price_raw")) in (int, float)
              and math.isfinite(item["price_raw"]) and item["price_raw"] > 0]
    purchase_brands = [set(str(item["brand_name"]) for item in event["purchased_items"]
                          if _known(item.get("brand_name"))) for event in completed
                       if event["purchased_items"]]
    pairs = [(left, right) for left, right in zip(purchase_brands, purchase_brands[1:])
             if left and right]
    return {
        "clock": clock,
        "history_events": len(events),
        "events_with_recorded_outcomes": len(completed),
        "mean_clicked_items_per_recorded_request": mean(clicks) if clicks else None,
        "mean_purchased_items_per_recorded_request": mean(purchases) if purchases else None,
        "requests_with_clicks": sum(count > 0 for count in clicks),
        "requests_with_purchases": sum(count > 0 for count in purchases),
        "distinct_query_count": len(set(queries)),
        "exact_query_repeat_fraction": 1 - len(set(queries)) / len(queries) if queries else None,
        "known_purchase_brand_observations": sum(brands.values()),
        "distinct_purchase_brand_count": len(brands),
        "largest_known_purchase_brand_share": max(brands.values()) / sum(brands.values()) if brands else None,
        "consecutive_purchase_requests_with_known_brands": len(pairs),
        "consecutive_purchase_brand_overlap_rate": mean(bool(a & b) for a, b in pairs) if pairs else None,
        "distinct_interacted_category_count": len(categories),
        "known_purchase_price_observations": len(prices),
        "median_purchased_price_raw": median(prices) if prices else None,
    }


def customer_behavior_features(observed, *, cutoff, simulated=(), simulation_before=None,
                               behavioral_summary=None, user_id=None):
    """Describe facts without converting correlations or missing data into traits.

    Callers supply authorized full history, never retrieved overlapping fragments.
    Current/equal-time and future outcomes are ignored; observed and simulated clocks
    remain separate. Background snapshots are reported separately, never summed.
    """
    if type(cutoff) is not int or cutoff <= 0:
        raise ValueError("Expected positive exclusive historical cutoff")
    if simulated and (type(simulation_before) is not int or simulation_before <= 0):
        raise ValueError("Simulated history requires an exclusive simulation cutoff")

    def prior(records, field, before):
        selected, seen = [], set()
        for event in records:
            if user_id is not None and event.get("user_id", user_id) != user_id:
                raise ValueError("Foreign Customer history")
            index = event.get(field)
            if type(index) is not int or index < 0:
                raise ValueError("History event lacks a valid ordering clock")
            if index >= before:
                continue
            identity = event.get("session_id", event.get("request_id"))
            if identity is not None:
                key = (index, str(identity))
                if key in seen:
                    raise ValueError("Duplicate historical request would double count features")
                seen.add(key)
            selected.append(event)
        return sorted(selected, key=lambda event: event[field])

    past = prior(observed, "time_index", cutoff)
    settled = prior(simulated, "settled_step", simulation_before) if simulated else []
    result = {
        "schema": "customer-descriptive-features-v1",
        "historical_cutoff_exclusive": cutoff,
        "observed": _summarize(past, "observed_time_index"),
        "limitations": [
            "Clicked-item counts describe recorded interactions, not the order or depth of comparison.",
            "Brand overlap describes observed purchases, not a verified loyalty trait.",
            "Price sensitivity is not identified: retained history omits the full historical exposure alternatives.",
            "Interacted categories do not cover all searched or considered categories.",
            "Sparse observations and zero recorded actions do not prove absence of attention or preference.",
        ],
    }
    if simulated:
        result["simulation_cutoff_exclusive"] = simulation_before
        result["simulated"] = _summarize(settled, "simulation_settled_step")
    if behavioral_summary is not None:
        when = behavioral_summary.get("observed_time_index")
        if type(when) is not int or when >= cutoff or when < 0:
            raise ValueError("Customer background is not strictly prior")
        result["background_snapshot"] = {
            "observed_time_index": when,
            "statistics_30d": deepcopy(behavioral_summary.get("statistics_30d")),
            "interpretation": "Separate source snapshot; overlaps history and is not an additive event count.",
        }
    return result
