"""Shared business measures and market price distributions."""

from bisect import bisect_left, bisect_right
import math
from agent_simulation.data.common import require

COUNTS = ("exposure_records", "click_records", "purchase_records")
QUANTITY_ASSUMPTION = 'Estimated units assume one unit per purchase record; actual purchased quantities are unavailable.'
METRICS = (
    "click_through_rate",
    "purchase_rate",
    "click_to_purchase_rate",
    "estimated_sales_amount_raw",
)
BUSINESS_NOTE = 'purchase_records counts purchase flags; estimated_units_sold assumes one unit per flag. Prices and estimated amounts retain stored values, interpreted as fen at the Agent interface. Observed time_index and simulation event indices are separate ordering clocks, not elapsed durations.'


def price_evidence(records, *, paired=False):
    """Aggregate value only for priced purchases, without treating missing prices as zero."""
    priced = [
        r
        for r in records
        if r.get("price_status") == "observed"
        and type(r.get("price_raw")) in (int, float)
        and math.isfinite(r["price_raw"])
        and r["price_raw"] > 0
    ]
    return {
        "clicked_and_purchased_records": (
            sum(r["purchase_records"] for r in records if r["click_records"])
            if paired
            else None
        ),
        "price_covered_purchase_records": sum(r["purchase_records"] for r in priced),
        "priced_amount_raw": math.fsum(
            r["price_raw"] * r["purchase_records"] for r in priced
        ),
    }


def statistics(counts, requested, evidence):
    """Return only requested metrics, with explicit denominators and valuation coverage."""
    require(
        len(requested) == len(set(requested)) and set(requested) <= set(METRICS),
        "Unknown or duplicate metric",
    )
    result = {}
    for name, numerator, denominator in (
        ("click_through_rate", counts["click_records"], counts["exposure_records"]),
        ("purchase_rate", counts["purchase_records"], counts["exposure_records"]),
        (
            "click_to_purchase_rate",
            evidence["clicked_and_purchased_records"],
            counts["click_records"],
        ),
    ):
        if name in requested:
            result[name] = {
                "value": (
                    numerator / denominator
                    if numerator is not None and denominator
                    else None
                ),
                "numerator": numerator,
                "denominator": denominator,
                "status": (
                    "unavailable_joint_labels"
                    if numerator is None
                    else "defined" if denominator else "zero_denominator"
                ),
            }
    if "estimated_sales_amount_raw" in requested:
        purchases, covered = (
            counts["purchase_records"],
            evidence["price_covered_purchase_records"],
        )
        result["estimated_sales_amount_raw"] = {
            "value": (
                evidence["priced_amount_raw"] if covered or not purchases else None
            ),
            "price_covered_purchase_records": covered,
            "purchase_records": purchases,
            "status": (
                "complete"
                if covered == purchases
                else "partial" if covered else "no_valid_price"
            ),
        }
    return result


def totals(events):
    exposures = clicks = purchases = 0
    for event in events:
        exposures += event["exposure_records"]
        clicks += event["click_records"]
        purchases += event["purchase_records"]
    return dict(
        exposure_records=exposures,
        click_records=clicks,
        purchase_records=purchases,
        estimated_units_sold=purchases,
    )


MARKET_METRICS = ("price_raw", *COUNTS, "ctr", "cvr")


def public_price_stats(stats):
    """One peer supports a price comparison but no useful market percentile."""
    return {k: v for k, v in stats.items() if k != "nonzero_products"
            and (k != "own_percentile" or stats["observed_products"] >= 2)}


def distribution(values, own_value):
    values = sorted(v for v in values if v is not None)
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in values),
        "Invalid market statistic",
    )
    if not values:
        return dict(
            observed_products=0,
            p25=None,
            median=None,
            p75=None,
            own_value=own_value,
            own_percentile=None,
            nonzero_products=0,
        )

    def quantile(q):
        p = (len(values) - 1) * q
        i = int(p)
        return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (p - i)

    rank = None
    if own_value is not None:
        lower = bisect_left(values, own_value)
        upper = bisect_right(values, own_value)
        rank = (lower + 0.5 * (upper - lower)) / len(values)
    return dict(
        observed_products=len(values),
        p25=quantile(0.25),
        median=quantile(0.5),
        p75=quantile(0.75),
        own_value=own_value,
        own_percentile=rank,
        nonzero_products=sum(v != 0 for v in values),
    )


def metrics(record, prefix=""):
    counts = {k: record.get(prefix + k) for k in COUNTS}
    exposures = counts["exposure_records"]
    return {
        "price_raw": record["price_raw"],
        **counts,
        "ctr": counts["click_records"] / exposures if exposures else None,
        "cvr": counts["purchase_records"] / exposures if exposures else None,
    }


def market_position(own, peers, *, window, clock):
    """Use one cohort with recent observations and usable prices for every comparison."""
    comparable = [
        p
        for p in peers
        if p["price_raw"] is not None and p.get("exposure_records") is not None
    ]
    own_eligible = (
        own["price_raw"] is not None and own.get("exposure_records") is not None
    )
    values = [metrics(p) for p in comparable]
    current = metrics(own)
    cumulative = [metrics(p, "cumulative_") for p in peers]
    own_cumulative = metrics(own, "cumulative_")
    price_indices = [
        p["price_time_index"]
        for p in comparable
        if p.get("price_time_index") is not None
    ]
    return {
        "clock": clock,
        "window": window,
        "peer_products": len(peers),
        "price_time_basis": {
            "clock": clock,
            "selection": "latest_observed_reference_before_cutoff",
            "cutoff_exclusive": window["end_exclusive"],
            "comparable_peer_observation_range": {
                "earliest_index": min(price_indices) if price_indices else None,
                "latest_index": max(price_indices) if price_indices else None,
            },
        },
        "coverage": {
            "known_peer_products": len(peers),
            "recent_observed_products": sum(
                p.get("exposure_records") is not None for p in peers
            ),
            "usable_reference_prices": sum(p["price_raw"] is not None for p in peers),
            "comparable_products": len(comparable),
            "comparable_fraction": len(comparable) / len(peers) if peers else None,
            "own_comparable": own_eligible,
        },
        "current_reference": {
            "price_raw": own["price_raw"],
            "observation_index": own.get("price_time_index"),
            "status": own.get("price_status", "active_simulation_price"),
            **(
                {"active_price_basis": own["active_price_basis"]}
                if "active_price_basis" in own
                else {}
            ),
        },
        **{
            k: distribution(
                [v[k] for v in values], current[k] if own_eligible else None
            )
            for k in MARKET_METRICS
        },
        "cumulative": {
            "scope": 'All visible records before the cutoff, reported separately from recent-window comparisons.',
            "clock": clock,
            "window": {
                "end_exclusive": window["end_exclusive"],
                "selection": "all_available_history_per_product_before_cutoff",
            },
            **{
                k: distribution([v[k] for v in cumulative], own_cumulative[k])
                for k in COUNTS
            },
        },
    }


MARKET_NOTE = (
    "Peers are other shops' products in the same complete three-level category. "
    "Recent comparisons use peers with observations in the stated window and a usable latest price before the cutoff. "
    "Missing observations and invalid latest prices remain unknown. "
    "CTR and CVR divide click and purchase records by exposure records. Zero exposure leaves both rates undefined. "
    "Cumulative counts use their own window. Purchase records do not include quantities. "
    "Peer aggregates describe this sandbox and do not imply access to private transactions."
)
