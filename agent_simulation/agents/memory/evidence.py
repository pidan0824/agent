"""Preserve authorized evidence and assemble owner-scoped factual experiences."""

from copy import deepcopy

from agent_simulation.data.common import require
from agent_simulation.agents.context.market import public_prices
from agent_simulation.agents.context.shopping import shopping_history_event


QUERY_TOOLS = {
    "query_business",
    "query_customer_history",
    "query_public_products",
    "discover_public_products",
    "query_market_position",
    "query_shop_products",
    "read_business_report",
    "query_displayed_products",
}
EVIDENCE_KINDS = QUERY_TOOLS | {"historical_segment", "market_position", "public_products"}

def context_parts(role, context):
    """Capture factual retrieval and market information, never generated summaries."""
    for segment in context.get("retrieved_context", {}).get("history", {}).get("segments", []):
        yield "historical_segment", segment
    if role == "seller":
        for product in context.get("owned_products", []):
            if "market_position" in product:
                yield "market_position", {
                    "item_id": product["item_id"],
                    "item_title": product.get("item_title"),
                    "market_position": product["market_position"],
                    "semantics": context.get("market_semantics", ""),
                }
        for page in context.get("public_product_pages", []):
            yield "public_products", page


def check_market_visibility(position):
    """Reject old market snapshots that included other shops' private traffic."""
    require("recent_observed_products" not in position.get("coverage", {}),
            "Stored market evidence contains legacy peer traffic; start a new episode")
    own_count = {"own_value"}
    own_rate = {"own_value", "numerator", "denominator", "numerator_field", "denominator_field"}
    for metric, allowed in (("click_records", own_count), ("exposure_records", own_count),
                            ("ctr", own_rate), ("cvr", own_rate)):
        if metric in position:
            require(isinstance(position[metric], dict) and set(position[metric]) <= allowed,
                    "Stored market evidence contains legacy peer traffic; start a new episode")
    for key in ("cumulative", "simulation"):
        if key in position:
            check_market_visibility(position[key])


def eligible_evidence(entries, owner, count, state):
    """Validate stored metadata before it becomes a citable summary source."""
    visible = []
    for entry in entries:
        require(entry["kind"] in EVIDENCE_KINDS, "Unsupported memory evidence kind")
        require(entry["owner"] == owner, "Foreign summary evidence")
        # A runtime is bound to the same historical data prefix for its lifetime.
        require(
            entry["historical_cutoff_exclusive"] == state["historical_cutoff_exclusive"],
            "Summary evidence cutoff mismatch",
        )
        require(
            entry["available_step"] <= state["simulation_step"] and entry["event_count"] <= count,
            "Future summary evidence",
        )
        time = entry.get("available_logical_time")
        require(
            time is None or ("clock" in state and time <= state["clock"]["time"]),
            "Future logical-time summary evidence",
        )
        content = entry["content"]
        if entry["kind"] == "market_position" and "market_position" in content:
            check_market_visibility(content["market_position"])
        elif entry["kind"] == "query_market_position":
            check_market_visibility(content["result"])
        if entry["kind"] in ("discover_public_products", "query_public_products"):
            content = {**content, "result": public_prices(content["result"], entry["kind"])}
        elif entry["kind"] == "query_customer_history":
            result = content["result"]
            content = {
                **content,
                "result": {
                    **result,
                    **{
                        field: [shopping_history_event(event) for event in result[field]]
                        for field in ("observed_history", "simulation_history")
                        if field in result
                    },
                },
            }
        elif owner.startswith("customer:") and entry["kind"] == "historical_segment":
            content = {
                **content,
                "records": [shopping_history_event(e) for e in content["records"]],
            }
        visible.append({**entry, "content": content})
    return visible


def retrieval_text(entry):
    """Index relevant values without repeating the whole profile or provenance.

    This view is only for semantic ranking. Citations resolve to the full snapshot.
    """
    fields = {
        "query",
        "item_title",
        "brand_name",
        "category_level1_name",
        "category_level2_name",
        "category_level3_name",
        "price_raw",
        "price_status",
        "reference_price_raw",
        "exposure_records",
        "click_records",
        "purchase_records",
        "median",
        "own_value",
        "own_percentile",
    }

    def values(value):
        if isinstance(value, list):
            return [text for row in value for text in values(row)]
        if isinstance(value, dict):
            return [
                text
                for key, child in value.items()
                for text in (
                    [f"{key}: {child}"]
                    if key in fields and not isinstance(child, (dict, list))
                    else values(child)
                )
            ]
        return []

    return entry["kind"] + "\n" + "\n".join(values(entry["content"]))


def observation_key(role, clock, record, item=None):
    """Match exact identities inside one authorized actor, dataset and runtime."""
    observed = clock == "observed_time_index"
    index_field = "time_index" if observed else "settled_step"
    if role == "seller":
        return clock, record.get("item_id", item), record.get(index_field, record.get("index"))
    return clock, record["session_id" if observed else "request_id"], record[index_field]


def segment_keys(role, segment):
    return [
        observation_key(role, segment["clock"], row, segment.get("item_id"))
        for row in segment["records"]
    ]


def source_rows(role, source):
    kind = source.get("kind")
    content = source["content"]
    if kind == "historical_segment":
        return zip(segment_keys(role, content), content["records"])
    if kind is None:
        clock = (
            "observed_time_index" if source["origin"] == "observed" else "simulation_settled_step"
        )
        rows = content if role == "seller" else [content]
        return [(observation_key(role, clock, row), row) for row in rows]
    result = content.get("result", {})
    if kind == "query_business":
        observed = result.get("series", [])
        simulated = result.get("simulation", {}).get("settled_series", [])
    elif kind == "query_customer_history":
        observed = result.get("observed_history", [])
        simulated = result.get("simulation_history", [])
    else:
        return []
    return [(observation_key(role, "observed_time_index", row), row) for row in observed] + [
        (observation_key(role, "simulation_settled_step", row), row) for row in simulated
    ]


def evidence_reference(source, role, positions):
    reference = {
        k: source[k]
        for k in (
            "origin",
            "time_index",
            "settled_step",
            "kind",
            "available_step",
            "available_logical_time",
            "historical_cutoff_exclusive",
        )
        if k in source
    }
    if positions:
        shared = sorted(
            {pos for key, _ in source_rows(role, source) for pos in positions.get(key, [])}
        )
        if shared:
            reference["same_observations_in_retrieved_history"] = [
                {"segment_number": segment, "record_number": record} for segment, record in shared
            ]
    return reference


def actor_key(state, role, actor):
    require(role in ("customer", "seller"), "Unknown memory role")
    known = (
        str(actor) in state["customers"]
        if role == "customer"
        else any(p["seller_id"] == actor for p in state["products"].values())
    )
    require(known, "Unknown memory owner")
    return f"{role}:{actor}"


def _seller_feedback(state, actor):
    return (event for event in state["feedback"] if event["seller_id"] == actor)


def experience_counts(state, role, actor):
    """Count a bound actor's observed events and distinct settlements."""
    if role == "customer":
        customer = state["customers"][str(actor)]
        return len(customer["observed_history"]), len(customer["simulation_history"])
    return (
        len(state["initial_seller_memory"][str(actor)]),
        len({event["settled_step"] for event in _seller_feedback(state, actor)}),
    )


def experiences(state, role, actor):
    actor_key(state, role, actor)
    if role == "customer":
        customer = state["customers"][str(actor)]
        observed = [
            {
                "source_id": f'{state["dataset"]}:customer:{actor}:session:{e["session_id"]}',
                "origin": "observed",
                "time_index": e["time_index"],
                "content": shopping_history_event(e),
            }
            for e in customer["observed_history"]
        ]
        simulated = [
            {
                "source_id": f'customer:{actor}:request:{e["request_id"]}',
                "origin": "simulated",
                "settled_step": e["settled_step"],
                "content": shopping_history_event(e),
            }
            for e in customer["simulation_history"]
        ]
    else:
        # Seed events already group the shop's retained product rows by time_index.
        observed = [
            {
                "source_id": f'{state["dataset"]}:seller:{actor}:time:{e["time_index"]}',
                "origin": "observed",
                **deepcopy(e),
            }
            for e in state["initial_seller_memory"][str(actor)]
        ]
        grouped = {}
        for event in _seller_feedback(state, actor):
            grouped.setdefault(event["settled_step"], []).append(deepcopy(event))
        simulated = [
            {
                "source_id": f"seller:{actor}:settlement:{step}",
                "origin": "simulated",
                "settled_step": step,
                "content": events,
            }
            for step, events in sorted(grouped.items())
        ]
    require(
        all(0 <= e["time_index"] < state["historical_cutoff_exclusive"] for e in observed),
        "Future initialization memory",
    )
    require(
        all(0 < e["settled_step"] <= state["simulation_step"] for e in simulated),
        "Future simulated memory",
    )
    events = observed + simulated
    require(len({e["source_id"] for e in events}) == len(events), "Duplicate memory source")
    return events


def with_settled_offers(selected, feedback):
    """Join displayed-price snapshots by request and settlement step within one runtime."""
    by_request = {
        (event["content"]["request_id"], event["settled_step"]): []
        for event in selected if event["origin"] == "simulated"
    }
    if not by_request:
        return selected
    for row in feedback:
        key = (row["request_id"], row["settled_step"])
        if key in by_request:
            by_request[key].append({k: row[k] for k in (
                "item_id", "price_raw", "click_records", "purchase_records"
            )})
    return [
        {**event, "displayed_offers": by_request[event["content"]["request_id"], event["settled_step"]]}
        if event["origin"] == "simulated" else event
        for event in selected
    ]


def reflection_profile(role, profile, selected):
    """Attach product descriptions for selected evidence, not the entire shop.

    The complete profile remains available to retrieval and the source store.
    No selected experience or its product description is truncated.
    """
    if role != "seller":
        return deepcopy(profile)
    from agent_simulation.data.market.products import PROFILE_FIELDS

    result = {k: deepcopy(v) for k, v in profile.items() if k != "products"}
    item_ids = {row["item_id"] for event in selected for row in event["content"]}
    products = profile["products"]
    result["shop_product_count"] = len(products)
    result["products"] = [
        {k: deepcopy(v) for k, v in p.items() if k in PROFILE_FIELDS}
        for p in products
        if p["item_id"] in item_ids
    ]
    require(
        {p["item_id"] for p in result["products"]} == item_ids,
        "Selected reflection evidence lacks a product description",
    )
    return result
