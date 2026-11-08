"""Select the Customer's visible shopping history and displayed product details."""

from copy import deepcopy

from agent_simulation.data.common import require
from agent_simulation.data.history_records import customer_history_limit
from agent_simulation.data.market.products import PROFILE_FIELDS
from agent_simulation.agents.retrieval.history import DEFAULT_HISTORY_ENCODER, prepare_customer_history


PRODUCT_FIELDS = (*PROFILE_FIELDS, "price_raw")


def context(
    profile,
    observed,
    simulated=(),
    long_term=(),
    limit=None,
    *,
    cutoff,
    history_query="",
    history_embedder=DEFAULT_HISTORY_ENCODER,
    simulation_before=None,
    prepared_history=None,
):
    """Assemble recent and retrieved history from the same authorized snapshot."""
    n = customer_history_limit() if limit is None else limit
    pool = prepare_customer_history(
        observed, simulated, cutoff=cutoff, recent_n=n, user_id=profile.get("user_id"),
        simulation_before=simulation_before, prepared=prepared_history,
    )
    return {
        "profile": deepcopy(profile),
        "history_cutoff_exclusive": cutoff,
        "short_term_memory": {"search_history": pool.recent_history()},
        "long_term_memory": deepcopy(list(long_term)),
        "retrieved_context": {"history": pool.retrieve(
            history_query, embedder=history_embedder, diagnostics=False)},
    }


def model_product(product):
    """Expose the displayed quote and an available reason when its price is unknown."""
    result = {key: product[key] for key in PRODUCT_FIELDS if key in product}
    if product.get("price_raw") is None and product.get("price_status") not in (None, "observed"):
        result["price_status"] = product["price_status"]
    return result


def shopping_history_event(event):
    """Expose only the product snapshots the Customer could see at decision time."""
    result = {
        k: deepcopy(v) for k, v in event.items() if k not in ("clicked_items", "purchased_items")
    }
    for field in ("clicked_items", "purchased_items"):
        if field in event:
            result[field] = [model_product(product) for product in event[field]]
    return result


def displayed_details(displayed, item_ids):
    requested = set(item_ids)
    require(requested and len(item_ids) == len(requested), "Empty or duplicate product query")
    by_id = {product["item_id"]: product for product in displayed}
    if not requested <= by_id.keys():
        raise PermissionError("Product is outside the current displayed candidate batch")
    allowed = (*PRODUCT_FIELDS, "price_status")
    return {
        "products": [{key: by_id[item][key] for key in allowed if key in by_id[item]} for item in item_ids]
    }


def project_customer(state):
    """Keep event boundaries and clocks while omitting internal search metadata."""
    require(isinstance(state, dict), "Customer context must be an object")
    profile = state.get("profile", {})
    require(isinstance(profile, dict), "Customer profile must be an object")
    owner = profile.get("user_id")

    def events(records):
        require(isinstance(records, list), "Customer history records must be a list")
        result = []
        for event in records:
            require(isinstance(event, dict), "Customer history record must be an object")
            # Catalog events are already owner-scoped and may omit the user ID.
            if "user_id" in event:
                require(event["user_id"] == owner, "Foreign Customer event")
            visible_event = dict(event)
            visible_event.pop("session_id", None)
            result.append(visible_event)
        return result

    visible = dict(state)
    memory = state.get("short_term_memory", {})
    require(isinstance(memory, dict), "Customer short-term memory must be an object")
    if "search_history" in memory:
        visible["short_term_memory"] = {**memory, "search_history": events(memory["search_history"])}

    retrieved = state.get("retrieved_context", {})
    require(isinstance(retrieved, dict), "Customer retrieved context must be an object")
    if "history" in retrieved:
        history = retrieved["history"]
        require(isinstance(history, dict), "Retrieved Customer history must be an object")
        segments = history.get("segments")
        require(isinstance(segments, list), "Retrieved Customer segments must be a list")
        selected = []
        for segment in segments:
            require(isinstance(segment, dict), "Retrieved Customer segment must be an object")
            if "owner_id" in segment:
                require(segment["owner_id"] == owner, "Foreign retrieved Customer history")
            require("records" in segment, "Retrieved Customer segment has no records")
            # The record carries time_index or settled_step, and the segment keeps its clock.
            visible_segment = {**segment, "records": events(segment["records"])}
            for key in ("owner_id", "start_inclusive", "end_inclusive"):
                visible_segment.pop(key, None)
            selected.append(visible_segment)
        visible_history = {**history, "segments": selected}
        for key in ("eligible_segments", "returned_segments"):
            visible_history.pop(key, None)
        visible["retrieved_context"] = {**retrieved, "history": visible_history}
    if "long_term_memory" in state:
        visible["long_term_memory"] = [
            {key: value for key, value in memory.items()
             if key not in ("memory_id", "owner", "source_ids")}
            for memory in state["long_term_memory"]
        ]
    return visible
