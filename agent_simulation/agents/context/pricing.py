"""Select owned products, their observed history, and the shop overview."""

from collections import defaultdict
from copy import deepcopy

from agent_simulation.data.common import require
from agent_simulation.data.market.statistics import BUSINESS_NOTE
from agent_simulation.agents.prompts.instructions import SHOP_OVERVIEW_NOTE
from agent_simulation.data.history_records import (
    ObservedPriceHistory, PRICE_HISTORY_POLICY, group_business_history, has_observed_price,
    price_point, recent_business_history,
)
from agent_simulation.data.market.products import PROFILE_FIELDS, PUBLIC_FIELDS


def history_window(cutoff, observations_per_item):
    return {
        "selection": "latest_observations_per_item",
        "observations_per_item": observations_per_item,
        "end_exclusive": cutoff,
    }


def build_pricing_context(
    observed,
    *,
    owner_id,
    cutoff,
    observations_per_item,
    public_pages,
    requested_item_ids,
    simulated=(),
    simulation_cutoff=None,
    active_prices=None,
    valid_references=None,
    price_histories=None,
):
    """Build shop facts from one seller's observed and settled rows."""
    grouped = group_business_history(
        observed, owner_id=owner_id, cutoff=cutoff,
        simulated=simulated, simulation_cutoff=simulation_cutoff,
    )
    require(bool(grouped), "Seller has no past observations")
    requested = set(requested_item_ids)
    require(len(requested) == len(requested_item_ids) and requested <= grouped.keys(),
            "Requested price product is not owned and previously observed")

    products = []
    first = None
    for item, events in grouped.items():
        clock, index, first_row = events[0]
        require(clock == 0, "Settled feedback requires a previously observed owned product")
        if first is None or (index, item) < first[:2]:
            first = (index, item, first_row["seller_name"])
        history = price_histories[item] if price_histories is not None else ObservedPriceHistory(owner_id, item)
        require(history.seller == owner_id and history.item == item, "Foreign cached price history")
        settled = None
        latest = first_row
        for clock, index, row in events:
            if clock == 0:
                latest = row
                if price_histories is None:
                    history.append(row)
            else:
                if settled is None:
                    settled = ObservedPriceHistory(owner_id, item)
                settled.append({**row, "time_index": index})

        product = {
            **{k: latest[k] for k in PROFILE_FIELDS},
            "last_observed_price_raw": latest["price_raw"],
            "price_status": latest["price_status"],
            "last_observed_time_index": latest["time_index"],
            "observed_price_history": history.view(cutoff),
        }
        if settled is not None:
            product["settled_price_history"] = dict(settled.view(simulation_cutoff), clock="simulation_settled_step")
        if not has_observed_price(latest):
            reference = (valid_references.get(item) if valid_references is not None else next(
                (price_point(row) for clock, _, row in reversed(events)
                 if clock == 0 and has_observed_price(row)), None))
            if reference is not None:
                product["last_valid_reference"] = dict(reference)
        products.append(product)
    memory, coverage = recent_business_history(grouped, observations_per_item)
    public_pages = list(public_pages)
    for page in public_pages:
        require(page["history_cutoff_exclusive"] == cutoff, "Public page cutoff mismatch")
        for r in page["candidates"]:
            require(set(r) == set(PUBLIC_FIELDS), "Nonpublic competitor fields")
            require(
                r["seller_id"] != owner_id and r["reference_price_time_index"] < cutoff,
                "Unauthorized or future public reference",
            )
    state = {
        "profile": {"seller_id": owner_id, "seller_name": first[2]},
        "history_cutoff_exclusive": cutoff,
        "owned_products": products,
        "report_window": history_window(cutoff, observations_per_item),
        "short_term_memory": memory,
        "long_term_memory": [],
        "public_product_pages": deepcopy(public_pages),
        "requested_price_item_ids": sorted(requested_item_ids),
        "observed_price_history_semantics": PRICE_HISTORY_POLICY['semantics'],
    }
    if simulation_cutoff is not None:
        state['simulation_cutoff_exclusive']=simulation_cutoff
    if active_prices is not None:
        require(simulation_cutoff is not None, "Active prices require a simulation cutoff")
        require(
            set(active_prices) == {str(item) for item in grouped},
            "Active-price ownership mismatch",
        )
        state["simulation_state"] = {
            "through_step_inclusive": simulation_cutoff - 1,
            "active_prices": deepcopy(active_prices),
        }
    return state, coverage


COUNTS = ("exposure_records", "click_records", "purchase_records")


def attach_retrieved_products(context, lookup):
    """Keep exactly the extra descriptions needed to interpret selected shop history."""
    details = {p["item_id"] for p in context["product_details"]}
    hits = context["retrieved_context"]["history"]["segments"]
    for item in sorted({s["item_id"] for s in hits} - details):
        product = lookup(item)
        require(product["item_id"] == item and product["seller_id"] == context["profile"]["seller_id"],
                "Foreign retrieved product")
        context["product_details"].append({k: deepcopy(product[k]) for k in PROFILE_FIELDS})


def overview_context(full, detail_item_ids):
    """Project one factual state into SFT or runtime context without selecting an action."""
    require("history_presentation" not in full, "Shop context was already projected")
    owned = {p["item_id"] for p in full["owned_products"]}
    require(len(owned) == len(full["owned_products"]), "Duplicate owned product")
    selected = set(detail_item_ids)
    require(
        len(selected) == len(detail_item_ids) and selected <= owned,
        "Details must identify distinct owned products",
    )
    records = defaultdict(lambda: {"observed": [], "simulation": []})
    memory = []
    for event in full["short_term_memory"]:
        require(len(event["source_ids"]) == len(event["content"]), "Event/source alignment differs")
        clock = "observed" if "time_index" in event else "simulation"
        rows, sources = [], []
        for row, source in zip(event["content"], event["source_ids"]):
            item = row["item_id"]
            require(item in owned, "Foreign record in shop memory")
            records[item][clock].append(row)
            if item in selected:
                rows.append(deepcopy(row))
                sources.append(source)
        if rows:
            retained = {k: deepcopy(v) for k, v in event.items() if k not in ("content", "source_ids")}
            retained.update(content=rows, source_ids=sources)
            memory.append(retained)

    result = {k: memory if k == "short_term_memory" else deepcopy(v) for k, v in full.items()}
    simulated = any(value["simulation"] for value in records.values())
    for product in result["owned_products"]:
        for clock in ("observed", "simulation"):
            if clock == "simulation" and not simulated:
                continue
            window = records[product["item_id"]][clock]
            field = "time_index" if clock == "observed" else "settled_step"
            product[f"{clock}_window_start"] = min((r[field] for r in window), default=None)
            product[f"{clock}_window_end"] = max((r[field] for r in window), default=None)
            product[f"{clock}_points"] = len(window)
            for name in COUNTS:
                # An empty observed window is unknown, never inferred to be zero.
                product[f"{clock}_{name}"] = sum(r[name] for r in window) if window else None

    # Store-wide statistics remain visible; full attributes are available on demand.
    result["product_details"] = [
        {k: p[k] for k in PROFILE_FIELDS}
        for p in result["owned_products"]
        if p["item_id"] in selected
    ]
    for product in result["owned_products"]:
        for field in PROFILE_FIELDS:
            if field not in ("item_id", "item_title"):
                product.pop(field)

    result["history_presentation"] = {
        "overview": SHOP_OVERVIEW_NOTE,
        "business_counts": BUSINESS_NOTE,
        "detail_item_ids": sorted(selected),
    }
    return result


SHARED_FIELDS = frozenset((
    "profile", "history_cutoff_exclusive", "report_window", "observed_price_history_semantics",
    "simulation_cutoff_exclusive", "decision_time", "price_scope",
    "shop_scope", "observed_shop_period", "business_report", "retrieved_context", "long_term_memory",
))


class PricingSnapshot:
    """Borrow one overview for a review or SFT cohort. Keep it unchanged until all views are built."""

    def __init__(self, overview):
        self.overview = overview
        self.products = {product["item_id"]: product for product in overview["owned_products"]}
        require(len(self.products) == len(overview["owned_products"]), "Duplicate owned product")
        presentation = overview["history_presentation"]
        self.detail_order = {item: index for index, item in enumerate(presentation["detail_item_ids"])}
        self.presentation = {key: value for key, value in presentation.items() if key != "detail_item_ids"}
        self.details = {product["item_id"]: (index, product)
                        for index, product in enumerate(overview.get("product_details", []))}
        self.events = defaultdict(list)
        for index, event in enumerate(overview["short_term_memory"]):
            if "source_ids" in event:
                require(len(event["source_ids"]) == len(event["content"]), "Event/source alignment differs")
            for position, row in enumerate(event["content"]):
                self.events[row["item_id"]].append((index, position))

    def view(self, item_ids):
        ids = list(item_ids)
        requested = set(ids)
        require(ids and len(ids) == len(requested), "Invalid pricing targets")
        require(requested <= self.products.keys(), "Foreign pricing target")
        require(requested <= self.detail_order.keys(), "Pricing target lacks prepared recent detail")
        ids.sort()

        result = {key: deepcopy(value) for key, value in self.overview.items() if key in SHARED_FIELDS}
        if "public_product_pages" in self.overview:
            result["public_product_pages"] = [
                deepcopy(page) for page in self.overview["public_product_pages"]
                if page.get("item_id") in requested
            ]
        result["owned_products"] = [deepcopy(self.products[item]) for item in ids]
        references = {segment["item_id"] for segment in
                      result.get("retrieved_context", {}).get("history", {}).get("segments", [])}
        described = sorted(self.details[item] for item in requested | references if item in self.details)
        result["product_details"] = [deepcopy(product) for _, product in described]

        positions = defaultdict(list)
        for item in ids:
            for index, position in self.events.get(item, ()):
                positions[index].append(position)
        result["short_term_memory"] = []
        for index in sorted(positions):
            event = self.overview["short_term_memory"][index]
            chosen = sorted(positions[index])
            entry = deepcopy({key: value for key, value in event.items()
                              if key not in ("content", "source_ids")})
            entry["content"] = [deepcopy(event["content"][i]) for i in chosen]
            if "source_ids" in event:
                entry["source_ids"] = [deepcopy(event["source_ids"][i]) for i in chosen]
            result["short_term_memory"].append(entry)

        if "simulation_state" in self.overview:
            state = self.overview["simulation_state"]
            result["simulation_state"] = deepcopy({key: value for key, value in state.items()
                                                   if key != "active_prices"})
            result["simulation_state"]["active_prices"] = {
                str(item): deepcopy(state["active_prices"][str(item)]) for item in ids
            }
        result["history_presentation"] = {
            **deepcopy(self.presentation),
            "detail_item_ids": sorted(ids, key=self.detail_order.__getitem__),
        }
        result["requested_price_item_ids"] = ids
        return result


def target_context(overview, item_ids):
    """Select targets once when the caller has no other views to build."""
    return PricingSnapshot(overview).view(item_ids)
