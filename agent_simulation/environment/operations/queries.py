"""Read authorized shop, market and customer records for a bound environment tool."""

from copy import deepcopy
from agent_simulation.agents.context.market import SELLER_MARKET_NOTE, context_view, simulation_position
from agent_simulation.agents.context.shopping import shopping_history_event
from agent_simulation.data.history_records import select_business_history
from agent_simulation.data.catalog import history_coverage
from agent_simulation.data.market.statistics import QUANTITY_ASSUMPTION, totals
from agent_simulation.environment.operations.reporting import cumulative_series, simulation_statistics, windowed_activity_timeline


def query_shop_products(state, owned, args):
    selected = [i for i in sorted(owned) if i > args["after_item_id"]]
    page = selected[: args["limit"]]
    return {
        "products": [
            {
                **deepcopy(state["products"][str(i)]),
                "simulation_current_price": deepcopy(state["active_prices"][str(i)]),
            }
            for i in page
        ],
        "shop_product_count": len(owned),
        "has_more": len(selected) > len(page),
        "next_after_item_id": page[-1] if len(selected) > len(page) else None,
    }


def query_market_position(runtime, state, actor, owned, args):
    if args["item_id"] not in owned:
        raise PermissionError("Market position must refer to an owned market product")
    result = runtime.data.market_position(actor, **args)
    result["simulation"] = simulation_position(state, args["item_id"])
    return {**context_view(result), "semantics": SELLER_MARKET_NOTE}


def validate_observed_window(args, cutoff):
    """Reject invalid observed bounds before any source records are queried."""
    before = args["before_time_index"]
    start = args.get("start_time_index", 0)
    if not 0 <= start <= before <= cutoff:
        raise ValueError(
            f"Observed bounds must satisfy 0 <= start_time_index <= before_time_index <= {cutoff}. "
            "before_time_index is exclusive and remains frozen during simulation. "
            "Do not substitute a logical Tick or simulation event index."
        )


def query_business(state, data, actor, args, *, ranking_connection=None):
    validate_observed_window(args, data.cutoff)
    before = args.get("before_simulation_step", state["simulation_step"] + 1)
    start = args.get("start_simulation_step", 0)
    if not 0 <= start <= before <= state["simulation_step"] + 1:
        raise ValueError(
            f"Invalid simulation cutoff: bounds must satisfy 0 <= start_simulation_step <= "
            f"before_simulation_step <= {state['simulation_step'] + 1}. "
            "Omit these optional fields for all settled feedback; do not use observed time_index."
        )
    logical = {"start_logical_time", "before_logical_time"} & args.keys()
    event_filters = {"start_simulation_step", "before_simulation_step"} & args.keys()
    if logical and "clock" not in state:
        raise ValueError("Logical-time queries require a Tick runtime; omit logical bounds here")
    if logical and event_filters:
        raise ValueError("Choose logical-time or event-step filters, not both")
    if "clock" in state:
        start_time = args.get("start_logical_time", state["clock"]["start"])
        end_time = args.get("before_logical_time", state["clock"]["time"])
        if not state["clock"]["start"] <= start_time <= end_time <= state["clock"]["time"]:
            raise ValueError(
                f"Logical bounds must lie between {state['clock']['start']} and {state['clock']['time']}, "
                "on completed Tick boundaries."
            )
        if not all(
            (t - state["clock"]["start"]) % state["clock"]["tick_size"] == 0
            for t in (start_time, end_time)
        ):
            raise ValueError("Logical-time bounds must be Tick boundaries")

    def in_logical_window(event, field):
        return not logical or start_time <= event[field] < end_time

    result = data.business(
        actor,
        **{
            k: v
            for k, v in args.items()
            if k
            not in (
                "before_simulation_step",
                "start_simulation_step",
                "start_logical_time",
                "before_logical_time",
            )
        },
        ranking_connection=ranking_connection,
    )
    series_ids = {r["item_id"] for r in result["series"]}
    result["products"] = [
        data.product(i, args["before_time_index"])
        for i in args["item_ids"]
        if i in series_ids
    ]
    simulated_ids = [i for i in args["item_ids"] if str(i) in state["products"]]
    simulated_items = set(simulated_ids)
    feedback = [f for f in state["feedback"] if f["item_id"] in simulated_items]
    matching = [
        f
        for f in feedback
        if start <= f["settled_step"] < before and in_logical_window(f, "settled_time")
    ]
    selected, _ = select_business_history(
        [],
        owner_id=actor,
        cutoff=data.cutoff,
        observations_per_item=args["limit"],
        simulated=matching,
        simulation_cutoff=before,
    )
    keys = {(r["item_id"], r["settled_step"]) for e in selected for r in e["content"]}
    visible = [f for f in matching if (f["item_id"], f["settled_step"]) in keys]
    matching_by_item = {item: [] for item in simulated_ids}
    visible_by_item = {item: [] for item in simulated_ids}
    for record in matching:
        matching_by_item[record["item_id"]].append(record)
    for record in visible:
        visible_by_item[record["item_id"]].append(record)
    actions_by_item = {item: [] for item in simulated_ids}
    for action in state["price_actions"]:
        item = action["item_id"]
        if (item in actions_by_item and action["seller_id"] == actor
                and start <= action["simulation_step"] < before and in_logical_window(action, "logical_time")):
            actions_by_item[item].append(action)
    returned_actions = {item: records[-args["limit"]:] for item, records in actions_by_item.items()}
    # The historical cutoff applies only to observed data; simulation uses its own clock.
    result["simulation"] = {
        "outside_market_item_ids": [i for i in args["item_ids"] if str(i) not in state["products"]],
        "quantity_assumption": QUANTITY_ASSUMPTION,
        "cumulative_scope": "From simulation initialization to each returned step, including earlier non-returned records; observed history is separate.",
        "cumulative_series": cumulative_series(
            [f for f in feedback if f["settled_step"] < before], keys
        ),
        "end_exclusive": before,
        "requested_start_inclusive": start,
        "limit_scope": "per_item",
        "settled_series": visible,
        "has_earlier_records": len(matching) > len(visible),
        "summary": [{"item_id": item, **totals(records)} for item, records in visible_by_item.items()],
        "executed_prices": sorted(
            (action for records in returned_actions.values() for action in records),
            key=lambda a: (a["simulation_step"], a["item_id"]),
        ),
        "settled_coverage": [
            {
                "item_id": i,
                **history_coverage(
                    matching_by_item[i], visible_by_item[i],
                    "settled_step",
                    "next_before_simulation_step",
                ),
            }
            for i in simulated_ids
        ],
        "price_action_coverage": [
            {
                "item_id": i,
                **history_coverage(
                    actions_by_item[i], returned_actions[i],
                    "simulation_step",
                    "next_before_simulation_step",
                ),
            }
            for i in simulated_ids
        ],
    }
    if args.get("metrics"):
        result["simulation"]["statistics"] = simulation_statistics(
            matching_by_item, args["metrics"], start, before
        )
    if logical:
        result["simulation"]["logical_window"] = {
            "start_inclusive": start_time,
            "end_exclusive": end_time,
        }
        if "statistics" in result["simulation"]:
            result["simulation"]["statistics"].update(
                clock="simulation_logical_time",
                start_inclusive=start_time,
                end_exclusive=end_time,
                scope="All settled item records in the requested logical-time interval; detail limit does not affect totals.",
            )
    if "clock" in state and not event_filters:
        result["simulation"]["activity_timeline"] = windowed_activity_timeline(
            state, actor, simulated_ids, start_time, end_time, args["limit"]
        )
    return result


def query_customer_history(state, data, catalog, actor, args):
    customer = state["customers"][str(actor)]
    before = args.get("before_simulation_step", state["simulation_step"] + 1)
    if not 0 <= before <= state["simulation_step"] + 1:
        raise ValueError(
            f"before_simulation_step must be between 0 and {state['simulation_step'] + 1}. "
            "This is a simulation event-step bound, not an observed time index."
        )
    if not 0 <= args["before_time_index"] <= data.cutoff:
        raise ValueError(
            f"History query exceeds cutoff: before_time_index must be between 0 and {data.cutoff}. "
            "Observed history stays frozen at initialization; simulated history is returned "
            "separately and can be filtered with before_simulation_step."
        )
    if catalog is not None:
        observed = (
            catalog.customer_history(actor, args["before_time_index"], args["limit"], "test")[
                "short_term_memory"
            ]["search_history"]
            if args["before_time_index"]
            else []
        )
    else:
        # Initialization validates ownership/cutoff and stores these seeds in time order.
        observed = deepcopy([
            e for e in customer["observed_history"] if e["time_index"] < args["before_time_index"]
        ][-args["limit"]:])
    simulated = [e for e in customer["simulation_history"] if e["settled_step"] < before]
    return {
        "observed_history": observed,
        "simulation_history": [shopping_history_event(e) for e in simulated[-args["limit"] :]],
        "through_simulation_step_inclusive": before - 1,
    }
