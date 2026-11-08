"""Validate and dispatch programmatic market operations."""

from copy import deepcopy
from functools import lru_cache
from agent_simulation.data.common import require
from agent_simulation.environment.operations import actions as market_actions, queries as history_queries
from agent_simulation.environment.operations import reporting as seller_reports
from agent_simulation.agents.context.market import public_prices
from agent_simulation.agents.schemas import ID, IDS, function, validate_arguments
from agent_simulation.data.market.statistics import METRICS
from agent_simulation.agents.actions import SUBMIT_PRICES, SEARCH_PRODUCTS, SUBMIT_RESPONSE

WINDOW = {
    "before_time_index": {
        "type": "integer",
        "minimum": 0,
    },
    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
}

MEMORY_TOOL = function(
    "query_long_term_memory",
    'Read relevant long-term memories.',
    {"query": {"type": "string", "minLength": 1}},
    ["query"],
)

SIMULATION_BEFORE = {
    "type": "integer",
    "minimum": 1,
}
SIMULATION_START = {
    "type": "integer",
    "minimum": 0,
}

CUSTOMER_QUERY_TOOLS = [
    function(
        "query_customer_history",
        "Read this customer's prior searches and choices.",
        {**WINDOW, "before_simulation_step": SIMULATION_BEFORE},
        ["before_time_index", "limit"],
    ),
    MEMORY_TOOL,
    function(
        "query_displayed_products",
        'Read attributes of currently displayed products.',
        {"item_ids": {**IDS, "minItems": 1, "maxItems": 20}},
        ["item_ids"],
    ),
]

SELLER_QUERY_TOOLS = [
    function(
        "query_shop_products",
        'Read a page of owned products.',
        {
            "after_item_id": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        ["after_item_id", "limit"],
    ),
    function(
        "discover_public_products",
        'Find public products from other sellers in the same category.',
        {
            "focal_item_id": {**ID},
            "before_time_index": WINDOW["before_time_index"],
            "after_item_id": {"type": "integer", "minimum": 0},
            "page_size": {"type": "integer", "minimum": 1, "maximum": 100},
            "query": {"type": "string", "minLength": 1},
            "brand_id": ID,
        },
        ["focal_item_id", "before_time_index"],
    ),
    function(
        "query_business",
        'Read owned-product history and requested window statistics.',
        {
            **WINDOW,
            "item_ids": {**IDS},
            "start_time_index": {
                "type": "integer",
                "minimum": 0,
            },
            "before_simulation_step": SIMULATION_BEFORE,
            "start_simulation_step": SIMULATION_START,
            "metrics": {
                "type": "array",
                "items": {"type": "string", "enum": list(METRICS)},
                "uniqueItems": True,
                "maxItems": len(METRICS),
            },
        },
        ["before_time_index", "limit", "item_ids"],
    ),
    function(
        "query_public_products",
        'Read public product attributes and reference-price histories.',
        {
            **WINDOW,
            "item_ids": {**IDS},
        },
        ["before_time_index", "limit", "item_ids"],
    ),
    MEMORY_TOOL,
    function(
        "query_market_position",
        'Read market price distributions and owned-product metrics.',
        {
            "item_id": ID,
            "before_time_index": {**WINDOW["before_time_index"], "minimum": 1},
            "start_time_index": {"type": "integer", "minimum": 0,
                                 },
        },
        ["item_id", "before_time_index"],
    ),
    function(
        "read_business_report",
        'Read settled shop outcomes and price intervals.',
        {},
        [],
    ),
]


def customer_runtime_tools(*, tick=False):
    """Environment operations for direct execution by the caller."""
    tools = deepcopy([SEARCH_PRODUCTS, SUBMIT_RESPONSE, *CUSTOMER_QUERY_TOOLS])
    if tick:
        tools.append(function("exit_customer", 'Exit the current simulation Episode.', {}, []))
    return tools


def runtime_tools(*, tick=False):
    """Environment operations for direct Seller execution and price settlement."""
    tools = deepcopy([SUBMIT_PRICES, *SELLER_QUERY_TOOLS])
    if tick:
        for schema in tools:
            f = schema["function"]
            if f["name"] == "query_business":
                for name in ("start_logical_time", "before_logical_time"):
                    f["parameters"]["properties"][name] = {
                        "type": "integer", "minimum": 0,
                    }
    return tools


@lru_cache(maxsize=4)
def _parameters(role, tick):
    tools = customer_runtime_tools(tick=tick) if role == "customer" else runtime_tools(tick=tick)
    return {t["function"]["name"]: t["function"]["parameters"] for t in tools}


def validate_bound_tool(role, tick, name, arguments):
    require(role in ("customer", "seller"), "Invalid actor binding")
    parameters = _parameters(role, tick)
    if name not in parameters:
        raise PermissionError("Tool is unavailable to this actor role")
    validate_arguments(arguments, parameters[name])


def dispatch(runtime, state, role, actor, owned, name, args):
    if name == "exit_customer":
        return market_actions.exit_customer(state, actor)
    if name == "query_shop_products":
        return history_queries.query_shop_products(state, owned, args)
    if name == "read_business_report":
        return seller_reports.read(state, actor)
    if name == "query_long_term_memory":
        return runtime.long_term_memory.context(role, actor, args["query"], state=state)
    if name == "query_market_position":
        return history_queries.query_market_position(runtime, state, actor, owned, args)
    if name == "discover_public_products":
        history_queries.validate_observed_window(args, runtime.data.cutoff)
        result = runtime.data.discover(actor, owned, **args)
        return public_prices(result, name, active_prices=state["active_prices"])
    if name == "query_business":
        return history_queries.query_business(
            state,
            runtime.data,
            actor,
            args,
            ranking_connection=(
                runtime.catalog.connection
                if runtime.catalog is not None and args.get("metrics")
                else None
            ),
        )
    if name == "query_public_products":
        history_queries.validate_observed_window(args, runtime.data.cutoff)
        result = runtime.data.public(actor, owned, **args)
        return public_prices(result, name, active_prices=state["active_prices"])
    if name == "query_customer_history":
        return history_queries.query_customer_history(
            state, runtime.data, runtime.catalog, actor, args
        )
    if name == "query_displayed_products":
        from agent_simulation.agents.context.shopping import displayed_details

        require(str(actor) in state["pending"], "No pending product list")
        return displayed_details(state["pending"][str(actor)]["displayed_items"], args["item_ids"])
    if name == "search_products":
        return market_actions.search_products(
            state, actor, args, runtime.search, search_pipeline=runtime.search_pipeline
        )
    if name == "submit_prices":
        return market_actions.submit_prices(state, actor, owned, args)
    require(name == "submit_response", "Unknown tool dispatch")
    return market_actions.submit_response(state, actor, args)
