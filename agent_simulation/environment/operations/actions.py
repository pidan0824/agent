"""Market state transitions; MarketEnvironment validates, commits and audits each call."""

from copy import deepcopy
from agent_simulation.platform.search import search_market
from agent_simulation.data.common import require
from agent_simulation.agents.actions import validate_response, price_schema
from agent_simulation.agents.schemas import validate_arguments


_REVIEW_PARAMETERS = price_schema(allow_unknown=True)


def require_open_market(state):
    require(state.get("clock", {}).get("status") not in ("failed", "complete"),
            "Episode is closed to further actions")


def search_products(state, actor, args, search_index, *, search_pipeline=None):
    require_open_market(state)
    require(
        str(actor) not in state["pending"],
        "Submit the pending response before another search",
    )
    ids, search_details = search_market(state, actor, args["query"], search_index, search_pipeline)
    state["simulation_step"] += 1
    displayed = []
    for i in ids:
        product = deepcopy(state["products"][str(i)])
        product["price_raw"] = state["active_prices"][str(i)]["price_raw"]
        if product["price_raw"] is not None:
            product["price_status"] = "observed"
        product["price_provenance"] = deepcopy(state["active_prices"][str(i)])
        displayed.append(product)
    result = {
        "request_id": state["simulation_step"],
        "simulation_step": state["simulation_step"],
        "query": args["query"],
        "displayed_items": displayed,
        **search_details,
        "catalog_size": len(state["products"]),
    }
    state["pending"][str(actor)] = deepcopy(result)
    if "clock" in state:
        result["logical_time"] = state["clock"]["time"]
        state["pending"][str(actor)]["logical_time"] = result["logical_time"]
    return result


def submit_prices(state, actor, owned, args):
    require_open_market(state)
    prices = args["prices"]
    require(len({p["item_id"] for p in prices}) == len(prices), "Duplicate price submission")
    if not {p["item_id"] for p in prices} <= set(owned):
        raise PermissionError("Cannot change another seller price")
    prices = [p for p in prices if p["price_raw"] != state["active_prices"][str(p["item_id"])]["price_raw"]]
    if prices:
        state["simulation_step"] += 1
    for p in prices:
        action = {
            **p,
            "seller_id": actor,
            "simulation_step": state["simulation_step"],
            "previous_price_raw": state["active_prices"][str(p["item_id"])]["price_raw"],
            "source": "simulation:submitted_price",
        }
        state["price_actions"].append(action)
        if "clock" in state:
            action["logical_time"] = state["clock"]["time"]
        state["active_prices"][str(p["item_id"])] = {
            "price_raw": p["price_raw"],
            "source": action["source"],
            "simulation_step": state["simulation_step"],
        }
        if "clock" in state:
            state["active_prices"][str(p["item_id"])]["logical_time"] = action["logical_time"]
    return {
        "simulation_step": state["simulation_step"],
        "updated": prices,
        "status": "applied" if prices else "no_change",
        "effective_from_simulation_step": state["simulation_step"] if prices else None,
        "applies_to": "Subsequent product searches; already displayed price snapshots are unchanged.",
    }


def exit_customer(state, actor):
    """Record voluntary Episode departure without inventing a shopping event."""
    require_open_market(state)
    require("clock" in state, "Exit requires an Episode clock")
    require(not state["customers"][str(actor)].get("exit"), "Customer already exited")
    require(str(actor) not in state["pending"], "Settle the displayed list before exit")
    event = {"tick": state["clock"]["tick"], "logical_time": state["clock"]["time"]}
    state["customers"][str(actor)]["exit"] = event
    return {"action": "exit", **event}


def submit_response(state, actor, args):
    require_open_market(state)
    require(str(actor) in state["pending"], "No pending displayed list")
    request = state["pending"][str(actor)]
    response = validate_response(args, [p["item_id"] for p in request["displayed_items"]])
    state["simulation_step"] += 1
    event = {
        "request_id": request["request_id"],
        "search_step": request["simulation_step"],
        "settled_step": state["simulation_step"],
        "query": request["query"],
        "source": "simulation:customer_response",
        **{
            key: [p for p in request["displayed_items"] if p["item_id"] in response[ids]]
            for key, ids in (
                ("clicked_items", "clicked_item_ids"),
                ("purchased_items", "purchased_item_ids"),
            )
        },
    }
    state["customers"][str(actor)]["simulation_history"].append(event)
    if "clock" in state:
        event.update(search_time=request["logical_time"], settled_time=state["clock"]["time"])
    for p in request["displayed_items"]:
        state["feedback"].append(
            {
                "item_id": p["item_id"],
                "seller_id": p["seller_id"],
                "request_id": request["request_id"],
                "search_step": request["simulation_step"],
                "settled_step": state["simulation_step"],
                "price_raw": p["price_raw"],
                "price_status": "observed" if p["price_raw"] is not None else p["price_status"],
                "exposure_records": 1,
                "click_records": int(p["item_id"] in response["clicked_item_ids"]),
                "purchase_records": int(p["item_id"] in response["purchased_item_ids"]),
                "source": "simulation:settled_response",
            }
        )
        if "clock" in state:
            state["feedback"][-1].update(
                search_time=request["logical_time"], settled_time=state["clock"]["time"]
            )
    del state["pending"][str(actor)]
    return {
        "settled_step": state["simulation_step"],
        "request_id": request["request_id"],
        **response,
    }


class SellerReview:
    def __init__(self, runtime, actor, snapshot):
        self.runtime, self.actor = runtime, actor
        self.revision = snapshot.revision
        self.owned = sorted(
            int(i) for i, p in snapshot.state["products"].items() if p["seller_id"] == actor
        )
        require(bool(self.owned), "Seller is not registered")
        self.original_prices = {i: snapshot.state["active_prices"][str(i)]["price_raw"] for i in self.owned}
        self.decisions = {}
        self.committed = False

    def stage(self, args, batch_ids):
        """Validate and collect this batch's prices without changing the market."""
        require(not self.committed, "Review was already committed")
        validate_arguments(args, _REVIEW_PARAMETERS)
        ids = [p["item_id"] for p in args["prices"]]
        if not set(ids) <= set(batch_ids):
            raise PermissionError("Price decision is outside the current product batch")
        require(
            len(ids) == len(set(ids)) and set(ids) == set(batch_ids),
            "Submit an explicit price for every product in the current batch, including unchanged prices",
        )
        require(not set(ids) & self.decisions.keys(), "Product already has a staged decision")
        for quote in args["prices"]:
            if quote["price_raw"] is None:
                require(self.original_prices[quote["item_id"]] is None,
                        "A null price can only keep an unknown active price")
        self.decisions.update({p["item_id"]: deepcopy(p) for p in args["prices"]})
        return {
            "status": "staged",
            "products_reviewed": len(ids),
            "reviewed_total": len(self.decisions),
            "shop_product_count": len(self.owned),
            "prices_effective": False,
        }

    def proposed_prices(self):
        """Return drafts without changing market prices or acknowledging feedback."""
        require(set(self.decisions) == set(self.owned), "Incomplete shop review")
        return [
            deepcopy(self.decisions[i])
            for i in sorted(self.decisions)
            if self.decisions[i]["price_raw"] != self.original_prices[i]
        ]

    def commit(self, run_id):
        require(not self.committed, "Review was already committed")
        prices = self.proposed_prices()
        result = self.runtime.commit_seller_review(self.actor, prices, self.revision,
            dict(reviewed_products=len(self.decisions), kept_products=len(self.owned)-len(prices),
                 review_id=run_id))
        self.committed = True
        return result
