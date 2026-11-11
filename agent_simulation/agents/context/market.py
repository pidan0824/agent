"""Time-qualified market summaries with explicit comparable-product coverage."""

from collections import defaultdict
from copy import deepcopy

from agent_simulation.data.market.statistics import COUNTS, distribution, market_position, public_price_stats
from agent_simulation.data.market.quotes import iter_decision_positions, shop_period, RECENT_MARKET_SPAN
from agent_simulation.data.market.products import CATEGORY_FIELDS, complete_category

from agent_simulation.agents.retrieval.history import DEFAULT_HISTORY_ENCODER
from agent_simulation.agents.retrieval.competitors import competitor_examples

SELLER_MARKET_NOTE = (
    "Ordinary seller view: comparable offers are from other shops in the same three-level category. "
    "Public peer evidence includes observed prices and recorded purchases (events, not units). "
    "Peer exposures, clicks, CTR and purchase rates are not disclosed or inferable as rates here. "
    "Traffic metrics marked own_value refer only to this shop. Missing is unknown. "
    "Default observed prices and purchase records share the recent window. Older quotes and simulation prices are separate. "
    "Price-band purchases use the recorded display price and a shared historical window; "
    "Sparse purchase counts are insufficient for stable comparisons. Competitor examples are a bounded relevance shortlist, not the whole market. Counts do not establish price effects."
)


SELLER_DECISION_MARKET_NOTE = (
    "Recent observed prices use each product's latest quote in the stated window, from other shops "
    "in the same three-level category. Products/shops and price quantiles cover valid quotes; "
    "known_products also includes unknown prices. Each product has equal weight. "
    "Own percentile is a 0..1 midrank of the latest observed own quote, not the simulation price. "
    "Same-brand summaries are subsets, not additional observations or proof of equivalent specifications. "
    "Largest shop product share describes listed products, not sales or market share. "
    "Competitor examples may include older quotes. Simulation prices are active at the decision "
    "and simulation purchases count settled request-item events."
)


def quote_comparison(market):
    """Describe existing quote statistics without querying or changing the source."""
    recent = (market or {}).get("recent", {})
    own = recent.get("own_quote", {})
    price = own.get("price_raw")
    if price is None:
        return ""
    start = (market or {}).get("window", {}).get("start_inclusive")
    observed = own.get("observation_index")
    basis = ("Latest observed quote before the market window" if start is not None and observed is not None and observed < start
             else "Latest observed quote")
    parts = [f"{basis}: {price:.15g}"]
    low, high = recent.get("p25_price"), recent.get("p75_price")
    where = {"below_p25": "below", "above_p75": "above", "p25_to_p75": "within"}.get(own.get("price_band"))
    if where is not None and low is not None and high is not None:
        parts.append(f"It is {where} the middle half of category quotes [{low:.15g}, {high:.15g}]")
    brand = recent.get("same_brand", {})
    median = brand.get("median")
    if median is not None and median > 0:
        change = 100 * (price / median - 1)
        brand_name = brand.get("brand_name")
        brand_label = f"the {brand_name} median" if brand_name else "the same-brand median"
        median_label = f"{brand_label} ({median:.15g})"
        if change == 0:
            parts.append(f"It equals {median_label}")
        else:
            direction = "below" if change < 0 else "above"
            parts.append(f"It is {abs(change):.2f}% {direction} {median_label}")
    return ". ".join(parts) + "."


def market_comparisons(market):
    """Describe the supplied category quote statistics."""
    recent = (market or {}).get("recent", {})
    views = {}
    low, high = recent.get("p25_price"), recent.get("p75_price")
    maximum = recent.get("max_price")
    if recent.get("products", 0) >= 2 and low is not None and high is not None:
        text = f"The middle half of category quotes ranges from {low:.15g} to {high:.15g}"
        if maximum is not None and high > 0:
            text += f". The highest quote ({maximum:.15g}) is {maximum / high:.2f} times the 75th percentile"
        views["middle_and_upper_end"] = text + "."
    brand_median = recent.get("same_brand", {}).get("median")
    category_median = recent.get("median_price")
    if brand_median is not None and category_median is not None and category_median > 0:
        views["brand_and_category"] = (
            f"The same-brand median ({brand_median:.15g}) is "
            f"{brand_median / category_median:.2f} times the category median ({category_median:.15g}).")
    known, valid = recent.get("known_products"), recent.get("products")
    if known and valid is not None:
        views["quote_coverage"] = (
            f"{valid} of {known} recently observed products in the category have valid quotes. "
            "Price statistics cover those valid quotes.")
    share = recent.get("largest_shop_product_share")
    if share is not None:
        views["shop_offer_concentration"] = (
            f"The shop with the most quoted products accounts for {share:.1%} of products with valid quotes. "
            "This share measures product count.")
    return views


def market_analysis(market):
    """Interpret every available market statistic, independently of competitor selection."""
    analysis = {"source": "derived_analysis"}
    views = market_comparisons(market)
    if views:
        analysis["market_overview"] = views
    position = quote_comparison(market)
    if position:
        analysis["price_position"] = position
    return analysis


def context_view(position):
    """Filter full market_position results for public queries; decision summaries bypass this view."""
    result = {k: deepcopy(position[k]) for k in (
        "item_id", "status", "clock", "window", "scope", "category_ids", "peer_products",
        "recent_price_scope", "expanded_price_sample", "historical_reference", "price_time_basis",
        "peer_price_observation_range", "current_reference", "price_raw", "purchase_records"
    ) if k in position}
    if "coverage" not in position:
        if "simulation" in position:
            result["simulation"] = context_view(position["simulation"])
        return result
    result["coverage"] = {k: position["coverage"][k] for k in (
        "known_peer_products", "usable_reference_prices", "comparable_products",
        "comparable_fraction", "own_comparable",
    )}
    for metric in ("exposure_records", "click_records"):
        result[metric] = {"own_value": position[metric]["own_value"]}
    cumulative = position["cumulative"]
    result["cumulative"] = {k: deepcopy(cumulative[k]) for k in (
        "clock", "window", "purchase_records") if k in cumulative}
    for metric in ("exposure_records", "click_records"):
        result["cumulative"][metric] = {"own_value": cumulative[metric]["own_value"]}
    result["price_raw"] = public_price_stats(result["price_raw"])
    for metric, numerator in (("ctr", "click_records"), ("cvr", "purchase_records")):
        result[metric] = {
            "own_value": position[metric]["own_value"],
            "numerator": position[numerator]["own_value"],
            "denominator": position["exposure_records"]["own_value"],
            "numerator_field": numerator,
            "denominator_field": "exposure_records",
        }
    if "simulation" in position:
        result["simulation"] = context_view(position["simulation"])
    return result


def simulation_market_cache(state):
    """Index one frozen market snapshot for all target contexts in this review."""
    feedback, categories = defaultdict(list), defaultdict(list)
    for event in state["feedback"]:
        feedback[event["item_id"]].append(event)
    for item, product in state["products"].items():
        categories[tuple(product[k] for k in CATEGORY_FIELDS)].append(
            (int(item), product["seller_id"]))
    return {"feedback": feedback, "categories": categories, "summaries": {}}


def simulation_position(state, item_id, *, cache=None, decision_only=False):
    """Read current peer evidence from one snapshot, with a compact view for pricing."""
    focal = state["products"][str(item_id)]
    if not complete_category(focal):
        return {"status": "missing_category"}
    clock = state.get("clock")
    if clock:
        start = (
            clock["last_decisions"]
            .get(f"seller:{focal['seller_id']}", {})
            .get("time", clock["start"])
        )
        end, field, name = clock["time"], "search_time", "simulation_logical_time"
    else:
        # Without Tick scheduling, use the available settled interval and label it explicitly.
        start, end, field, name = (
            0,
            state["simulation_step"] + 1,
            "settled_step",
            "simulation_settled_step",
        )

    window = {
        "start_inclusive": start,
        "end_exclusive": end,
        "selection": "since_previous_seller_decision" if clock else "all_settled",
    }
    price_basis = {
        "clock": name,
        "selection": "active_price_at_decision",
        "as_of": clock["time"] if clock else state["simulation_step"],
    }
    if cache is None:
        cache = simulation_market_cache(state)
    category = tuple(focal[k] for k in CATEGORY_FIELDS)
    seller = focal["seller_id"]

    if decision_only:
        key = (seller, category, start, end, field)
        summaries = cache["summaries"]
        if key not in summaries:
            peers = [item for item, owner in cache["categories"].get(category, ()) if owner != seller]
            prices, purchases = [], []
            for peer in peers:
                price = state["active_prices"][str(peer)]["price_raw"]
                # Match the priced-peer cohort used by full market_position comparisons.
                if price is not None:
                    prices.append(price)
                    purchases.append(sum(
                        event["purchase_records"] for event in cache["feedback"].get(peer, ())
                        if start <= event[field] < end
                    ))
            summaries[key] = {
                "active_products": len(peers),
                "priced_products": len(prices),
                "price_as_of": price_basis,
                "median_active_price": distribution(prices, None)["median"],
                "purchase_window": window,
                "purchase_scope": "currently_priced_peer_products",
                "median_purchase_records": distribution(purchases, None)["median"],
            }
        return deepcopy(summaries[key])

    def product(item):
        events = cache["feedback"].get(item, ())
        recent = [event for event in events if start <= event[field] < end]
        active = state["active_prices"][str(item)]
        return {
            "price_raw": active["price_raw"],
            "active_price_basis": dict(active),
            **{k: sum(event[k] for event in recent) for k in COUNTS},
            **{"cumulative_" + k: sum(event[k] for event in events) for k in COUNTS},
        }

    peers = [product(item) for item, owner in cache["categories"].get(category, ()) if owner != seller]
    result = {
        "scope": "Registered products and currently active prices. Recent counts include "
                 + ("settled request-item records whose exposure occurred within the window."
                    if clock else "request-item records settled within the window."),
        **market_position(product(item_id), peers, window=window, clock=name),
    }
    result["price_time_basis"] = price_basis
    result["cumulative"]["window"] = {
        "start_inclusive": clock["start"] if clock else 0,
        "end_exclusive": end,
        "selection": "since_episode_initialization",
    }
    return result


def decision_market(position):
    """Keep the historical fields used in the default pricing decision."""
    if "recent_price_scope" not in position:
        return {k: position[k] for k in ("status",) if k in position}
    scope = position["recent_price_scope"]
    return {
        "category_ids": position["category_ids"],
        "window": {k: position["window"][k] for k in ("start_inclusive", "end_exclusive")},
        "recent": {
            **deepcopy(scope),
            "median_price": position["price_raw"]["median"],
            **{key + "_price": value for key, value in position["price_raw"].items() if key != "median"},
        },
    }


def load_market_context(context, data, *, simulation=None, simulation_cache=None,
                        history_embedder=DEFAULT_HISTORY_ENCODER, product_lookup=None):
    """Attach the historical pricing panel and optional simulation evidence."""
    cutoff = context["history_cutoff_exclusive"]
    seller = context["profile"]["seller_id"]
    items = [p["item_id"] for p in context["owned_products"]]
    products = {p["item_id"]: p for p in context["owned_products"]}
    context["observed_shop_period"] = shop_period(data, seller, cutoff)
    context["market_semantics"] = SELLER_DECISION_MARKET_NOTE
    profiles = [product_lookup(item) for item in items] if product_lookup is not None else None
    if simulation is not None and simulation_cache is None:
        simulation_cache = simulation_market_cache(simulation)
    start = max(0, cutoff - RECENT_MARKET_SPAN)
    for item, position in iter_decision_positions(data, seller, items, cutoff, profiles=profiles):
        visible = decision_market(position)
        if simulation is not None:
            visible["simulation"] = simulation_position(
                simulation, item, cache=simulation_cache, decision_only=True,
            )
        visible["analysis"] = market_analysis(visible)
        if position.get("category_ids"):
            visible["competitor_examples"] = competitor_examples(
                data, seller, item, cutoff, start, embedder=history_embedder, simulation=simulation,
                own_quote={k: products[item][k] for k in ("last_observed_price_raw",
                    "last_observed_time_index", "price_status") if k in products[item]},
            )
        products[item]["market_position"] = visible
    return context


def public_prices(result, name, *, active_prices=None):
    """Add posted prices to already-public catalog rows, omitting active-price action metadata."""
    key = "candidates" if name == "discover_public_products" else "products"
    rows = []
    for row in result[key]:
        if active_prices is None:
            price = row.get("simulation_current_price")
        else:
            profile = row if key == "candidates" else row["profile"]
            price = active_prices.get(str(profile["item_id"]))
        rows.append({**row, "simulation_current_price": (
            {"price_raw": price["price_raw"]} if price is not None else None
        )})
    return {**result, key: rows}
