"""Read a verified Full/Lite catalog through existing history and public-product rules."""


import argparse
import json
import duckdb
import math
from pathlib import Path
from agent_simulation.data.common import Config, require, rows
from agent_simulation.data.history_records import (
    shopping_event,
    customer_history_limit,
    select_business_history,
    seller_history_policy,
)
from agent_simulation.data.market.products import (
    discover_candidates,
    CATEGORY_FIELDS,
    complete_category,
    PROFILE_FIELDS,
    history,
)
from collections import OrderedDict
from agent_simulation.data.market.statistics import (
    BUSINESS_NOTE,
    COUNTS,
    METRICS,
    totals,
    price_evidence,
    statistics,
)

# Customer inputs use NULL for unknown prices; ranking_observations retains the source value.
DISPLAY_PRICE_SQL = "CASE WHEN isfinite(o.price_raw) AND o.price_raw > 0 THEN o.price_raw END"




def history_coverage(records, returned, time_field, cursor_field):
    """Describe one product's selected window and its exclusive continuation cursor."""
    times = [r[time_field] for r in returned]
    earlier = len(records) > len(returned)
    return {
        "matched_points": len(records),
        "returned_points": len(returned),
        "returned_start_inclusive": min(times) if times else None,
        "returned_end_inclusive": max(times) if times else None,
        "has_earlier_records": earlier,
        cursor_field: min(times) if earlier and times else None,
    }


class DataRepository:
    """Query facade: product facts, owned history, public products and market panels."""

    def __init__(self, connection, config, cutoff, *, history_points=None, category_snapshots=None):
        require(type(cutoff) is int and cutoff > 0, "Invalid historical cutoff")
        self.connection, self.config, self.cutoff = connection, config, cutoff
        self.history_points = seller_history_policy()["observations_per_item"] if history_points is None else history_points
        require(type(self.history_points) is int and self.history_points > 0, "Invalid historical window")
        self._histories, self._market_rows, self._market_panels = {}, {}, {}
        self._decision_cache = OrderedDict()
        self.category_snapshots = category_snapshots
        self.competitor_features=category_snapshots.derived if category_snapshots is not None else OrderedDict()

    def decision_cached(self, key, compute):
        """Bounded immutable query results, scoped to this connection and cutoff.

        Callers must project/copy returned values before exposing them to agents.
        Keys include the exact historical boundary; no time-bucket approximation.
        """
        if key not in self._decision_cache:
            value = compute()
            self._decision_cache[key] = value
            if len(self._decision_cache) > 32:
                self._decision_cache.popitem(last=False)
        self._decision_cache.move_to_end(key)
        return self._decision_cache[key]

    # Historical product facts and cutoff validation.
    def records(self, item_id):
        require(type(item_id) is int and item_id > 0, "Invalid product ID")
        if item_id not in self._histories:
            records = history(self.connection, self.config, item_id, self.cutoff)
            require(
                len({r["time_index"] for r in records}) == len(records),
                "Duplicate item/time observation",
            )
            for r in records:
                require(
                    r["item_id"] == item_id
                    and type(r["time_index"]) is int
                    and 0 <= r["time_index"] < self.cutoff,
                    "Invalid historical event",
                )
                require(
                    all(type(r[k]) is int and 0 <= r[k] <= r["exposure_records"] for k in COUNTS),
                    "Invalid or unknown observed outcome count",
                )
                price = r["price_raw"]
                if r["price_status"] == "observed":
                    require(
                        type(price) in (int, float) and math.isfinite(price) and price > 0,
                        "Invalid observed price",
                    )
                else:
                    require(
                        r["price_status"] in ("invalid_price", "conflicting_prices")
                        and price is None,
                        "Unknown price must remain null with an explicit status",
                    )
            self._histories[item_id] = records
        return self._histories[item_id]

    def product(self, item_id, before=None):
        before = self.cutoff if before is None else before
        self.window(before, 1)
        def read():
            visible = [r for r in self.records(item_id) if r["time_index"] < before]
            require(bool(visible), "Product has no observation before cutoff")
            return visible[-1]
        r = self.decision_cached(('product_profile', item_id, before), read)
        return {**{k: r[k] for k in PROFILE_FIELDS},
                "reference_price_raw": r["price_raw"],
                "reference_price_time_index": r["time_index"], "price_status": r["price_status"]}

    def window(self, before, limit):
        require(
            type(before) is int and 0 <= before <= self.cutoff, "Query exceeds historical cutoff"
        )
        require(type(limit) is int and 1 <= limit <= 100, "Invalid time-point limit")

    # Owned product trajectories and derived business statistics.
    def business(self, seller_id, item_ids, before_time_index, limit, start_time_index=0,
                 metrics=(), *, ranking_connection=None):
        self.window(before_time_index, limit)
        require(len(item_ids) == len(set(item_ids)), "Duplicate product IDs")
        require(len(metrics) == len(set(metrics)) and set(metrics) <= set(METRICS), "Unknown or duplicate metric")
        require(
            type(start_time_index) is int and 0 <= start_time_index <= before_time_index,
            "Invalid historical window",
        )
        selected = []
        for item_id in item_ids:
            records = self.records(item_id)
            require(bool(records), "Unknown owned product")
            if any(r["seller_id"] != seller_id for r in records):
                raise PermissionError("Cannot query another seller business history")
            selected.extend(
                r for r in records if start_time_index <= r["time_index"] < before_time_index
            )
        times = sorted({r["time_index"] for r in selected})
        memory, _ = (
            select_business_history(
                selected, owner_id=seller_id, cutoff=before_time_index, observations_per_item=limit
            )
            if selected
            else ([], [])
        )
        series = [r for event in memory for r in event["content"]]
        visible_times = {r["time_index"] for r in series}
        series.sort(key=lambda r: (r["time_index"], r["item_id"]))
        summaries = [
            {
                "item_id": i,
                **{k: sum(r[k] for r in series if r["item_id"] == i) for k in COUNTS},
                "estimated_units_sold": sum(r["purchase_records"] for r in series if r["item_id"] == i),
                "observed_time_points": sum(r["item_id"] == i for r in series),
            }
            for i in sorted({r["item_id"] for r in series})
        ]
        result = {
            "clock": "observed_time_index",
            "end_exclusive": before_time_index,
            "requested_start_inclusive": start_time_index,
            "returned_start_inclusive": min(visible_times) if visible_times else None,
            "matched_time_points": len(times),
            "returned_time_points": len(visible_times),
            "limit_scope": "per_item",
            "has_earlier_records": len(selected) > len(series),
            "item_coverage": [
                {"item_id": i, **history_coverage(
                    [r for r in selected if r["item_id"] == i],
                    [r for r in series if r["item_id"] == i],
                    "time_index", "next_before_time_index",
                )}
                for i in item_ids
            ],
            "series": series,
            "summary_of_returned_series": summaries,
            "semantics": "Observed records only. Gaps are not zero. " + BUSINESS_NOTE,
        }
        if metrics:
            # Query the verified request-item table only when a derived metric is requested.
            connection = ranking_connection if ranking_connection is not None else self.connection
            available = connection.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name='ranking_observations'"
            ).fetchone()[0] > 0
            evidence = {}
            if available:
                evidence = {r["item_id"]: r for r in rows(connection, """
                    SELECT item_id, count(*) exposure_records,
                        sum(is_clicked)::BIGINT click_records, sum(is_purchased)::BIGINT purchase_records,
                        sum(CASE WHEN is_clicked=1 AND is_purchased=1 THEN 1 ELSE 0 END)::BIGINT clicked_and_purchased_records,
                        sum(CASE WHEN is_purchased=1 AND isfinite(price_raw) AND price_raw>0 THEN 1 ELSE 0 END)::BIGINT price_covered_purchase_records,
                        sum(CASE WHEN is_purchased=1 AND isfinite(price_raw) AND price_raw>0 THEN price_raw ELSE 0 END) priced_amount_raw
                    FROM ranking_observations WHERE item_id IN (SELECT unnest(?))
                        AND time_index>=? AND time_index<? GROUP BY item_id
                    """, [item_ids, start_time_index, before_time_index])}
            items = []
            for item in item_ids:
                records = [r for r in selected if r["item_id"] == item]
                counts = totals(records)
                detail = evidence.get(item, price_evidence(records))
                if available:
                    require(all(detail.get(k, 0) == counts[k] for k in COUNTS),
                            "Ranking/business statistics differ within the requested window")
                    if not records:
                        detail["clicked_and_purchased_records"] = 0
                items.append({"item_id": item, **counts, "metrics": statistics(counts, metrics, detail)})
            result["statistics"] = {
                "clock": "observed_time_index", "start_inclusive": start_time_index,
                "end_exclusive": before_time_index,
                "scope": "All observed request-item records in the requested window, independent of detail limit; gaps remain unobserved.",
                "amount_basis": "Recorded reference price per purchase record; one estimated unit; not actual paid revenue. Partial values cover only priced purchases.",
                "rate_basis": "Same request-item records; click-to-purchase uses the intersection of click and purchase flags. Rates are fractions, not percentages.",
                "items": items,
            }
        return result

    # Other sellers' eligible public attributes and reference-price histories.
    def competitor_snapshot(self, seller_id, focal_item_id, before):
        """Current pricing references share the recent market window."""
        self.window(before,1)
        profile=self.product(focal_item_id,before)
        require(profile['seller_id']==seller_id,'Invalid owned product binding')
        if not complete_category(profile):return []
        from agent_simulation.data.market.quotes import category_snapshot, RECENT_MARKET_SPAN
        return category_snapshot(self,tuple(profile[k] for k in CATEGORY_FIELDS),before,
                                 start=max(0,before-RECENT_MARKET_SPAN))

    def competitor_pool(self, seller_id, focal_item_id, before):
        """Authorized same-category quotes; reuse immutable per-observation projections."""
        self.window(before,1)
        profile=self.product(focal_item_id,before)
        require(profile['seller_id']==seller_id,'Invalid owned product binding')
        if not complete_category(profile):return []
        from agent_simulation.data.market.quotes import category_snapshot
        category=tuple(profile[k] for k in CATEGORY_FIELDS)
        return self.decision_cached(('competitor_pool',seller_id,category,before),lambda:[
            r['_public'] for r in category_snapshot(self,category,before)
            if r['_complete_category'] and r['seller_id']!=seller_id and r['price_time_index']<before])

    def public_pool(self, seller_id, focal_item_id, before_time_index):
        """Share one category/time-bound pool between discovery and detail lookup."""
        self.window(before_time_index, 1)
        profile = self.product(focal_item_id, before_time_index)
        require(profile["seller_id"] == seller_id, "Invalid owned product binding")
        candidates = []
        if complete_category(profile):
            from agent_simulation.data.market.quotes import category_snapshot
            category = tuple(profile[k] for k in CATEGORY_FIELDS)
            def read():
                panel = category_snapshot(self, category, before_time_index)
                return [dict(row["_public"]) for row in panel
                        if row["_complete_category"] and row["seller_id"] != seller_id]
            candidates = self.decision_cached(('public_pool', seller_id, category, before_time_index), read)
        return {
            "item_id": focal_item_id,
            "history_cutoff_exclusive": before_time_index,
            "candidates": candidates,
        }

    def discover(
        self,
        seller_id,
        owned_ids,
        focal_item_id,
        before_time_index,
        after_item_id=0,
        page_size=5,
        query="",
        brand_id=None,
    ):
        if focal_item_id not in owned_ids:
            raise PermissionError("Discovery must start from an owned market product")
        pool = self.public_pool(seller_id, focal_item_id, before_time_index)
        return discover_candidates(
            pool, after_item_id=after_item_id, page_size=page_size, query=query, brand_id=brand_id
        )

    def public(self, seller_id, owned_ids, item_ids, before_time_index, limit):
        self.window(before_time_index, limit)
        allowed = set()
        visited_categories = set()
        for own_id in owned_ids:
            records = self.records(own_id)
            require(records and records[-1]["seller_id"] == seller_id, "Invalid owned product binding")
            profile = next((r for r in reversed(records) if r["time_index"] < before_time_index), None)
            if profile is None:
                continue
            require(profile["seller_id"] == seller_id, "Invalid owned product binding")
            category = tuple(profile[k] for k in CATEGORY_FIELDS)
            if not complete_category(profile) or category in visited_categories:
                continue
            visited_categories.add(category)
            pool = self.public_pool(seller_id, own_id, before_time_index)
            allowed.update(r["item_id"] for r in pool["candidates"])
        if not set(item_ids) <= allowed:
            raise PermissionError("Product is outside the visible same-category public catalog")
        result = []
        for item_id in item_ids:
            records = [r for r in self.records(item_id) if r["time_index"] < before_time_index]
            result.append(
                {
                    "profile": self.product(item_id, before_time_index),
                    "reference_price_history": [
                        {k: r[k] for k in ("time_index", "price_raw", "price_status")}
                        for r in records[-limit:]
                    ],
                    "has_earlier_records": len(records) > limit,
                    "coverage": history_coverage(
                        records, records[-limit:], "time_index", "next_before_time_index"
                    ),
                }
            )
        return {
            "clock": "observed_time_index",
            "end_exclusive": before_time_index,
            "products": result,
        }

    # Aggregate market panels; calculations live in market.py.
    def market_position(self, seller_id, item_id, before_time_index, start_time_index=None):
        from agent_simulation.data.market.quotes import historical_position

        return historical_position(self, seller_id, item_id, before_time_index, start_time_index)

    def market_positions(self, seller_id, item_ids, before_time_index, *, decision_only=False, profiles=None):
        from agent_simulation.data.market.quotes import historical_positions

        return historical_positions(self, seller_id, item_ids, before_time_index,
                                    decision_only=decision_only, profiles=profiles)


class Catalog:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        require(self.manifest["schema"] == "kuaisearch-catalog-v1", "Unexpected catalog schema")
        value = self.manifest["pipeline_config"]
        self.config = Config(
            value["dataset"],
            Path(value["output"]),
            value["sources"],
            value["item_parts"],
        )
        self.connection = duckdb.connect(str(self.directory / "catalog.duckdb"), read_only=True)
        self.connection.execute("SET threads=2; SET memory_limit='2GB'")

    def close(self):
        self.connection.close()

    def customer_history(self, user_id, cutoff, limit=None, official_split="train", *, all_prior=False):
        if all_prior:
            require(limit is None, "Full historical retrieval cannot specify a row limit")
        else:
            limit = customer_history_limit() if limit is None else limit
            require(type(limit) is int and 1 <= limit <= 100, "History limit must be 1..100")
        require(type(user_id) is int and user_id > 0, "Invalid user ID")
        require(type(cutoff) is int and cutoff > 0, "Invalid exclusive cutoff")
        require(official_split in ("train", "test"), "Unknown official split")
        profiles = rows(self.connection, "SELECT * FROM users WHERE user_id=?", [user_id])
        require(len(profiles) == 1, "User profile absent or nonunique; no invented profile")
        # Keep other shopping topics; keywords do not silently remove evidence of habits.
        source = "FROM requests WHERE user_id=? AND time_index<? AND (?='test' OR split='train')"
        params = [user_id, cutoff, official_split]
        query = f"SELECT * {source} ORDER BY time_index DESC, session_id DESC"
        if all_prior:
            events = rows(self.connection, query, params)
            eligible = len(events)
        else:
            events = rows(self.connection, query + " LIMIT ?", [*params, limit])
            eligible = self.connection.execute(f"SELECT count(*) {source}", params).fetchone()[0]
        events.reverse()
        # A product can appear at different prices in the same user's past requests.
        keys = [(r["session_id"], item) for r in events
                for item in sorted(set(r["clicked_item_ids"]) | set(r["purchased_item_ids"]))]
        snapshots = {r["session_id"]: {} for r in events}
        if keys:
            sessions, items = zip(*keys)
            products = rows(
                self.connection,
                f"""WITH selected AS (SELECT unnest(?::BIGINT[]) session_id, unnest(?::BIGINT[]) item_id)
                SELECT o.session_id,p.*, {DISPLAY_PRICE_SQL} AS price_raw
                FROM selected JOIN ranking_observations o USING(session_id,item_id)
                JOIN products p USING(item_id)""",
                [list(sessions), list(items)],
            )
            require(len(products) == len(keys), "Historical exposure/product join mismatch")
            for product in products:
                session = product.pop("session_id")
                snapshots[session][product["item_id"]] = product
        return {
            "profile": profiles[0],
            "history_cutoff_exclusive": cutoff,
            "short_term_memory": {"search_history": [shopping_event(r, snapshots[r["session_id"]]) for r in events]},
            "long_term_memory": [],
            "provenance": {
                "kind": "observed_kuaisearch",
                "dataset": self.config.dataset,
                "request_ids": [r["session_id"] for r in events],
                "query_limit": limit,
                "eligible_history_count": eligible,
                "has_earlier_records": eligible > len(events),
            },
            "memory_note": "Long-term experiences are generated during a model run; none are fabricated by this query.",
        }

    def request(self, session_id, history_limit=None, *, include_displayed=True, all_prior=False):
        """Read one request for inspection or direct sampling, with labels separate from context."""
        request = rows(self.connection, "SELECT * FROM requests WHERE session_id=?", [session_id])
        require(len(request) == 1, "Unknown request")
        r = request[0]
        context = self.customer_history(r["user_id"], r["time_index"], history_limit, r["split"],
                                        all_prior=all_prior)
        current = rows(
            self.connection,
            f"""SELECT p.*, {DISPLAY_PRICE_SQL} AS price_raw FROM ranking_observations o
            JOIN products p USING(item_id) WHERE o.session_id=?""",
            [session_id],
        ) if include_displayed else []
        by_id = {p["item_id"]: p for p in current}
        require(
            not include_displayed or (
                len(current) == len(by_id) and set(by_id) == set(r["impressed_item_ids"])
            ),
            "Request product join mismatch",
        )
        return {
            "session_id": session_id,
            "assigned_split": rows(
                self.connection,
                "SELECT assigned_split FROM customer_splits WHERE session_id=?",
                [session_id],
            )[0]["assigned_split"],
            "query_input": context,
            "query_target": {"query": r["query"]},
            "response_input": {
                **context,
                "query": r["query"],
                "displayed_items": [by_id[i] for i in r["impressed_item_ids"]] if include_displayed else [],
            },
            "response_target": {k: r[k] for k in ("clicked_item_ids", "purchased_item_ids")},
        }

    def business(self, seller_id, item_ids, cutoff, limit=10, start=0, metrics=()):
        data = DataRepository(self.connection, self.config, cutoff)
        result = data.business(seller_id, item_ids, cutoff, limit, start, metrics)
        result["products"] = []
        for item in item_ids:
            product = data.product(item)
            product["last_observed_price_raw"] = product.pop("reference_price_raw")
            product["last_observed_time_index"] = product.pop("reference_price_time_index")
            result["products"].append(product)
        return result

    def candidates(
        self,
        seller_id,
        focal_item_id,
        cutoff,
        page_size=20,
        after_item_id=0,
        query="",
        brand_id=None,
    ):
        data = DataRepository(self.connection, self.config, cutoff)
        pool = data.public_pool(seller_id, focal_item_id, cutoff)
        return discover_candidates(
            pool, page_size=page_size, after_item_id=after_item_id, query=query, brand_id=brand_id
        )

    def public_history(self, seller_id, owned_ids, item_ids, cutoff, limit=10):
        data = DataRepository(self.connection, self.config, cutoff)
        return data.public(seller_id, owned_ids, item_ids, cutoff, limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("customer")
    p.add_argument("--user", type=int, required=True)
    p.add_argument("--before", type=int, required=True)
    p.add_argument("--limit", type=int, default=customer_history_limit())
    p.add_argument("--official-split", choices=["train", "test"], default="train")
    p = sub.add_parser("request")
    p.add_argument("--session", type=int, required=True)
    p = sub.add_parser("seller")
    p.add_argument("--seller", type=int, required=True)
    p.add_argument("--items", type=int, nargs="+", required=True)
    p.add_argument("--before", type=int, required=True)
    p.add_argument("--limit", type=int, default=10)
    p = sub.add_parser("public")
    p.add_argument("--seller", type=int, required=True)
    p.add_argument("--item", type=int, required=True)
    p.add_argument("--before", type=int, required=True)
    p.add_argument("--after-item", type=int, default=0)
    p.add_argument("--query", default="")
    p = sub.add_parser("public-history")
    p.add_argument("--seller", type=int, required=True)
    p.add_argument("--owned-items", type=int, nargs="+", required=True)
    p.add_argument("--items", type=int, nargs="+", required=True)
    p.add_argument("--before", type=int, required=True)
    p.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    catalog = Catalog(args.catalog)
    try:
        if args.action == "customer":
            result = catalog.customer_history(
                args.user, args.before, args.limit, args.official_split
            )
        elif args.action == "request":
            result = catalog.request(args.session)
        elif args.action == "seller":
            result = catalog.business(args.seller, args.items, args.before, args.limit)
        elif args.action == "public-history":
            result = catalog.public_history(
                args.seller, args.owned_items, args.items, args.before, args.limit
            )
        else:
            result = catalog.candidates(
                args.seller, args.item, args.before, after_item_id=args.after_item, query=args.query
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    finally:
        catalog.close()
