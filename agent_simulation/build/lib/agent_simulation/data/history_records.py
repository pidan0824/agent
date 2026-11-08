"""Select historical records and summarize observed price changes."""

from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from functools import cache
from itertools import islice
import json
import math
from pathlib import Path

from agent_simulation.data.market.statistics import COUNTS
from agent_simulation.data.common import require


@cache
def _policy():
    path = Path(__file__).parents[1] / "inference" / "model_policy.json"
    return json.loads(path.read_text())["history_windows"]


def has_observed_price(row):
    value = row["price_raw"]
    return (row["price_status"] == "observed" and type(value) in (int, float)
            and math.isfinite(value) and value > 0)


def price_point(row):
    return {"price_raw": row["price_raw"], "time_index": row["time_index"]}


def last_valid_reference(records, before):
    """The most recent usable observation in one item's chronological history."""
    return next((price_point(r) for r in reversed(records)
                 if r["time_index"] < before and has_observed_price(r)), None)


def seller_history_policy():
    policy = _policy()["seller"]
    require(
        policy["selection"] == "latest_observations_per_item",
        "Unexpected history selection",
    )
    require(
        type(policy["observations_per_item"]) is int
        and policy["observations_per_item"] > 0,
        "Invalid Seller history window",
    )
    return policy.copy()


def customer_history_limit():
    n = _policy()["customer"]["recent_requests"]
    require(type(n) is int and 1 <= n <= 100, "Invalid Customer history window")
    return n


EVENT_FIELDS = (
    "item_id",
    "time_index",
    "price_raw",
    "price_status",
    "exposure_records",
    "click_records",
    "purchase_records",
)


def group_business_history(observed, *, owner_id, cutoff, simulated=(), simulation_cutoff=None):
    """Validate ownership and order each product's observations on separate clocks."""
    require(type(cutoff) is int and cutoff > 0, "Invalid observation cutoff")
    require(
        not simulated or (type(simulation_cutoff) is int and simulation_cutoff > 0),
        "Simulation cutoff is required for simulated feedback",
    )
    by_item, seen = defaultdict(list), set()
    for clock, events, time_field, bound in (
        (0, observed, "time_index", cutoff),
        (1, simulated, "settled_step", simulation_cutoff),
    ):
        for row in events:
            require(row["seller_id"] == owner_id, "Foreign private business records")
            index = row[time_field]
            require(type(index) is int and index >= 0, "Invalid business event index")
            if index >= bound:
                continue
            key = (clock, index, row["item_id"])
            require(key not in seen, "Duplicate product observation point")
            seen.add(key)
            if clock:
                require(
                    type(row["search_step"]) is int and row["search_step"] < index,
                    "Feedback must settle after its displayed list",
                )
            by_item[row["item_id"]].append((clock, index, row))
    ordered = dict(sorted(by_item.items()))
    for events in ordered.values():
        events.sort(key=lambda entry: entry[:2])
    return ordered


def recent_business_history(by_item, observations_per_item):
    """Read recent events and coverage from authorized, ordered product histories."""
    require(
        type(observations_per_item) is int and observations_per_item > 0,
        "Invalid per-item memory limit",
    )
    grouped, coverage = defaultdict(list), []
    for item, events in by_item.items():
        kept = events[-observations_per_item:]
        for clock, index, row in kept:
            grouped[clock, index].append(row)
        coverage.append(
            {
                "item_id": item,
                "available_points": len(events),
                "retained_points": len(kept),
                "observed_points": sum(clock == 0 for clock, _, _ in kept),
                "simulated_points": sum(clock == 1 for clock, _, _ in kept),
            }
        )
    memory = []
    for (clock, index), events in sorted(grouped.items()):
        if clock == 0:
            memory.append(
                {
                    "time_index": index,
                    "generator": "environment:observed_business_event",
                    "source_ids": [f"item:{r['item_id']}:time:{index}" for r in events],
                    "content": [
                        {k: deepcopy(r[k]) for k in EVENT_FIELDS} for r in events
                    ],
                }
            )
        else:
            fields = (
                "item_id",
                "price_raw",
                "price_status",
                "exposure_records",
                "click_records",
                "purchase_records",
            )
            memory.append(
                {
                    "simulation_step": index,
                    "generator": "environment:simulated_business_event",
                    "source_ids": [
                        f"simulation:request:{r['request_id']}:item:{r['item_id']}"
                        for r in events
                    ],
                    "content": [
                        {
                            **{k: deepcopy(r[k]) for k in fields},
                            "search_step": r["search_step"],
                            "settled_step": index,
                            **{
                                k: r[k]
                                for k in ("search_time", "settled_time")
                                if k in r
                            },
                        }
                        for r in events
                    ],
                }
            )
    return memory, coverage


def select_business_history(
    observed,
    *,
    owner_id,
    cutoff,
    observations_per_item,
    simulated=(),
    simulation_cutoff=None,
):
    """Keep N points per item across the seed history and subsequent settled feedback."""
    grouped = group_business_history(
        observed, owner_id=owner_id, cutoff=cutoff,
        simulated=simulated, simulation_cutoff=simulation_cutoff,
    )
    return recent_business_history(grouped, observations_per_item)


def shopping_event(row, items):
    """Do not copy the full exposure list or infer a reason for the observed choice."""
    require(
        isinstance(row["query"], str) and bool(row["query"].strip()),
        "Empty historical Query",
    )
    for key in ("clicked_item_ids", "purchased_item_ids"):
        ids = row[key]
        require(len(ids) == len(set(ids)), "Duplicate historical response ID")
        require(
            set(ids) <= set(row["impressed_item_ids"]),
            "Historical response outside exposure set",
        )
        require(set(ids) <= set(items), "Missing historical product attributes")
    return {
        "session_id": row["session_id"],
        "time_index": row["time_index"],
        "query": row["query"],
        "clicked_items": [deepcopy(items[i]) for i in row["clicked_item_ids"]],
        "purchased_items": [deepcopy(items[i]) for i in row["purchased_item_ids"]],
    }


PRICE_HISTORY_POLICY = {
    'schema': 'observed-price-history-v1',
    'recent_runs': 4,
    'recent_price_levels': 4,
    'semantics': 'All statistics use observations strictly before the decision cutoff. '
                 'Runs group consecutive observations with the same price and price status. '
                 'Unknown prices remain explicit. Gaps do not establish continuous pricing or zero activity. '
                 'Valid quote differences are observations, not actual price-edit actions. '
                 'Price-level totals combine all prior records at that price; they are not causal effects.',
}


def _totals(index):
    return dict(first_index=index, last_index=index, observations=0, **dict.fromkeys(COUNTS, 0))


def _add(total, row):
    total['last_index'] = row['time_index']
    total['observations'] += 1
    for field in COUNTS:
        total[field] += row[field]


class ObservedPriceHistory:
    """One item, constant update work; bounded views never scan the observation prefix."""

    def __init__(self, seller, item):
        self.seller, self.item = seller, item
        self.totals = None
        self.levels = OrderedDict()
        self.runs = deque(maxlen=PRICE_HISTORY_POLICY['recent_runs'])
        self.run_count = self.changes = self.unknown = 0
        self.last_valid = None
        self.minimum = self.maximum = None

    def append(self, row):
        require(row['seller_id'] == self.seller and row['item_id'] == self.item,
                'Foreign price-history observation')
        index = row['time_index']
        require(type(index) is int and index >= 0, 'Invalid price-history index')
        require(self.totals is None or index > self.totals['last_index'],
                'Price history must be strictly chronological')
        require(all(type(row[k]) is int and row[k] >= 0 for k in COUNTS), 'Invalid price-history counts')
        price = row['price_raw'] if has_observed_price(row) else None
        if self.totals is None:
            self.totals = _totals(index)
        _add(self.totals, row)
        if price is None:
            self.unknown += 1
        else:
            self.changes += int(self.last_valid is not None and price != self.last_valid)
            self.last_valid = price
            self.minimum = price if self.minimum is None else min(price, self.minimum)
            self.maximum = price if self.maximum is None else max(price, self.maximum)
            level = self.levels.get(price)
            if level is None:
                level = self.levels[price] = _totals(index)
            _add(level, row)
            self.levels.move_to_end(price)
        status = row['price_status']
        last = self.runs[-1] if self.runs else None
        if last is None or last['price_raw'] != price or last['price_status'] != status:
            self.runs.append(dict(price_raw=price, price_status=status, **_totals(index)))
            self.run_count += 1
        _add(self.runs[-1], row)

    def view(self, cutoff):
        require(type(cutoff) is int and self.totals is not None and self.totals['last_index'] < cutoff,
                'Future or empty price history')
        # OrderedDict reverse iteration takes at most the configured number of levels.
        prices = list(islice(reversed(self.levels), PRICE_HISTORY_POLICY['recent_price_levels']))
        return deepcopy(dict(
            clock='observed_time_index', end_exclusive=cutoff,
            all_observations=self.totals,
            unknown_price_observations=self.unknown,
            distinct_valid_prices=len(self.levels), valid_quote_changes=self.changes,
            min_price_raw=self.minimum, max_price_raw=self.maximum,
            omitted_earlier_runs=self.run_count-len(self.runs),
            recent_observation_runs=list(self.runs),
            omitted_price_levels=len(self.levels)-len(prices),
            price_level_totals=[dict(price_raw=p, **self.levels[p]) for p in prices],
        ))
