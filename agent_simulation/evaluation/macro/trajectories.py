"""Compare simulated and historical trajectories for the same customers and products."""

import argparse
from collections import Counter
from itertools import pairwise
import json
import math
import statistics
from pathlib import Path

from agent_simulation.data.catalog import Catalog
from agent_simulation.data.common import rows, require, sha256
from agent_simulation.evaluation import price_direction
from agent_simulation.logs import write_report


def customer_distribution(events):
    """Request-level rates, separate from item exposure rates or physical quantities."""
    clicks = [len(e["clicked_item_ids"]) for e in events]
    buys = [len(e["purchased_item_ids"]) for e in events]
    return {
        "requests": len(events),
        "click_request_rate": statistics.mean(n > 0 for n in clicks) if events else None,
        "purchase_request_rate": statistics.mean(n > 0 for n in buys) if events else None,
        "no_response_rate": statistics.mean(not c and not b for c, b in zip(clicks, buys)) if events else None,
        "clicked_counts": clicks,
        "purchased_counts": buys,
        "repeated_query_share": (
            statistics.mean(
                a["query"].strip() == b["query"].strip() for a, b in zip(events, events[1:])
            ) if len(events) > 1 else None
        ),
    }


def dtw(x, y):
    """Sqrt of the minimum sum of squared costs, with monotone unrestricted warping."""
    if not x or not y or not all(math.isfinite(v) for v in [*x, *y]):
        raise ValueError("DTW needs two nonempty finite sequences")
    previous = [0.0] + [math.inf] * len(y)
    for value in x:
        current = [math.inf]
        for j, other in enumerate(y, 1):
            current.append((value - other) ** 2 + min(previous[j], previous[j - 1], current[-1]))
        previous = current
    return math.sqrt(previous[-1])


def price_dynamics(values, *, initial_price=None):
    """Summarize sampled prices and include the first adjustment if its baseline is known."""
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Expected positive finite prices")
    path = values if initial_price is None else [initial_price, *values]
    changes = [(b - a) / a for a, b in pairwise(path)]
    directions = Counter(map(price_direction, changes))
    runs = [1]
    for a, b in pairwise(values):
        if a == b:
            runs[-1] += 1
        else:
            runs.append(1)
    return {
        "relative_levels": [v / path[0] for v in values],
        "coefficient_of_variation": statistics.pstdev(values) / statistics.mean(values),
        "direction_shares": {
            k: directions[k] / len(changes) if changes else None
            for k in ("keep", "increase", "decrease")
        },
        "relative_changes": changes,
        "same_price_runs_observations": runs,
    }


def align_prices(observations, tick_times):
    """Keep the first observed quote in each tick's half-open interval."""
    matched = {}
    end = tick_times.start + len(tick_times) * tick_times.step
    for observation in observations:
        time = observation["time_index"]
        if tick_times.start <= time < end:
            tick = (time - tick_times.start) // tick_times.step
            if tick not in matched:
                matched[tick] = dict(observation, tick=tick, logical_time=tick_times[tick])
    return list(matched.values())


def compare(directory, catalog_path, output):
    initial = json.loads((directory / "initial_state.json").read_text())
    final = json.loads((directory / "final_state.json").read_text())
    runtime_report = json.loads((directory / "report.json").read_text())
    catalog = Catalog(catalog_path)
    try:
        cutoff = initial["historical_cutoff_exclusive"]
        require(sha256(catalog.directory / "manifest.json") == initial["catalog_manifest_sha256"],
                "Different source catalog release")
        clock = final["clock"]
        completed = clock["tick"]
        tick_size = clock["tick_size"]
        completed_before = clock["start"] + completed * tick_size
        tick_times = range(clock["start"], completed_before, tick_size)
        # Actions are in commit order. The last quote at a given Tick takes effect.
        quotes = {(a["item_id"], a["logical_time"]): a["price_raw"] for a in final["price_actions"]}
        prices = []
        for item, product in initial["products"].items():
            item_id = int(item)
            initial_price = initial["active_prices"][item]
            start_price = initial_price["price_raw"]
            price = {"item_id": item_id, "seller_id": product["seller_id"]}
            prices.append(price)
            if start_price is None or not completed:
                price["status"] = "missing_initial_price" if start_price is None else "no_completed_ticks"
                continue
            generated = []
            changed_ticks = set()
            current = start_price
            for tick, time in enumerate(tick_times):
                updated = quotes.get((item_id, time), current)
                if updated != current:
                    changed_ticks.add(tick)
                current = updated
                generated.append({"tick": tick, "logical_time": time, "price_raw": current})
            observed = rows(
                catalog.connection,
                """SELECT time_index,price_raw FROM seller_history
                WHERE item_id=? AND time_index>=? AND time_index<? AND price_status='observed'
                      AND isfinite(price_raw) AND price_raw>0
                ORDER BY time_index""",
                [item_id, cutoff, completed_before],
            )
            price.update(
                initial_price=initial_price,
                simulated=generated,
                simulated_full_dynamics=price_dynamics(
                    [e["price_raw"] for e in generated], initial_price=start_price
                ),
                reference_observations=len(observed),
                matched_future_points=0,
                matched_tick_share=0.0,
                price_change_ticks=len(changed_ticks),
                price_change_ticks_with_reference=0,
            )
            if not observed:
                price["status"] = "no_future_reference"
                continue
            observed = align_prices(observed, tick_times)
            x = [e["price_raw"] for e in observed]
            y = [generated[e["tick"]]["price_raw"] for e in observed]
            observed_dynamics, simulated_dynamics = price_dynamics(x), price_dynamics(y)
            shape_distance = dtw(x, y)
            price.update(
                status="aligned",
                observed=observed,
                matched_future_points=len(x),
                matched_tick_share=len(x) / completed,
                price_change_ticks_with_reference=sum(e["tick"] in changed_ticks for e in observed),
                keep_initial_price_baseline={
                    "price_mae_raw": statistics.mean(abs(actual - start_price) for actual in x),
                    "mean_relative_absolute_error": statistics.mean(
                        abs(actual - start_price) / actual for actual in x
                    ),
                },
                price_mae_raw=statistics.mean(abs(actual - simulated) for actual, simulated in zip(x, y)),
                mean_relative_absolute_error=statistics.mean(
                    abs(actual - simulated) / actual for actual, simulated in zip(x, y)
                ),
                direction_agreement=statistics.mean(
                    price_direction(actual) == price_direction(simulated)
                    for actual, simulated in zip(
                        observed_dynamics["relative_changes"], simulated_dynamics["relative_changes"]
                    )
                ) if len(x) > 1 else None,
                dtw_raw=shape_distance,
                dtw_relative_to_initial=shape_distance / start_price,
                observed_dynamics=observed_dynamics,
                simulated_dynamics=simulated_dynamics,
            )
        customers = []
        for user, state in final["customers"].items():
            generated = [
                {
                    "query": e["query"],
                    "clicked_item_ids": [p["item_id"] for p in e["clicked_items"]],
                    "purchased_item_ids": [p["item_id"] for p in e["purchased_items"]],
                    "logical_time": e["search_time"],
                }
                for e in state["simulation_history"]
                if e["search_time"] < completed_before
            ]
            observed = rows(
                catalog.connection,
                """SELECT session_id,time_index,query,clicked_item_ids,purchased_item_ids FROM requests
                WHERE user_id=? AND time_index>=? AND time_index<? ORDER BY time_index,session_id""",
                [int(user), cutoff, completed_before],
            )
            exit_event = state.get("exit")
            if exit_event and exit_event["tick"] >= completed:
                exit_event = None
            opportunities = exit_event["tick"] + 1 if exit_event else completed
            customers.append(
                {
                    "user_id": int(user),
                    "observed": observed,
                    "simulated": generated,
                    "observed_distribution": customer_distribution(observed),
                    "simulated_distribution": customer_distribution(generated),
                    "observed_search_intervals": [
                        b["time_index"] - a["time_index"] for a, b in pairwise(observed)
                    ],
                    "simulated_search_intervals": [
                        b["logical_time"] - a["logical_time"] for a, b in pairwise(generated)
                    ],
                    "simulated_opportunities": opportunities,
                    "simulated_no_action_count": opportunities - len(generated) - int(exit_event is not None),
                    "simulated_search_rate": len(generated) / opportunities if opportunities else None,
                    "exit": exit_event,
                }
            )
        result = {
            "episode_status": runtime_report["status"],
            "completed_ticks": completed,
            "reference_window": {"start_inclusive": cutoff, "end_exclusive": completed_before},
            "tick_size": tick_size,
            "price_alignment": "Use the first valid quote in each completed tick interval and compare it with the simulated price effective in that interval.",
            "limits": [
                "The shared clock measures source index steps. Differences can reflect platform policy as well as agent behavior.",
                "DTW is an exploratory shape metric and permits position shifts.",
                "Reference prices are sparse. Price error and direction agreement use ticks with matched observations.",
                "The historical log records searches and has no labels for decision opportunities, waiting or exit.",
            ],
            "seller": prices,
            "customer": customers,
            "dtw_relative_distribution": [
                r["dtw_relative_to_initial"] for r in prices if r["status"] == "aligned"
            ],
        }
        write_report(output, result)
        return result
    finally:
        catalog.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    compare(a.run, a.catalog, a.output)
