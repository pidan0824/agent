"""Paired Customer price interventions on frozen exposure lists."""

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
from math import isfinite
from statistics import mean, median

from agent_simulation.agents.prompts.customer import messages


PRICE_MULTIPLIERS = (0.8, 1.0, 1.2)


class PriceResponseStudy:
    """Accumulate paired decisions and observed exposures for the same test cohort."""

    def __init__(self, seed):
        self.seed = seed
        self.trials = []
        self.categories = {}
        self.exposures = defaultdict(list)

    def add_request(self, batches, outputs, merged, model, predict):
        self.trials.append(compare_prices(batches, outputs, model, predict, self.seed))
        if merged["status"] != "ok":
            return
        predicted = set(merged["prediction"]["purchased_item_ids"])
        purchased = set(merged["target"]["purchased_item_ids"])
        for batch in batches:
            for product in batch["input"]["displayed_items"]:
                item = product["item_id"]
                self.categories[item] = str(product.get("category_level1_id"))
                price = product.get("price_raw")
                if price is not None and isfinite(price) and price > 0:
                    self.exposures[item].append((price, item in predicted, item in purchased))

    def report(self, records):
        return {**population(records, self.categories, self.trials),
                "observational_within_item": observed_price_gradient(self.exposures)}


def compare_prices(samples, baseline, model, predict, seed):
    """Choose one displayed product per request independently of its choice label."""
    request_id = samples[0]["sample_id"].partition(":batch:")[0]
    failed = [r for r in baseline if r["status"] != "ok"]
    if failed:
        return {"status": "invalid", "sample_id": request_id, "reason": "invalid baseline",
                "failures": [{"sample_id": r["sample_id"], "error": r["error"]} for r in failed]}
    candidates = [(batch, item) for batch, sample in enumerate(samples)
                  for item in sample["input"]["displayed_items"]
                  if isinstance(item.get("price_raw"), (float, int))
                  and isfinite(item["price_raw"]) and item["price_raw"] > 0]
    if not candidates:
        return {"status": "ineligible", "sample_id": request_id, "reason": "no positive displayed price"}

    def order(candidate):
        _, item = candidate
        return sha256(f'{seed}:{request_id}:{item["item_id"]}'.encode()).digest()
    batch, item = min(candidates, key=order)
    item_id = item["item_id"]
    arms = []
    for multiplier in PRICE_MULTIPLIERS:
        price = item["price_raw"] if multiplier == 1 else item["price_raw"] * multiplier
        result = baseline[batch]
        if multiplier != 1:
            variant = deepcopy(dict(samples[batch]))  # Copy business fields, excluding the reader's tokenizer cache.
            for product in variant["input"]["displayed_items"]:
                if product["item_id"] == item_id:
                    product["price_raw"] = price
            variant["messages"] = [*messages(variant["input"], response=True), variant["messages"][-1]]
            variant["sample_id"] += f":price-{multiplier}"
            result = predict(variant, model)
        ok = result["status"] == "ok"
        purchases = None
        if ok:
            outputs = [*baseline[:batch], result, *baseline[batch + 1:]]
            purchases = [i for output in outputs for i in output["prediction"]["purchased_item_ids"]]
        arms.append({"multiplier": multiplier, "price_raw": price, "status": result["status"],
                     **({"error": result["error"]} if not ok else {}),
                     "purchased": item_id in purchases if ok else None,
                     "purchased_item_ids": purchases})
    return {"status": "ok" if all(a["status"] == "ok" for a in arms) else "invalid",
            "sample_id": samples[batch]["sample_id"], "item_id": item_id, "arms": arms}


def observed_price_gradient(exposures):
    """Compare actual and predicted buying at observed low vs high prices within item."""
    differences = {"prediction": [], "target": []}
    counts = {}
    for item, rows in exposures.items():
        middle = median({price for price, _, _ in rows})
        low = [(p, t) for price, p, t in rows if price < middle]
        high = [(p, t) for price, p, t in rows if price > middle]
        if low and high:
            high_predicted, high_actual = zip(*high)
            low_predicted, low_actual = zip(*low)
            differences["prediction"].append(mean(high_predicted) - mean(low_predicted))
            differences["target"].append(mean(high_actual) - mean(low_actual))
            counts[item] = {"low": len(low), "high": len(high)}
    return {
        "comparable_items": len(counts),
        "grouping": "below/above median distinct observed price within item; equal item weights",
        "exposure_counts_by_item": counts,
        "high_minus_low_purchase_rate": {
            name: mean(values) if values else None for name, values in differences.items()
        },
    }


def population(records, categories, trials):
    responses = [r for r in records if r["stage"] == "response"]
    valid = [r for r in responses if r["status"] == "ok"]
    distributions = {}
    for field in ("item", "category"):
        counts = {}
        for side in ("prediction", "target"):
            counts[side] = Counter(
                (item if field == "item" else categories[item])
                for r in valid for item in r[side]["purchased_item_ids"])
        p, t = counts["prediction"], counts["target"]
        pn, tn = sum(p.values()), sum(t.values())
        distributions[field] = {"counts": counts, "total_variation":
            sum(abs(p[k] / pn - t[k] / tn) for k in p.keys() | t.keys()) / 2 if pn and tn else None}
    paired = [r for r in trials if r["status"] == "ok"]
    focal_rates, other_rates, request_rates = [], [], []
    for index in range(len(PRICE_MULTIPLIERS)):
        outcomes = [(r["item_id"], r["arms"][index]["purchased_item_ids"]) for r in paired]
        focal_rates.append(mean(item in purchases for item, purchases in outcomes) if outcomes else None)
        other_rates.append(mean(
            any(i != item for i in purchases) for item, purchases in outcomes
        ) if outcomes else None)
        request_rates.append(mean(bool(purchases) for _, purchases in outcomes) if outcomes else None)
    return {"requests": len(responses), "valid_requests": len(valid),
            "purchase_distributions": distributions, "price_trials": trials,
            "valid_pairs": len(paired), "multipliers": list(PRICE_MULTIPLIERS),
            "purchase_rate_scope": "Share of paired requests with each outcome from one greedy decision per batch.",
            "focal_purchase_rates": focal_rates,
            "other_purchase_rates": other_rates,
            "request_purchase_rates": request_rates}
