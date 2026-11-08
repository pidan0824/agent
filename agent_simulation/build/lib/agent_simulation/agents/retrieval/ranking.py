"""Shared rank fusion and a bounded freshness preference for factual evidence."""

from collections import defaultdict

FRESHNESS_POLICY = {
    "maximum_penalty": 0.1,
    "half_life": {"observed_time_index": 10000, "simulation_settled_step": 10},
    "calibrated": False,
}


def average_ranks(values):
    """Descending midranks; an unknown signal casts no vote."""
    positions = defaultdict(list)
    for position, value in enumerate(sorted((v for v in values if v is not None), reverse=True), 1):
        positions[value].append(position)
    ranks = {value: sum(p) / len(p) for value, p in positions.items()}
    return [ranks.get(v) if v is not None else None for v in values]


def reciprocal_rank_scores(signals, constant=60):
    ranks = [average_ranks(values) for values in signals]
    return [sum(1 / (constant + r[i]) for r in ranks if r[i] is not None)
            for i in range(len(signals[0]))]


def freshness_factor(index, before, clock="observed_time_index"):
    """Clock indices are not elapsed days; never decay generated memories here."""
    if before is None:
        return 1.0
    age = max(0, before - 1 - index)
    return 1 - FRESHNESS_POLICY["maximum_penalty"] * (
        1 - 2 ** (-age / FRESHNESS_POLICY["half_life"][clock]))
