"""Agent behavior metrics."""


def price_direction(change):
    if change == 0:
        return "keep"
    return "increase" if change > 0 else "decrease"
