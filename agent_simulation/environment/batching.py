"""Split the stored display list into independent, ordered Customer response batches."""

from copy import deepcopy

from agent_simulation.inference.requests import model_policy
from agent_simulation.data.common import require

CUSTOMER_RESPONSE_BATCH_SIZE = model_policy()["customer_response_batch_size"]


def response_batches(state, *, max_items=None):
    """Partition by item count without rendering or tokenizing trial prompts."""
    require("candidate_batch" not in state, "Response already partitioned")
    items = state["displayed_items"]
    require(len({p["item_id"] for p in items}) == len(items), "Duplicate exposure item")
    size = CUSTOMER_RESPONSE_BATCH_SIZE if max_items is None else max_items
    require(type(size) is int and size > 0, "Customer response batch size must be positive")
    if len(items) <= size:
        return [deepcopy(state)]
    count = (len(items) + size - 1) // size
    common = {k: v for k, v in state.items() if k != "displayed_items"}
    return [
        {
            **deepcopy(common),
            "displayed_items": deepcopy(items[start:start + size]),
            "candidate_batch": {"index": start // size + 1, "count": count,
                                "request_item_count": len(items)},
        }
        for start in range(0, len(items), size)
    ]
