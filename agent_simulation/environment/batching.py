"""Fit the ordered Customer display into the model's input budget."""

from copy import deepcopy

from agent_simulation.agents.context.shopping import project_customer
from agent_simulation.agents.prompts.customer import messages, prepare_history
from agent_simulation.data.common import require
from agent_simulation.inference.requests import ContextCapacityError, input_token_budget, model_policy, prompt_tokens


def response_batch_policy():
    policy = model_policy()
    return {"method": "prompt-tokens-v1", "context_limit": policy["context_limit"],
            "reserved_output_tokens": policy["reserved_output_tokens"]}


def response_batches(state, tokenizer, *, max_output_tokens=0, compact_background=False):
    """Return each display slice with its exact, label-free model messages."""
    require("candidate_batch" not in state, "Response already partitioned")
    items = state["displayed_items"]
    require(len({p["item_id"] for p in items}) == len(items), "Duplicate exposure item")
    budget = input_token_budget(max_output_tokens)
    history = prepare_history(project_customer(state))

    def render(view):
        prompt = messages(view, response=True, prepared_history=history,
                          compact_background=compact_background)
        return prompt, len(prompt_tokens(prompt, [], tokenizer))

    prompt, size = render(state)
    if size <= budget:
        return [(deepcopy(state), prompt)]
    if len(items) <= 1:
        raise ContextCapacityError(f"Customer Response needs {size} input tokens with a budget of {budget}")

    batches = []
    start = 0
    while start < len(items):
        # Find a fitting consecutive prefix. Only the probed prompts can be emitted.
        low, high = start + 1, len(items)
        best = None
        while low <= high:
            end = (low + high) // 2
            view = {**state, "displayed_items": items[start:end],
                    "candidate_batch": {"request_item_count": len(items)}}
            prompt, size = render(view)
            if size <= budget:
                best = end, view, prompt
                low = end + 1
            else:
                high = end - 1
        if best is None:
            raise ContextCapacityError(
                f"Customer Response with only item {items[start]['item_id']} "
                f"needs {size} input tokens with a budget of {budget}"
            )
        start, view, prompt = best
        batches.append((deepcopy(view), prompt))

    for index, (view, _) in enumerate(batches, 1):
        view["candidate_batch"].update(index=index, count=len(batches))
    return batches
