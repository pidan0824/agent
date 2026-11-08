"""Read and constrain structured JSON model decisions."""

from copy import deepcopy
from inspect import isawaitable

from agent_simulation.data.common import require
from agent_simulation.agents.schemas import strict_json
from agent_simulation.inference.requests import prepare_request
from agent_simulation.inference.client import ModelOutputError, response_tool_calls
from agent_simulation.agents.actions import SEARCH_PRODUCTS, SUBMIT_RESPONSE, price_schema


class UnavailableModelAction(PermissionError):
    """A model requested a tool that the decision stage does not expose."""


async def run_workflow(*, messages, complete, validate_final):
    """Accept one JSON action from a sync or async model call."""
    response = complete(messages, [])
    if isawaitable(response):
        response = await response
    if response_tool_calls(response):
        raise UnavailableModelAction("Tool is unavailable in this decision")
    if response["content"] is None:
        raise ModelOutputError("Missing JSON answer", raw_output=response)
    try:
        args = strict_json(response["content"])
        require(isinstance(args, dict), "Expected a JSON object")
    except ValueError as error:
        raise ModelOutputError(str(error), raw_output=response) from error
    return validate_final(args)


def action_output(stage, *, allow_control_actions=False, displayed_ids=None,
                  price_ids=None, allow_unknown_prices=False):
    """Declare the final-answer contract from the decision scope."""
    if stage == "seller":
        require(price_ids is not None, "Seller generation requires price targets")
        alternatives = [price_schema(price_ids, allow_unknown=allow_unknown_prices)]
    elif stage in {"query", "response"}:
        action = SEARCH_PRODUCTS if stage == "query" else SUBMIT_RESPONSE
        alternatives = [deepcopy(action["function"]["parameters"])]
    else:
        raise ValueError("Unknown action stage")
    if stage == "response":
        require(displayed_ids is not None, "Response generation requires displayed IDs")
        ids = sorted(set(displayed_ids))
        require(all(type(item) is int and item > 0 for item in ids), "Invalid displayed IDs")
        for field in alternatives[0]["properties"].values():
            field["items"]["enum"] = ids
        if not ids:
            alternatives = [{"const": {"clicked_item_ids": [], "purchased_item_ids": []}}]
    if allow_control_actions and stage == "query":
        alternatives.append(
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["no_action", "exit"],
                    }
                },
                "required": ["action"],
                "additionalProperties": False,
            }
        )
    return {"final_schemas": alternatives}


def decision_request(messages, tools, tokenizer, *, role, stage, max_output_tokens=0,
                     allow_wait=False, displayed_ids=None, price_ids=None):
    request, count = prepare_request(messages, tools, tokenizer, max_output_tokens=max_output_tokens)
    request['agent_role'] = role
    request['action_output'] = action_output(stage, allow_control_actions=allow_wait,
                                            displayed_ids=displayed_ids, price_ids=price_ids,
                                            allow_unknown_prices=stage == "seller")
    return request, count
