"""Customer and Seller action schemas shared by supervision and the environment."""

from copy import deepcopy
from agent_simulation.agents.schemas import ID, IDS, function
from agent_simulation.data.common import require


SEARCH_PRODUCTS = function(
    "search_products",
    "Submit a product search query to retrieve the platform's current product list.",
    {"query": {"type": "string", "minLength": 1}},
    ["query"],
)

SUBMIT_RESPONSE = function(
    "submit_response",
    "Submit click and purchase sets drawn from the current displayed products; either set may be empty.",
    {"clicked_item_ids": IDS, "purchased_item_ids": IDS},
    ["clicked_item_ids", "purchased_item_ids"],
)


def validate_response(response, displayed_ids):
    """Validate a proposed response without inventing a response or recording a purchase."""
    require(
        set(response) == {"clicked_item_ids", "purchased_item_ids"}, "Unexpected response fields"
    )
    allowed = set(displayed_ids)
    for ids in response.values():
        require(
            isinstance(ids, list) and all(type(i) is int and i > 0 for i in ids),
            "Invalid response IDs",
        )
        selected = set(ids)
        require(len(ids) == len(selected), "Duplicate response IDs")
        require(selected <= allowed, "Response selects an unshown item")
    return deepcopy(response)


SUBMIT_PRICES = function(
    "submit_prices",
    "Submit prices for selected owned products on the stored numerical scale; an empty list requests no changes.",
    {
        "prices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_id": ID,
                    "price_raw": {"type": "number", "exclusiveMinimum": 0},
                },
                "required": ["item_id", "price_raw"],
                "additionalProperties": False,
            },
        }
    },
    ["prices"],
)


def price_schema(item_ids=None, *, allow_unknown=False):
    """Bind a JSON quote to its targets and, in simulation, allow unknown prices."""
    schema = deepcopy(SUBMIT_PRICES["function"]["parameters"])
    prices = schema["properties"]["prices"]
    fields = prices["items"]["properties"]
    if item_ids is not None:
        require(item_ids and len(item_ids) == len(set(item_ids))
                and all(type(item) is int and item > 0 for item in item_ids), "Invalid price targets")
        prices.update(minItems=len(item_ids), maxItems=len(item_ids))
        fields["item_id"]["enum"] = sorted(item_ids)
    if allow_unknown:
        fields["price_raw"] = {"anyOf": [fields["price_raw"], {"type": "null"}]}
    return schema
