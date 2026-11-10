"""Strict scalar, JSON and Function Calling schema validation."""

import json
import math
import unicodedata

from jsonschema import Draft202012Validator, validators, ValidationError


def fields(value, required, optional=()):
    if (
        not isinstance(value, dict)
        or not set(required) <= value.keys()
        or value.keys() - set(required) - set(optional)
    ):
        raise ValueError(f"Expected fields {sorted(required)}, optional {sorted(optional)}")


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Expected integer >= {minimum}")
    return value


def has_visible_text(value):
    """Require at least one letter, number, punctuation mark or symbol."""
    return any(unicodedata.category(char)[0] in "LNPS" for char in value)


def text(value):
    if not isinstance(value, str) or not has_visible_text(value):
        raise ValueError("Expected nonempty text")
    return value


def strict_json(value):
    """Reject duplicate keys, nonfinite numbers and Markdown wrappers, never repair."""

    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = item
        return result

    def invalid(constant):
        raise ValueError(f"Nonfinite JSON: {constant}")

    def finite_float(literal):
        number = float(literal)
        if not math.isfinite(number):
            invalid(literal)
        return number

    return json.loads(
        value, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite_float
    )


def function(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


# The sandbox expects integral JSON tokens and finite prices, with no coercion.
_types = Draft202012Validator.TYPE_CHECKER.redefine_many(
    {
        "integer": lambda _, v: type(v) is int,
        "number": lambda _, v: type(v) is int or (type(v) is float and math.isfinite(v)),
    }
)


def _min_length(validator, minimum, value, schema):
    if isinstance(value, str) and (
        len(value.strip()) < minimum or (minimum > 0 and not has_visible_text(value))
    ):
        yield ValidationError("String is too short or contains no visible text")


StrictSchema = validators.extend(
    Draft202012Validator, {"minLength": _min_length}, type_checker=_types
)


def validate_arguments(value, schema, path="arguments"):
    """Validate the published JSON schema using the same strict scalar rules as runtime."""
    error = next(StrictSchema(schema).iter_errors(value), None)
    if error:
        raise ValueError(f"{path}{error.json_path[1:]}: {error.message}")


ID = {"type": "integer", "minimum": 1}

IDS = {"type": "array", "items": ID, "uniqueItems": True}
