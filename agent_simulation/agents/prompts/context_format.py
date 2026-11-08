"""Lossless JSON table encoding for explicitly selected prompt fields."""

import json
from agent_simulation.data.common import require

MIN_TABLE_SAVINGS = 12


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def pack(value, *, table_paths=(), path=()):
    """Encode only named field paths with a meaningful compact JSON size saving."""
    def visit(node, fields):
        if isinstance(node, dict):
            require("$table" not in node, "Reserved shared-input table marker in source")
            return {
                key: visit(child, fields + (key,)) if isinstance(child, (dict, list)) else child
                for key, child in node.items()
            }
        if not isinstance(node, list):
            return node

        rows = [visit(row, fields) for row in node]
        if fields not in table_paths or len(rows) < 2 or not isinstance(rows[0], dict):
            return rows
        keys = list(rows[0])
        if not all(isinstance(row, dict) and list(row) == keys for row in rows):
            return rows
        shared = {
            k: v
            for k, v in rows[0].items()
            if not isinstance(v, (dict, list))
            and not (type(v) is float and v == 0.0)  # JSON distinguishes signed zeros.
            and all(type(row[k]) is type(v) and row[k] == v for row in rows)
        }
        columns = [key for key in keys if key not in shared]
        table = {
            "$table": {
                "columns": columns,
                "shared": shared,
                "rows": [[row[key] for key in columns] for row in rows],
            }
        }
        options = dict(ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        savings = len(json.dumps(rows, **options)) - len(json.dumps(table, **options))
        return table if savings >= MIN_TABLE_SAVINGS else rows

    return visit(value, path)


def unpack(value):
    """Decode a saved model input, including tables emitted by earlier renderers."""
    if isinstance(value, list):
        return [unpack(v) for v in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"$table"}:
        table = value["$table"]
        require(
            all(len(row) == len(table["columns"]) for row in table["rows"]),
            "Invalid shared-input table width",
        )
        return [
            {**table["shared"], **dict(zip(table["columns"], map(unpack, row)))}
            for row in table["rows"]
        ]
    return {k: unpack(v) for k, v in value.items()}
