"""Render shopping history, offers and responses as Customer messages."""

from agent_simulation.agents.prompts.instructions import (
    CUSTOMER_STEPS, FACT_RULES, HISTORY_NOTE, VALUE_NOTE, TABLE_NOTE, MEMORY_NOTE, TITLE_NOTE,
    QUERY_TASK,
    QUERY_OUTPUT,
    RESPONSE_SYSTEM,
)

import json
from copy import deepcopy
from agent_simulation.data.common import require
from agent_simulation.agents.prompts.context_format import answer_message, canonical, compact_json, pack, unpack

REPRESENTATION = "customer-shared-attributes-v3"

ATTRIBUTE_GROUPS = {
    "sellers": ("seller_id", "seller_name"),
    "brands": ("brand_id", "brand_name"),
    "categories": tuple(
        f"category_level{level}_{suffix}" for level in (1, 2, 3) for suffix in ("id", "name")
    ),
}

TABLE_PATHS = frozenset({
    ("behavioral_summary", "item_details"),
    ("displayed_items",),
    ("display_categories",),
    ("short_term_memory", "item_attributes"),
    *(("short_term_memory", "attribute_sets", name) for name in ATTRIBUTE_GROUPS),
})

QUERY_RUNTIME_TASK = 'Task\nYou are an e-commerce customer. Use your profile, shopping history and experience to choose search, wait or exit this Episode. For search, generate a Chinese query and preserve original product and brand names.\n'

QUERY_RUNTIME_OUTPUT = '\nOutput\nChoose one action for this global Tick. To search, return a JSON object with exactly one key, query, containing your specific Chinese product search query. To wait until the next Tick, return {"action":"no_action"}. To leave this Episode, return {"action":"exit"}. Decide from your profile, current index and history. Search leads to a separate displayed-list response; exit can be chosen at the next Tick. Empty click or purchase sets do not mean exit. Return only the chosen action, without placeholders or explanation.'


def _history_events(state):
    yield from state["short_term_memory"]["search_history"]
    for segment in state.get("retrieved_context", {}).get("history", {}).get("segments", []):
        yield from segment.get("records", [])


def _attribute_ref(attributes, records, indices):
    """Keep the first occurrence of an attribute record and return its index."""
    key = canonical(attributes)
    if key not in indices:
        indices[key] = len(records)
        records.append(attributes)
    return indices[key]


def factor_history(state):
    """Share identical historical snapshots, never overwrite an item's past attributes."""
    result = deepcopy(state)
    memory = result.get("short_term_memory") if isinstance(result, dict) else None
    if not isinstance(memory, dict) or not isinstance(memory.get("search_history"), list):
        return result
    reserved = {"history_encoding", "item_attributes", "attribute_sets"}
    require(not reserved.intersection(memory), "Reserved shared-history fields in source")
    items, item_index = [], {}
    groups = {name: [] for name in ATTRIBUTE_GROUPS}
    group_index = {name: {} for name in ATTRIBUTE_GROUPS}
    for event in _history_events(result):
        if not isinstance(event, dict):
            continue
        for field in ("clicked_items", "purchased_items"):
            if field not in event:
                continue
            reference = field.replace("_items", "_item_refs")
            require(reference not in event, "Reserved history reference in source")
            products = event[field]
            require(
                isinstance(products, list) and all(isinstance(p, dict) for p in products),
                "Historical products must be records",
            )
            refs = []
            for product in products:
                key = canonical(product)
                if key not in item_index:
                    item_index[key] = len(items)
                    record = dict(product)
                    for name, fields in ATTRIBUTE_GROUPS.items():
                        require(name + "_ref" not in record, "Reserved attribute reference")
                        values = {k: record.pop(k) for k in fields if k in record}
                        if values:
                            record[name + "_ref"] = _attribute_ref(values, groups[name], group_index[name])
                    items.append(record)
                refs.append(item_index[key])
            event.pop(field)
            event[reference] = refs
    memory.update(
        history_encoding="shared-attributes-v2", item_attributes=items, attribute_sets=groups
    )
    return result


def unfactor_history(state):
    """Restore complete event snapshots, including unknown values and changed prices."""
    result = deepcopy(state)
    memory = result.get("short_term_memory") if isinstance(result, dict) else None
    if not isinstance(memory, dict) or "history_encoding" not in memory:
        return result
    version = memory.pop("history_encoding")
    require(version == "shared-attributes-v2", "Unknown history encoding")
    items, groups = memory.pop("item_attributes"), memory.pop("attribute_sets")
    require(
        isinstance(items, list) and set(groups) == set(ATTRIBUTE_GROUPS),
        "Invalid shared-history dictionaries",
    )

    def resolve(records, index):
        require(
            isinstance(records, list) and type(index) is int and 0 <= index < len(records),
            "Invalid shared-history reference",
        )
        return deepcopy(records[index])

    restored = []
    for product in items:
        record = dict(product)
        for name in ATTRIBUTE_GROUPS:
            if name + "_ref" in record:
                values = resolve(groups[name], record.pop(name + "_ref"))
                require(not set(record).intersection(values), "Conflicting shared attributes")
                record.update(values)
        restored.append(record)
    for event in _history_events(result):
        if not isinstance(event, dict):
            continue
        for field in ("clicked_items", "purchased_items"):
            reference = field.replace("_items", "_item_refs")
            if reference in event:
                require(field not in event, "Conflicting historical products")
                event[field] = [resolve(restored, i) for i in event.pop(reference)]
    return result


def _wire(value, path=()):
    return compact_json(pack(value, table_paths=TABLE_PATHS, path=path))


def _share_titles(value, background):
    """Reference identical titles; every event, quote and other attribute stays in place."""
    details = (background or {}).get("item_details", [])
    titles = {(p["item_id"], p["item_title"]): i for i, p in enumerate(details)
              if "item_id" in p and "item_title" in p}

    def visit(node):
        if isinstance(node, list):
            return [visit(x) for x in node]
        if not isinstance(node, dict):
            return node
        result = {key: visit(child) for key, child in node.items()}
        key = (node.get("item_id"), node.get("item_title"))
        if key in titles:
            result["item_title"] = {"title_ref": titles[key]}
        return result

    if not titles:
        return value
    return visit(value)


def _encoded_history(history, background=None):
    compact = factor_history(history)
    variants = [(history, 0), (compact, len(HISTORY_NOTE) + 1)]
    if background and background.get("item_details"):
        variants.extend([(_share_titles(history, background), len(TITLE_NOTE) + 1),
                         (_share_titles(compact, background), len(TITLE_NOTE) + len(HISTORY_NOTE) + 2)])
    encoded = []
    for value, note_size in variants:
        fields = {key: _wire(part, (key,)) for key, part in value.items()}
        encoded.append((sum(map(len, fields.values())) + note_size, fields))
    return min(encoded, key=lambda pair: pair[0])[1]


def _prepare_display(state):
    """Serialize offers once per representation and share repeated category tuples."""
    items = state["displayed_items"]
    path = ("displayed_items",)
    original = _wire(items, path)
    if len(items) < 2:
        return {"displayed_items": original}
    require("display_categories" not in state, "Reserved displayed category dictionary")
    groups, indices, offers = [], {}, []
    for item in items:
        require("category_ref" not in item, "Reserved displayed category reference")
        attrs = {k:item[k] for k in ATTRIBUTE_GROUPS["categories"] if k in item}
        if not attrs:
            offers.append(item)
            continue
        reference = _attribute_ref(attrs, groups, indices)
        offers.append({k: v for k, v in item.items() if k not in attrs} | {"category_ref": reference})
    if groups:
        encoded_offers = _wire(offers, path)
        encoded_groups = _wire(groups, ("display_categories",))
        extra_field = len(',"display_categories":') + len(encoded_groups)
        if len(encoded_offers) + extra_field < len(original):
            return {"displayed_items": encoded_offers, "display_categories": encoded_groups}
    return {"displayed_items": original}


def prepare_history(state, *, recent_summary=...):
    """Freeze serialized history once for all batches of one Response decision."""
    history = {key: value for key, value in state.items()
               if key in ("short_term_memory", "retrieved_context")}
    from agent_simulation.agents.context import context_model
    model = context_model()
    if model is not None:
        recent = history.get("short_term_memory", {})
        summary = (model.summarize(recent.get("search_history", []), scope="Customer recent history",
                                  compact={k: json.loads(v) for k, v in
                                           _encoded_history({"short_term_memory": recent}).items()})
                   if recent_summary is ... else recent_summary)
        if summary is not None:
            history["short_term_memory"] = {**recent, "search_history": [], "summary": summary}
        retrieved = history.get("retrieved_context", {})
        older = retrieved.get("history", {})
        # Citations use segment and record positions. Keep their numbering intact.
        cited = any(evidence.get("same_observations_in_retrieved_history")
                    for memory in state.get("long_term_memory", ())
                    for evidence in memory.get("evidence_times", ()))
        if not cited:
            summary = model.summarize(older.get("segments", []), scope="Customer retrieved older history",
                                      compact={k: json.loads(v) for k, v in _encoded_history({
                                          "short_term_memory": {"search_history": []},
                                          "retrieved_context": {"history": older}}).items()})
            if summary is not None:
                history["retrieved_context"] = {**retrieved, "history": {**older, "segments": [], "summary": summary}}
    return _encoded_history(history, state.get("behavioral_summary"))


def encode_input(state, *, prepared_history=None):
    require(isinstance(state, dict), "Customer context must be an object")
    require("$table" not in state, "Reserved shared-input table marker in source")
    history = prepare_history(state) if prepared_history is None else prepared_history
    display = _prepare_display(state) if "displayed_items" in state else {}
    if display and (state.get("behavioral_summary") or {}).get("item_details"):
        shared = _prepare_display({**state, "displayed_items": _share_titles(
            state["displayed_items"], state["behavioral_summary"])})
        if sum(map(len, shared.values())) + len(TITLE_NOTE) + 1 < sum(map(len, display.values())):
            display = shared
    encoded = {**history, **display}
    return "{" + ",".join(
        json.dumps(key, ensure_ascii=False) + ":" + (encoded[key] if key in encoded else _wire(value, (key,)))
        for key, value in {**state, **display}.items()
    ) + "}"


def decode_input(text):
    state = unfactor_history(unpack(json.loads(text)))
    if isinstance(state, dict) and "display_categories" in state:
        groups = state.pop("display_categories")
        for item in state["displayed_items"]:
            if "category_ref" in item:
                i = item.pop("category_ref")
                require(type(i) is int and 0 <= i < len(groups), "Invalid displayed category reference")
                require(not set(groups[i]).intersection(item), "Conflicting displayed category")
                item.update(deepcopy(groups[i]))
    details = (state.get("behavioral_summary") or {}).get("item_details", [])

    def restore_titles(node):
        if isinstance(node, list):
            for child in node:
                restore_titles(child)
        elif isinstance(node, dict):
            title = node.get("item_title")
            if isinstance(title, dict) and "title_ref" in title:
                node["item_title"] = details[title["title_ref"]]["item_title"]
            for child in node.values():
                restore_titles(child)

    restore_titles(state)
    return state


def messages(
    state, target=None, *, response=False, query_output=QUERY_OUTPUT, query_task=QUERY_TASK,
    prepared_history=None,
):
    from agent_simulation.agents.context.shopping import project_customer
    visible = project_customer(state)
    content = encode_input(visible, prepared_history=prepared_history)
    notes = TABLE_NOTE if '"$table":{' in content else VALUE_NOTE
    if '"history_encoding":' in content:
        notes += " " + HISTORY_NOTE
    if '"title_ref"' in content:
        notes += " " + TITLE_NOTE
    if visible.get("long_term_memory"):
        notes += " " + MEMORY_NOTE
    if '"display_categories":' in content:
        notes += " Current offers use category_ref to index display_categories (zero-based)."
    result = [
        {
            "role": "system",
            "content": (
                RESPONSE_SYSTEM
                if response
                else query_task + FACT_RULES + CUSTOMER_STEPS + query_output
            )
            + "\n"
            + notes,
        },
        {
            "role": "user",
            "content": content,
        },
    ]
    if target is not None:
        result.append(answer_message(target))
    return result


def customer_decision_messages(
    context, *, response=False, prepared_history=None,
):
    return messages(
        context,
        response=response, prepared_history=prepared_history,
        query_output=(QUERY_RUNTIME_OUTPUT if "decision_time" in context else QUERY_OUTPUT),
        query_task=(QUERY_RUNTIME_TASK if "decision_time" in context else QUERY_TASK),
    )
