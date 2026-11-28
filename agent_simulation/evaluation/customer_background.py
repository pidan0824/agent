"""Prior snapshot facts and input-only candidate/history associations."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import re
import unicodedata

from agent_simulation.data.common import require

ATTRIBUTES = ("item_title", "brand_name", "category_level2_name",
              "category_level3_name", "seller_id", "seller_name")
UNKNOWN = {"", "unknown", "null", "none", "无品牌", "其他/other", "other"}
SNAPSHOT_NOTE = (
    "Occurrences are in retained snapshot lists, not lifetime totals or extra events. "
    "Lists overlap search history; never add their counts. Zero means absent from "
    "these lists, not never interested. Item details may be selected by the history budget."
)


def known(value):
    return value is not None and str(value).strip().casefold() not in UNKNOWN


def snapshot_facts(summary, actor, cutoff):
    """Preserve both complete ID sequences; details remain separately budgetable."""
    when = summary.get("observed_time_index")
    require(type(actor) is int and type(when) is int and 0 <= when < cutoff,
            "Invalid Customer background owner or cutoff")
    require(summary.get("user_id", actor) == actor, "Foreign Customer background")
    fields = ("recent_clicked_item_ids", "recent_purchased_item_ids")
    lists = {field: deepcopy(summary[field]) for field in fields if field in summary}
    for values in lists.values():
        require(isinstance(values, list) and all(type(i) is int for i in values),
                "Invalid Customer background item IDs")
    if not lists:
        return {}
    clicks, purchases = (Counter(lists.get(field, [])) for field in fields)
    facts = {}
    def add(field, value, item=None):
        source = f"background:{actor}:{when}" + (f":item:{item}" if item is not None else "")
        identity = json.dumps([source, field, value], ensure_ascii=False, sort_keys=True)
        key = "B" + hashlib.sha256(identity.encode()).hexdigest()[:12]
        facts[key] = dict(source_id=source, field=field, value=deepcopy(value),
                          time_index=when, item_id=item, actor_id=actor, kind="background")
    add("recent_item_ids", lists)
    seen = set()
    for product in summary.get("item_details", []):
        item = product.get("item_id")
        require(type(item) is int and item not in seen, "Duplicate or invalid background product")
        seen.add(item)
        if item not in clicks and item not in purchases:
            continue
        value = {"item_id": item, **{f: product[f] for f in ATTRIBUTES if f in product}}
        for field, counts, label in zip(fields, (clicks, purchases), ("click_occurrences", "purchase_occurrences")):
            if field in lists:
                value[label] = counts[item]
        add("item_history", value, item)
    return facts


def _terms(text):
    text = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return {word[i:i + 2] for word in re.findall(r"[^\W_]+", text)
            for i in range(len(word) - 1) if not word[i:i + 2].isdigit()}


def product_match(candidate, past):
    """Describe observed similarities, never infer a preference or desired action."""
    fields = []
    if candidate.get("item_id") == past.get("item_id"):
        fields.append("item_id")
    for field in ("brand_name", "seller_id", "seller_name", "category_level3_name", "category_level2_name"):
        value = candidate.get(field)
        if known(value) and value == past.get(field) and (field != "seller_id" or type(value) is int and value > 0):
            fields.append(field)
    a, b = _terms(candidate.get("item_title")), _terms(past.get("item_title"))
    shared = sorted(a & b)
    lexical = len(shared) / len(a | b) if a or b else 0.
    if len(shared) >= 2 and lexical >= .15:
        fields.append("item_title")
    return fields, lexical


def match_priority(candidate, past):
    fields, lexical = product_match(candidate, past)
    return ("item_id" in fields, "brand_name" in fields,
            bool({"seller_id", "seller_name"} & set(fields)), lexical,
            "category_level3_name" in fields, "category_level2_name" in fields)


def candidate_links(context):
    """Colocate candidate attributes and references to relevant visible history."""
    facts = context["facts"]
    snapshot = next(((key, fact) for key, fact in facts.items()
                     if fact["kind"] == "background" and fact["field"] == "recent_item_ids"), None)
    if snapshot is None:
        return
    key, fact = snapshot
    lists = fact["value"]
    counts = {field: Counter(values) for field, values in lists.items()}
    history = []
    for ref, source in facts.items():
        if source["kind"] == "background" and source["field"] == "item_history":
            history.append((ref, source["value"], source["time_index"]))
        elif source["kind"] == "history" and source["field"] in ("clicked_items", "purchased_items"):
            history.extend((ref, item, source["time_index"]) for item in source["value"])
    query_terms = _terms(context.get("query"))
    for row in context["candidate_context"]:
        attributes = {field: facts[ref]["value"] for field, ref in row["fact_refs"].items()}
        row["attributes"] = attributes
        row["historical_price_status"] = "observed" if "price_raw" in attributes else "unknown"
        row["snapshot_interactions"] = {
            **{label: counts[field][row["item_id"]] for field, label in (
                ("recent_clicked_item_ids", "click_occurrences"),
                ("recent_purchased_item_ids", "purchase_occurrences")) if field in counts},
            "fact_ref": key}
        candidate = {"item_id": row["item_id"], **attributes}
        candidate_terms = _terms(candidate.get("item_title"))
        observed = [(ref, item, when, *product_match(candidate, item))
                    for ref, item, when in history]
        # Repair missing/coarse-only associations without replacing already specific
        # item, brand, shop, leaf-category or title matches.
        query_fallback = not any(set(fields) - {"category_level2_name"}
                                 for _, _, _, fields, _ in observed)
        matches = []
        for ref, item, when, fields, lexical in observed:
            shared = candidate_terms & _terms(item.get("item_title")) if query_fallback else set()
            # Long titles can share query-relevant features across category labels
            # despite a low whole-title Jaccard score. Keep the literal basis visible.
            query_shared = sorted(shared & query_terms) if len(shared) >= 2 else []
            if query_shared and "item_title" not in fields:
                fields.append("item_title")
            if not fields:
                continue
            priority = ("item_id" in fields, "brand_name" in fields,
                        bool({"seller_id", "seller_name"} & set(fields)), len(query_shared), lexical,
                        "category_level3_name" in fields, "category_level2_name" in fields,
                        when, -item["item_id"])
            matches.append((priority, ref, item["item_id"], fields, query_shared))
        related, seen = [], set()
        for _, ref, item_id, fields, query_shared in sorted(matches, key=lambda m: m[0], reverse=True):
            identity = (ref, item_id)
            if identity in seen:
                continue
            seen.add(identity)
            related.append({"fact_ref": ref, "item_id": item_id, "matched_attributes": fields,
                            **({"matched_query_terms": query_shared} if query_shared else {})})
            if len(related) == 3:
                break
        row["related_history"] = related
    context["background_history_scope"] = SNAPSHOT_NOTE
