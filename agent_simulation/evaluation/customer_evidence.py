"""Deterministic provenance checks, not a semantic truth judge for EKB prose."""
from copy import deepcopy
import json

import hashlib
STAGES = ("stimulus", "knowledge", "evaluation")
EVIDENCE_SCHEMA = {"type": "object", "properties": {
    stage: {"type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": True}
    for stage in STAGES}, "required": list(STAGES), "additionalProperties": False}
EVIDENCE_NOTE = """
Keep the original stimulus, knowledge and evaluation explanation. In evidence_refs,
list only supplied fact IDs for each stage; use [] when evidence is unavailable.
Never generate field paths, serialized values or copies of the facts.
When fact_layout is present, each facts row follows those columns; fact_sources
holds shared metadata and an optional row metadata object supplies other fields.
Facts with clicked_items describe clicks, purchased_items describe purchases.
A citation proves source availability, not that a preference inference is true.
Mark tentative inferences and unknowns in ordinary EKB prose. Do not infer price
preferences from a click alone, purchases from clicks, or currency from raw prices.
Predict observed behavior. Relevance alone does not establish a click.
"""


def attach_sources(context):
    """Build deterministic IDs for visible facts; replace duplicate raw fact bodies."""
    facts = dict(context.pop("facts", {}))
    actor = context.get("evidence_actor_id")
    def add(prefix, source, field, value, time_index=None, item_id=None, owner=None):
        if value is None:
            return
        identity = json.dumps([source, field, value], ensure_ascii=False, sort_keys=True)
        key = prefix + hashlib.sha256(identity.encode()).hexdigest()[:12]
        facts[key] = {"source_id": source, "field": field, "value": deepcopy(value),
                      "time_index": time_index, "item_id": item_id, "actor_id": owner,
                      "kind": "history" if prefix == "H" else "product" if prefix == "P" else "query"}
    for event in context.pop("observed_history", []):
        source = "history:" + str(event["session_id"])
        for field in ("query", "clicked_items", "purchased_items"):
            if field in event:
                add("H", source, field, event[field], event["time_index"],
                    owner=event.get("user_id", actor))
    for product in context.pop("displayed_items", []):
        for field in ("item_title", "brand_name", "category_level3_name", "price_raw"):
            if field in product:
                add("P", "item:" + str(product["item_id"]), field, product[field],
                    product.get("price_observed_time_index") if field == "price_raw" else None,
                    item_id=product["item_id"])
    if "query" in context:
        add("Q", "current_query", "query", context["query"])
    context["facts"] = facts
    events, products = {}, {}
    for key, fact in facts.items():
        if fact["kind"] == "history":
            event = events.setdefault(fact["source_id"], {"event_id": fact["source_id"],
                "time_index": fact["time_index"], "fact_refs": {}})
            event["fact_refs"][fact["field"]] = key
        elif fact["kind"] == "product":
            item = products.setdefault(fact["item_id"], {"item_id": fact["item_id"],
                "current_price_status": "unknown", "fact_refs": {}})
            item["fact_refs"][fact["field"]] = key
            if fact["field"] == "price_raw":
                fact["price_scope"] = "historical_observation" if fact["time_index"] is not None else "unconfirmed_time"
                fact["not_current_quote"] = True
    context["history_events"] = sorted(events.values(), key=lambda e: (e["time_index"], e["event_id"]))
    context["candidate_context"] = list(products.values())
    return context


FACT_LAYOUT = ["source_id", "field", "value", "extra_metadata_optional"]
FACT_REFS_ENCODING = "derive-from-facts-v1"
FACT_REFS_NOTE = (
    "Fact references are indexed once in facts. For each history event or candidate, "
    "derive its fact_refs by matching source_id or item_id and mapping field to the fact ID."
)


def _fact_ref_indexes(facts):
    """Derive references without changing source metadata or action values."""
    from agent_simulation.data.common import require
    require(isinstance(facts, dict), "Invalid facts for derived references")
    indexes = {"history_events": {}, "candidate_context": {}}
    for key, fact in facts.items():
        require(isinstance(key, str) and key and isinstance(fact, dict), "Invalid fact record")
        kind = fact.get("kind")
        if kind not in ("history", "product"):
            continue
        field = fact.get("field")
        identity = fact.get("source_id" if kind == "history" else "item_id")
        require(isinstance(field, str) and field and
                (isinstance(identity, str) if kind == "history" else type(identity) is int),
                "Invalid derived fact identity")
        refs = indexes["history_events" if kind == "history" else "candidate_context"].setdefault(identity, {})
        require(field not in refs, "Ambiguous derived fact reference")
        refs[field] = key
    return indexes


def _fact_ref_rows(context, indexes, *, encoded):
    """Only canonical, complete indexes can omit their repeated fact references."""
    from agent_simulation.data.common import require
    result = []
    for collection, identity_key in (("history_events", "event_id"), ("candidate_context", "item_id")):
        rows = context.get(collection)
        require(isinstance(rows, list), "Invalid derived fact index")
        seen = set()
        for row in rows:
            require(isinstance(row, dict), "Invalid derived fact index row")
            identity = row.get(identity_key)
            require(isinstance(identity, str) if identity_key == "event_id" else type(identity) is int,
                    "Invalid derived fact index identity")
            require(identity in indexes[collection] and identity not in seen, "Unknown or duplicate fact index identity")
            refs = indexes[collection][identity]
            require("fact_refs" not in row if encoded else row.get("fact_refs") == refs,
                    "Conflicting derived fact references")
            seen.add(identity)
            result.append((row, refs))
        require(seen == set(indexes[collection]), "Incomplete derived fact index")
    return result


def _compact_fact_refs(context):
    try:
        rows = _fact_ref_rows(context, _fact_ref_indexes(context["facts"]), encoded=False)
    except ValueError:
        return  # Preserve nonstandard or ambiguous indexes in the legacy format.
    if not rows:
        return
    from agent_simulation.agents.prompts.context_format import compact_json, MIN_TABLE_SAVINGS
    metadata = {"fact_refs_encoding": FACT_REFS_ENCODING, "fact_refs_note": FACT_REFS_NOTE}
    # Each affected object keeps other fields: replacing braces with one comma
    # gives the exact compact JSON size change, including both metadata fields.
    removed_size = sum(len(compact_json({"fact_refs": refs})) - 1 for _, refs in rows)
    if removed_size - (len(compact_json(metadata)) - 1) < MIN_TABLE_SAVINGS:
        return
    for row, _ in rows:
        row.pop("fact_refs")
    context.update(metadata)


def compact_sources(context):
    """Lossless wire format: source metadata once, unchanged fact IDs and values."""
    from collections import defaultdict
    from agent_simulation.data.common import require
    require(not {"fact_layout", "fact_sources", "fact_refs_encoding", "fact_refs_note"}.intersection(context),
            "Context is already encoded or uses reserved fact fields")
    result = deepcopy(context)
    if "facts" not in result:
        return result
    _compact_fact_refs(result)
    grouped = defaultdict(list)
    for key, fact in result["facts"].items():
        grouped[fact["source_id"]].append((key, fact))
    sources, packed = {}, {}
    for source, entries in grouped.items():
        first = entries[0][1]
        common = {k: v for k, v in first.items()
                  if k not in ("source_id", "field", "value")
                  and all(k in f and f[k] == v for _, f in entries)}
        sources[source] = common
        for key, fact in entries:
            extra = {k: v for k, v in fact.items()
                     if k not in ("source_id", "field", "value") and k not in common}
            packed[key] = [source, fact["field"], fact["value"]]
            if extra:
                packed[key].append(extra)
    result.update(fact_layout=list(FACT_LAYOUT), fact_sources=sources, facts=packed)
    return result


def expand_sources(context):
    """Read both original transcripts and the reversible compact context format."""
    from agent_simulation.data.common import require
    derived_refs = "fact_refs_encoding" in context or "fact_refs_note" in context
    if derived_refs:
        require(context.get("fact_refs_encoding") == FACT_REFS_ENCODING and
                context.get("fact_refs_note") == FACT_REFS_NOTE and "fact_layout" in context,
                "Invalid derived fact reference encoding")
    if "fact_layout" not in context:
        require("fact_sources" not in context, "Missing compact fact layout")
        return context
    require(context["fact_layout"] == FACT_LAYOUT, "Unknown compact fact layout")
    result = deepcopy(context)
    sources = result.pop("fact_sources", None)
    result.pop("fact_layout")
    require(isinstance(sources, dict) and isinstance(result.get("facts"), dict),
            "Invalid compact fact tables")
    facts = {}
    for key, row in result["facts"].items():
        require(isinstance(row, list) and len(row) in (3, 4), "Invalid compact fact row")
        source, field, value = row[:3]
        require(isinstance(source, str) and source in sources and isinstance(field, str),
                "Unknown compact fact source")
        common, extra = sources[source], row[3] if len(row) == 4 else {}
        require(isinstance(common, dict) and isinstance(extra, dict), "Invalid fact metadata")
        require(not (set(common) & set(extra)) and
                not ({"source_id", "field", "value"} & (set(common) | set(extra))),
                "Conflicting compact fact metadata")
        facts[key] = {**common, **extra, "source_id": source, "field": field, "value": value}
    result["facts"] = facts
    if derived_refs:
        for row, refs in _fact_ref_rows(result, _fact_ref_indexes(facts), encoded=True):
            row["fact_refs"] = refs
        result.pop("fact_refs_encoding")
        result.pop("fact_refs_note")
    return result


def encode_context(context):
    return json.dumps(compact_sources(context), ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def check_evidence(trajectory):
    """Recompute from actual transcript; never trust a cached model quality verdict."""
    if trajectory.get("stage") != "response":
        return {"accepted": True, "status": "not_applicable", "reasons": []}
    if trajectory.get("response_style") == "direct":
        from agent_simulation.agents.schemas import strict_json
        try:
            final = trajectory["messages"][-1]
            valid = (final["role"] == "assistant" and not final.get("tool_calls")
                     and strict_json(final["content"]) == trajectory["final_action"])
        except (KeyError, IndexError, TypeError, ValueError):
            valid = False
        return {"accepted": valid, "status": "not_applicable" if valid else "failed",
                "reasons": [] if valid else ["original_transcript_mismatch"]}
    rationale = trajectory.get("rationale", {})
    annotations = rationale.get("evidence_refs")
    if not annotations:
        return {"accepted": False, "status": "unverified", "reasons": ["missing_evidence_refs"]}
    from agent_simulation.agents.schemas import validate_arguments, strict_json
    try:
        validate_arguments(annotations, EVIDENCE_SCHEMA)
    except ValueError:
        return {"accepted": False, "status": "failed", "reasons": ["invalid_evidence_refs"]}
    sources, cutoff, actor = {}, None, None
    reasons = []
    for message in trajectory.get("messages", []):
        if message.get("role") not in ("user", "tool"):
            continue
        try:
            obj = json.loads(message.get("content") or "")
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            try:
                obj = expand_sources(obj)
            except ValueError:
                reasons.append("invalid_fact_encoding")
                continue
            sources.update(obj.get("facts", {}))
            actor = obj.get("evidence_actor_id", actor)
            if "history_cutoff_exclusive" in obj:
                cutoff = obj["history_cutoff_exclusive"]
    final_messages = [m for m in trajectory.get("messages", [])
                      if m.get("role") == "assistant" and not m.get("tool_calls")]
    if final_messages:
        try:
            final = strict_json(final_messages[-1]["content"])
            if final.get("rationale") != rationale or final.get("action") != trajectory.get("final_action"):
                reasons.append("original_transcript_mismatch")
        except (ValueError, TypeError, AttributeError):
            reasons.append("invalid_original_final_json")
    for stage, refs in annotations.items():
        for ref in refs:
            fact = sources.get(ref)
            if fact is None:
                reasons.append("unknown_fact_id")
                continue
            if fact["kind"] == "history":
                if cutoff is None or fact["time_index"] >= cutoff:
                    reasons.append("non_prior_evidence")
                if actor is not None and fact.get("actor_id") != actor:
                    reasons.append("wrong_actor_evidence")
                if fact["field"] not in ("query", "clicked_items", "purchased_items"):
                    reasons.append("invalid_behavior_type")
            if fact["kind"] == "product":
                if fact["source_id"] != "item:" + str(fact["item_id"]):
                    reasons.append("wrong_item_evidence")
                if fact["time_index"] is not None and (cutoff is None or fact["time_index"] >= cutoff):
                    reasons.append("non_prior_price")
    price_flags = price_claim_flags(rationale, sources)
    reasons.extend(price_flags)
    # Optional external/manual review; never manufacture a semantic pass.
    review = trajectory.get("semantic_review", {})
    if review.get("verdict") == "fail":
        reasons.append("semantic_review_failed")
    reasons = sorted(set(reasons))
    return {"accepted": not reasons, "status": "failed" if reasons else "machine_checked",
            "reasons": reasons, "semantic_status": trajectory.get("semantic_review", {}).get("verdict", "not_reviewed"),
            "scope": "code-owned fact references, visibility, time, actor/item identity; prose entailment is not certified"}


def price_claim_flags(rationale, facts):
    """Conservative explicit item/price checks; no claim to verify all natural language."""
    import re
    products = {}
    for fact in facts.values():
        if fact.get("kind") == "product":
            item = products.setdefault(str(fact["item_id"]), [])
            if fact["field"] == "price_raw" and isinstance(fact["value"], (int, float)):
                item.append(float(fact["value"]))
    flags = []
    for stage in STAGES:
        for sentence in re.split(r"(?<!\d)[.!?。](?!\d)", rationale.get(stage, "")):
            if not re.search(r"price|priced|价格|售价", sentence, re.I):
                continue
            ids = [item for item in products if re.search(r"\b"+re.escape(item)+r"\b", sentence)]
            if len(ids) != 1:
                continue  # Ambiguous multi-item prose requires semantic review.
            item = ids[0]
            numbers = [float(n) for n in re.findall(r"(?<!\w)\d+(?:\.\d+)?(?!\w)", sentence)
                       if n != item]
            if not numbers:
                continue
            if not products[item]:
                flags.append("price_claim_for_unknown_item_price")
            elif any(n not in products[item] for n in numbers):
                flags.append("item_price_value_mismatch")
            elif not re.search(r"histor|previous|past|prior|observed|历史|过去|曾|观测", sentence, re.I):
                flags.append("historical_price_presented_without_time_qualification")
    return sorted(set(flags))
