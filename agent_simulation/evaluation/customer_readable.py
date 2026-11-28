"""Readable fixed Response views; canonical provenance stays outside model messages."""
from collections import Counter
from copy import deepcopy
import json

from agent_simulation.data.common import require

PROTOCOL = "readable-v1"
STAGES = ("A", "B", "C")


def readable_view(context):
    """Render stored facts directly, without hash joins in the model input."""
    from agent_simulation.evaluation.customer_evidence import expand_sources
    c = expand_sources(deepcopy(context))
    facts = c["facts"]
    exclude = {"facts", "history_events", "candidate_context", "context_protocol",
               "readable_stage", "readable_history_policy", "_folded_snapshot"}
    view = {k: deepcopy(v) for k, v in c.items() if k not in exclude}
    view.update(context_protocol=PROTOCOL, readable_stage=c["readable_stage"],
                readable_history_policy=c.get("readable_history_policy", "frozen"))
    events = []
    for row in c["history_events"]:
        event = {"event_id": row["event_id"].replace("history:", "event:", 1),
                 "time_index": row["time_index"]}
        for field, ref in row["fact_refs"].items():
            event[field] = deepcopy(facts[ref]["value"])
        events.append(event)
    view["history_events"] = events
    snapshot = next((f for f in facts.values() if f["kind"] == "background" and
                     f["field"] == "recent_item_ids"), None)
    if snapshot:
        snap = {"observed_time_index": snapshot["time_index"],
                "scope": "历史快照；与搜索事件重叠，不相加。次数仅限保留列表，缺少记录不等于从未交互。"}
        if view["readable_history_policy"] == "frozen":
            snap.update(deepcopy(snapshot["value"]))
        snap["products"] = [deepcopy(f["value"]) for f in facts.values()
                            if f["kind"] == "background" and f["field"] == "item_history"]
        if "_folded_snapshot" in c:
            snap["folded"] = deepcopy(c["_folded_snapshot"])
        view["prior_interacted_products"] = snap
    products = []
    for row in c["candidate_context"]:
        refs = row["fact_refs"]
        p = {"item_id": row["item_id"], "attributes": {
            field: deepcopy(facts[ref]["value"]) for field, ref in refs.items()},
            "current_price_status": row.get("current_price_status", "unknown"),
            "historical_price_status": row.get("historical_price_status", "unknown")}
        if "price_raw" in refs:
            p["historical_price_time_index"] = facts[refs["price_raw"]]["time_index"]
        else:
            p["historical_price"] = "未知"
        if c["readable_stage"] == "A":
            if "snapshot_interactions" in row:
                p["snapshot_interactions"] = {k: deepcopy(v) for k, v in
                    row["snapshot_interactions"].items() if k != "fact_ref"}
            p["related_history"] = []
            for link in row.get("related_history", []):
                f = facts[link["fact_ref"]]
                value = f["value"]
                historical = value if f["field"] == "item_history" else next(
                    item for item in value if item["item_id"] == link["item_id"])
                p["related_history"].append({
                    "item_id": link["item_id"], "item_title": historical.get("item_title"),
                    "source": ("snapshot" if f["kind"] == "background" else
                               f["source_id"].replace("history:", "event:", 1)),
                    "behavior_field": f["field"],
                    **{k: deepcopy(v) for k, v in link.items()
                       if k not in ("fact_ref", "item_id")}})
        products.append(p)
    view["candidate_context"] = products
    return view


def reference_map(context, view=None):
    """Readable references resolve only to facts that are actually visible."""
    view = view or readable_view(context)
    mapping = {}
    snapshot = view.get("prior_interacted_products", {})
    visible_ids = {p["item_id"] for p in snapshot.get("products", [])}
    for field in ("recent_clicked_item_ids", "recent_purchased_item_ids"):
        visible_ids.update(snapshot.get(field, []))
    for ref, fact in context["facts"].items():
        kind, field = fact["kind"], fact["field"]
        aliases = []
        if kind == "query":
            aliases.append("query")
        elif kind == "history":
            aliases.append(fact["source_id"].replace("history:", "event:", 1))
        elif kind == "product":
            aliases.append("item:" + str(fact["item_id"]))
        elif kind == "background" and field == "item_history":
            aliases.append("item:" + str(fact["item_id"]))
        elif kind == "background" and field == "recent_item_ids":
            if view.get("readable_history_policy") == "frozen":
                aliases.append("snapshot")
            for ids in fact["value"].values():
                aliases.extend("item:" + str(i) for i in set(ids) & visible_ids)
        for alias in set(aliases):
            mapping.setdefault(alias, []).append(ref)
    return mapping


def validate_sources(context):
    """Validate every visible source, including uncited prices."""
    actor, cutoff = context["evidence_actor_id"], context["history_cutoff_exclusive"]
    for fact in context["facts"].values():
        if fact["kind"] in ("history", "background"):
            require(fact.get("actor_id") == actor, "Readable source actor mismatch")
            require(type(fact.get("time_index")) is int and
                    0 <= fact["time_index"] < cutoff, "Readable non-prior history")
        if fact["kind"] == "product" and fact["field"] == "price_raw":
            require(type(fact.get("time_index")) is int and
                    0 <= fact["time_index"] < cutoff, "Readable price needs a strictly-prior timestamp")


def mark_readable(context, stage="A", policy="frozen"):
    require(stage in STAGES and policy in ("frozen", "budgeted"), "Unknown readable stage or history policy")
    result = deepcopy(context)
    result.update(context_protocol=PROTOCOL, readable_stage=stage, readable_history_policy=policy)
    validate_sources(result)
    return result


def semantic_values(context):
    """Inventory for the independent A-view comparison (values, not transport IDs)."""
    return {(f["kind"], f["source_id"], f["field"]): f["value"]
            for f in context["facts"].values()}


def view_values(view):
    result = {}
    if "query" in view:
        result[("query", "current_query", "query")] = view["query"]
    for event in view["history_events"]:
        for field in ("query", "clicked_items", "purchased_items"):
            if field in event:
                result[("history", event["event_id"].replace("event:", "history:", 1), field)] = event[field]
    for p in view["candidate_context"]:
        for field, value in p["attributes"].items():
            result[("product", "item:" + str(p["item_id"]), field)] = value
    snapshot = view.get("prior_interacted_products")
    if snapshot:
        source = f'background:{view["evidence_actor_id"]}:{snapshot["observed_time_index"]}'
        lists = {f: snapshot[f] for f in ("recent_clicked_item_ids", "recent_purchased_item_ids") if f in snapshot}
        result[("background", source, "recent_item_ids")] = lists
        for p in snapshot["products"]:
            result[("background", source + ":item:" + str(p["item_id"]), "item_history")] = p
    return result


def budget_readable(context, *, tokenizer, budget, event_limit=10):
    """Fit whole events/products with exact serialized history tokens; never use labels."""
    require(type(budget) is int and budget >= 0, "Invalid readable history budget")
    full = mark_readable(context, context["readable_stage"], "budgeted")
    facts = full["facts"]
    snapshots = [f for f in facts.values() if f["kind"] == "background" and f["field"] == "recent_item_ids"]
    lists = snapshots[0]["value"] if snapshots else {}
    # Snapshot sequences have no individual timestamps. Preserve source sequence order;
    # do not claim it is independently verified event recency.
    ids = list(dict.fromkeys(lists.get("recent_clicked_item_ids", []) +
                             lists.get("recent_purchased_item_ids", [])))
    candidates = {p["item_id"] for p in full["candidate_context"]}
    details = {f["item_id"]: ref for ref, f in facts.items()
               if f["kind"] == "background" and f["field"] == "item_history"}
    ordered = sorted(details, key=lambda i: (i not in candidates, ids.index(i) if i in ids else len(ids), i))
    event_ids = [e["event_id"] for e in sorted(full["history_events"],
                  key=lambda e: (e["time_index"], e["event_id"]), reverse=True)][:event_limit]
    def render(chosen_events, chosen_items):
        c = deepcopy(full)
        c["facts"] = {ref: f for ref, f in facts.items()
                      if f["kind"] != "history" and
                      not (f["kind"] == "background" and f["field"] == "item_history")}
        c["facts"].update({ref: f for ref, f in facts.items()
                          if f["kind"] == "history" and f["source_id"] in chosen_events})
        c["facts"].update({details[i]: facts[details[i]] for i in chosen_items})
        c["history_events"] = [e for e in full["history_events"] if e["event_id"] in chosen_events]
        # Rebuild only links to the selected visible history.
        from agent_simulation.evaluation.customer_background import candidate_links
        candidate_links(c)
        omitted = [i for i in ids if i not in chosen_items]
        cats = Counter(facts[details[i]]["value"].get("category_level2_name", "未知")
                       if i in details else "未知" for i in omitted)
        c["_folded_snapshot"] = {"omitted_products": len(omitted), "categories": dict(sorted(cats.items())),
                                "order": "原始快照序列，无逐商品时间"}
        c["selected_history_events"] = len(c["history_events"])
        if "snapshot_detail_selection" in c:
            c["snapshot_detail_selection"] = {"available": len(details), "shown": len(chosen_items)}
        return c
    def cost(c):
        v = readable_view(c)
        fields = {key: v[key] for key in ("history_events", "prior_interacted_products") if key in v}
        return len(tokenizer.encode(json.dumps(fields, ensure_ascii=False, separators=(",", ":")),
                                    add_special_tokens=False))
    selected_events, selected_items = [], []
    value = render([], [])
    require(cost(value) <= budget, "Readable history budget cannot hold empty history metadata")
    # Candidate same-ID history is selected uniformly, never by target label or count.
    units = [("item", i) for i in ordered if i in candidates]
    units += [("event", e) for e in event_ids]
    units += [("item", i) for i in ordered if i not in candidates]
    for kind, ident in units:
        es = selected_events + [ident] if kind == "event" else selected_events
        its = selected_items + [ident] if kind == "item" else selected_items
        proposed = render(es, its)
        if cost(proposed) <= budget:
            value, selected_events, selected_items = proposed, es, its
    value["readable_history_tokens"] = cost(value)
    return value


def instructions(stage, *, concise=True):
    from agent_simulation.agents.prompts.customer_ekb import concise_ekb_instructions, ekb_instructions
    from agent_simulation.evaluation.customer_evidence import EVIDENCE_NOTE
    text = (concise_ekb_instructions() if concise else ekb_instructions()).replace(EVIDENCE_NOTE, "")
    text = text.replace("Cite the\nsupporting facts;", "Cite the supporting visible references;")
    text = text.replace("Write the explanation first, then evidence_refs: each preference\ncomparison needs both the historical B/H facts and the candidate P attribute IDs.",
                        "Write the explanation first, then cite the supporting visible references in evidence_refs.")
    if stage == "C":
        text = text.replace("Past interaction counts indicate familiarity, not a rule to choose the most\nfrequent item.",
            "同一商品的历史点击和购买是重要依据，需结合当前查询和候选之间的差异判断；历史交互不等于本次一定点击。")
    return text + """
价格未知的商品不能参与价格高低比较，但仍然可以被选择。历史价格不代表当前报价。
历史记录可支持推断，不能机械地选次数最多的商品；不要把短列表构造当作行为规则。
evidence_refs use only visible readable IDs: query, event:<event_id number>,
item:<item_id>, or snapshot when the full snapshot lists are shown.
Cite events for historical searches/actions and item IDs for supplied product attributes
or snapshot interactions. A citation establishes source availability, not a true preference.
"""


def readable_check(trajectory):
    """Bind the sidecar to the actual view, map aliases, then reuse legacy provenance checks."""
    from agent_simulation.evaluation.customer_evidence import check_evidence
    from agent_simulation.agents.schemas import strict_json
    try:
        context = trajectory["evidence_context"]
        require(context.get("context_protocol") == PROTOCOL, "Missing readable sidecar")
        validate_sources(context)
        view = readable_view(context)
        require(any(m.get("role") == "user" and strict_json(m["content"]) == view
                    for m in trajectory["messages"] if str(m.get("content", "")).startswith("{")),
                "Readable transcript differs from source sidecar")
        final = trajectory["messages"][-1]
        value = strict_json(final["content"])
        require(final["role"] == "assistant" and value.get("rationale") == trajectory["rationale"] and
                value.get("action") == trajectory["final_action"], "Readable original transcript mismatch")
        from agent_simulation.agents.schemas import validate_arguments
        from agent_simulation.evaluation.customer_evidence import EVIDENCE_SCHEMA
        validate_arguments(trajectory["rationale"]["evidence_refs"], EVIDENCE_SCHEMA)
        mapping = reference_map(context, view)
        mapped = deepcopy(trajectory["rationale"])
        for stage, refs in mapped["evidence_refs"].items():
            require(len(set(refs)) == len(refs), "Duplicate readable references")
            require(all(ref in mapping for ref in refs), "Unknown readable evidence reference")
            mapped["evidence_refs"][stage] = list(dict.fromkeys(x for ref in refs for x in mapping[ref]))
        canonical = deepcopy(context)
        for field in ("context_protocol", "readable_stage", "readable_history_policy"):
            canonical.pop(field, None)
        record = {k: deepcopy(v) for k, v in trajectory.items()
                  if k not in ("context_protocol", "evidence_context")}
        record.update(rationale=mapped, messages=[
            {"role": "user", "content": json.dumps(canonical, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps({"rationale": mapped,
                "action": trajectory["final_action"]}, ensure_ascii=False)}])
        return check_evidence(record)
    except (ValueError, KeyError, TypeError, StopIteration) as exc:
        return {"accepted": False, "status": "failed", "reasons": ["invalid_readable_evidence"],
                "detail": str(exc)}
