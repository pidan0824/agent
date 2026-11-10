"""Retrieve active long-term summaries and resolve their cited evidence."""

from collections import defaultdict
from math import log1p
from functools import lru_cache
import json

from agent_simulation.data.common import require

from agent_simulation.agents.memory.evidence import (
    actor_key,
    eligible_evidence,
    evidence_reference,
    experiences,
    segment_keys,
    source_rows,
)
from agent_simulation.agents.memory.versions import active, memory_id
from agent_simulation.agents.retrieval.bge_encoder import encode_query
from agent_simulation.agents.retrieval.history import HISTORY_RETRIEVAL_POLICY
from agent_simulation.agents.retrieval.retrieval_text import query_text
from agent_simulation.agents.retrieval.similarity import related_memories


def available_entries(state, key, count, encoder_id):
    entries = state["long_term"].get(key, [])
    require(all(e["owner"] == key for e in entries), "Foreign long-term memory")
    entries = active(entries, count, state["simulation_step"], state.get("clock", {}).get("time"))
    require(
        not entries or bool(encoder_id),
        "Long-term retrieval requires an explicit embedding model and identity",
    )
    require(
        all(e["embedding_model"] == encoder_id for e in entries), "Memory embedding model differs"
    )
    return entries


def retrieve(entries, query, count, embedder, *, limit=5):
    """Rank entries already scoped by available_entries()."""
    if not entries:
        return []
    require(callable(embedder), "Long-term retrieval requires an explicit embedding model")
    return related_memories(
        encode_query(embedder, query),
        entries,
        [e["embedding"] for e in entries],
        [e["event_count"] for e in entries],
        count,
        limit=limit,
    )


def mean_counts(rows):
    """Measure traffic and purchase volume per observation point on a log scale."""
    return [
        log1p(sum(row[field] for row in rows) / len(rows))
        for field in ("exposure_records", "purchase_records")
    ]


def rank_by_seller_evidence(entries, context, sources):
    """Fuse semantic and comparable same-item evidence ranks within their existing slots."""
    current = defaultdict(list)
    for event in context.get("short_term_memory", []):
        for row in event["content"]:
            current[row["item_id"]].append(row)
    means = {item: mean_counts(rows) for item, rows in current.items()}
    comparable = []
    for slot, entry in enumerate(entries):
        unique = {
            key: row
            for source in entry["source_ids"]
            for key, row in source_rows("seller", sources[source])
            if key[1] in means
        }
        rows_by_item = defaultdict(list)
        for (_, item, _), row in unique.items():
            rows_by_item[item].append(row)
        if rows_by_item:
            distance = sum(
                abs(a - b)
                for item, rows in rows_by_item.items()
                for a, b in zip(mean_counts(rows), means[item])
            ) / len(rows_by_item)
            comparable.append((slot, entry, distance))
    # Reuse the factual retriever's equal-weight rank fusion. Ties favor measured similarity.
    from agent_simulation.agents.retrieval.ranking import average_ranks

    numeric_ranks = average_ranks([-d for _, _, d in comparable])
    constant = HISTORY_RETRIEVAL_POLICY["rank_constant"]
    scored = [
        (1 / (constant + slot + 1) + 1 / (constant + rank), -distance, -slot, entry)
        for (slot, entry, distance), rank in zip(comparable, numeric_ranks)
    ]
    ranked = list(entries)
    for (slot, _, _), (_, _, _, entry) in zip(comparable, sorted(scored, reverse=True)):
        ranked[slot] = entry
    return ranked[:5]


def retrieve_long_term_memory(state, role, actor, query, embedder, encoder_id, evidence=(), *, events=None, tokenizer=None):
    key = actor_key(state, role, actor)
    if events is None:
        events = experiences(state, role, actor)
    selected, sources = select_memories(state, role, key, query, embedder, encoder_id, events, evidence)
    return present_memories(selected, sources, role, key, query, tokenizer=tokenizer)


def select_memories(state, role, key, query, embedder, encoder_id, events, evidence):
    """Return ranked entries and their authorized sources before prompt formatting."""
    count = len(events)
    entries = available_entries(state, key, count, encoder_id)
    compare_counts = (
        role == "seller" and isinstance(query, dict) and bool(query.get("short_term_memory"))
    )
    selected = retrieve(
        entries,
        query_text(role, query),
        count,
        embedder,
        limit=None if compare_counts else 5,
    )
    sources = {
        e["source_id"]: e for e in [*events, *eligible_evidence(evidence, key, count, state)]
    }
    require(
        all(source in sources for e in selected for source in e["source_ids"]),
        "Memory cites unavailable or foreign history",
    )
    if compare_counts:
        selected = rank_by_seller_evidence(selected, query, sources)
    return selected, sources


MEMORY_INPUT_TOKEN_LIMIT = 1024


@lru_cache(maxsize=1)
def _memory_tokenizer():
    from agent_simulation import PROJECT_ROOT
    from agent_simulation.inference.requests import load_tokenizer
    return load_tokenizer(PROJECT_ROOT / "models/Qwen2.5-3B-Instruct", local_files_only=True)


def budget_memories(memories, *, tokenizer=None, limit=MEMORY_INPUT_TOKEN_LIMIT):
    """Keep complete ranked summaries with citations; stored memories are untouched."""
    if not memories:
        return []
    tokenizer = _memory_tokenizer() if tokenizer is None else tokenizer

    def fits(entries):
        text = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
        return len(tokenizer.encode(text, add_special_tokens=False)) <= limit

    if fits(memories):
        return list(memories)
    selected = []
    for memory in memories:
        if fits([*selected, memory]):
            selected.append(memory)
    return selected


def present_memories(selected, sources, role, key, query, *, tokenizer=None):
    """Resolve citations against the exact retrieved facts shown for this decision."""
    positions = {}
    if selected and isinstance(query, dict):
        segments = query.get("retrieved_context", {}).get("history", {}).get("segments", [])
        for number, segment in enumerate(segments, 1):
            for row, identity in enumerate(segment_keys(role, segment), 1):
                positions.setdefault(identity, []).append((number, row))
    return budget_memories([
        {
            "memory_id": memory_id(e),
            "owner": key,
            "simulation_step": e["generated_step"],
            "content": e["content"],
            "generated_logical_time": e.get("generated_logical_time"),
            "evidence_kind": "model_summary_or_hypothesis",
            "source_ids": e["source_ids"],
            "evidence_times": [
                evidence_reference(sources[source], role, positions) for source in e["source_ids"]
            ],
        }
        for e in selected
    ], tokenizer=tokenizer)
