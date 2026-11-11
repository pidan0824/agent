"""Schedule, generate and validate evidence-backed long-term experience updates."""

import json
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from uuid import uuid4

import numpy as np
from agent_simulation.data.common import require

from agent_simulation.agents.memory.evidence import (
    actor_key,
    eligible_evidence,
    experiences,
    reflection_profile,
    retrieval_text,
    with_settled_offers,
)
from agent_simulation.agents.memory.validation import (
    ReflectionDraftError,
    quality_issues,
    read_draft,
)
from agent_simulation.agents.memory.versions import memory_id, retire
from agent_simulation.agents.retrieval.bge_encoder import encode_query
from agent_simulation.agents.retrieval.long_term import (
    available_entries,
    retrieve,
)
from agent_simulation.agents.retrieval.retrieval_text import event_text
from agent_simulation.agents.retrieval.similarity import cosine_scores, related_memories
from agent_simulation.inference.client import ModelClient, ModelOutputError
from agent_simulation.inference.requests import prepare_request



SUMMARY_PROMPTS = {
    'customer': (
        'Reflect as this Customer on product preferences, recurring interests, search and '
        'comparison habits, purchases, and possible price sensitivity. Repeated searches can '
        'suggest interest. Claims about stable habits, satisfaction, motives or budget need '
        'specific evidence. Price sensitivity needs observed prices and shopping choices. Use '
        'purchase_evidence for actual purchases by item and shop when checking a prior claim '
        'of never buying.'
    ),
    'seller': (
        'Reflect as this Seller on pricing and demand, differences among your products, '
        'competitive position, and how the platform may allocate exposure. Compare the same '
        'item across observations before describing a price change. Keep counts within their '
        'recorded windows and treat purchase records as events. Treat possible price effects '
        'and allocation rules as working hypotheses. Avoid claims about unobserved costs, '
        'inventory, competitor performance or the actual ranking algorithm.'
    ),
}

SUMMARY_INSTRUCTION = """
Make one update useful for future decisions. Separate observations from tentative beliefs,
consider conflicting evidence, and scope conclusions to the cited products, shops and
observations. Preserve relevant names and IDs. Keep historical indices and simulation
steps separate and use them only for ordering. Keep prices as raw values without assigning
a currency unit or converting them.

Choose one action.
- no_update: No useful new insight or correction.
- add: Record a new insight unrelated to correcting a supplied memory.
- revise: Replace a supplied memory, retaining what the evidence still supports.
- retract: Withdraw a misleading or unsupported memory without replacement.

For add or revise, write one concise English paragraph and cite 1–3 supporting E cards in
evidence_refs, including a new one. For revise or retract, identify the affected M entries
in memory_refs. Return only JSON matching the schema.
"""

EVIDENCE_ORDER = (
    '\n\nReview all evidence cards marked new=true and check previous memories for conflicts. '
    'Use older cards as relevant background.\n'
)


@dataclass(frozen=True)
class ReflectionTrigger:
    """Process each complete evidence batch once, including initialized history."""

    batch_size: int = 10

    def due(self, total, processed):
        return total - processed >= self.batch_size

    def next_boundary(self, processed):
        return processed + self.batch_size


REFLECTION_TRIGGER = ReflectionTrigger()


def _document_vectors(embedder, texts):
    batch = getattr(embedder, "encode_many", None)
    return batch(texts) if callable(batch) else [embedder(text) for text in texts]


def retrieve_prior_events(prior, role, profile, query, embedder, limit=5, *, query_encoder=None):
    """Select optional background only from already processed facts. Prefer recent ties."""
    if not prior:
        return [], {}
    documents = [event_text(role, event, profile.get("products", [])) for event in prior]
    query_vector = encode_query(embedder, query) if query_encoder is None else query_encoder(query)
    scores = cosine_scores(query_vector, _document_vectors(embedder, documents))
    ranked = sorted((i for i in scores if scores[i] > 0), key=lambda i: (scores[i], i), reverse=True)
    chosen = ranked[:limit]
    return [prior[i] for i in chosen], {prior[i]["source_id"]: documents[i] for i in chosen}


def update(state, role, actor, profile, generate, embedder, model_id, encoder_id, evidence=()):
    """Generate and persist a memory revision using the supplied evidence and model."""
    key = actor_key(state, role, actor)
    events = experiences(state, role, actor)
    processed = state["memory_processed"].get(key, 0)
    require(
        REFLECTION_TRIGGER.due(len(events), processed), "Fewer than ten unprocessed memory events"
    )
    require(
        callable(generate) and callable(embedder), "Generation and embedding models are required"
    )
    require(
        isinstance(model_id, str) and bool(model_id.strip()), "Missing generation model identity"
    )
    require(
        type(encoder_id) is str and bool(encoder_id.strip()), "Missing embedding model identity"
    )
    events = events[: REFLECTION_TRIGGER.next_boundary(processed)]
    count = len(events)
    step = state["simulation_step"]
    logical_time = state.get("clock", {}).get("time")
    memories = state["long_term"].get(key, [])
    existing_counts = {entry["event_count"] for entry in memories}
    require(len(existing_counts) == len(memories) and count not in existing_counts,
            "Duplicate long-term memory ID")

    new_events = events[processed:]
    new_ids = [event["source_id"] for event in new_events]
    query = "\n".join(event_text(role, event, profile.get("products", [])) for event in new_events)

    @cache
    def query_encoder(text):
        return encode_query(embedder, text)

    background, documents = retrieve_prior_events(
        events[:processed], role, profile, query, embedder, query_encoder=query_encoder
    )
    selected = [*new_events, *background]
    if role == "customer":
        selected = with_settled_offers(selected, state["feedback"])
    prior_entries = available_entries(state, key, count, encoder_id)
    previous = retrieve(prior_entries, query, count, query_encoder)
    previous_content = [
        {
            "memory_id": memory_id(e),
            **{k: e[k] for k in ("generated_step", "content", "source_ids")},
        }
        for e in previous
    ]
    supplemental = eligible_evidence(evidence, key, count, state)
    available_sources = {e["source_id"] for e in events}
    available_sources.update(e["source_id"] for e in supplemental)
    require(
        all(s in available_sources for e in previous for s in e["source_ids"]),
        "Previous memory cites unavailable evidence",
    )
    if supplemental:
        texts = [retrieval_text(e) for e in supplemental]
        supplemental = related_memories(
            query_encoder(query),
            supplemental,
            _document_vectors(embedder, texts),
            [e["event_count"] for e in supplemental],
            count,
        )
    request = [
        {
            "role": "system",
            "content": SUMMARY_PROMPTS[role]
            + EVIDENCE_ORDER
            + SUMMARY_INSTRUCTION,
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "actor_role": role,
                    "profile": reflection_profile(role, profile, selected),
                    "experiences": selected,
                    "new_evidence_ids": new_ids,
                    "supplemental_evidence": supplemental,
                    "previous_long_term_memory": previous_content,
                },
                ensure_ascii=False,
            ),
        },
    ]
    try:
        raw = generate(request)
    except ModelOutputError as error:
        raise ReflectionDraftError(error.raw_output, str(error), request=request) from error
    result, action = read_draft(raw, [*selected, *supplemental], previous, request=request)
    issues = quality_issues(result, new_ids, prior_entries, action=action)
    if issues:
        raise ReflectionDraftError(raw, "; ".join(issues), request=request)
    entry = None
    if action in ("add", "revise"):
        vector = np.asarray(embedder(result["content"]), dtype=float)
        require(vector.ndim == 1 and vector.size > 0 and np.isfinite(vector).all(),
                "Invalid summary embedding")
        entry = {
            **deepcopy(result),
            "generated_step": step,
            "generated_logical_time": logical_time,
            "event_count": count,
            "generator": model_id,
            "embedding": vector.tolist(),
            "embedding_model": encoder_id,
            "owner": key,
            "parent_memory_event_counts": [e["event_count"] for e in previous],
        }
        if getattr(generate, 'memory_format', None):
            entry['content_format'] = generate.memory_format
        memories = [*memories, entry]
    if action in ("revise", "retract"):
        memories = retire(
            memories,
            result["supersedes"],
            count=count,
            step=step,
            replacement=memory_id(entry) if entry else None,
            source_ids=result["source_ids"],
            logical_time=logical_time,
            context_source_ids=new_ids,
        )
    if action != "no_update":
        state["long_term"][key] = memories
    # Only successful generations consume a factual batch; failures propagate unchanged.
    state["memory_processed"][key] = count
    return {
        "saved": entry,
        "operation": action,
        "status": "saved" if entry else "retired" if result.get("supersedes") else "no_update",
        "request": request,
        "raw_output": raw,
        "processed_events": count,
        "retrieval_query": query,
        "retrieval_documents": documents,
        "review_context_source_ids": new_ids,
    }


class SummaryGenerator:
    """Generate one explicit operation, then resolve its local reference IDs."""

    memory_format = 'role_experience_summary_v1'

    def __init__(self, model: ModelClient):
        self.model = model

    def __call__(self, messages):
        payload = json.loads(messages[1]["content"])
        from agent_simulation.agents.memory.protocol import prepare, decode
        visible, schema, references = prepare(payload)
        model_messages = [messages[0], {'role': 'user', 'content': json.dumps(visible, ensure_ascii=False)}]
        request, _ = prepare_request(
            model_messages, [], self.model.tokenizer, max_output_tokens=self.model.max_new_tokens
        )
        request["memory_generation_method"] = "explicit-memory-operation-v2"
        request["json_schema"] = schema
        reply = self.model.complete(request, call_id="memory:" + uuid4().hex)
        if (reply.get("role") != "assistant" or reply.get("tool_calls", []) != []
                or not isinstance(reply.get("content"), str)):
            raise ModelOutputError("Expected an assistant text response", raw_output=reply)
        try:
            return decode(reply['content'], schema, references)
        except ValueError as error:
            raise ModelOutputError(str(error), raw_output=reply['content']) from error
