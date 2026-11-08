"""Read and commit each actor's private memory in the current market database."""

from contextlib import nullcontext
import hashlib
import json

from agent_simulation.data.common import require
from agent_simulation.agents.memory.evidence import (
    QUERY_TOOLS, actor_key, context_parts, experience_counts, experiences,
)
from agent_simulation.agents.memory.reflection import REFLECTION_TRIGGER, update as update_memory
from agent_simulation.agents.memory.validation import ReflectionDraftError
from agent_simulation.agents.retrieval.long_term import retrieve_long_term_memory
from agent_simulation.agents.retrieval import bge_encoder


class EvidenceStore:
    def __init__(self, db):
        self.db = db
        with db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS memory_evidence ("
                "source_id TEXT PRIMARY KEY, owner TEXT NOT NULL, "
                "event_count INTEGER NOT NULL, available_step INTEGER NOT NULL, "
                "payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS memory_evidence_owner "
                "ON memory_evidence(owner,event_count,available_step)"
            )

    def record(self, owner, state, event_count, kind, content):
        entry = {
            "owner": owner,
            "origin": "context",
            "kind": kind,
            "event_count": event_count,
            "available_step": state["simulation_step"],
            "available_logical_time": state.get("clock", {}).get("time"),
            "historical_cutoff_exclusive": state["historical_cutoff_exclusive"],
            "content": content,
        }
        identity = {
            k: v
            for k, v in entry.items()
            if k not in {"event_count", "available_step", "available_logical_time"}
        }
        encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, allow_nan=False)
        source = "context:" + hashlib.sha256(encoded.encode()).hexdigest()[:24]
        entry["source_id"] = source
        self.db.execute(
            "INSERT OR IGNORE INTO memory_evidence VALUES (?,?,?,?,?)",
            (
                source,
                owner,
                event_count,
                state["simulation_step"],
                json.dumps(entry, ensure_ascii=False, allow_nan=False),
            ),
        )
        return source

    def eligible(self, owner, count, step, *, source_ids=None, logical_time=None):
        if source_ids is not None and not source_ids:
            return []
        clause, parameters = "", [owner, count, step]
        if source_ids is not None:
            clause = " AND source_id IN (SELECT value FROM json_each(?))"
            parameters.append(json.dumps(list(source_ids)))
        rows = self.db.execute(
            "SELECT payload FROM memory_evidence WHERE owner=? AND event_count<=? "
            "AND available_step<=?" + clause + " ORDER BY event_count,available_step,source_id",
            parameters,
        )
        return [
            entry
            for (payload,) in rows
            for entry in [json.loads(payload)]
            if (
                logical_time is None
                or entry.get("available_logical_time") is None
                or entry["available_logical_time"] <= logical_time
            )
        ]


class LongTermMemoryService:
    def __init__(self, runtime):
        self.runtime = runtime
        self.embedder = None
        self.encoder_id = None
        self._failed_tick_batches = {}
        self.evidence = EvidenceStore(runtime.db)

    def observe_context(self, role, actor, context, state):
        """Persist authorized factual inputs without advancing the market or event count."""
        self._observe(role, actor, context_parts(role, context), state)

    def observe_tool(self, role, actor, name, args, result, state):
        """Call only after successful environment authorization and execution."""
        if name in QUERY_TOOLS:
            self._observe(role, actor, ((name, {"arguments": args, "result": result}),), state)

    def _observe(self, role, actor, parts, state):
        key = actor_key(state, role, actor)
        count = sum(experience_counts(state, role, actor))
        db = self.runtime.db
        # A nested SQLite connection context would commit the caller's transaction.
        with (nullcontext() if db.in_transaction else db):
            for kind, content in parts:
                self.evidence.record(key, state, count, kind, content)

    def bind_encoder(self, embedder, encoder_id):
        require(
            callable(embedder) and type(encoder_id) is str and bool(encoder_id.strip()),
            "Explicit embedding model and identity required",
        )
        require(
            all(
                e["embedding_model"] == encoder_id
                for entries in self.runtime.snapshot()["long_term"].values()
                for e in entries
            ),
            "Memory embedding model differs",
        )
        self.embedder = embedder
        self.encoder_id = encoder_id

    def ensure_encoder(self):
        """Load no generative model; bind local Chinese BGE when retrieval is needed."""
        if self.embedder is None:
            self.bind_encoder(*bge_encoder.default_encoder())

    def context(self, role, actor, query, *, state=None, tokenizer=None):
        state = self.runtime.snapshot() if state is None else state
        key = actor_key(state, role, actor)
        entries = state["long_term"].get(key, [])
        if not entries:
            return []
        self.ensure_encoder()
        sources = {s for e in entries for s in e["source_ids"]}
        events = experiences(state, role, actor)
        evidence = self.evidence.eligible(
            key,
            len(events),
            state["simulation_step"],
            source_ids=sources,
            logical_time=state.get("clock", {}).get("time"),
        )
        return retrieve_long_term_memory(state, role, actor, query, self.embedder, self.encoder_id,
                                         evidence, events=events, tokenizer=tokenizer)

    def status(self, role, actor, *, state=None):
        state = self.runtime.snapshot() if state is None else state
        key = actor_key(state, role, actor)
        initial, settled = experience_counts(state, role, actor)
        count = initial + settled
        processed = state["memory_processed"].get(key, 0)
        return {
            "initial_events": initial,
            "settled_events": settled,
            "total_events": count,
            "processed_events": processed,
            "generation_due": REFLECTION_TRIGGER.due(count, processed),
        }

    def can_attempt(self, role, actor, *, state=None):
        """Retry a rejected factual batch on a later Tick, shared by both Agents."""
        state = self.runtime.snapshot() if state is None else state
        key = actor_key(state, role, actor)
        if "clock" not in state:
            return True
        batch = (state["clock"]["tick"], state["memory_processed"].get(key, 0))
        return self._failed_tick_batches.get(key) != batch

    def update(self, role, actor, *, generate, model_id, embedder=None, encoder_id=None):
        """Explicit model callback boundary; failures leave the prior memory unchanged."""
        require(
            (embedder is None) == (encoder_id is None),
            "Provide both embedding callback and identity",
        )
        state, revision = self.runtime.freeze()
        key = actor_key(state, role, actor)
        processed = state["memory_processed"].get(key, 0)
        try:
            if embedder is None:
                self.ensure_encoder()
                embedder, encoder_id = self.embedder, self.encoder_id
            else:
                self.bind_encoder(embedder, encoder_id)
            # Snapshot availability prevents later context leaking into earlier batches.
            profile, facts = self.runtime.inputs.reflection_evidence(role, actor, state=state)
            self.observe_context(role, actor, facts, state)
            result = update_memory(
                state,
                role,
                actor,
                profile,
                generate,
                embedder,
                model_id,
                encoder_id,
                self.evidence.eligible(
                    key,
                    REFLECTION_TRIGGER.next_boundary(processed),
                    state["simulation_step"],
                    logical_time=state.get("clock", {}).get("time"),
                ),
            )
            with self.runtime.store.transaction():
                require(self.runtime.revision == revision, "State changed during generation")
                self.runtime.store.write(state)
                cursor = self.runtime.db.execute(
                    "INSERT INTO calls(role,actor,name,arguments,result) VALUES(?,?,?,?,?)",
                    (
                        role,
                        actor,
                        "generate_long_term",
                        json.dumps({"model": model_id, "encoder": encoder_id}),
                        json.dumps(result, ensure_ascii=False),
                    ),
                )
            return {**result, "audit_id": cursor.lastrowid}
        except Exception as error:
            if isinstance(error, ReflectionDraftError):
                require(self.runtime.revision == revision, "State changed during generation")
                if "clock" in state:
                    self._failed_tick_batches[key] = (state["clock"]["tick"], processed)
            with self.runtime.db:
                cursor = self.runtime.db.execute(
                    "INSERT INTO calls(role,actor,name,arguments,error,result) VALUES(?,?,?,?,?,?)",
                    (
                        role,
                        actor,
                        "generate_long_term",
                        json.dumps({"model": model_id, "encoder": encoder_id}),
                        f"{type(error).__name__}: {error}",
                        json.dumps({"request": error.request, "raw_output": error.output}, ensure_ascii=False)
                        if isinstance(error, ReflectionDraftError) else None,
                    ),
                )
            if isinstance(error, ReflectionDraftError):
                error.audit_id = cursor.lastrowid
            raise
