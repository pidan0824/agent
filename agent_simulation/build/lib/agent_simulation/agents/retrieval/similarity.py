"""Exact authorized cosine scoring without rebuilding a vector database per query."""

import operator
import numpy as np

_OPERATORS = {"==": operator.eq, "!=": operator.ne, "<": operator.lt,
              "<=": operator.le, ">": operator.gt, ">=": operator.ge}


class PreparedVector:
    """One validated immutable document vector and its exact float64 norm."""
    __slots__ = ("values", "norm")

    def __init__(self, values):
        vector = np.array(values, dtype=float, copy=True)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError("Invalid document embedding")
        vector.setflags(write=False)
        self.values, self.norm = vector, np.linalg.norm(vector)


def cosine_scores(query, embeddings, *, metadata=None, predicates=()):
    """Filter by metadata before validating documents and computing cosine scores."""
    q = np.asarray(query, dtype=float)
    if q.ndim != 1 or not q.size or not np.isfinite(q).all():
        raise ValueError("Invalid query embedding")
    if metadata is not None and len(metadata) != len(embeddings):
        raise ValueError("Embedding metadata alignment differs")
    checks = []
    for key, op, value in predicates:
        if op not in _OPERATORS:
            raise ValueError("Unsupported metadata predicate: " + str(op))
        checks.append((key, _OPERATORS[op], value))
    vectors, ids = [], []
    for i, embedding in enumerate(embeddings):
        fields = metadata[i] if metadata is not None else {}
        if not all(key in fields and fields[key] is not None and test(fields[key], value)
                   for key, test, value in checks):
            continue
        prepared = isinstance(embedding, PreparedVector)
        v = embedding.values if prepared else np.asarray(embedding, dtype=float)
        if v.shape != q.shape or (not prepared and not np.isfinite(v).all()):
            raise ValueError("Invalid or incompatible embedding")
        ids.append(i)
        vectors.append((v, embedding.norm if prepared else np.linalg.norm(v),
                        embedding if prepared else None))
    if not vectors:
        return {}
    # Match the previous scalar cosine arithmetic, including exact score ties.
    # Reusing already encoded vectors avoids TextNode/index/retriever creation.
    norm = np.linalg.norm(q)
    result, shared = {}, {}
    for i, (vector, other, prepared) in zip(ids, vectors):
        # Identical text can occur at many dates. Reuse its score, not its facts.
        if prepared is not None and prepared in shared:
            result[i] = shared[prepared]
        else:
            result[i] = float(np.dot(q, vector) / (norm * other)) if norm and other else 0.0
            if prepared is not None:
                shared[prepared] = result[i]
    return result


# Adapted from PolicySim commit 045446b7dd5120437233e75da0e997d68d7c267a.
# Original sources: src/agents/memory/twitter_memory.py and src/agent/base_agent.py.
# License declaration: docs/third-party-notices.txt.
from copy import deepcopy


def related_memories(
    query, contents, embeddings, times, cnt, *, metadata=None, predicates=(), limit=5
):
    """Keep older relevant evidence; event age is not a calibrated forgetting rate."""
    if not (len(contents) == len(embeddings) == len(times)):
        raise ValueError("Memory content/vector/time alignment differs")
    if any(t > cnt for t in times):
        raise ValueError("Cannot retrieve future memory")
    scores = cosine_scores(query, embeddings, metadata=metadata, predicates=predicates)
    ranked = sorted(scores, key=lambda i: (scores[i], times[i], -i), reverse=True)
    return [deepcopy(contents[i]) for i in ranked[:limit]]
