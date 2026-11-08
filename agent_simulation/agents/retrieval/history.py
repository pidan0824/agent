"""Retrieve older factual history with Chinese BGE and reciprocal rank fusion.

RRF reference: https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf"""

from agent_simulation.agents.retrieval.ranking import (
    reciprocal_rank_scores, freshness_factor, FRESHNESS_POLICY,
)

DEFAULT_HISTORY_ENCODER = object()

HISTORY_RETRIEVAL_POLICY = {
    "default_mode": "bge",
    "fusion": "reciprocal_rank_fusion",
    "rank_constant": 60,
    "signal_weights": "equal",
    "tied_signal_ranks": "average",
    "missing_numeric_signal": "no_vote",
    "unknown_category": "no_vote",
    "fused_ties": "semantic_then_numeric_then_clock_recency",
    "calibrated": False,
    "freshness": FRESHNESS_POLICY,
    "seller_deduplication": "exact_same_item_clock_observation_records_including_indices",
}


def history_encoder(encoder):
    """Load local BGE lazily; None explicitly requests the structured-only ablation."""
    if encoder is DEFAULT_HISTORY_ENCODER:
        from agent_simulation.agents.retrieval.bge_encoder import default_encoder

        return default_encoder()[0]
    return encoder


from collections import defaultdict, deque, OrderedDict
from itertools import groupby
from copy import deepcopy
import math
from agent_simulation.data.common import require
from agent_simulation.agents.retrieval.retrieval_text import event_text


LIMIT = 5
SEGMENT_POINTS = 3


class CachedHistoryEncoder:
    """Actor-scoped exact text vectors; changed text cannot reuse a stale vector."""
    def __init__(self, embedder, max_entries=2048):
        self.embedder, self.max_entries = embedder, max_entries
        self.documents = OrderedDict()
        self._resolved = embedder

    def _base(self):
        if self._resolved is DEFAULT_HISTORY_ENCODER:
            self._resolved = history_encoder(self.embedder)
        return self._resolved

    def prepared_many(self, texts):
        from agent_simulation.agents.retrieval.similarity import PreparedVector
        texts = list(texts)
        result = {text: self.documents[text] for text in set(texts) if text in self.documents}
        missing = list(dict.fromkeys(text for text in texts if text not in result))
        if missing:
            encoder = self._base()
            vectors = (encoder.encode_many(missing) if callable(getattr(encoder, 'encode_many', None))
                       else [encoder(text) for text in missing])
            for text, vector in zip(missing, vectors, strict=True):
                result[text] = PreparedVector(vector)
        for text in texts:
            self.documents[text] = result[text]
            self.documents.move_to_end(text)
            if len(self.documents) > self.max_entries:
                self.documents.popitem(last=False)
        return [result[text] for text in texts]

    def encode_many(self, texts):
        return [entry.values for entry in self.prepared_many(texts)]

    def __call__(self, text):
        return self.encode_many([text])[0]

    def encode_query(self, text):
        encoder = self._base()
        return getattr(encoder, "encode_query", encoder)(text)


def _rank(candidates, query, embedder, limit, *, owner=None, cutoff=None, simulation_before=None, deduplicate=False):
    require(type(limit) is int and limit >= 0, "History limit must be nonnegative")
    use_bge = embedder is not None
    if candidates:
        embedder = history_encoder(embedder)
    if embedder is not None and candidates:
        from agent_simulation.agents.retrieval.similarity import cosine_scores

        groups = defaultdict(list)
        for entry in candidates:
            groups[entry.get("query_text", query)].append(entry)
        predicates = [
            (key, op, value)
            for key, op, value in (
                ("owner", "==", owner),
                ("observed_index", "<", cutoff),
                ("settled_step", "<", simulation_before),
            )
            if value is not None
        ]
        for text, group in groups.items():
            query_vector = getattr(embedder, "encode_query", embedder)(text)
            metadata = [
                {
                    "owner": e.get("owner_id"),
                    "observed_index": (
                        e["end_inclusive"] if e["clock"] == "observed_time_index" else -1
                    ),
                    "settled_step": (
                        e["end_inclusive"] if e["clock"] == "simulation_settled_step" else -1
                    ),
                }
                for e in group
            ]
            scores = cosine_scores(
                query_vector,
                (embedder.prepared_many([e["search_text"] for e in group])
                 if isinstance(embedder, CachedHistoryEncoder)
                 else embedder.encode_many([e["search_text"] for e in group])
                 if callable(getattr(embedder, "encode_many", None))
                 else [embedder(e["search_text"]) for e in group]),
                metadata=metadata,
                predicates=predicates,
            )
            for i, score in scores.items():
                group[i]["match"]["semantic_similarity"] = score
        candidates = [e for e in candidates if "semantic_similarity" in e["match"]]
    signals = [[e["match"]["entity_match"] for e in candidates]]
    if use_bge:
        signals.append([e["match"]["semantic_similarity"] for e in candidates])
    signals.append(
        [
            (
                -e["match"]["numeric_distance"]
                if e["match"].get("numeric_distance") is not None
                else None
            )
            for e in candidates
        ]
    )
    constant = HISTORY_RETRIEVAL_POLICY["rank_constant"]
    scores = reciprocal_rank_scores(signals, constant)
    for entry, score in zip(candidates, scores):
        before = simulation_before if entry["clock"] == "simulation_settled_step" else cutoff
        freshness = freshness_factor(entry["end_inclusive"], before, entry["clock"])
        entry["match"].update(base_retrieval_score=score, freshness_factor=freshness,
                              retrieval_score=score * freshness)
    # No item/category tier can override the fused score. Separate clocks only break ties.
    candidates.sort(
        key=lambda e: (
            e["match"]["retrieval_score"],
            e["match"].get("semantic_similarity", 0),
            (
                -e["match"]["numeric_distance"]
                if e["match"].get("numeric_distance") is not None
                else -math.inf
            ),
            e["clock"] == "simulation_settled_step",
            e["end_inclusive"],
            -e.get("item_id", 0),
        ),
        reverse=True,
    )
    selected, seen = [], set()
    for entry in candidates:
        if len(selected) == limit:
            break
        if deduplicate:
            signature = (entry["item_id"], entry["clock"], tuple(
                tuple(row.get(k) for k in ("index", "price_raw", "price_status", "exposure_records",
                                           "click_records", "purchase_records"))
                for row in entry["records"]))
            if signature in seen:
                continue
            seen.add(signature)
        selected.append(entry)
    for entry in selected:
        entry.pop("search_text")
        entry.pop("query_text", None)
    return {
        "kind": "retrieved_factual_history",
        "ranking": "rrf_bge" if use_bge else "rrf_structured",
        "rank_constant": constant,
        "eligible_segments": len(candidates),
        "returned_segments": len(selected),
        "segments": selected,
    }


def _features(rows):
    def price(row):
        value = row.get("price_raw")
        return (
            value
            if row.get("price_status") == "observed" and value is not None and value > 0
            else None
        )

    def rate(part, key):
        exposure = sum(r["exposure_records"] for r in part)
        return sum(r[key] for r in part) / exposure if exposure else None

    first, last = price(rows[0]), price(rows[-1])
    midpoint = len(rows) // 2

    def trend(key):
        if not midpoint:
            return None
        before, after = rate(rows[:midpoint], key), rate(rows[midpoint:], key)
        return after - before if before is not None and after is not None else None

    return (
        last,
        rate(rows, "click_records"),
        rate(rows, "purchase_records"),
        (last - first) / first if len(rows) > 1 and first and last else None,
        trend("click_records"),
        trend("purchase_records"),
    )


def _feature_text(values):
    names = (
        "最近参考价",
        "点击/曝光",
        "购买/曝光",
        "区间价格相对变化",
        "点击率前后差",
        "购买率前后差",
    )
    return "；".join(
        f'{name}：{value if value is not None else "未知"}' for name, value in zip(names, values)
    )


def _distance(a, b):
    terms = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x is not None and y is not None:
            terms.append(abs(math.log(x / y)) if i == 0 and x > 0 and y > 0 else abs(x - y))
    return sum(terms) / len(terms) if terms else None


def seller_segment(item, clock, rows):
    """Prepare one full or partial factual segment, independent of a decision."""
    field = "time_index" if clock == "observed_time_index" else "settled_step"
    owner = rows[0]["seller_id"]
    require(all(r["seller_id"] == owner and r["item_id"] == item for r in rows),
            "Mixed Seller segment")
    require(all(a[field] < b[field] for a, b in zip(rows, rows[1:])),
            "Seller segment must be chronological")
    return {
        "seller_id": owner,
        "item_id": item,
        "clock": clock,
        "start_inclusive": rows[0][field],
        "end_inclusive": rows[-1][field],
        "records": [
            {
                k: r[k]
                for k in (
                    "price_raw",
                    "price_status",
                    "exposure_records",
                    "click_records",
                    "purchase_records",
                )
            }
            | {"index": r[field]}
            for r in rows
        ],
        "features": _features(rows),
    }


class SellerItemWindow:
    """Keep the latest observations per item and segment evicted rows within each clock."""

    def __init__(self, item, recent_n, stride=SEGMENT_POINTS):
        require(type(recent_n) is int and recent_n > 0, "Seller window must be positive")
        require(type(stride) is int and 1 <= stride <= SEGMENT_POINTS,
                "Segment stride must be within the window length")
        self.item = item
        self.stride = stride
        self._recent = deque(maxlen=recent_n)
        self._segments = []
        self._recent_segments = None
        self._recent_features = None
        self._clock = None
        self._partial = []
        self._pending = False
        self._partial_segment = None

    def append(self, clock, row):
        evicted = None
        if len(self._recent) == self._recent.maxlen:
            old_clock, old_row = evicted = self._recent.popleft()
            if old_clock != self._clock:
                if self._pending:
                    self._segments.append(self._pending_segment())
                self._clock, self._partial = old_clock, []
            self._partial.append(old_row)
            self._pending, self._partial_segment = True, None
            if len(self._partial) == SEGMENT_POINTS:
                self._segments.append(self._pending_segment())
                self._partial = self._partial[self.stride:]
                self._pending, self._partial_segment = False, None
        self._recent.append((clock, row))
        self._recent_segments = self._recent_features = None
        return evicted

    def _pending_segment(self):
        if self._partial_segment is None:
            self._partial_segment = seller_segment(self.item, self._clock, self._partial)
        return self._partial_segment

    @property
    def first_recent_row(self):
        return self._recent[0][1]

    def recent_rows(self):
        return [row for _, row in self._recent]

    def recent_features(self):
        if self._recent_features is None:
            self._recent_features = _features(self.recent_rows())
        return self._recent_features

    def segments(self, *, include_recent=False):
        yield from self._segments
        if self._pending:
            yield self._pending_segment()
        if include_recent:
            if self._recent_segments is None:
                self._recent_segments = []
                for clock, entries in groupby(self._recent, key=lambda entry: entry[0]):
                    rows = [row for _, row in entries]
                    for start in range(0, len(rows), SEGMENT_POINTS):
                        self._recent_segments.append({
                            **seller_segment(self.item, clock, rows[start:start + SEGMENT_POINTS]),
                            "recent": True,
                        })
            yield from self._recent_segments


def category_id(product, level):
    key = f"category_level{level}"
    value = product.get(key + "_id")
    if value is None or value <= 0 or product.get(key + "_name") == "UNKNOWN":
        return None
    return value


def _seller_search_text(segment, profile):
    """Reuse factual text until the segment or its visible product title changes."""
    title = str(profile.get("item_title", ""))
    if segment.get("_search_title") != title:
        segment["_search_title"] = title
        segment["_search_text"] = title + " " + _feature_text(segment["features"])
    return segment["_search_text"]


def _rank_seller_history(history, focus_ids, embedder, limit):
    recent, profiles = history.recent, history.profiles
    seller_id, cutoff = history.seller_id, history.cutoff
    simulation_before = history.simulation_before
    focal = set(focus_ids) or set(recent)
    require(focal <= set(recent), "Unknown focus product")
    current = {item: (history.window_features[item]) for item in focal}
    queries = {item: str(profiles[item].get("item_title", "")) + " 当前经营 "
               + _feature_text(current[item]) for item in focal}
    categories = history.categories
    ordered_focal = sorted(focal)
    related = {item: [i for i in ordered_focal if None not in categories[item]
                     and categories[item] == categories[i]] for item in recent}
    candidates = []
    # Repeated operating patterns share distances, but remain distinct evidence.
    cross_item_distances = {}
    for segment, base in history.candidates:
        item, clock, features = segment["item_id"], segment["clock"], segment["features"]
        if item in focal and segment.get("recent", False):
            continue  # Target recent facts are already shown directly.
        if item in focal:
            distance = _distance(features, current[item])
            entity = 2
            query_item = item
        else:
            same_category = related[item]
            entity = int(bool(same_category))
            pattern = features[1:]
            if pattern not in cross_item_distances:
                cross_item_distances[pattern] = {
                    i: _distance((None, *pattern), (None, *current[i][1:])) for i in focal
                }
            distances = cross_item_distances[pattern]
            query_item = min(
                same_category or ordered_focal,
                key=lambda i: (distances[i] if distances[i] is not None else math.inf, i),
            )
            distance = distances[query_item]
        candidates.append({
            **base,
            "match": {"entity_match": entity, "numeric_distance": distance},
            "query_text": queries[query_item],
        })
    result = _rank(
        candidates,
        "",
        embedder,
        limit,
        owner=seller_id,
        deduplicate=True,
        cutoff=cutoff,
        simulation_before=simulation_before,
    )
    # Returned samples must not expose mutable references to cached segment records.
    return deepcopy(result)


def prepare_seller_history(
    observed,
    *,
    seller_id,
    cutoff,
    recent_n,
    simulated=(),
    simulation_before=None,
    segment_stride=SEGMENT_POINTS,
):
    """Prepare full same-shop history; each query excludes its own recent facts.

    Cross-item references include recent and older segments. All records precede
    their clock's cutoff; relative prices are only compared within an item.
    """
    require(
        type(segment_stride) is int and 1 <= segment_stride <= SEGMENT_POINTS,
        "Segment stride must be within the window length",
    )
    groups = defaultdict(list)
    profiles = {}
    for clock, events, field, bound in (
        ("observed_time_index", observed, "time_index", cutoff),
        ("simulation_settled_step", simulated, "settled_step", simulation_before),
    ):
        require(bound is not None or not events, "Missing simulation cutoff")
        for row in events:
            require(row["seller_id"] == seller_id, "Foreign history owner")
            if row[field] >= bound:
                continue
            groups[row["item_id"]].append((clock, row[field], row))
            if clock == "observed_time_index":
                prior = profiles.get(row["item_id"])
                if prior is None or row[field] > prior[field]:
                    profiles[row["item_id"]] = row
    recent, segments = {}, []
    for item, events in groups.items():
        events.sort(key=lambda e: (e[0] != "observed_time_index", e[1]))
        window = SellerItemWindow(item, recent_n, segment_stride)
        for clock, _, row in events:
            window.append(clock, row)
        recent[item] = window.recent_rows()
        segments.extend(window.segments(include_recent=True))
    return PreparedSellerHistory(segments, recent, profiles, seller_id, cutoff, simulation_before)


class PreparedSellerHistory:
    """Authorized history prefix; callers freeze its records until the review ends."""

    def __init__(self, segments, recent, profiles, seller_id, cutoff, simulation_before=None,
                 *, window_features=None):
        self.segments, self.recent, self.profiles = segments, recent, profiles
        self.seller_id, self.cutoff, self.simulation_before = seller_id, cutoff, simulation_before
        self._encoder = None
        self.window_features = (window_features if window_features is not None else
                                {item: _features(window) for item, window in recent.items()})
        self.categories = {item: tuple(category_id(profile, n) for n in (1, 2, 3))
                           for item, profile in profiles.items()}
        # Ownership and time are invariant across all target queries in this prefix.
        for item, window in recent.items():
            profile = profiles[item]
            require(profile["seller_id"] == seller_id and profile["item_id"] == item
                    and profile["time_index"] < cutoff, "Foreign or future Seller profile")
            for row in window:
                require(row["seller_id"] == seller_id and row["item_id"] == item, "Foreign Seller window")
                field, bound = (("settled_step", simulation_before) if "settled_step" in row
                                else ("time_index", cutoff))
                require(bound is not None and row[field] < bound, "Future Seller window")
        self.candidates = []
        for segment in segments:
            item, clock = segment["item_id"], segment["clock"]
            bound = cutoff if clock == "observed_time_index" else simulation_before
            require(segment["seller_id"] == seller_id and item in recent, "Foreign Seller segment")
            require(bound is not None and segment["end_inclusive"] < bound, "Future Seller segment")
            self.candidates.append((segment, {
                "owner_id": seller_id, "item_id": item, "clock": clock,
                "start_inclusive": segment["start_inclusive"],
                "end_inclusive": segment["end_inclusive"], "records": segment["records"],
                "evidence_kind": "observed_fact" if clock.startswith("observed") else "simulated_fact",
                "search_text": _seller_search_text(segment, profiles[item]),
            }))

    def retrieve(self, focus_ids=(), *, embedder=DEFAULT_HISTORY_ENCODER, limit=LIMIT):
        if embedder is not None and not isinstance(embedder, CachedHistoryEncoder):
            if self._encoder is None or self._encoder.embedder is not embedder:
                self._encoder = CachedHistoryEncoder(embedder)
            embedder = self._encoder
        return _rank_seller_history(self, focus_ids, embedder, limit)


def seller_history(observed, *, seller_id, cutoff, recent_n, focus_ids=(),
                   simulated=(), simulation_before=None, embedder=DEFAULT_HISTORY_ENCODER,
                   limit=LIMIT, segment_stride=SEGMENT_POINTS):
    return prepare_seller_history(observed, seller_id=seller_id, cutoff=cutoff,
        recent_n=recent_n, simulated=simulated, simulation_before=simulation_before,
        segment_stride=segment_stride).retrieve(focus_ids, embedder=embedder, limit=limit)


# Customer candidates grow across decisions; each query ranks against its current recent window.

class PreparedCustomerHistory:
    """Incremental factual pool for one Customer, with explicit correction invalidation."""

    def __init__(self, user_id, recent_n):
        require(type(recent_n) is int and recent_n > 0, "Invalid Customer window")
        self.user_id, self.recent_n = user_id, recent_n
        self._observed, self._simulated = [], []
        self._recent = deque()
        self.candidates, self._categories = [], []
        self._encoder = None

    def update(self, observed, simulated, *, cutoff, simulation_before=None, append_only=False):
        """Sync a snapshot; trusted chronological producers may skip prefix comparisons.

        append_only requires unchanged existing records. Runtime snapshots keep
        correction detection; the SFT stream owns and only appends to its lists.
        """
        require(type(cutoff) is int and cutoff > 0, "Invalid historical cutoff")
        require(simulation_before is not None or not simulated, "Missing simulation cutoff")
        require(simulation_before is None or type(simulation_before) is int and simulation_before >= 0,
                "Invalid simulation cutoff")
        # Corrections, rewind or insertion before simulated events invalidate the pool.
        if append_only:
            require(len(observed) >= len(self._observed) and len(simulated) >= len(self._simulated),
                    "Append-only Customer history shrank")
            changed = False
        else:
            changed = (self._observed != list(observed[:len(self._observed)])
                       or self._simulated != list(simulated[:len(self._simulated)]))
        changed = changed or bool(self._simulated and len(observed) != len(self._observed))
        recent = deque() if changed else self._recent.copy()
        added_rows, candidates, categories = [], [], []
        for events, stored, field, clock, before in (
            (observed, self._observed, "time_index", "observed_time_index", cutoff),
            (simulated, self._simulated, "settled_step", "simulation_settled_step", simulation_before),
        ):
            start = 0 if changed else len(stored)
            previous = stored[-1][field] if start else -1
            require(not start or before is not None and previous < before, "Future Customer history")
            added = []
            for event in events[start:]:
                require(event.get("user_id", self.user_id) == self.user_id, "Foreign Customer history")
                index = event[field]
                require(type(index) is int and index >= 0, "Invalid Customer history index")
                require(index < before, "Future Customer history")
                require(previous <= index, "Unordered Customer history")
                previous = index
                saved = deepcopy(event)
                added.append(saved)
                recent.append((saved, clock, index))
                if len(recent) > self.recent_n:
                    old, old_clock, old_index = recent.popleft()
                    candidates.append({
                        "owner_id": self.user_id, "clock": old_clock,
                        "start_inclusive": old_index, "end_inclusive": old_index,
                        "evidence_kind": "observed_fact" if old_clock == "observed_time_index" else "simulated_fact",
                        "records": [old], "search_text": event_text("customer", {"content": old}),
                    })
                    categories.append(self._event_categories(old))
            added_rows.append(added)

        default_query = " ".join(event["query"] for event, _, _ in recent)
        recent_categories = set().union(*(self._event_categories(event) for event, _, _ in recent))
        # Commit the update only after both clocks and the new recent window are valid.
        if changed:
            self._observed, self._simulated = added_rows
            self.candidates, self._categories = candidates, categories
        else:
            self._observed.extend(added_rows[0])
            self._simulated.extend(added_rows[1])
            self.candidates.extend(candidates)
            self._categories.extend(categories)
        self._recent = recent
        self.cutoff, self.simulation_before = cutoff, simulation_before
        self.default_query, self._recent_categories = default_query, recent_categories
        return self

    def recent_history(self):
        """Return the same recent requests that were excluded from older candidates."""
        return deepcopy([event for event, _, _ in self._recent])

    @staticmethod
    def _event_categories(event):
        return {category for field in ("clicked_items", "purchased_items")
                for product in event.get(field, [])
                if (category := category_id(product, 3)) is not None}

    def retrieve(self, query="", *, embedder=DEFAULT_HISTORY_ENCODER, limit=LIMIT, diagnostics=True):
        if embedder is not None and not isinstance(embedder, CachedHistoryEncoder):
            if self._encoder is None or self._encoder.embedder is not embedder:
                self._encoder = CachedHistoryEncoder(embedder)
            embedder = self._encoder
        # Recompute current relevance, while keeping event text and exact vectors reusable.
        candidates = [{**entry, "match": {"entity_match": int(bool(categories & self._recent_categories))}}
                      for entry, categories in zip(self.candidates, self._categories, strict=True)]
        result = _rank(candidates, query or self.default_query, embedder, limit,
                       owner=self.user_id, cutoff=self.cutoff,
                       simulation_before=self.simulation_before)
        if not diagnostics:
            # Stored memory evidence should have the same identity for the same facts.
            result.pop("ranking")
            result.pop("rank_constant")
            for segment in result["segments"]:
                segment.pop("match")
        return deepcopy(result)


def prepare_customer_history(observed, simulated, *, cutoff, recent_n,
                             user_id=None, simulation_before=None, prepared=None):
    """Build a new pool or use a matching, already validated decision snapshot."""
    if prepared is None:
        return PreparedCustomerHistory(user_id, recent_n).update(
            observed, simulated, cutoff=cutoff, simulation_before=simulation_before)
    require(type(cutoff) is int and type(recent_n) is int
            and (prepared.user_id, prepared.cutoff, prepared.recent_n, prepared.simulation_before)
            == (user_id, cutoff, recent_n, simulation_before), "Customer history snapshot differs")
    return prepared


def customer_history(observed, simulated, *, cutoff, recent_n, query="",
                     embedder=DEFAULT_HISTORY_ENCODER, limit=LIMIT, user_id=None,
                     simulation_before=None, prepared=None):
    pool = prepare_customer_history(
        observed, simulated, cutoff=cutoff, recent_n=recent_n, user_id=user_id,
        simulation_before=simulation_before, prepared=prepared,
    )
    return pool.retrieve(query, embedder=embedder, limit=limit)
