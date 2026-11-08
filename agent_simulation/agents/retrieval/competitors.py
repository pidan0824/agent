"""Small, diverse public product examples; reuse the factual-history encoder."""

from collections import Counter, defaultdict
from functools import lru_cache
from time import perf_counter
from agent_simulation.data.market.quotes import purchase_relation

from agent_simulation.data.common import rows
from agent_simulation.data.market.products import CATEGORY_FIELDS, complete_category
from agent_simulation.platform.search import tokens
from agent_simulation.agents.retrieval.history import DEFAULT_HISTORY_ENCODER, history_encoder
from agent_simulation.agents.retrieval.ranking import reciprocal_rank_scores, freshness_factor
from agent_simulation.agents.retrieval.retrieval_text import product_text
from agent_simulation.agents.retrieval.similarity import cosine_scores

COMPETITOR_POLICY = {"limit": 3, "shortlist": 64, "per_shop": 1,
                     "shortlist_per_shop": 4,
                     "quote_scope": "recent_market_window",
                     "ranking": "lexical_brand_freshness_then_rrf_semantic_v4"}


@lru_cache(maxsize=131072)
def title_terms(title):
    """Text-only features; visibility and price signals are never cached here."""
    return frozenset(tokens(title))


@lru_cache(maxsize=131072)
def title_similarity(left, right):
    a, b = title_terms(left), title_terms(right)
    return len(a & b) / len(a | b) if a or b else 0.


class CoarseFeatures:
    """Reuse title postings within immutable, time-bounded market snapshots."""

    def __init__(self):
        self.snapshot_view = None
        self.entries = {}
        self.postings = defaultdict(set)
        self.arrays = {}
        self.terms = []
        self.records = []
        self.numeric = {}
        self.capacity = 0
        self.last_work = {}
        self.last_timings = {}

    def _grow(self, size):
        import numpy as np
        if size <= self.capacity:
            return
        capacity = max(256, self.capacity * 2, size)
        for name, dtype in [('brands', np.int64), ('times', np.int64),
                            ('ids', np.int64), ('owners', np.int64),
                            ('lengths', np.int32), ('active', bool), ('complete', bool)]:
            values = np.zeros(capacity, dtype=dtype)
            if name in self.numeric:
                values[:self.capacity] = self.numeric[name]
            self.numeric[name] = values
        self.capacity = capacity

    def _update(self, product, complete=True):
        item = product['item_id']
        slot = self.entries.get(item)
        if slot is None:
            slot = len(self.records)
            self._grow(slot + 1)
            self.entries[item] = slot
            self.records.append(None)
            self.terms.append(frozenset())
        old = self.terms[slot]
        terms = title_terms(product['item_title'])
        for term in old - terms:
            self.postings[term].discard(slot)
            self.arrays.pop(term, None)
        for term in terms - old:
            self.postings[term].add(slot)
            self.arrays.pop(term, None)
        self.terms[slot] = terms
        self.records[slot] = product
        for name, value in [('brands', product['brand_id'] or -1),
                            ('times', product['reference_price_time_index']),
                            ('ids', -item), ('owners', product['seller_id']),
                            ('lengths', len(terms)), ('active', True), ('complete', complete)]:
            self.numeric[name][slot] = value
        return slot

    def prepare(self, peers):
        self.snapshot_view = None
        if self.capacity:
            self.numeric['active'][:len(self.records)] = False
        return [self._update(p) for p in peers]

    def prepare_snapshot(self, snapshot):
        if snapshot is self.snapshot_view:
            return
        # Recent windows are complete views. Rebuild membership on replacement,
        # retaining title postings but deactivating expired or removed products.
        if self.capacity:
            self.numeric['active'][:len(self.records)] = False
        for row in snapshot:
            self._update(row['_public'], row['_complete_category'])
        self.snapshot_view = snapshot

    def _recall(self, focal):
        import numpy as np
        words = title_terms(focal['item_title'])
        postings = []
        for word in words:
            if word not in self.arrays:
                self.arrays[word] = np.fromiter(self.postings.get(word, ()), dtype=np.intp)
            if len(self.arrays[word]):
                postings.append(self.arrays[word])
        self.last_work['posting_entries'] = sum(map(len, postings))
        if not postings:
            return np.array([], dtype=np.intp), np.array([], dtype=float)
        matched, counts = np.unique(np.concatenate(postings), return_counts=True)
        values = counts / (len(words) + self.numeric['lengths'][matched] - counts)
        return matched, values

    def _select(self, focal, eligible):
        import numpy as np
        start = perf_counter()
        slots, lexical = self._recall(focal)
        accepted = eligible[slots]
        slots, lexical = slots[accepted], lexical[accepted]
        self.last_timings['text_seconds'] = perf_counter() - start
        self.last_work.update(eligible=int(np.count_nonzero(eligible)), positive_text=len(slots),
                              zero_fallback_candidates=0)
        start = perf_counter()
        selected, shops = [], Counter()

        def select_group(group, scores):
            if not len(group):
                return
            brand = ((self.numeric['brands'][group] == focal['brand_id'])
                     if focal['brand_id'] and focal['brand_id'] > 0 else np.zeros(len(group), dtype=bool))
            order = np.lexsort((self.numeric['ids'][group], self.numeric['times'][group], brand, scores))[::-1]
            for slot in group[order]:
                owner = int(self.numeric['owners'][slot])
                if shops[owner] >= COMPETITOR_POLICY['shortlist_per_shop']:
                    continue
                selected.append(self.records[slot])
                shops[owner] += 1
                if len(selected) == COMPETITOR_POLICY['shortlist']:
                    break

        select_group(slots, lexical)
        if len(selected) < COMPETITOR_POLICY['shortlist']:
            # Only fill remaining capacity with zero-overlap products.
            eligible[slots] = False
            zero = np.flatnonzero(eligible)
            self.last_work['zero_fallback_candidates'] = len(zero)
            select_group(zero, np.zeros(len(zero)))
        self.last_timings['selection_seconds'] = perf_counter() - start
        return selected

    def select(self, focal, slots):
        import numpy as np
        eligible = np.zeros(len(self.records), dtype=bool)
        eligible[slots] = True
        return self._select(focal, eligible)

    def select_snapshot(self, focal, snapshot, cutoff):
        start = perf_counter()
        self.prepare_snapshot(snapshot)
        self.last_timings = {'prepare_seconds': perf_counter() - start}
        if not self.records:
            self.last_work = {'eligible': 0, 'positive_text': 0, 'posting_entries': 0,
                              'zero_fallback_candidates': 0}
            return [], 0
        start = perf_counter()
        n = len(self.records)
        eligible = (self.numeric['active'][:n] & self.numeric['complete'][:n]
                    & (self.numeric['owners'][:n] != focal['seller_id'])
                    & (self.numeric['times'][:n] < cutoff))
        self.last_timings['eligibility_seconds'] = perf_counter() - start
        selected = self._select(focal, eligible)
        return selected, self.last_work['eligible']


def coarse_candidates(focal, peers, *, features=None):
    features=features or CoarseFeatures()
    return features.select(focal,features.prepare(peers))


def rank_competitors(focal, candidates, cutoff, *, embedder=DEFAULT_HISTORY_ENCODER, qualified=False, features=None, snapshot=None):
    """Preselect bounded public candidates before encoding, never private shop facts."""
    category = tuple(focal[k] for k in CATEGORY_FIELDS)
    peers = candidates if qualified else [p for p in candidates if p["seller_id"] != focal["seller_id"]
             and complete_category(p) and tuple(p[k] for k in CATEGORY_FIELDS) == category
             and p["reference_price_time_index"] < cutoff] if complete_category(focal) else []
    if snapshot is not None:
        pool,eligible=features.select_snapshot(focal,snapshot,cutoff)
    else:
        eligible=len(peers)
        pool=coarse_candidates(focal,peers,features=features)
    query = product_text(focal)

    def signals(pool):
        lexical, brand = [], []
        for p in pool:
            lexical.append(title_similarity(focal['item_title'],p['item_title']))
            brand.append(int(p["brand_id"] == focal["brand_id"]) if focal["brand_id"] and focal["brand_id"] > 0 else None)
        return [lexical, brand]

    def order(pool, values):
        scores = reciprocal_rank_scores(values)
        return sorted(range(len(pool)), key=lambda i: (
            scores[i] * freshness_factor(pool[i]["reference_price_time_index"], cutoff),
            pool[i]["reference_price_time_index"], -pool[i]["item_id"]), reverse=True)

    values = signals(pool)
    start = perf_counter()
    if pool and embedder is not None:
        encoder = history_encoder(embedder)
        texts = [product_text(p) for p in pool]
        vectors = (encoder.encode_many(texts) if callable(getattr(encoder, "encode_many", None))
                   else [encoder(t) for t in texts])
        scores = cosine_scores(getattr(encoder, "encode_query", encoder)(query), vectors)
        values.append([scores[i] for i in range(len(pool))])
    if features is not None:
        features.last_timings["semantic_seconds"] = perf_counter() - start
    start = perf_counter()
    selected, shops = [], set()
    for i in order(pool, values):
        peer = pool[i]
        if peer["seller_id"] not in shops:
            selected.append(peer)
            shops.add(peer["seller_id"])
        if len(selected) == COMPETITOR_POLICY["limit"]:
            break
    if features is not None:
        features.last_timings["fusion_seconds"] = perf_counter() - start
    return selected, {"eligible_products": eligible, "coarse_candidates":len(pool), "encoded_candidates": len(pool) if embedder is not None else 0,
                      "shortlisted_products": len(pool), "selection": COMPETITOR_POLICY["ranking"]}


def competitor_examples(data, seller, item, cutoff, start, *, connection=None,
                        embedder=DEFAULT_HISTORY_ENCODER, simulation=None, include_purchases=False):
    focal = data.product(item, cutoff)
    # Each index covers just one bounded quote window, never an accumulating past pool.
    key=(cutoff,*(focal[k] for k in CATEGORY_FIELDS))
    cache=data.competitor_features
    if key not in cache:cache[key]=CoarseFeatures()
    cache.move_to_end(key)
    while len(cache)>8:cache.popitem(last=False)
    features=cache[key]
    selected, selection = rank_competitors(focal, [], cutoff, embedder=embedder,qualified=True,
                                           features=features,snapshot=data.competitor_snapshot(seller,item,cutoff))
    ids = tuple(sorted(p["item_id"] for p in selected))
    counts = None
    if include_purchases and connection is not None and ids:
        def read():
            return {r["item_id"]: r["purchases"] for r in rows(connection, f"""
                SELECT item_id,count(*) purchases FROM {purchase_relation(connection)}
                WHERE time_index>=? AND time_index<?
                  AND item_id IN (SELECT unnest(?::BIGINT[])) GROUP BY item_id
                """, [start, cutoff, list(ids)])}
        counts = data.decision_cached(("competitor_purchases", start, cutoff, ids), read)
    cards = []
    for peer in selected:
        card = {k: peer[k] for k in ("item_id", "seller_id", "item_title", "brand_name",
                "reference_price_raw", "reference_price_time_index", "price_status")}
        if counts is not None:
            card["purchase_records"] = counts.get(peer["item_id"], 0)
        if simulation is not None:
            active = simulation["active_prices"].get(str(peer["item_id"]))
            if active is not None:
                clock = simulation.get("clock")
                card["simulation_price"] = dict(price_raw=active["price_raw"],
                    clock="simulation_logical_time" if clock else "simulation_settled_step",
                    as_of=clock["time"] if clock else simulation["simulation_step"])
        cards.append(card)
    return {"selection": selection, "purchase_window": {"clock": "observed_time_index",
            "start_inclusive": start, "end_exclusive": cutoff}, "products": cards}
