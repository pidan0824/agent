"""UserMirrorer candidate construction with KuaiSearch request-level adapters.

Offline, outcome-conditioned training scenes: sample distractors, insert every
observed positive, and retain no-click requests without inserting any target.
No LLM inference or behavior evaluation is performed by this module.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import random
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from agent_simulation.data.common import require
from agent_simulation.vendor.usermirrorer import sampler as original

_RNG_LOCK = RLock()


@contextmanager
def seeded(seed):
    # Upstream uses module-global random/torch RNGs. Isolate construction seeds.
    with _RNG_LOCK, torch.random.fork_rng(devices=[]):
        py, np_state = random.getstate(), np.random.get_state()
        random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(py); np.random.set_state(np_state)


class SampledCandidateBuilder:
    def __init__(self, products, interactions, encoder, *, min_items=2, max_items=12,
                 methods=("random", "content", "collaborative"), seed=42, min_history=3, fit_cutoff=None):
        require(1 <= min_items <= max_items, "Invalid candidate length range")
        require(methods and set(methods) <= {"random", "content", "collaborative"}, "Unknown sampler method")
        require(min_history >= 0, "Negative minimum history")
        self.products = {p["item_id"]: deepcopy(p) for p in products}
        require(len(self.products) == len(products), "Duplicate candidate pool item")
        self.interactions = list(interactions)
        require(all(type(r["time_index"]) is int and type(r["user_id"]) is int for r in interactions),
                "Interaction pool must have integer user/time IDs")
        self.encoder, self.methods, self.seed = encoder, tuple(methods), seed
        self.min_items, self.max_items, self.min_history = min_items, max_items, min_history
        self._vectors = None
        self.fit_cutoff, self._cf_cache = fit_cutoff, {}

    def _content_vectors(self):
        if self._vectors is None:
            require(self.encoder is not None, "Content sampling requires an embedding encoder")
            self._vectors = {item: vector for item, vector in zip(self.products,
                self.encoder.encode_many([" ".join(str(p.get(k) or "") for k in
                    ("item_title", "brand_name", "category_level3_name")) or str(item)
                    for item, p in self.products.items()]), strict=True)}
        return self._vectors

    def build(self, request):
        context = deepcopy(request["response_input"])
        target = request["response_target"]
        positives = sorted(set(target["clicked_item_ids"]) | set(target["purchased_item_ids"]))
        cutoff = context["history_cutoff_exclusive"]
        user = context["profile"]["user_id"]
        prior = [r for r in self.interactions if r["time_index"] < cutoff]
        owned = [r for r in prior if r["user_id"] == user]
        known = {(r["item_id"],r["time_index"]) for r in owned}
        for event in context["short_term_memory"]["search_history"]:
            require(event["time_index"] < cutoff, "Non-prior Customer history")
            for field in ("clicked_items","purchased_items"):
                for product in event.get(field,[]):
                    key = (product["item_id"],event["time_index"])
                    if key not in known:
                        owned.append({"user_id":user,"item_id":key[0],"time_index":key[1]})
                        known.add(key)
        owned.sort(key=lambda r:(r["time_index"],r["item_id"]))
        require(len(owned) >= self.min_history, "insufficient_prior_interactions")
        require(set(positives) <= set(self.products), "True action missing from candidate pool")
        # Use data time indices directly; they are not Unix timestamps.
        observed = {p["item_id"] for p in context["displayed_items"]}
        ids = sorted(item for item, p in self.products.items()
                     if p.get("first_seen_time_index", 0) < cutoff or item in observed or item in positives)
        require(ids, "empty_available_item_pool")
        mapping = {item: i for i, item in enumerate(ids)}
        seen = sorted({r["item_id"] for r in owned})
        prior = [r for r in prior if r["item_id"] in mapping]
        # No-click has no current positive: use prior interests, then the query.
        anchors = positives or list(dict.fromkeys(r["item_id"] for r in reversed(owned)
                                                  if r["item_id"] in mapping))[:10]
        event_rows = [{"uid": 0, "iid": mapping[item]} for item in anchors]
        if not event_rows:
            event_rows = [{"uid": 0, "iid": 0}]
        sf = SimpleNamespace(item_df=pd.DataFrame({"item_id": ids}),
                             interaction_df=pd.DataFrame(event_rows),
                             user_interaction_list=pd.Series({user: seen}))
        seed = int(hashlib.sha256(f"{self.seed}:{request['session_id']}".encode()).hexdigest()[:8], 16)
        with seeded(seed):
            rankings = []
            sample_num = min(len(ids), max(32, self.max_items * 3))
            if "random" in self.methods:
                scores = original.random_scoring(sf, torch.tensor([0]))
                rankings.extend([[ids[i] for i in row] for row in original.topk_sampling(
                    torch.softmax(scores, dim=1), sample_num).tolist()])
            if "content" in self.methods:
                vectors = self._content_vectors()
                sf.item_content_embedding = torch.tensor(np.stack([vectors[item] for item in ids]), dtype=torch.float)
                if anchors:
                    scores = original.content_based_scoring(sf, torch.arange(len(anchors)))
                else:
                    query = torch.tensor(self.encoder.encode_many([context["query"]], query=True)[0], dtype=torch.float)
                    scores = (query @ sf.item_content_embedding.T).unsqueeze(0)
                rankings.extend([[ids[i] for i in row] for row in original.topk_sampling(
                    torch.softmax(scores, dim=1), sample_num).tolist()])
            if "collaborative" in self.methods:
                fit_cutoff = self.fit_cutoff if self.fit_cutoff is not None else cutoff
                require(fit_cutoff <= cutoff, "Collaborative snapshot exceeds decision cutoff")
                if fit_cutoff not in self._cf_cache:
                    global_ids = sorted(self.products)
                    global_map = {item: i for i, item in enumerate(global_ids)}
                    fit = [r for r in self.interactions if r["time_index"] < fit_cutoff and r["item_id"] in global_map]
                    users = {uid: i for i, uid in enumerate(sorted({r["user_id"] for r in fit} | {user}))}
                    shape = (max(32, len(users)), max(32, len(global_ids)))
                    indices = torch.tensor([[users[r["user_id"]] for r in fit],
                                            [global_map[r["item_id"]] for r in fit]], dtype=torch.long)
                    matrix = torch.sparse_coo_tensor(indices, torch.ones(len(fit)), shape).coalesce()
                    rank = max(1, min(64, min(shape) // 4))
                    cf = original.get_spectral_info(matrix, rank)[:len(global_ids)]
                    self._cf_cache[fit_cutoff] = (global_map, cf)
                global_map, cf = self._cf_cache[fit_cutoff]
                vectors = cf[[global_map[item] for item in ids]]
                history = torch.zeros((1, len(ids)))
                for row in owned:
                    if row["item_id"] in mapping:
                        history[0, mapping[row["item_id"]]] += 1
                scores = original.collaborative_predict(vectors, history)
                rankings.extend([[ids[i] for i in row] for row in original.topk_sampling(
                    torch.softmax(scores, dim=1), sample_num).tolist()])
            merged = original.merge_ranking_list(rankings)
            frame = pd.DataFrame([{"user_id": user, "impression_list": merged}])
            filtered = original.interaction_filtering(sf, frame).iloc[0].impression_list
            negatives = [item for item in filtered if item not in positives
                         and (positives or item in observed)]
            size = max(len(positives), random.randint(self.min_items, self.max_items))
            candidates = negatives[:max(0, size-len(positives))]
            for item in positives:
                if candidates:
                    candidates, _ = original.add_gt_to_impression(candidates, item)
                else:
                    # Upstream gt_position requires a nonempty distractor list.
                    candidates = [item]
        require(len(candidates) == len(set(candidates)), "Duplicate constructed candidate")
        require(set(positives) <= set(candidates), "Lost true multi-action label")
        require(candidates, "empty_constructed_list")
        context["displayed_items"] = [deepcopy(self.products[item]) for item in candidates]
        for product in context["displayed_items"]:
            product.pop("first_seen_time_index", None)
        original_ids = [p["item_id"] for p in request["response_input"]["displayed_items"]]
        categories = [str(self.products[item].get("category_level1_name") or "unknown") for item in candidates]
        manifest = {"policy": "sampled", "protocol": "offline_outcome_conditioned", "outcome_conditioned": True,
            "original_item_ids": original_ids, "candidate_item_ids": candidates,
            "original_count": len(original_ids), "candidate_count": len(candidates),
            "domain": Counter(categories).most_common(1)[0][0], "methods": self.methods,
            "seed": seed, "target_size": size, "prior_interactions": len(owned),
            "distractor_count": len(candidates)-len(positives), "positive_count": len(positives),
            "pool_size": len(ids), "no_click_pool": "recorded_exposure_only" if not positives else None, "short_pool": len(candidates) < self.min_items,
            "over_max_to_preserve_positives": len(candidates) > self.max_items,
            "target_positions": {name: [candidates.index(item) for item in target[name]]
                                 for name in ("clicked_item_ids", "purchased_item_ids")},
            "embedding_identity": getattr(self.encoder, "identity", None),
            "input_sha256": hashlib.sha256(json.dumps(request["response_input"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
        result = deepcopy(request)
        result["response_input"] = context
        require(result["query_input"] == request["query_input"], "Candidate construction changed Query input")
        return {"request": result, "candidate": manifest}
