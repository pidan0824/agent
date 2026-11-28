"""UserMirrorer candidate construction with KuaiSearch request-level adapters.

Offline, outcome-conditioned training scenes: sample distractors, insert one
observed positive per scene, and retain no-click requests without inserting any target.
No LLM inference or behavior evaluation is performed by this module.
"""
from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
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


CANDIDATE_DEDUPLICATION = "same-shop-content-v1"
INTERACTION_MATCHED = "interaction-matched-v1"
LEGACY_CANDIDATES = "legacy"

_CONTENT_FIELDS = (
    "item_title", "seller_id", "seller_name", "brand_id", "brand_name",
    "category_level1_id", "category_level1_name",
    "category_level2_id", "category_level2_name",
    "category_level3_id", "category_level3_name",
)


def deduplicate_pool(ids, products, prices, history_ids, *, target_item_id, seed):
    """Collapse complete, identical descriptions within one exposure pool only.

    Prices must be the frozen strictly-prior observations used by the scene.
    Observation timestamps remain attached to the retained ID, not in this key.
    Any prior interaction with a member protects the entire group.
    """
    audit = {"protocol": CANDIDATE_DEDUPLICATION if prices is not None else None,
             "removed_to_retained": [], "history_protected_groups": []}
    if prices is None:
        return ids, audit  # Legacy callers without frozen prices cannot compare them.
    groups = {}
    for item in ids:
        product = products[item]
        values = tuple(product.get(field) for field in _CONTENT_FIELDS)
        if any(value is None or isinstance(value, str) and
               value.strip().casefold() in ("", "unknown", "null") for value in values):
            continue  # Missing metadata is not proof of identical content.
        price = prices.get(item, (None, None))[0]
        key = (*values, price, "observed" if price is not None else "unknown")
        groups.setdefault(key, []).append(item)
    removed = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        if history_ids.intersection(group):
            audit["history_protected_groups"].append(group)
            continue
        kept = target_item_id if target_item_id in group else min(
            group, key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).digest())
        for item in group:
            if item != kept:
                removed.add(item)
                audit["removed_to_retained"].append({"removed": item, "retained": kept})
    return [item for item in ids if item not in removed], audit



def match_interacted_distractors(negatives, exposed_ids, prior_ids, positive_ids, *,
                                 rate, seed, scene_id, calibration_sha256=None, acceptance_probability=None):
    """Replace equal slots using calibrated binomial admission of exposed seen items.

    The pure helper is shared by new construction and frozen-list replay, which
    need not rerun embedding/CF ranking just to audit this one protocol change.
    """
    acceptance_probability = rate if acceptance_probability is None else acceptance_probability
    require(math.isfinite(rate) and 0 <= rate <= 1 and math.isfinite(acceptance_probability)
            and 0 <= acceptance_probability <= 1, "Invalid history interaction rate")
    rng = random.Random(f"{seed}:{scene_id}:{INTERACTION_MATCHED}")
    eligible_seen = sorted(set(exposed_ids) & set(prior_ids) - set(positive_ids),
        key=lambda item: hashlib.sha256(f"{seed}:{scene_id}:seen:{item}".encode()).digest())
    requested = sum(rng.random() < acceptance_probability for _ in eligible_seen)
    realized = min(requested, len(negatives))
    result = list(negatives)
    require(not (set(result) & set(prior_ids)), "Baseline negatives must follow legacy filtering")
    for slot, item in zip(sorted(rng.sample(range(len(result)), realized)),
                          eligible_seen[:realized], strict=True):
        result[slot] = item
    return result, {
        "protocol": INTERACTION_MATCHED, "fit_rate": rate,
        "seen_acceptance_probability": acceptance_probability,
        "legacy_distractor_count": len(negatives), "requested_seen_count": requested,
        "eligible_seen_count": len(eligible_seen), "selected_seen_count": realized,
        "unfilled_seen_count": requested - realized,
        "calibration_request_ids_sha256": calibration_sha256,
        "count_scope": "strictly-prior event interactions; snapshots excluded"}

class SampledCandidateBuilder:
    def __init__(self, products, interactions, encoder, *, min_items=2, max_items=12,
                 methods=("random", "content", "collaborative"), seed=42, min_history=3, fit_cutoff=None,
                 candidate_protocol=LEGACY_CANDIDATES, interaction_calibration=None):
        require(2 <= min_items <= max_items, "Invalid candidate length range")
        require(methods and set(methods) <= {"random", "content", "collaborative"}, "Unknown sampler method")
        require(min_history >= 0, "Negative minimum history")
        require(candidate_protocol in (LEGACY_CANDIDATES, INTERACTION_MATCHED), "Unknown candidate protocol")
        if candidate_protocol == INTERACTION_MATCHED:
            require(isinstance(interaction_calibration, dict)
                    and interaction_calibration.get("scope") == "frozen_train_fit_full_exposures"
                    and interaction_calibration.get("source_split") == "train",
                    "Interaction matching requires Train fit exposure calibration")
            denominator = interaction_calibration.get("nonpositive_exposure_items")
            numerator = interaction_calibration.get("interacted_nonpositive_items")
            require(type(denominator) is int and denominator > 0 and type(numerator) is int
                    and 0 <= numerator <= denominator, "Invalid interaction calibration counts")
            require(interaction_calibration.get("rate") == numerator / denominator,
                    "Interaction calibration rate differs from counts")
            allocation = interaction_calibration.get("allocation", {})
            require(not allocation or all(allocation.get(k) == v for k, v in
                (("seed", seed), ("min_items", min_items), ("max_items", max_items), ("min_history", min_history))),
                "Candidate settings differ from fit interaction calibration")
            probability = interaction_calibration.get("seen_acceptance_probability")
            require(type(probability) in (int, float) and math.isfinite(probability)
                    and 0 <= probability <= 1, "Missing valid calibrated seen acceptance probability")
        self.candidate_protocol = candidate_protocol
        self.interaction_calibration = deepcopy(interaction_calibration)
        self.products = {p["item_id"]: deepcopy(p) for p in products}
        require(len(self.products) == len(products), "Duplicate candidate pool item")
        self.interactions = list(interactions)
        require(all(type(r["time_index"]) is int and type(r["user_id"]) is int for r in self.interactions),
                "Interaction pool must have integer user/time IDs")
        self._user_interactions = defaultdict(list)
        for row in self.interactions:
            self._user_interactions[row["user_id"]].append(row)
        for history in self._user_interactions.values():
            history.sort(key=lambda row: (row["time_index"], row["item_id"]))
        self.encoder, self.methods, self.seed = encoder, tuple(methods), seed
        self.min_items, self.max_items, self.min_history = min_items, max_items, min_history
        self._vectors = {}
        self.fit_cutoff, self._cf_cache = fit_cutoff, {}

    def _content_vectors(self, ids):
        # Encode only requested exposure items, never the whole catalog.
        missing = [item for item in ids if item not in self._vectors]
        if missing:
            require(self.encoder is not None, "Content sampling requires an embedding encoder")
            encoded = self.encoder.encode_many([" ".join(str(self.products[item].get(k) or "")
                for k in ("item_title", "brand_name", "category_level3_name")) or str(item)
                for item in missing])
            for item, vector in zip(missing, encoded, strict=True):
                vector = np.asarray(vector, dtype=np.float32)
                require(bool(np.isfinite(vector).all()) and np.linalg.norm(vector) > 0, "Invalid item vector")
                self._vectors[item] = vector / np.linalg.norm(vector)
        return self._vectors

    def build_all(self, request, *, price_observations=None):
        target = request["response_target"]
        positives = sorted(set(target["clicked_item_ids"]) | set(target["purchased_item_ids"]))
        return [self.build(request, target_item_id=item, price_observations=price_observations) for item in (positives or [None])]

    def build(self, request, *, target_item_id=None, price_observations=None):
        context = deepcopy(request["response_input"])
        source_target = request["response_target"]
        source_positives = sorted(set(source_target["clicked_item_ids"]) | set(source_target["purchased_item_ids"]))
        if target_item_id is None and source_positives:
            require(len(source_positives) == 1, "Multi-positive request requires build_all or an explicit target")
            target_item_id = source_positives[0]
        require(target_item_id is None or target_item_id in source_positives, "Unknown target item")
        positives = [] if target_item_id is None else [target_item_id]
        target = {field: [item for item in source_target[field] if item == target_item_id]
                  for field in ("clicked_item_ids", "purchased_item_ids")}
        scene_id = f"{request['session_id']}:positive:{target_item_id}" if positives else f"{request['session_id']}:no_action"
        cutoff = context["history_cutoff_exclusive"]
        user = context["profile"]["user_id"]
        history = self._user_interactions.get(user, ())
        stop = bisect_left(history, cutoff, key=lambda row: row["time_index"])
        owned = list(history[:stop])
        known = {(r["item_id"],r["time_index"]) for r in owned}
        for event in context["short_term_memory"]["search_history"]:
            require(event["time_index"] < cutoff, "Non-prior Customer history")
            require(event.get("user_id", user) == user, "Foreign Customer history")
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
        require(set(source_positives) <= observed, "True action missing from recorded exposure")
        require(observed <= self.products.keys(), "Observed item missing from candidate pool")
        require(observed, "empty_available_item_pool")
        # All three routes rank the same recorded-exposure pool. No semantic or
        # category gate; other true positives never enter this scene's distractors.
        ids = sorted(observed - (set(source_positives) - set(positives)))
        scope = {"candidate_pool": "request_exposures", "exposure_pool_size": len(observed),
                 "sampling_pool_size": len(ids), "query_category_filter": False}
        history_ids = {r["item_id"] for r in owned}
        summary = context.get("behavioral_summary")
        if summary is not None:
            when = summary.get("observed_time_index")
            require(type(when) is int and 0 <= when < cutoff, "Customer background is not strictly prior")
            require(summary.get("user_id", user) == user, "Foreign Customer background")
            for field in ("recent_clicked_item_ids", "recent_purchased_item_ids"):
                history_ids.update(summary.get(field) or [])
        if price_observations is not None:
            require(all(item in observed and type(time) is int and 0 <= time < cutoff
                        and math.isfinite(price) and price > 0
                        for item, (price, time) in price_observations.items()),
                    "Invalid strictly-prior candidate prices")
        ids, deduplication = deduplicate_pool(ids, self.products, price_observations, history_ids,
            target_item_id=target_item_id, seed=f"{self.seed}:{request['session_id']}")
        scope.update(sampling_pool_size=len(ids), deduplication=deduplication)
        mapping = {item: i for i, item in enumerate(ids)}
        seen = sorted({r["item_id"] for r in owned})
        # No-click has no current positive: use prior interests, then the query.
        anchors = positives or list(dict.fromkeys(r["item_id"] for r in reversed(owned)
                                                  if r["item_id"] in mapping))[:10]
        event_rows = [{"uid": 0, "iid": mapping[item]} for item in anchors]
        if not event_rows:
            event_rows = [{"uid": 0, "iid": 0}]
        sf = SimpleNamespace(item_df=pd.DataFrame({"item_id": ids}),
                             interaction_df=pd.DataFrame(event_rows),
                             user_interaction_list=pd.Series({user: seen}))
        seed = int(hashlib.sha256(f"{self.seed}:{scene_id}".encode()).hexdigest()[:8], 16)
        # List length must not depend on shuffle consumption or exposure-pool size.
        size = random.Random(f"{self.seed}:{scene_id}:target-size-v1").randint(self.min_items, self.max_items)
        with seeded(seed):
            rankings = []
            sample_num = min(len(ids), max(32, self.max_items * 3))
            if "random" in self.methods:
                scores = original.random_scoring(sf, torch.tensor([0]))
                rankings.extend([[ids[i] for i in row] for row in original.topk_sampling(
                    torch.softmax(scores, dim=1), sample_num).tolist()])
            if "content" in self.methods:
                vectors = self._content_vectors(ids)
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
                    fit = [r for r in self.interactions if r["time_index"] < fit_cutoff and r["item_id"] in self.products]
                    global_ids = sorted({r["item_id"] for r in fit})
                    global_map = {item: i for i, item in enumerate(global_ids)}
                    if fit:
                        users = {uid: i for i, uid in enumerate(sorted({r["user_id"] for r in fit} | {user}))}
                        shape = (max(32, len(users)), max(32, len(global_ids)))
                        indices = torch.tensor([[users[r["user_id"]] for r in fit],
                                                [global_map[r["item_id"]] for r in fit]], dtype=torch.long)
                        matrix = torch.sparse_coo_tensor(indices, torch.ones(len(fit)), shape).coalesce()
                        rank = max(1, min(64, min(shape) // 4))
                        cf = original.get_spectral_info(matrix, rank)[:len(global_ids)]
                    else:
                        cf = torch.empty((0, 0))
                    self._cf_cache[fit_cutoff] = (global_map, cf)
                global_map, cf = self._cf_cache[fit_cutoff]
                # Unseen candidates have no CF signal; retain all repeated known history.
                scores = cf.new_zeros((1, len(ids)))
                positions = [i for i, item in enumerate(ids) if item in global_map]
                if positions:
                    history_indices = [global_map[row["item_id"]] for row in owned if row["item_id"] in global_map]
                    user_embedding = cf[history_indices].sum(dim=0, keepdim=True)
                    scores[:, positions] = user_embedding @ cf[[global_map[ids[i]] for i in positions]].T
                rankings.extend([[ids[i] for i in row] for row in original.topk_sampling(
                    torch.softmax(scores, dim=1), sample_num).tolist()])
            merged = original.merge_ranking_list(rankings)
            frame = pd.DataFrame([{"user_id": user, "impression_list": merged}])
            filtered = original.interaction_filtering(sf, frame).iloc[0].impression_list
            negatives = [item for item in filtered if item not in source_positives]
            # The length range is a sampling target, not a reason to lose behavior
            # supervision. Keep any nonempty shortlist, including a lone positive.
            require(bool(negatives or positives), "insufficient_distractors")
            candidates = negatives[:max(0, size-len(positives))]
            if self.candidate_protocol == INTERACTION_MATCHED:
                # Preserve the legacy realized length and insertion position. Replace
                # equal numbers of negatives; never import an unexposed product.
                candidates, scope["interaction_matching"] = match_interacted_distractors(
                    candidates, ids, seen, source_positives,
                    rate=self.interaction_calibration["rate"], seed=self.seed, scene_id=scene_id,
                    calibration_sha256=self.interaction_calibration.get("request_ids_sha256"),
                    acceptance_probability=self.interaction_calibration["seen_acceptance_probability"])
            if positives:
                # Include the end slot; the upstream helper excludes it.
                position = random.Random(f"{self.seed}:{scene_id}:target-position-v1")
                candidates.insert(position.randrange(len(candidates) + 1), target_item_id)
        require(len(candidates) == len(set(candidates)), "Duplicate constructed candidate")
        require(set(positives) <= set(candidates), "Lost true multi-action label")
        require(candidates, "empty_constructed_list")
        context["displayed_items"] = [deepcopy(self.products[item]) for item in candidates]
        for product in context["displayed_items"]:
            product.pop("first_seen_time_index", None)
        original_ids = [p["item_id"] for p in request["response_input"]["displayed_items"]]
        categories = [str(self.products[item].get("category_level1_name") or "unknown") for item in candidates]
        manifest = {"policy": "sampled", "protocol": "offline_single_positive",
            "scene_id": scene_id, "source_request_id": request["session_id"],
            "source_response_target": deepcopy(source_target),
            "target_item_id": target_item_id, "excluded_positive_ids": [i for i in source_positives if i != target_item_id], "outcome_conditioned": True,
            "original_item_ids": original_ids, "candidate_item_ids": candidates,
            "original_count": len(original_ids), "candidate_count": len(candidates),
            "domain": Counter(categories).most_common(1)[0][0], "methods": self.methods,
            "seed": seed, "target_size": size, "target_size_policy": "scene-seeded-v1",
            "target_position_policy": "scene-seeded-uniform-v1",
            "prior_interactions": len(owned),
            "distractor_count": len(candidates)-len(positives), "positive_count": len(positives),
            "scope": scope, "pool_size": len(ids), "no_click_pool": "recorded_exposure_only" if not positives else None, "short_pool": len(candidates) < self.min_items,
            "target_positions": {name: [candidates.index(item) for item in target[name]]
                                 for name in ("clicked_item_ids", "purchased_item_ids")},
            "embedding_identity": getattr(self.encoder, "identity", None),
            "input_sha256": hashlib.sha256(json.dumps(request["response_input"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
        if self.candidate_protocol == INTERACTION_MATCHED:
            manifest["candidate_protocol_version"] = INTERACTION_MATCHED
        result = deepcopy(request)
        result["response_input"] = context
        result["response_target"] = deepcopy(target)
        require(result["query_input"] == request["query_input"], "Candidate construction changed Query input")
        return {"request": result, "candidate": manifest}
