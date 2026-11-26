"""Multi-label adapter around the original UserMirrorer filtering functions.

Strong/weak *explanations* are evaluated by one common behavior predictor.
DPO exports the original generator action and explanation, never resampled actions.
Per-item A/B/C/D = neither/click/purchase/both; sampled sets may be empty or plural.
Entropy uses a mixture over item/state pairs: H = log2(N) + mean item entropy.
It is not the joint entropy over all 4**N sets. Comparisons use identical N.
"""
from copy import deepcopy
import json
import math
import numpy as np
import pandas as pd

from agent_simulation.data.common import require
from agent_simulation.inference.requests import prompt_tokens, ContextCapacityError
from agent_simulation.vendor.usermirrorer.utils import get_entropy, get_prob, get_logloss, calculate_metrics
from agent_simulation.vendor.usermirrorer.filtering import behavior_sampling, data_filtering

from agent_simulation.evaluation.customer_quality import FIELDS, assess_record, behavior_stratum
from agent_simulation.evaluation.customer_evidence import check_evidence
from agent_simulation.evaluation.customer_contract import single_item_actions, validate_request_splits


def decode_states(ids, states):
    return {field: [item for item, state in zip(ids, states, strict=True) if int(state) & (1 << bit)]
            for bit, field in enumerate(FIELDS)}


def true_states(ids, target):
    return [int(item in target[FIELDS[0]]) + 2 * int(item in target[FIELDS[1]]) for item in ids]


def probability_record(probabilities, ids, target, *, rng):
    require(bool(ids), "Cannot score an empty exposure")
    if isinstance(probabilities, dict):
        actions = single_item_actions(ids)
        require(probabilities.get("actions") == actions, "Mismatched single-item option ordering")
        probs = np.asarray(probabilities["probabilities"], dtype=float)
        require(probs.shape == (len(actions),) and np.isfinite(probs).all()
                and (probs >= 0).all() and probs.sum() > 0, "Invalid joint option probabilities")
        # Reuse the author's probability floor before entropy (0*log(0) is NaN).
        probs = get_prob({chr(65+i): float(value) for i, value in enumerate(probs)}, len(actions))
        matches = [i for i, action in enumerate(actions) if all(
            set(action[f]) == set(target[f]) for f in FIELDS)]
        index = matches[0] if matches else None
        sampled = int(rng.choice(len(actions), p=probs))
        return {"prob": probs, "option_probabilities": probs.tolist(), "option_actions": actions,
                "candidate_item_ids": list(ids), "choice_cnt": len(ids),
                "logloss_unit": "bits_per_decision",
                "probability_protocol": "joint categorical: no action or one item with click/purchase/both",
                "entropy": float(get_entropy(probs)), "mean_item_entropy_bits": None,
                "logloss": float(-math.log2(max(float(probs[index]), 1e-12))) if index is not None else None,
                "full_target_log_probability": float(math.log(max(float(probs[index]), 1e-12))) if index is not None else None,
                "behavior": json.dumps(actions[sampled], sort_keys=True),
                "preference": sampled == index, "sampled_action": deepcopy(actions[sampled]),
                "argmax_action": deepcopy(actions[int(probs.argmax())])}

    require(len(probabilities) == len(ids), "Incomplete item probabilities")
    raw = np.asarray(probabilities, dtype=float)
    require(raw.shape == (len(ids), 4) and np.isfinite(raw).all() and (raw >= 0).all() and (raw.sum(axis=1) > 0).all(), "Invalid four-state probabilities")
    probs = np.array([get_prob(dict(zip("ABCD", row, strict=True)), 4) for row in probabilities])
    require(np.isfinite(probs).all(), "Non-finite probabilities")
    states = true_states(ids, target)
    sampled = [rng.choice(4, p=row) for row in probs]
    action = decode_states(ids, sampled)
    distribution = probs.reshape(-1) / len(ids)
    # Exact match to *full* observed sets, including positives missing from retrieval.
    preference = all(set(action[field]) == set(target[field]) for field in FIELDS)
    return {"prob": distribution, "item_probabilities": probs.tolist(),
            "candidate_item_ids": list(ids), "logloss_unit": "bits_per_item",
            "probability_protocol": "factorized item marginals; not a calibrated joint distribution",
            "entropy": float(get_entropy(distribution)),
            "mean_item_entropy_bits": float(np.mean([get_entropy(row) for row in probs])),
            "logloss": float(np.mean([get_logloss(row, int(state)) for row, state in zip(probs, states, strict=True)])),
            "full_target_log_probability": (float(sum(math.log(row[state]) for row, state in zip(probs, states, strict=True)))
                 if set(target[FIELDS[0]] + target[FIELDS[1]]) <= set(ids) else None),
            "behavior": json.dumps(action, sort_keys=True), "preference": preference,
            "sampled_action": action, "argmax_action": decode_states(ids, probs.argmax(axis=1))}


class OptionPredictor:
    """Author's allowed-label next-token probabilities, using an existing HTTP client."""
    def __init__(self, client, *, context_limit=16384):
        self.client, self.context_limit = client, context_limit
        labels = [client.tokenizer.encode(x, add_special_tokens=False) for x in "ABCD"]
        require(all(len(x) == 1 for x in labels), "Option labels must each be one token")
        self.labels = [x[0] for x in labels]

    def _single_score(self, binding, trajectory):
        actions = single_item_actions([p["item_id"] for p in binding.displayed])
        labels = [str(i) for i in range(len(actions))]
        token_ids = [self.client.tokenizer.encode(label, add_special_tokens=False) for label in labels]
        require(all(len(ids) == 1 for ids in token_ids), "Single-item labels must each be one token")
        allowed = [ids[0] for ids in token_ids]
        require(len(set(allowed)) == len(allowed), "Duplicate option tokens")
        evidence = {"context": binding.initial_context(), "rationale": trajectory.get("rationale", {}),
                    "observations": [{"name": call["name"], "returned_content": call.get("returned_content")}
                        for call in trajectory["trace"].get("tools", [])],
                    "options": dict(zip(labels, actions, strict=True))}
        messages = [{"role": "system", "content": "Predict this user's observed behavior. Output exactly one "
            "option label. No action is a normal option. Do not infer a purchase from a click. "
            "The supplied explanation, if any, is a hypothesis, not a fact."},
            {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}]
        prompt = prompt_tokens(messages, [], self.client.tokenizer)
        if len(prompt) + 1 > self.context_limit:
            raise ContextCapacityError("Option predictor context overflow")
        response = self.client.http.post("completions", json={"model": self.client.model, "prompt": [prompt],
            "max_tokens": 1, "temperature": 0, "top_p": 1, "top_k": -1, "n": 1,
            "add_special_tokens": False, "logprobs": len(allowed), "allowed_token_ids": allowed,
            "return_tokens_as_token_ids": True})
        response.raise_for_status()
        choices = response.json()["choices"]
        require(len(choices) == 1, "Incomplete single-item option scores")
        top, = choices[0]["logprobs"]["top_logprobs"]
        keys = [f"token_id:{token}" for token in allowed]
        require(all(key in top for key in keys), "Server must expose all joint option logprobs")
        values = np.asarray([top[key] for key in keys])
        values = np.exp(values - values.max())
        return {"actions": actions, "probabilities": (values / values.sum()).tolist()}

    def score(self, binding, trajectory):
        if binding.context.get("response_action_mode") == "single_item":
            return self._single_score(binding, trajectory)
        # No generator action and no target enter the predictor context.
        evidence = {"context": binding.initial_context(), "rationale": trajectory["rationale"],
                    "observations": [{"name": call["name"], "returned_content": call.get("returned_content"),
                                      "error": call.get("error")} for call in trajectory["trace"].get("tools", [])]}
        messages = [{"role": "system", "content":
            "Predict this user's observed behavior, given the supplied explanation and facts. "
            "For the indicated candidate output exactly one label: A=neither clicked nor purchased; "
            "B=clicked only; C=purchased only; D=both clicked and purchased. "
            "Other candidates can also be clicked/purchased; all candidates can receive A. "
            "Do not output an explanation. Treat the supplied explanation as a hypothesis, not a fact."},
            {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}]
        prompts = [prompt_tokens([*messages, {"role": "user", "content": f"Candidate item ID: {product['item_id']}"}],
                                 [], self.client.tokenizer) for product in binding.displayed]
        if not prompts:
            return []
        if max(map(len, prompts)) + 1 > self.context_limit:
            raise ContextCapacityError("Option predictor context overflow")
        response = self.client.http.post("completions", json={"model": self.client.model, "prompt": prompts,
            "max_tokens": 1, "temperature": 0, "top_p": 1, "top_k": -1, "n": 1, "add_special_tokens": False,
            "logprobs": 4, "allowed_token_ids": self.labels, "return_tokens_as_token_ids": True})
        response.raise_for_status()
        choices = sorted(response.json()["choices"], key=lambda c: c["index"])
        require([c["index"] for c in choices] == list(range(len(prompts))), "Incomplete option scores")
        probabilities = []
        for choice in choices:
            top, = choice["logprobs"]["top_logprobs"]
            keys = [f"token_id:{label}" for label in self.labels]
            require(all(key in top for key in keys), "Server must expose all four option logprobs")
            values = np.array([top[key] for key in keys]); values = np.exp(values - values.max())
            probabilities.append((values / values.sum()).tolist())
        return probabilities



def select_sft(records, *, variants=("strong",)):
    """Keep original outcome-accepted transcripts independently of predictor/DPO."""
    validate_request_splits(records)
    selected = {}
    for row in records:
        t = row["trajectory"]
        quality = assess_record(row)
        if (row["source_split"] != "train" or t["source_split"] != "train"
            or row["variant"] not in variants or not quality["accepted"] or not check_evidence(t)["accepted"]
            or not row.get("reachable", True)):
            continue
        meta = {key: t[key] for key in ("role", "actor_id", "stage", "initial_state_id", "source_split")}
        meta.update({key: deepcopy(t[key]) for key in ("source_request_id", "scene_id", "source_response_target", "dataset_manifest_sha256", "response_style", "action_mode", "evaluation_protocol") if key in t})
        value = {**meta, "trajectory_id": t["trajectory_id"], "tools": deepcopy(t["tools"]),
                 "messages": deepcopy(t["messages"]), "score": quality["metrics"]["score"],
                 "quality": quality, "evidence_quality": check_evidence(t),
                 "behavior_source": "generator_final_action; no predictor resampling"}
        # Do not collapse distinct actors/scenes just because their text matches.
        key = json.dumps([meta, value["messages"], value["tools"]], sort_keys=True)
        selected.setdefault(key, value)
    return list(selected.values())


def _uncertainty_metrics(usable):
    """Original author metrics on matched strong/weak samples; no generated labels."""
    if not usable:
        return []
    frame = pd.DataFrame(usable)
    complete = frame.groupby(["dataset", "cid", "tid"])["variant"].agg(set)
    complete = complete[complete.apply(lambda values: values == {"strong", "weak"})].index
    frame = frame.set_index(["dataset", "cid", "tid"]).loc[lambda x: x.index.isin(complete)].reset_index()
    if frame.empty:
        return []
    frame["uncertainty"] = frame.groupby(["dataset", "cid", "variant"])["prob"].transform(
        lambda values: get_entropy(np.mean(np.stack(values.to_list()), axis=0)))
    strong = frame[frame.variant == "strong"].set_index(["dataset", "cid", "tid"])
    weak = frame[frame.variant == "weak"].set_index(["dataset", "cid", "tid"])
    metrics = calculate_metrics(strong, weak).reset_index()
    for name, data in (("teacher", strong), ("student", weak)):
        hit = data.preference.groupby(level=[0, 1]).mean().rename(name + "_hit")
        metrics[name + "_hit"] = metrics.set_index(["dataset", "cid"]).index.map(hit)
    metrics["diff_hit"] = metrics.student_hit - metrics.teacher_hit
    columns = ["dataset", "cid", "choice_cnt",
               *[c for c in metrics if c.startswith(("teacher_", "student_", "diff_"))]]
    return metrics[columns].to_dict("records")


def _validate_group(rows):
    first = rows[0]["trajectory"]
    for row in rows:
        t = row["trajectory"]
        require(all(t[k] == first[k] for k in
                    ("role", "actor_id", "stage", "initial_state_id", "source_split")),
                "Mixed preference identity")
        require(t["prompt"] == first["prompt"] and t["tools"] == first["tools"],
                "Mixed preference contexts")
        require(t.get("target") == first.get("target"), "Mixed preference targets")
        require(all(t.get(k) == first.get(k) for k in
                    ("response_style", "action_mode", "evaluation_protocol", "dataset_manifest_sha256")),
                "Mixed preference protocols")
        require(t["messages"][:len(t["prompt"])] == t["prompt"], "Invalid trajectory prefix")
        # Exact identity/order, when recorded, prevents scoring different candidate lists.
        require(row.get("candidate_item_ids") == rows[0].get("candidate_item_ids")
                and row.get("choice_cnt") == rows[0].get("choice_cnt"), "Mixed candidate lists")


def _pair(good, bad, *, protocol, selection_metric, selection_value):
    a, b = good["trajectory"], bad["trajectory"]
    prefix = len(a["prompt"])
    require(a["messages"][prefix]["role"] == b["messages"][prefix]["role"] == "assistant",
            "Preference continuation must start with assistant")
    meta = {key: a[key] for key in ("role", "actor_id", "stage", "initial_state_id", "source_split")}
    meta.update({key: deepcopy(a[key]) for key in ("source_request_id", "scene_id", "source_response_target", "dataset_manifest_sha256", "response_style", "action_mode", "evaluation_protocol") if key in a})
    return {**meta, "prompt": deepcopy(a["prompt"]), "tools": deepcopy(a["tools"]),
        "chosen": deepcopy(a["messages"][prefix:]), "rejected": deepcopy(b["messages"][prefix:]),
        "chosen_trajectory_id": a["trajectory_id"], "rejected_trajectory_id": b["trajectory_id"],
        "chosen_variant": good["variant"], "rejected_variant": bad["variant"],
        "chosen_score": good["_quality"]["metrics"]["score"],
        "rejected_score": bad["_quality"]["metrics"]["score"],
        "chosen_logloss_bits": good["logloss"], "rejected_logloss_bits": bad["logloss"],
        "logloss_gap_bits": bad["logloss"] - good["logloss"],
        "logloss_unit": good.get("logloss_unit", "bits_per_item"),
        "chosen_evidence_quality": good["_evidence"],
        "rejected_evidence_quality": bad["_evidence"],
        "pair_reason": "evidence_quality" if bad["_quality"]["metrics"]["exact_match"] else "behavior_quality",
        "preference_source": protocol,
        "behavior_source": "original_generator_transcript; predictor_scores_only",
        "selection_metric": selection_metric, "selection_value": selection_value}


def select_preferences(records, *, max_pairs=10000, filter_column="diff_model_uncertainty",
                       sampling_column="entropy", ascending=False, pair_protocol="logloss",
                       logloss_min_gap=.05, action_min_gap=.05, diagnostics=None):
    """Rank scenarios by uncertainty; pair original actions and explanations.

    logloss: chosen matches both action heads exactly and passes provenance checks.
    Rejected has a worse action or a demonstrated evidence error; chosen also has
    lower observed-label NLL. NLL never substitutes for explanation quality.
    sampled_match: original author sampling/filter functions with explicit
    original-action safeguards; no resampled action is ever exported.
    """
    from collections import defaultdict
    require(max_pairs > 0, "Positive pair limit required")
    require(pair_protocol in ("logloss", "sampled_match"), "Unknown pair protocol")
    require(all(math.isfinite(v) and v > 0 for v in (logloss_min_gap, action_min_gap)),
            "Pair gaps must be positive and finite")
    sft_rows = select_sft(records)
    groups = defaultdict(list)
    for row in records:
        groups[(row.get("dataset", "unknown"), row.get("cid", row["trajectory"]["trajectory_id"]))].append(row)
    usable, candidates, statuses = [], [], {}
    for key, rows in groups.items():
        valid = [r for r in rows if r["source_split"] == "train"
                 and r["trajectory"]["source_split"] == "train"
                 and r["trajectory"]["status"] == "ok" and r.get("reachable", True)]
        if valid:
            _validate_group(valid)
        for row in valid:
            if row.get("prob") is not None and row.get("logloss") is not None and math.isfinite(row["logloss"]):
                quality = assess_record(row)
                if "score" in quality["metrics"]:
                    usable.append({**row, "_quality": quality, "_evidence": check_evidence(row["trajectory"])})
        statuses[key] = "no_eligible_train_trajectory" if not valid else "missing_probabilities"
    metrics = _uncertainty_metrics(usable)
    metric_map = {(r["dataset"], r["cid"]): r for r in metrics}
    scored = defaultdict(list)
    for row in usable:
        scored[(row["dataset"], row["cid"])].append(row)

    def compatible(good, bad):
        gm, bm = good["_quality"]["metrics"], bad["_quality"]["metrics"]
        if good is bad or not gm["exact_match"] or not good["_evidence"]["accepted"]:
            return False
        if good.get("logloss_unit") != bad.get("logloss_unit"):
            return False
        if bm["exact_match"]:
            # Same correct action: only a demonstrated provenance violation qualifies.
            # Missing annotations are unverified, not evidence of a worse explanation.
            return bad["_evidence"]["status"] == "failed"
        return (gm["score"] - bm["score"] >= action_min_gap
                and all(gm[f + "_f1"] >= bm[f + "_f1"] for f in FIELDS))

    if pair_protocol == "logloss":
        for key, rows in scored.items():
            eligible = [r for r in rows if r["_quality"]["metrics"]["exact_match"] and r["_evidence"]["accepted"]]
            statuses[key] = ("insufficient_scored_branches" if len(rows) < 2 else
                             "no_exact_behavior_and_evidence_chosen" if not eligible else "behavior_or_evidence_gap_or_conflict")
            action_pairs = [(a, b) for a in eligible for b in rows if compatible(a, b)]
            if not action_pairs:
                continue
            choices = [(a, b) for a, b in action_pairs if b["logloss"] - a["logloss"] >= logloss_min_gap]
            if not choices:
                statuses[key] = "logloss_gap_or_direction_conflict"
                continue
            good, bad = max(choices, key=lambda ab: (
                ab[1]["logloss"] - ab[0]["logloss"],
                ab[0]["_quality"]["metrics"]["score"] - ab[1]["_quality"]["metrics"]["score"],
                ab[0]["trajectory"]["trajectory_id"], ab[1]["trajectory"]["trajectory_id"]))
            candidates.append((key, good, bad))
    elif usable and metrics:
        # Preserve vendor functions. Adapt which branches may form a complete-agent
        # pair: sampled behavior AND original action must agree on correctness.
        legacy = []
        for row in usable:
            exact = row["_quality"]["metrics"]["exact_match"]
            if (row["preference"] and exact) or (not row["preference"] and not exact):
                legacy.append(row)
        if legacy:
            frame = pd.DataFrame(legacy).set_index(["dataset", "cid", "tid"])
            available = set(zip(frame.index.get_level_values(0), frame.index.get_level_values(1)))
            report = pd.DataFrame([m for m in metrics if (m["dataset"], m["cid"]) in available])
            if not report.empty:
                sampled = behavior_sampling(frame, report, filter_column, sampling_column).reset_index()
                lookup = {(r["dataset"], r["cid"], r["variant"], r["tid"]): r for r in legacy}
                for (dataset, cid), group in sampled.groupby(["dataset", "cid"]):
                    good_row = group[group.preference].iloc[0]
                    bad_row = group[~group.preference].iloc[0]
                    good = lookup[(dataset, cid, good_row.variant, good_row.tid)]
                    bad = lookup[(dataset, cid, bad_row.variant, bad_row.tid)]
                    if compatible(good, bad):
                        candidates.append(((dataset, cid), good, bad))
        for key in scored:
            statuses[key] = "no_consistent_sampled_match_pair"

    # Reuse author's stratified scene selection with matched pair rows.
    # The budget is counted in PAIRS here; the vendor function counts rows.
    selection, pairs_by_key = [], {}
    for key, good, bad in candidates:
        metric = metric_map.get(key, {}).get(filter_column)
        rank = float(metric) if metric is not None and math.isfinite(metric) else 0.
        pairs_by_key[key] = _pair(good, bad, protocol=pair_protocol,
            selection_metric=filter_column, selection_value=rank)
        pairs_by_key[key]["selection_metric_available"] = metric is not None and math.isfinite(metric)
        for preference in (True, False):
            selection.append({"dataset": key[0], "cid": key[1], "choice_cnt": good["choice_cnt"],
                "behavior_group": behavior_stratum(good["trajectory"]["target"]),
                "preference": preference, filter_column: rank})
        statuses[key] = "pair_limit"
    pairs = []
    if selection:
        selected = data_filtering(pd.DataFrame(selection), filter_column,
            ["dataset", "choice_cnt", "behavior_group"], max_pairs, ascending=ascending)
        for _, row in selected[selected.preference].iterrows():
            key = (row.dataset, row.cid)
            pairs.append(pairs_by_key[key])
            statuses[key] = "paired"
    if diagnostics is not None:
        diagnostics.extend({"dataset": key[0], "cid": key[1], "reason": reason}
                           for key, reason in statuses.items())
    return sft_rows, pairs, metrics


def validate_filter_options(*, query_min=None, response_min=None, click_min=None,
                            purchase_min=None, max_pairs=10000,
                            logloss_min_gap=.05, action_min_gap=.05):
    require(all(v is None or (math.isfinite(v) and 0 <= v <= 1)
                for v in (query_min, response_min, click_min, purchase_min)),
            "Quality thresholds must be finite and in [0,1]")
    require(max_pairs > 0, "Positive pair limit required")
    require(all(math.isfinite(v) and v > 0 for v in (logloss_min_gap, action_min_gap)),
            "Pair gaps must be positive and finite")


def export_training(records, output, *, max_pairs=10000, filter_column="diff_model_uncertainty",
                    sampling_column="entropy", ascending=False, query_min=None, response_min=None,
                    click_min=None, purchase_min=None, pair_protocol="logloss",
                    logloss_min_gap=.05, action_min_gap=.05):
    """Offline replay with row-level gates, scene-level pair reasons and coverage."""
    from collections import Counter, defaultdict
    from pathlib import Path
    from agent_simulation.evaluation.trajectories import training_rows, digest
    validate_filter_options(query_min=query_min, response_min=response_min,
        click_min=click_min, purchase_min=purchase_min, max_pairs=max_pairs,
        logloss_min_gap=logloss_min_gap, action_min_gap=action_min_gap)
    output = Path(output)
    records = deepcopy(records)
    validate_request_splits(records)
    decisions, groups, response = [], defaultdict(list), []
    for row in records:
        t = row["trajectory"]
        quality = assess_record(row, query_min=query_min, response_min=response_min,
                                click_min=click_min, purchase_min=purchase_min)
        t["accepted"] = quality["accepted"]
        if quality.get("thresholds"):
            t["quality_thresholds"] = quality["thresholds"]
        reasons = []
        if row["source_split"] != "train" or t["source_split"] != "train":
            reasons.append("held_out")
        if row["variant"] != "strong":
            reasons.append("sft_variant_excluded")
        if not row.get("reachable", True):
            reasons.append("target_unreachable")
        reasons.extend(quality["reasons"])
        evidence = check_evidence(t)
        reasons.extend(evidence["reasons"])
        stratum = ("query" if row["stage"] == "query" else
                   behavior_stratum(t["target"]) if "target" in t else "unknown")
        decisions.append({"trajectory_id": t["trajectory_id"], "dataset": row.get("dataset"),
            "cid": row.get("cid"), "stage": row["stage"], "variant": row["variant"],
            "source_request_id": t.get("source_request_id"), "scene_id": t.get("scene_id"),
            "source_stratum": behavior_stratum(t["source_response_target"]) if "source_response_target" in t else stratum,
            "stratum": stratum, "quality": quality, "evidence_quality": evidence,
            "domain": row.get("dataset", "unknown"), "candidate_count": row.get("choice_cnt"),
            "policy": row.get("policy", "unknown"), "response_style": t.get("response_style", "legacy"),
            "sft_exclusion_reasons": reasons,
            "probability_available": row.get("prob") is not None,
            "probability_error": row.get("probability_error")})
        if row["stage"] == "query":
            # Both row and trajectory split guard must hold.
            if row["source_split"] == t["source_split"] == "train":
                groups[(row["dataset"], row["cid"])].append(t)
        else:
            response.append(row)
    query_pairs, pair_decisions = [], []
    for (dataset, cid), group in groups.items():
        _, pairs, reason = training_rows(group, min_gap=action_min_gap)
        query_pairs.extend(pairs)
        pair_decisions.append({"stage": "query", "dataset": dataset, "cid": cid, "reason": reason})
    response_decisions = []
    response_sft, response_pairs, metrics = select_preferences(response, max_pairs=max_pairs,
        filter_column=filter_column, sampling_column=sampling_column, ascending=ascending,
        pair_protocol=pair_protocol, logloss_min_gap=logloss_min_gap,
        action_min_gap=action_min_gap, diagnostics=response_decisions)
    pair_decisions.extend({"stage": "response", **item} for item in response_decisions)
    sft = select_sft([r for r in records if r["stage"] == "query"]) + response_sft
    pairs = list({digest([r["initial_state_id"], r["actor_id"], r["stage"], r["prompt"],
                          r["tools"], r["chosen"], r["rejected"]]): r
                  for r in query_pairs + response_pairs}.values())
    sft_ids = Counter(r["trajectory_id"] for r in sft)
    strata = {}
    for decision in decisions:
        retained = not decision["sft_exclusion_reasons"] and sft_ids[decision["trajectory_id"]] > 0
        if retained:
            sft_ids[decision["trajectory_id"]] -= 1
        decision["sft_retained"] = retained
        if not retained and not decision["sft_exclusion_reasons"]:
            decision["sft_exclusion_reasons"].append("duplicate_transcript")
        key = (decision["stage"], decision["variant"], decision["stratum"],
               decision["domain"], decision["candidate_count"], decision["source_stratum"],
               decision["policy"], decision["response_style"])
        counts = strata.setdefault(key, {"stage": key[0], "variant": key[1], "stratum": key[2],
            "domain": key[3], "candidate_count": key[4], "source_stratum": key[5],
            "policy": key[6], "response_style": key[7],
            "raw": 0, "quality_passed": 0, "evidence_passed": 0, "sft_retained": 0})
        counts["raw"] += 1
        counts["evidence_passed"] += int(decision["evidence_quality"]["accepted"])
        counts["quality_passed"] += int(decision["quality"]["accepted"])
        counts["sft_retained"] += int(retained)
    for counts in strata.values():
        counts["sft_retention_rate"] = counts["sft_retained"] / counts["raw"]
    for name, values in [("sft", sft), ("preferences", pairs), ("uncertainty-metrics", metrics),
                         ("filter-decisions", decisions), ("pair-decisions", pair_decisions)]:
        with (output / (name + ".jsonl")).open("w") as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    funnel = {"raw_trajectories": len(records),
        "valid_trajectories": sum(r["trajectory"]["status"] == "ok" for r in records),
        "generator_outcome_accepted": sum(r["trajectory"]["accepted"] for r in records),
        "query_sft": sum(r["stage"] == "query" for r in sft),
        "response_sft": sum(r["stage"] == "response" for r in sft),
        "query_pairs": sum(r["stage"] == "query" for r in pairs),
        "response_pairs": sum(r["stage"] == "response" for r in pairs),
        "sft_total": len(sft), "pairs_total": len(pairs),
        "sft_exclusion_reasons": dict(Counter(reason for d in decisions for reason in d["sft_exclusion_reasons"])),
        "response_probability_missing": sum(r.get("prob") is None for r in response),
        "pair_reasons": dict(Counter(d["stage"] + "/" + d["reason"] for d in pair_decisions)),
        "strata": list(strata.values()),
        "counting_note": "Row exclusion reasons can overlap; pair reasons count scenarios, not trajectories.",
        "sft_protocol": "Strong original transcripts passing separate action and evidence gates; independent of DPO",
        "response_dpo_protocol": pair_protocol,
        "thresholds": {"query_min": query_min, "response_min": response_min,
            "click_min": click_min, "purchase_min": purchase_min,
            "logloss_min_gap_bits": logloss_min_gap, "action_min_gap": action_min_gap},
        "dpo_protocol": "Original generator transcripts only; Query outcome gap; Response action-consistent " + pair_protocol}
    (output / "filter-funnel.json").write_text(json.dumps(funnel, ensure_ascii=False, indent=2) + "\n")
    return funnel


def add_filter_arguments(parser):
    parser.add_argument("--click-min", type=float, help="Positive-click F1 gate; defaults to response-min")
    parser.add_argument("--purchase-min", type=float, help="Positive-purchase F1 gate; defaults to response-min")
    parser.add_argument("--pair-protocol", choices=["logloss", "sampled_match"], default="logloss")
    parser.add_argument("--logloss-min-gap", type=float, default=.05, help="NLL gap in bits: per decision for single-item lists, per item for full exposure")
    parser.add_argument("--action-min-gap", type=float, default=.05)


def main(argv=None):
    import argparse
    from pathlib import Path
    from agent_simulation.data.common import sha256
    p = argparse.ArgumentParser(description="Filter saved Customer trajectories; no model inference")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--query-min", type=float)
    p.add_argument("--response-min", type=float)
    p.add_argument("--max-pairs", type=int, default=10000)
    p.add_argument("--filter-column", default="diff_model_uncertainty")
    p.add_argument("--sampling-column", choices=["entropy", "logloss"], default="entropy")
    p.add_argument("--ascending", action="store_true")
    add_filter_arguments(p)
    args = p.parse_args(argv)
    validate_filter_options(query_min=args.query_min, response_min=args.response_min,
        click_min=args.click_min, purchase_min=args.purchase_min, max_pairs=args.max_pairs,
        logloss_min_gap=args.logloss_min_gap, action_min_gap=args.action_min_gap)
    require(not args.output.exists(), "Use a new filtering output directory")
    records = [json.loads(s) for s in args.input.read_text().splitlines() if s.strip()]
    args.output.mkdir(parents=True)
    report = export_training(records, args.output, max_pairs=args.max_pairs, filter_column=args.filter_column,
        sampling_column=args.sampling_column, ascending=args.ascending,
        query_min=args.query_min, response_min=args.response_min,
        click_min=args.click_min, purchase_min=args.purchase_min,
        pair_protocol=args.pair_protocol, logloss_min_gap=args.logloss_min_gap, action_min_gap=args.action_min_gap)
    (args.output / "manifest.json").write_text(json.dumps({"schema": "customer-filter-export-v2",
        "input": str(args.input.resolve()), "input_sha256": sha256(args.input),
        "filter_column": args.filter_column, "sampling_column": args.sampling_column,
        "ascending": args.ascending, "pair_protocol": args.pair_protocol,
        "thresholds": report["thresholds"]}, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
