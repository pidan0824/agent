"""Bounded Customer generation and optional, auditable teacher budget allocation."""
import asyncio
from collections import defaultdict, deque
import heapq
import json

from agent_simulation.data.common import require
from agent_simulation.evaluation.trajectories import digest
from agent_simulation.evaluation.customer_quality import behavior_stratum


def resolve_generation_options(args):
    from agent_simulation.evaluation.customer_context import resolve_context_options
    context = resolve_context_options(args.cot_preset, args.modes, args.history_token_budget)
    args.history_token_budget = context["history_token_budget"]
    light = context["concise"]
    for name, default in (("samples", 1 if light else 10),
                          ("temperature", 0. if light else .7),
                          ("max_new_tokens", 768 if light else 2048),
                          ("scene_concurrency", 4 if light else 1)):
        if getattr(args, name) is None:
            setattr(args, name, default)
    require(args.scene_concurrency > 0, "Positive scene concurrency required")
    if light:
        require(args.sft_only, "Lightweight preset requires --sft-only")
    if args.teacher_budget is not None:
        require(light and args.split == "train" and not args.evaluation_only
                and args.teacher_budget >= 0 and args.response_styles in (None, ["ekb"])
                and args.theory and args.candidate_policies == ["sampled"],
                "Teacher budget requires lightweight Train sampled EKB SFT")


async def bounded_map(function, items, concurrency):
    """Lazy ordered results, at most concurrency scene tasks (including completed)."""
    require(concurrency > 0, "Positive scene concurrency required")
    pending = deque()
    iterator = iter(items)
    try:
        for _ in range(concurrency):
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append(asyncio.create_task(function(item)))
        while pending:
            task = pending[0]
            result = await task
            pending.popleft()
            yield result
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.append(asyncio.create_task(function(item)))
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def scene_evidence(binding):
    """Rank input support, not agreement with current labels or explanation length."""
    context = binding.initial_context()  # Existing ownership and time guards.
    facts = context["facts"]
    products = defaultdict(dict)
    for fact in facts.values():
        if fact["kind"] == "product":
            products[fact["item_id"]][fact["field"]] = fact["value"]
    from agent_simulation.evaluation.customer_evidence import product_collisions
    collisions = product_collisions([{"item_id": identity, **product}
                                     for identity, product in products.items()], include_shop=True)
    supported_ids = set()
    query_match = False
    from agent_simulation.evaluation.customer_context import normalize_query
    query = normalize_query(binding.context.get("query", ""))
    for fact in facts.values():
        if fact["kind"] != "history":
            continue
        if fact["field"] in ("clicked_items", "purchased_items"):
            supported_ids.update(p["item_id"] for p in fact["value"])
        elif fact["field"] == "query" and query:
            query_match |= normalize_query(fact["value"]) == query
    same_item = bool(supported_ids.intersection(products))
    # An identical pair remains unsupported if neither ID has visible interaction evidence.
    unresolved = [ids for ids in collisions if not supported_ids.intersection(ids)]
    complete = len(products) == len(binding.displayed) > 0 and all(str(p.get("item_title") or "").strip() for p in products.values())
    return {"complete_titles": complete, "has_query": bool(query),
            "single_candidate": len(binding.displayed) == 1,
            "same_item_history": same_item, "exact_query_history": query_match,
            "identical_visible_groups": collisions, "unsupported_identical_groups": unresolved}


def budget_scenes(jobs, budget, seed, audit, *, history_token_budget=None, tokenizer=None):
    """Top evidence within behavior/domain strata, round-robin allocation.

    Labels are used only for coverage strata. Priority uses visible inputs only.
    Keep at most budget descriptors per stratum, not one asyncio task per scene.
    Raw observed rows are collected by the upstream iterator before selection.
    """
    from agent_simulation.evaluation.customer_context import CurationTools
    heaps = defaultdict(list)
    total = 0
    for order, job in enumerate(jobs):
        identity, actor, stage, policy, adapted, candidate, mode, style = job
        require(stage == "response" and mode == "fixed" and style == "ekb",
                "Teacher budget expects fixed Response EKB scenes")
        features = scene_evidence(CurationTools(adapted, stage, mode=mode, concise=True,
            history_token_budget=history_token_budget, tokenizer=tokenizer))
        scene_id = str(candidate.get("scene_id", identity))
        stratum = (candidate["domain"], behavior_stratum(adapted["response_target"]))
        priority = (not features["single_candidate"],
                    features["has_query"] and features["complete_titles"],
                    not features["unsupported_identical_groups"],
                    features["same_item_history"], features["exact_query_history"],
                    digest([seed, scene_id]))
        row = {"scene_id": scene_id, "stratum": list(stratum), **features}
        audit.write(json.dumps(row, ensure_ascii=False) + "\n")
        total += 1
        if budget:
            heap = heaps[stratum]
            entry = (priority, order, job)
            if len(heap) < budget:
                heapq.heappush(heap, entry)
            elif entry[:2] > heap[0][:2]:
                heapq.heapreplace(heap, entry)
    queues = [deque(sorted(heaps[key], reverse=True)) for key in sorted(heaps, key=lambda key: digest([seed, key]))]
    selected = []
    while queues and len(selected) < budget:
        next_round = []
        for queue in queues:
            if len(selected) == budget:
                break
            selected.append(queue.popleft())
            if queue:
                next_round.append(queue)
        queues = next_round
    selected.sort(key=lambda entry: entry[1])
    return [entry[2] for entry in selected], {
        "considered": total, "selected": len(selected), "budget": budget,
        "selected_scene_ids": [str(e[2][5].get("scene_id", e[2][0])) for e in selected],
        "protocol": "Input-only evidence priority within domain/observed-behavior strata; round-robin budget. Raw actions retained.",
    }
