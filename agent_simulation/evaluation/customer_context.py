"""Shared frozen Customer contexts for scene previews, evaluation and generation."""
from collections import OrderedDict
from copy import deepcopy
from typing import NamedTuple
import unicodedata

from agent_simulation.agents.agentic import AgenticOptions
from agent_simulation.data.common import require
from agent_simulation.evaluation.micro.customer_agentic import FrozenSampleTools


def normalize_query(value):
    return " ".join(unicodedata.normalize("NFKC", value or "").casefold().split())


def resolve_context_options(cot_preset, modes, history_token_budget=None):
    """One preset contract for preprocessing and the generation entry point."""
    require(cot_preset in ("original", "lightweight"), "Unknown CoT preset")
    require(modes and len(set(modes)) == len(modes) and set(modes) <= {"fixed", "autonomous"},
            "Choose distinct supported context modes")
    light = cot_preset == "lightweight"
    if light and history_token_budget is None:
        history_token_budget = 2048
    require(history_token_budget is None or
            (type(history_token_budget) is int and history_token_budget >= 0 and tuple(modes) == ("fixed",)),
            "History token budget requires a nonnegative limit and fixed mode")
    require(not light or tuple(modes) == ("fixed",), "Lightweight preset requires fixed mode")
    return {"concise": light, "history_token_budget": history_token_budget}


class HistoryRelevance(NamedTuple):
    same_item: bool
    same_query: bool
    lexical: float = 0.
    category: int = 0


def history_relevance(context, *, include_weak=True):
    """Cheap input-only recall: exact evidence first, lexical/category hints next."""
    import re
    from functools import lru_cache
    @lru_cache(maxsize=1024)
    def grams(text):
        return frozenset(word[i:i + 2] for word in re.findall(r"[^\W_]+", normalize_query(text))
                         for i in range(len(word) - 1) if not word[i:i + 2].isdigit())
    def categories(product):
        return {(level, normalize_query(product.get(f"category_level{level}_name"))) for level in (2, 3)
                if normalize_query(product.get(f"category_level{level}_name")) not in ("", "unknown", "null")}
    displayed = context.get("displayed_items", [])
    ids = {p["item_id"] for p in displayed}
    query = normalize_query(context.get("query"))
    references, category_keys = set(), set()
    if include_weak:
        references = {grams(context.get("query") or "")}
        references.update(grams(p.get("item_title") or "") for p in displayed)
        references.discard(frozenset())
        category_keys = set().union(*(categories(p) for p in displayed))
    def score(event):
        products = [p for field in ("clicked_items", "purchased_items") for p in (event.get(field) or [])]
        same_item = any(p.get("item_id") in ids for p in products)
        same_query = bool(query) and normalize_query(event.get("query")) == query
        if not include_weak:
            return HistoryRelevance(same_item, same_query)
        lexical = 0.
        for text in [event.get("query"), *(p.get("item_title") for p in products)]:
            tokens = grams(text or "")
            for reference in references:
                shared = len(tokens & reference)
                if shared >= 2:
                    similarity = shared / len(tokens | reference)
                    if similarity >= .3:
                        lexical = max(lexical, similarity)
        category = max((level for p in products for level, name in categories(p)
                        if (level, name) in category_keys), default=0)
        return HistoryRelevance(same_item, same_query, lexical, category)
    return score


def budget_history(context, history, *, stage, tokenizer, budget, event_limit=None, relevance=None):
    """Fit complete search events and snapshot item details in one history budget."""
    from agent_simulation.evaluation.customer_evidence import attach_sources, encode_context
    from agent_simulation.evaluation.customer_background import match_priority
    # Full retained ID sequences remain visible, even if a detail record cannot fit.
    base = attach_sources({k: deepcopy(v) for k, v in context.items() if k != "observed_history"})
    details = {key: fact for key, fact in base["facts"].items()
               if fact["kind"] == "background" and fact["field"] == "item_history"}
    base["facts"] = {key: fact for key, fact in base["facts"].items() if key not in details}
    def render(events, items=()):
        value = deepcopy(base)
        value["facts"].update({key: details[key] for key in items})
        value["observed_history"] = sorted(events, key=lambda e: (e["time_index"], e["session_id"]))
        value["selected_history_events"] = len(events)
        value["history_token_budget"] = budget
        if details:
            value["snapshot_detail_selection"] = {"available": len(details), "shown": len(items)}
        return attach_sources(value)
    empty = render([])
    if not budget or not (history or details):
        return empty
    def tokens(value):
        return len(tokenizer.encode(encode_context(value), add_special_tokens=False))
    baseline = tokens(empty)
    if event_limit is None or len(history) <= event_limit:
        complete = render(history, details)
        if tokens(complete) - baseline <= budget:
            return complete
    if relevance is None:
        relevance = history_relevance(context) if stage == "response" else lambda event: HistoryRelevance(False, False)
    latest = max(history, key=lambda e: (e["time_index"], e["session_id"])) if history else None
    def priority(event):
        same_item, same_query, lexical, category = relevance(event)
        return (event is not latest, not same_item, not same_query, -lexical, -category,
                -event["time_index"], event["session_id"])
    units = [(priority(event), "event", event) for event in history]
    for key, fact in details.items():
        item = fact["value"]
        matches = [match_priority(product, item) for product in context.get("displayed_items", [])]
        same, brand, shop, lexical, cat3, cat2 = max(matches, default=(False, False, False, 0., False, False))
        units.append(((True, not same, True, -(brand + shop + lexical), -(3 if cat3 else 2 if cat2 else 0),
                       -fact["time_index"], item["item_id"]), "detail", key))
    selected, shown, result = [], [], empty
    for _, kind, value in sorted(units, key=lambda unit: unit[0]):
        if kind == "event" and event_limit is not None and len(selected) >= event_limit:
            continue
        events = [*selected, value] if kind == "event" else selected
        items = [*shown, value] if kind == "detail" else shown
        proposed = render(events, items)
        if tokens(proposed) - baseline <= budget:
            selected, shown, result = events, items, proposed
    return result


def select_history(history, limit, relevance, *, budgeted):
    """Recall bounded older evidence; the token fitter makes the final selection."""
    recent = history[-limit:] if limit else []
    if len(history) <= limit or limit <= 1:
        return recent
    scored = [(event, relevance(event)) for event in history[:-limit]]
    exact = sorted((pair for pair in scored if pair[1].same_item or pair[1].same_query),
                   key=lambda pair: (not pair[1].same_item, -pair[0]["time_index"], pair[0]["session_id"]))
    recalled = [event for event, _ in exact[:min(2, limit - 1)]]
    if not budgeted:
        return sorted(recalled + recent[len(recalled):], key=lambda e: (e["time_index"], e["session_id"]))
    queries = sorted((event for event, score in scored if score.same_query),
                     key=lambda e: (-e["time_index"], e["session_id"]))
    weak = sorted((pair for pair in scored if not (pair[1].same_item or pair[1].same_query)
                   and (pair[1].lexical or pair[1].category)),
                  key=lambda pair: (-pair[1].lexical, -pair[1].category,
                                    -pair[0]["time_index"], pair[0]["session_id"]))
    candidates = {event["session_id"]: event for event in
                  recent + recalled + queries[:2] + [event for event, _ in weak[:2]]}
    return list(candidates.values())


class ContextCache:
    """Run-local bounded cache of input-only contexts, shared across curation stages."""
    def __init__(self, maxsize=128):
        require(type(maxsize) is int and maxsize > 0, "Positive context cache size required")
        self.maxsize = maxsize
        self._contexts = OrderedDict()

    def get(self, key, tokenizer):
        entry = self._contexts.get(key)
        if entry is None or entry[0] is not tokenizer:
            return None
        self._contexts.move_to_end(key)
        return deepcopy(entry[1])

    def put(self, key, tokenizer, context):
        # Keep the tokenizer alive so its object identity cannot be recycled.
        self._contexts[key] = (tokenizer, deepcopy(context))
        self._contexts.move_to_end(key)
        while len(self._contexts) > self.maxsize:
            self._contexts.popitem(last=False)


class CurationTools(FrozenSampleTools):
    def __init__(self, request, stage, *, mode, theory=True, history_limit=10, concise=False,
                 history_token_budget=None, tokenizer=None, context_cache=None):
        super().__init__(request, stage, theory=theory)
        require(mode in ("fixed", "autonomous"), "Unknown information mode")
        require(type(history_limit) is int and history_limit >= 0, "Nonnegative integer history limit required")
        self.mode, self.history_limit = mode, history_limit
        require(history_token_budget is None or (type(history_token_budget) is int and history_token_budget >= 0),
                "Nonnegative history token budget required")
        require(history_token_budget is None or mode == "fixed", "History token budget requires fixed mode")
        require(not history_token_budget or callable(getattr(tokenizer, "encode", None)),
                "History token budget requires the model tokenizer")
        self.history_token_budget, self.tokenizer = history_token_budget, tokenizer
        self._budget_context = None
        self._context_cache = context_cache if mode == "fixed" else None
        self._context_key = (self.fingerprint, stage, mode, history_limit,
                             history_token_budget, id(tokenizer))
        self.concise = concise
        if mode == "fixed":
            self.schemas = []
            self.parameters = {}

    def initial_context(self):
        if self._budget_context is not None:
            self.guard()
            return deepcopy(self._budget_context)
        if self._context_cache is not None:
            self.guard()
            cached = self._context_cache.get(self._context_key, self.tokenizer)
            if cached is not None:
                self._budget_context = cached
                return deepcopy(cached)
        context = self._base_context()
        relevance = None
        context["information_mode"] = self.mode
        if "candidate_protocol" in self.context:
            context["candidate_protocol"] = self.context["candidate_protocol"]
        if self.mode == "fixed":
            history = self.history[-self.history_limit:] if self.history_limit else []
            if self.stage == "response":
                relevance = history_relevance(self.context, include_weak=self.history_token_budget is not None)
                history = select_history(self.history, self.history_limit, relevance,
                                         budgeted=self.history_token_budget is not None)
                context["displayed_items"] = deepcopy(self.displayed)
                context["selected_history_events"] = len(history)
                context["history_scope"] = (
                    "Selected events from available history; unshown events may contain other actions.")
            context["observed_history"] = deepcopy(history)
        if self.history_token_budget is not None:
            context = budget_history(context, history, stage=self.stage,
                tokenizer=self.tokenizer, budget=self.history_token_budget, event_limit=self.history_limit,
                relevance=relevance)
        else:
            from agent_simulation.evaluation.customer_evidence import attach_sources
            context = attach_sources(context)
        if self.history_token_budget is not None or self._context_cache is not None:
            self._budget_context = context
            if self._context_cache is not None:
                self._context_cache.put(self._context_key, self.tokenizer, context)
            return deepcopy(context)
        return context


def fixed_response_requests(response_input, *, history_limit=10, history_token_budget=None, tokenizer=None):
    """Use the exact fixed curation prompts and schemas in controlled evaluation."""
    from agent_simulation.agents.agentic import initial_agentic_request, teacher_output
    from agent_simulation.evaluation.trajectories import trajectory_messages
    result = {}
    for style in ("direct", "ekb"):
        theory = style == "ekb"
        binding = CurationTools({"response_input": response_input}, "response",
                               mode="fixed", theory=theory, history_limit=history_limit,
                               history_token_budget=history_token_budget, tokenizer=tokenizer)
        schemas, messages = trajectory_messages(binding, role="customer", stage="response", theory=theory)
        messages, tools, _ = initial_agentic_request(messages, [], AgenticOptions(max_model_calls=1, theory=theory))
        result[style] = {"messages": messages, "tools": tools,
                        "final_schema": teacher_output(schemas, customer_ekb=True) if theory else schemas[0]}
    return binding.initial_context(), result
