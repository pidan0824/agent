"""Build a market search request and run the selected search backend."""

from copy import deepcopy
import hashlib
import json
import random
import re

from agent_simulation.data.common import require


def tokens(text):
    """Use CJK characters and Latin words for a dependency-free lexical baseline."""
    return re.findall(r"[\u3400-\u9fff]|[a-z0-9]+", text.lower())


def index_titles(connection, products):
    """Index market titles for the local BM25 backend."""
    connection.execute("CREATE VIRTUAL TABLE titles USING fts5(text)")
    connection.executemany(
        "INSERT INTO titles(rowid,text) VALUES (?,?)",
        ((int(item), " ".join(tokens(product["item_title"]))) for item, product in products.items()),
    )


def _search_titles(search_index, query, top_k):
    words = tokens(query)
    require(bool(words), "Query has no searchable text")
    match = " OR ".join('"' + w + '"' for w in dict.fromkeys(words))
    return [
        r[0]
        for r in search_index.execute(
            "SELECT rowid FROM titles WHERE titles MATCH ? ORDER BY bm25(titles),rowid LIMIT ?",
            (match, top_k),
        )
    ]


def random_list(state, actor, query):
    """Sample candidates and shuffle them independently for a reproducible wiring test."""
    config = state["search_config"]
    request_number = len(state["customers"][str(actor)]["simulation_history"])
    key = json.dumps([config["seed"], actor, request_number, query])

    def generator(stage):
        return random.Random(int.from_bytes(hashlib.sha256((key + stage).encode()).digest(), "big"))

    pool = sorted(int(i) for i in state["products"])
    candidates = generator("recall").sample(pool, min(config["candidate_size"], len(pool)))
    ordered = candidates.copy()
    generator("ranking").shuffle(ordered)
    return ordered[: state["top_k"]], candidates


def request_context(state, actor, query):
    """Build a search request from settled clicks and the current market."""
    customer = state["customers"][str(actor)]
    history = [*customer["observed_history"], *customer["simulation_history"]]
    return {
        "user_id": actor,
        "query": query,
        "request_id": state["simulation_step"] + 1,
        "recently_clicked_item_ids": [p["item_id"] for e in history for p in e["clicked_items"]],
        "market_item_ids": sorted(int(i) for i in state["products"]),
        "active_prices": deepcopy(state["active_prices"]),
        "products": deepcopy(state["products"]),
    }


def search_market(state, actor, query, search_index, pipeline):
    """Return ordered IDs and the trace for the configured search backend."""
    backend = state["search_config"]["backend"]
    if backend == "kuaisearch":
        ids, trace = pipeline.search(
            request_context(state, actor, query), state["search_config"]["candidate_size"], state["top_k"]
        )
        return ids, {"backend": "kuaisearch_recall_ranking_reranking", "platform_trace": trace}
    if backend == "random":
        ids, candidates = random_list(state, actor, query)
        return ids, {"backend": "random_recall_random_ranking", "candidate_item_ids": candidates,
                     "random_seed": state["search_config"]["seed"]}
    if backend == "bm25":
        return _search_titles(search_index, query, state["top_k"]), {"backend": "local_title_FTS5_BM25"}
    raise ValueError("Unknown search backend: " + str(backend))
