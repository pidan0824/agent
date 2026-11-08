"""Build paired Query and Response supervision, individually or chronologically."""

import json
from itertools import groupby
from agent_simulation.data.catalog import DISPLAY_PRICE_SQL
from agent_simulation.agents.context.shopping import context, project_customer
from agent_simulation.agents.prompts.customer import messages, prepare_history
from agent_simulation.data.common import iter_rows, require
from agent_simulation.data.history_records import shopping_event, customer_history_limit
from agent_simulation.environment.batching import response_batches
from agent_simulation.agents.retrieval.history import DEFAULT_HISTORY_ENCODER, prepare_customer_history


def customer_sample(catalog, session, *, history_embedder=DEFAULT_HISTORY_ENCODER, stage=None, render=True):
    r = catalog.request(session, all_prior=True, include_displayed=stage != "query")
    return sample_from_request(
        catalog.config.dataset, session, r, history_embedder=history_embedder, stage=stage, render=render
    )


def sample_from_request(dataset, session, r, *, history_embedder=DEFAULT_HISTORY_ENCODER, stage=None, render=True,
                        prepared_history=None):
    """Build the observed stages, optionally leaving message rendering to the caller."""
    require(stage in (None, "query", "response"), "Unknown Customer stage")
    base = r["query_input"]
    prepared = prepared_history
    if prepared is None:
        prepared = prepare_customer_history(
            base["short_term_memory"]["search_history"], (),
            cutoff=base["history_cutoff_exclusive"], recent_n=customer_history_limit(),
            user_id=base["profile"].get("user_id"),
        )

    def stage_context(query):
        return context(
            base["profile"], base["short_term_memory"]["search_history"],
            cutoff=base["history_cutoff_exclusive"], history_embedder=history_embedder,
            history_query=query, prepared_history=prepared,
        )

    state = stage_context("") if stage != "response" else None
    shared = {
        "key": {
            "session_id": session,
            "user_id": base["profile"]["user_id"],
            "time_index": base["history_cutoff_exclusive"],
        },
        "split": r["assigned_split"],
        "tools": [],
        "source": {
            "dataset": dataset,
            "request_id": session,
        },
    }
    if "provenance" in base:
        shared["source"]["history_request_ids"] = base["provenance"]["request_ids"]
    query = None if stage == "response" else {
        **shared,
        "sample_id": f"query:{session}",
        "input": state,
        "target": r["query_target"],
        **({"messages": messages(state, r["query_target"])} if render else {}),
    }
    if stage == "query":
        return query, None
    response_state = {
        **stage_context(r["query_target"]["query"]),
        "query": r["query_target"]["query"],
        "displayed_items": r["response_input"]["displayed_items"],
    }
    response = {
        **shared,
        "sample_id": f"response:{session}",
        "tools": [],
        "input": response_state,
        "target": r["response_target"],
        **({"messages": messages(response_state, r["response_target"], response=True)} if render else {}),
    }
    return query, response


def split_supervision(sample):
    """Intersect original labels only after computing label-blind partitions."""
    batches = response_batches(sample["input"])
    history = prepare_history(project_customer(sample["input"])) if len(batches) > 1 else None
    result = []
    for index, state in enumerate(batches):
        selected = {p["item_id"] for p in state["displayed_items"]}
        target = {
            key: [i for i in value if i in selected] for key, value in sample["target"].items()
        }
        row = dict(sample)
        row.update(input=state, target=target,
                   messages=messages(state, target, response=True, prepared_history=history))
        row["batching"] = {
            "parent_sample_id": sample["sample_id"],
            "index": index + 1,
            "count": len(batches),
            "request_weight": 1 / len(batches),
        }
        if len(batches) > 1:
            row["sample_id"] = f"{sample['sample_id']}:batch:{index + 1}"
        result.append(row)
    for key in sample["target"]:
        require(
            set(i for row in result for i in row["target"][key]) == set(sample["target"][key]),
            "Batch labels do not reconstruct the original response",
        )
    return result



class CustomerHistory:
    def __init__(self, user, dataset, *, include_history_ids=False):
        self.user, self.dataset = user, dataset
        self.include_history_ids = include_history_ids
        self.all = []
        self.train = []
        self.last_time = -1
        self.prepared = {}

    def samples(self, cohort, *, split=None, stage=None, render=True):
        cutoff = cohort[0]["time_index"]
        require(cutoff > self.last_time, "Customer cohorts must be strictly chronological")
        require(
            all(r["user_id"] == self.user and r["time_index"] == cutoff for r in cohort),
            "Mixed Customer/time cohort",
        )
        require(len({r["session_id"] for r in cohort}) == len(cohort), "Duplicate request")
        new_events = []
        for row in sorted(cohort, key=lambda r: r["session_id"]):
            supervise = row["assigned_split"] is not None and (split is None or row["assigned_split"] == split)
            by_id = {p["item_id"]: p for p in row["displayed_products"]}
            needed = (
                set(row["impressed_item_ids"])
                if supervise and stage != "query"
                else set(row["clicked_item_ids"]) | set(row["purchased_item_ids"])
            )
            require(
                len(by_id) == len(row["displayed_products"])
                and needed <= set(by_id) <= set(row["impressed_item_ids"]),
                "Exposure/product join differs",
            )
            require(
                row["user_profile"] is not None and row["user_profile"]["user_id"] == self.user,
                "Missing verified user profile",
            )
            if supervise:
                source = "train" if row["split"] == "train" else "all"
                history = self.train if source == "train" else self.all
                request = {
                    "assigned_split": row["assigned_split"],
                    "query_input": {
                        "profile": row["user_profile"],
                        "history_cutoff_exclusive": cutoff,
                        "short_term_memory": {"search_history": history},
                    },
                    "query_target": {"query": row["query"]},
                    "response_input": {
                        "displayed_items": (
                            [by_id[i] for i in row["impressed_item_ids"]]
                            if stage != "query" else []
                        )
                    },
                    "response_target": {k: row[k] for k in ["clicked_item_ids", "purchased_item_ids"]},
                }
                if self.include_history_ids:
                    request["query_input"]["provenance"] = {
                        "request_ids": [event["session_id"] for event in history],
                    }
                prepared = self.prepared.get(source)
                if prepared is None:
                    prepared = self.prepared[source] = prepare_customer_history(
                        history, (), cutoff=cutoff, recent_n=customer_history_limit(), user_id=self.user)
                elif prepared.cutoff != cutoff:
                    prepared.update(history, (), cutoff=cutoff, append_only=True)
                yield sample_from_request(self.dataset, row["session_id"], request,
                                          stage=stage, render=render, prepared_history=prepared)
            new_events.append((row["split"], shopping_event(row, by_id)))
        # Equal-time requests are excluded from each other's history.
        for split, event in new_events:
            self.all.append(event)
            if split == "train":
                self.train.append(event)
        self.last_time = cutoff


def customer_records(catalog, bucket, buckets, *, split=None, stage=None, include_history_ids=False):
    """Keep a request together so capacity screening cannot retain a partial label."""
    require(0 <= bucket < buckets, "Invalid partition")
    cursor = catalog.connection.execute(
        f"""WITH selected AS (
          SELECT r.*,s.assigned_split FROM requests r
          JOIN customer_splits s USING(session_id) WHERE r.user_id % ? = ?
        ), displayed AS (
          SELECT o.session_id,to_json(list(struct_insert(p,price_raw := {DISPLAY_PRICE_SQL}))) AS products_json
          FROM ranking_observations o JOIN selected r USING(session_id)
          JOIN products p USING(item_id)
          WHERE (r.assigned_split IS NOT NULL AND (? IS NULL OR r.assigned_split=?) AND ?)
             OR list_contains(r.clicked_item_ids,o.item_id)
             OR list_contains(r.purchased_item_ids,o.item_id)
          GROUP BY o.session_id
        )
        SELECT r.*,to_json(u) AS profile_json,d.products_json
        FROM selected r
        LEFT JOIN users u ON u.user_id=r.user_id LEFT JOIN displayed d ON d.session_id=r.session_id
        ORDER BY r.user_id,r.time_index,r.session_id""",
        [buckets, bucket, split, split, stage != "query"],
    )

    def decode():
        for row in iter_rows(cursor, 64):
            row["user_profile"] = json.loads(row.pop("profile_json"))
            raw = row.pop("products_json")
            row["displayed_products"] = json.loads(raw) if raw is not None else []
            yield row

    for user, group in groupby(decode(), key=lambda r: r["user_id"]):
        state = CustomerHistory(user, catalog.config.dataset, include_history_ids=include_history_ids)
        for _, cohort in groupby(group, key=lambda r: r["time_index"]):
            yield from state.samples(list(cohort), split=split, stage=stage, render=False)
