"""Build real price supervision, individually or with incremental Seller history."""

from itertools import groupby
from agent_simulation.inference.requests import ContextCapacityError
from agent_simulation.data.market.quotes import CategorySnapshots
from agent_simulation.data.common import iter_rows, require, rows
from agent_simulation.agents.context.pricing import (
    PricingSnapshot, attach_retrieved_products, build_pricing_context, overview_context, target_context,
)
from agent_simulation.data.history_records import (
    ObservedPriceHistory, has_observed_price, price_point, seller_history_policy,
)
from agent_simulation.agents.prompts.seller import pricing_messages
from agent_simulation.agents.retrieval.history import (
    DEFAULT_HISTORY_ENCODER, SellerItemWindow, PreparedSellerHistory, CachedHistoryEncoder,
)


def seller_sample(catalog, seller, item, cutoff, *, history_embedder=DEFAULT_HISTORY_ENCODER):
    targets = rows(
        catalog.connection,
        """SELECT b.* ,t.assigned_split FROM seller_history b
        JOIN seller_targets t USING(seller_id,item_id,time_index)
        WHERE b.seller_id=? AND b.item_id=? AND b.time_index=?""",
        [seller, item, cutoff],
    )
    require(len(targets) == 1, "Missing reliable item-time target with earlier observation")
    target = targets[0]
    events = rows(
        catalog.connection,
        """SELECT * FROM seller_history WHERE seller_id=? AND time_index<?
        ORDER BY time_index,item_id""",
        [seller, cutoff],
    )
    eligible = {r["item_id"] for r in events if has_observed_price(r)}
    events = [r for r in events if r["item_id"] in eligible]
    state, _ = build_pricing_context(
        events,
        owner_id=seller,
        cutoff=cutoff,
        observations_per_item=seller_history_policy()["observations_per_item"],
        public_pages=[],
        requested_item_ids=[item],
    )
    state = target_context(overview_context(state, [item]), [item])
    return with_market_context(
        catalog,
        sample_from_context(catalog.config.dataset, target, state, render=False),
        observed=events,
        history_embedder=history_embedder,
    )


def with_market_context(
    catalog,
    sample,
    *,
    connection=None,
    observed=None,
    retrieved_history=None,
    shared_market=None,
    product_lookup=None,
    category_snapshots=None,
    history_embedder=DEFAULT_HISTORY_ENCODER,
):
    """Use the same market evidence contract in direct and streaming sample readers."""
    from agent_simulation.data.catalog import DataRepository
    from agent_simulation.agents.context.market import load_market_context

    seller, cutoff = sample["source"]["seller_id"], sample["source"]["time_index"]
    cache_key = (seller, cutoff)
    if shared_market is not None and shared_market.get("key") == cache_key:
        data = shared_market["data"]
    else:
        data = DataRepository(
            connection if connection is not None else catalog.connection,
            catalog.config, cutoff,
            history_points=seller_history_policy()["observations_per_item"],
            category_snapshots=category_snapshots,
        )
        if shared_market is not None:
            shared_market.clear()
            shared_market.update(key=cache_key, data=data)
    context = sample["input"]
    load_market_context(context, data,
                        history_embedder=history_embedder, product_lookup=product_lookup)
    from agent_simulation.agents.retrieval.history import seller_history

    if retrieved_history is None:
        if observed is None:
            observed = rows(
                data.connection,
                "SELECT * FROM seller_history WHERE seller_id=? AND time_index<? ORDER BY time_index,item_id",
                [seller, cutoff],
            )
        # Selection narrows the model view, not the authorized retrieval pool.
        owned = {r["item_id"] for r in observed if has_observed_price(r)}
        retrieved_history = seller_history(
            [r for r in observed if r["item_id"] in owned],
            seller_id=seller,
            cutoff=cutoff,
            recent_n=seller_history_policy()["observations_per_item"],
            focus_ids=sample["input"]["requested_price_item_ids"],
            embedder=history_embedder,
        )
    sample["input"]["retrieved_context"] = {"history": retrieved_history}
    attach_retrieved_products(sample["input"], product_lookup or data.product)
    sample["messages"] = pricing_messages(sample["input"], sample["target"])
    return sample


def sample_from_context(dataset, target, state, *, render=True):
    """Serialize one target using shared SFT/runtime messages, without query labels."""
    seller, item, cutoff = (target[k] for k in ("seller_id", "item_id", "time_index"))
    require(
        state["requested_price_item_ids"] == [item], "SFT must supervise one observed item-time"
    )
    require(
        item in state["history_presentation"]["detail_item_ids"], "Target history must be visible"
    )
    answer = {"prices": [{"item_id": item, "price_raw": target["price_raw"]}]}
    return {
        "sample_id": f"seller:{seller}:{item}:{cutoff}",
        "split": target["assigned_split"],
        "input": state,
        "target": answer,
        **({"messages": pricing_messages(state, answer)} if render else {}),
        "tools": [],
        "source": {
            "dataset": dataset,
            "seller_id": seller,
            "item_id": item,
            "time_index": cutoff,
            "price_status": target["price_status"],
        },
    }


class SellerHistory:
    """Process complete time-index cohorts; targets never see same-index feedback."""

    def __init__(self, seller, dataset):
        self.seller, self.dataset = seller, dataset
        self.limit = seller_history_policy()["observations_per_item"]
        self.windows = {}
        self.valid_references = {}
        self.first_observation = {}
        self.profiles = {}
        self.price_histories = {}
        self._first_eligible = None
        self.last_time = -1
        self._retrieval_encoder = None
        self._retrieval_pool = None

    def samples(self, cohort, *, split=None, render=True):
        """Exhaust this iterator before processing the next cohort.

        Until then, retrieval and product lookups use the prefix before this cohort.
        """
        require(bool(cohort), "Empty Seller cohort")
        cutoff = cohort[0]["time_index"]
        require(cutoff > self.last_time, "Seller cohorts must be strictly chronological")
        require(
            all(r["seller_id"] == self.seller and r["time_index"] == cutoff for r in cohort),
            "Mixed Seller/time cohort",
        )
        require(len({r["item_id"] for r in cohort}) == len(cohort), "Duplicate item-time")
        targets = [
            r for r in cohort
            if r.get("assigned_split") is not None
            and (split is None or r["assigned_split"] == split)
        ]
        # Render only requested items; other eligible products stay in the
        # retrieval pool. Common-period shop totals are loaded with market context.
        retained = [r for item in sorted({t['item_id'] for t in targets})
                    if item in self.valid_references for r in self.windows[item].recent_rows()]
        if targets:
            state, _ = build_pricing_context(
                retained,
                owner_id=self.seller,
                cutoff=cutoff,
                observations_per_item=self.limit,
                public_pages=[],
                requested_item_ids=[],
                valid_references=self.valid_references,
                price_histories=self.price_histories,
            )
            state["profile"]["seller_name"] = self._first_eligible[2]
            state = overview_context(state, [r["item_id"] for r in targets])
            snapshot = PricingSnapshot(state)
        for target in targets:
            require(
                target["item_id"] in self.valid_references and has_observed_price(target),
                "Target requires reliable price and a strictly earlier observation",
            )
            projected = snapshot.view([target["item_id"]])
            yield sample_from_context(self.dataset, target, projected, render=render)
        # Update only after every target at this time has been constructed.
        for row in sorted(cohort, key=lambda r: r["item_id"]):
            item = row["item_id"]
            if item not in self.price_histories:
                self.price_histories[item] = ObservedPriceHistory(self.seller, item)
            self.price_histories[item].append(row)
            self.first_observation.setdefault(item, (cutoff, item, row["seller_name"]))
            if has_observed_price(row):
                self.valid_references[item] = price_point(row)
                first = self.first_observation[item]
                if self._first_eligible is None or first < self._first_eligible:
                    self._first_eligible = first
            if item not in self.windows:
                self.windows[item] = SellerItemWindow(item, self.limit)
            window = self.windows[item]
            window.append("observed_time_index", row)
            self.profiles[item] = row
        self.last_time = cutoff
        self._retrieval_pool = None

    def product_profile(self, item):
        """Latest consumed profile, copied by the caller into selected evidence only."""
        require(item in self.valid_references, "Unknown eligible Seller product")
        return self.profiles[item]

    def retrieve(self, cutoff, focus_ids, *, embedder=DEFAULT_HISTORY_ENCODER):
        """Read only the prefix consumed before this supervision cohort."""
        require(cutoff > self.last_time, "Retrieval cutoff must follow the consumed history")
        if self._retrieval_pool is None:
            recent = {item: window.recent_rows() for item, window in self.windows.items()
                      if item in self.valid_references}
            profiles = {item: rows[-1] for item, rows in recent.items()}
            segments = [segment for item in recent
                        for segment in self.windows[item].segments(include_recent=True)]
            features = {item: self.windows[item].recent_features() for item in recent}
            self._retrieval_pool = PreparedSellerHistory(
                segments, recent, profiles, self.seller, cutoff, window_features=features)
        if embedder is not None:
            if self._retrieval_encoder is None or self._retrieval_encoder.embedder is not embedder:
                self._retrieval_encoder = CachedHistoryEncoder(embedder)
            embedder = self._retrieval_encoder
        return self._retrieval_pool.retrieve(focus_ids, embedder=embedder)


def seller_records(catalog, bucket, buckets, *, split=None, on_capacity_error=None):
    """Yield chronological contexts; read only each target's recent quote window."""
    require(0 <= bucket < buckets, "Invalid partition")
    cursor = catalog.connection.execute(
        """SELECT h.*,t.assigned_split FROM seller_history h
        LEFT JOIN seller_targets t ON h.seller_id=t.seller_id AND h.item_id=t.item_id
          AND h.time_index=t.time_index AND (? IS NULL OR t.assigned_split=?)
        WHERE h.seller_id % ? = ? ORDER BY h.seller_id,h.time_index,h.item_id""",
        [split, split, buckets, bucket],
    )
    # A separate cursor prevents market lookups from invalidating the streaming source scan.
    market_connection = catalog.connection.cursor()
    category_snapshots = CategorySnapshots()
    try:
        for seller, group in groupby(iter_rows(cursor), key=lambda r: r["seller_id"]):
            state = SellerHistory(seller, catalog.config.dataset)
            for _, cohort in groupby(group, key=lambda r: r["time_index"]):
                rows = list(cohort)
                # Every target in this shop/time cohort sees the same shop and market.
                shared_market = {} if sum(r["assigned_split"] is not None for r in rows) > 1 else None
                for sample in state.samples(rows, render=False):
                    try:
                        with_market_context(
                            catalog,
                            sample,
                            connection=market_connection,
                            shared_market=shared_market,
                            category_snapshots=category_snapshots,
                            product_lookup=state.product_profile,
                            retrieved_history=state.retrieve(
                                sample["source"]["time_index"],
                                sample["input"]["requested_price_item_ids"],
                            ),
                        )
                    except ContextCapacityError as error:
                        if on_capacity_error is None:
                            raise
                        on_capacity_error(sample["sample_id"], sample["split"], error)
                        continue
                    except ValueError as error:
                        raise ValueError(sample["sample_id"] + ": " + str(error)) from error
                    yield sample
    finally:
        market_connection.close()
