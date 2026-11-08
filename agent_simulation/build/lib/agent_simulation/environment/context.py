"""Assemble role-specific inputs from market state, history and memory."""

from copy import deepcopy

from agent_simulation.data.common import require
from agent_simulation.agents.context.shopping import context as customer_context
from agent_simulation.agents.context.pricing import (
    attach_retrieved_products, build_pricing_context, overview_context,
)
from agent_simulation.agents.memory.evidence import actor_key
from agent_simulation.agents.context.shopping import shopping_history_event
from agent_simulation.environment.operations import reporting as seller_reports
from agent_simulation.agents.retrieval.history import (
    DEFAULT_HISTORY_ENCODER, PreparedCustomerHistory,
)
from agent_simulation.environment.batching import response_batches


class AgentContextBuilder:
    def __init__(self, runtime):
        self.runtime = runtime
        self.history_embedder = DEFAULT_HISTORY_ENCODER
        self._customer_history = {}
        self._customer_observed = {}

    def use_semantic_history(self):
        """Use the existing local BGE for older factual segments as well as memory."""
        self.runtime.long_term_memory.ensure_encoder()
        self.history_embedder = self.runtime.long_term_memory.embedder

    def _history_encoder(self):
        """Share one bound encoder for factual history and long-term experiences."""
        if self.history_embedder is DEFAULT_HISTORY_ENCODER:
            self.use_semantic_history()
        return self.history_embedder

    def seller(self, actor_id, *, public_pages=(), detail_item_ids=(), retrieve=True, observe=True,
               state=None, include_market=True):
        """Build a shop view. Empty detail_item_ids selects all registered products."""
        if state is None:
            state = self.runtime.snapshot()
        context, coverage, history_index = self._seller_overview(
            actor_id, state, public_pages=public_pages, detail_item_ids=detail_item_ids,
            retrieve=retrieve,
        )
        if include_market:
            from agent_simulation.agents.context.market import load_market_context

            load_market_context(context, self.runtime.data, simulation=state,
                                history_embedder=self._history_encoder())
        if retrieve:
            self._attach_seller_evidence(actor_id, context, state=state, history_index=history_index)
        if observe:
            self.runtime.long_term_memory.observe_context("seller", actor_id, context, state)
        return context, coverage

    def _seller_records(self, actor_id, state):
        owned = [int(i) for i, p in state["products"].items() if p["seller_id"] == actor_id]
        require(bool(owned), "Seller is not registered in this market")
        observed = [r for i in sorted(owned) for r in self.runtime.data.records(i)]
        feedback = [r for r in state["feedback"] if r["seller_id"] == actor_id]
        return owned, observed, feedback

    def _seller_history_index(self, actor_id, state, observed, feedback):
        from agent_simulation.agents.retrieval.history import prepare_seller_history

        return prepare_seller_history(
            observed, seller_id=actor_id, cutoff=self.runtime.data.cutoff,
            recent_n=state["history_per_item"], simulated=feedback,
            simulation_before=state["simulation_step"] + 1,
        )

    def _seller_overview(self, actor_id, state, *, public_pages=(), detail_item_ids=(), retrieve):
        """Share one collection of records between the shop view and retrieval."""
        owned, observed, feedback = self._seller_records(actor_id, state)
        context, coverage = build_pricing_context(
            observed,
            owner_id=actor_id,
            cutoff=self.runtime.data.cutoff,
            observations_per_item=state["history_per_item"],
            public_pages=public_pages,
            requested_item_ids=[],
            simulated=feedback,
            simulation_cutoff=state["simulation_step"] + 1,
            active_prices={str(i): state["active_prices"][str(i)] for i in owned},
        )
        # Every visible product keeps its recent window in automatic retrieval.
        selected = detail_item_ids or owned
        context = overview_context(context, selected)
        context["shop_scope"] = {
            "decision_products": "This seller's products registered in the simulation market.",
            "recent_and_retrieved_history": "Registered products only; observed and simulated facts use separate clocks.",
            "settled_shop_period": "Registered products only; settled reporting periods are separate from observed history.",
            "observed_shop_period": "All historically observed products of this seller, including products outside this simulation market.",
        }
        context.pop("requested_price_item_ids")
        context["price_scope"] = (
            "Decide which owned prices to change; an empty prices list requests no changes."
        )
        if "clock" in state:
            context["decision_time"] = self.decision_time(state, "seller", actor_id)
            report = seller_reports.read(state, actor_id)
            context["business_report"] = {
                k: report[k]
                for k in (
                    "logical_window",
                    "shop_totals",
                    "shop_cumulative_totals",
                    "shop_exposed_requests",
                    "quantity_assumption",
                )
            }
            products = {p["item_id"]: p for p in report["products"]}
            for product in context["owned_products"]:
                row = products[product["item_id"]]
                spans = row["activity_intervals"]
                product["new_feedback"] = {
                    k: row[k] for k in (*seller_reports.COUNTS, "estimated_units_sold")
                }
                product["activity_intervals"] = spans[-state["history_per_item"] :]
                product["earlier_intervals_available"] = len(spans) > state["history_per_item"]
        history_index = self._seller_history_index(actor_id, state, observed, feedback) if retrieve else None
        return context, coverage, history_index

    def _attach_seller_evidence(self, actor_id, context, *, state, history_index, memory_tokenizer=None):
        """Enrich a private target view from the authorized whole-shop history."""
        actor_key(state, "seller", actor_id)
        products = state["products"]
        batch_ids = {product["item_id"] for product in context["owned_products"]}
        focal = context["history_presentation"]["detail_item_ids"]
        require(
            bool(batch_ids) and set(focal) <= batch_ids
            and all(products.get(str(item), {}).get("seller_id") == actor_id for item in batch_ids),
            "Foreign Seller batch product",
        )
        context["retrieved_context"] = {
            "history": history_index.retrieve(focal, embedder=self._history_encoder()),
        }
        context["long_term_memory"] = self.runtime.long_term_memory.context(
            "seller", actor_id, context, state=state, tokenizer=memory_tokenizer
        )
        attach_retrieved_products(context, lambda item: products[str(item)])

    def seller_targets(self, actor_id, *, state, batch_size=1, memory_tokenizer=None):
        """Prepare once, select targets first, then read only their market evidence.

        Every yielded decision uses the same snapshot. The caller stages all
        prices and commits at the existing Seller/episode barrier.
        """
        from agent_simulation.agents.context.market import load_market_context, simulation_market_cache
        from agent_simulation.agents.context.pricing import PricingSnapshot

        require(type(batch_size) is int and 1 <= batch_size <= 16, "Expected 1..16 price targets")
        overview, _, index = self._seller_overview(actor_id, state, retrieve=True)
        snapshot = PricingSnapshot(overview)
        owned = list(snapshot.products)
        simulation_cache = simulation_market_cache(state)
        count = (len(owned) + batch_size - 1) // batch_size
        for start in range(0, len(owned), batch_size):
            context = snapshot.view(owned[start:start + batch_size])
            context["product_batch"] = {"index": start // batch_size + 1, "count": count,
                                        "shop_product_count": len(owned)}
            context["price_scope"] = "Submit one price per requested_price_item_ids entry, including unchanged prices."
            load_market_context(context, self.runtime.data, simulation=state,
                simulation_cache=simulation_cache, history_embedder=self._history_encoder())
            self._attach_seller_evidence(actor_id, context, state=state, history_index=index,
                                         memory_tokenizer=memory_tokenizer)
            yield context

    def _customer_observed_history(self, actor_id, seed):
        """Reuse observed facts while the read-only catalog, cutoff and seed agree."""
        catalog = self.runtime.catalog
        cutoff = self.runtime.data.cutoff
        cached = self._customer_observed.get(actor_id)
        if cached is not None:
            source, end, previous_seed, profile, pool = cached
            if source is catalog and end == cutoff and previous_seed == seed:
                return profile, pool

        observed = catalog.customer_history(actor_id, cutoff, official_split="test", all_prior=True)
        seed = deepcopy(seed)
        # For shared sessions the catalog supplies the canonical product snapshots.
        combined = {event["session_id"]: event for event in seed}
        combined.update(
            (event["session_id"], event) for event in observed["short_term_memory"]["search_history"]
        )
        pool = sorted(combined.values(), key=lambda e: (e["time_index"], e["session_id"]))
        profile = observed["profile"]
        self._customer_observed[actor_id] = catalog, cutoff, seed, profile, pool
        return profile, pool

    def _prepare_customer_history(self, actor_id, state):
        customer = state["customers"][str(actor_id)]
        profile, pool = self._customer_observed_history(actor_id, customer["observed_history"])
        simulated = [shopping_history_event(event) for event in customer["simulation_history"]]
        key = (actor_id, state["customer_history_limit"])
        if key not in self._customer_history:
            self._customer_history[key] = PreparedCustomerHistory(*key)
        return profile, self._customer_history[key].update(
            pool, simulated, cutoff=self.runtime.data.cutoff,
            simulation_before=state["simulation_step"] + 1,
        )

    def reflection_evidence(self, role, actor, *, state):
        """Collect the factual sources that can be cited during reflection."""
        actor_key(state, role, actor)
        if role == "customer":
            require(self.runtime.catalog is not None, "Customer memory needs a verified profile")
            profile, history = self._prepare_customer_history(actor, state)
            facts = history.retrieve(embedder=self._history_encoder(), diagnostics=False)
            return profile, {"retrieved_context": {"history": facts}}

        from agent_simulation.agents.context.market import load_market_context

        owned, observed, feedback = self._seller_records(actor, state)
        history = self._seller_history_index(actor, state, observed, feedback)
        profile = {
            "seller_id": actor,
            "products": [state["products"][str(item)] for item in owned],
        }
        context = {
            "profile": {"seller_id": actor},
            "history_cutoff_exclusive": self.runtime.data.cutoff,
            "owned_products": [
                {"item_id": item, "item_title": history.profiles[item]["item_title"]}
                for item in sorted(owned)
            ],
        }
        load_market_context(context, self.runtime.data, simulation=state,
                            history_embedder=self._history_encoder())
        context["retrieved_context"] = {
            "history": history.retrieve(owned, embedder=self._history_encoder()),
        }
        return profile, context

    def customer_responses(self, actor_id, *, state=None, batch_size=None,
                           expected_request_id=None, observe=True, memory_tokenizer=None):
        """Build one response context, then partition its ordered display list."""
        context = self.customer(
            actor_id, response=True, expected_request_id=expected_request_id, observe=observe,
            state=state, memory_tokenizer=memory_tokenizer,
        )
        return response_batches(context, max_items=batch_size)

    def customer(self, actor_id, *, response=False, expected_request_id=None, observe=True,
                 state=None, memory_tokenizer=None):
        require(
            self.runtime.catalog is not None,
            "Customer model input requires the verified catalog profile",
        )
        state = self.runtime.snapshot() if state is None else state
        actor_key(state, "customer", actor_id)
        pending = state["pending"].get(str(actor_id))
        if response:
            require(pending is not None, "No pending list for response input")
        if expected_request_id is not None:
            require(response and pending["request_id"] == expected_request_id,
                    "Customer displayed request changed before response input")
        profile, prepared = self._prepare_customer_history(actor_id, state)
        searches = state["customers"][str(actor_id)]["simulation_history"]
        result = customer_context(
            profile, (), prepared_history=prepared,
            limit=state["customer_history_limit"],
            cutoff=self.runtime.data.cutoff,
            history_query=pending["query"] if response else "",
            history_embedder=self._history_encoder(),
            simulation_before=state["simulation_step"] + 1,
        )
        if state["simulation_step"]:
            result["simulation_step"] = state["simulation_step"]
        if "clock" in state:
            result["decision_time"] = self.decision_time(state, "customer", actor_id)
            clock = state["clock"]
            last_settled_search = searches[-1]["search_time"] if searches else None
            result["last_search_time"] = pending["logical_time"] if pending is not None else last_settled_search
            start = (clock["start"] if last_settled_search is None
                     else last_settled_search + clock["tick_size"])
            end = pending["logical_time"] if pending is not None else clock["time"]
            if start < end:
                result["no_new_search_interval"] = {"start_inclusive": start, "end_exclusive": end}
        if response:
            from agent_simulation.agents.context.shopping import model_product

            result.update(
                query=pending["query"],
                displayed_items=[model_product(p) for p in pending["displayed_items"]],
            )
        result["long_term_memory"] = self.runtime.long_term_memory.context(
            "customer", actor_id, result, state=state, tokenizer=memory_tokenizer
        )
        if observe:
            self.runtime.long_term_memory.observe_context("customer", actor_id, result, state)
        return result

    @staticmethod
    def decision_time(state, role, actor):
        clock = state["clock"]
        return {
            "tick": clock["tick"],
            "logical_time": clock["time"],
            "tick_size": clock["tick_size"],
            "last_decision_time": clock["last_decisions"].get(f"{role}:{actor}", {}).get("time"),
            "unit": "Simulation-relative index units, not elapsed duration; the observed-history cutoff remains frozen.",
        }

