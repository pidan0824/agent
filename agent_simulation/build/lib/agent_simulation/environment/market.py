"""Maintain market state, authorize operations and commit settlements."""

from collections import defaultdict
from contextlib import ExitStack, closing, contextmanager
from copy import deepcopy
from itertools import chain
import json
import argparse
from pathlib import Path
import sqlite3
import time

from agent_simulation.data.common import Config, connect, require, sha256
from agent_simulation.data.catalog import DataRepository
from agent_simulation.data.history_records import last_valid_reference, select_business_history, seller_history_policy
from agent_simulation.data.history_records import customer_history_limit
from agent_simulation.platform.search import index_titles
from agent_simulation.environment.memory import LongTermMemoryService
from agent_simulation.environment.context import AgentContextBuilder
from agent_simulation.logs import AgentRunLog
from agent_simulation.environment.operations.api import dispatch, validate_bound_tool
from agent_simulation.environment.store import MarketStore


class MarketEnvironment:
    @classmethod
    def create(
        cls,
        path,
        data,
        item_ids,
        customer_histories,
        top_k=10,
        history_per_item=None,
        catalog=None,
        search_config=None,
        search_pipeline=None,
    ):
        require(not Path(path).exists(), "Runtime store already exists")
        require(type(top_k) is int and top_k > 0, "Invalid list size")
        search_config = deepcopy({"backend": "bm25"} if search_config is None else search_config)
        require(search_config.get("backend") in ("bm25", "random", "kuaisearch"), "Unknown search backend")
        cls.validate_pipeline(search_config, search_pipeline)
        if search_config["backend"] in ("random", "kuaisearch"):
            if search_config["backend"] == "random":
                require(type(search_config.get("seed")) is int, "Random search needs an integer seed")
            require(
                type(search_config.get("candidate_size")) is int
                and search_config["candidate_size"] >= top_k,
                "Candidate pool is smaller than Top-K",
            )
        require(
            item_ids and len(item_ids) == len(set(item_ids)), "Empty or duplicate market catalog"
        )
        history_per_item = (
            seller_history_policy()["observations_per_item"]
            if history_per_item is None
            else history_per_item
        )
        require(
            type(history_per_item) is int and history_per_item > 0, "Invalid per-item memory limit"
        )
        products, active_prices, excluded = {}, {}, []
        history_by_seller = defaultdict(list)
        for item in item_ids:
            # Keep the latest product facts distinct from the last usable quotation.
            records = data.records(item)
            price = last_valid_reference(records, data.cutoff)
            if price is None:
                excluded.append(item)
                continue
            product = data.product(item)
            products[str(item)] = product
            history_by_seller[product["seller_id"]].append(records)
            active_prices[str(item)] = {
                "price_raw": price["price_raw"],
                "source": "observed_reference",
                "time_index": price["time_index"],
            }
        require(products, "No market product has a valid observed price before cutoff")
        if catalog is not None:
            require(
                catalog.config.dataset == data.config.dataset
                and catalog.config.output.resolve() == data.config.output.resolve(),
                "Catalog release or business source mismatch",
            )
        customers = {}
        for user, events in customer_histories.items():
            require(type(user) is int and user > 0, "Invalid customer identity")
            if catalog is not None:
                canonical = catalog.customer_history(user, data.cutoff, customer_history_limit(), "test")
                canonical = canonical["short_term_memory"]["search_history"]
                require(
                    not events or events == canonical,
                    "Customer seed differs from catalog initialization history",
                )
                events = canonical
            require(all(e.get("user_id", user) == user for e in events), "Foreign customer history")
            require(all(e["time_index"] < data.cutoff for e in events), "Future event in history store")
            require(
                len({e["session_id"] for e in events}) == len(events), "Duplicate customer history"
            )
            customers[str(user)] = {
                "observed_history": sorted(
                    deepcopy(events), key=lambda e: (e["time_index"], e["session_id"])
                ),
                "simulation_history": [],
            }
        seeds = {}
        for seller, histories in sorted(history_by_seller.items()):
            seeds[str(seller)] = select_business_history(
                chain.from_iterable(histories),
                owner_id=seller,
                cutoff=data.cutoff,
                observations_per_item=history_per_item,
            )[0]
        state = {
            "schema": "kuaisearch-runtime-v5",
            "dataset": data.config.dataset,
            "historical_cutoff_exclusive": data.cutoff,
            "simulation_step": 0,
            "history_per_item": history_per_item,
            "catalog": str(catalog.directory) if catalog is not None else None,
            "catalog_manifest_sha256": (
                sha256(catalog.directory / "manifest.json") if catalog is not None else None
            ),
            "customer_history_limit": customer_history_limit(),
            "long_term": {},
            "memory_processed": {},
            "initial_seller_memory": seeds,
            "top_k": top_k,
            "search_config": search_config,
            "products": products,
            "customers": customers,
            "active_prices": active_prices,
            "excluded_items_without_initial_price": sorted(excluded),
            "pending": {},
            "feedback": [],
            "price_actions": [],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute(
                "CREATE TABLE state (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL)"
            )
            db.execute("INSERT INTO state VALUES (1,?)", (json.dumps(state, ensure_ascii=False),))
            db.execute(
                "CREATE TABLE calls (id INTEGER PRIMARY KEY, role TEXT, actor INTEGER, name TEXT, "
                "arguments TEXT, result TEXT, error TEXT, seconds REAL)"
            )
        return cls(path, data, search_pipeline=search_pipeline)

    @staticmethod
    def validate_pipeline(config, pipeline):
        if config.get("backend") == "kuaisearch":
            require(pipeline is not None, "KuaiSearch requires a bound, trained platform pipeline")
            require(config.get("pipeline_id") == pipeline.identity
                    and config.get("reranker_id") == pipeline.reranker_id,
                    "Runtime platform pipeline identity mismatch")
        else:
            require(pipeline is None, "A platform pipeline requires the kuaisearch backend")

    def __init__(self, path, data, *, search_pipeline=None):
        require(Path(path).is_file(), "Missing runtime store")
        with ExitStack() as resources:
            self.db = resources.enter_context(closing(sqlite3.connect(path)))
            self.store = MarketStore(self.db)
            self.data = data
            self.catalog = None
            state = self.snapshot()
            self.validate_pipeline(state["search_config"], search_pipeline)
            self.search_pipeline = search_pipeline
            require(
                state.get("schema") == "kuaisearch-runtime-v5",
                "Unsupported runtime schema; create a new runtime store",
            )
            if state.get("catalog") is not None:
                from agent_simulation.data.catalog import Catalog

                require(
                    sha256(Path(state["catalog"]) / "manifest.json")
                    == state["catalog_manifest_sha256"],
                    "Runtime catalog changed",
                )
                self.catalog = resources.enter_context(closing(Catalog(state["catalog"])))
                require(
                    self.catalog.config.dataset == data.config.dataset
                    and self.catalog.config.output.resolve() == data.config.output.resolve(),
                    "Catalog release or business source mismatch",
                )
            require(
                type(state.get("history_per_item")) is int and state["history_per_item"] > 0,
                "Runtime store lacks the per-item memory policy; create a new store",
            )
            require(
                state["dataset"] == data.config.dataset
                and state["historical_cutoff_exclusive"] == data.cutoff,
                "Runtime dataset/cutoff mismatch",
            )
            self.search = None
            if state["search_config"]["backend"] == "bm25":
                self.search = resources.enter_context(closing(sqlite3.connect(":memory:")))
                index_titles(self.search, state["products"])
            self.long_term_memory = LongTermMemoryService(self)
            self.inputs = AgentContextBuilder(self)
            self.logs = AgentRunLog(self.db)
            self._resources = resources.pop_all()

    def close(self):
        self._resources.close()

    def snapshot(self):
        return self.store.snapshot()

    def freeze(self):
        return self.store.freeze()

    @property
    def revision(self):
        return self.store.revision

    def start_episode(self, clock):
        with self.store.transaction():
            state = self.snapshot()
            require('clock' not in state and state['simulation_step'] == 0 and not state['pending'],
                    'Start an Episode from a fresh initialized market')
            state['clock'] = clock
            self.db.execute('CREATE TABLE IF NOT EXISTS ticks (tick INTEGER PRIMARY KEY, trace TEXT)')
            self.store.write(state)

    def claim_tick(self):
        with self.store.transaction():
            state = self.snapshot()
            clock = state['clock']
            require(clock['status'] == 'ready', 'Episode is not ready for another Tick')
            require(not state['pending'], 'A Tick cannot start with unsettled searches')
            trace = dict(tick=clock['tick'], logical_time=clock['time'], status='running',
                         sellers=[], customers=[])
            clock['status'] = 'running'
            self.store.write(state)
            self.store.write_tick_trace(trace)
        return state, trace

    @contextmanager
    def record_tick(self, trace):
        """Edit the current Tick state and commit its trace in the same transaction."""
        with self.store.transaction():
            state = self.snapshot()
            clock = state["clock"]
            require(clock["status"] == "running" and clock["tick"] == trace["tick"]
                    and clock["time"] == trace["logical_time"], "Tick is no longer running")
            yield state
            self.store.write(state)
            self.store.write_tick_trace(trace)

    def commit_seller_review(self, actor, prices, revision, metadata):
        return self._apply_staged('seller', [(actor, 'submit_prices', {'prices': prices})],
                                  expected_revision=revision, metadata={actor: metadata})[0]

    def apply_seller_quotes(self, quotes, *, expected_revision=None):
        """Publish every staged shop decision in one SQLite transaction."""
        actions = [(actor, "submit_prices", args) for actor, args in quotes]
        return self._apply_staged("seller", actions, expected_revision=expected_revision)

    def settle_customers(self, actions, *, request_ids=None):
        """Publish the Tick's exits and responses together.

        Each response gets its own settled_step and one feedback row per displayed item.
        """
        actions = list(actions)
        require(all(name in ("exit_customer", "submit_response") for _, name, _ in actions),
                "Only Customer outcomes may be settled")
        if request_ids is not None:
            require(set(request_ids) == {actor for actor, name, _ in actions if name == "submit_response"}
                    and all(type(rid) is int for rid in request_ids.values()),
                    "Expected one displayed request ID per Customer response")
        return self._apply_staged("customer", actions, request_ids=request_ids)

    def _apply_staged(self, role, actions, *, request_ids=None, expected_revision=None, metadata=None):
        if not actions:
            return []
        require(len({actor for actor, _, _ in actions}) == len(actions),
                f"Duplicate {role.title()} in staged batch")
        with self.store.transaction():
            if expected_revision is not None:
                require(self.revision == expected_revision, "Market changed during Seller review; discard staged quotes")
            state = self.snapshot()
            results = [
                self._execute_bound(role, actor, name, arguments,
                                    json.dumps(arguments, allow_nan=False), time.monotonic(),
                                    state=state, expected_request_id=request_ids.get(actor) if request_ids else None,
                                    metadata=metadata.get(actor) if metadata else None)
                for actor, name, arguments in actions
            ]
            self.store.write(state)
            return results

    def _execute_bound(self, role, actor_id, name, arguments, encoded, started, *, state=None, expected_request_id=None, metadata=None):
        require(type(actor_id) is int, "Invalid actor binding")
        write_state = state is None
        if write_state:
            state = self.snapshot()
        validate_bound_tool(role, "clock" in state, name, arguments)
        owned = []
        if role == "seller":
            owned = [int(i) for i, p in state["products"].items() if p["seller_id"] == actor_id]
            registered = bool(owned)
        else:
            registered = str(actor_id) in state["customers"]
        require(registered, "Actor is not registered in this market")
        if role == "customer" and name in (
            "search_products",
            "submit_response",
            "exit_customer",
        ):
            require(
                not state["customers"][str(actor_id)].get("exit"), "Customer already exited"
            )
        if expected_request_id is not None:
            pending = state["pending"].get(str(actor_id))
            require(role == "customer" and name == "submit_response" and pending is not None
                    and pending["request_id"] == expected_request_id, "Customer displayed request changed")
        result = dispatch(self, state, role, actor_id, owned, name, arguments)
        if metadata is not None:
            result.update(metadata)
        self.long_term_memory.observe_tool(role, actor_id, name, arguments, result, state)
        if name not in ("submit_prices", "search_products", "submit_response", "exit_customer"):
            return result
        if write_state:
            self.store.write(state)
        self.db.execute(
            "INSERT INTO calls(role,actor,name,arguments,result,seconds) VALUES(?,?,?,?,?,?)",
            (
                role,
                actor_id,
                name,
                encoded,
                json.dumps(result, ensure_ascii=False, allow_nan=False),
                time.monotonic() - started,
            ),
        )
        return result

    def execute(self, role, actor_id, name, arguments, *, expected_request_id=None):
        """Authorize the operation; record actions and failures, without duplicating query results."""
        started = time.monotonic()
        encoded = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        try:
            with self.store.transaction():
                return self._execute_bound(role, actor_id, name, arguments, encoded, started,
                                           expected_request_id=expected_request_id)
        except Exception as error:
            with self.db:
                self.db.execute(
                    "INSERT INTO calls(role,actor,name,arguments,error,seconds) VALUES(?,?,?,?,?,?)",
                    (
                        role,
                        actor_id,
                        name,
                        encoded,
                        f"{type(error).__name__}: {error}",
                        time.monotonic() - started,
                    ),
                )
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Execute one bound tool against an existing isolated runtime store."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--role", choices=("customer", "seller"), required=True)
    parser.add_argument("--actor-id", type=int, required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument(
        "--arguments", required=True, help="JSON object containing tool arguments only"
    )
    args = parser.parse_args()
    require(args.store.is_file(), "Missing runtime store")
    with closing(sqlite3.connect(f"{args.store.resolve().as_uri()}?mode=ro", uri=True)) as db:
        state = json.loads(db.execute("SELECT value FROM state WHERE id=1").fetchone()[0])
    cfg = Config.load(args.config)
    c = connect(cfg)
    runtime = None
    try:
        runtime = MarketEnvironment(args.store, DataRepository(c, cfg, state["historical_cutoff_exclusive"]))
        print(
            json.dumps(
                runtime.execute(args.role, args.actor_id, args.tool, json.loads(args.arguments)),
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        if runtime is not None:
            runtime.close()
        c.close()


if __name__ == "__main__":
    main()
