"""Coordinate shop pricing decisions and staged product batches."""

import asyncio
from copy import deepcopy
import hashlib
import json
from uuid import uuid4
from agent_simulation.data.common import require
from agent_simulation.agents.context import seller_decision_messages
from agent_simulation.environment.operations.actions import SellerReview, require_open_market
from agent_simulation.agents.base import BaseAgent
from agent_simulation.agents.concurrency import map_decisions
from agent_simulation.agents.decisions import run_workflow
from agent_simulation.inference.client import ModelOutputError


class SellerAgent(BaseAgent):
    def __init__(self, *args, price_batch_size=1, price_concurrency=2,
                 agentic_model=None, agentic_options=None, **kwargs):
        super().__init__(*args, **kwargs)
        require(type(price_batch_size) is int and 1 <= price_batch_size <= 16,
                "Expected 1..16 price targets")
        require(type(price_concurrency) is int and price_concurrency > 0,
                "Expected positive price concurrency")
        self.price_batch_size = price_batch_size
        self.price_concurrency = price_concurrency
        self.agentic_model = agentic_model
        from agent_simulation.agents.agentic import options_from
        self.agentic_options = options_from(agentic_options)

    @property
    def engine(self):
        return "seller-agentic-frozen-v1" if self.agentic_model is not None else super().engine

    def run(self, actor):
        """Run one isolated decision; Episode uses propose() to defer price application."""
        require("clock" not in self.runtime.snapshot(), "Use Episode for Tick decisions")
        return asyncio.run(self.review_prices(self.prepare_review(actor), commit=True))

    def propose(self, actor):
        """Collect a Tick proposal without applying prices."""
        state = self.runtime.snapshot()
        require("clock" in state, "Tick proposals require an Episode clock")
        require_open_market(state)
        return asyncio.run(self.review_prices(self.prepare_review(actor)))

    def prepare_review(self, actor):
        """Record the review and finish pending reflections before prices are frozen."""
        run = self.logs.start("seller", actor)
        run["engine"] = self.engine
        try:
            self._update_long_term_memory(run, [("seller", actor)])
            return run
        except Exception as error:
            self.logs.save(run, "failed", str(error))
            raise

    def prepare_requests(self, actor):
        """Prepare target requests, including optional context-model calls; do not submit prices or write state."""
        baseline = self.runtime.snapshot()
        for context, messages in self.prepare_prices(actor, baseline):
            if self.agentic_model is not None:
                from agent_simulation.agents.agentic import FrozenSellerTools, initial_agentic_request
                from agent_simulation.inference.requests import prepare_request
                bound = FrozenSellerTools(self.runtime, actor, baseline, theory=self.agentic_options.theory)
                messages, tools, _ = initial_agentic_request(messages, bound.schemas, self.agentic_options)
                yield prepare_request(messages, tools, self.agentic_model.tokenizer,
                                      max_output_tokens=self.agentic_model.max_new_tokens)[0]
                continue
            request, _ = self.prepare_request(messages, [], 'seller', role='seller',
                price_ids=context["requested_price_item_ids"])
            yield request

    def prepare_prices(self, actor, baseline):
        """Finish database reads and optional context inference before dispatching quotes."""
        if self.agentic_model is not None:
            from agent_simulation.agents.context.seller_features import seller_behavior_features
            owned = sorted(int(item) for item, product in baseline["products"].items()
                           if product["seller_id"] == actor)
            require(owned, "Seller is not registered")
            batches = []
            for start in range(0, len(owned), self.price_batch_size):
                ids = owned[start:start + self.price_batch_size]
                records = [row for item in ids for row in self.runtime.data.records(item)]
                context = {"seller_id": actor, "requested_price_item_ids": ids,
                    "history_cutoff_exclusive": self.runtime.data.cutoff,
                    "simulation_before_exclusive": baseline["simulation_step"] + 1,
                    "shop_product_count": len(owned),
                    "products": [{**{key: product[key] for key in ("item_id", "item_title", "brand_name",
                        "category_level3_name") if key in product},
                        "active_price_raw": baseline["active_prices"][str(item)]["price_raw"]}
                        for item in ids for product in [baseline["products"][str(item)]]],
                    "behavior_features": seller_behavior_features(records, seller_id=actor,
                                                                  cutoff=self.runtime.data.cutoff)}
                if "clock" in baseline:
                    context["decision_time"] = self.runtime.inputs.decision_time(baseline, "seller", actor)
                batches.append((context, self._agentic_messages(context)))
            return batches
        return [(context, seller_decision_messages(context))
                for context in self.runtime.inputs.seller_targets(
                    actor, state=baseline, batch_size=self.price_batch_size,
                    memory_tokenizer=self.tokenizer)]

    async def _quote(self, run, prepared, review, baseline, request_slots):
        """One isolated target workflow; only its validated draft enters the review."""
        batch, prompt = prepared
        ids = batch["requested_price_item_ids"]
        if self.agentic_model is None:
            self.runtime.long_term_memory.observe_context("seller", run["actor"], batch, baseline)

        def stage(arguments):
            entry = {"name": "submit_prices", "arguments": deepcopy(arguments), "staged_review": True}
            run["tools"].append(entry)
            try:
                entry["result"] = review.stage(arguments, ids)
            except (ValueError, PermissionError) as error:
                entry["error"] = f"{type(error).__name__}: {error}"
                raise ModelOutputError(str(error), raw_output=arguments) from error
            return entry["result"]

        # _complete checks the actual prompt once; no hypothetical target reply
        # or extra tokenization pass is needed before dispatch.
        async with request_slots:
            if self.agentic_model is not None:
                action = await self._agentic_quote(run, batch, prompt, baseline)
                return {"item_ids": ids, "result": stage(action)}
            result = await run_workflow(
                messages=prompt,
                complete=lambda messages, schemas: self._complete(
                    run, messages, schemas, "seller", price_ids=ids),
                validate_final=stage,
            )
        return {"item_ids": ids, "result": result}

    def _agentic_messages(self, context):
        from agent_simulation.agents.agentic import teacher_output
        from agent_simulation.agents.actions import price_schema
        from agent_simulation.agents.prompts.seller_agentic import agentic_seller_instructions
        return [{"role": "system", "content": agentic_seller_instructions(theory=self.agentic_options.theory)},
                {"role": "user", "content": json.dumps({"stage": "seller", "context": context,
                    "final_output": teacher_output([price_schema(context["requested_price_item_ids"],
                                                               allow_unknown=True)])},
                    ensure_ascii=False, allow_nan=False)}]

    async def _agentic_quote(self, run, context, messages, state):
        from agent_simulation.agents.agentic import FrozenSellerTools, run_agentic_workflow, validate_teacher_final
        from agent_simulation.agents.actions import price_schema
        from agent_simulation.inference.client import complete_once
        from agent_simulation.inference.requests import prepare_request
        frozen = self.runtime.freeze()
        require(state == frozen.state, "Market changed before Seller agentic decision")
        actor = run["actor"]
        bound = FrozenSellerTools(self.runtime, actor, frozen.state, theory=self.agentic_options.theory)
        fingerprint = hashlib.sha256(json.dumps(frozen.state, ensure_ascii=False, sort_keys=True,
                                               allow_nan=False).encode()).hexdigest()
        trace = {"role": "seller", "actor_id": actor, "actor": actor, "stage": "seller",
                 "trajectory_id": uuid4().hex, "initial_state_id": fingerprint,
                 "snapshot_sha256": fingerprint, "snapshot_revision": frozen.revision,
                 "model": self.agentic_model.generator_id,
                 "budget_scope": "one_target_batch",
                 "requested_price_item_ids": context["requested_price_item_ids"]}
        run.setdefault("agentic_traces", []).append(trace)

        def guard():
            require(self.runtime.revision == frozen.revision, "Market changed during Seller agentic decision")

        async def complete(prompt, tools):
            request, count = prepare_request(prompt, tools, self.agentic_model.tokenizer,
                                             max_output_tokens=self.agentic_model.max_new_tokens)
            request.update(agent_role="seller", decision_stage="seller")
            call_id = f'{run["id"]}:agentic:{len(run["model_calls"]) + 1}'
            call = {"call_id": call_id, "stage": "seller", "input_tokens": count,
                    "request": deepcopy(request), "model": self.agentic_model.generator_id}
            run["model_calls"].append(call)
            try:
                reply = await complete_once(self.agentic_model, request, call_id=call_id)
                call["response"] = deepcopy(reply)
                return reply
            except BaseException as error:
                call["error"] = f"{type(error).__name__}: {error}"
                raise

        schema = [price_schema(context["requested_price_item_ids"], allow_unknown=True)]

        def validate_action(action):
            for quote in action["prices"]:
                require(quote["price_raw"] is not None or
                        frozen.state["active_prices"][str(quote["item_id"])]["price_raw"] is None,
                        "A null price can only keep an unknown active price")
            return action

        try:
            return await run_agentic_workflow(messages=messages, tools=bound.schemas, complete=complete,
                execute_tool=bound.execute,
                validate_final=lambda value: validate_teacher_final(value, schema, trace=trace,
                                                                    validate_action=validate_action),
                options=self.agentic_options, trace=trace, guard=guard)
        finally:
            run["tools"].extend({**deepcopy(entry), "stage": "seller", "read_only": True}
                                for entry in trace.get("tools", []))

    async def review_prices(self, run, *, baseline=None, commit=False,
                            request_slots=None, prepared_batches=None):
        """Prepare one frozen input set, run quotes concurrently, then publish atomically."""
        try:
            baseline = self.runtime.freeze() if baseline is None else baseline
            actor = run["actor"]
            batches = (self.prepare_prices(actor, baseline.state)
                       if prepared_batches is None else prepared_batches)
            review = SellerReview(self.runtime, actor, baseline)
            concurrency = self.price_concurrency
            request_slots = request_slots or asyncio.Semaphore(concurrency)
            run.update(price_batches=[], pricing_mode="target", price_concurrency=concurrency)

            run["price_batches"] = await map_decisions(
                batches, lambda ctx: self._quote(run, ctx, review, baseline.state, request_slots),
                concurrency=concurrency)

            if commit:
                run["result"] = review.commit(run["id"])
            else:
                require(self.runtime.revision == baseline.revision, "Market changed during Seller proposal")
                run["result"] = {"status": "staged", "prices": review.proposed_prices()}
            self.logs.save(run, "complete" if commit else "staged")
            return run
        except Exception as error:
            self.logs.save(run, "failed", f"{type(error).__name__}: {error}")
            raise
