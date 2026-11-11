"""Coordinate shop pricing decisions and staged product batches."""

import asyncio
from copy import deepcopy
from agent_simulation.data.common import require
from agent_simulation.agents.context import seller_decision_messages
from agent_simulation.environment.operations.actions import SellerReview, require_open_market
from agent_simulation.agents.base import BaseAgent
from agent_simulation.agents.concurrency import map_decisions
from agent_simulation.agents.decisions import run_workflow
from agent_simulation.inference.client import ModelOutputError


class SellerAgent(BaseAgent):
    def __init__(self, *args, price_batch_size=1, price_concurrency=2, **kwargs):
        super().__init__(*args, **kwargs)
        require(type(price_batch_size) is int and 1 <= price_batch_size <= 16,
                "Expected 1..16 price targets")
        require(type(price_concurrency) is int and price_concurrency > 0,
                "Expected positive price concurrency")
        self.price_batch_size = price_batch_size
        self.price_concurrency = price_concurrency

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
            request, _ = self.prepare_request(messages, [], 'seller', role='seller',
                price_ids=context["requested_price_item_ids"])
            yield request

    def prepare_prices(self, actor, baseline):
        """Finish database reads and optional context inference before dispatching quotes."""
        return [(context, seller_decision_messages(context))
                for context in self.runtime.inputs.seller_targets(
                    actor, state=baseline, batch_size=self.price_batch_size,
                    memory_tokenizer=self.tokenizer)]

    async def _quote(self, run, prepared, review, baseline, request_slots):
        """One isolated target workflow; only its validated draft enters the review."""
        batch, prompt = prepared
        ids = batch["requested_price_item_ids"]
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
            result = await run_workflow(
                messages=prompt,
                complete=lambda messages, schemas: self._complete(
                    run, messages, schemas, "seller", price_ids=ids),
                validate_final=stage,
            )
        return {"item_ids": ids, "result": result}

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
