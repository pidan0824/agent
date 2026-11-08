"""Coordinate shop pricing decisions and staged product batches."""

import asyncio
from copy import deepcopy
from itertools import chain
from agent_simulation.data.common import require
from agent_simulation.agents.prompts.seller import seller_can_wait, seller_decision_messages
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

    def preview_requests(self, actor):
        """Prepare the same initial target requests as execution, without model calls or writes."""
        baseline = self.runtime.snapshot()
        for context in self.runtime.inputs.seller_targets(
            actor, state=baseline, batch_size=self.price_batch_size, memory_tokenizer=self.tokenizer
        ):
            messages = seller_decision_messages(context)
            request, _ = self.prepare_request(messages, [], 'seller', role='seller',
                allow_wait=seller_can_wait(context), price_ids=context["requested_price_item_ids"])
            yield request

    async def _quote(self, run, batch, review, baseline, request_slots):
        """One isolated target workflow; only its validated draft enters the review."""
        can_wait = seller_can_wait(batch)
        ids = batch["requested_price_item_ids"]
        prompt = seller_decision_messages(batch)
        self.runtime.long_term_memory.observe_context("seller", run["actor"], batch, baseline)

        def stage(arguments):
            if arguments == {"action": "no_action"}:
                require(can_wait, "Only the first Seller batch can wait")
                return arguments
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
                    run, messages, schemas, "seller", allow_wait=can_wait, price_ids=ids),
                validate_final=stage,
            )
        return {"item_ids": ids, "result": result}

    async def review_prices(self, run, *, baseline=None, commit=False,
                            request_slots=None):
        """Share one versioned snapshot, keep quote workers busy, then publish atomically."""
        try:
            baseline = self.runtime.freeze() if baseline is None else baseline
            actor = run["actor"]
            contexts = self.runtime.inputs.seller_targets(
                actor, state=baseline.state, batch_size=self.price_batch_size,
                memory_tokenizer=self.tokenizer)
            first = next(contexts)
            review = SellerReview(self.runtime, actor, baseline)
            concurrency = self.price_concurrency
            request_slots = request_slots or asyncio.Semaphore(concurrency)
            run.update(price_batches=[], pricing_mode="target", price_concurrency=concurrency)

            # A whole-shop wait must be decided before dispatching the rest.
            if seller_can_wait(first):
                result = await self._quote(run, first, review, baseline.state, request_slots)
                run["price_batches"].append(result)
                if result["result"] == {"action": "no_action"}:
                    run["result"] = result["result"]
                    self.logs.save(run, "complete" if commit else "staged")
                    return run
            else:
                contexts = chain([first], contexts)
            results = await map_decisions(
                contexts, lambda ctx: self._quote(run, ctx, review, baseline.state, request_slots),
                concurrency=concurrency)
            run["price_batches"].extend(results)

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
