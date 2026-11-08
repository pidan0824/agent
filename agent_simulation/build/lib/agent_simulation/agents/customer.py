"""Customer Query and Response decisions with automatically prepared history."""

import asyncio
from contextlib import nullcontext
from copy import deepcopy
from agent_simulation.data.common import require
from agent_simulation.agents.base import BaseAgent
from agent_simulation.agents.concurrency import map_decisions
from agent_simulation.agents.decisions import action_output, run_workflow
from agent_simulation.environment.operations.actions import require_open_market
from agent_simulation.agents.schemas import validate_arguments
from agent_simulation.agents.actions import validate_response
import agent_simulation.environment.batching as batching
from agent_simulation.agents.prompts.customer import prepare_history
from agent_simulation.agents.context.shopping import project_customer
from agent_simulation.agents.prompts.customer import customer_decision_messages


class CustomerAgent(BaseAgent):
    def __init__(self, *args, response_concurrency=2, response_batch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        require(type(response_concurrency) is int and response_concurrency > 0,
                "Expected positive response concurrency")
        self.response_concurrency = response_concurrency
        self.response_batch_size = (batching.CUSTOMER_RESPONSE_BATCH_SIZE
                                    if response_batch_size is None else response_batch_size)
        require(type(self.response_batch_size) is int and self.response_batch_size > 0,
                "Customer response batch size must be positive")

    def _response_inputs(self, actor, *, state=None, expected_request_id=None, observe=True):
        batches = self.runtime.inputs.customer_responses(
            actor, state=state, batch_size=self.response_batch_size,
            expected_request_id=expected_request_id, observe=observe, memory_tokenizer=self.tokenizer,
        )
        history = prepare_history(project_customer(batches[0])) if len(batches) > 1 else None
        return batches, history

    def preview_requests(self, actor, *, response=False):
        """Render requests from the current state without running memory reflection."""
        stage = "response" if response else "query"
        if response:
            batches, history = self._response_inputs(actor, observe=False)
        else:
            context = self.runtime.inputs.customer(
                actor, observe=False, memory_tokenizer=self.tokenizer)
            batches, history = [context], None
        for state in batches:
            request, _ = self.prepare_request(
                customer_decision_messages(state, response=stage == "response", prepared_history=history), [], stage, role="customer",
                allow_wait=stage == "query" and "decision_time" in state,
                displayed_ids=[p["item_id"] for p in state["displayed_items"]] if response else None,
            )
            yield request

    def prepare_decision(self, actor, *, state=None):
        run = self.logs.start("customer", actor)
        run["engine"] = self.engine
        try:
            initial = self.runtime.snapshot() if state is None else state
            customer = initial["customers"].get(str(actor))
            require(customer is not None, "Customer is not registered in this market")
            require(not customer.get("exit"), "Customer already exited")
            require_open_market(initial)
            self._update_long_term_memory(run, [("customer", actor)])
            return run
        except Exception as error:
            self.logs.save(run, "failed", str(error))
            raise

    async def choose_query(self, run, *, state=None):
        actor = run["actor"]
        context = self.runtime.inputs.customer(actor, state=state, memory_tokenizer=self.tokenizer)
        choice = await self._stage(run, "query", state=context)
        if "action" in choice:
            run["result"] = choice
        else:
            run["search_query"] = choice

    def open_display(self, run):
        if "search_query" in run:
            run["displayed_list"] = self._execute(run, "search_products", run["search_query"])
            del run["search_query"]

    async def choose_response(self, run, *, request_slots=None, state=None):
        actor = run["actor"]
        pending = run["displayed_list"]
        batches, prepared_history = self._response_inputs(
            actor, state=state, expected_request_id=pending["request_id"])
        selected = {"clicked_item_ids": set(), "purchased_item_ids": set()}
        run["response_batches"] = []

        async def respond(batch):
            # Episode supplies a shared limit; map_decisions already bounds this Customer.
            async with (request_slots if request_slots is not None else nullcontext()):
                return await self._stage(run, "response", state=batch, prepared_history=prepared_history)

        responses = await map_decisions(batches, respond, concurrency=min(self.response_concurrency, len(batches)))
        for batch, args in zip(batches, responses, strict=True):
            run["response_batches"].append({
                "item_ids": [p["item_id"] for p in batch["displayed_items"]],
                "response": deepcopy(args),
            })
            for key in selected:
                selected[key].update(args[key])
        # Keep the original request order; intermediate choices never enter history.
        args = {
            key: [p["item_id"] for p in pending["displayed_items"] if p["item_id"] in ids]
            for key, ids in selected.items()
        }
        run["result"] = {"status": "staged", "response": args}

    def run(self, actor, *, defer_settlement=False):
        """Run one Customer, resuming its existing pending search when present."""
        run = self.prepare_decision(actor)
        try:
            pending = self.runtime.snapshot()["pending"].get(str(actor))
            if pending is None:
                asyncio.run(self.choose_query(run))
                self.open_display(run)
            else:
                run["displayed_list"] = pending
            if "displayed_list" in run:
                asyncio.run(self.choose_response(run))
            if defer_settlement:
                self.logs.save(run, "staged")
                return run
            if "displayed_list" in run:
                run["result"] = self._execute(
                    run, "submit_response", run["result"]["response"],
                    expected_request_id=run["displayed_list"]["request_id"],
                )
                run["settled_result"] = run["result"]
                self.logs.save(run, "settled")
            else:
                if run["result"] == {"action": "exit"}:
                    run["applied_result"] = self._execute(run, "exit_customer", {})
                self.logs.save(run, "complete")
                return run
        except Exception as error:
            self.logs.save(run, "failed", f"{type(error).__name__}: {error}")
            raise

        self.finish_response(run)
        return run

    def finish_response(self, run):
        require("settled_result" in run, "Customer response has not been settled")
        try:
            self.reflect_on_feedback(run)
        except Exception as error:
            self.logs.save(run, "failed", f"Post-settlement reflection: {type(error).__name__}: {error}")
            raise
        self.logs.save(run, "complete")

    def settle_staged(self, runs):
        """Commit all outcomes, record receipts, then reflect on the settled feedback."""
        runs = list(runs)
        staged = []
        for run in runs:
            require("settled_result" not in run and "applied_result" not in run, "Customer decision already applied")
            decision = run["result"]
            if decision == {"action": "exit"}:
                staged.append((run, "exit_customer", {}))
            elif decision != {"action": "no_action"}:
                staged.append((run, "submit_response", decision["response"]))
        actions = [(run["actor"], name, args) for run, name, args in staged]
        request_ids = {run["actor"]: run["displayed_list"]["request_id"]
                       for run, name, _ in staged if name == "submit_response"}
        try:
            results = self.runtime.settle_customers(actions, request_ids=request_ids)
            require(len(results) == len(staged), "Customer settlement result count mismatch")
            require(
                all(result.get("request_id") == request_ids[run["actor"]]
                    for (run, name, _), result in zip(staged, results, strict=True)
                    if name == "submit_response"),
                "Customer settlement request ID mismatch",
            )
        except Exception as error:
            for run in runs:
                self.logs.save(run, "failed", f"Customer settlement failed: {type(error).__name__}: {error}")
            raise
        for (run, name, args), result in zip(staged, results, strict=True):
            run["tools"].append({"name": name, "arguments": args, "result": result})
            run["applied_result" if name == "exit_customer" else "settled_result"] = result
        for run in runs:
            self.logs.save(run, "settled" if "settled_result" in run else "complete")
        for run, name, _ in staged:
            if name == "submit_response":
                self.finish_response(run)

    def reflect_on_feedback(self, run):
        sellers = [("seller", item["seller_id"]) for item in run["displayed_list"]["displayed_items"]]
        self._update_long_term_memory(run, [("customer", run["actor"]), *sellers])

    async def _stage(self, run, stage, *, state, prepared_history=None):
        displayed_ids = [product["item_id"] for product in state["displayed_items"]] if stage == "response" else None
        allow_wait = stage == "query" and "decision_time" in state

        def validate(args):
            if allow_wait and args in ({"action": "no_action"}, {"action": "exit"}):
                return args
            if stage == "response":
                return validate_response(args, displayed_ids)
            validate_arguments(args, action_output(stage)["final_schemas"][0])
            return args

        prompt = customer_decision_messages(state, response=stage == "response",
                                            prepared_history=prepared_history)
        recent = state.get("short_term_memory", {}).get("search_history", [])
        history = state.get("retrieved_context", {}).get("history", {})
        provenance = {
            "stage": stage,
            "candidate_batch": state.get("candidate_batch"),
            "recent_session_ids": [event.get("session_id") for event in recent],
            "retrieved_session_ids": [
                [event.get("session_id") for event in segment["records"]]
                for segment in history.get("segments", [])
            ],
        }
        if state.get("long_term_memory"):
            provenance["memories"] = [
                {key: deepcopy(memory[key]) for key in ("memory_id", "source_ids", "evidence_times")
                 if key in memory}
                for memory in state["long_term_memory"]
            ]
        run.setdefault("context_provenance", []).append(provenance)
        return await run_workflow(
            messages=prompt,
            complete=lambda prompt, tools: self._complete(run, prompt, tools, stage,
                allow_wait=allow_wait, displayed_ids=displayed_ids),
            validate_final=validate,
        )
