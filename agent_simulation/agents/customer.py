"""Customer Query and Response decisions with automatically prepared history."""

import asyncio
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from agent_simulation.data.common import require
from agent_simulation.agents.base import BaseAgent
from agent_simulation.agents.concurrency import map_decisions
from agent_simulation.agents.decisions import action_output, run_workflow
from agent_simulation.environment.operations.actions import require_open_market
from agent_simulation.agents.schemas import validate_arguments
from agent_simulation.agents.actions import validate_response
from agent_simulation.agents.context import customer_decision_messages


class CustomerAgent(BaseAgent):
    def __init__(self, *args, response_concurrency=2, agentic_model=None, agentic_options=None, **kwargs):
        super().__init__(*args, **kwargs)
        require(type(response_concurrency) is int and response_concurrency > 0,
                "Expected positive response concurrency")
        self.response_concurrency = response_concurrency
        self.agentic_model = agentic_model
        from agent_simulation.agents.agentic import options_from
        self.agentic_options = options_from(agentic_options)

    @property
    def engine(self):
        return "customer-agentic-frozen-v1" if self.agentic_model is not None else super().engine

    def _response_inputs(self, actor, *, state=None, expected_request_id=None, observe=True):
        return self.runtime.inputs.customer_responses(
            actor, self.tokenizer, state=state, max_output_tokens=getattr(self.model, "max_new_tokens", 0),
            expected_request_id=expected_request_id, observe=observe,
        )

    def prepare_requests(self, actor, *, response=False):
        """Prepare requests, including optional history summaries; do not run decisions or memory reflection."""
        stage = "response" if response else "query"
        if self.agentic_model is not None:
            from agent_simulation.agents.agentic import FrozenCustomerTools, initial_agentic_request
            from agent_simulation.inference.requests import prepare_request
            state = self.runtime.snapshot()
            context = self._agentic_context(actor, state, stage)
            bound = FrozenCustomerTools(self.runtime, actor, state, stage, theory=self.agentic_options.theory)
            messages, schemas, _ = initial_agentic_request(self._agentic_messages(context, stage),
                                                         bound.schemas, self.agentic_options)
            request, _ = prepare_request(messages, schemas,
                self.agentic_model.tokenizer, max_output_tokens=self.agentic_model.max_new_tokens)
            yield request
            return
        if response:
            batches = self._response_inputs(actor, observe=False)
        else:
            context = self.runtime.inputs.customer(
                actor, observe=False, memory_tokenizer=self.tokenizer)
            batches = [(context, customer_decision_messages(context))]
        for state, prompt in batches:
            request, _ = self.prepare_request(
                prompt, [], stage, role="customer",
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
        if self.agentic_model is not None:
            choice = await self._agentic_stage(run, "query", state=state)
        else:
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
        if self.agentic_model is not None:
            async with (request_slots if request_slots is not None else nullcontext()):
                args = await self._agentic_stage(run, "response", state=state,
                                                expected_request_id=pending["request_id"])
            run["response_batches"] = [{"item_ids": [p["item_id"] for p in pending["displayed_items"]],
                                        "response": deepcopy(args)}]
            run["result"] = {"status": "staged", "response": args}
            return
        batches = self._response_inputs(
            actor, state=state, expected_request_id=pending["request_id"])
        selected = {"clicked_item_ids": set(), "purchased_item_ids": set()}
        run["response_batches"] = []
        rankers = getattr(self.model, "customer_rankers", {})

        async def respond(batch):
            state, prompt = batch
            # Episode supplies a shared limit; map_decisions already bounds this Customer.
            async with (request_slots if request_slots is not None else nullcontext()):
                if "click" in rankers and "purchase" in rankers:
                    args = {}
                else:
                    args = await self._stage(run, "response", state=state, prompt=prompt)
                scores = {}
                if rankers:
                    ids = [p["item_id"] for p in state["displayed_items"]]
                    revision = self.runtime.revision
                    suffix = ids[0] if ids else "empty"
                    for action in rankers:
                        values = await self.model.score_customer_items(
                            prompt, ids, action, call_id=f'{run["id"]}:{action}:{suffix}')
                        require(self.runtime.revision == revision, "State changed during Customer scoring")
                        require(values.keys() == set(ids), "Customer scores do not match the displayed batch")
                        scores[action] = values
                return args, scores

        responses = await map_decisions(batches, respond, concurrency=min(self.response_concurrency, len(batches)))
        for action, (served, threshold) in rankers.items():
            scores = {item: value for _, batch_scores in responses for item, value in batch_scores[action].items()}
            ranked = sorted(scores, key=lambda item: (-scores[item], item))
            chosen = {item for item in ranked[:3] if scores[item] >= threshold}
            run[f"{action}_ranking"] = {"model": served, "threshold": threshold, "scores": scores}
            field = "clicked_item_ids" if action == "click" else "purchased_item_ids"
            for args, batch_scores in responses:
                args[field] = [item for item in batch_scores[action] if item in chosen]
        for (batch, _), (args, _) in zip(batches, responses, strict=True):
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

    def settle_staged(self, runs, *, on_settled=None):
        """Commit all outcomes, notify the caller, then record receipts and reflect."""
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
            if on_settled is not None:
                on_settled()
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

    def _agentic_context(self, actor, state, stage):
        """Expose descriptive features; the model chooses which underlying facts to read."""
        from agent_simulation.agents.context.customer_features import customer_behavior_features
        require(self.runtime.catalog is not None, "Customer agentic input requires a verified catalog")
        seed = state["customers"][str(actor)]["observed_history"]
        profile, history, summary = self.runtime.inputs._customer_observed_history(actor, seed)
        result = {
            "profile": {key: deepcopy(value) for key, value in profile.items() if key != "user_id"},
            "history_cutoff_exclusive": self.runtime.data.cutoff,
            "simulation_before_exclusive": state["simulation_step"] + 1,
            "behavior_features": customer_behavior_features(
                history, cutoff=self.runtime.data.cutoff,
                simulated=state["customers"][str(actor)]["simulation_history"],
                simulation_before=state["simulation_step"] + 1,
                behavioral_summary=summary, user_id=actor),
            "evidence_access": "Read prior facts and private memories with the bound tools. Unknown is not zero.",
        }
        if "clock" in state:
            result["decision_time"] = self.runtime.inputs.decision_time(state, "customer", actor)
        if stage == "response":
            pending = state["pending"][str(actor)]
            result.update(query=pending["query"], displayed_items=[
                {key: product[key] for key in ("item_id", "item_title", "price_raw", "price_status")
                 if key in product}
                for product in pending["displayed_items"]])
        return result

    def _agentic_messages(self, context, stage, previous=()):
        from agent_simulation.agents.prompts.customer_agentic import agentic_customer_instructions
        from agent_simulation.agents.agentic import teacher_output
        displayed = [p["item_id"] for p in context["displayed_items"]] if stage == "response" else None
        output = action_output(stage, allow_control_actions=stage == "query" and "decision_time" in context,
                               displayed_ids=displayed)
        messages = deepcopy(list(previous)) if previous else [{"role": "system", "content": (
            agentic_customer_instructions(theory=self.agentic_options.theory)
            + "\nUse only the tools currently offered. Queries are optional. Return the newest stage's final JSON "
              "when enough evidence is available. Tools cannot search, settle, edit memory or change prices. "
              "The caller executes one final Query search before the Response stage. "
              "Tool and model-call budgets restart for each stage; no budget exhaustion is an implicit action. "
              "A brief behavior hypothesis may be recorded with record_behavior_hypothesis; it is unverified."
        )}]
        messages.append({"role": "user", "content": json.dumps(
            {"stage": stage, "final_output": teacher_output(output["final_schemas"], customer_ekb=self.agentic_options.theory), "context": context}, ensure_ascii=False, allow_nan=False)})
        return messages

    async def _agentic_stage(self, run, stage, *, state=None, expected_request_id=None):
        from agent_simulation.agents.agentic import FrozenCustomerTools, run_agentic_workflow, validate_teacher_final, teacher_output
        from uuid import uuid4
        from agent_simulation.inference.client import complete_once
        from agent_simulation.inference.requests import prepare_request
        # Pair the decision facts and revision in one database read. A supplied
        # phase snapshot must still describe this revision before dispatch.
        frozen = self.runtime.freeze()
        require(state is None or state == frozen.state,
                "Market changed before Customer agentic decision")
        state, revision = frozen
        actor = run["actor"]
        if expected_request_id is not None:
            require(state["pending"].get(str(actor), {}).get("request_id") == expected_request_id,
                    "Customer displayed request changed before agentic decision")
        context = self._agentic_context(actor, state, stage)
        bound = FrozenCustomerTools(self.runtime, actor, state, stage, theory=self.agentic_options.theory)
        previous = ()
        if stage == "response" and run.get("agentic_traces"):
            queries = [t for t in run["agentic_traces"] if t["stage"] == "query" and t["status"] == "complete"]
            if queries:
                previous = queries[-1]["messages"]
        messages = self._agentic_messages(context, stage, previous)
        trace = {"stage": stage, "actor": actor, "snapshot_revision": revision,
                 "snapshot_sha256": hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False,
                                                               allow_nan=False).encode()).hexdigest(),
                 "model": self.agentic_model.generator_id, "budget_scope": "one_stage",
                 "expected_request_id": expected_request_id}
        trace.update(role="customer", actor_id=actor, trajectory_id=uuid4().hex,
                     initial_state_id=trace["snapshot_sha256"])
        run.setdefault("agentic_traces", []).append(trace)
        ids = [p["item_id"] for p in context["displayed_items"]] if stage == "response" else None
        output = action_output(stage, allow_control_actions=stage == "query" and "decision_time" in context,
                               displayed_ids=ids)["final_schemas"]

        def guard():
            require(self.runtime.revision == revision, "Market changed during Customer agentic decision")

        async def complete(prompt, tools):
            request, count = prepare_request(prompt, tools, self.agentic_model.tokenizer,
                                             max_output_tokens=self.agentic_model.max_new_tokens)
            request.update(agent_role="customer", decision_stage=stage,
                           final_schema=teacher_output(output, customer_ekb=self.agentic_options.theory))
            call_id = f'{run["id"]}:agentic:{len(run["model_calls"]) + 1}'
            call = {"call_id": call_id, "stage": stage, "input_tokens": count,
                    "request": deepcopy(request), "model": self.agentic_model.generator_id}
            run["model_calls"].append(call)
            try:
                response = await complete_once(self.agentic_model, request, call_id=call_id)
                call["response"] = deepcopy(response)
                return response
            except Exception as error:
                call["error"] = f"{type(error).__name__}: {error}"
                raise

        def validate(value):
            return validate_teacher_final(value, output, trace=trace, customer_ekb=self.agentic_options.theory,
                validate_action=(lambda action: validate_response(action, ids)) if stage == "response" else None)

        try:
            return await run_agentic_workflow(messages=messages, tools=bound.schemas,
                complete=complete, execute_tool=bound.execute, validate_final=validate,
                options=self.agentic_options, trace=trace, guard=guard)
        finally:
            run["tools"].extend({**deepcopy(entry), "stage": stage, "read_only": True}
                                for entry in trace.get("tools", []))

    async def _stage(self, run, stage, *, state, prompt=None):
        displayed_ids = [product["item_id"] for product in state["displayed_items"]] if stage == "response" else None
        allow_wait = stage == "query" and "decision_time" in state

        def validate(args):
            if allow_wait and args in ({"action": "no_action"}, {"action": "exit"}):
                return args
            if stage == "response":
                return validate_response(args, displayed_ids)
            validate_arguments(args, action_output(stage)["final_schemas"][0])
            return args

        if prompt is None:
            prompt = customer_decision_messages(state, response=stage == "response")
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
