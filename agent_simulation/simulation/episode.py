"""Run each Tick through pricing, shopping and unified settlement."""

import asyncio
from copy import deepcopy
from agent_simulation.data.common import require
from agent_simulation.agents.concurrency import map_decisions


def create_clock(start, tick_size, max_ticks):
    """Create a clock from the validated Episode settings."""
    return {
        "start": start,
        "time": start,
        "tick": 0,
        "tick_size": tick_size,
        "max_ticks": max_ticks,
        "status": "ready",
        "last_decisions": {},
    }


def advance_clock(clock, *, all_customers_exited):
    """Complete one Tick and return its termination reason, if any."""
    clock["tick"] += 1
    clock["time"] += clock["tick_size"]
    reason = None
    if all_customers_exited:
        reason = "all_customers_exited"
    elif clock["tick"] >= clock["max_ticks"]:
        reason = "max_ticks"
    clock["status"] = "complete" if reason else "ready"
    return reason


class Episode:
    """All Sellers propose, prices apply, all Customers respond, feedback settles.

    Logical time is a configurable index, not reconstructed wall-clock time.
    Failed ticks remain failed for inspection; they are never converted to waits.
    """

    def __init__(self, runtime, seller_agent, customer_agent, *, tick_size, max_ticks,
                 max_concurrent_requests=1, fixed_horizon=False):
        require(type(max_concurrent_requests) is int and max_concurrent_requests > 0,
                "Invalid inference concurrency")
        self.concurrency = max_concurrent_requests
        require(type(tick_size) is int and tick_size > 0, "Invalid Tick size")
        require(type(max_ticks) is int and max_ticks > 0, "Invalid Episode horizon")
        require(
            seller_agent.runtime is runtime and customer_agent.runtime is runtime,
            "Agents must share the Episode runtime",
        )
        self.runtime, self.seller, self.customer = runtime, seller_agent, customer_agent
        self.fixed_horizon = fixed_horizon
        cutoff = runtime.snapshot()['historical_cutoff_exclusive']
        runtime.start_episode(create_clock(cutoff, tick_size, max_ticks))

    def _collect(self, runs, decide, *, share_request_limit=False, **kwargs):
        if not runs:
            return

        async def collect():
            if share_request_limit:
                kwargs["request_slots"] = asyncio.Semaphore(self.concurrency)
            await map_decisions(runs, lambda run: decide(run, **kwargs),
                                concurrency=min(self.concurrency, len(runs)))
        asyncio.run(collect())

    def _seller_proposals(self, sellers):
        runs = []
        try:
            for actor in sellers:
                runs.append(self.seller.prepare_review(actor))
            baseline = self.runtime.freeze()
            self._collect(runs, self.seller.review_prices, baseline=baseline, share_request_limit=True)
        except Exception as error:
            for run in runs:
                self.seller.logs.save(run, "failed", f"Seller phase aborted: {error}")
            raise
        return runs, baseline

    def _customer_decisions(self, customers, *, state):
        runs = []
        try:
            for actor in customers:
                runs.append(self.customer.prepare_decision(actor, state=state))
            # Reflections finish before Query; searches finish before Response.
            self._collect(runs, self.customer.choose_query, state=self.runtime.snapshot())
            for run in runs:
                self.customer.open_display(run)
            shopping = [run for run in runs if "displayed_list" in run]
            if shopping:
                self._collect(shopping, self.customer.choose_response, share_request_limit=True,
                              state=self.runtime.snapshot())
        except Exception as error:
            for run in runs:
                self.customer.logs.save(run, "failed", f"Customer phase aborted: {error}")
            raise
        return runs

    def step(self):
        state, trace = self.runtime.claim_tick()
        clock = state['clock']
        sellers = sorted({p["seller_id"] for p in state["products"].values()})
        customers = sorted(
            int(actor) for actor, customer in state["customers"].items() if not customer.get("exit")
        )
        seller_cutoff = state["simulation_step"]
        settled_phases = []
        try:
            trace["sellers"], baseline = self._seller_proposals(sellers)
            # No Seller can see another Seller's proposal until this barrier.
            proposals = [run for run in trace["sellers"] if run["result"] != {"action": "no_action"}]
            quotes = [(run["actor"], {"prices": run["result"]["prices"]}) for run in proposals]
            try:
                applied = self.runtime.apply_seller_quotes(quotes, expected_revision=baseline.revision)
            except Exception as error:
                for run in trace["sellers"]:
                    self.seller.logs.save(run, "failed", f"Seller settlement aborted: {type(error).__name__}: {error}")
                raise
            settled_phases.append("seller")
            for run, (_, args), result in zip(proposals, quotes, applied):
                run["applied_result"] = result
                run["tools"].append({"name": "submit_prices", "arguments": args, "result": result})
            for run in trace["sellers"]:
                self.seller.logs.save(run, "complete")
            customer_state = self.runtime.snapshot()
            customer_cutoff = customer_state["simulation_step"]
            trace["customers"] = self._customer_decisions(customers, state=customer_state)
            # All response decisions precede all new feedback and memory updates.
            self.customer.settle_staged(trace["customers"], on_settled=lambda: settled_phases.append("customer"))
            with self.runtime.record_tick(trace) as state:
                require(not state["pending"], "Cannot finish Tick with unsettled searches")
                for role, runs, cutoff in (
                    ("seller", trace["sellers"], seller_cutoff),
                    ("customer", trace["customers"], customer_cutoff),
                ):
                    for run in runs:
                        state["clock"]["last_decisions"][f'{role}:{run["actor"]}'] = {
                            "time": clock["time"],
                            "input_step": cutoff,
                        }
                reason = advance_clock(
                    state["clock"],
                    all_customers_exited=not self.fixed_horizon and all(
                        customer.get("exit") for customer in state["customers"].values()
                    ),
                )
                if reason:
                    state["clock"]["termination_reason"] = reason
                    trace["termination_reason"] = reason
                trace["status"] = "complete"
        except Exception as error:
            trace.pop("termination_reason", None)
            trace.update(status="failed", error=f"{type(error).__name__}: {error}", settled_phases=settled_phases)
            with self.runtime.record_tick(trace) as state:
                state["clock"]["status"] = "failed"
                state["clock"]["settled_phases"] = settled_phases
            raise
        return deepcopy(trace)

    def run(self, *, progress=None):
        """Run to the horizon, optionally stopping when all Customers exit."""
        clock = self.runtime.snapshot()["clock"]
        while clock["status"] == "ready":
            tick = clock["tick"]
            if progress is not None:
                progress(tick, "started")
            self.step()
            if progress is not None:
                progress(tick, "complete")
            clock = self.runtime.snapshot()["clock"]
        require(clock["status"] == "complete", "Episode failed")
        return clock
