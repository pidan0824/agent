"""Shared single-call decisions, tool execution and memory updates."""

from copy import deepcopy
from time import perf_counter

from agent_simulation.data.common import require
from agent_simulation.agents.decisions import decision_request
from agent_simulation.inference.client import ModelClient, complete_once
from agent_simulation.environment.market import MarketEnvironment
from agent_simulation.agents.schemas import fields
from agent_simulation.agents.memory.validation import ReflectionDraftError


class BaseAgent:
    def __init__(
        self,
        runtime: MarketEnvironment,
        model: ModelClient,
        tokenizer,
        *,
        summary_models=None,
    ):
        require(callable(getattr(model, "complete", None)), "Model must implement complete")
        require(bool(tokenizer.chat_template), "Official chat template is missing")
        self.runtime, self.model, self.tokenizer = runtime, model, tokenizer
        self.summary_models = dict(summary_models) if summary_models is not None else None
        if self.summary_models is not None:
            fields(self.summary_models, {"generate", "model_id"}, {"embedder", "encoder_id"})
            require(
                callable(self.summary_models["generate"]), "Memory generation callback is missing"
            )
            require(
                ("embedder" in self.summary_models) == ("encoder_id" in self.summary_models),
                "Provide both embedding callback and identity",
            )
            if "embedder" in self.summary_models:
                runtime.long_term_memory.bind_encoder(
                    self.summary_models["embedder"], self.summary_models["encoder_id"]
                )
        self.logs = runtime.logs

    @property
    def engine(self):
        return "automatic-context-workflow-v2"

    def prepare_request(self, messages, tools, stage, *, role, allow_wait=False,
                        displayed_ids=None, price_ids=None):
        return decision_request(messages, tools, self.tokenizer, role=role, stage=stage,
            max_output_tokens=getattr(self.model, 'max_new_tokens', 0),
            allow_wait=allow_wait, displayed_ids=displayed_ids, price_ids=price_ids)

    async def _complete(self, run, messages, tools, stage, *, allow_wait=False,
                        displayed_ids=None, price_ids=None):
        request, count = self.prepare_request(messages, tools, stage, role=run['role'],
                                             allow_wait=allow_wait,
                                             displayed_ids=displayed_ids, price_ids=price_ids)
        before = self.runtime.revision
        call_id = f'{run["id"]}:{len(run["model_calls"]) + 1}'
        call = {"call_id": call_id, "stage": stage, "input_tokens": count,
                "request": deepcopy(request)}
        run["model_calls"].append(call)
        try:
            response = await complete_once(self.model, request, call_id=call_id)
            require(self.runtime.revision == before, "State changed during model generation")
            call["response"] = deepcopy(response)
            return response
        except Exception as error:
            call["error"] = f"{type(error).__name__}: {error}"
            raise

    def _execute(self, run, name, arguments, *, expected_request_id=None):
        entry = {"name": name, "arguments": deepcopy(arguments)}
        run["tools"].append(entry)
        try:
            result = self.runtime.execute(run["role"], run["actor"], name, arguments,
                                          expected_request_id=expected_request_id)
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
            raise
        entry["result"] = deepcopy(result)
        return result

    def _update_long_term_memory(self, run, actors):
        if self.summary_models is None:
            return
        for role, actor in sorted(set(actors)):
            while True:
                state = self.runtime.snapshot()
                if not self.runtime.long_term_memory.status(role, actor, state=state)["generation_due"]:
                    break
                if not self.runtime.long_term_memory.can_attempt(role, actor, state=state):
                    break
                started = perf_counter()
                try:
                    result = self.runtime.long_term_memory.update(
                        role, actor, **self.summary_models
                    )
                except ReflectionDraftError as error:
                    run["memory_updates"].append(
                        {
                            "role": role,
                            "actor": actor,
                            "seconds": perf_counter() - started,
                            "result": {
                                "status": "failed",
                                "error": str(error),
                                "audit_id": error.audit_id,
                            },
                        }
                    )
                    break
                receipt = {k: result[k] for k in ("status", "operation", "processed_events", "audit_id")}
                run["memory_updates"].append({"role": role, "actor": actor,
                                             "seconds": perf_counter() - started,
                                             "result": receipt})
