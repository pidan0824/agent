"""OpenAI-compatible tool calling for the optional Qwen3-32B Customer policy."""

import asyncio
from copy import deepcopy
import math
import json
import os
import time

import httpx

from agent_simulation.data.common import require
from agent_simulation.inference.client import ModelOutputError, response_tool_calls
from agent_simulation.inference.requests import prepare_request
from agent_simulation.logs import ModelLog


class AgenticClient:
    """Keep tool calls intact; the bounded workflow validates tools and final actions."""

    def __init__(self, url, model, tokenizer, *, max_new_tokens=2048, timeout=120,
                 temperature=0, seed=None, log_path=None, transport=None):
        require(bool(model) and type(max_new_tokens) is int and max_new_tokens > 0,
                "Invalid agentic model or generation budget")
        require(math.isfinite(timeout) and timeout > 0, "Invalid agentic timeout")
        require(math.isfinite(temperature) and 0 <= temperature <= 2,
                "Invalid agentic temperature")
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens, self.temperature, self.seed = max_new_tokens, temperature, seed
        self.log = ModelLog(log_path) if log_path is not None else None
        key = os.environ.get("AGENTIC_API_KEY") or os.environ.get("VLLM_API_KEY")
        self.http = httpx.Client(base_url=url.rstrip("/") + "/", timeout=timeout,
                                 headers={"Authorization": "Bearer " + key} if key else {},
                                 transport=transport)

    @property
    def generator_id(self):
        # Multi-trajectory generation changes the sampling seed between calls.
        return (f"agentic:{self.model}:temperature={self.temperature}:"
                f"max_new_tokens={self.max_new_tokens}:seed={self.seed}")

    def check_models(self):
        response = self.http.get("models")
        response.raise_for_status()
        require(self.model in {m["id"] for m in response.json()["data"]},
                "Agentic model is not advertised by the configured server")
        return {"backend": "openai-compatible-tools", "model": self.model,
                "remote_weights_verified": False}

    def complete(self, request, *, call_id):
        started = time.monotonic()
        # Recount the growing conversation, including tool definitions/results, each turn.
        prepared, count = prepare_request(request["messages"], request.get("tools", []),
                                          self.tokenizer, max_output_tokens=self.max_new_tokens)
        payload = {"model": self.model, "messages": prepared["messages"],
                   "max_tokens": self.max_new_tokens, "temperature": self.temperature,
                   "n": 1, "stream": False,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if self.seed is not None:
            payload["seed"] = self.seed
        if prepared["tools"]:
            budget = next((message["content"] for message in reversed(prepared["messages"])
                           if message.get("role") == "user" and isinstance(message.get("content"), str)
                           and message["content"].startswith("Runtime budget for this decision stage: ")), None)
            disabled = budget is not None and json.loads(budget.split(": ", 1)[1])["instruction"].startswith("No further tools")
            payload.update(tools=prepared["tools"], tool_choice="none" if disabled else "auto")
        record = {"call_id": call_id, "model": self.generator_id,
                  "seed": self.seed, "temperature": self.temperature,
                  "served_model": self.model, "request": deepcopy(request),
                  "input_tokens": count, "output_tokens": 0, "finished": False}
        try:
            response = self.http.post("chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            choice, = data["choices"]
            reason, message = choice.get("finish_reason"), choice["message"]
            record.update(raw_output=deepcopy(message), finish_reason=reason,
                          input_tokens=data.get("usage", {}).get("prompt_tokens", count),
                          output_tokens=data.get("usage", {}).get("completion_tokens", 0))
            if reason not in ("stop", "tool_calls"):
                raise ModelOutputError(f"Incomplete agentic response: {reason}",
                                       raw_output=message, finished=False)
            reply = {"role": message.get("role"), "content": message.get("content"),
                     "tool_calls": message.get("tool_calls") or []}
            calls = response_tool_calls(reply)
            if reason == "tool_calls" and not calls:
                raise ModelOutputError("Tool-call finish reason without calls", raw_output=message)
            if not calls and not reply["content"]:
                raise ModelOutputError("Empty agentic response", raw_output=message)
            record["finished"] = True
            return reply
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            if self.log is not None:
                self.log.append(record)

    async def acomplete(self, request, *, call_id):
        return await asyncio.to_thread(self.complete, request, call_id=call_id)

    def close(self):
        self.http.close()


def add_agentic_arguments(parser):
    parser.add_argument("--customer-agentic", action="store_true",
                        help="Use one external model for Customer evidence selection and decisions")
    parser.add_argument("--seller-agentic", action="store_true",
                        help="Use one external model for Seller evidence selection and pricing")
    parser.add_argument("--agentic-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--agentic-model", default="Qwen3-32B")
    parser.add_argument("--agentic-tokenizer", default="models/Qwen3-32B")
    parser.add_argument("--agentic-max-calls", type=int, default=6)
    parser.add_argument("--agentic-max-tools", type=int, default=8)
    parser.add_argument("--agentic-max-new-tokens", type=int, default=2048)
    parser.add_argument("--agentic-temperature", type=float, default=0)
    import argparse
    parser.add_argument("--agentic-theory", action=argparse.BooleanOptionalAction, default=True)


def agentic_options(args):
    return {"max_model_calls": args.agentic_max_calls,
            "max_tool_calls": args.agentic_max_tools, "theory": args.agentic_theory}
