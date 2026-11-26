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
from agent_simulation.inference.requests import prepare_request, ContextCapacityError
from agent_simulation.logs import ModelLog
from agent_simulation.inference.vllm import _generation_schema


class AgenticClient:
    """Keep tool calls intact; the bounded workflow validates tools and final actions."""

    def __init__(self, url, model, tokenizer, *, max_new_tokens=2048, timeout=120,
                 temperature=0, seed=None, log_path=None, transport=None, context_limit=None, stream=False):
        require(bool(model) and type(max_new_tokens) is int and max_new_tokens > 0,
                "Invalid agentic model or generation budget")
        require(math.isfinite(timeout) and timeout > 0, "Invalid agentic timeout")
        require(math.isfinite(temperature) and 0 <= temperature <= 2,
                "Invalid agentic temperature")
        require(context_limit is None or (type(context_limit) is int and context_limit > max_new_tokens),
                "Invalid served context limit")
        self.stream = stream
        self.context_limit = context_limit
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
        if self.context_limit is not None and count + self.max_new_tokens > self.context_limit:
            raise ContextCapacityError("Agentic request exceeds the served model context limit")
        payload = {"model": self.model, "messages": prepared["messages"],
                   "max_tokens": self.max_new_tokens, "temperature": self.temperature,
                   "n": 1, "stream": self.stream,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if self.seed is not None:
            payload["seed"] = self.seed
        if prepared["tools"]:
            budget = next((message["content"] for message in reversed(prepared["messages"])
                           if message.get("role") == "user" and isinstance(message.get("content"), str)
                           and message["content"].startswith("Runtime budget for this decision stage: ")), None)
            disabled = budget is not None and json.loads(budget.split(": ", 1)[1])["instruction"].startswith("No further tools")
            payload.update(tools=prepared["tools"], tool_choice="none" if disabled else "auto")
        if request.get("final_schema") and (not prepared["tools"] or payload.get("tool_choice") == "none"):
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "final_decision", "schema": _generation_schema(request["final_schema"]), "strict": True}}
        record = {"call_id": call_id, "model": self.generator_id,
                  "seed": self.seed, "temperature": self.temperature,
                  "served_model": self.model, "request": deepcopy(request),
                  "input_tokens": count, "output_tokens": 0, "finished": False}
        try:
            if self.stream:
                payload["stream_options"] = {"include_usage": True}
                data = self._stream_response(payload, record)
            else:
                response = self.http.post("chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                record.update(ttft_seconds=None, decode_tokens_per_second=None,
                              timing_protocol="non-streaming; TTFT unavailable")
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

    def _stream_response(self, payload, record):
        started = time.monotonic()
        first = last = None
        content, calls, usage, reason = [], {}, {}, None
        with self.http.stream("POST", "chat/completions", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if value == "[DONE]":
                    break
                chunk = json.loads(value)
                if chunk.get("error"):
                    raise RuntimeError(str(chunk["error"]))
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") or delta.get("tool_calls"):
                        now = time.monotonic()
                        first = now if first is None else first
                        last = now
                    if delta.get("content"):
                        content.append(delta["content"])
                    for fragment in delta.get("tool_calls", []):
                        call = calls.setdefault(fragment["index"], {"id": "", "type": "function",
                            "function": {"name": "", "arguments": ""}})
                        if fragment.get("id"):
                            call["id"] = fragment["id"]
                        for key in ("name", "arguments"):
                            call["function"][key] += fragment.get("function", {}).get(key) or ""
                    reason = choice.get("finish_reason") or reason
        record.update(ttft_seconds=first-started if first is not None else None,
            decode_tokens_per_second=((usage.get("completion_tokens", 0)-1)/(last-first)
                if first is not None and last > first and usage.get("completion_tokens", 0)>1 else None),
            timing_protocol="client first nonempty SSE delta; decode rate approximated from server token usage",
            http_seconds=time.monotonic()-started)
        return {"usage": usage, "choices": [{"finish_reason": reason, "message": {
            "role": "assistant", "content": "".join(content) or None,
            "tool_calls": [calls[i] for i in sorted(calls)]}}]}

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
