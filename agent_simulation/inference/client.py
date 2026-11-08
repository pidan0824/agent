"""Backend-independent model contract and PydanticAI model adapters."""

import asyncio
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai.direct import model_request
from pydantic_ai.messages import (
    ModelRequest, SystemPromptPart, UserPromptPart, TextPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.output import OutputObjectDefinition

from agent_simulation.agents.schemas import fields, strict_json, validate_arguments


class ModelClient(Protocol):
    """Adapters expose their tokenizer, generation budget and logged model identity."""

    tokenizer: object
    max_new_tokens: int
    generator_id: str

    def complete(self, request: dict, *, call_id: str) -> dict:
        """Generate a normalized assistant reply or raise an explicit error."""
        ...


class ModelOutputError(ValueError):
    """Invalid or unfinished model output, preserving the original response."""

    def __init__(self, message, *, raw_output=None, finished=None):
        super().__init__(message)
        self.raw_output = raw_output
        self.finished = finished


def response_tool_calls(reply):
    """Validate an assistant envelope before consuming its content or translating tool calls."""
    try:
        fields(reply, {"role", "content"}, {"tool_calls"})
    except ValueError as error:
        raise ModelOutputError(str(error), raw_output=reply) from error
    if reply["role"] != "assistant":
        raise ModelOutputError("Expected an assistant message", raw_output=reply)
    if reply["content"] is not None and not isinstance(reply["content"], str):
        raise ModelOutputError("Invalid assistant content", raw_output=reply)
    calls = reply.get("tool_calls", [])
    if not isinstance(calls, list):
        raise ModelOutputError("tool_calls must be a list", raw_output=reply)
    for call in calls:
        try:
            fields(call, {"id", "type", "function"})
            fields(call["function"], {"name", "arguments"})
        except ValueError as error:
            raise ModelOutputError(str(error), raw_output=reply) from error
        if call["type"] != "function" or not isinstance(call["id"], str) or not call["id"]:
            raise ModelOutputError("Invalid tool call identity", raw_output=reply)
        function = call["function"]
        if (not isinstance(function["name"], str) or not function["name"]
                or not isinstance(function["arguments"], (dict, str))):
            raise ModelOutputError("Invalid tool call function", raw_output=reply)
    return calls


def output_schema(request):
    """Resolve the same JSON decision schema for all inference providers."""
    if request.get("tools"):
        raise ValueError("Model decisions require JSON without tools")
    if "action_output" not in request:
        return request.get("json_schema")
    if "json_schema" in request:
        raise ValueError("Choose one output contract")
    alternatives = request["action_output"]["final_schemas"]
    if not alternatives:
        raise ValueError("No legal action output")
    return alternatives[0] if len(alternatives) == 1 else {"type": "object", "anyOf": alternatives}


def finish_reply(raw, *, finished, request):
    """Normalize a logged generation and preserve raw reflection drafts."""
    if not finished:
        raise ModelOutputError("Generation did not finish; refusing a truncated decision",
                               raw_output=raw, finished=False)
    if "memory_generation_method" not in request:
        try:
            value = strict_json(raw)
            schema = output_schema(request)
            if schema is not None:
                validate_arguments(value, schema, path="output")
        except (ValueError, TypeError) as error:
            raise ModelOutputError(str(error), raw_output=raw, finished=True) from error
        raw = raw.strip()
    return {"role": "assistant", "content": raw, "tool_calls": []}


async def complete_once(client, request, *, call_id):
    """Use native asynchronous dispatch when declared by the adapter."""
    if callable(getattr(type(client), "acomplete", None)):
        return await client.acomplete(request, call_id=call_id)
    return client.complete(request, call_id=call_id)


@dataclass(kw_only=True)
class DecisionParameters(ModelRequestParameters):
    call_id: str
    decoding: dict


class FrameworkClient:
    def __init__(self, provider, tokenizer, max_new_tokens, generator_id=None):
        self.provider = provider
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.generator_id = generator_id or provider.model_name

    async def acomplete(self, request, *, call_id):
        parts = {"system": SystemPromptPart, "user": UserPromptPart}
        messages = ModelRequest(parts=[parts[m["role"]](m["content"]) for m in request["messages"]])
        schema = output_schema(request)
        params = DecisionParameters(
            call_id=call_id,
            decoding={k: v for k, v in request.items() if k not in ("messages", "tools")},
        )
        if schema is not None:
            params.output_mode = "native"
            params.output_object = OutputObjectDefinition(json_schema=schema)
        params.allow_text_output = True
        response = await model_request(
            self.provider,
            [messages],
            model_request_parameters=params,
            model_settings={"max_tokens": self.max_new_tokens},
            instrument=False,
        )
        if response.finish_reason == "error":
            raise RuntimeError("Model provider reported a generation error")
        if response.finish_reason in ("length", "content_filter"):
            raise ModelOutputError(f"Incomplete model response: {response.finish_reason}",
                                   raw_output=response.text, finished=False)
        if any(not isinstance(p, TextPart) for p in response.parts):
            raise ModelOutputError("Expected assistant JSON text", raw_output=response.text)
        # Reflection validates the raw text with its authorized evidence at write time.
        if schema is not None and "memory_generation_method" not in request:
            try:
                validate_arguments(strict_json(response.text), schema, path="output")
            except (ValueError, TypeError) as exc:
                raise ModelOutputError(str(exc), raw_output=response.text, finished=True) from exc
        return {"role": "assistant", "content": response.text, "tool_calls": []}

    def complete(self, request, *, call_id):
        return asyncio.run(self.acomplete(request, call_id=call_id))
