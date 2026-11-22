"""Bounded, observable tool use over a private Customer decision snapshot."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from inspect import isawaitable
import json

from agent_simulation.agents.schemas import function, strict_json, validate_arguments
from agent_simulation.data.common import require
from agent_simulation.inference.client import ModelOutputError, response_tool_calls


@dataclass(frozen=True)
class AgenticOptions:
    max_model_calls: int = 6
    max_tool_calls: int = 8
    max_tools_per_call: int = 3
    max_tool_output_chars: int = 12000
    max_total_tool_output_chars: int = 40000
    max_total_model_output_chars: int = 32000
    theory: bool = True

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name != "theory":
                require(type(value) is int and value > 0, f"{name} must be a positive integer")
        require(type(self.theory) is bool, "theory must be boolean")


class AgenticBudgetExceeded(ModelOutputError):
    """A proposal exhausted its budget; no implicit action may be committed."""


def options_from(value):
    return value if isinstance(value, AgenticOptions) else AgenticOptions(**(value or {}))


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


BUDGET_PREFIX = "Runtime budget for this decision stage: "


def budget_message(options, *, turn=0, tools_used=0, tool_output_chars=0,
                   model_output_chars=0, available=True):
    return {"role": "user", "content": BUDGET_PREFIX + _json({
        "model_calls_remaining_including_this": options.max_model_calls - turn,
        "tool_calls_remaining": options.max_tool_calls - tools_used,
        "tools_per_call_limit": options.max_tools_per_call,
        "tool_output_chars_remaining": options.max_total_tool_output_chars - tool_output_chars,
        "model_output_chars_remaining": options.max_total_model_output_chars - model_output_chars,
        "instruction": ("Use optional tools or return the required final JSON."
                        if available else "No further tools are allowed. Return the required final JSON now."),
    })}


def initial_agentic_request(messages, tools, options=None):
    """Prepare the exact initial visible conversation for previews and execution."""
    options = options_from(options)
    conversation = deepcopy(messages)
    available = deepcopy(tools)
    has_budget = any(m.get("role") == "user" and str(m.get("content", "")).startswith(BUDGET_PREFIX)
                     for m in conversation)
    budget_index = (len(conversation) - 1 if has_budget and conversation[-1].get("role") == "user"
                    else 1 if conversation and conversation[0].get("role") == "system" else 0)
    conversation.insert(budget_index, budget_message(options, available=bool(available) and options.max_model_calls > 1))
    return conversation, available, budget_index


def _bounded_result(result, limit):
    raw = _json(result)
    if len(raw) <= limit:
        return raw, False
    # Leave enough room for metadata and JSON escaping; count the serialized result.
    preview = raw[:max(0, limit // 2)]
    while preview:
        value = _json({"truncated": True, "original_chars": len(raw), "preview": preview,
                       "instruction": "Incomplete evidence; request a smaller result if needed."})
        if len(value) <= limit:
            return value, True
        preview = preview[:len(preview) // 2]
    minimal = '{"truncated":true}'
    if len(minimal) > limit:
        raise AgenticBudgetExceeded("Tool output budget exhausted")
    return minimal, True


async def run_agentic_workflow(*, messages, tools, complete, execute_tool, validate_final,
                              options=None, trace=None, guard=None):
    """Run one decision. All side effects are delegated to a read-only tool binding.

    The trace contains the exact visible conversation and full raw tool results,
    including the bounded result actually returned to the model. It is suitable
    for replay and later filtering; no label is passed to this function.
    """
    options = options_from(options)
    trace = {} if trace is None else trace
    conversation, _, budget_index = initial_agentic_request(messages, tools, options)
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    require(len(schemas) == len(tools), "Duplicate agentic tool name")
    trace.update(status="running", options=asdict(options), model_calls=[], tools=[], available_tools=deepcopy(tools),
                 messages=conversation, tool_output_chars=0, model_output_chars=0)
    seen_ids = {c["id"] for message in conversation for c in message.get("tool_calls", [])}

    def check():
        if guard is not None:
            guard()

    try:
        for turn in range(options.max_model_calls):
            check()
            # Reserve a final call for the decision, even when tools used all their budget.
            available = tools if (turn + 1 < options.max_model_calls
                                  and len(trace["tools"]) < options.max_tool_calls
                                  and trace["tool_output_chars"] < options.max_total_tool_output_chars) else []
            budget = budget_message(options, turn=turn, tools_used=len(trace["tools"]),
                tool_output_chars=trace["tool_output_chars"], model_output_chars=trace["model_output_chars"],
                available=bool(available))
            if turn:
                conversation.append(budget)
            else:
                conversation[budget_index] = budget
            call = {"index": turn + 1, "messages": deepcopy(conversation),
                    "tools": deepcopy(tools), "tools_enabled": bool(available)}
            trace["model_calls"].append(call)
            # Keep the tool-definition prefix stable for trajectory training.
            # The runtime user message and HTTP tool_choice disable further calls.
            reply = complete(deepcopy(conversation), deepcopy(tools))
            if isawaitable(reply):
                reply = await reply
            call["response"] = deepcopy(reply)
            check()
            trace["model_output_chars"] += len(_json(reply))
            if trace["model_output_chars"] > options.max_total_model_output_chars:
                raise AgenticBudgetExceeded("Total model output budget exceeded", raw_output=reply)
            calls = response_tool_calls(reply)
            conversation.append(deepcopy(reply))
            if not calls:
                if reply["content"] is None:
                    raise ModelOutputError("Missing final JSON decision", raw_output=reply)
                try:
                    final = strict_json(reply["content"])
                    require(isinstance(final, dict), "Expected a JSON object")
                    final = validate_final(final)
                except (ValueError, TypeError) as error:
                    raise ModelOutputError(str(error), raw_output=reply) from error
                trace.update(status="complete", result=deepcopy(final), final_action=deepcopy(final))
                return final
            if not available:
                raise AgenticBudgetExceeded("Final call must return a decision without tools", raw_output=reply)
            if len(calls) > options.max_tools_per_call:
                raise AgenticBudgetExceeded("Per-call tool budget exceeded", raw_output=reply)
            if len(trace["tools"]) + len(calls) > options.max_tool_calls:
                raise AgenticBudgetExceeded("Total tool-call budget exceeded", raw_output=reply)
            ids = [c["id"] for c in calls]
            if len(ids) != len(set(ids)) or seen_ids.intersection(ids):
                raise ModelOutputError("Tool-call IDs must be unique within a trajectory", raw_output=reply)
            seen_ids.update(ids)
            for requested in calls:
                check()
                name = requested["function"]["name"]
                entry = {"id": requested["id"], "name": name,
                         "arguments": deepcopy(requested["function"]["arguments"])}
                trace["tools"].append(entry)
                try:
                    args = entry["arguments"]
                    if isinstance(args, str):
                        args = strict_json(args)
                    if name not in schemas:
                        raise PermissionError("Tool unavailable in this decision stage")
                    validate_arguments(args, schemas[name])
                    entry["arguments"] = deepcopy(args)
                    result = execute_tool(name, deepcopy(args))
                    if isawaitable(result):
                        result = await result
                    entry["result"] = deepcopy(result)
                except (ValueError, PermissionError) as error:
                    result = {"error": f"{type(error).__name__}: {error}"}
                    entry["error"] = result["error"]
                check()
                remaining = options.max_total_tool_output_chars - trace["tool_output_chars"]
                encoded, truncated = _bounded_result(result, min(options.max_tool_output_chars, remaining))
                trace["tool_output_chars"] += len(encoded)
                entry.update(returned_content=encoded, truncated=truncated)
                conversation.append({"role": "tool", "tool_call_id": requested["id"],
                                     "name": name, "content": encoded})
        raise AgenticBudgetExceeded("Model-call budget exhausted")
    except BaseException as error:
        # Sibling decision failures cancel this coroutine. Preserve that terminal
        # state too, so an interrupted trajectory never remains marked running.
        trace.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise


def teacher_output(action_schemas, *, customer_ekb=False):
    """Final action envelope; Customer theory runs use the structured EKB contract."""
    schema = {"type": "object", "properties": {
        "rationale": {"type": "object", "properties": {
            "hypothesis": {"type": "string", "minLength": 1, "maxLength": 240},
            "evidence": {"type": "array", "minItems": 1, "maxItems": 5,
                         "items": {"type": "string", "minLength": 1, "maxLength": 500}},
            "uncertainty": {"type": "string", "minLength": 1, "maxLength": 500}},
            "required": ["hypothesis", "evidence", "uncertainty"], "additionalProperties": False},
        "action": action_schemas[0] if len(action_schemas) == 1 else {"anyOf": action_schemas}},
        "required": ["rationale", "action"], "additionalProperties": False}
    if customer_ekb:
        from agent_simulation.agents.prompts.customer_ekb import EKB_SCHEMA
        schema["properties"]["rationale"] = deepcopy(EKB_SCHEMA)
    return schema


def validate_teacher_final(value, action_schemas, *, trace, validate_action=None, customer_ekb=False):
    validate_arguments(value, teacher_output(action_schemas, customer_ekb=customer_ekb))
    action = deepcopy(value["action"])
    if "prices" in action:
        ids = [quote["item_id"] for quote in action["prices"]]
        require(len(ids) == len(set(ids)), "Duplicate price target")
    if validate_action is not None:
        action = validate_action(action)
    trace["rationale"] = deepcopy(value["rationale"])
    return action


TEACHER_OUTPUT_NOTE = (
    "Return the required JSON envelope with rationale and action. The rationale is a brief visible "
    "decision justification: a tentative hypothesis, 1–5 specific available observations (or explicit "
    "absence of evidence), and uncertainty. It is generated interpretation, not a claim to know the "
    "person's real thoughts. Tools are optional; a rationale is required even without tools. "
    "Do not output a hidden thought transcript or invent observations. Only action is executed."
)


def customer_auxiliary_tools(*, theory=True):
    from agent_simulation.agents.prompts.customer_agentic import skill_reference_names

    return [function("read_skill_reference", "Read an optional interpretation reference, not behavioral evidence.",
                     {"name": {"type": "string", "enum": list(skill_reference_names(theory=theory))}}, ["name"]),
            function("record_behavior_hypothesis", "Record a tentative behavior description with brief visible evidence. This does not change market state.",
                     {"decision_type": {"type": "string", "minLength": 1, "maxLength": 160},
                      "evidence": {"type": "array", "maxItems": 5,
                                   "items": {"type": "string", "maxLength": 400}},
                      "confidence": {"type": "string", "enum": ["low", "medium", "high"]}},
                     ["decision_type", "evidence", "confidence"])]


def execute_customer_auxiliary(name, args, *, theory=True):
    if name == "read_skill_reference":
        from agent_simulation.agents.prompts.customer_agentic import read_reference
        return {"name": args["name"], "content": read_reference(args["name"], theory=theory)}
    if name == "record_behavior_hypothesis":
        return {"recorded": True, "hypothesis": deepcopy(args), "status": "unverified_hypothesis"}
    raise PermissionError("Unknown auxiliary tool")


class FrozenCustomerTools:
    """Bind query backends to a private snapshot; never call runtime.execute."""

    def __init__(self, runtime, actor, state, stage, *, theory=True):
        from agent_simulation.environment.operations.api import CUSTOMER_QUERY_TOOLS
        require(stage in ("query", "response"), "Unknown Customer stage")
        require(str(actor) in state["customers"], "Unregistered Customer")
        self.runtime, self.actor = runtime, actor
        self.state, self.stage, self.theory = deepcopy(state), stage, theory
        allowed = {"query_customer_history", "query_long_term_memory"}
        if stage == "response":
            require(str(actor) in state["pending"], "No frozen displayed request")
            allowed.add("query_displayed_products")
        self.schemas = [deepcopy(t) for t in CUSTOMER_QUERY_TOOLS if t["function"]["name"] in allowed]
        self.schemas += customer_auxiliary_tools(theory=theory)
        self.parameters = {t["function"]["name"]: t["function"]["parameters"] for t in self.schemas}

    def execute(self, name, arguments):
        from agent_simulation.environment.operations.queries import query_customer_history
        from agent_simulation.agents.context.shopping import displayed_details
        if name not in self.parameters:
            raise PermissionError("Tool unavailable in this Customer stage")
        validate_arguments(arguments, self.parameters[name])
        if name == "query_customer_history":
            result = query_customer_history(self.state, self.runtime.data, self.runtime.catalog,
                                            self.actor, arguments)
        elif name == "query_long_term_memory":
            result = self.runtime.long_term_memory.context("customer", self.actor, arguments["query"],
                                                          state=self.state)
        elif name == "query_displayed_products":
            result = displayed_details(self.state["pending"][str(self.actor)]["displayed_items"],
                                       arguments["item_ids"])
        else:
            result = execute_customer_auxiliary(name, arguments, theory=self.theory)
        return deepcopy(result)


def seller_auxiliary_tools(*, theory=True):
    from agent_simulation.agents.prompts.seller_agentic import skill_reference_names
    tools = customer_auxiliary_tools(theory=theory)
    tools[0]["function"]["parameters"]["properties"]["name"]["enum"] = skill_reference_names(theory=theory)
    return tools


def execute_seller_auxiliary(name, args, *, theory=True):
    if name == "read_skill_reference":
        from agent_simulation.agents.prompts.seller_agentic import read_reference
        return {"name": args["name"], "content": read_reference(args["name"], theory=theory)}
    return execute_customer_auxiliary(name, args, theory=theory)


class FrozenSellerTools:
    """Owner-bound Seller queries; dispatch pure backends without observing tools."""

    def __init__(self, runtime, actor, state, stage="seller", *, theory=True):
        from agent_simulation.environment.operations.api import runtime_tools, SELLER_QUERY_TOOLS
        require(stage == "seller", "Unknown Seller stage")
        self.runtime, self.actor = runtime, actor
        self.state, self.theory = deepcopy(state), theory
        self.owned = sorted(int(item) for item, product in state["products"].items()
                            if product["seller_id"] == actor)
        require(self.owned, "Unregistered Seller")
        allowed = {tool["function"]["name"] for tool in SELLER_QUERY_TOOLS}
        self.schemas = [tool for tool in runtime_tools(tick="clock" in state)
                        if tool["function"]["name"] in allowed]
        self.schemas += seller_auxiliary_tools(theory=theory)
        self.parameters = {tool["function"]["name"]: tool["function"]["parameters"] for tool in self.schemas}

    def execute(self, name, arguments):
        from agent_simulation.environment.operations.api import dispatch
        if name not in self.parameters:
            raise PermissionError("Tool unavailable in this Seller stage")
        validate_arguments(arguments, self.parameters[name])
        if name in ("read_skill_reference", "record_behavior_hypothesis"):
            result = execute_seller_auxiliary(name, arguments, theory=self.theory)
        else:
            result = dispatch(self.runtime, self.state, "seller", self.actor, self.owned, name, arguments)
        return deepcopy(result)
