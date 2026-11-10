"""Official chat-template rendering and model request capacity checks."""

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from agent_simulation.data.common import require


class ContextCapacityError(ValueError):
    """A complete decision cannot fit; distinct from corrupt facts or broken code."""


_MODEL_POLICY = {
    key: value for key, value in json.loads(Path(__file__).with_name("model_policy.json").read_text()).items()
    if key != "history_windows"
}


def model_policy():
    """Return process-level model settings without rereading the file per sample."""
    return _MODEL_POLICY.copy()


def render_chat(messages, tools, tokenizer, *, tokenize, add_generation_prompt, **options):
    options.setdefault("enable_thinking", False)
    return tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
        return_dict=False, **options,
    )


def prompt_tokens(messages, tools, tokenizer):
    require(bool(getattr(tokenizer, "chat_template", None)), "Official chat template is missing")
    return render_chat(
        messages, tools, tokenizer, tokenize=True, add_generation_prompt=True,
        tokenizer_kwargs={"verbose": False},
    )


def input_token_budget(max_output_tokens=0):
    reserve = max(_MODEL_POLICY["reserved_output_tokens"], max_output_tokens)
    return _MODEL_POLICY["context_limit"] - reserve


def prepare_request(messages, tools, tokenizer, *, max_output_tokens=0):
    """Reject overflow before inference; preserve all supplied history and tool results."""
    request = {"messages": deepcopy(messages), "tools": deepcopy(tools)}
    tokens = prompt_tokens(request["messages"], request["tools"], tokenizer)
    count = len(tokens)
    budget = input_token_budget(max_output_tokens)
    require(
        count <= budget,
        f"Model context overflow: {count} input tokens plus {_MODEL_POLICY['context_limit'] - budget} output reserve",
    )
    _attach_tokens(request, tokenizer, tokens)
    return request, count


def load_tokenizer(model, *, revision=None, local_files_only=False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=revision, local_files_only=local_files_only, trust_remote_code=False
    )
    if not tokenizer.chat_template:
        raise ValueError("Selected checkpoint has no official chat template")
    return tokenizer


def render_request(tokenizer, request, **template_options):
    """Return an official generation prompt; model-specific options are explicit."""
    return render_chat(
        request["messages"], request["tools"], tokenizer,
        tokenize=False,
        add_generation_prompt=True,
        **template_options,
    )


def _prompt_digest(messages, tools):
    encoded = json.dumps([messages, tools], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@lru_cache(maxsize=8)
def tokenizer_identity(tokenizer, template):
    template_identity = [template, tokenizer.special_tokens_map]
    if "enable_thinking" in template:
        template_identity.append({"enable_thinking": False})
    encoded = tokenizer.backend_tokenizer.to_str() + json.dumps(
        template_identity, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _attach_tokens(request, tokenizer, tokens):
    if getattr(tokenizer, "backend_tokenizer", None) is not None:
        request["_prompt_tokens"] = {
            "ids": list(tokens),
            "source": _prompt_digest(request["messages"], request["tools"]),
            "tokenizer": tokenizer_identity(tokenizer, tokenizer.chat_template),
        }


def read_tokens(request, tokenizer):
    """Reuse the exact prompt tokens counted during request preparation."""
    prepared = request["_prompt_tokens"]
    require(
        prepared["source"] == _prompt_digest(request["messages"], request["tools"]),
        "Prepared prompt content changed",
    )
    require(
        prepared["tokenizer"] == tokenizer_identity(tokenizer, tokenizer.chat_template),
        "Prepared prompt tokenizer differs",
    )
    return prepared["ids"]
