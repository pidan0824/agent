"""Check capacity and tokenize SFT messages with final-answer-only supervision."""

import json
import math

from agent_simulation.inference.requests import ContextCapacityError, model_policy, render_chat
from agent_simulation.data.common import require


def _payload(sample):
    messages, tools = sample["messages"], sample["tools"]
    if type(tools) is list and not tools and all(
        type(message) is dict
        and all(type(key) is str and type(value) is str for key, value in message.items())
        for message in messages
    ):
        # The strings are immutable, so keep references and preserve message field order.
        return tuple(tuple(message.items()) for message in messages)
    return json.dumps((messages, tools), ensure_ascii=False, allow_nan=False)


class _TokenizedSample(dict):
    """Keep this read's tokens in memory, outside exported sample fields."""

    def __init__(self, sample, tokenizer, answer_start, complete):
        super().__init__(sample)
        self._tokenizer = tokenizer
        self._template = tokenizer.chat_template
        self._payload = _payload(sample)
        self._tokens = answer_start, complete


def _anchored_tokens(messages, tools, tokenizer):
    """Encode the shared prefix once, checking its final boundary at an atomic special token.

    Text before an unstripped special token is an independent tokenizer segment.
    Verify the short header after that token separately; unsupported tokenizers or
    templates retain the original complete-prefix comparison below.
    """
    backend=getattr(tokenizer,'backend_tokenizer',None)
    if not getattr(tokenizer,'is_fast',False) or backend is None:
        return None
    # Qwen uses NFC inside the segments separated by unnormalized special tokens.
    if backend.normalizer is not None and type(backend.normalizer).__name__!='NFC':
        return None
    prefix=render_chat(messages[:-1],tools,tokenizer,tokenize=False,add_generation_prompt=True)
    complete=render_chat(messages,tools,tokenizer,tokenize=False,add_generation_prompt=False)
    if not isinstance(prefix,str) or not isinstance(complete,str) or not complete.startswith(prefix):return None
    candidates=[(prefix.rfind(token),token) for token in tokenizer.all_special_tokens]
    if not candidates:return None
    start,anchor=max(candidates,key=lambda p:(p[0],len(p[1])))
    if start<0 or len(prefix)-start>256:return None
    token_id=tokenizer.convert_tokens_to_ids(anchor)
    added=tokenizer.added_tokens_decoder.get(token_id)
    if added is None or not added.special or added.lstrip or added.rstrip or added.single_word or added.normalized:
        return None
    encoded=tokenizer(complete,add_special_tokens=False,truncation=False,padding=False)
    encoding=encoded.encodings[0]
    index=encoding.char_to_token(start)
    end=encoding.char_to_token(len(prefix)-1)
    if index is None or end is None:return None
    ids=encoded['input_ids']
    if ids[index]!=token_id or encoding.token_to_chars(index)!=(start,start+len(anchor)):return None
    # A merged token crossing into the answer must never receive a prefix mask.
    if encoding.token_to_chars(end)[1]!=len(prefix):return None
    tail=tokenizer(prefix[start:],add_special_tokens=False,truncation=False,padding=False)['input_ids']
    if ids[index:end+1]!=tail:return None
    return end + 1, tuple(ids)


def _tokens(sample, tokenizer):
    if (
        isinstance(sample, _TokenizedSample)
        and sample._tokenizer is tokenizer
        and sample._template == tokenizer.chat_template
        and sample._payload == _payload(sample)
    ):
        return sample._tokens
    messages, tools = sample["messages"], sample["tools"]
    require(messages[-1]["role"] == "assistant", "Missing supervised answer")
    anchored=_anchored_tokens(messages,tools,tokenizer)
    if anchored is not None:return anchored
    prefix = render_chat(messages[:-1], tools, tokenizer, tokenize=True, add_generation_prompt=True)
    complete = render_chat(messages, tools, tokenizer, tokenize=True, add_generation_prompt=False)
    require(complete[: len(prefix)] == prefix, "Answer boundary changed")
    return len(prefix), tuple(complete)


def _capacity(answer_start, complete):
    answer = len(complete) - answer_start
    policy = model_policy()
    return {
        "input_tokens": answer_start,
        "answer_tokens": answer,
        "fits_context": answer_start + max(answer, policy["reserved_output_tokens"])
        <= policy["context_limit"],
    }


def prepare_sample(sample, tokenizer):
    """Reuse one tokenization for capacity checks and the later training batch."""
    answer_start, complete = _tokens(sample, tokenizer)
    result = _TokenizedSample(sample, tokenizer, answer_start, complete)
    result["capacity"] = _capacity(answer_start, complete)
    return result


IGNORE_INDEX = -100


def encode_tokens(sample, tokenizer):
    """Return checked tokens and supervision boundaries without padded training arrays."""
    require(
        [m["role"] for m in sample["messages"]] == ["system", "user", "assistant"],
        "Price/Query/Response SFT must contain one final answer, without invented tool trajectories",
    )
    answer_start, complete = _tokens(sample, tokenizer)
    require(len(complete) > answer_start, "Answer token boundary changed")
    capacity = _capacity(answer_start, complete)
    if not capacity["fits_context"]:
        policy = model_policy()
        reserve = max(capacity["answer_tokens"], policy["reserved_output_tokens"])
        raise ContextCapacityError(
            f"Unresolved context overflow: {sample['sample_id']}; "
            f"input={answer_start}, reserve={reserve}, limit={policy['context_limit']}"
        )
    weight = 1.0
    if sample["sample_id"].startswith("response:"):
        meta = sample.get("batching", {"count": 1, "request_weight": 1.0})
        require(type(meta["count"]) is int and meta["count"] > 0, "Invalid response batch count")
        require(
            math.isclose(meta["request_weight"], 1 / meta["count"], rel_tol=0, abs_tol=1e-12),
            "Response weight differs from one request divided across its batches",
        )
        weight = float(meta["request_weight"])
    return {
        "input_ids": complete,
        "answer_start": answer_start,
        "sample_weight": weight,
        "sample_id": sample["sample_id"],
        "split": sample["split"],
    }


def encode_sample(sample, tokenizer):
    """Materialize model arrays only for training consumers."""
    row = encode_tokens(sample, tokenizer)
    tokens, start = row["input_ids"], row["answer_start"]
    return {**row, "input_ids": list(tokens), "attention_mask": [1] * len(tokens),
            "labels": [IGNORE_INDEX] * start + list(tokens[start:])}


def collate(samples, tokenizer):
    """Right-pad model inputs; keep per-example weights and audit IDs outside model kwargs."""
    require(bool(samples), "Empty SFT batch")
    require(type(tokenizer.pad_token_id) is int, "Tokenizer pad ID is missing")
    encoded = [encode_sample(s, tokenizer) for s in samples]
    length = max(len(s["input_ids"]) for s in encoded)
    inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for sample in encoded:
        pad = length - len(sample["input_ids"])
        inputs["input_ids"].append(sample["input_ids"] + [tokenizer.pad_token_id] * pad)
        inputs["attention_mask"].append(sample["attention_mask"] + [0] * pad)
        inputs["labels"].append(sample["labels"] + [IGNORE_INDEX] * pad)
    return {
        "model_inputs": inputs,
        "sample_weights": [s["sample_weight"] for s in encoded],
        "metadata": [{k: s[k] for k in ("sample_id", "split", "answer_start")} for s in encoded],
    }
