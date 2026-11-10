"""Local Qwen generation and bounded prefix KV reuse."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import json
import time

from agent_simulation.data.common import require

from agent_simulation.inference.requests import load_tokenizer, prompt_tokens, read_tokens, render_request
from agent_simulation.inference import resolve_device
from agent_simulation.agents.schemas import integer, text
from agent_simulation.logs import ModelLog
from agent_simulation.inference.client import ModelOutputError, finish_reply, output_schema

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


class QwenLocal:
    role_adapters = ()

    def __init__(
        self,
        model_path=DEFAULT_MODEL,
        *,
        revision=DEFAULT_REVISION,
        max_new_tokens=2048,
        device="auto",
        dtype="auto",
        log_path=None,
        role_adapters=None,
        checkpoint=None,
        prefix_cache_tokens=2048,
        prefix_cache_entries=4,
    ):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        text(revision)
        integer(max_new_tokens, 1)
        device = str(resolve_device(device))
        if dtype not in ("auto", "float32", "float16", "bfloat16"):
            raise ValueError("Unsupported inference dtype")
        self.log = ModelLog(log_path) if log_path is not None else None
        self.tokenizer = load_tokenizer(model_path, revision=revision, local_files_only=True)
        config = AutoConfig.from_pretrained(model_path, revision=revision, local_files_only=True,
                                            trust_remote_code=False)
        model_class = AutoModelForCausalLM
        if config.model_type == "qwen3_5":
            from transformers import Qwen3_5ForCausalLM
            model_class = Qwen3_5ForCausalLM
            config = config.get_text_config()
        if config.model_type not in {"qwen2", "qwen3", "qwen3_5_text"}:
            raise ValueError("Local generation supports Qwen2.5, Qwen3 and Qwen3.5 text checkpoints")
        self.model = model_class.from_pretrained(
            model_path,
            config=config,
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype if dtype == "auto" else getattr(torch, dtype),
            attn_implementation="sdpa",
        ).to(device)
        if config.model_type == "qwen3_5_text":
            generation = self.model.generation_config
            eos = generation.eos_token_id
            stop = [eos] if isinstance(eos, int) else (eos or [])
            generation.eos_token_id = list(dict.fromkeys([self.tokenizer.eos_token_id, *stop]))
            generation.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.model.requires_grad_(False)
        self.generator_id = (
            f"{model_path}@{revision}:greedy:max_new_tokens={max_new_tokens}"
            f":device={device}:dtype={self.model.dtype}"
        )
        self.max_new_tokens, self.device, self._torch = max_new_tokens, device, torch
        self._grammar_tokenizer = None
        require(type(prefix_cache_tokens) is int and prefix_cache_tokens >= 0
                and type(prefix_cache_entries) is int and prefix_cache_entries >= 0,
                'Expected nonnegative prefix cache limits')
        # The shared-prefix implementation has not been verified for recurrent Qwen3.5 cache states.
        entries = 0 if config.model_type == "qwen3_5_text" else prefix_cache_entries
        self.prefix_cache = PrefixCache(prefix_cache_tokens, entries)
        self.role_adapters = tuple(role_adapters or ())
        if role_adapters:
            from agent_simulation.inference.local.checkpoints import attach_adapter
            for role, directory in role_adapters.items():
                attach_adapter(self, directory, name=role, expected_role=role, checkpoint=checkpoint)

    def _adapter_args(self, requests):
        names = [r.get("agent_role") for r in requests]
        require(all(n in (None, "customer", "seller") for n in names), "Unknown inference agent role")
        if not self.role_adapters:
            return {}
        return {"adapter_names": [n if n in self.role_adapters else "__base__" for n in names]}

    def _decoding(self, request):
        """Build the requested output grammar, caching only tokenizer metadata."""
        schema = output_schema(request)
        if schema is None:
            return {}
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

        if self._grammar_tokenizer is None:
            self._grammar_tokenizer = build_token_enforcer_tokenizer_data(self.tokenizer)
        parser = action_parser(schema, self.max_new_tokens, self._grammar_tokenizer.tokenizer_alphabet)
        return {"prefix_allowed_tokens_fn": build_prefix_allowed_tokens_fn(self._grammar_tokenizer, parser)}

    def _inputs(self, requests):
        if not any('_prompt_tokens' in request for request in requests):
            prompts = [render_request(self.tokenizer, request) for request in requests]
            return self.tokenizer(prompts, padding=True, padding_side='left',
                                  return_tensors='pt', add_special_tokens=False).to(self.device)
        ids = [read_tokens(request, self.tokenizer) if '_prompt_tokens' in request else
               prompt_tokens(request['messages'], request['tools'], self.tokenizer) for request in requests]
        from transformers import BatchEncoding
        width = max(map(len, ids))
        return BatchEncoding({
            'input_ids': self._torch.tensor([[self.tokenizer.pad_token_id] * (width-len(row)) + row
                                            for row in ids], device=self.device),
            'attention_mask': self._torch.tensor([[0] * (width-len(row)) + [1] * len(row)
                                                 for row in ids], device=self.device),
        })

    def _reuse_prefix(self, inputs, requests):
        cache = getattr(self, 'prefix_cache', None)
        return cache.prepare(self, inputs, requests) if cache is not None else (inputs, {})

    def complete(self, request, *, call_id):
        started = time.monotonic()
        inputs = self._inputs([request])
        if (
            inputs.input_ids.shape[1] + self.max_new_tokens
            > self.model.config.max_position_embeddings
        ):
            raise ValueError("Context exceeds checkpoint capacity; no silent history truncation")
        inputs, cached = self._reuse_prefix(inputs, [request])
        decoding = self._decoding(request)
        with self._torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_new_tokens, **decoding, **cached,
                **self._adapter_args([request])
            )
        return self._reply(request, call_id, output[0, inputs.input_ids.shape[1] :],
                           inputs.input_ids.shape[1], started)

    def complete_batch(self, jobs):
        """One left-padded greedy generate, with independent grammars and EOS per row."""
        started = time.monotonic()
        requests = [request for _, request, _ in jobs]
        inputs = self._inputs(requests)
        width = inputs.input_ids.shape[1]
        require(width + self.max_new_tokens <= self.model.config.max_position_embeddings,
                "Context exceeds checkpoint capacity")
        inputs, cached = self._reuse_prefix(inputs, requests)
        grammars = [self._decoding(request).get("prefix_allowed_tokens_fn") for _, request, _ in jobs]
        unrestricted = list(range(self.model.config.vocab_size)) if None in grammars else None
        eos = self.model.generation_config.eos_token_id
        stop = {eos} if isinstance(eos, int) else set(eos)
        finished_tokens = stop | {self.tokenizer.pad_token_id}

        def allowed(row, tokens):
            if time.monotonic() >= jobs[row][2] or (
                len(tokens) > width and int(tokens[-1]) in finished_tokens
            ):
                return [min(stop)]
            grammar = grammars[row]
            return unrestricted if grammar is None else grammar(0, tokens)

        with self._torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                                         prefix_allowed_tokens_fn=allowed, **cached,
                                         **self._adapter_args(requests))
        results = []
        for row, (call_id, request, deadline) in enumerate(jobs):
            if time.monotonic() >= deadline:
                results.append(TimeoutError(f"Inference deadline exceeded: {call_id}"))
                continue
            tokens = output[row, width:].tolist()
            end = next((i + 1 for i, token in enumerate(tokens) if token in stop), len(tokens))
            try:
                results.append(self._reply(request, call_id, tokens[:end],
                                           int(inputs.attention_mask[row].sum()), started))
            except ModelOutputError as error:
                results.append(error)
        return results

    def _reply(self, request, call_id, generated, input_length, started):
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        finished = bool(len(generated) and eos and int(generated[-1]) in eos)
        # Persist unedited output before parsing, including malformed or truncated answers.
        if self.log is not None:
            self.log.append(
                {
                    "call_id": call_id,
                    "model": self.generator_id,
                    **({"active_adapter": self._adapter_args([request])["adapter_names"][0]}
                       if self.role_adapters else {}),
                    "request": request,
                    "raw_output": raw,
                    "finished": finished,
                    "input_tokens": input_length,
                    "output_tokens": len(generated),
                    "seconds": time.monotonic() - started,
                }
            )
        return finish_reply(raw, finished=finished, request=request)


class PrefixCache:
    """Reuse exact prefix tokens for one model, with separate entries per adapter."""

    def __init__(self, max_tokens=2048, max_entries=4, min_tokens=32):
        self.max_tokens, self.max_entries, self.min_tokens = max_tokens, max_entries, min_tokens
        self.entries = OrderedDict()
        self.hits = self.misses = self.reused_tokens = 0

    def prepare(self, backend, inputs, requests):
        import torch
        from transformers import BatchEncoding

        if not self.max_tokens or not self.max_entries:
            return inputs, {}
        adapters = backend._adapter_args(requests).get('adapter_names', ['__base__'] * len(requests))
        if len(set(adapters)) != 1:
            return inputs, {}  # Different adapter weights cannot share a prefill.
        rows = [row[mask.bool()].tolist() for row, mask in zip(inputs.input_ids, inputs.attention_mask)]
        length = min(self.max_tokens, min(map(len, rows)) - 1)
        for i in range(length):
            if any(row[i] != rows[0][i] for row in rows[1:]):
                length = i
                break
        if length < self.min_tokens:
            return inputs, {}
        matching = [key for key in self.entries if key[0] == adapters[0]
                    and len(key[1]) <= length and tuple(rows[0][:len(key[1])]) == key[1]]
        if matching:
            key = max(matching, key=lambda key: len(key[1]))
            length = len(key[1])
            self.hits += 1
            self.reused_tokens += length * len(rows)
        else:
            key = adapters[0], tuple(rows[0][:length])
            prefix = torch.tensor([key[1]], device=backend.device)
            with torch.inference_mode():
                output = backend.model(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                                       use_cache=True, logits_to_keep=1,
                                       **backend._adapter_args([requests[0]]))
            self.entries[key] = output.past_key_values
            self.misses += 1
            self.reused_tokens += length * (len(rows) - 1)
            if len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
        self.entries.move_to_end(key)
        # generate() extends its cache in place; stored prefixes remain immutable.
        cache = deepcopy(self.entries[key])
        cache.batch_repeat_interleave(len(rows))
        width = max(map(len, rows))
        padded, masks = [], []
        for row in rows:
            pad = width - len(row)
            padded.append(row[:length] + [backend.tokenizer.pad_token_id] * pad + row[length:])
            masks.append([1] * length + [0] * pad + [1] * (len(row) - length))
        return BatchEncoding(dict(input_ids=torch.tensor(padded, device=backend.device),
                                  attention_mask=torch.tensor(masks, device=backend.device))), {
                                      'past_key_values': cache}


@lru_cache(maxsize=2)
def _string_boundary_tokens(tokenizer_data):
    """Tokens crossing a closing quote are omitted by LMFE's free-string cache."""
    from lmformatenforcer.tokenizerprefixtree import TokenizerPrefixTreeNode

    cache = tokenizer_data.tokenizer_tree.json_freetext_tokens
    cached = set(cache.regular_tokens_length_cache.tokens) | set(cache.quote_tokens_length_cache.tokens)
    root = TokenizerPrefixTreeNode()
    for token, text, _ in tokenizer_data.regular_tokens:
        if '"' in text[1:-1] and token not in cached:
            node = root
            for character in text:
                node = node.children.setdefault(character, TokenizerPrefixTreeNode())
            node.tokens.append(token)
    return root


def build_prefix_allowed_tokens_fn(tokenizer_data, parser):
    """Keep cached free text fast, but grammar-check tokens such as .\", too."""
    from lmformatenforcer import TokenEnforcer
    from lmformatenforcer.integrations.transformers import TransformersPrefixAllowedTokensFn

    boundaries = _string_boundary_tokens(tokenizer_data)

    class CompleteStringTokens(TokenEnforcer):
        def _collect_allowed_tokens(self, parser, node, allowed, shortcut):
            super()._collect_allowed_tokens(parser, node, allowed, shortcut)
            if isinstance(shortcut, tuple) and shortcut[0] == 'json_freetext':
                super()._collect_allowed_tokens(parser, boundaries, allowed, None)

    config = getattr(parser, "config", None)
    enforcer = CompleteStringTokens(tokenizer_data, parser)
    if config is not None:
        # LMFE resets the root parser config during construction, including its array limit.
        parser.config = replace(config, alphabet=tokenizer_data.tokenizer_alphabet)
    return TransformersPrefixAllowedTokensFn(enforcer)


def action_parser(schema, max_output_tokens, alphabet):
    """Constrain JSON and allow arrays larger than LMFE's default of 20."""
    from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser, StringParser

    config = CharacterLevelParserConfig(max_json_array_length=max_output_tokens, alphabet=alphabet)
    # LMFE cannot enforce empty arrays through maxItems=0; bind empty responses literally.
    if set(schema) == {"const"}:
        parser = StringParser(json.dumps(schema["const"], separators=(",", ":")))
        parser.config = config
        return parser
    return JsonSchemaParser(schema, config=config)
