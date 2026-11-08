"""vLLM HTTP completions with the exact local chat template and action contracts."""

import asyncio
from copy import deepcopy
import os
import time

import httpx

from agent_simulation.data.common import require
from agent_simulation.inference.client import finish_reply, output_schema
from agent_simulation.inference.requests import prepare_request, read_tokens
from agent_simulation.logs import ModelLog


def structured_output(request):
    """Use vLLM's native JSON-schema constraint for decisions and reflections."""
    schema = output_schema(request)
    require(schema is not None, "vLLM requires a JSON output contract")
    return {"json": schema}


class VLLMClient:
    """One shared connection pool; async callers release their event loop during HTTP I/O."""

    def __init__(self, url, model, tokenizer, *, max_new_tokens=2048, timeout=120,
                 role_models=None, log_path=None):
        require(bool(model) and max_new_tokens > 0 and timeout > 0, 'Invalid vLLM settings')
        self.model, self.tokenizer, self.max_new_tokens = model, tokenizer, max_new_tokens
        self.role_models = dict(role_models or {})
        require(set(self.role_models) <= {'customer', 'seller'}, 'Unknown inference role')
        self.generator_id = f'vllm:{model}:greedy:max_new_tokens={max_new_tokens}:roles={self.role_models}'
        self.log = ModelLog(log_path) if log_path is not None else None
        key = os.environ.get('VLLM_API_KEY')
        self.http = httpx.Client(base_url=url.rstrip('/') + '/', timeout=timeout,
                                 headers={'Authorization': 'Bearer ' + key} if key else {})

    def check_models(self):
        """Check advertised model/LoRA names once at startup, without generating a sample."""
        response = self.http.get('models')
        response.raise_for_status()
        available = {entry['id'] for entry in response.json()['data']}
        require({self.model, *self.role_models.values()} <= available, 'vLLM model is not served')
        return {'backend': 'vllm', 'model': self.model, 'role_models': self.role_models,
                'remote_weights_verified': False}

    def complete(self, request, *, call_id):
        started = time.monotonic()
        if '_prompt_tokens' not in request:
            prepared, _ = prepare_request(request['messages'], request['tools'], self.tokenizer,
                                          max_output_tokens=self.max_new_tokens)
            request = {**request, **prepared}
        model = self.role_models.get(request.get('agent_role'), self.model)
        payload = dict(model=model, prompt=read_tokens(request, self.tokenizer),
                       max_tokens=self.max_new_tokens, temperature=0, n=1,
                       add_special_tokens=False, structured_outputs=structured_output(request))
        response = self.http.post('completions', json=payload)
        response.raise_for_status()
        data = response.json()
        choice, = data['choices']
        raw = choice['text']
        finished = choice['finish_reason'] == 'stop'
        if self.log is not None:
            self.log.append(dict(call_id=call_id, model=self.generator_id, served_model=model,
                request=deepcopy(request), raw_output=raw, finished=finished,
                input_tokens=data['usage']['prompt_tokens'], output_tokens=data['usage']['completion_tokens'],
                seconds=time.monotonic() - started))
        return finish_reply(raw, finished=finished, request=request)

    async def acomplete(self, request, *, call_id):
        return await asyncio.to_thread(self.complete, request, call_id=call_id)

    def close(self):
        self.http.close()


def add_server_arguments(parser):
    parser.add_argument('--inference-backend', choices=('transformers', 'vllm'), default='transformers')
    parser.add_argument('--vllm-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--served-model', help='Default vLLM model name for reflection and unbound roles (qwen)')
    parser.add_argument('--customer-served-model', help='Customer SFT model or LoRA name advertised by vLLM')
    parser.add_argument('--seller-served-model', help='Seller SFT model or LoRA name advertised by vLLM')


def server_client(args, tokenizer):
    model = 'qwen' if args.served_model is None else args.served_model
    return VLLMClient(args.vllm_url, model, tokenizer,
        max_new_tokens=args.max_new_tokens, timeout=args.request_timeout,
        role_models={role: name for role in ('customer', 'seller')
                     if (name := getattr(args, role + '_served_model'))},
        log_path=args.output / 'model_calls.jsonl')
