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


def export_lora(source, output, *, model):
    """Copy a PEFT adapter for serving; Qwen3.5 keeps its language-model namespace."""
    import json
    from pathlib import Path
    from shutil import copyfile
    from tempfile import TemporaryDirectory
    from safetensors.torch import load_file, save_file

    source, output, model = Path(source), Path(output), Path(model)
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((model / 'config.json').read_text())
    weights = load_file(source / 'adapter_model.safetensors')
    if config.get('architectures') == ['Qwen3_5ForConditionalGeneration']:
        original = 'base_model.model.model.layers.'
        serving = 'base_model.model.model.language_model.layers.'
        if not weights or not all(key.startswith(original) for key in weights):
            raise ValueError('Expected a Qwen3.5 text LoRA checkpoint, not an exported serving copy')
        weights = {serving + key[len(original):]: value for key, value in weights.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.lora-') as temp:
        stage = Path(temp) / 'adapter'
        stage.mkdir()
        copyfile(source / 'adapter_config.json', stage / 'adapter_config.json')
        save_file(weights, stage / 'adapter_model.safetensors', metadata={'format': 'pt'})
        stage.rename(output)
    return output


def _generation_schema(value):
    # The vLLM grammar cannot express uniqueness. finish_reply checks the original schema.
    if isinstance(value, dict):
        return {key: _generation_schema(child) for key, child in value.items()
                if key != "uniqueItems" or not isinstance(child, bool)}
    if isinstance(value, list):
        return [_generation_schema(child) for child in value]
    return value


def structured_output(request):
    """Use vLLM's native JSON-schema constraint for decisions and reflections."""
    schema = output_schema(request)
    require(schema is not None, "vLLM requires a JSON output contract")
    return {"json": _generation_schema(schema)}


class VLLMClient:
    """One shared connection pool; async callers release their event loop during HTTP I/O."""

    def __init__(self, url, model, tokenizer, *, max_new_tokens=2048, timeout=120,
                 role_models=None, customer_stage_models=None, log_path=None):
        require(bool(model) and max_new_tokens > 0 and timeout > 0, 'Invalid vLLM settings')
        self.model, self.tokenizer, self.max_new_tokens = model, tokenizer, max_new_tokens
        self.role_models = dict(role_models or {})
        require(set(self.role_models) <= {'customer', 'seller'}, 'Unknown inference role')
        self.customer_stage_models = dict(customer_stage_models or {})
        require(set(self.customer_stage_models) <= {'query', 'response'}, 'Unknown Customer stage')
        self.generator_id = f'vllm:{model}:greedy:max_new_tokens={max_new_tokens}:roles={self.role_models}'
        if self.customer_stage_models:
            self.generator_id += f':customer_stages={self.customer_stage_models}'
        self.log = ModelLog(log_path) if log_path is not None else None
        key = os.environ.get('VLLM_API_KEY')
        self.http = httpx.Client(base_url=url.rstrip('/') + '/', timeout=timeout,
                                 headers={'Authorization': 'Bearer ' + key} if key else {})

    def check_models(self):
        """Check advertised model/LoRA names once at startup, without generating a sample."""
        response = self.http.get('models')
        response.raise_for_status()
        available = {entry['id'] for entry in response.json()['data']}
        require({self.model, *self.role_models.values(), *self.customer_stage_models.values()} <= available,
                'vLLM model is not served')
        return {'backend': 'vllm', 'model': self.model, 'role_models': self.role_models,
                'customer_stage_models': self.customer_stage_models, 'remote_weights_verified': False}

    def complete(self, request, *, call_id):
        started = time.monotonic()
        if '_prompt_tokens' not in request:
            prepared, _ = prepare_request(request['messages'], request['tools'], self.tokenizer,
                                          max_output_tokens=self.max_new_tokens)
            request = {**request, **prepared}
        role = request.get('agent_role')
        model = self.role_models.get(role, self.model)
        if role == 'customer':
            model = self.customer_stage_models.get(request.get('decision_stage'), model)
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
    parser.add_argument('--customer-query-served-model', help='Override the Customer model for Query')
    parser.add_argument('--customer-response-served-model', help='Override the Customer model for Response')
    parser.add_argument('--seller-served-model', help='Seller SFT model or LoRA name advertised by vLLM')


def server_client(args, tokenizer):
    model = 'qwen' if args.served_model is None else args.served_model
    return VLLMClient(args.vllm_url, model, tokenizer,
        max_new_tokens=args.max_new_tokens, timeout=args.request_timeout,
        role_models={role: name for role in ('customer', 'seller')
                     if (name := getattr(args, role + '_served_model'))},
        customer_stage_models={stage: name for stage in ('query', 'response')
                               if (name := getattr(args, 'customer_' + stage + '_served_model', None))},
        log_path=args.output / 'model_calls.jsonl')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Export a training LoRA for vLLM serving')
    parser.add_argument('--model', required=True, help='Base model directory served by vLLM')
    parser.add_argument('--adapter', required=True, help='Original PEFT checkpoint directory')
    parser.add_argument('--output', required=True, help='New serving-only adapter directory')
    args = parser.parse_args()
    print(export_lora(args.adapter, args.output, model=args.model))
