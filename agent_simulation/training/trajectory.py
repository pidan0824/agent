"""Train a local LoRA policy on tool-trajectory SFT or assistant-only DPO.

This separate entry point never changes legacy final-answer SFT. It requires a
new output directory: restart from an exported adapter in a new run, rather
than silently resuming optimizer state. DPO requires the SFT adapter as its
initial policy and computes its frozen reference scores before any updates.
"""

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from agent_simulation.data.common import require, sha256
from agent_simulation.datasets.trajectory import FORMAT, IGNORE_INDEX, read_training_records
from agent_simulation.inference import resolve_device
from agent_simulation.inference.requests import load_tokenizer, tokenizer_identity
from agent_simulation.logs import write_report


def collate_trajectories(rows, pad_token_id, device):
    require(rows and type(pad_token_id) is int, 'Missing rows or padding token')
    length = max(len(row.input_ids) for row in rows)
    ids = torch.full((len(rows), length), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, IGNORE_INDEX)
    for index, row in enumerate(rows):
        size = len(row.input_ids)
        ids[index, :size] = torch.tensor(row.input_ids, device=device)
        attention[index, :size] = 1
        labels[index, :size] = torch.tensor(row.labels, device=device)
    return {'input_ids': ids, 'attention_mask': attention, 'labels': labels}


def trajectory_log_probs(model, batch):
    """Sum log p of a decision's assistant tokens under its actual observations."""
    labels = batch['labels']
    positions = labels[:, 1:].ne(IGNORE_INDEX).any(dim=0).nonzero().flatten()
    require(positions.numel() > 0, 'No assistant prediction positions')
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    use_cache=False, logits_to_keep=positions)
    targets = labels[:, positions + 1]
    counts = targets.ne(IGNORE_INDEX).sum(-1)
    require(bool(torch.all(counts > 0)), 'Empty assistant supervision in a branch')
    losses = F.cross_entropy(outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
                             targets.reshape(-1), ignore_index=IGNORE_INDEX,
                             reduction='none').reshape_as(targets)
    return -losses.sum(-1), counts


def complete_trajectory_log_probs(model, trajectories, pad_token_id, device):
    """Sum turn logprobs without supervising earlier assistant context again.

    Process one decision depth at a time, keeping the configured batch size;
    do not turn one six-step trajectory into a six-times-larger padded batch.
    """
    values, counts = [[] for _ in trajectories], [[] for _ in trajectories]
    for depth in range(max(len(trajectory.turns) for trajectory in trajectories)):
        indices = [index for index, trajectory in enumerate(trajectories) if len(trajectory.turns) > depth]
        batch = collate_trajectories([trajectories[index].turns[depth] for index in indices],
                                     pad_token_id, device)
        turn_values, turn_counts = trajectory_log_probs(model, batch)
        for position, index in enumerate(indices):
            values[index].append(turn_values[position])
            counts[index].append(turn_counts[position])
    return (torch.stack([torch.stack(turns).sum() for turns in values]),
            torch.stack([torch.stack(turns).sum() for turns in counts]))


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta=0.1):
    """Full trajectory DPO, with system/user/tool terms absent from both policies."""
    require(math.isfinite(beta) and beta > 0, 'DPO beta must be finite and positive')
    margin = beta * ((chosen - rejected) - (reference_chosen - reference_rejected))
    return -F.logsigmoid(margin), margin


@contextmanager
def frozen_reference(model):
    """Temporarily freeze the starting SFT policy while caching fixed reference values."""
    training = model.training
    flags = [parameter.requires_grad for parameter in model.parameters()]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        with torch.no_grad():
            yield
    finally:
        for parameter, flag in zip(model.parameters(), flags, strict=True):
            parameter.requires_grad_(flag)
        model.train(training)


def reference_scores(model, records, pad_token_id, device, *, batch_size=1):
    """Cache the frozen SFT reference once; no second 9B base model is necessary."""
    scores = []
    with frozen_reference(model):
        for start in range(0, len(records), batch_size):
            group = records[start:start + batch_size]
            by_branch = {}
            for branch in ('chosen', 'rejected'):
                values, counts = complete_trajectory_log_probs(
                    model, [row['encoded'][branch] for row in group], pad_token_id, device)
                by_branch[branch] = list(zip(values.cpu().tolist(), counts.cpu().tolist()))
            for index, row in enumerate(group):
                scores.append({**row['metadata'], **{
                    branch: by_branch[branch][index][0] for branch in ('chosen', 'rejected')},
                    'assistant_tokens': {branch: by_branch[branch][index][1]
                                         for branch in ('chosen', 'rejected')}})
    require(all(math.isfinite(row[branch]) for row in scores for branch in ('chosen', 'rejected')),
            'Non-finite frozen reference score')
    return scores


def training_batch_loss(model, records, pad_token_id, device, mode, *, reference=None, beta=0.1):
    if mode == 'sft':
        values, counts = complete_trajectory_log_probs(
            model, [row['encoded']['trajectory'] for row in records], pad_token_id, device)
        loss = -(values / counts).mean()
        return loss, {'assistant_nll': float(loss.detach())}
    require(reference is not None and len(reference) == len(records), 'Missing fixed DPO reference')
    values = {}
    for branch in ('chosen', 'rejected'):
        values[branch], _ = complete_trajectory_log_probs(
            model, [row['encoded'][branch] for row in records], pad_token_id, device)
    refs = {branch: torch.tensor([row[branch] for row in reference], device=device,
                                dtype=values[branch].dtype) for branch in ('chosen', 'rejected')}
    losses, margins = dpo_loss(values['chosen'], values['rejected'], refs['chosen'], refs['rejected'], beta)
    return losses.mean(), {'dpo_margin': float(margins.detach().mean()),
                           'preference_accuracy': float((margins.detach() > 0).float().mean())}


def adapter_digest(directory):
    files = sorted(path for path in Path(directory).iterdir()
                   if path.is_file() and (path.name == 'adapter_config.json' or path.name.startswith('adapter_model')))
    require(any(path.name == 'adapter_config.json' for path in files)
            and any(path.name.startswith('adapter_model') for path in files), 'Incomplete initial LoRA adapter')
    return {path.name: sha256(path) for path in files}


def base_weight_digest(directory):
    index = directory / 'model.safetensors.index.json'
    names = (sorted(set(json.loads(index.read_text())['weight_map'].values()))
             if index.is_file() else ['model.safetensors'])
    require(names and all(Path(name).name == name and (directory / name).is_file() for name in names),
            'Complete local base weights are required')
    return {name: sha256(directory / name) for name in names}


def save_checkpoint(model, tokenizer, optimizer, scheduler, output, step, state):
    destination = output / f'checkpoint-{step}'
    require(not destination.exists(), 'Checkpoint already exists')
    with TemporaryDirectory(dir=output, prefix='.checkpoint-') as temporary:
        staged = Path(temporary) / 'ready'
        staged.mkdir()
        model.save_pretrained(staged, safe_serialization=True)
        tokenizer.save_pretrained(staged)
        torch.save(optimizer.state_dict(), staged / 'optimizer.pt')
        torch.save(scheduler.state_dict(), staged / 'scheduler.pt')
        rng = {'torch': torch.get_rng_state(), 'python': random.getstate()}
        if torch.cuda.is_available():
            rng['cuda'] = torch.cuda.get_rng_state_all()
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            rng['mps'] = torch.mps.get_rng_state()
        torch.save(rng, staged / 'rng_state.pt')
        write_report(staged / 'state.json', {**state, 'global_step': step,
                                            'restart_policy': 'new_run_from_adapter; optimizer resume not implemented'})
        staged.rename(destination)
    return destination


def train(args):
    require(not args.output.exists(), 'Use a new output directory; trajectory runs do not silently resume')
    require(args.mode != 'dpo' or args.init_adapter is not None, 'DPO requires --init-adapter from SFT')
    require(min(args.epochs, args.batch_size, args.accumulation, args.rank, args.save_steps) > 0,
            'Epochs, batch sizes, rank and save interval must be positive')
    require(args.max_steps is None or args.max_steps > 0, 'max-steps must be positive')
    require(math.isfinite(args.learning_rate) and args.learning_rate > 0, 'Invalid learning rate')
    require(math.isfinite(args.beta) and args.beta > 0, 'Invalid DPO beta')
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import set_seed
    from agent_simulation.training.run import load_base_model

    device = resolve_device(args.device)
    require(not (device.type == 'cuda' and torch.cuda.device_count() > 1),
            'Select one GPU with CUDA_VISIBLE_DEVICES for trajectory training')
    args.device = device.type
    if args.dtype == 'auto':
        args.dtype = 'bfloat16' if device.type == 'cuda' and torch.cuda.is_bf16_supported() else 'float32'
    tokenizer = load_tokenizer(str(args.model), local_files_only=True)
    require(tokenizer.pad_token_id is not None, 'Model tokenizer has no padding token')
    records = read_training_records(args.input, args.mode, tokenizer,
                                    max_length=args.max_length, limit=args.limit)
    init_identity = adapter_digest(args.init_adapter) if args.init_adapter else None
    identity = {'format': FORMAT, 'mode': args.mode, 'role': records[0]['metadata']['role'],
                'input_sha256': sha256(args.input), 'model': str(args.model.resolve()),
                'base_config_sha256': sha256(args.model / 'config.json'),
                'base_weight_sha256': base_weight_digest(args.model),
                'tokenizer_sha256': tokenizer_identity(tokenizer, tokenizer.chat_template),
                'initial_adapter_sha256': init_identity, 'records': len(records),
                'mask': 'generated assistant content/tool calls/end tokens only; observations excluded',
                'dpo_logprob': 'sum over all assistant tokens of each complete actual branch',
                'turn_encoding': 'official template per decision prefix; earlier assistant turns context-only',
                'source_split': 'train', 'restart_policy': 'new output required; no optimizer resume'}
    if args.init_adapter:
        from agent_simulation.inference.local.checkpoints import adapter_training_identity
        source = adapter_training_identity(args.init_adapter, identity['role'])
        require(all(source.get(key) == identity[key] for key in ('base_config_sha256', 'base_weight_sha256')),
                'Initial SFT adapter base weights differ from the training model')
        if args.mode == 'dpo':
            require(source.get('mode', 'sft') == 'sft', 'DPO reference must be an SFT policy')
    options = {key: str(value.resolve()) if isinstance(value, Path) else value
               for key, value in vars(args).items()}
    args.output.mkdir(parents=True)
    write_report(args.output / 'config.json', {'identity': identity, 'options': options})
    # Existing local agent/evaluation loaders verify role and base using this file.
    write_report(args.output / 'run_config.json', {'identity': identity, 'options': options})
    try:
        set_seed(args.seed)
        model = load_base_model(args)
        if args.init_adapter:
            adapter = json.loads((args.init_adapter / 'adapter_config.json').read_text())
            require(adapter.get('peft_type') == 'LORA', 'Initial adapter must be LoRA')
            model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True, local_files_only=True)
        else:
            targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
            if model.config.model_type == 'qwen3_5_text':
                targets.extend(['in_proj_qkv', 'out_proj'])
            model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=args.rank,
                lora_alpha=2 * args.rank, lora_dropout=0.0, bias='none', target_modules=targets))
        model.to(device)
        # DPO likelihood ratios require a deterministic policy/reference, including
        # adapters warm-started from legacy SFT with nonzero LoRA dropout.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        frozen = None
        if args.mode == 'dpo':
            frozen = reference_scores(model, records, tokenizer.pad_token_id, device, batch_size=args.batch_size)
            with (args.output / 'reference_scores.jsonl').open('x', encoding='utf-8') as stream:
                for row in frozen:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            write_report(args.output / 'reference.json', {
                'initial_adapter_sha256': init_identity,
                'input_sha256': identity['input_sha256'],
                'tokenizer_sha256': identity['tokenizer_sha256'],
                'turn_encoding': identity['turn_encoding'],
                'scores_sha256': sha256(args.output / 'reference_scores.jsonl'),
                'computed_before_any_optimizer_update': True, 'frozen': True,
                'same_assistant_mask': True, 'evaluation_mode': True})
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.enable_input_require_grads()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        require(bool(trainable), 'No trainable LoRA parameters')
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
        effective_batch = args.batch_size * args.accumulation
        steps = math.ceil(len(records) / effective_batch) * args.epochs
        if args.max_steps is not None:
            steps = min(steps, args.max_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1.0 - step / steps))
        global_step, last_saved = 0, 0
        last_metrics = {}
        model.train()
        with (args.output / 'metrics.jsonl').open('x', encoding='utf-8') as log:
            for epoch in range(args.epochs):
                order = list(range(len(records)))
                random.Random(args.seed + epoch).shuffle(order)
                for start in range(0, len(order), effective_batch):
                    selected = order[start:start + effective_batch]
                    optimizer.zero_grad(set_to_none=True)
                    aggregate = {'loss': 0.0}
                    for offset in range(0, len(selected), args.batch_size):
                        indices = selected[offset:offset + args.batch_size]
                        loss, details = training_batch_loss(model, [records[i] for i in indices],
                            tokenizer.pad_token_id, device, args.mode,
                            reference=[frozen[i] for i in indices] if frozen is not None else None,
                            beta=args.beta)
                        require(bool(torch.isfinite(loss)), 'Non-finite trajectory loss')
                        fraction = len(indices) / len(selected)
                        (loss * fraction).backward()
                        for key, value in {'loss': float(loss.detach()), **details}.items():
                            aggregate[key] = aggregate.get(key, 0.0) + fraction * value
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                    last_metrics = dict(global_step=global_step,
                        epoch=epoch + min(start + effective_batch, len(order)) / len(order),
                        learning_rate=scheduler.get_last_lr()[0], records=len(selected), **aggregate)
                    log.write(json.dumps(last_metrics, allow_nan=False) + '\n')
                    log.flush()
                    print(json.dumps(last_metrics, allow_nan=False), flush=True)
                    if global_step % args.save_steps == 0 or global_step == steps:
                        save_checkpoint(model, tokenizer, optimizer, scheduler, args.output, global_step,
                                        {'identity': identity, 'metrics': last_metrics})
                        last_saved = global_step
                    if global_step >= steps:
                        break
                if global_step >= steps:
                    break
        require(last_saved == global_step and global_step > 0, 'Missing final trajectory checkpoint')
        model.save_pretrained(args.output / 'adapter', safe_serialization=True)
        tokenizer.save_pretrained(args.output / 'adapter')
        result = {'status': 'completed', 'mode': args.mode, 'steps': global_step,
                  'checkpoint': str(args.output / f'checkpoint-{global_step}'),
                  'adapter': str(args.output / 'adapter'), 'metrics': last_metrics,
                  'reference_frozen': frozen is not None, 'test_used': False,
                  'behavior_evaluation_performed': False}
        write_report(args.output / 'result.json', result)
        return result
    except Exception as error:
        write_report(args.output / 'failure.json', {'status': 'incomplete', 'error': str(error)})
        raise


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('sft', 'dpo'), required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'models/Qwen3.5-9B')
    parser.add_argument('--init-adapter', type=Path, help='SFT warm start; required frozen SFT reference for DPO')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu', 'mps'), default='auto')
    parser.add_argument('--dtype', choices=('auto', 'float32', 'bfloat16'), default='auto')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-steps', type=int, help='Cap optimizer steps within the requested epochs')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--rank', type=int, default=64)
    parser.add_argument('--max-length', type=int, default=32768)
    parser.add_argument('--limit', type=int, default=0, help='Optional input-row cap for smoke runs')
    parser.add_argument('--save-steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == '__main__':
    train(parser().parse_args())
