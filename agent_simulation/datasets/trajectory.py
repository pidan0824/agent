"""Encode observed tool trajectories with loss on assistant generations only.

SFT rows contain ``messages``. DPO rows contain a shared ``prompt`` plus
``chosen``/``rejected`` continuation message lists, each with its actual tool
results. Both formats retain ``tools`` and ``source_split='train'``.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

from agent_simulation.data.common import require
from agent_simulation.inference.requests import load_tokenizer, render_chat

IGNORE_INDEX = -100
FORMAT = 'assistant-tool-trajectory-v1'


def normalized_messages(messages, tools):
    """Check call/result correspondence; adapt OpenAI arguments to HF templates."""
    require(isinstance(messages, list) and bool(messages), 'Empty trajectory messages')
    require(isinstance(tools, list), 'Trajectory tools must be a list')
    tool_names = {tool['function']['name'] for tool in tools}
    result = deepcopy(messages)
    pending, seen = set(), set()
    for message in result:
        role = message.get('role')
        require(role in ('system', 'user', 'assistant', 'tool'), 'Unsupported trajectory role')
        content = message.get('content')
        if content is None and role == 'assistant' and message.get('tool_calls'):
            message['content'] = ''
        else:
            require(isinstance(content, str), 'Trajectory messages must contain text')
        if role == 'tool':
            call_id = message.get('tool_call_id')
            require(call_id in pending, 'Tool result does not match a pending call')
            pending.remove(call_id)
            require(not message.get('tool_calls'), 'Tool observation cannot call another tool')
            continue
        require(not pending, 'Missing tool results before the next model turn')
        calls = message.get('tool_calls') or []
        require(not calls or role == 'assistant', 'Only the assistant can call tools')
        require(isinstance(calls, list), 'tool_calls must be a list')
        for call in calls:
            require(call.get('type') == 'function', 'Only function tool calls are supported')
            call_id = call.get('id')
            require(isinstance(call_id, str) and call_id and call_id not in seen,
                    'Missing or repeated tool call ID')
            function = call.get('function', {})
            require(function.get('name') in tool_names, 'Tool call has no declared tool schema')
            arguments = function.get('arguments')
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            require(isinstance(arguments, dict), 'Tool arguments must encode an object')
            function['arguments'] = arguments
            pending.add(call_id)
            seen.add(call_id)
    require(not pending, 'Trajectory ends with unresolved tool calls')
    require(result[-1]['role'] == 'assistant' and not result[-1].get('tool_calls'),
            'Trajectory must end with the assistant final action')
    return result


@dataclass
class AssistantTurnTokens:
    input_ids: list
    labels: list

    @property
    def assistant_tokens(self):
        return sum(token != IGNORE_INDEX for token in self.labels)


@dataclass
class TrajectoryTokens:
    turns: list

    @property
    def assistant_tokens(self):
        return sum(turn.assistant_tokens for turn in self.turns)

    @property
    def max_tokens(self):
        return max(len(turn.input_ids) for turn in self.turns)


def encode_trajectory(messages, tools, tokenizer, *, prompt_messages=0, max_length=32768):
    """Encode each decision under the official template's actual historical prefix.

    Template-inserted assistant headers are conditioning, not generated actions.
    Qwen3.5 removes an earlier turn's empty thinking scaffold when a new user
    message arrives. Thus one static full-chat rendering cannot represent every
    decision context. Keep one encoding per decision, supervise only its newest
    assistant content and end-of-turn token, and sum those decision logprobs.
    """
    require(type(prompt_messages) is int and 0 <= prompt_messages < len(messages),
            'Invalid common prompt boundary')
    require(type(max_length) is int and max_length > 1, 'Invalid trajectory capacity')
    messages = normalized_messages(messages, tools)
    turns = []
    for index, message in enumerate(messages):
        if index < prompt_messages or message['role'] != 'assistant':
            continue
        prefix = list(render_chat(messages[:index], tools, tokenizer,
                                  tokenize=True, add_generation_prompt=True))
        through = list(render_chat(messages[:index + 1], tools, tokenizer,
                                   tokenize=True, add_generation_prompt=False))
        require(through[:len(prefix)] == prefix,
                'Chat template rewrites an assistant boundary; cannot safely mask this trajectory')
        require(len(through) > len(prefix), 'Assistant turn has no generated tokens')
        require(len(through) <= max_length,
                f'Trajectory exceeds context capacity ({len(through)} > {max_length}); no truncation')
        require(len(prefix) > 0, 'Assistant action needs a causal prompt')
        turns.append(AssistantTurnTokens(through, [IGNORE_INDEX] * len(prefix) + through[len(prefix):]))
    require(bool(turns), 'Trajectory has no causally supervised assistant tokens')
    return TrajectoryTokens(turns)


def encode_record(row, mode, tokenizer, *, max_length=32768):
    require(row.get('source_split', row.get('split')) == 'train',
            'Trajectory training requires explicit official Train provenance')
    require(row.get('role') in ('customer', 'seller'), 'Unknown trajectory actor role')
    for field in ('actor_id', 'initial_state_id', 'stage'):
        require(row.get(field) is not None, 'Missing trajectory identity: ' + field)
    tools = row.get('tools', [])
    if mode == 'sft':
        return {'trajectory': encode_trajectory(row['messages'], tools, tokenizer, max_length=max_length)}
    require(mode == 'dpo', 'Unknown trajectory training mode')
    prompt = row['prompt']
    require(isinstance(prompt, list) and prompt, 'DPO needs a common initial prompt')
    require(prompt[-1].get('role') in ('user', 'system'),
            'DPO prompt must end before the first assistant decision')
    encoded = {}
    for branch in ('chosen', 'rejected'):
        continuation = row[branch]
        require(isinstance(continuation, list) and continuation
                and continuation[0].get('role') == 'assistant',
                'DPO branches must be assistant-first continuations, not full conversations')
        encoded[branch] = encode_trajectory(prompt + continuation, tools, tokenizer,
                                            prompt_messages=len(prompt), max_length=max_length)
    require(row['chosen'] != row['rejected'], 'Identical DPO branches contain no preference signal')
    return encoded


def read_training_records(path, mode, tokenizer, *, max_length=32768, limit=0):
    """Read a bounded training release; corrupt/oversized rows fail explicitly."""
    require(type(limit) is int and limit >= 0, 'Invalid training row cap')
    records, identities = [], set()
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = (row.get('trajectory_id') if mode == 'sft' else
                        (row.get('initial_state_id'), row.get('chosen_trajectory_id'), row.get('rejected_trajectory_id')))
            require(identity is not None and (mode == 'sft' or all(part is not None for part in identity)),
                    f'Missing trajectory/pair IDs at line {line_number}')
            require(identity not in identities, f'Duplicate training example at line {line_number}')
            identities.add(identity)
            encoded = encode_record(row, mode, tokenizer, max_length=max_length)
            records.append({'encoded': encoded, 'metadata': {
                key: row[key] for key in ('role', 'actor_id', 'stage', 'initial_state_id', 'trajectory_id',
                                         'chosen_trajectory_id', 'rejected_trajectory_id') if key in row}})
            if limit and len(records) >= limit:
                break
    require(bool(records), 'Empty trajectory training release')
    require(len({row['metadata']['role'] for row in records}) == 1,
            'Train separate Customer and Seller adapters')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--mode', choices=('sft', 'dpo'), required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=32768)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    tokenizer = load_tokenizer(str(args.tokenizer), local_files_only=True)
    rows = read_training_records(args.input, args.mode, tokenizer,
                                 max_length=args.max_length, limit=args.limit)
    encoded = [value for row in rows for value in row['encoded'].values()]
    print(json.dumps(dict(format=FORMAT, mode=args.mode, records=len(rows),
                          max_tokens=max(value.max_tokens for value in encoded),
                          assistant_tokens=sum(value.assistant_tokens for value in encoded))))


if __name__ == '__main__':
    main()
