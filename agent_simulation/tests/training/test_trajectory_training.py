"""Small CPU checks for assistant masks, branch conditioning and DPO gradients."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from agent_simulation.datasets.trajectory import (
    IGNORE_INDEX, encode_record, encode_trajectory, read_training_records,
)
from agent_simulation.training.trajectory import (
    collate_trajectories, complete_trajectory_log_probs, dpo_loss, reference_scores,
    trajectory_log_probs, training_batch_loss,
)


TOOLS = [{'type': 'function', 'function': {'name': 'lookup', 'description': 'Read frozen facts',
          'parameters': {'type': 'object', 'properties': {'item': {'type': 'integer'}}}}}]
PROMPT = [{'role': 'system', 'content': 'Choose as this actor.'},
          {'role': 'user', 'content': 'Frozen market.'}]


def branch(item, observation, action):
    return [
        {'role': 'assistant', 'content': 'Check evidence.', 'tool_calls': [
            {'id': 'call-1', 'type': 'function', 'function': {
                'name': 'lookup', 'arguments': json.dumps({'item': item})}}]},
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': observation},
        {'role': 'user', 'content': 'One decision remains.'},
        {'role': 'assistant', 'content': json.dumps({
            'rationale': {'hypothesis': 'habit', 'evidence': [observation], 'uncertainty': 'limited'},
            'action': {'query': action}})},
    ]


def pair_row():
    return dict(role='customer', actor_id=7, stage='query', source_split='train', initial_state_id='state-1',
                chosen_trajectory_id='good', rejected_trajectory_id='bad', tools=TOOLS, prompt=PROMPT,
                chosen=branch(1, 'OBS_A', 'tea'), rejected=branch(2, 'OBS_B', 'coffee'))


class CharacterChatTokenizer:
    """An exact-prefix template with one token per ASCII character."""
    pad_token_id = 0

    def apply_chat_template(self, messages, tools, *, tokenize, add_generation_prompt, **kwargs):
        text = 'TOOLS:' + json.dumps(tools, sort_keys=True) + '\n'
        for message in messages:
            text += '<' + message['role'] + '>' + message['content']
            if message.get('tool_calls'):
                text += json.dumps(message['tool_calls'], sort_keys=True)
            text += '</turn>'
        if add_generation_prompt:
            text += '<assistant>'
        return [ord(char) + 1 for char in text] if tokenize else text


class ContextModel(torch.nn.Module):
    """Tiny causal model whose later predictions depend on actual tool observations."""
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(129, 8)
        self.output = torch.nn.Linear(8, 129)

    def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=None):
        embedded = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        # Keep the tool effects visible without long-fixture numerical explosion.
        states = embedded.cumsum(1) / attention_mask.cumsum(1).clamp_min(1).unsqueeze(-1).sqrt()
        return SimpleNamespace(logits=self.output(states[:, logits_to_keep]))


class TrajectoryTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.tokenizer = CharacterChatTokenizer()

    def test_mask_supervises_tool_calls_and_final_rationale_but_never_observations(self):
        messages = PROMPT + branch(1, 'OBS_A', 'tea')
        original = deepcopy(messages)
        encoded = encode_trajectory(messages, TOOLS, self.tokenizer)
        for turn in encoded.turns:
            text = ''.join(chr(token - 1) for token in turn.input_ids)
            for needle in ('Choose as this actor.', 'Frozen market.', '<tool>OBS_A</turn>',
                           '<user>One decision remains.</turn>'):
                if needle in text:
                    start = text.index(needle)
                    self.assertTrue(all(value == IGNORE_INDEX for value in turn.labels[start:start + len(needle)]))
        generated = ''.join(chr(token - 1) for turn in encoded.turns for token in turn.labels if token != IGNORE_INDEX)
        for needle in ('Check evidence.', '"name": "lookup"', '"hypothesis": "habit"', '"query": "tea"'):
            self.assertIn(needle, generated)
        self.assertEqual(generated.count('Check evidence.'), 1)
        self.assertEqual(messages, original)  # Arguments were normalized only in a rendering copy.

    def test_dpo_uses_each_branch_actual_tool_results_and_no_common_prompt_loss(self):
        row = pair_row()
        encoded = encode_record(row, 'dpo', self.tokenizer)
        model = ContextModel()
        actual, _ = complete_trajectory_log_probs(model, [encoded['chosen']], 0, 'cpu')
        different = deepcopy(row)
        different['chosen'][1]['content'] = 'CHANGED_OBSERVATION'
        # Keep the assistant generation identical; only its conditioning changes.
        revised = encode_record(different, 'dpo', self.tokenizer)
        counterfactual, _ = complete_trajectory_log_probs(model, [revised['chosen']], 0, 'cpu')
        self.assertFalse(torch.allclose(actual, counterfactual))
        self.assertNotEqual(encoded['chosen'].turns[-1].input_ids, encoded['rejected'].turns[-1].input_ids)
        common = self.tokenizer.apply_chat_template(PROMPT, TOOLS, tokenize=True, add_generation_prompt=True)
        self.assertTrue(all(value == IGNORE_INDEX for value in encoded['chosen'].turns[0].labels[:len(common)]))

    def test_dpo_gradient_prefers_chosen_and_frozen_reference_has_no_gradients(self):
        chosen = torch.tensor([-5.0], requires_grad=True)
        rejected = torch.tensor([-5.0], requires_grad=True)
        loss, _ = dpo_loss(chosen, rejected, torch.tensor([-5.0]), torch.tensor([-5.0]), beta=0.1)
        loss.sum().backward()
        self.assertLess(chosen.grad.item(), 0)
        self.assertGreater(rejected.grad.item(), 0)
        row = pair_row()
        record = {'encoded': encode_record(row, 'dpo', self.tokenizer),
                  'metadata': {key: row[key] for key in ('initial_state_id', 'chosen_trajectory_id', 'rejected_trajectory_id')}}
        model = ContextModel()
        model.embedding.weight.requires_grad_(False)
        flags = [parameter.requires_grad for parameter in model.parameters()]
        reference = reference_scores(model, [record], 0, 'cpu')
        frozen = deepcopy(reference)
        self.assertTrue(model.training)
        self.assertEqual([parameter.requires_grad for parameter in model.parameters()], flags)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        before, _ = training_batch_loss(model, [record], 0, 'cpu', 'dpo', reference=reference)
        before.backward()
        self.assertGreater(model.output.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        after, details = training_batch_loss(model, [record], 0, 'cpu', 'dpo', reference=reference)
        self.assertLess(after.item(), before.item())
        self.assertGreater(details['dpo_margin'], 0)
        self.assertEqual(reference, frozen)

    def test_sft_loss_and_gradients_match_assistant_only_manual_cross_entropy(self):
        encoded = encode_trajectory(PROMPT + branch(1, 'OBS_A', 'tea'), TOOLS, self.tokenizer)
        model = ContextModel()
        losses = []
        for turn in encoded.turns:
            batch = collate_trajectories([turn], 0, 'cpu')
            logits = model(batch['input_ids'], batch['attention_mask'], logits_to_keep=slice(None)).logits[:, :-1]
            losses.append(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                batch['labels'][:, 1:].reshape(-1), ignore_index=IGNORE_INDEX, reduction='sum'))
        manual = torch.stack(losses).sum() / encoded.assistant_tokens
        measured, _ = training_batch_loss(model, [{'encoded': {'trajectory': encoded}}], 0, 'cpu', 'sft')
        self.assertTrue(torch.allclose(manual, measured, atol=1e-6))
        measured.backward()
        self.assertGreater(model.output.weight.grad.abs().sum().item(), 0)

    def test_train_only_nonempty_release_and_broken_tool_pairs_fail(self):
        row = pair_row()
        row['source_split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'Train provenance'):
            encode_record(row, 'dpo', self.tokenizer)
        bad = PROMPT + branch(1, 'OBS_A', 'tea')
        bad[3]['tool_call_id'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'pending call'):
            encode_trajectory(bad, TOOLS, self.tokenizer)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'empty.jsonl'
            path.touch()
            with self.assertRaisesRegex(ValueError, 'Empty trajectory training release'):
                read_training_records(path, 'dpo', self.tokenizer)
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            encode_trajectory(PROMPT + branch(1, 'OBS_A', 'tea'), TOOLS, self.tokenizer, max_length=20)

    def test_tiny_qwen_sft_then_dpo_exports_loadable_adapters_and_complete_checkpoints(self):
        from tokenizers import Tokenizer, decoders, pre_tokenizers
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        from agent_simulation.tests.fixtures import tiny_base
        from agent_simulation.training.trajectory import parser, train
        from agent_simulation.inference.local.checkpoints import adapter_training_identity

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / 'base'
            vocab = {'[PAD]': 0, '[UNK]': 1, **{chr(number): number + 2 for number in range(128)}}
            backend = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
            backend.pre_tokenizer = pre_tokenizers.Split('', behavior='isolated')
            backend.decoder = decoders.Fuse()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]',
                chat_template="{% if tools %}TOOLS:{{tools|tojson}}{% endif %}"
                "{% for m in messages %}<{{m.role}}>{{m.content}}"
                "{% if m.tool_calls %}{{m.tool_calls|tojson}}{% endif %}</turn>{% endfor %}"
                "{% if add_generation_prompt %}<assistant>{% endif %}")
            tiny_base(len(vocab)).save_pretrained(base)
            tokenizer.save_pretrained(base)
            row = pair_row()
            sft = {key: value for key, value in row.items() if key not in
                   ('prompt', 'chosen', 'rejected', 'chosen_trajectory_id', 'rejected_trajectory_id')}
            sft.update(trajectory_id='good', messages=row['prompt'] + row['chosen'])
            for mode, record in (('sft', sft), ('dpo', row)):
                source = root / f'{mode}.jsonl'
                source.write_text(json.dumps(record) + '\n')
                argv = ['--mode', mode, '--input', str(source), '--output', str(root / mode),
                        '--model', str(base), '--device', 'cpu', '--rank', '2', '--accumulation', '1',
                        '--max-steps', '1', '--no-gradient-checkpointing']
                if mode == 'dpo':
                    argv += ['--init-adapter', str(root / 'sft/adapter')]
                args = parser().parse_args(argv)
                with redirect_stdout(io.StringIO()):
                    result = train(args)
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(result['steps'], 1)
                self.assertEqual(adapter_training_identity(root / mode / 'adapter', 'customer')['mode'], mode)
                checkpoint = root / mode / 'checkpoint-1'
                for name in ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt',
                             'scheduler.pt', 'rng_state.pt', 'state.json'):
                    self.assertGreater((checkpoint / name).stat().st_size, 0)
                with self.assertRaisesRegex(ValueError, 'new output directory'):
                    train(args)
            reference = json.loads((root / 'dpo/reference.json').read_text())
            self.assertTrue(reference['computed_before_any_optimizer_update'])
            self.assertTrue(reference['same_assistant_mask'])


if __name__ == '__main__':
    unittest.main()
