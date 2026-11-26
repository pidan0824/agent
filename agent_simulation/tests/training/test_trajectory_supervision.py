"""Regression checks for component masks, request mass and actual gradients."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import torch
from agent_simulation.datasets.trajectory import encode_record, read_training_records, COMPONENTS, IGNORE_INDEX
from agent_simulation.training.trajectory import collate_trajectories, trajectory_token_losses, training_batch_loss
from agent_simulation.tests.training.test_trajectory_training import CharacterChatTokenizer, ContextModel, PROMPT


class OffsetTokenizer(CharacterChatTokenizer):
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) + 1 for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def sample(request=1, scene="one", identity="a"):
    return {"role": "customer", "actor_id": 7, "stage": "response", "source_split": "train",
            "source_request_id": request, "scene_id": scene, "initial_state_id": scene,
            "trajectory_id": identity, "tools": [], "messages": PROMPT + [{"role": "assistant",
            "content": json.dumps({"rationale": {"note": 'the word "action" is not a field here'},
                                  "action": {"clicked_item_ids": [1], "purchased_item_ids": []}})}]}


class SupervisionTests(unittest.TestCase):
    def test_value_spans_handle_quoted_keys_and_do_not_supervise_context(self):
        row = sample()
        encoded = encode_record(row, "sft", OffsetTokenizer())["trajectory"]
        turn = encoded.turns[-1]
        text = lambda group: "".join(chr(t - 1) for t, c in zip(turn.input_ids, turn.components) if c == group)
        self.assertEqual(text(COMPONENTS["action"]), "[1][]")
        self.assertIn("clicked_item_ids",text(COMPONENTS["format"]))
        self.assertNotIn("clicked_item_ids",text(COMPONENTS["action"]))
        self.assertIn("not a field", text(COMPONENTS["rationale"]))
        self.assertTrue(all(c == 0 for c, label in zip(turn.components, turn.labels) if label == IGNORE_INDEX))
        self.assertIn("</turn>", text(COMPONENTS["format"]))

    def test_separate_loss_equals_manual_components_and_weights_survive_single_row_batches(self):
        torch.manual_seed(7)
        encoded = encode_record(sample(), "sft", OffsetTokenizer())["trajectory"]
        model = ContextModel()
        batch = collate_trajectories(encoded.turns, 0, "cpu")
        losses, targets, groups = trajectory_token_losses(model, batch)
        expected = sum((losses[groups == c]).mean() * weight for c, weight in ((1, 1.), (2, .3), (4, .2)))
        row = {"encoded": {"trajectory": encoded}, "weight": .4}
        measured, metrics = training_batch_loss(model, [row], 0, "cpu", "sft", rationale_weight=.3, format_weight=.2)
        self.assertTrue(torch.allclose(measured, expected * .4))
        measured.backward()
        self.assertGreater(model.output.weight.grad.abs().sum().item(), 0)
        self.assertGreater(metrics["action_tokens"], 0)
        self.assertGreater(metrics["rationale_tokens"], 0)
        self.assertEqual(metrics["tool_tokens"], 0)
        row["weight"] = .8
        doubled, _ = training_batch_loss(model, [row], 0, "cpu", "sft", rationale_weight=.3, format_weight=.2)
        self.assertTrue(torch.allclose(doubled, measured * 2))

    def test_original_request_and_sibling_scene_mass_are_balanced(self):
        rows = [sample(1, "x", "a"), sample(1, "x", "b"), sample(1, "y", "c"), sample(2, "z", "d")]
        with TemporaryDirectory() as d:
            p = Path(d) / "sft.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            records = read_training_records(p, "sft", OffsetTokenizer())
        weights = [r["weight"] for r in records]
        self.assertEqual(weights, [.5, .5, 1., 2.])
        self.assertEqual(sum(weights), len(weights))

    def test_positive_request_weights_preserve_sibling_mass_and_keep_negatives(self):
        rows=[sample(1,"x","a"),sample(1,"x","b"),sample(2,"y","c"),sample(3,"z","d")]
        rows[2]["messages"][-1]["content"]=json.dumps({"clicked_item_ids":[],"purchased_item_ids":[2]})
        rows[3]["messages"][-1]["content"]=json.dumps({"clicked_item_ids":[],"purchased_item_ids":[]})
        with TemporaryDirectory() as d:
            p=Path(d)/"sft.jsonl"
            p.write_text("".join(json.dumps(r)+"\n" for r in rows))
            records=read_training_records(p,"sft",OffsetTokenizer(),click_weight=2,purchase_weight=4)
        weights=[r["weight"] for r in records]
        self.assertAlmostEqual(weights[0]+weights[1],8/3)
        self.assertAlmostEqual(weights[2],16/3)
        self.assertAlmostEqual(weights[3],4/3)

    def test_response_decisions_mark_first_value_or_close_instead_of_opening_bracket(self):
        for wrapped in (False, True):
            for click, purchase in (([], []), ([123], []), ([], [456]), ([123, 7], [456])):
                row = sample()
                action = {"clicked_item_ids": click, "purchased_item_ids": purchase}
                row["messages"][-1]["content"] = json.dumps(
                    {"rationale": {"note": "unknown"}, "action": action} if wrapped else action, indent=2)
                encoded = encode_record(row, "sft", OffsetTokenizer(), response_decisions=True)["trajectory"]
                turn = encoded.turns[-1]
                for code, expected in ((5, "1" if click else "]"), (6, "4" if purchase else "]")):
                    actual = [chr(t - 1) for t, c in zip(turn.input_ids, turn.components) if c == code]
                    self.assertEqual(actual, [expected])
                self.assertTrue(all(c == 0 for c, label in zip(turn.components, turn.labels)
                                    if label == IGNORE_INDEX))

    def test_response_auxiliary_preserves_action_loss_and_adds_real_decision_gradients(self):
        torch.manual_seed(9)
        row = sample()
        old = encode_record(row, "sft", OffsetTokenizer())["trajectory"]
        marked = encode_record(row, "sft", OffsetTokenizer(), response_decisions=True)["trajectory"]
        self.assertEqual(old.turns[-1].input_ids, marked.turns[-1].input_ids)
        self.assertEqual(old.turns[-1].labels, marked.turns[-1].labels)
        model = ContextModel()
        records = [{"encoded": {"trajectory": marked}, "metadata": {"stage": "response"}}]
        base, _ = training_batch_loss(model, records, 0, "cpu", "sft")
        previous, _ = training_batch_loss(model, [{"encoded": {"trajectory": old}}], 0, "cpu", "sft")
        self.assertTrue(torch.allclose(base, previous))
        enhanced, metrics = training_batch_loss(model, records, 0, "cpu", "sft", response_decision_weight=2)
        batch = collate_trajectories(marked.turns, 0, "cpu")
        losses, _, groups = trajectory_token_losses(model, batch)
        manual = losses[groups == 5].sum() + losses[groups == 6].sum()
        self.assertTrue(torch.allclose(enhanced - base, manual))
        actual_grad = torch.autograd.grad(enhanced - base, model.output.weight, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(manual, model.output.weight)[0]
        self.assertTrue(torch.allclose(actual_grad, expected_grad, atol=1e-6))
        self.assertGreater(actual_grad.abs().sum().item(), 0)
        self.assertEqual(metrics["click_decision_tokens"], 1)
        self.assertEqual(metrics["purchase_decision_tokens"], 1)
        with self.assertRaises(ValueError):
            training_batch_loss(model, [{"encoded": {"trajectory": old}, "metadata": {"stage": "response"}}],
                                0, "cpu", "sft", response_decision_weight=1)

    def test_direct_action_has_no_rationale_loss(self):
        row = sample()
        row["messages"][-1]["content"] = json.dumps({"clicked_item_ids": [], "purchased_item_ids": []})
        encoded = encode_record(row, "sft", OffsetTokenizer())["trajectory"]
        self.assertNotIn(COMPONENTS["rationale"], encoded.turns[-1].components)
        loss, metrics = training_batch_loss(ContextModel(), [{"encoded": {"trajectory": encoded}}], 0, "cpu", "sft")
        self.assertEqual(metrics["rationale_tokens"], 0)
        self.assertEqual(metrics["rationale_nll"], 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
