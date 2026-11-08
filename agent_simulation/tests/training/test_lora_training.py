"""Weighted loss, exact accumulation and adapter checkpoints on tiny Qwen fixtures."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import duckdb
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import TrainingArguments, TrainerCallback

from agent_simulation.tests.fixtures import tiny_base
from agent_simulation.datasets.labels import select_record_ids
from agent_simulation.training.data_loader import (
    SampleStream,
    EvaluationStream,
)
from agent_simulation.training.trainer import BehaviorTrainer, answer_losses, make_trainer
from agent_simulation.training.run import parser, resume_checkpoint, train


def tiny_model(vocab=32):
    return get_peft_model(
        tiny_base(vocab),
        LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0,
        ),
    )


def tensor_batch():
    return {
        "input_ids": torch.tensor([[1, 3, 4, 5], [1, 6, 7, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        "labels": torch.tensor([[-100, -100, 4, 5], [-100, -100, 7, -100]]),
        "sample_weights": torch.tensor([1.0, 0.25]),
    }


class LoRATrainingTests(unittest.TestCase):
    def test_official_protocol_uses_test_only_in_evaluation_stream(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from agent_simulation.datasets.labels import selection_split
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'manifest.json').write_text(json.dumps({'counts':{
                role:{'train':2,'test':1} for role in ('customer','seller')}}))
            self.assertEqual(selection_split(root), 'test')
            data = SimpleNamespace(close=Mock())
            sample = {'split':'test','sample_id':'query:99'}
            with patch('agent_simulation.training.data_loader.SFTDataset',return_value=data), \
                 patch('agent_simulation.training.data_loader.record_samples',return_value=[sample]) as read:
                self.assertEqual(list(EvaluationStream(root, root, 'customer', 'test', [99])), [sample])
                read.assert_called_once_with(data, 'customer', 99)
            data.close.assert_called_once()

    def test_no_validation_trains_and_saves_final_without_evaluation(self):
        batch = tensor_batch()
        samples = [{k:v[i] for k,v in batch.items()} for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            args = parser().parse_args(['train','--model','unused','--dataset','unused',
                '--tokenizer','unused','--role','customer','--output',tmp,'--device','cpu',
                '--max-steps','2','--batch-size','1','--accumulation','1'])
            trainer = make_trainer(tiny_model(), None, args, samples, None)
            trainer.data_collator = lambda rows:{k:torch.stack([r[k] for r in rows]) for k in rows[0]}
            self.assertEqual(trainer.args.eval_strategy, 'no')
            self.assertFalse(trainer.args.load_best_model_at_end)
            with patch.object(trainer,'evaluate',side_effect=AssertionError('Validation must not run')):
                trainer.train()
                trainer.save_model(str(Path(tmp)/'adapter'))
            self.assertEqual(trainer.state.global_step, 2)
            self.assertIsNone(trainer.state.best_model_checkpoint)
            self.assertTrue((Path(tmp)/'adapter/adapter_model.safetensors').is_file())

    def test_resumed_outcome_has_one_current_status_and_preserves_failure(self):
        from agent_simulation.training.run import write_outcome
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            failure = {"status": "incomplete", "error": "interrupted"}
            success = {"status": "configured_steps_completed", "steps": 2}
            write_outcome(output, completed=False, payload=failure)
            write_outcome(output, completed=True, payload=success)
            self.assertFalse((output / "failure.json").exists())
            self.assertEqual(json.loads((output / "previous_failure.json").read_text()), failure)
            self.assertEqual(json.loads((output / "result.json").read_text()), success)
            write_outcome(output, completed=False, payload=failure)
            self.assertFalse((output / "result.json").exists())

    def test_epoch_size_never_constructs_inputs(self):
        stream = SampleStream("unused", "unused", "customer", "train")
        with patch('agent_simulation.datasets.labels.customer_rows', return_value=7) as count, \
             patch.object(SampleStream, '__iter__', side_effect=AssertionError('Input construction')):
            self.assertEqual(stream.measure_epoch(3), 7)
            self.assertEqual(stream.measure_epoch(3), 7)
            stream.limit = 4
            self.assertEqual(stream.measure_epoch(3), 4)
            self.assertEqual(SampleStream('unused','unused','seller','train').measure_epoch(12),12)
        self.assertEqual(count.call_count, 3)

    def test_mps_checkpoint_roundtrips_device_rng_state(self):
        from types import SimpleNamespace
        from transformers import Trainer
        trainer = object.__new__(BehaviorTrainer)
        trainer.args = SimpleNamespace(device=torch.device("mps"))
        rng = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with tempfile.TemporaryDirectory() as folder, patch.object(Trainer, "_save_rng_state"), \
             patch.object(Trainer, "_load_rng_state"), patch.object(torch.mps, "get_rng_state", return_value=rng), \
             patch.object(torch.mps, "set_rng_state") as restore:
            trainer._save_rng_state(folder)
            trainer._load_rng_state(folder)
            restore.assert_called_once()
            torch.testing.assert_close(restore.call_args.args[0], rng)

    def test_ten_stream_epochs_resume_mid_epoch_and_retain_best_and_latest(self):
        from safetensors.torch import load_file

        batch = tensor_batch()
        class Stream(torch.utils.data.IterableDataset):
            epoch = 0
            def __len__(self):
                return 5
            def set_epoch(self, epoch):
                self.epoch = epoch
            def __iter__(self):
                for i in range(len(self)):
                    yield {k: v[(i + self.epoch) % 2] for k, v in batch.items()}

        class StopAfterFive(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 5:
                    control.should_training_stop = True
                return control

        def make(directory):
            args = parser().parse_args(["train", "--model", "unused", "--dataset", "unused",
                "--role", "seller", "--output", str(directory), "--device", "cpu",
                "--epochs", "10",
                "--accumulation", "2", "--eval-steps", "5", "--learning-rate", "0.01",
                "--no-gradient-checkpointing"])
            self.assertEqual((args.rank, args.epochs, args.max_steps), (64, 10, None))
            model = get_peft_model(tiny_base(), LoraConfig(
                task_type="CAUSAL_LM", r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
            trainer = make_trainer(model, None, args, Stream(), Stream())
            trainer.data_collator = lambda rows: {k: torch.stack([r[k] for r in rows]) for k in batch}
            return trainer

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            reference, resumed = path / "reference", path / "resumed"
            ref = make(reference)
            ref.train()
            partial = make(resumed)
            partial.add_callback(StopAfterFive())
            partial.train()
            interrupted = resumed / "checkpoint-6"
            interrupted.mkdir()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                resume_checkpoint(resumed, interrupted)
            latest = resume_checkpoint(resumed, Path("latest"))
            self.assertEqual(latest.name, "checkpoint-5")
            interrupted.rmdir()
            restored = make(resumed)
            restored.train(resume_from_checkpoint=str(latest))
            for directory, trainer in [(reference, ref), (resumed, restored)]:
                self.assertEqual(trainer.state.global_step, 30)
                self.assertEqual(trainer.state.epoch, 10)
                self.assertTrue(Path(trainer.state.best_model_checkpoint).is_dir())
                self.assertEqual(resume_checkpoint(directory, Path("latest")).name, "checkpoint-30")
                self.assertLessEqual(len(list(directory.glob("checkpoint-*"))), 2)
            self.assertEqual(ref.state.best_metric, restored.state.best_metric)
            for trainer in (ref, restored):
                state = json.loads((Path(trainer.state.best_model_checkpoint).parent
                                    / "checkpoint-30/trainer_state.json").read_text())
                last = next(entry for entry in state["log_history"] if entry["step"] == 30 and "loss" in entry)
                self.assertEqual(last["epoch_train_steps"], 3)
            def epoch_two_loss(trainer):
                return next(entry["epoch_train_loss"] for entry in trainer.state.log_history
                            if entry["epoch"] == 2.0 and "epoch_train_loss" in entry)
            self.assertEqual(epoch_two_loss(ref), epoch_two_loss(restored))
            def weights(directory):
                return load_file(directory / "checkpoint-30/adapter_model.safetensors")
            for key, value in weights(reference).items():
                torch.testing.assert_close(value, weights(resumed)[key], rtol=0, atol=0)
            for name in ("optimizer.pt", "scheduler.pt"):
                torch.testing.assert_close(torch.load(reference / "checkpoint-30" / name),
                                           torch.load(resumed / "checkpoint-30" / name), rtol=0, atol=0)

    def test_optional_cache_release_preserves_the_training_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            args = parser().parse_args([
                'train', '--dataset', 'data', '--model', 'model', '--role', 'seller',
                '--output', folder, '--device', 'cpu', '--max-steps', '12',
                '--accumulation', '2', '--empty-cache-steps', '1',
            ])
            trainer = make_trainer(tiny_model(), None, args, [{}], [{}])
            self.assertEqual(trainer.args.torch_empty_cache_steps, 1)
            self.assertEqual(trainer.args.max_steps, 12)
            self.assertEqual(trainer.args.gradient_accumulation_steps, 2)
            self.assertTrue(trainer.args.gradient_checkpointing)
            args.empty_cache_steps = 0
            with patch('agent_simulation.training.run.check_data') as check:
                with self.assertRaisesRegex(ValueError, 'Cache release interval'):
                    train(args)
            check.assert_not_called()

    def trainer(self, model, output):
        args = TrainingArguments(
            output_dir=output,
            use_cpu=True,
            report_to=[],
            remove_unused_columns=False,
            prediction_loss_only=True,
        )
        return BehaviorTrainer(model=model, args=args)

    def test_five_epochs_can_keep_all_snapshots_on_a_ten_epoch_lr_schedule(self):
        batch = tensor_batch()
        rows = [{k: v[0] for k, v in batch.items()}]
        with tempfile.TemporaryDirectory() as folder:
            args = parser().parse_args(["train", "--model", "unused", "--dataset", "unused",
                "--role", "customer", "--output", folder, "--device", "cpu", "--epochs", "5",
                "--lr-schedule-epochs", "10", "--accumulation", "1", "--keep-checkpoints", "0"])
            trainer = make_trainer(tiny_model(), None, args, rows, None)
            trainer.data_collator = lambda items: {k: torch.stack([r[k] for r in items]) for k in batch}
            trainer.train()
            self.assertEqual(trainer.state.global_step, 5)
            self.assertEqual(len(list(Path(folder).glob("checkpoint-*"))), 5)
            self.assertAlmostEqual(trainer.lr_scheduler.get_last_lr()[0], args.learning_rate / 2)
            self.assertTrue(all((p / "optimizer.pt").exists() for p in Path(folder).glob("checkpoint-*")))

    def test_loss_parts_add_up_and_evaluation_does_not_update_weights(self):
        batch = tensor_batch()
        batch["answer_parts"] = torch.tensor([[-1, -1, 0, 1], [-1, -1, 2, -1]])
        batch["stages"] = torch.tensor([1, 1])
        rows = [{k: v[i] for k, v in batch.items()} for i in range(2)]
        def collator(items):
            return {k: torch.stack([row[k] for row in items]) for k in batch}
        collator.loss_breakdown = True
        with tempfile.TemporaryDirectory() as folder:
            model = tiny_model()
            trainer = self.trainer(model, folder)
            trainer.data_collator = collator
            trainer.eval_dataset = rows
            before = {k: v.detach().clone() for k, v in model.named_parameters()}
            metrics = trainer.evaluate()
            self.assertEqual(metrics["eval_response_rows"], 2)
            self.assertEqual(metrics["eval_response_weight"], 1.25)
            for part in ("structure", "click_id", "purchase_id"):
                self.assertEqual(metrics[f"eval_response_{part}_tokens"], 1)
            self.assertAlmostEqual(metrics["eval_response_answer_loss"], sum(
                metrics[f"eval_response_{part}_loss_contribution"]
                for part in ("structure", "click_id", "purchase_id")), places=6)
            self.assertAlmostEqual(metrics["eval_loss"] * 2 / 1.25, metrics["eval_response_answer_loss"], places=5)
            for name, value in model.named_parameters():
                torch.testing.assert_close(value, before[name])

    def test_answer_mask_and_weights_match_hand_computation(self):
        logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]], [[1.0, 0.0], [2.0, 0.0]]])
        target = torch.tensor([[0, 1], [1, -100]])
        expected = torch.stack(
            [torch.log1p(torch.exp(torch.tensor(-2.0))), torch.log1p(torch.exp(torch.tensor(1.0)))]
        )
        torch.testing.assert_close(answer_losses(logits, target), expected)

    def test_selected_logits_equal_full_context_loss_and_weights_do_not_cancel(self):
        model, batch = tiny_model(), tensor_batch()
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.trainer(model, folder)
            model.eval()
            full = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
            expected = (
                answer_losses(full[:, :-1], batch["labels"][:, 1:]) * batch["sample_weights"]
            ).mean()
            actual = trainer.compute_loss(model, batch)
            torch.testing.assert_close(actual, expected)
            one = {k: v[1:] for k, v in batch.items()}
            weighted = trainer.compute_loss(model, one)
            one["sample_weights"] = torch.ones(1)
            torch.testing.assert_close(weighted, trainer.compute_loss(model, one) / 4)

    def test_accumulated_gradient_matches_uneven_combined_batch(self):
        combined, accumulated = tiny_model(), tiny_model()
        batch = tensor_batch()
        with tempfile.TemporaryDirectory() as folder:
            a, b = self.trainer(combined, folder), self.trainer(accumulated, folder)
            a.compute_loss(combined, batch).backward()
            for index in range(2):
                part = {k: v[index : index + 1] for k, v in batch.items()}
                b.compute_loss(accumulated, part, num_items_in_batch=2).backward()
            for (_, x), (_, y) in zip(combined.named_parameters(), accumulated.named_parameters()):
                if x.requires_grad:
                    torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-4)

    def test_selected_logits_and_gradients_match_distinct_answer_boundaries(self):
        selected, full = tiny_model(), tiny_model()
        batch = {
            "input_ids": torch.tensor([[1, 3, 4, 5, 0, 0, 0, 0], [1, 6, 7, 8, 9, 10, 11, 12]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]]),
            "labels": torch.tensor([[-100, -100, 4, 5, -100, -100, -100, -100],
                                    [-100, -100, -100, -100, -100, -100, 11, 12]]),
            "sample_weights": torch.tensor([1.0, 0.25]),
        }
        with tempfile.TemporaryDirectory() as folder:
            actual = self.trainer(selected, folder).compute_loss(selected, batch)
            logits = full(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                          use_cache=False).logits
            expected = (answer_losses(logits[:, :-1], batch["labels"][:, 1:])
                        * batch["sample_weights"]).mean()
            torch.testing.assert_close(actual, expected)
            actual.backward()
            expected.backward()
            for (name, a), (_, b) in zip(selected.named_parameters(), full.named_parameters()):
                if a.requires_grad:
                    with self.subTest(parameter=name):
                        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)

    def test_optimizer_validation_save_reload_and_resume(self):
        # Short synthetic token fixtures test checkpoint mechanics, not behavior fidelity.
        batch = tensor_batch()
        rows = [{k: v[i] for k, v in batch.items()} for i in range(2)]

        def collator(items):
            return {k: torch.stack([x[k] for x in items]) for k in batch}

        with tempfile.TemporaryDirectory() as folder:

            def make(model, steps, output=folder):
                args = TrainingArguments(
                    output_dir=str(output),
                    use_cpu=True,
                    report_to=[],
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    gradient_accumulation_steps=2,
                    max_steps=steps,
                    learning_rate=0.01,
                    eval_strategy="steps",
                    eval_steps=1,
                    save_steps=1,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    remove_unused_columns=False,
                    prediction_loss_only=True,
                    label_names=["labels"],
                    disable_tqdm=True,
                    gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
                return BehaviorTrainer(
                    model=model,
                    args=args,
                    train_dataset=rows,
                    eval_dataset=rows,
                    data_collator=collator,
                )

            model = tiny_model()
            original = {n: p.detach().clone() for n, p in model.named_parameters()}

            class StopAfterOne(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step == 1:
                        control.should_training_stop = True
                    return control

            trainer = make(model, 2)
            trainer.add_callback(StopAfterOne())
            trainer.train()
            self.assertTrue((Path(folder) / "checkpoint-1/optimizer.pt").is_file())
            self.assertIsNotNone(trainer.state.best_model_checkpoint)
            self.assertTrue(
                any(
                    not torch.equal(original[n], p)
                    for n, p in model.named_parameters()
                    if p.requires_grad
                )
            )
            self.assertTrue(
                all(
                    torch.equal(original[n], p)
                    for n, p in model.named_parameters()
                    if not p.requires_grad
                )
            )
            trainer.save_model(str(Path(folder) / "adapter"))
            reloaded = PeftModel.from_pretrained(tiny_base(), Path(folder) / "adapter")
            model.eval()
            reloaded.eval()
            with torch.no_grad():
                torch.testing.assert_close(
                    model(batch["input_ids"]).logits, reloaded(batch["input_ids"]).logits
                )
            resumed = make(tiny_model(), 2)
            resumed.train(resume_from_checkpoint=str(Path(folder) / "checkpoint-1"))
            self.assertEqual(resumed.state.global_step, 2)
            reference = make(tiny_model(), 2, Path(folder) / "uninterrupted")
            reference.train()
            for (name, a), (_, b) in zip(resumed.model.named_parameters(), reference.model.named_parameters()):
                with self.subTest(resumed_parameter=name):
                    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)

    def test_trainer_accumulation_keeps_the_same_effective_batch_update(self):
        batch = tensor_batch()
        rows = [{k: v[i % 2].clone() for k, v in batch.items()} for i in range(16)]
        for i, row in enumerate(rows):
            row["input_ids"][1] = i + 8

        def collator(items):
            return {k: torch.stack([x[k] for x in items]) for k in batch}

        class CaptureGradients(TrainerCallback):
            def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
                self.gradients = {name: param.grad.detach().clone() for name, param in model.named_parameters()
                                  if param.requires_grad}

        models = [tiny_model() for _ in range(3)]
        runs = []
        with tempfile.TemporaryDirectory() as folder:
            for index, (model, (size, accumulation)) in enumerate(zip(models, ((16, 1), (1, 16), (4, 4)))):
                args = TrainingArguments(
                    output_dir=str(Path(folder) / str(index)),
                    use_cpu=True,
                    max_steps=1,
                    learning_rate=0.01,
                    report_to=[],
                    disable_tqdm=True,
                    per_device_train_batch_size=size,
                    gradient_accumulation_steps=accumulation,
                    max_grad_norm=0,
                    remove_unused_columns=False,
                    save_strategy="no",
                )
                capture = CaptureGradients()
                BehaviorTrainer(
                    model=model, args=args, train_dataset=rows, data_collator=collator, callbacks=[capture]
                ).train()
                runs.append(capture.gradients)
            for model, gradients in zip(models[1:], runs[1:]):
                torch.testing.assert_close(gradients, runs[0], atol=1e-6, rtol=1e-4)
                torch.testing.assert_close(model.state_dict(), models[0].state_dict(), atol=1e-6, rtol=1e-4)

    def test_split_stream_keeps_prior_history_and_excludes_future_test_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            labels = root / "customer/validation"
            labels.mkdir(parents=True)
            connection = duckdb.connect(str(root / "catalog.duckdb"))
            connection.execute(
                "CREATE TABLE requests AS SELECT * FROM (VALUES (1),(2),(3)) t(time_index)"
            )
            connection.execute("CREATE TABLE seller_history AS SELECT * FROM requests")
            connection.execute(
                "COPY (SELECT 2 history_cutoff_exclusive) TO ? (FORMAT PARQUET)",
                [str(labels / "labels.parquet")],
            )
            calls = []

            class Fixture:
                def __init__(self, *a, **kw):
                    self.catalog = type("Catalog", (), {"connection": connection})()

                def iter_samples(self, role, split, stage=None):
                    calls.append((role, split))
                    for table in ("requests", "seller_history"):
                        assert connection.execute(
                            f"SELECT list(time_index ORDER BY time_index) FROM {table}"
                        ).fetchone()[0] == [1, 2]
                    assert stage == "response"
                    yield {"sample_id": "response:2"}

                def close(self):
                    calls.append("closed")

            with patch("agent_simulation.training.data_loader.SFTDataset", Fixture):
                stream = SampleStream(
                    root, root, "customer", "validation", stage="response", limit=1
                )
                self.assertEqual(list(stream), [{"sample_id": "response:2"}])
            self.assertEqual(calls, [("customer", "validation"), "closed"])
            connection.close()
            connection = duckdb.connect(str(root / "catalog.duckdb"))
            self.assertEqual(
                connection.execute("SELECT max(time_index) FROM requests").fetchone()[0], 3
            )
            connection.close()

    def test_seeded_evaluation_selects_complete_requests_from_test_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "customer/test"
            directory.mkdir(parents=True)
            connection = duckdb.connect()
            connection.execute(
                "COPY (SELECT range+1 session_id FROM range(20)) TO ? (FORMAT PARQUET)",
                [str(directory / "labels.parquet")],
            )
            first = select_record_ids(connection, root, "customer", "test", 5, 42)
            self.assertEqual(first, select_record_ids(connection, root, "customer", "test", 5, 42))
            self.assertNotEqual(first, select_record_ids(connection, root, "customer", "test", 5, 43))
            self.assertEqual(len(set(first)), 5)

            class Fixture:
                def __init__(self, *a, **kw):
                    self.catalog = type("Catalog", (), {"connection": connection})()

                def sample(self, sample_id):
                    return {"sample_id": sample_id, "split": "test"}

                def request_samples(self, session_id):
                    return [self.sample(f"query:{session_id}")] + [
                        {
                            "sample_id": f"response:{session_id}:batch:{i}",
                            "split": "test",
                            "batching": {"count": 2, "request_weight": 0.5},
                        }
                        for i in (1, 2)
                    ]

                def close(self):
                    pass

            with patch("agent_simulation.training.data_loader.SFTDataset", Fixture):
                result = list(EvaluationStream(root, root, "customer", "test", first))
                self.assertEqual(len(result), 15)
                self.assertEqual(
                    [s["sample_id"] for s in result[::3]], [f"query:{i}" for i in first]
                )
                for index in range(0, len(result), 3):
                    self.assertEqual(
                        sum(s["batching"]["request_weight"] for s in result[index + 1 : index + 3]),
                        1,
                    )
            connection.close()
