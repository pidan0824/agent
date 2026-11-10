"""Weighted answer loss, checkpointing and evaluation on the configured selection split."""

import math
from itertools import repeat
from pathlib import Path
import torch
from torch.nn import functional as F
from accelerate.utils import gather_object
from transformers import Trainer, TrainerCallback, TrainingArguments

from agent_simulation.training.data_loader import TensorCollator, RESPONSE_FORMAT_WEIGHT


def answer_losses(logits, targets, *, token_weights=None, return_tokens=False):
    """Normalize weighted token CE inside each answer, ignoring prompts and padding."""
    valid = targets.ne(-100)
    counts = valid.sum(-1)
    if torch.any(counts == 0):
        raise ValueError("Each example must have a supervised answer")
    losses = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        ignore_index=-100,
    )
    tokens = losses.reshape_as(targets)
    weights = valid if token_weights is None else token_weights * valid
    means = (tokens * weights).sum(-1) / weights.sum(-1)
    return (means, tokens) if return_tokens else means


class EpochCheckpoint(TrainerCallback):
    """Evaluate and save at epoch boundaries and the final step as well as periodic saves."""

    def on_step_end(self, args, state, control, **kwargs):
        if args.eval_strategy == "epoch":
            return control
        if state.global_step >= state.max_steps or (state.epoch is not None and state.epoch % 1 == 0):
            control.should_save = True
            control.should_evaluate = args.eval_strategy != 'no'
        return control


class EpochStream(torch.utils.data.IterableDataset):
    """Keep shuffle state on resume and give every rank a full last batch."""

    def __init__(self, stream, multiple=1):
        self.stream, self.multiple = stream, multiple

    def __len__(self):
        size = len(self.stream)
        return size + (-size % self.multiple)

    def __iter__(self):
        yield from self.stream
        if self.multiple > 1:
            yield from repeat(None, -len(self.stream) % self.multiple)


class RestoreShuffleEpoch(TrainerCallback):
    def __init__(self, stream):
        self.stream = stream

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.stream.set_epoch(int(state.epoch or 0))


class BehaviorTrainer(Trainer):
    """Use a fixed example denominator so 1/batch_count weights do not cancel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This subclass consumes the accumulation denominator, not model.forward.
        self.model_accepts_loss_kwargs = True
        # compute_loss already divides by the number of examples across all microbatches.
        self.accelerator.gradient_accumulation_steps = 1
        self._loss_totals = None
        self._evaluating = False
        self.schedule_epochs = None
        self.activation_offload_tokens = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        limit = self.activation_offload_tokens
        if limit is not None and inputs["input_ids"].shape[-1] >= limit:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                return super().training_step(model, inputs, num_items_in_batch)
        return super().training_step(model, inputs, num_items_in_batch)

    def create_scheduler(self, num_training_steps, optimizer=None):
        if self.schedule_epochs is not None:
            num_training_steps = math.ceil(num_training_steps * self.schedule_epochs / self.args.num_train_epochs)
        return super().create_scheduler(num_training_steps, optimizer)

    def log(self, logs, start_time=None):
        epoch = self.state.epoch
        if "loss" in logs and epoch is not None:
            ending = epoch % 1 == 0 or self.control.should_training_stop or self.control.should_epoch_stop
            if ending:
                start = math.ceil(epoch) - 1
                # With logging_steps=1, history contains one loss per optimizer step,
                # including steps restored from a checkpoint inside this epoch.
                losses = [entry["loss"] for entry in self.state.log_history
                          if "loss" in entry and start < entry.get("epoch", 0) <= epoch]
                losses.append(logs["loss"])
                logs["epoch_train_loss"] = math.fsum(losses) / len(losses)
                logs["epoch_train_steps"] = len(losses)
        return super().log(logs, start_time=start_time)

    def evaluation_loop(self, dataloader, *args, metric_key_prefix="eval", **kwargs):
        self._loss_totals = {} if getattr(self.data_collator, "loss_breakdown", False) else None
        self._evaluating = True
        try:
            result = super().evaluation_loop(dataloader, *args, metric_key_prefix=metric_key_prefix, **kwargs)
            totals_by_stage = self._loss_totals or {}
            if self.args.world_size > 1:
                dataset = dataloader.dataset.dataset
                # Trainer averages over the full padded batches. Only real rows contribute loss.
                samples = len(dataset.stream)
                result.metrics[f"{metric_key_prefix}_loss"] *= len(dataset) / samples
                result = result._replace(num_samples=samples)
                if self._loss_totals is not None:
                    totals_by_stage = self._merge_loss_totals(gather_object([self._loss_totals]))
            for stage, totals in totals_by_stage.items():
                prefix = f"{metric_key_prefix}_{stage}"
                rows, weight, answer = totals["rows"], totals["weight"], totals["answer"]
                result.metrics.update({prefix + "_rows": rows, prefix + "_weight": weight,
                                       prefix + "_answer_loss": answer / weight})
                if stage == "response":
                    for part, (tokens, weighted_tokens, loss, contribution) in totals["parts"].items():
                        result.metrics.update({
                            f"{prefix}_{part}_tokens": tokens,
                            f"{prefix}_{part}_weighted_tokens": weighted_tokens,
                            f"{prefix}_{part}_token_loss": loss / weighted_tokens if weighted_tokens else None,
                            f"{prefix}_{part}_loss_contribution": contribution / weight,
                        })
            return result
        finally:
            self._loss_totals = None
            self._evaluating = False

    @staticmethod
    def _merge_loss_totals(by_rank):
        merged = {}
        for rank_totals in by_rank:
            for stage, totals in rank_totals.items():
                if stage not in merged:
                    merged[stage] = totals
                    continue
                current = merged[stage]
                for field in ("rows", "weight", "answer"):
                    current[field] += totals[field]
                for part, values in totals["parts"].items():
                    current["parts"][part] = [a + b for a, b in zip(current["parts"][part], values)]
        return merged

    def _record_loss_parts(self, inputs, targets, token_losses, positions, effective_weights, token_weights):
        parts = inputs["answer_parts"][:, positions + 1].detach().cpu()
        valid = targets.ne(-100).detach().cpu()
        factors = valid.double() if token_weights is None else token_weights.detach().double().cpu() * valid
        losses = token_losses.detach().double().cpu() * factors
        for i, (stage, request_weight, weight) in enumerate(zip(
            inputs["stages"].tolist(), inputs["sample_weights"].tolist(), effective_weights.tolist()
        )):
            if weight == 0:
                continue
            stage = ("query", "response", "seller")[stage]
            totals = self._loss_totals.setdefault(stage, {"rows": 0, "weight": 0., "answer": 0.,
                "parts": {p: [0, 0., 0., 0.] for p in ("structure", "click_id", "purchase_id", "decision")}})
            count = float(factors[i].sum())
            totals["rows"] += 1
            totals["weight"] += request_weight
            totals["answer"] += weight * float(losses[i][valid[i]].sum()) / count
            if stage == "response":
                for number, name in enumerate(totals["parts"]):
                    selected = valid[i] & parts[i].eq(number)
                    n, loss = int(selected.sum()), float(losses[i][selected].sum())
                    item = totals["parts"][name]
                    item[0] += n
                    item[1] += weight * float(factors[i][selected].sum())
                    item[2] += weight * loss
                    item[3] += weight * loss / count

    def _get_num_items_in_batch(self, batch_samples, device):
        if self._evaluating:
            # Trainer reports a mean over physical rows; remove padding once for the full evaluation.
            return sum(batch["labels"].shape[0] for batch in batch_samples)
        count = sum(batch["real_rows"].sum() if "real_rows" in batch else batch["labels"].shape[0]
                    for batch in batch_samples)
        if self.args.world_size > 1:
            # DDP averages gradients across ranks, so divide by global real rows / ranks.
            count = self.accelerator.reduce(torch.as_tensor(count, device=device), reduction="sum")
            count = count / self.args.world_size
        return count

    def _save_rng_state(self, output_dir):
        super()._save_rng_state(output_dir)
        # The installed Trainer persists CPU and CUDA generators but omits Metal.
        if self.args.device.type == "mps":
            torch.save(torch.mps.get_rng_state(), Path(output_dir) / "mps_rng_state.pt")

    def _load_rng_state(self, checkpoint):
        super()._load_rng_state(checkpoint)
        if checkpoint is not None and self.args.device.type == "mps":
            state = torch.load(Path(checkpoint) / "mps_rng_state.pt", map_location="cpu", weights_only=True)
            torch.mps.set_rng_state(state)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels, weights = inputs["labels"], inputs["sample_weights"]
        if torch.any(~torch.isfinite(weights)) or torch.any(weights <= 0) or torch.any(weights > 1):
            raise ValueError("Invalid sample weights")
        if "action_multipliers" in inputs:
            weights = weights * inputs["action_multipliers"]
        if "real_rows" in inputs:
            weights = weights.masked_fill(~inputs["real_rows"], 0)
        # Qwen can project only positions needed to predict answer tokens. Context
        # attention remains complete; neither facts nor supervised tokens are cut.
        positions = labels[:, 1:].ne(-100).any(0).nonzero().flatten()
        if positions.numel() == 0:
            raise ValueError("No answer tokens")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            logits_to_keep=positions,
        )
        targets = labels[:, positions + 1]
        token_weights = inputs.get("token_weights")
        if token_weights is not None:
            token_weights = token_weights[:, positions + 1]
        if self._loss_totals is not None:
            losses, tokens = answer_losses(outputs.logits, targets, token_weights=token_weights, return_tokens=True)
            self._record_loss_parts(inputs, targets, tokens, positions, weights, token_weights)
        else:
            losses = answer_losses(outputs.logits, targets, token_weights=token_weights)
        denominator = num_items_in_batch if num_items_in_batch is not None else labels.shape[0]
        loss = (losses * weights).sum() / denominator
        return (loss, outputs) if return_outputs else loss


def make_trainer(model, tokenizer, args, train_data, eval_data, *, evaluate_each_epoch=False,
                 action_weights=None):
    """Share the identical training setup with bounded integration tests."""
    interval = min(args.eval_steps, args.max_steps) if args.max_steps else args.eval_steps
    evaluate = eval_data is not None
    strategy = "epoch" if evaluate_each_epoch and not args.max_steps else "steps"
    options = TrainingArguments(
        output_dir=str(args.output),
        use_cpu=args.device == "cpu",
        max_steps=args.max_steps or -1,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        eval_strategy=strategy if evaluate else "no",
        eval_steps=interval,
        save_strategy=strategy,
        save_steps=interval,
        logging_steps=1,
        save_total_limit=getattr(args, "keep_checkpoints", 2) or None,
        save_only_model=False,
        load_best_model_at_end=evaluate,
        metric_for_best_model="eval_loss" if evaluate else None,
        greater_is_better=False,
        prediction_loss_only=True,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
        dataloader_pin_memory=args.device == "cuda",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        torch_empty_cache_steps=getattr(args, "empty_cache_steps", None),
        report_to=["tensorboard"] if getattr(args, "tensorboard", False) else [],
        logging_dir=str(args.output / "tensorboard"),
        disable_tqdm=getattr(args, "tensorboard", False),
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
        accelerator_config={"dispatch_batches": False},
        ddp_find_unused_parameters=False,
    )
    if options.device.type != args.device or options.n_gpu > 1:
        raise ValueError("Select one device, or use torchrun with one process per GPU")
    callbacks = [EpochCheckpoint()]
    if hasattr(train_data, "set_epoch"):
        callbacks.append(RestoreShuffleEpoch(train_data))
    if isinstance(train_data, torch.utils.data.IterableDataset) or options.world_size > 1:
        train_data = EpochStream(train_data, options.world_size * args.batch_size if options.world_size > 1 else 1)
    if evaluate and options.world_size > 1:
        eval_data = EpochStream(eval_data, options.world_size * args.batch_size)
    trainer = BehaviorTrainer(
        model=model,
        args=options,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        data_collator=TensorCollator(tokenizer, loss_breakdown=getattr(args, "loss_breakdown", False),
                                     action_weights=action_weights,
                                     response_format_weight=(
                                         getattr(args, "response_format_weight", RESPONSE_FORMAT_WEIGHT)
                                         if getattr(args, "role", None) == "customer"
                                         and getattr(args, "stage", None) != "query" else 1.0)),
        callbacks=callbacks,
    )
    trainer.schedule_epochs = getattr(args, "lr_schedule_epochs", None)
    trainer.activation_offload_tokens = getattr(args, "activation_offload_tokens", None)
    return trainer
