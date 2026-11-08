"""Weighted answer loss, checkpointing and evaluation on the configured selection split."""

from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import Trainer, TrainerCallback, TrainingArguments

from agent_simulation.training.data_loader import TensorCollator


def answer_losses(logits, targets):
    """Average token cross-entropy inside each answer, ignoring padding and prompts."""
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
    return (losses.reshape_as(targets) * valid).sum(-1) / counts


class EpochCheckpoint(TrainerCallback):
    """Evaluate and save at epoch boundaries and the final step as well as periodic saves."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps or (state.epoch is not None and state.epoch % 1 == 0):
            control.should_save = True
            control.should_evaluate = args.eval_strategy != 'no'
        return control


class EpochStream(torch.utils.data.IterableDataset):
    """Keep Accelerate's replacement skip-loader from resetting the underlying shuffle epoch."""

    def __init__(self, stream):
        self.stream = stream

    def __len__(self):
        return len(self.stream)

    def __iter__(self):
        return iter(self.stream)


class RestoreShuffleEpoch(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        train_dataloader.dataset.stream.set_epoch(int(state.epoch or 0))


class BehaviorTrainer(Trainer):
    """Use a fixed example denominator so 1/batch_count weights do not cancel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This subclass consumes the accumulation denominator, not model.forward.
        self.model_accepts_loss_kwargs = True

    def _get_num_items_in_batch(self, batch_samples, device):
        return sum(batch["labels"].shape[0] for batch in batch_samples)

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
        losses = answer_losses(outputs.logits, labels[:, positions + 1])
        denominator = num_items_in_batch if num_items_in_batch is not None else labels.shape[0]
        loss = (losses * weights).sum() / denominator
        return (loss, outputs) if return_outputs else loss


def make_trainer(model, tokenizer, args, train_data, eval_data):
    """Share the identical training setup with bounded integration tests."""
    interval = min(args.eval_steps, args.max_steps) if args.max_steps else args.eval_steps
    evaluate = eval_data is not None
    options = TrainingArguments(
        output_dir=str(args.output),
        use_cpu=args.device == "cpu",
        max_steps=args.max_steps or -1,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        eval_strategy="steps" if evaluate else "no",
        eval_steps=interval,
        save_strategy="steps",
        save_steps=interval,
        logging_steps=1,
        save_total_limit=2,
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
    )
    if options.device.type != args.device or options.world_size != 1:
        raise ValueError(
            "This entry point requires one explicit device/process; no implicit fallback"
        )
    callbacks = [EpochCheckpoint()]
    if isinstance(train_data, torch.utils.data.IterableDataset):
        train_data = EpochStream(train_data)
        callbacks.append(RestoreShuffleEpoch())
    return BehaviorTrainer(
        model=model,
        args=options,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        data_collator=TensorCollator(tokenizer),
        callbacks=callbacks,
    )
