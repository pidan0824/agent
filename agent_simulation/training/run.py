"""Check or train separate KuaiSearch Customer/Seller LoRA adapters, locally."""


import argparse
import importlib.metadata
import json
import math
import torch
from itertools import islice
from pathlib import Path
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, set_seed
from agent_simulation.inference.local.checkpoints import adapter_training_identity
from agent_simulation.inference.requests import load_tokenizer
from agent_simulation.inference import resolve_device
from agent_simulation.logs import write_report
from agent_simulation.datasets.labels import SFTDataset, record_samples, select_record_ids, selection_split
from agent_simulation.datasets.tokens import cache_tokenizer_digest, validate_release
from agent_simulation.environment.batching import response_batch_policy
from agent_simulation.training.data_loader import (
    prepare_datasets,
    action_multipliers,
    ACTION_WEIGHTS,
    RESPONSE_FORMAT_WEIGHT,
    TensorCollator,
    TokenStream,
)
from agent_simulation.training.trainer import make_trainer
from agent_simulation.data.common import sha256


def input_report(args, shards, eval_split, counts, checks, *, cached=False):
    return {
        "role": args.role,
        "counts": counts,
        "checks": checks,
        "test_used": eval_split == "test",
        "test_used_for_gradient_updates": False,
        "weights_loaded": False,
        "local_model_shards_present": shards,
        "tokenizer_matches_model": True,
        "capacity_checked_during_cache_build": cached,
    }


def sample_probe(collator, sample, sample_id, split):
    batch = collator([sample])
    return {
        "sample_id": sample_id,
        "split": split,
        "tokens": batch["input_ids"].shape[1],
        "answer_tokens": batch["labels"].ne(-100).sum().item(),
        "weight": batch["sample_weights"].item(),
    }


def check_cached_data(args, data, shards, *, verify=False):
    """Read cached split counts and optionally verify files and sample rows."""
    spec = validate_release(args.token_cache, args.dataset, args.seed, role=args.role,
                            tokenizer_path=getattr(args, 'tokenizer', None))
    collator = TensorCollator(data.tokenizer) if verify else None
    checks, counts = [], {}
    eval_split = selection_split(args.dataset)
    for split in ("train", eval_split):
        stream = TokenStream(args.token_cache, args.role, split, verify=verify, stage=getattr(args, "stage", None))
        if not len(stream):
            raise ValueError('Empty cached supervision split: ' + split)
        counts[split] = len(stream)
        if verify:
            for sample in islice(stream, 2):
                checks.append(sample_probe(collator, sample, sample.sample_id, split))
    return {
        **input_report(args, shards, eval_split, counts, checks, cached=True),
        "count_unit": "expanded_cached_rows",
        "token_cache_identity": spec,
        "evaluation_selection": "All evaluation rows in the frozen token cache; no resampling",
    }


def check_data(args):
    """Bind training inputs, optionally scanning files and collating sample rows."""
    verify = args.command == "check" or getattr(args, "verify", False)
    if getattr(args, "stage", None) and (args.role != "customer" or not args.token_cache):
        raise ValueError("Stage selection requires Customer token cache")
    if getattr(args, "loss_breakdown", False) and not args.token_cache:
        raise ValueError("Loss diagnostics require a token cache")
    config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    index = args.model / "model.safetensors.index.json"
    shards = (sorted(set(json.loads(index.read_text())["weight_map"].values()))
              if index.is_file() else ["model.safetensors"])
    if not shards or any(Path(p).name != p or not (args.model / p).is_file() for p in shards):
        raise ValueError("Complete local model shards are required")
    args.tokenizer = args.tokenizer or args.model
    if (args.tokenizer.resolve() != args.model.resolve()
            and cache_tokenizer_digest(args.tokenizer) != cache_tokenizer_digest(args.model)):
        raise ValueError("Training tokenizer or chat template differs from the selected model")
    if args.token_cache and config.model_type != "qwen2":
        identity = json.loads((args.token_cache / "identity.json").read_text())
        if "tokenizer_sha256" not in identity:
            raise ValueError("Build a new token cache with the selected model's tokenizer")
    if getattr(args, "token_cache", None) and not verify:
        return check_cached_data(args, None, shards)
    data = SFTDataset(args.dataset, args.tokenizer, verify_hashes=verify)
    checks = []
    try:
        if getattr(args, "token_cache", None):
            return check_cached_data(args, data, shards, verify=verify)
        eval_split = selection_split(args.dataset)
        if verify:
            collator = TensorCollator(data.tokenizer)
            for split in ("train", eval_split):
                path = args.dataset / args.role / split / "labels.parquet"
                key = "sample_id" if args.role == "seller" else "session_id"
                rows = data.catalog.connection.execute(
                    f"SELECT {key} FROM read_parquet(?) LIMIT 2", [str(path)]
                ).fetchall()
                if not rows:
                    raise ValueError("Empty supervision split: " + split)
                samples = (s for (record,) in rows for s in record_samples(data, args.role, record))
                for sample in samples:
                    if sample["split"] != split:
                        raise ValueError(f"Expected {split} sample, got {sample['sample_id']}")
                    checks.append(sample_probe(collator, sample, sample["sample_id"], split))
        result = input_report(args, shards, eval_split, data.manifest["counts"][args.role], checks)
        if args.command == "train":
            result["evaluation_selection"] = {
                "seed": args.seed,
                "unit": "request" if args.role == "customer" else "item_time",
                "record_ids": (
                    select_record_ids(data.catalog.connection, args.dataset, args.role, eval_split,
                                      args.validation_samples, args.seed)
                    if args.validation_samples
                    else None
                ),
                "full_evaluation": args.validation_samples == 0,
                "split": eval_split,
            }
        return result
    finally:
        data.close()


def training_identity(args, data_check, eval_split):
    """Bind resume to the same data, base weights and training settings."""
    value = {
        k: str(v.resolve()) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("command", "resume", "verify")
    }
    # Only neutral weights are omitted: new defaults must not silently resume an unweighted run.
    for key, default in (("stage", None), ("loss_breakdown", False),
                         ("keep_checkpoints", 2), ("lr_schedule_epochs", None),
                         ("activation_offload_tokens", None), ("click_weight", 1.0),
                         ("purchase_weight", 1.0), ("price_change_weight", 1.0),
                         ("response_format_weight", 1.0)):
        if value.get(key) == default:
            value.pop(key, None)
    if args.role != "customer" or getattr(args, "stage", None) == "query":
        value.pop("response_format_weight", None)
    if value.get("init_adapter") is None:
        value.pop("init_adapter", None)
    value["dataset_manifest_sha256"] = sha256(args.dataset / "manifest.json")
    if getattr(args, 'token_cache', None):
        value['token_cache_identity_sha256'] = sha256(args.token_cache/'identity.json')
        buckets = data_check["token_cache_identity"]["buckets"]
        role_cache = args.token_cache / args.role
        value["token_cache_bucket_sha256"] = {
            split: [sha256(role_cache / split / f"{bucket:03d}" / "complete.json")
                    for bucket in range(buckets)]
            for split in ("train", eval_split)
        }
    elif args.role == "customer" and getattr(args, "stage", None) != "query":
        value["customer_response_batching"] = response_batch_policy()
    value["base_weight_sha256"] = {
        p: sha256(args.model / p) for p in data_check["local_model_shards_present"]
    }
    value["base_config_sha256"] = sha256(args.model / "config.json")
    return value


def resume_checkpoint(output, requested, *, device="cpu"):
    """Select a completed single-process PEFT training snapshot in this run."""
    output = output.resolve()
    latest = requested == Path("latest")
    if latest:
        candidates = sorted((p for p in output.glob("checkpoint-*") if p.name[11:].isdigit()),
                            key=lambda p: int(p.name[11:]), reverse=True)
    else:
        candidates = [requested.resolve(strict=True)]
    for path in candidates:
        if path.resolve().parent != output or not path.name.startswith("checkpoint-") or not path.name[11:].isdigit():
            raise ValueError("Resume must name a numbered checkpoint in this output directory")
        # Trainer writes trainer_state.json last, after the weights, optimizer, scheduler and RNG.
        required = ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt",
                    "scheduler.pt", "rng_state.pth", "trainer_state.json")
        if device == "mps":
            required += ("mps_rng_state.pt",)
        if all((path / name).is_file() and (path / name).stat().st_size for name in required):
            try:
                state = json.loads((path / "trainer_state.json").read_text())
            except json.JSONDecodeError:
                if latest:
                    continue
                raise
            if state.get("global_step") == int(path.name[11:]):
                return path
        if not latest:
            raise ValueError(f"Incomplete training checkpoint: {path}")
    raise ValueError("No complete training checkpoint in this output directory")


def load_base_model(args):
    config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model_class = AutoModelForCausalLM
    if config.model_type == "qwen3_5":
        from transformers import Qwen3_5ForCausalLM

        model_class = Qwen3_5ForCausalLM
        config = config.get_text_config()
    model = model_class.from_pretrained(
        args.model, config=config, local_files_only=True, trust_remote_code=False,
        dtype=getattr(torch, args.dtype), attn_implementation="sdpa",
    )
    model.config.use_cache = False
    return model


def train(args):
    if not 0 < getattr(args, "response_format_weight", RESPONSE_FORMAT_WEIGHT) <= 1:
        raise ValueError("Response format weight must be in (0, 1]")
    multipliers = {key: getattr(args, key, default) for key, default in ACTION_WEIGHTS.items()}
    if any(not math.isfinite(value) or value < 1 for value in multipliers.values()):
        raise ValueError("Action weights must be finite and at least 1")
    if getattr(args, "empty_cache_steps", None) is not None and args.empty_cache_steps < 1:
        raise ValueError("Cache release interval must be positive")
    if getattr(args, "lr_schedule_epochs", None) is not None and (args.max_steps or args.lr_schedule_epochs < args.epochs):
        raise ValueError("Learning-rate schedule epochs must cover all training epochs")
    budget = args.max_steps if args.max_steps is not None else args.epochs
    if min(budget, args.eval_steps, args.batch_size, args.accumulation, args.rank) < 1:
        raise ValueError("Positive step, batch, accumulation and rank values required")
    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
        or args.validation_samples < 0
        or args.train_samples < 0
    ):
        raise ValueError("Invalid learning rate or sample limit")
    if not 0 <= getattr(args, "data_workers", 0) <= 32:
        raise ValueError("Expected 0..32 data workers")
    if getattr(args, "data_workers", 0) and args.train_samples:
        raise ValueError("Parallel readers require the complete training pool")
    if args.shuffle_buffer < 0:
        raise ValueError("Invalid shuffle buffer")
    relevant_weight = (multipliers['price_change_weight'] if args.role == "seller" else
                       1.0 if getattr(args, "stage", None) == "query" else
                       max(multipliers['click_weight'], multipliers['purchase_weight']))
    if relevant_weight > 1 and not args.token_cache:
        raise ValueError("Action weighting requires a completed token cache")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Use a new output directory or explicitly resume its checkpoint")
    args.device = resolve_device(args.device).type
    offload = getattr(args, "activation_offload_tokens", None)
    if offload is not None and (offload < 1 or args.device != "cuda" or not args.gradient_checkpointing):
        raise ValueError("Activation offload requires a positive token threshold, CUDA and gradient checkpointing")
    checkpoint = None
    adapter = args.init_adapter
    if args.resume:
        checkpoint = resume_checkpoint(args.output, args.resume, device=args.device)
        config = json.loads((args.output / "run_config.json").read_text())
        # Retired evaluation-only settings do not affect the saved optimizer or data stream.
        saved_identity = {k: v for k, v in config["identity"].items()
                          if k not in ("training_code_sha256", "evaluate_train")}
        if saved_identity.get("init_adapter"):
            adapter = checkpoint
        if args.tokenizer is None and saved_identity.get("tokenizer"):
            args.tokenizer = Path(saved_identity["tokenizer"])
        if args.dtype == "auto":
            args.dtype = saved_identity["dtype"]
    args.tokenizer = args.tokenizer or args.model
    if args.dtype == "auto":
        args.dtype = "bfloat16" if args.device == "cuda" and torch.cuda.is_bf16_supported() else "float32"
    if adapter is not None:
        adapter_identity = adapter_training_identity(adapter, args.role)
        adapter_config = json.loads((adapter / "adapter_config.json").read_text())
        if adapter_config["peft_type"] != "LORA":
            raise ValueError("Expected a LoRA adapter")
        args.rank = adapter_config["r"]
    # Bind the current data and model before loading weights.
    data_check = check_data(args)
    eval_split = selection_split(args.dataset)
    identity = training_identity(args, data_check, eval_split)
    if relevant_weight > 1:
        # Earlier checkpoints selected by unweighted loss cannot share the best-metric state.
        identity["evaluation_action_weights"] = True
    if adapter is not None and any(adapter_identity.get(key) != identity[key]
                                   for key in ("base_config_sha256", "base_weight_sha256")):
        raise ValueError("Adapter training base differs from --model")
    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "accelerate", "duckdb", "numpy", "tokenizers")
    }
    if args.resume:
        if adapter is not None:
            identity["init_adapter"] = saved_identity["init_adapter"]
        if saved_identity != identity:
            raise ValueError("Resume data/model/config differs")
        if config["packages"] != packages:
            raise ValueError("Resume package versions differ")
    tokenizer = load_tokenizer(str(args.tokenizer), local_files_only=True)
    train_data, evaluation_data, samples_per_epoch = prepare_datasets(
        args, tokenizer, data_check, eval_split,
        resume_epoch_size=config["samples_per_epoch"] if checkpoint and not args.token_cache else None,
    )
    if samples_per_epoch == 0:
        raise ValueError("Empty training epoch")
    action_weights = action_multipliers(args, tokenizer, train_data, evaluation_data)
    if not args.resume:
        args.output.mkdir(parents=True)
        write_report(
            args.output / "run_config.json",
            {
                "identity": identity,
                "samples_per_epoch": samples_per_epoch,
                "input_check": data_check,
                "packages": packages,
                "loss": "mean_over_examples(request_weight * action_multiplier * sum(token_weight * CE) / sum(token_weight))",
                "action_weighted_rows_across_splits": len(action_weights),
                "checkpoint_selection": f"lowest {eval_split} answer loss",
                "evaluation_split": eval_split,
                "lora_reference": "PolicySim rank 64; targets/alpha/dropout are explicit project choices",
            },
        )
    try:
        set_seed(args.seed)
        model = load_base_model(args)
        if adapter is not None:
            model = PeftModel.from_pretrained(model, adapter, is_trainable=True, local_files_only=True)
        else:
            targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
            if model.config.model_type == "qwen3_5_text":
                targets.extend(["in_proj_qkv", "out_proj"])
            model = get_peft_model(model, LoraConfig(
                task_type="CAUSAL_LM",
                r=args.rank,
                lora_alpha=2 * args.rank,
                lora_dropout=0.05,
                bias="none",
                target_modules=targets,
            ))
        model.print_trainable_parameters()
        trainer = make_trainer(
            model, tokenizer, args, train_data, evaluation_data, evaluate_each_epoch=True,
            action_weights=action_weights,
        )
        initial_metrics = trainer.evaluate() if not args.resume else None
        if initial_metrics is not None:
            write_report(args.output / "initial_evaluation.json", initial_metrics)
        result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        best = trainer.state.best_model_checkpoint
        if best is None or not math.isfinite(trainer.state.best_metric):
            raise ValueError("No checkpoint with finite evaluation loss")
        best_step = trainer.state.best_global_step
        metrics = next(entry for entry in reversed(trainer.state.log_history)
                       if entry.get("step") == best_step and "eval_loss" in entry)
        trainer.save_model(str(args.output / "adapter"))
        trainer.save_state()
        write_outcome(
            args.output, completed=True, payload={
                "status": "configured_steps_completed",
                "steps": trainer.state.global_step,
                "epochs_completed": trainer.state.epoch if samples_per_epoch is not None else None,
                "train": result.metrics,
                "evaluation": metrics,
                "evaluation_split": eval_split,
                "best_checkpoint": best,
                "initial_evaluation": initial_metrics,
                "test_used": eval_split == "test",
                "test_used_for_gradient_updates": False,
                "behavior_generation_evaluated": False,
            },
        )
        return metrics
    except Exception as error:
        write_outcome(args.output, completed=False, payload={"status": "incomplete", "error": repr(error)})
        raise


def write_outcome(directory, *, completed, payload):
    current, previous = ("result", "failure") if completed else ("failure", "result")
    marker = directory / (previous + ".json")
    if marker.exists():
        marker.replace(directory / ("previous_" + previous + ".json"))
    write_report(directory / (current + ".json"), payload)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("check", "train"):
        q = sub.add_parser(command)
        q.add_argument("--dataset", type=Path, required=True)
        q.add_argument("--model", type=Path,
                       default=Path(__file__).resolve().parents[2] / "models/Qwen3.5-9B",
                       help="Local base model (default: models/Qwen3.5-9B); tokenizer defaults to this directory")
        q.add_argument("--tokenizer", type=Path, help="Defaults to the tokenizer bundled with --model")
        q.add_argument("--role", choices=("seller", "customer"), required=True)
        q.add_argument("--stage", choices=("query", "response"),
                       help="Train/evaluate one Customer stage from an existing token cache")
        q.add_argument(
            "--eval-samples", "--validation-samples", dest="validation_samples",
            type=int,
            default=128,
            help="Seeded evaluation requests or item-time records; 0 evaluates the complete selection split",
        )
        q.add_argument("--token-cache", type=Path, help="Read a completed token-cache release")
        q.add_argument("--seed", type=int, default=42)
        if command == "check":
            continue
        q.add_argument("--verify", action="store_true",
                       help="Run the file checks and sample probes from the check command before training")
        q.add_argument("--output", type=Path, required=True)
        q.add_argument("--tensorboard", action="store_true", help="Write training and evaluation curves under OUTPUT/tensorboard")
        q.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
        q.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto",
                       help="Auto uses bfloat16 on supported CUDA devices, float32 otherwise")
        budget = q.add_mutually_exclusive_group()
        budget.add_argument("--epochs", type=int, default=5, help="Complete passes over the training stream (default: 5)")
        budget.add_argument("--max-steps", type=int, help="Explicit optimizer-step budget for bounded runs")
        q.add_argument("--save-steps", "--eval-steps", dest="eval_steps", type=int, default=100,
                       help="Checkpoint interval for explicit step-budget runs")
        q.add_argument("--learning-rate", type=float, default=1e-6)
        q.add_argument("--click-weight", type=float, default=ACTION_WEIGHTS["click_weight"], help="Train and evaluation loss multiplier for Customer click rows (default: 2)")
        q.add_argument("--purchase-weight", type=float, default=ACTION_WEIGHTS["purchase_weight"], help="Train and evaluation loss multiplier for Customer purchase rows (default: 4)")
        q.add_argument("--price-change-weight", type=float, default=ACTION_WEIGHTS["price_change_weight"], help="Train and evaluation loss multiplier for Seller quote changes (default: 2)")
        q.add_argument("--response-format-weight", type=float, default=RESPONSE_FORMAT_WEIGHT,
                       help="Customer Response fixed-format token weight in train/eval; decisions stay 1 (default: 0.25)")
        q.add_argument("--rank", type=int, default=64)
        q.add_argument("--loss-breakdown", action="store_true", help="Report stage and Response token-group evaluation losses")
        q.add_argument("--keep-checkpoints", type=int, default=2, help="Retained complete checkpoints; 0 keeps all")
        q.add_argument("--lr-schedule-epochs", type=int, help="Optional longer learning-rate decay horizon for matched experiments")
        q.add_argument("--batch-size", type=int, default=1)
        q.add_argument("--accumulation", type=int, default=16)
        q.add_argument(
            "--empty-cache-steps", type=int,
            help="Use Transformers' device-cache release interval; no input or loss changes",
        )
        q.add_argument(
            "--train-samples",
            type=int,
            default=0,
            help="0 retains the complete training stream; a positive cap is for smoke checks",
        )
        q.add_argument("--data-workers", type=int, default=0, help="CPU workers for complete actor buckets and cached sampled evaluation")
        q.add_argument("--shuffle-buffer", type=int, default=64)
        q.add_argument("--activation-offload-tokens", type=int,
                       help="Save backward activations on CPU at or above this input length (CUDA)")
        q.add_argument(
            "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
        )
        start = q.add_mutually_exclusive_group()
        start.add_argument("--resume", type=Path, help="Resume the same run from a checkpoint or 'latest'")
        start.add_argument("--init-adapter", type=Path,
                           help="Start a new training run from an existing LoRA adapter")
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    print(json.dumps(check_data(a) if a.command == "check" else train(a), ensure_ascii=False))
