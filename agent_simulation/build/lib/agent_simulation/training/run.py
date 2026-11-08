"""Check or train separate KuaiSearch Customer/Seller LoRA adapters, locally."""


import argparse
import importlib.metadata
import json
import math
import sys
import time
import torch
from itertools import islice
from pathlib import Path
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, set_seed
from agent_simulation.inference.requests import load_tokenizer
from agent_simulation.inference import resolve_device
from agent_simulation.logs import write_report
from agent_simulation.datasets.labels import SFTDataset, record_samples, select_record_ids, selection_split
from agent_simulation.datasets.tokens import validate_release
from agent_simulation.training.data_loader import (
    SampleStream,
    EvaluationStream,
    WorkerRecords,
    worker_loader,
    TensorCollator,
    TokenStream,
)
from agent_simulation.training.trainer import make_trainer
from agent_simulation.data.common import sha256

def check_cached_data(args, data, shards, *, verify=False):
    """Read cached split counts and optionally verify files and sample rows."""
    spec = validate_release(args.token_cache, args.dataset, args.seed, role=args.role)
    collator = TensorCollator(data.tokenizer) if verify else None
    checks, counts = [], {}
    eval_split = selection_split(args.dataset)
    for split in ("train", eval_split):
        stream = TokenStream(args.token_cache, args.role, split, verify=verify)
        if not len(stream):
            raise ValueError('Empty cached supervision split: ' + split)
        counts[split] = len(stream)
        if verify:
            for sample in islice(stream, 2):
                batch = collator([sample])
                checks.append(dict(sample_id=sample.sample_id, split=split,
                    tokens=batch['input_ids'].shape[1], answer_tokens=batch['labels'].ne(-100).sum().item(),
                    weight=batch['sample_weights'].item()))
    return dict(role=args.role,counts=counts,count_unit='expanded_cached_rows',checks=checks,
        test_used=eval_split == 'test',weights_loaded=False,local_model_shards_present=shards,
        tokenizer_matches_model=True,all_inputs_capacity_checked=False,
        token_cache_identity=spec,evaluation_selection='All evaluation rows in the frozen token cache; no resampling')


def check_data(args):
    """Bind training inputs, optionally scanning files and collating sample rows."""
    verify = args.command == "check" or getattr(args, "verify", False)
    config = json.loads((args.model / "config.json").read_text())
    if (config.get("model_type"), config.get("hidden_size"), config.get("num_hidden_layers")) != (
        "qwen2",
        2048,
        36,
    ):
        raise ValueError("Expected the existing local Qwen2.5-3B-Instruct")
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if not shards or any(Path(p).name != p or not (args.model / p).is_file() for p in shards):
        raise ValueError("Complete local model shards are required")
    for name in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
        model_file, tokenizer_file = args.model / name, args.tokenizer / name
        if not model_file.samefile(tokenizer_file) and sha256(model_file) != sha256(tokenizer_file):
            raise ValueError("Training tokenizer differs from local Qwen: " + name)
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
                    batch = collator([sample])
                    checks.append({
                        "sample_id": sample["sample_id"],
                        "split": split,
                        "tokens": batch["input_ids"].shape[1],
                        "answer_tokens": batch["labels"].ne(-100).sum().item(),
                        "weight": batch["sample_weights"].item(),
                    })
        result = {
            "role": args.role,
            "counts": data.manifest["counts"][args.role],
            "checks": checks,
            "test_used": eval_split == "test",
            "test_used_for_gradient_updates": False,
            "weights_loaded": False,
            "local_model_shards_present": shards,
            "tokenizer_matches_model": True,
            "all_inputs_capacity_checked": False,
        }
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


def train(args):
    if getattr(args, "empty_cache_steps", None) is not None and args.empty_cache_steps < 1:
        raise ValueError("Cache release interval must be positive")
    if min(args.epochs, args.eval_steps, args.batch_size, args.accumulation, args.rank) < 1 or (
        args.max_steps is not None and args.max_steps < 1
    ):
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
    if args.output.exists() and not args.resume:
        raise FileExistsError("Use a new output directory or explicitly resume its checkpoint")
    args.device = resolve_device(args.device).type
    # Bind the current data and model before loading weights.
    data_check = check_data(args)
    eval_split = selection_split(args.dataset)
    identity = training_identity(args, data_check, eval_split)
    packages = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "accelerate", "duckdb", "numpy", "tokenizers")
    }
    checkpoint = None
    if args.resume:
        checkpoint = resume_checkpoint(args.output, args.resume, device=args.device)
        config = json.loads((args.output / "run_config.json").read_text())
        saved_identity = {k: v for k, v in config["identity"].items() if k != "training_code_sha256"}
        if saved_identity != identity:
            raise ValueError("Resume data/model/config differs")
        if config["packages"] != packages:
            raise ValueError("Resume package versions differ")
    tokenizer = load_tokenizer(str(args.tokenizer), local_files_only=True)
    if getattr(args, "token_cache", None):
        if args.train_samples or args.data_workers:
            raise ValueError("Use the finished token cache without caps or live workers")
        train_data = TokenStream(args.token_cache, args.role, "train", seed=args.seed,
                                 shuffle_buffer=args.shuffle_buffer)
        evaluation_data = TokenStream(args.token_cache, args.role, eval_split)
        samples_per_epoch = len(train_data)
    else:
        workers = getattr(args, "data_workers", 0)
        train_tokenizer = args.tokenizer if workers else tokenizer
        train_data = SampleStream(args.dataset, train_tokenizer, args.role, "train", limit=args.train_samples,
                                  shuffle_buffer=args.shuffle_buffer, seed=args.seed,
                                  workers=workers)
        samples_per_epoch = (config["samples_per_epoch"] if checkpoint else train_data.measure_epoch(
            data_check["counts"]["train"]))
        train_data.epoch_size = samples_per_epoch
        records = data_check["evaluation_selection"]["record_ids"]
        if records is None:
            evaluation_data = SampleStream(args.dataset, tokenizer, args.role, eval_split)
        elif not records:
            raise ValueError("Empty evaluation selection")
        elif not workers:
            evaluation_data = EvaluationStream(args.dataset, tokenizer, args.role, eval_split, records)
        else:
            groups = worker_loader(WorkerRecords(args.dataset, args.tokenizer, args.role,
                                                 eval_split, records=records), workers)
            evaluation_data = []
            started = time.monotonic()
            print(f"Preparing {len(records)} evaluation records with {args.data_workers} workers",
                  file=sys.stderr, flush=True)
            for count, group in enumerate(groups, 1):
                evaluation_data.extend(group)
                if count % 16 == 0 or count == len(records):
                    print(f"Evaluation inputs {count}/{len(records)} records, "
                          f"{len(evaluation_data)} samples, {time.monotonic()-started:.1f}s",
                          file=sys.stderr, flush=True)
    if samples_per_epoch == 0:
        raise ValueError("Empty training epoch")
    if not args.resume:
        args.output.mkdir(parents=True)
        write_report(
            args.output / "run_config.json",
            {
                "identity": identity,
                "samples_per_epoch": samples_per_epoch,
                "input_check": data_check,
                "packages": packages,
                "loss": "mean_over_examples(request_weight * mean_answer_token_CE)",
                "checkpoint_selection": f"{eval_split} weighted answer loss; no evaluation-label gradient updates",
                "evaluation_split": eval_split,
                "lora_reference": "PolicySim rank 64; targets/alpha/dropout are explicit project choices",
            },
        )
    try:
        set_seed(args.seed)
        dtype = getattr(torch, args.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=args.rank,
                lora_alpha=2 * args.rank,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        model.print_trainable_parameters()
        trainer = make_trainer(model, tokenizer, args, train_data, evaluation_data)
        initial_metrics = trainer.evaluate() if not args.resume else None
        if initial_metrics is not None:
            write_report(args.output / "initial_evaluation.json", initial_metrics)
        result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        if trainer.state.best_model_checkpoint is None or not math.isfinite(trainer.state.best_metric):
            raise ValueError("Cannot export an adapter without a finite best evaluation loss")
        metrics = trainer.evaluate()
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
                "initial_evaluation": initial_metrics,
                "best_checkpoint": trainer.state.best_model_checkpoint,
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
        q.add_argument("--model", type=Path, required=True)
        q.add_argument(
            "--tokenizer",
            type=Path,
            default=Path(__file__).resolve().parents[2] / "models/Qwen2.5-3B-Instruct-tokenizer",
        )
        q.add_argument("--role", choices=("seller", "customer"), required=True)
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
        q.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
        budget = q.add_mutually_exclusive_group()
        budget.add_argument("--epochs", type=int, default=10, help="Complete passes over the training stream (default: 10)")
        budget.add_argument("--max-steps", type=int, help="Explicit optimizer-step budget for bounded runs")
        q.add_argument("--save-steps", "--eval-steps", dest="eval_steps", type=int, default=100,
                       help="Checkpoint and model-selection evaluation interval")
        q.add_argument("--learning-rate", type=float, default=1e-6)
        q.add_argument("--rank", type=int, default=64)
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
        q.add_argument(
            "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
        )
        q.add_argument("--resume", type=Path, help="Checkpoint directory, or 'latest' for the newest complete checkpoint")
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    print(json.dumps(check_data(a) if a.command == "check" else train(a), ensure_ascii=False))
