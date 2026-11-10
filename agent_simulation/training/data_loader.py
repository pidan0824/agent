"""Stream frozen inputs and tensorize official-template answer-only batches."""


import json
import re
import numpy as np
import random
import sys
import time
import torch
from contextlib import closing
from itertools import islice
from pathlib import Path
from agent_simulation.data.common import require, sha256
from agent_simulation.datasets.tokenization import IGNORE_INDEX, collate
from agent_simulation.datasets.labels import bind_split_history, SFTDataset, record_samples
from agent_simulation.datasets.tokens import PreparedRow, read_shard

ACTION_WEIGHTS = {"click_weight": 2.0, "purchase_weight": 4.0, "price_change_weight": 2.0}


class WorkerRecords(torch.utils.data.IterableDataset):
    """Each worker owns whole actor buckets, or disjoint complete evaluation records."""

    def __init__(self, directory, tokenizer, role, split, stage=None, records=None):
        self.directory, self.tokenizer = directory, tokenizer
        self.role, self.split, self.stage, self.records = role, split, stage, records

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        with closing(SFTDataset(self.directory, self.tokenizer)) as data:
            if self.records is not None:
                for record in self.records[worker.id::worker.num_workers]:
                    group = record_samples(data, self.role, record)
                    if not group or any(s["split"] != self.split for s in group):
                        raise ValueError("Wrong split or empty evaluation record")
                    yield [PreparedRow.from_sample(s, data.tokenizer) for s in group]
                return
            bind_split_history(data.catalog.connection, self.directory, self.role, self.split)
            for bucket in range(worker.id, 32, worker.num_workers):
                for sample in data.iter_samples(self.role, bucket=bucket, buckets=32,
                                                split=self.split, stage=self.stage):
                    yield PreparedRow.from_sample(sample, data.tokenizer)


def _identity(value):
    return value


def worker_loader(dataset, workers):
    if not 1 <= workers <= 32:
        raise ValueError("Data workers must be between 1 and 32")
    # Spawn isolates DuckDB, tokenizer and CUDA state. Keep the Trainer's main
    # loader single-process so shuffling, batching and resume skip stay global.
    return torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=workers,
                                       multiprocessing_context="spawn", prefetch_factor=2,
                                       collate_fn=_identity, generator=torch.Generator().manual_seed(0))


class SampleStream(torch.utils.data.IterableDataset):
    """Open a private read-only catalog per iterator; never materialize the label pool."""

    def __init__(
        self, directory, tokenizer, role, split, *, limit=0, stage=None, shuffle_buffer=0, seed=42, workers=0
    ):
        super().__init__()
        if role not in ("seller", "customer") or split not in ("train", "validation", "test"):
            raise ValueError("Unknown role or split")
        if stage not in (None, "query", "response", "seller") or limit < 0 or shuffle_buffer < 0:
            raise ValueError("Invalid stream selection")
        self.directory, self.tokenizer, self.role, self.split = directory, tokenizer, role, split
        self.limit, self.stage = limit, stage
        self.shuffle_buffer, self.seed, self.epoch = shuffle_buffer, seed, 0
        self.workers = workers
        self.epoch_size = None

    def __len__(self):
        if self.epoch_size is None:
            raise TypeError("Stream length has not been measured")
        return self.epoch_size

    def measure_epoch(self, label_count):
        if self.role == "seller" or self.stage == "query":
            return min(self.limit, label_count) if self.limit else label_count
        # Response row counts depend on the rendered contexts. Token caches already store them.
        return sum(1 for _ in self)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError("Use dataloader_num_workers=0; a stream must not be duplicated")
        if self.workers:
            source = worker_loader(WorkerRecords(self.directory, self.tokenizer, self.role,
                                                  self.split, self.stage), self.workers)
            yield from self._shuffle(iter(source))
            return
        with closing(SFTDataset(self.directory, self.tokenizer)) as data:
            bind_split_history(data.catalog.connection, self.directory, self.role, self.split)
            yield from self._shuffle(data.iter_samples(self.role, split=self.split, stage=self.stage))

    def _shuffle(self, source):
        if self.limit:
            source = islice(source, self.limit)
        yield from _buffered_shuffle(source, self.shuffle_buffer, random.Random(self.seed + self.epoch))


def _buffered_shuffle(source, size, rng):
    if not size:
        yield from source
        return
    buffer = []
    for sample in source:
        if len(buffer) < size:
            buffer.append(sample)
        else:
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = sample
    rng.shuffle(buffer)
    yield from buffer


class TokenStream(torch.utils.data.IterableDataset):
    """Read prepared NPZ shards in deterministic epoch order."""

    def __init__(self, root, role, split, *, seed=42, shuffle_buffer=0, verify=False, stage=None):
        super().__init__()
        self.root, self.seed, self.epoch = root / role / split, seed, 0
        self.stage = stage
        self.shuffle_buffer = shuffle_buffer
        spec = json.loads((root / "identity.json").read_text())
        self.files = []
        self.size = 0
        for bucket in range(spec["buckets"]):
            base = self.root / f"{bucket:03d}"
            manifest = json.loads((base / "complete.json").read_text())
            if verify and 'selection_sha256' in manifest:
                require(sha256(base/'selection.json') == manifest['selection_sha256'], 'Selected record audit changed')
            for f in manifest["files"]:
                path = base / f["file"]
                require(path.is_file(), "Missing token shard: " + str(path))
                if verify:
                    require(sha256(path) == f["sha256"], "Changed token shard: " + str(path))
                self.files.append(path)
                if stage is None:
                    self.size += f["rows"]
                else:
                    with np.load(path, allow_pickle=False) as shard:
                        self.size += sum(str(sid).startswith(stage + ":") for sid in shard["ids"])

    def __len__(self):
        return self.size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        require(torch.utils.data.get_worker_info() is None, "Use one reader for cached tokens")
        rng = random.Random(self.seed + self.epoch)
        files = list(self.files)
        if self.shuffle_buffer:
            rng.shuffle(files)
        rows = (sample for path in files for sample in read_shard(path)
                if self.stage is None or sample.sample_id.startswith(self.stage + ":"))
        yield from _buffered_shuffle(rows, self.shuffle_buffer, rng)


def action_multipliers(args, tokenizer, train_data):
    """Index positive observed actions once from the frozen training stream."""
    if args.role == "customer":
        click = getattr(args, "click_weight", ACTION_WEIGHTS["click_weight"])
        purchase = getattr(args, "purchase_weight", ACTION_WEIGHTS["purchase_weight"])
        if max(click, purchase) == 1 or train_data.stage == "query":
            return {}
        result = {}
        for row in train_data:
            if not row.sample_id.startswith("response:"):
                continue
            answer = json.loads(tokenizer.decode(row.tokens[row.answer_start:], skip_special_tokens=True))
            factor = max(click if answer["clicked_item_ids"] else 1.0,
                         purchase if answer["purchased_item_ids"] else 1.0)
            if factor > 1:
                result[row.sample_id] = factor
        return result

    factor = getattr(args, "price_change_weight", ACTION_WEIGHTS["price_change_weight"])
    if factor == 1:
        return {}
    keys = []
    for path in train_data.files:
        with np.load(path, allow_pickle=False) as shard:
            keys.extend(tuple(map(int, sid.split(":")[1:])) for sid in shard["ids"])
    sellers, items, times = map(list, zip(*keys))
    with closing(SFTDataset(args.dataset, None)) as data:
        from agent_simulation.datasets.sampling import changed_seller_records
        rows = changed_seller_records(data.catalog.connection, """
            SELECT unnest(?::BIGINT[]) seller_id, unnest(?::BIGINT[]) item_id,
                   unnest(?::BIGINT[]) time_index
        """, [sellers, items, times])
    return {f"seller:{seller}:{item}:{time}": factor for seller, item, time in rows}


class EvaluationStream(torch.utils.data.IterableDataset):
    """Read the fixed list of complete records chosen at training startup."""

    def __init__(self, directory, tokenizer, role, split, records):
        self.directory, self.tokenizer, self.role = directory, tokenizer, role
        self.split, self.records = split, records

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError("Use dataloader_num_workers=0; evaluation must not be duplicated")
        with closing(SFTDataset(self.directory, self.tokenizer)) as data:
            for record in self.records:
                for sample in record_samples(data, self.role, record):
                    if sample["split"] != self.split:
                        raise ValueError("Wrong split in evaluation selection")
                    yield sample


def response_token_parts(tokenizer, tokens):
    """0: structure (including terminator), 1: click ID, 2: purchase ID."""
    text = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != list(tokens):
        raise ValueError("Cannot align answer tokens for loss breakdown")
    spans = []
    for field in re.finditer(r'"(clicked_item_ids|purchased_item_ids)"\s*:\s*\[([^]]*)\]', text):
        part = 1 if field[1] == "clicked_item_ids" else 2
        spans.extend((field.start(2) + m.start(), field.start(2) + m.end(), part)
                     for m in re.finditer(r"\d+", field[2]))
    return [next((part for left, right, part in spans if start < right and end > left), 0)
            for start, end in encoded["offset_mapping"]]


class TensorCollator:
    """Keep IDs and provenance outside model kwargs; preserve request weights."""

    def __init__(self, tokenizer, *, loss_breakdown=False, action_weights=None):
        self.tokenizer = tokenizer
        self.loss_breakdown = loss_breakdown
        self.action_weights = action_weights or {}

    def __call__(self, samples):
        if isinstance(samples[0], PreparedRow):
            length = max(len(s.tokens) for s in samples)
            ids = torch.full((len(samples), length), self.tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            labels = torch.full_like(ids, IGNORE_INDEX)
            for i, sample in enumerate(samples):
                n, start = len(sample.tokens), sample.answer_start
                ids[i, :n] = torch.from_numpy(sample.tokens)
                mask[i, :n] = 1
                labels[i, start:n] = ids[i, start:n]
            result = {"input_ids": ids, "attention_mask": mask, "labels": labels,
                      "sample_weights": torch.tensor([s.weight for s in samples], dtype=torch.float32)}
            if self.action_weights:
                result["action_multipliers"] = torch.tensor([
                    self.action_weights.get(sample.sample_id, 1.0) for sample in samples
                ], dtype=torch.float32)
            if self.loss_breakdown:
                stages = [s.sample_id.split(":", 1)[0] for s in samples]
                parts = torch.full_like(ids, -1)
                for i, sample in enumerate(samples):
                    start, n = sample.answer_start, len(sample.tokens)
                    values = (response_token_parts(self.tokenizer, sample.tokens[start:])
                              if stages[i] == "response" else [0] * (n - start))
                    parts[i, start:n] = torch.tensor(values)
                result["answer_parts"] = parts
                result["stages"] = torch.tensor([("query", "response", "seller").index(s) for s in stages])
            return result
        batch = collate(samples, self.tokenizer)
        result = {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in batch["model_inputs"].items()
        }
        result["sample_weights"] = torch.tensor(batch["sample_weights"], dtype=torch.float32)
        return result


def prepare_datasets(args, tokenizer, data_check, eval_split, *, resume_epoch_size=None):
    """Prepare the training and held-out streams before loading model weights."""
    if getattr(args, "token_cache", None):
        if args.train_samples or args.data_workers:
            raise ValueError("Use the finished token cache without caps or live workers")
        train_data = TokenStream(args.token_cache, args.role, "train", seed=args.seed,
                                 shuffle_buffer=args.shuffle_buffer, stage=getattr(args, "stage", None))
        evaluation_data = TokenStream(args.token_cache, args.role, eval_split, stage=getattr(args, "stage", None))
        return train_data, evaluation_data, len(train_data)

    workers = getattr(args, "data_workers", 0)
    train_tokenizer = args.tokenizer if workers else tokenizer
    train_data = SampleStream(args.dataset, train_tokenizer, args.role, "train", limit=args.train_samples,
                              shuffle_buffer=args.shuffle_buffer, seed=args.seed,
                              stage=getattr(args, "stage", None), workers=workers)
    samples_per_epoch = (resume_epoch_size if resume_epoch_size is not None else train_data.measure_epoch(
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
    return train_data, evaluation_data, samples_per_epoch
