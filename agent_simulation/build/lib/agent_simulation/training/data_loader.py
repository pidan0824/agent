"""Stream frozen inputs and tensorize official-template answer-only batches."""


import json
import random
import torch
from contextlib import closing
from itertools import islice
from pathlib import Path
from agent_simulation.data.common import require, sha256
from agent_simulation.datasets.tokenization import IGNORE_INDEX, collate
from agent_simulation.datasets.labels import bind_split_history, SFTDataset, record_samples
from agent_simulation.datasets.tokens import PreparedRow, read_shard

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
        # Count labels and fixed partitions only; never invoke the input iterator.
        if self.role == "seller" or self.stage == "query":
            count = label_count
        else:
            from agent_simulation.datasets.labels import customer_rows
            count = customer_rows(self.directory, self.split, response_only=self.stage == 'response')
        return min(self.limit, count) if self.limit else count

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

    def __init__(self, root, role, split, *, seed=42, shuffle_buffer=0, verify=False):
        super().__init__()
        self.root, self.seed, self.epoch = root / role / split, seed, 0
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
                self.size += f["rows"]

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
        rows = (sample for path in files for sample in read_shard(path))
        yield from _buffered_shuffle(rows, self.shuffle_buffer, rng)


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


class TensorCollator:
    """Keep IDs and provenance outside model kwargs; preserve request weights."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

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
            return {"input_ids": ids, "attention_mask": mask, "labels": labels,
                    "sample_weights": torch.tensor([s.weight for s in samples], dtype=torch.float32)}
        batch = collate(samples, self.tokenizer)
        result = {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in batch["model_inputs"].items()
        }
        result["sample_weights"] = torch.tensor(batch["sample_weights"], dtype=torch.float32)
        return result
