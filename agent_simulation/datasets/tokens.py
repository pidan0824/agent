"""Read, write and resume Token shards bound to their dataset and selection."""
import argparse
import fcntl
import json
import multiprocessing
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agent_simulation.data.common import literal, require, sha256
from agent_simulation.datasets.labels import (
    SFTDataset, expand_record, customer_records, seller_records,
    ordered_splits, release_splits,
)
from agent_simulation.datasets.tokenization import encode_tokens
from agent_simulation.environment.batching import response_batch_policy
from agent_simulation.inference import resolve_device
from agent_simulation.inference.requests import ContextCapacityError, load_tokenizer, tokenizer_identity
from agent_simulation.logs import write_report


def cache_tokenizer_digest(path):
    tokenizer = load_tokenizer(str(path), local_files_only=True)
    return tokenizer_identity(tokenizer, tokenizer.chat_template)


def selected_identity(directory, selection, selection_hash, seed, buckets, *, tokenizer_digest=None):
    require(selection.get('schema') == 'sft-split-candidates-v2', 'Expected split candidate manifest')
    require(seed == selection['seed'], 'Candidate seed differs; use the frozen selection seed')
    partitions = [ordered_splits(entries) for entries in selection['roles'].values()]
    require(partitions and all(p == partitions[0] for p in partitions), 'Role candidate splits differ')
    spec = dict(format=1, dataset=sha256(directory / "manifest.json"), seed=seed,
                validation_records=0, buckets=buckets)
    spec.update(selection_sha256=selection_hash, coverage='selected_complete_records',
                capacity_exclusion='whole_selected_record_only', selection_splits=list(partitions[0]))
    if tokenizer_digest is not None:
        spec['tokenizer_sha256'] = tokenizer_digest
    return spec


def validate_release(root, directory, seed, *, role=None, tokenizer_path=None):
    """Read and verify the cache identity, selection and completion markers."""
    spec = json.loads((root / 'identity.json').read_text())
    seed = spec['seed'] if seed is None else seed
    path = root / 'candidate_manifest.json'
    selection_hash = sha256(path)
    require(selection_hash == spec['selection_sha256'], 'Selected cache manifest changed')
    selection = json.loads(path.read_text())
    digest = spec.get('tokenizer_sha256')
    if digest is not None and tokenizer_path is not None:
        require(digest == cache_tokenizer_digest(tokenizer_path), 'Token cache tokenizer or chat template differs')
    expected = selected_identity(directory, selection, selection_hash, seed, spec['buckets'],
                                 tokenizer_digest=digest)
    require(selection['dataset_manifest_sha256'] == expected['dataset'], 'Selected cache source differs')
    for key in ('background_sha256', 'compact_customer_background', 'customer_response_batching'):
        if key in spec:
            expected[key] = spec[key]
    require(spec == expected,
            'Token cache dataset, seed or candidate selection differs')
    complete = root / 'complete.json'
    if role is None or complete.exists():
        require(json.loads(complete.read_text()) == expected,
                'Token cache completion identity differs')
    if role is not None:
        require(role in ('customer', 'seller'), 'Invalid training role')
        splits = spec['selection_splits']
        require(all((root / role / split / f'{bucket:03d}' / 'complete.json').is_file()
                    for split in splits for bucket in range(spec['buckets'])),
                'Token cache role has unfinished buckets: ' + role)
    return spec


@dataclass
class PreparedRow:
    sample_id: str
    tokens: np.ndarray
    answer_start: int
    weight: float

    @classmethod
    def from_sample(cls, sample, tokenizer):
        row = encode_tokens(sample, tokenizer)
        return cls(row["sample_id"], np.asarray(row["input_ids"], dtype=np.int32),
                   row["answer_start"], row["sample_weight"])


def read_shard(path):
    with np.load(path, allow_pickle=False) as block:
        ids, tokens, ends, starts, weights = (block[k] for k in ("ids", "tokens", "ends", "starts", "weights"))
    begin = 0
    for sid, end, start, weight in zip(ids, ends, starts, weights, strict=True):
        yield PreparedRow(str(sid), tokens[begin:end], int(start), float(weight))
        begin = end


def write_shard(path, rows):
    # Keep token IDs contiguous in sample order so read_shard can return slices.
    arrays = dict(ids=np.array([r.sample_id for r in rows]), tokens=np.concatenate([r.tokens for r in rows]),
                  ends=np.cumsum([len(r.tokens) for r in rows]), starts=np.array([r.answer_start for r in rows]),
                  weights=np.array([r.weight for r in rows]))
    partial = path.with_suffix(".partial")
    with partial.open("wb") as dest:
        np.savez_compressed(dest, **arrays)
    partial.replace(path)


class RecordWriter:
    def __init__(self, args, role, split, bucket, expected=None):
        self.dest = args.output / role / split / f'{bucket:03d}'
        self.dest.mkdir(parents=True, exist_ok=True)
        self.role, self.split, self.expected = role, split, expected
        self.reserve_gib = args.reserve_gib
        self.files, self.buffer, self.accepted, self.excluded = [], [], [], []
        self.total = self.seen = 0
        self.failures = self.dest / 'failures.jsonl'
        checkpoint = self.dest / 'progress.json'
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            require('files' in saved and 'accepted' in saved,
                    'Old partial cache has no resumable checkpoint; use a new output directory')
            self.files, self.accepted = saved['files'], saved['accepted']
            self.total = saved['rows']
            for entry in self.files:
                path = self.dest / entry['file']
                require(path.is_file(), 'Missing committed shard: ' + str(path))
                if getattr(args, 'verify', False):
                    require(sha256(path) == entry['sha256'], 'Committed shard changed')
        if self.failures.exists():
            self.excluded = [json.loads(line) for line in self.failures.read_text().splitlines()]
        self.seen = len(self.accepted) + len(self.excluded)
        self.restored = set(self.accepted) | {f['record_id'] for f in self.excluded}

    def add(self, record, rows):
        self.seen += 1
        self.accepted.append(record)
        self.buffer.extend(rows)
        if len(self.buffer) >= (16 if not self.files else 256):
            self.flush()

    def reject(self, record, error):
        self.seen += 1
        failure = dict(record_id=record, role=self.role, split=self.split,
                       reason='context_capacity', detail=str(error))
        self.excluded.append(failure)
        with self.failures.open('a') as log:
            log.write(json.dumps(failure, ensure_ascii=False) + '\n')

    def flush(self):
        if self.buffer:
            require(shutil.disk_usage(self.dest).free > self.reserve_gib * 1024**3, 'Disk reserve reached')
            path = self.dest / f'{len(self.files):06d}.npz'
            # A shard without an atomic checkpoint is uncommitted and can be replaced.
            write_shard(path, self.buffer)
            self.total += len(self.buffer)
            self.files.append(dict(file=path.name, sha256=sha256(path), rows=len(self.buffer)))
        write_report(self.dest / 'progress.json', dict(rows=self.total, records=self.seen,
                    total_records=self.expected, accepted_records=len(self.accepted),
                    excluded_records=len(self.excluded),
                    shards=len(self.files), files=self.files, accepted=self.accepted))
        self.buffer.clear()

    def finish(self):
        self.flush()
        require(self.seen == self.expected, 'Incomplete candidate coverage')
        write_report(self.dest / 'selection.json', dict(accepted=self.accepted,
                    excluded=self.excluded, candidate_records=self.seen))
        result = dict(rows=self.total, records=len(self.accepted), candidate_records=self.seen,
                      excluded_records=len(self.excluded),
                      files=self.files, selection_sha256=sha256(self.dest / 'selection.json'))
        write_report(self.dest / 'complete.json', result)
        return result


def initialize_vector_cache():
    """Initialize SQLite once in the parent, before any encoding workers start."""
    from agent_simulation.agents.retrieval.bge_encoder import default_encoder
    cache = default_encoder()[0].disk_cache
    cache.path.parent.mkdir(parents=True, exist_ok=True)
    with cache.path.with_suffix('.init.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _ = cache.connection


# Candidate selection and chronological history.

def bind_split_candidates(connection, paths, role, bucket, buckets, restored_ids=()):
    """Bind any requested splits through one chronological history scan."""
    selects = [f"SELECT *, '{split}' AS cache_split FROM read_parquet({literal(path)})"
               for split, path in paths.items()]
    connection.execute('CREATE TEMP TABLE selected_records AS SELECT * FROM (' +
                       ' UNION ALL '.join(selects) + ') WHERE actor_id % ? = ?', [buckets, bucket])
    order = bind_selected_history(connection, role, restored_ids)
    return [(str(record), split) for record, split in connection.execute(
        f'SELECT record_id,cache_split FROM selected_records ORDER BY {order}').fetchall()]


def bind_selected_history(connection, role, restored_ids):
    connection.execute('CREATE TEMP TABLE restored_records(record_id VARCHAR PRIMARY KEY)')
    if restored_ids:
        connection.executemany('INSERT INTO restored_records VALUES (?)', [(str(i),) for i in restored_ids])
    connection.execute('CREATE TEMP VIEW pending_records AS SELECT * FROM selected_records WHERE record_id NOT IN (SELECT record_id FROM restored_records)')
    owner_column, table, index = (
        ('user_id', 'requests', 'customer_splits') if role == 'customer'
        else ('seller_id', 'seller_history', 'seller_targets')
    )
    database=connection.execute('SELECT current_database()').fetchone()[0].replace('"','""')
    base=f'"{database}".main'
    history = f'{base}.{table}'
    if role == 'seller':
        from agent_simulation.data.market.quotes import ALIAS, TABLES
        if connection.execute('SELECT count(*) FROM duckdb_databases() WHERE database_name=?',
                              [ALIAS]).fetchone()[0]:
            # The optional market cache provides the same business facts.
            # Its seller ordering avoids decoding every source Parquet row.
            history = f'{ALIAS}.{TABLES["seller"]}'
    connection.execute(f'''CREATE TEMP VIEW {table} AS SELECT h.* FROM {history} h
        JOIN (SELECT actor_id,max(history_cutoff_exclusive) cutoff FROM pending_records GROUP BY actor_id) s
        ON h.{owner_column}=s.actor_id AND h.time_index<=s.cutoff''')
    if role=='customer':
        # Non-selected requests still join and update memory; only their labels disappear.
        connection.execute(f'''CREATE TEMP VIEW {index} AS SELECT * REPLACE (
            CASE WHEN session_id IN (SELECT session_id FROM pending_records) THEN assigned_split END AS assigned_split
            ) FROM {base}.{index}''')
        order='actor_id,history_cutoff_exclusive,session_id'
    else:
        connection.execute(f'''CREATE TEMP VIEW {index} AS SELECT t.* FROM {base}.{index} t
            JOIN pending_records s ON t.seller_id=s.seller_id AND t.item_id=s.item_id
            AND t.time_index=s.history_cutoff_exclusive''')
        order='actor_id,history_cutoff_exclusive,item_id'
    return order


# Record construction, failure logging and bucket output.

def build_splits(args, role, bucket, splits):
    results, paths = {}, {}
    for split in splits:
        complete = args.output / role / split / f'{bucket:03d}' / 'complete.json'
        if complete.exists():
            results[split] = json.loads(complete.read_text())
        else:
            paths[split] = args.candidates.parent / f'{role}-{split}-candidates.parquet'
    if not paths:
        return results
    writers = {split: RecordWriter(args, role, split, bucket) for split in paths}
    restored = set().union(*(writer.restored for writer in writers.values()))
    with closing(SFTDataset(args.dataset, args.tokenizer)) as data:
        connection = data.catalog.connection
        connection.execute('SET enable_progress_bar=false')
        if role == 'seller' and getattr(args, 'market_cache', None):
            from agent_simulation.data.market.quotes import attach_cache
            attach_cache(connection, data.catalog.config, args.market_cache, verify_hashes=False)
        expected = bind_split_candidates(connection, paths, role, bucket, args.buckets, restored)
        counts = Counter(split for _, split in expected)
        for split, writer in writers.items():
            writer.expected = counts[split]
            require(writer.restored <= {r for r, s in expected if s == split},
                    'Checkpoint records differ from frozen candidates')
        if role == 'customer':
            # Token shards only need the final model payload and sample weight.
            source = customer_records(data.catalog, bucket, args.buckets, include_history_ids=False)
        else:
            source = seller_records(data.catalog, bucket, args.buckets, split=None)
        for record, split in expected:
            writer = writers[split]
            if record in writer.restored:
                continue
            raw = next(source, None)
            require(raw is not None, 'Missing selected record')
            actual = raw['sample_id'] if role == 'seller' else str(raw[0]['key']['session_id'])
            require(actual == record, 'Selected label sequence differs')
            try:
                rows = []
                for sample in expand_record(
                    role, raw, tokenizer=data.tokenizer,
                    compact_background=getattr(args, 'compact_customer_background', False),
                ):
                    require(sample['split'] == split, 'Wrong selected split')
                    rows.append(PreparedRow.from_sample(sample, data.tokenizer))
            except ContextCapacityError as error:
                writer.reject(record, error)
                continue
            # Publish only after Query and every Response batch have encoded successfully.
            writer.add(record, rows)
        require(next(source, None) is None, 'Unexpected extra selected record')
        results.update({split: writer.finish() for split, writer in writers.items()})
    return results


# One-time preflight, release binding and worker scheduling.

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'tokenizer', 'candidates', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--market-cache', type=Path,
                        help='Immutable Seller query cache')
    parser.add_argument('--verify', action='store_true',
                        help='Verify source and candidate checksums, including resumed shards')
    parser.add_argument('--role', choices=('customer', 'seller'))
    parser.add_argument('--compact-customer-background', action='store_true',
                        help='Keep ID, title, brand and leaf category in Customer background item descriptions')
    parser.add_argument('--workers', type=int,
                        help='Default: 1 on GPU or 24 on CPU; workers each load their own encoder')
    parser.add_argument('--buckets', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reserve-gib', type=int, default=8)
    args = parser.parse_args()
    if args.workers is None:
        device = resolve_device(os.environ.get('MEMORY_BGE_DEVICE', 'auto'))
        args.workers = 24 if device.type == 'cpu' else 1
    require(min(args.workers, args.buckets, args.reserve_gib) > 0,
            'Positive settings required')
    return args


def preflight(args):
    """Bind frozen inputs before workers write shards, with optional full-file checks."""
    verify = getattr(args, 'verify', False)
    selection = json.loads(args.candidates.read_text())
    spec = selected_identity(args.dataset, selection, sha256(args.candidates),
                             args.seed, args.buckets)
    splits = release_splits(args.dataset)
    require(selection['dataset_manifest_sha256'] == spec['dataset'],
            'Candidate source release changed')
    for role, entries in selection['roles'].items():
        require(set(entries) == set(splits), 'Missing candidate split')
        for split, entry in entries.items():
            require(entry['file'] == f'{role}-{split}-candidates.parquet',
                    'Unexpected candidate filename')
            if verify:
                require(sha256(args.candidates.parent / entry['file']) == entry['sha256'],
                        'Candidate file changed')
    if verify or args.market_cache:
        # This step only reads data files; workers load the tokenizer when encoding.
        with closing(SFTDataset(args.dataset, None, verify_hashes=verify)) as data:
            if args.market_cache:
                from agent_simulation.data.market.quotes import attach_cache
                sources = {Path(entry["path"]).name: entry["sha256"]
                           for entry in data.catalog.manifest["business_sources"]}
                market = json.loads((args.market_cache / "manifest.json").read_text())
                require(market["sources"] == sources, "Market cache source release differs")
                attach_cache(data.catalog.connection, data.catalog.config, args.market_cache, verify_hashes=verify)
    catalog = Path(json.loads((args.dataset / 'manifest.json').read_text())['catalog'])
    spec['background_sha256'] = sha256(catalog / 'facts/customer_background.parquet')
    spec['tokenizer_sha256'] = cache_tokenizer_digest(args.tokenizer)
    spec['customer_response_batching'] = response_batch_policy()
    if getattr(args, 'compact_customer_background', False):
        spec['compact_customer_background'] = True
    return spec, splits


def bind_release(args, spec):
    header = args.output / 'identity.json'
    if header.exists():
        saved = json.loads(header.read_text())
        require(saved.get('tokenizer_sha256') == spec['tokenizer_sha256'],
                'Token cache tokenizer or chat template differs or was not recorded; choose a new output directory')
        if getattr(args, 'role', None) == 'seller' and 'customer_response_batching' not in saved:
            spec.pop('customer_response_batching', None)
        require(saved == spec, 'Different selected cache release; choose a new output directory')
    else:
        write_report(header, spec)
    copied = args.output / 'candidate_manifest.json'
    if copied.exists():
        require(sha256(copied) == spec['selection_sha256'], 'Copied candidate manifest changed')
    else:
        shutil.copyfile(args.candidates, copied)


def run_jobs(args, splits):
    roles = (args.role,) if args.role else ('customer', 'seller')
    jobs = {
        role: [bucket for bucket in range(args.buckets)
               if any(not (args.output / role / split / f'{bucket:03d}' / 'complete.json').is_file()
                      for split in splits)]
        for role in roles
    }
    if not any(jobs.values()):
        return
    initialize_vector_cache()
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for role, buckets in jobs.items():
            pending = {pool.submit(build_splits, args, role, bucket, splits): bucket
                       for bucket in buckets}
            for future in as_completed(pending):
                for split, value in future.result().items():
                    print(role, split, pending[future],
                          {k: v for k, v in value.items() if k != 'files'}, flush=True)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec, splits = preflight(args)
        bind_release(args, spec)
        run_jobs(args, splits)
        if all((args.output / role / split / f'{bucket:03d}' / 'complete.json').is_file()
               for role in ('customer', 'seller') for split in splits for bucket in range(args.buckets)):
            write_report(args.output / 'complete.json', spec)


if __name__ == '__main__':
    main()
