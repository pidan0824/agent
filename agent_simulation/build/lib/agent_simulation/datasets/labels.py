"""Build label pools, read supervised samples and preview their model inputs."""


import argparse
import json
from os import PathLike
import duckdb
import agent_simulation.environment.batching as batching
from contextlib import ExitStack, closing
from tempfile import TemporaryDirectory
from agent_simulation.agents.prompts.customer import messages
from pathlib import Path
from agent_simulation.data.common import require, sha256, literal
from agent_simulation.data.catalog import Catalog
from agent_simulation.inference.requests import load_tokenizer, ContextCapacityError, model_policy
from agent_simulation.datasets.samples.seller import seller_sample, seller_records
from agent_simulation.datasets.samples.customer import customer_sample, split_supervision, customer_records
from agent_simulation.datasets.tokenization import prepare_sample

class SFTDataset:
    """Read full messages with a loaded tokenizer or a local tokenizer path."""

    def __init__(self, directory, tokenizer_path, *, verify_hashes=False):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        require(self.manifest["schema"] == "kuaisearch-on-demand-sft-v1", "Wrong dataset format")
        with ExitStack() as resources:
            self.catalog = resources.enter_context(closing(Catalog(self.manifest["catalog"])))
            require(
                sha256(self.catalog.directory / "manifest.json")
                == self.manifest["catalog_manifest_sha256"],
                "Catalog changed",
            )
            self.tokenizer = (
                load_tokenizer(str(tokenizer_path), local_files_only=True)
                if isinstance(tokenizer_path, (str, PathLike)) else tokenizer_path
            )
            if verify_hashes:
                for f in self.manifest["files"]:
                    require(sha256(self.directory / f["path"]) == f["sha256"], "Label pool changed")
                sources = self.catalog.manifest
                for name, info in sources["files"].items():
                    path = self.catalog.directory / "facts" / (name + ".parquet")
                    require(sha256(path) == info["sha256"], "Source facts changed")
                for info in sources["business_sources"] + sources["split_files"]:
                    require(sha256(Path(info["path"])) == info["sha256"], "Source facts changed")
            self._resources = resources.pop_all()

    def close(self):
        self._resources.close()

    def sample(self, sample_id):
        parts = sample_id.split(":")
        if parts[0] == "seller":
            require(len(parts) == 4, "Invalid Seller sample ID")
            seller, item, cutoff = map(int, parts[1:])
            return self.check_sample(seller_sample(self.catalog, seller, item, cutoff))
        if parts[0] == "query":
            require(len(parts) == 2, "Invalid Query sample ID")
            query, _ = customer_sample(self.catalog, int(parts[1]), stage="query")
            return self.check_sample(query)
        require(
            parts[0] == "response" and len(parts) in (2, 4), "Invalid Customer sample ID"
        )
        _, response = customer_sample(self.catalog, int(parts[1]), stage="response", render=False)
        for batch in split_supervision(response):
            if batch["sample_id"] == sample_id:
                return self.check_sample(batch)
        raise ValueError(
            "Response batch ID is absent; use response_samples for the complete request"
        )

    def check_sample(self, sample):
        sample = prepare_sample(sample, self.tokenizer)
        if not sample["capacity"]["fits_context"]:
            capacity, policy = sample['capacity'], model_policy()
            reserve = max(capacity['answer_tokens'], policy['reserved_output_tokens'])
            raise ContextCapacityError(
                f"Unresolved context overflow: {sample['sample_id']}; "
                f"input={capacity['input_tokens']}, reserve={reserve}, limit={policy['context_limit']}"
            )
        return sample

    def request_samples(self, session_id):
        """Read one historical request once for its Query and all Response batches."""
        query, response = customer_sample(self.catalog, session_id, render=False)
        return checked_record(self, "customer", (query, response))

    def response_samples(self, session_id):
        _, response = customer_sample(self.catalog, session_id, stage="response", render=False)
        return checked_record(self, "customer", (None, response))

    def iter_samples(self, role, *, bucket=None, buckets=32, split=None, stage=None):
        """Traverse every actor bucket by default; an explicit bucket supports workers."""
        require(role in ("seller", "customer"), "Unknown dataset role")
        require(split is None or split in ("train", "test"), "Invalid split")
        require(
            stage is None or stage in (("seller",) if role == "seller" else ("query", "response")),
            "Stage does not belong to this role",
        )
        require(type(buckets) is int and buckets > 0, "Invalid actor partition count")
        require(
            bucket is None or (type(bucket) is int and 0 <= bucket < buckets),
            "Invalid actor bucket",
        )
        selected = range(buckets) if bucket is None else [bucket]
        for current in selected:
            source = (
                seller_records(self.catalog, current, buckets, split=split)
                if role == "seller" else customer_samples(
                    self.catalog, current, buckets, split=split, stage=stage
                )
            )
            for sample in source:
                yield self.check_sample(sample)


def expand_record(role, record):
    if role == "seller":
        yield record
    elif role == "customer":
        query, response = record
        if query is not None:
            if "messages" not in query:
                query = {**query, "messages": messages(query["input"], query["target"])}
            yield query
        if response is not None:
            yield from split_supervision(response)
    else:
        raise ValueError("Unknown supervision role")


def checked_record(data, role, record):
    """Prepare a complete request before returning its samples to a reader."""
    return [data.check_sample(sample) for sample in expand_record(role, record)]


def customer_rows(directory, split, *, response_only=False):
    """Count the rows from fixed Customer batches directly from their labels."""
    labels = Path(directory) / "customer" / split / "labels.parquet"
    size = batching.CUSTOMER_RESPONSE_BATCH_SIZE
    query_rows = 0 if response_only else 1
    with closing(duckdb.connect()) as connection:
        return connection.execute(
            "SELECT coalesce(sum(? + greatest(1, (len(impressed_item_ids) + ? - 1) // ?)), 0) "
            "FROM read_parquet(?)", [query_rows, size, size, str(labels)]
        ).fetchone()[0]


def record_samples(data, role, record_id):
    if role == "seller":
        return [data.sample(record_id)]
    if role != "customer":
        raise ValueError("Unknown selection role")
    return data.request_samples(int(record_id))


def select_record_ids(connection, directory, role, split, count, seed):
    if role not in ("seller", "customer") or count < 0:
        raise ValueError("Invalid record selection")
    path = Path(directory) / role / split / "labels.parquet"
    identity = "sample_id" if role == "seller" else "session_id"
    query = (
        f"SELECT {identity} FROM read_parquet(?) "
        f"ORDER BY sha256(CAST({identity} AS VARCHAR) || ':' || CAST(? AS VARCHAR)),{identity}"
    )
    params = [str(path), seed]
    if count:
        query += " LIMIT ?"
        params.append(count)
    return [row[0] for row in connection.execute(query, params).fetchall()]


def bind_split_history(connection, directory, role, split):
    """Restrict this stream's private connection to history through its last label."""
    path = Path(directory) / role / split / "labels.parquet"
    end = connection.execute(
        "SELECT max(history_cutoff_exclusive) FROM read_parquet(?)", [str(path)]
    ).fetchone()[0]
    if end is None:
        raise ValueError("Empty supervision split")
    database = connection.execute("SELECT current_database()").fetchone()[0]
    database = '"' + database.replace('"', '""') + '"'
    for table in ("requests", "seller_history"):
        connection.execute(
            f"CREATE TEMP VIEW {table} AS SELECT * FROM {database}.main.{table} "
            f"WHERE time_index<={int(end)}"
        )


def customer_samples(catalog, bucket, buckets, *, split=None, stage=None):
    for query, response in customer_records(catalog, bucket, buckets, split=split, stage=stage):
        yield from expand_record("customer", (query, response))


def export(directory, tokenizer_path, sample_ids, output):
    require(not output.exists(), "Choose a new example directory")
    output.mkdir(parents=True)
    data = SFTDataset(directory, tokenizer_path)
    try:
        for sample_id in sample_ids:
            sample = data.sample(sample_id)
            stem = sample_id.replace(":", "_")
            (output / (stem + ".json")).write_text(
                json.dumps(sample, ensure_ascii=False, indent=2) + "\n"
            )
            messages = sample["messages"]
            text = "SYSTEM\n" + messages[0]["content"] + "\n\nUSER\n" + messages[1]["content"]
            text += "\n\nTOOLS\n" + json.dumps(sample["tools"], ensure_ascii=False, indent=2)
            text += "\n\nSUPERVISED ANSWER\n" + json.dumps(
                messages[-1], ensure_ascii=False, indent=2
            )
            (output / (stem + "_model_input.txt")).write_text(text + "\n")
    finally:
        data.close()


def ordered_splits(names):
    names = set(names)
    require(names == {'train', 'test'}, 'Expected official train/test splits')
    return ('train', 'test')


def release_splits(directory):
    counts = json.loads((Path(directory) / 'manifest.json').read_text())['counts']
    splits = ordered_splits(counts['customer'])
    require(set(counts['seller']) == set(splits), 'Role split protocols differ')
    return splits


def selection_split(directory):
    """Official two-split releases use test for selection, as configured for this project."""
    return release_splits(directory)[-1]


def build_labels(catalog_path, output):
    """Save complete supervision pools; contexts remain reconstructible from frozen facts."""
    require(not output.exists(), "Choose a new dataset release")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix="." + output.name + "-") as tmp:
        staged = Path(tmp) / "release"
        staged.mkdir()
        with closing(Catalog(catalog_path)) as catalog:
            counts = _write_labels(catalog, staged)
        staged.rename(output)
    return counts


def _write_labels(catalog, output):
    from agent_simulation.logs import write_report

    select = {
        "seller": """SELECT 'seller:' || t.seller_id || ':' || t.item_id || ':' || t.time_index sample_id,
            t.assigned_split split, t.seller_id, t.item_id, t.time_index history_cutoff_exclusive,
            h.price_raw target_price_raw, h.price_status
            FROM seller_targets t LEFT JOIN seller_history h USING(seller_id,item_id,time_index)""",
        "customer": """SELECT r.session_id,r.user_id,r.time_index history_cutoff_exclusive,s.assigned_split split,
            r.query query_target,r.impressed_item_ids,r.clicked_item_ids,r.purchased_item_ids,
            r.split source_split FROM customer_splits s LEFT JOIN requests r USING(session_id)""",
    }
    files, counts = [], {}
    splits = ordered_splits(r[0] for r in catalog.connection.execute(
        'SELECT DISTINCT assigned_split FROM customer_splits').fetchall())
    for role, query in select.items():
        index, split_key = (("seller_targets", "seller_id,item_id,time_index") if role == "seller"
                            else ("customer_splits", "session_id"))
        invalid_split = catalog.connection.execute(f"""
            SELECT 1 FROM {index} GROUP BY {split_key}
            HAVING count(*) != 1 OR count(*) FILTER (
                WHERE assigned_split IN ('train','test')) != 1 LIMIT 1
        """).fetchone()
        require(invalid_split is None, "Invalid or duplicate supervision split assignment")
        counts[role] = {}
        for split in splits:
            directory = output / role / split
            directory.mkdir(parents=True)
            path = directory / "labels.parquet"
            if role == "seller":
                order = "seller_id,history_cutoff_exclusive,item_id"
            else:
                order = "user_id,history_cutoff_exclusive,session_id"
            expected = f"SELECT * FROM ({query}) WHERE split='{split}'"
            n, = catalog.connection.execute(
                f"COPY ({expected} ORDER BY {order}) TO {literal(path)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 8192)"
            ).fetchone()
            if role == "seller":
                key = "sample_id"
                invalid = """price_status IS DISTINCT FROM 'observed' OR target_price_raw IS NULL
                    OR NOT isfinite(target_price_raw) OR target_price_raw<=0"""
            else:
                key = "session_id"
                invalid = """query_target IS NULL OR impressed_item_ids IS NULL OR clicked_item_ids IS NULL
                    OR purchased_item_ids IS NULL OR NOT list_has_all(impressed_item_ids,clicked_item_ids)
                    OR NOT list_has_all(impressed_item_ids,purchased_item_ids)"""
            stored = f"SELECT * FROM read_parquet({literal(path)})"
            rows, unique, bad = catalog.connection.execute(f"""
                SELECT count(*), count(DISTINCT {key}), count(*) FILTER (WHERE {invalid})
                FROM ({stored})
            """).fetchone()
            require(rows == n and unique == n and bad == 0, "Invalid, duplicate or missing label")
            counts[role][split] = n
            files.append(
                {
                    "path": str(path.relative_to(output)),
                    "rows": n,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_report(
        output / "manifest.json",
        {
            "schema": "kuaisearch-on-demand-sft-v1",
            "complete_label_pools": True,
            "all_prompts_materialized": False,
            "all_samples_capacity_checked": False,
            "dataset": catalog.config.dataset,
            "catalog": str(catalog.directory),
            "catalog_manifest_sha256": sha256(catalog.directory / "manifest.json"),
            "counts": counts,
            "files": files,
            "stages": {
                "customer": "Each request supplies one Query and one or more label-blind Response batches.",
                "seller": "One finite positive price label per item-time with a strictly earlier valid observed price.",
            },
            "history": "The frozen catalog is required; use SFTDataset to reconstruct full messages and tools.",
            "inference": False,
            "training": False,
        },
    )
    return counts


def main():
    parser = argparse.ArgumentParser(description="Build SFT labels or preview selected samples.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Export labels from the fact catalog")
    build.add_argument("--catalog", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    preview = commands.add_parser("preview", help="Read model inputs and answers")
    preview.add_argument("--dataset", type=Path, required=True)
    preview.add_argument("--tokenizer", type=Path, required=True)
    preview.add_argument("--sample-id", action="append", required=True)
    preview.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_labels(args.catalog.resolve(), args.output.resolve()), ensure_ascii=False))
    else:
        export(args.dataset, args.tokenizer, args.sample_id, args.output)


if __name__ == "__main__":
    main()
