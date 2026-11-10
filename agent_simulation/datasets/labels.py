"""Build label pools, read supervised samples and preview their model inputs."""


import argparse
import json
from os import PathLike
from contextlib import ExitStack, closing
from tempfile import TemporaryDirectory
from agent_simulation.agents.prompts.customer import messages
from pathlib import Path
from agent_simulation.data.common import require, sha256
from agent_simulation.data.catalog import Catalog
from agent_simulation.datasets.build_labels import build_labels, ordered_splits
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
        else:
            require(
                parts[0] == "response" and len(parts) in (2, 4), "Invalid Customer sample ID"
            )
        record = customer_sample(self.catalog, int(parts[1]), stage=parts[0], render=False)
        for sample in expand_record("customer", record, tokenizer=self.tokenizer):
            if sample["sample_id"] == sample_id:
                return self.check_sample(sample)
        raise ValueError(
            "Sample ID is absent; use request_samples for the complete request"
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

    def request_samples(self, session_id, *, stage=None, max_output_tokens=0):
        """Read one request for both stages or just the requested stage."""
        record = customer_sample(self.catalog, session_id, stage=stage, render=False)
        return checked_record(self, "customer", record, max_output_tokens=max_output_tokens)

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
                    self.catalog, current, buckets, tokenizer=self.tokenizer, split=split, stage=stage
                )
            )
            for sample in source:
                yield self.check_sample(sample)


def expand_record(role, record, *, tokenizer=None, compact_background=False, max_output_tokens=0):
    if role == "seller":
        yield record
    elif role == "customer":
        query, response = record
        if query is not None:
            if "messages" not in query or compact_background:
                query = {**query, "messages": messages(
                    query["input"], query["target"], compact_background=compact_background)}
            yield query
        if response is not None:
            yield from split_supervision(response, tokenizer, compact_background=compact_background,
                                         max_output_tokens=max_output_tokens)
    else:
        raise ValueError("Unknown supervision role")


def checked_record(data, role, record, *, max_output_tokens=0):
    return [data.check_sample(sample) for sample in expand_record(
        role, record, tokenizer=data.tokenizer, max_output_tokens=max_output_tokens,
    )]


def record_samples(data, role, record_id, *, max_output_tokens=0):
    if role == "seller":
        return [data.sample(record_id)]
    if role != "customer":
        raise ValueError("Unknown selection role")
    return data.request_samples(int(record_id), max_output_tokens=max_output_tokens)


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


def customer_samples(catalog, bucket, buckets, *, tokenizer=None, split=None, stage=None):
    for record in customer_records(catalog, bucket, buckets, split=split, stage=stage):
        yield from expand_record("customer", record, tokenizer=tokenizer)


def export(directory, tokenizer_path, sample_ids, output):
    """Publish a complete preview only after all requested samples have rendered."""
    require(not output.exists(), "Choose a new example directory")
    with closing(SFTDataset(directory, tokenizer_path)) as data:
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent, prefix="." + output.name + "-") as tmp:
            staged = Path(tmp) / "preview"
            staged.mkdir()
            for sample_id in sample_ids:
                sample = data.sample(sample_id)
                stem = sample_id.replace(":", "_")
                (staged / (stem + ".json")).write_text(
                    json.dumps(sample, ensure_ascii=False, indent=2) + "\n"
                )
                messages = sample["messages"]
                text = "SYSTEM\n" + messages[0]["content"] + "\n\nUSER\n" + messages[1]["content"]
                text += "\n\nTOOLS\n" + json.dumps(sample["tools"], ensure_ascii=False, indent=2)
                text += "\n\nSUPERVISED ANSWER\n" + json.dumps(
                    messages[-1], ensure_ascii=False, indent=2
                )
                (staged / (stem + "_model_input.txt")).write_text(text + "\n")
            staged.rename(output)


def release_splits(directory):
    counts = json.loads((Path(directory) / 'manifest.json').read_text())['counts']
    splits = ordered_splits(counts['customer'])
    require(set(counts['seller']) == set(splits), 'Role split protocols differ')
    return splits


def selection_split(directory):
    """Return the official test split for evaluation."""
    return release_splits(directory)[-1]


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
