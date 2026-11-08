"""Shared IO and schema transformations; all dataset paths come from configuration."""

import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import duckdb


@dataclass(frozen=True)
class Config:
    dataset: str
    output: Path
    sources: dict
    item_parts: int = 8

    @property
    def work(self):
        return self.output / "intermediate"

    @property
    def data(self):
        return self.output / "data"

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        value = json.loads(path.read_text())
        required = {"dataset", "output", "sources"}
        require(
            isinstance(value, dict) and required <= value.keys() <= required | {"item_parts"},
            "Unexpected config keys",
        )
        require(bool(value["dataset"]), "Dataset identity required")
        require(
            {"items", "recall", "ranking"} <= set(value["sources"]),
            "Items, Recall and Ranking sources required",
        )
        for name, source in value["sources"].items():
            require(
                set(source) <= {"path", "sha256", "bytes", "official_file"},
                f"Unexpected source keys: {name}",
            )
            require(
                isinstance(source.get("path"), str) and bool(source["path"]),
                f"Source path required: {name}",
            )
            source["path"] = str((path.parent / source["path"]).resolve())
        output = (path.parent / value.pop("output")).resolve()
        config = cls(output=output, **value)
        require(type(config.item_parts) is int and config.item_parts > 0, "Invalid item_parts")
        return config


def connect(cfg):
    cfg.work.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect()
    c.execute("SET threads=2")
    c.execute("SET memory_limit='2GB'")
    c.execute("SET max_temp_directory_size='6GB'")
    c.execute(f"SET temp_directory={literal(cfg.work / 'spill')}")
    c.execute("SET preserve_insertion_order=false")
    return c


def bind_config(cfg):
    """Prevent mixing sources or partition schemes in an existing build."""
    path = cfg.output / "build_config.json"
    payload = asdict(cfg)
    payload["output"] = str(cfg.output)
    if path.exists():
        require(
            json.loads(path.read_text()) == payload,
            "Output belongs to a different build configuration",
        )
    else:
        save_json(path, payload)


def business_parts(cfg):
    """Require exact partition membership, including empty partitions."""
    directory = cfg.data / "seller_business"
    expected = [directory / f"part_{i:02d}.parquet" for i in range(cfg.item_parts)]
    require(
        set(directory.glob("part_*.parquet")) == set(expected),
        "Missing or unexpected business partitions",
    )
    return expected


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def rows(c, sql, parameters=None):
    """Materialize a query whose result the caller needs as a list."""
    return list(iter_rows(c.execute(sql, parameters or [])))


def copy_parquet(c, query, path):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count, = c.execute(
        f"COPY ({query}) TO {literal(path)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    ).fetchone()
    return count


def export(c, query, path, root):
    copy_parquet(c, query, path)
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def difference(c, left, right):
    return c.execute(
        f"""SELECT count(*) FROM (
        (SELECT * FROM ({left}) EXCEPT ALL SELECT * FROM ({right}))
        UNION ALL (SELECT * FROM ({right}) EXCEPT ALL SELECT * FROM ({left})))"""
    ).fetchone()[0]


def iter_rows(cursor, size=256):
    """Read an executed query in batches for a single pass over its rows."""
    columns = [x[0] for x in cursor.description]
    while batch := cursor.fetchmany(size):
        for values in batch:
            yield dict(zip(columns, values))
