"""Publish the Customer and Seller supervision labels from the fact catalog."""

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_simulation.data.catalog import Catalog
from agent_simulation.data.common import literal, require, sha256


def ordered_splits(names):
    names = set(names)
    require(names == {'train', 'test'}, 'Expected official train/test splits')
    return ('train', 'test')


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
