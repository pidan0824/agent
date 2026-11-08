"""Index official train/test membership directly, without creating a validation split."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import closing
import duckdb

from agent_simulation.data.common import Config, business_parts, export, literal, require, rows, save_json, sha256


def validate_sources(cfg, manifest):
    """Verify the split's source files and return the checked Recall hash."""
    require(cfg.dataset == manifest["dataset"], "Split manifest belongs to another release")
    recall = Path(cfg.sources["recall"]["path"])
    require(
        recall.resolve() == Path(manifest["recall_source"]).resolve(), "Different Recall source"
    )
    recall_hash = sha256(recall)
    require(recall_hash == manifest["recall_sha256"], "Recall source hash changed")
    actual = business_parts(cfg)
    expected = manifest["business_sources"]
    require(len(actual) == cfg.item_parts == len(expected), "Incomplete business partitions")
    require(
        {p.resolve() for p in actual} == {Path(r["path"]).resolve() for r in expected},
        "Different business sources",
    )
    for record in expected:
        require(sha256(Path(record["path"])) == record["sha256"], "Business source hash changed")
    return recall_hash


def build(cfg, output):
    require(not output.exists(), "Use a new split output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix="." + output.name + "-") as tmp:
        staged = Path(tmp) / "split"
        staged.mkdir()
        with closing(duckdb.connect()) as connection:
            manifest = _build(cfg, staged, connection)
        staged.rename(output)
    return manifest


def _build(cfg, output, c):
    c.execute("SET threads=2; SET memory_limit='2GB'; SET preserve_insertion_order=false")
    c.execute("SET max_temp_directory_size='2GB'")
    c.execute(f"SET temp_directory={literal(output/'spill')}")
    recall = Path(cfg.sources["recall"]["path"])
    recall_stat = (recall.stat().st_size, recall.stat().st_mtime_ns)
    recall_hash = sha256(recall)
    # Check feedback against its exposure list on read. The chronological index
    # only needs identifiers, time, split and the outcome of that check.
    c.execute(f"""CREATE TABLE requests AS SELECT session_id,user_id,time_index,split,
        (session_id IS NULL OR user_id IS NULL OR time_index IS NULL
          OR time_index<1 OR query IS NULL OR impressed_item_ids IS NULL
          OR clicked_item_ids IS NULL OR purchased_item_ids IS NULL
          OR len(impressed_item_ids)<>list_unique(impressed_item_ids)
          OR len(clicked_item_ids)<>list_unique(clicked_item_ids)
          OR len(purchased_item_ids)<>list_unique(purchased_item_ids)
          OR NOT list_has_all(impressed_item_ids,clicked_item_ids)
          OR NOT list_has_all(impressed_item_ids,purchased_item_ids)) AS invalid_request
        FROM read_json_auto({literal(recall)})""")
    original = {
        r["split"]: {k: v for k, v in r.items() if k != "split"}
        for r in rows(
            c,
            "SELECT split,count(*) n,min(time_index) AS first,max(time_index) AS last FROM requests GROUP BY split",
        )
    }
    require(set(original) == {"train", "test"}, "Expected explicit train/test source flags")
    require(original["train"]["last"] < original["test"]["first"], "Official train/test overlap")
    require(
        c.execute(
            """SELECT count(*)=count(DISTINCT session_id) AND
        count(*) FILTER(WHERE invalid_request)=0 FROM requests"""
        ).fetchone()[0],
        "Duplicate or invalid request / feedback",
    )
    cutoff = original["train"]["last"]
    split_names = ('train', 'test')
    customer = [dict(assigned_split=split, **original[split]) for split in split_names]
    files = [
        export(
            c,
            """SELECT session_id,user_id,time_index,split AS source_split,split AS assigned_split
               FROM requests ORDER BY time_index,session_id""",
            output / "customer_requests.parquet",
            output,
        )
    ]
    totals = {}
    excluded = {"all_observations": 0, "first_observation": 0, "invalid_target_after_first": 0,
                "valid_target_without_prior_valid_price": 0}
    sources = []
    parts = business_parts(cfg)
    for part in parts:
        stamp = (part.stat().st_size, part.stat().st_mtime_ns)
        sources.append({"path": str(part.resolve()), "sha256": sha256(part)})
        bucket = int(part.stem.split("_")[1])
        good = "price_status='observed' AND price_raw IS NOT NULL AND isfinite(price_raw) AND price_raw>0"
        c.execute(
            f"""CREATE OR REPLACE TEMP TABLE events AS SELECT item_id,seller_id,time_index,
            price_raw,price_status,source_split AS split,
            lag(time_index) OVER(PARTITION BY item_id ORDER BY time_index) AS previous_time,
            min(time_index) FILTER (WHERE {good}) OVER (PARTITION BY item_id) AS first_valid_time
            FROM read_parquet({literal(part)})"""
        )
        bad = c.execute(
            """SELECT count(*) FROM events WHERE item_id IS NULL OR seller_id IS NULL
            OR item_id<=0 OR seller_id<=0
            OR time_index IS NULL OR split IS NULL OR price_status IS NULL OR time_index<1
            OR item_id % ? <> ? OR previous_time>=time_index OR split NOT IN ('train','test')
            OR (split='train' AND time_index NOT BETWEEN ? AND ?)
            OR (split='test' AND time_index NOT BETWEEN ? AND ?)""",
            [
                cfg.item_parts,
                bucket,
                original["train"]["first"],
                original["train"]["last"],
                original["test"]["first"],
                original["test"]["last"],
            ],
        ).fetchone()[0]
        require(bad == 0, "Invalid business key, partition, order or source split")
        n, first, invalid, no_prior = c.execute(
            f"""SELECT count(*),count(*) FILTER(WHERE previous_time IS NULL),
            count(*) FILTER(WHERE previous_time IS NOT NULL AND NOT ({good})),
            count(*) FILTER(WHERE previous_time IS NOT NULL AND {good} AND first_valid_time=time_index) FROM events"""
        ).fetchone()
        for key, value in zip(excluded, (n, first, invalid, no_prior)):
            excluded[key] += value
        c.execute(
            f"""CREATE OR REPLACE TEMP VIEW targets AS SELECT seller_id,item_id,time_index,
            previous_time,split AS assigned_split FROM events WHERE first_valid_time < time_index AND {good}"""
        )
        for r in rows(
            c,
            "SELECT assigned_split,count(*) n,min(time_index) AS first,max(time_index) AS last FROM targets GROUP BY assigned_split",
        ):
            v = totals.setdefault(
                r["assigned_split"], {"n": 0, "first": r["first"], "last": r["last"]}
            )
            v["n"] += r["n"]
            v["first"] = min(v["first"], r["first"])
            v["last"] = max(v["last"], r["last"])
        files.append(
            export(
                c,
                "SELECT * FROM targets ORDER BY item_id,time_index",
                output / "seller_targets" / part.name,
                output,
            )
        )
        require(
            stamp == (part.stat().st_size, part.stat().st_mtime_ns),
            "Business source changed during split",
        )
        print(cfg.dataset, part.name, flush=True)
    require(
        sum(v["n"] for v in totals.values())
        + excluded["first_observation"]
        + excluded["invalid_target_after_first"]
        + excluded["valid_target_without_prior_valid_price"]
        == excluded["all_observations"],
        "Seller targets do not reconcile",
    )
    require(
        recall_stat == (recall.stat().st_size, recall.stat().st_mtime_ns),
        "Recall changed during split",
    )
    require(set(totals) == set(split_names), "Empty Seller target split")
    manifest = {
        "schema": "kuaisearch-split-index-v1",
        "dataset": cfg.dataset,
        "rule": {
            "membership": "official_source_split",
            "last_train_time_index": cutoff,
            "official_test_unchanged": True,
            "same_time_index_kept_together": True,
        },
        "official_requests": original,
        "customer_request_pools": customer,
        "seller_label_pools": totals,
        "seller_exclusions": excluded,
        "recall_source": str(recall.resolve()),
        "recall_sha256": recall_hash,
        "business_sources": sources,
        "files": files,
        "units": {
            "customer": "One request index shared by Query and Response.",
            "seller": "One item-time price label with a strictly earlier valid observed price. Unchanged prices included.",
        },
        "history_protocol": "Strictly earlier real observations in held-out replay. Earlier held-out outcomes may enter later context, never fitting or parameter updates.",
        "status": "Complete split membership indices. These contain no SFT prompts or tokens.",
    }
    save_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(Config.load(args.config), args.output)
