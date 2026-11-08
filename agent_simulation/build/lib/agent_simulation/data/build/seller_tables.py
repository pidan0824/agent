"""One configurable pipeline for Full, Lite or independently supplied KuaiSearch files."""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from agent_simulation.data.common import (
    Config,
    connect,
    literal,
    require,
    rows,
    save_json,
    sha256,
    copy_parquet,
    bind_config,
)


def raw_ranking(path, *, with_history=False):
    """Expose the same typed request, product, price and feedback fields in both builds."""
    columns = {
        "session_id": "BIGINT",
        "target_item_id": "BIGINT",
        "time_index": "BIGINT",
        "target_item_price": "DOUBLE",
        "is_clicked": "INTEGER",
        "is_purchased": "INTEGER",
        "split": "VARCHAR",
    }
    if with_history:
        columns["target_item_statistical_features"] = (
            "STRUCT(item_show_cnt_30d_hist DOUBLE, item_click_cnt_30d_hist DOUBLE, "
            "item_order_cnt_30d_hist DOUBLE)"
        )
    schema = ", ".join(f"{key}: {literal(dtype)}" for key, dtype in columns.items())
    return f"read_json({literal(path)}, format='newline_delimited', columns={{{schema}}})"


def attach_sources(c, cfg):
    c.execute(
        f"CREATE VIEW profiles AS SELECT * FROM read_parquet({literal(cfg.work / 'profiles.parquet')})"
    )
    c.execute(
        f"CREATE VIEW requests AS SELECT * FROM read_parquet({literal(cfg.work / 'requests.parquet')})"
    )


def attach_rank(c, cfg):
    c.execute(
        f"CREATE VIEW ranking AS SELECT * FROM read_parquet({literal(cfg.work / 'ranking.parquet')})"
    )


def observations_sql(partitions=1, bucket=0):
    require(type(partitions) is int and type(bucket) is int and 0 <= bucket < partitions,
            "Invalid observation partition")
    where = f" WHERE item_id % {partitions} = {bucket}" if partitions > 1 else ""
    return f"""SELECT item_id,time_index,min(price) AS price_min_raw,max(price) AS price_max_raw,
        count(*) AS exposure_records,sum(is_clicked)::BIGINT AS click_records,
        sum(is_purchased)::BIGINT AS purchase_records,min(split) AS source_split,
        min(shows_30d) AS item_shows_30d_hist,min(clicks_30d) AS item_clicks_30d_hist,
        min(orders_30d) AS item_orders_30d_hist,
        count(*) FILTER(WHERE price IS NULL OR NOT isfinite(price) OR price<=0) AS bad_prices,
        count(DISTINCT struct_pack(s:=split,a:=shows_30d,b:=clicks_30d,c:=orders_30d)) AS context_variants
        FROM ranking{where} GROUP BY item_id,time_index"""


def business_sql():
    return """SELECT p.*,o.time_index,
        CASE WHEN bad_prices=0 AND price_min_raw=price_max_raw THEN price_min_raw END AS price_raw,
        price_min_raw,price_max_raw,
        CASE WHEN bad_prices>0 THEN 'invalid_price'
             WHEN price_min_raw<>price_max_raw THEN 'conflicting_prices' ELSE 'observed' END AS price_status,
        exposure_records,click_records,purchase_records,
        item_shows_30d_hist,item_clicks_30d_hist,item_orders_30d_hist,source_split
        FROM observations o JOIN profiles p USING(item_id)"""

def sources(c, cfg):
    manifest = cfg.output / "source_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    evidence = []
    for name in (key for key in cfg.sources if key != "ranking"):
        official = cfg.sources[name]
        path = Path(official["path"])
        digest = sha256(path)
        require(
            "bytes" not in official or path.stat().st_size == official["bytes"],
            f"{name} source size mismatch",
        )
        require(
            "sha256" not in official or digest == official["sha256"],
            f"{name} source SHA256 mismatch",
        )
        evidence.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "official_file": official.get("official_file"),
                "expected_hash_matches": "sha256" in official,
            }
        )
    copy_parquet(
        c,
        f"SELECT * FROM read_json_auto({literal(cfg.sources['items']['path'])})",
        cfg.work / "profiles.parquet",
    )
    copy_parquet(
        c,
        f"SELECT * FROM read_json_auto({literal(cfg.sources['recall']['path'])})",
        cfg.work / "requests.parquet",
    )
    attach_sources(c, cfg)
    profile_stats = rows(
        c,
        """SELECT count(*) AS items,count(DISTINCT item_id) AS unique_items,
        count(DISTINCT seller_id) AS sellers,count(*) FILTER(WHERE item_id IS NULL OR item_id<=0
        OR seller_id IS NULL OR seller_id<=0) AS invalid_ids FROM profiles""",
    )[0]
    require(
        profile_stats["items"] == profile_stats["unique_items"]
        and profile_stats["invalid_ids"] == 0,
        "Missing or duplicate product identity",
    )
    request_stats = rows(
        c,
        """SELECT count(*) AS requests,count(DISTINCT session_id) AS sessions,
        sum(len(impressed_item_ids)) AS exposures,sum(len(clicked_item_ids)) AS clicks,
        sum(len(purchased_item_ids)) AS purchases,
        count(*) FILTER(WHERE session_id IS NULL OR time_index IS NULL OR split IS NULL
          OR impressed_item_ids IS NULL OR clicked_item_ids IS NULL OR purchased_item_ids IS NULL
          OR len(impressed_item_ids)<>list_unique(impressed_item_ids)
          OR len(clicked_item_ids)<>list_unique(clicked_item_ids)
          OR len(purchased_item_ids)<>list_unique(purchased_item_ids)
          OR NOT list_has_all(impressed_item_ids,clicked_item_ids)
          OR NOT list_has_all(impressed_item_ids,purchased_item_ids)) AS invalid_requests FROM requests""",
    )[0]
    require(
        request_stats["requests"] == request_stats["sessions"]
        and request_stats["invalid_requests"] == 0,
        "Missing/duplicate request identity, lists or feedback outside exposure",
    )
    coverage = rows(
        c,
        """SELECT count(*) AS exposures,count(*) FILTER(WHERE p.seller_id IS NULL) AS missing_products,
        count(*) FILTER(WHERE p.category_level3_id>0 AND p.category_level3_name<>'UNKNOWN')
            AS known_category_exposures
        FROM requests r CROSS JOIN unnest(impressed_item_ids) u(item_id)
        LEFT JOIN profiles p USING(item_id)""",
    )[0]
    require(coverage.pop("missing_products") == 0, "Recall has products absent from the configured Item table")
    save_json(
        manifest,
        {
            "dataset": cfg.dataset,
            "sources": evidence,
            "profiles": profile_stats,
            "requests": request_stats,
            "coverage": coverage,
        },
    )
    print(
        json.dumps({"profiles": profile_stats, "requests": request_stats, "coverage": coverage}),
        flush=True,
    )


def ranking(c, cfg):
    evidence = cfg.sources["ranking"]
    path = evidence["path"]
    if "sha256" in evidence:
        require(sha256(Path(path)) == evidence["sha256"], "Ranking source SHA256 mismatch")
    count = copy_parquet(
        c,
        f"""SELECT session_id,target_item_id AS item_id,time_index,target_item_price AS price,
        target_item_statistical_features.item_show_cnt_30d_hist AS shows_30d,
        target_item_statistical_features.item_click_cnt_30d_hist AS clicks_30d,
        target_item_statistical_features.item_order_cnt_30d_hist AS orders_30d,is_clicked,is_purchased,split
        FROM {raw_ranking(path, with_history=True)}""",
        cfg.work / "ranking.parquet",
    )
    attach_rank(c, cfg)
    print(json.dumps({"rows": count}), flush=True)


def business(c, cfg):
    output = cfg.data / "seller_business"
    manifest = cfg.output / "business_manifest.json"
    for path in (output, manifest):
        if path.exists():
            raise FileExistsError(path)
    attach_sources(c, cfg)
    attach_rank(c, cfg)
    invalid_ranking = c.execute("""SELECT EXISTS (
        SELECT 1 FROM ranking WHERE session_id IS NULL OR item_id IS NULL OR item_id <= 0
        OR time_index IS NULL OR split IS NULL OR split NOT IN ('train','test')
        OR is_clicked IS NULL OR is_clicked NOT IN (0,1)
        OR is_purchased IS NULL OR is_purchased NOT IN (0,1)
    )""").fetchone()[0]
    require(not invalid_ranking, "Invalid Ranking keys/labels or official split")
    cfg.data.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cfg.data, prefix=".seller-business-") as tmp:
        staged = Path(tmp) / output.name
        files = _write_business_parts(c, cfg, staged, output)
        # Seller IDs can span item partitions.
        stats = rows(
            c,
            f"""SELECT count(*) AS rows,count(DISTINCT item_id) AS items,count(DISTINCT seller_id) AS sellers,
            sum(exposure_records) AS exposures,sum(click_records) AS clicks,sum(purchase_records) AS purchases,
            count(*) FILTER(WHERE price_status='invalid_price') AS invalid_prices,
            count(*) FILTER(WHERE price_status='conflicting_prices') AS conflicting_prices
            FROM read_parquet({literal(staged / 'part_*.parquet')})""",
        )[0]
        staged.rename(output)
    c.execute(f"CREATE VIEW business AS SELECT * FROM read_parquet({literal(output / 'part_*.parquet')})")
    save_json(
        manifest,
        {
            "dataset": cfg.dataset,
            "grain": "item_id x time_index",
            "totals": stats,
            "parts": files,
            "price_unit": "unconfirmed; preserve raw values",
            "time_unit": "relative ordering index, not calendar time",
            "purchases": "binary purchase record counts, not units sold",
            "history_rule": "time_index < decision cutoff; current feedback excluded from decision input",
        },
    )
    print(json.dumps(stats), flush=True)


def _write_business_parts(c, cfg, staged, output):
    files = []
    for bucket in range(cfg.item_parts):
        c.execute("CREATE OR REPLACE TEMP TABLE observations AS " + observations_sql(cfg.item_parts, bucket))
        expected, invalid = c.execute(
            "SELECT count(*), count(*) FILTER (WHERE context_variants<>1) FROM observations"
        ).fetchone()
        require(
            invalid == 0,
            "Conflicting or mixed missing/present historical features or splits at item/time",
        )
        path = staged / f"part_{bucket:02d}.parquet"
        count = copy_parquet(c, business_sql() + " ORDER BY item_id,time_index", path)
        require(count == expected, "Product join lost or multiplied observations")
        files.append({
            "path": str((output / path.name).relative_to(cfg.output)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
        print("Business", bucket + 1, "/", cfg.item_parts, flush=True)
    c.execute("DROP TABLE observations")
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("stage", choices=("sources", "ranking", "business", "all"))
    args = parser.parse_args()
    cfg = Config.load(args.config)
    cfg.output.mkdir(parents=True, exist_ok=True)
    bind_config(cfg)
    require(shutil.disk_usage(cfg.output).free > 8 * 1024**3, "At least 8 GiB free required")
    stages = {"sources": sources, "ranking": ranking, "business": business}
    selected = [sources, ranking, business] if args.stage == "all" else [stages[args.stage]]
    for stage in selected:
        c = connect(cfg)
        try:
            stage(c, cfg)
        finally:
            c.close()


if __name__ == "__main__":
    main()
