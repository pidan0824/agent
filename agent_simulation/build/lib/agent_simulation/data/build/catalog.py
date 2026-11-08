"""Build one release-bound research catalog without duplicating existing business tables."""

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import duckdb
from agent_simulation.data.common import Config, copy_parquet, difference, literal, require, save_json, sha256
from agent_simulation.data.build.splits import validate_sources
from agent_simulation.data.build.seller_tables import raw_ranking

RANK_SELECT = """session_id,target_item_id AS item_id,time_index,target_item_price AS price_raw,
    is_clicked,is_purchased,split AS source_split"""
TABLE_NOTES = {
    "products": ("商品静态快照；item_id唯一", "Item全部字段；无历史快照日期；不是实时商品状态"),
    "users": ("用户静态资料；user_id唯一", "User全部字段；无资料记录日期"),
    "requests": ("一次搜索请求；session_id唯一", "Recall全部字段；曝光数组顺序不代表屏幕位置"),
    "ranking_observations": (
        "请求×商品",
        "Ranking价格与响应投影；购买标记不是件数；完整原行保留在源文件",
    ),
    "seller_history": ("商品×相对时间索引", "复用已有经营表；计数只覆盖数据中的观测，不补缺失时段"),
    "customer_splits": ("一次搜索请求", "复用冻结划分；Query与Response共用"),
    "seller_targets": ("商品×相对时间索引", "复用价格监督索引；不等于已生成SFT样本"),
    "customer_history": (
        "用户的一次既往搜索",
        "查询前必须限制用户和截止点；不默认附完整历史曝光列表",
    ),
    "public_price_history": ("商品×相对时间索引", "仅公开属性与参考价格；不含竞品私有经营结果"),
}
FIELD_NOTES = {
    "item_id": "商品编号",
    "seller_id": "商家编号",
    "seller_name": "商家名称",
    "item_title": "原始商品标题",
    "brand_id": "品牌编号",
    "brand_name": "品牌名称",
    "user_id": "用户编号",
    "session_id": "搜索请求编号；不据此推断一段多轮会话",
    "time_index": "版本内相对时间索引；不是自然日期或秒",
    "query": "真实中文搜索词",
    "impressed_item_ids": "该请求记录的曝光商品集合；不是已证实的屏幕位置",
    "clicked_item_ids": "该请求真实点击商品集合",
    "purchased_item_ids": "该请求真实购买商品集合",
    "price_raw": "原始价格尺度；不换算元/分；异常或冲突的经营价格保持空值",
    "price_status": "价格观测质量；observed / invalid_price / conflicting_prices",
    "price_min_raw": "同商品同索引观测价格最小值",
    "price_max_raw": "同商品同索引观测价格最大值",
    "exposure_records": "该商品在该观测索引下的曝光记录数",
    "click_records": "该商品在该观测索引下的点击标记数",
    "purchase_records": "该商品在该观测索引下的购买标记数；件数未知",
    "is_clicked": "本次请求对该商品的点击标记",
    "is_purchased": "本次请求对该商品的购买标记",
    "split": "官方train/test",
    "source_split": "官方train/test",
    "assigned_split": "官方train/test，与source_split一致",
    "previous_time": "该商品上一条观测的索引，包含价格缺失或无效的观测",
    "gender": "用户性别字段",
    "age_bucket": "用户年龄段",
    "fre_country": "用户常驻国家字段",
    "fre_province": "用户常驻省份字段",
    "fre_city": "用户常驻城市字段",
    "item_shows_30d_hist": "原Ranking提供的30日历史曝光特征；不是本次窗口汇总",
    "item_clicks_30d_hist": "原Ranking提供的30日历史点击特征；不是本次窗口汇总",
    "item_orders_30d_hist": "原Ranking提供的30日历史订单特征；不是本次窗口汇总",
}
for level in (1, 2, 3):
    FIELD_NOTES[f"category_level{level}_id"] = f"第{level}级类目编号；0可表示未知"
    FIELD_NOTES[f"category_level{level}_name"] = f"第{level}级类目名称；UNKNOWN保留原值"


def convert(c, query, path, order):
    return {
        "rows": copy_parquet(c, f"{query} ORDER BY {order}", path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def create_views(c, output, cfg, splits):
    for name in ("products", "users", "requests", "ranking_observations"):
        c.execute(
            f'CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({literal(output / "facts" / (name+".parquet"))})'
        )
    for name, path in [
        ("seller_history", cfg.data / "seller_business/part_*.parquet"),
        ("customer_splits", splits / "customer_requests.parquet"),
        ("seller_targets", splits / "seller_targets/part_*.parquet"),
    ]:
        c.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({literal(path)})")
    c.execute(
        """CREATE OR REPLACE VIEW customer_history AS SELECT user_id,session_id,time_index,query,
        clicked_item_ids,purchased_item_ids,split AS source_split FROM requests"""
    )
    from agent_simulation.data.market.products import PROFILE_FIELDS

    c.execute(
        "CREATE OR REPLACE VIEW public_price_history AS SELECT "
        + ",".join(PROFILE_FIELDS)
        + ",time_index,price_raw,price_status FROM seller_history"
    )


def verify_catalog(c, cfg):
    checks = {}
    for table, key in [("products", "item_id"), ("users", "user_id"), ("requests", "session_id")]:
        n, unique, bad = c.execute(
            f"""SELECT count(*),count(DISTINCT {key}),
            count(*) FILTER(WHERE {key} IS NULL OR {key}<=0) FROM {table}"""
        ).fetchone()
        require(n == unique and bad == 0, f"Invalid or duplicate {table} identity")
        checks[table] = n
    require(
        c.execute(
            """SELECT count(*) FROM requests WHERE user_id IS NULL OR time_index IS NULL OR time_index<=0
        OR query IS NULL OR length(trim(query))=0 OR split IS NULL OR split NOT IN ('train','test')
        OR impressed_item_ids IS NULL OR clicked_item_ids IS NULL OR purchased_item_ids IS NULL
        OR len(impressed_item_ids)<>list_unique(impressed_item_ids)
        OR len(clicked_item_ids)<>list_unique(clicked_item_ids)
        OR len(purchased_item_ids)<>list_unique(purchased_item_ids)
        OR NOT list_has_all(impressed_item_ids,clicked_item_ids)
        OR NOT list_has_all(impressed_item_ids,purchased_item_ids)"""
        ).fetchone()[0]
        == 0,
        "Invalid request fields",
    )
    # Missing profiles are reported, never manufactured; a query fails explicitly if one is required.
    checks["requests_missing_user_profile"] = c.execute(
        "SELECT count(*) FROM requests r ANTI JOIN users u USING(user_id)"
    ).fetchone()[0]
    checks["exposures"] = c.execute("SELECT sum(len(impressed_item_ids)) FROM requests").fetchone()[
        0
    ]
    require(
        c.execute(
            """SELECT count(*) FROM ranking_observations WHERE session_id IS NULL OR item_id IS NULL
        OR time_index IS NULL OR source_split IS NULL OR source_split NOT IN ('train','test')
        OR is_clicked IS NULL OR is_purchased IS NULL OR is_clicked NOT IN (0,1) OR is_purchased NOT IN (0,1)"""
        ).fetchone()[0]
        == 0,
        "Invalid Ranking identity/labels",
    )
    # Bucketed equality preserves duplicates and checks every request/item and response flag.

    for bucket in range(16):
        actual = f"""SELECT session_id,item_id,time_index,is_clicked,is_purchased,source_split
            FROM ranking_observations WHERE session_id%16={bucket}"""
        expected = f"""SELECT session_id,u.item_id,time_index,list_contains(clicked_item_ids,u.item_id)::INTEGER,
            list_contains(purchased_item_ids,u.item_id)::INTEGER,split
            FROM requests CROSS JOIN unnest(impressed_item_ids) u(item_id) WHERE session_id%16={bucket}"""
        require(
            difference(c, actual, expected) == 0,
            f"Recall/Ranking keys or responses differ: {bucket}",
        )
    checks["all_recall_ranking_keys_and_responses_exact"] = True
    expected = """SELECT session_id,user_id,time_index,split AS source_split,
        split AS assigned_split FROM requests"""
    require(
        difference(c, "SELECT * FROM customer_splits", expected) == 0, "Customer split mismatch"
    )
    checks["customer_split_exact"] = True
    # Business aggregation and Seller eligibility are checked when those tables are built.
    # Their source hashes are bound by validate_sources; do not rebuild both tables here.
    checks["sellers"] = c.execute("SELECT count(DISTINCT seller_id) FROM products").fetchone()[0]
    checks["ranking_rows"] = c.execute("SELECT count(*) FROM ranking_observations").fetchone()[0]
    return checks


def build(cfg, users, splits, output):
    started = time.monotonic()
    require(not output.exists(), "Use a new catalog directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(output.parent).free > 6 * 1024**3, "At least 6 GiB free required")
    split_manifest = json.loads((splits / "manifest.json").read_text())
    recall_stat = Path(cfg.sources["recall"]["path"]).stat()
    recall_hash = validate_sources(cfg, split_manifest)
    # An explicit CLI path must not discard the configured user-file hash.
    user_source = dict(cfg.sources.get("users", {}))
    if users is None:
        require("path" in user_source, "Configure the users source or pass --users")
        users = Path(user_source["path"])
    if user_source:
        require(
            Path(user_source["path"]).resolve() == users.resolve(),
            "Different configured User source",
        )
    user_source["path"] = str(users.resolve())
    sources = {**cfg.sources, "users": user_source}
    output.mkdir()
    (output / "facts").mkdir()
    (output / "facts/seller_history").symlink_to(
        (cfg.data / "seller_business").resolve(), target_is_directory=True
    )
    (output / "splits").symlink_to(splits.resolve(), target_is_directory=True)
    c = duckdb.connect(str(output / "catalog.duckdb"))
    c.execute(
        "SET threads=2; SET memory_limit='2GB'; SET max_temp_directory_size='4GB'; SET preserve_insertion_order=false"
    )
    c.execute(f'SET temp_directory={literal(output/"spill")}')
    source_records, files = {}, {}
    try:
        for key in ("items", "users", "recall", "ranking"):
            path = Path(sources[key]["path"])
            stat = recall_stat if key == "recall" else path.stat()
            digest = recall_hash if key == "recall" else sha256(path)
            require(
                "sha256" not in sources[key] or sources[key]["sha256"] == digest,
                f"{key} source hash mismatch",
            )
            source_records[key] = {
                "path": str(path.resolve()),
                "bytes": stat.st_size,
                "sha256": digest,
                "mtime_ns": stat.st_mtime_ns,
            }
        for table, source, order in [
            ("products", "items", "item_id"),
            ("users", "users", "user_id"),
            ("requests", "recall", "user_id,time_index,session_id"),
            ("ranking_observations", "ranking", "item_id,time_index,session_id"),
        ]:
            path = Path(sources[source]["path"])
            if source == "ranking":
                query = f"SELECT {RANK_SELECT} FROM {raw_ranking(path)}"
            else:
                query = f"SELECT * FROM read_json_auto({literal(path)},format='newline_delimited')"
            files[table] = convert(c, query, output / "facts" / f"{table}.parquet", order)
            print(f'{cfg.dataset}: materialized {table}: {files[table]["rows"]} rows', flush=True)
        create_views(c, output, cfg, splits)
        checks = verify_catalog(c, cfg)
        for record in source_records.values():
            stat = Path(record["path"]).stat()
            require(
                (stat.st_size, stat.st_mtime_ns) == (record["bytes"], record["mtime_ns"]),
                "Source changed while building",
            )
        split_files = []
        for f in split_manifest["files"]:
            path = splits / f["path"]
            require(sha256(path) == f["sha256"], "Split index changed")
            split_files.append({"path": str(path.resolve()), "sha256": f["sha256"]})
        with (output / "data_dictionary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["table", "field", "type", "中文说明", "粒度", "来源与边界"])
            for table, (grain, note) in TABLE_NOTES.items():
                for field, dtype, *_ in c.execute("DESCRIBE " + table).fetchall():
                    require(field in FIELD_NOTES, f"Missing field definition: {table}.{field}")
                    writer.writerow([table, field, dtype, FIELD_NOTES[field], grain, note])
        manifest = {
            "schema": "kuaisearch-catalog-v1",
            "dataset": cfg.dataset,
            "sources": source_records,
            "files": files,
            "business_sources": split_manifest["business_sources"],
            "split_manifest": str((splits / "manifest.json").resolve()),
            "split_files": split_files,
            "pipeline_config": {
                "dataset": cfg.dataset,
                "output": str(cfg.output),
                "sources": sources,
                "item_parts": cfg.item_parts,
            },
            "checks": checks,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "limits": [
                "Relative time_index and unconfirmed raw price scale.",
                "Static profile snapshot time unavailable.",
                "Exposure list order is not confirmed display position.",
                "Purchases count binary records, not units.",
                "Full/Lite remain separate experiments.",
                "No LLM-generated memories, SFT prompts, tokens or trained models created.",
            ],
            "storage": (
                "DuckDB views reference local fact files and external Parquet files by path. "
                "External Seller history and split files are listed in business_sources and split_files."
            ),
        }
        save_json(output / "manifest.json", manifest)
        c.execute("CHECKPOINT")
    finally:
        c.close()
    if (output / "spill").exists():
        (output / "spill").rmdir()
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--users", type=Path, help="Defaults to the users source in --config")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(
        Config.load(args.config), args.users, args.splits.resolve(), args.output.resolve()
    )
