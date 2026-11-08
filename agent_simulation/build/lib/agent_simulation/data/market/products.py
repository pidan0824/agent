"""Expose an eligible public catalog; the caller decides which products to inspect."""

import argparse
import csv
import json
import math
import time
import unicodedata
from pathlib import Path
from agent_simulation.data.common import Config, business_parts, connect, literal, rows, require, save_json

PROFILE_FIELDS = (
    "item_id",
    "seller_id",
    "seller_name",
    "item_title",
    "brand_id",
    "brand_name",
    "category_level1_id",
    "category_level1_name",
    "category_level2_id",
    "category_level2_name",
    "category_level3_id",
    "category_level3_name",
)
CATEGORY_FIELDS = ("category_level1_id", "category_level2_id", "category_level3_id")
PRICE_FIELDS = ("reference_price_raw", "reference_price_time_index", "price_status")
PUBLIC_FIELDS = PROFILE_FIELDS + PRICE_FIELDS


def history(c, cfg, item_id, cutoff):
    return rows(
        c,
        f"""SELECT * FROM read_parquet({literal(cfg.data / 'seller_business' / f'part_{item_id % cfg.item_parts:02d}.parquet')})
        WHERE item_id=? AND time_index<? ORDER BY time_index""",
        [item_id, cutoff],
    )


def complete_category(profile):
    return all(profile[k] is not None and profile[k] > 0
               and profile[k.replace("_id", "_name")] not in (None, "", "UNKNOWN")
               for k in CATEGORY_FIELDS)


def public_candidates(c, cfg, item_id, cutoff, *, relation=None):
    """A prior observation establishes eligibility, not a true listing date or seller awareness."""
    require(
        type(item_id) is int and item_id > 0 and type(cutoff) is int and cutoff > 0,
        "Positive integer item and exclusive cutoff required",
    )
    business_parts(cfg)
    projection = (
        ",".join(PROFILE_FIELDS) + ",price_raw AS reference_price_raw,"
        "time_index AS reference_price_time_index,price_status"
    )
    own = rows(
        c,
        f"""SELECT {projection} FROM read_parquet(
        {literal(cfg.data / 'seller_business' / f'part_{item_id % cfg.item_parts:02d}.parquet')})
        WHERE item_id=? AND time_index<? ORDER BY time_index DESC LIMIT 1""",
        [item_id, cutoff],
    )
    if not own:
        return None, []
    own = own[0]
    if not complete_category(own):
        return own, []
    relation = relation or f"read_parquet({literal(cfg.data / 'seller_business/part_*.parquet')})"
    candidates = rows(
        c,
        f"""SELECT {projection} FROM {relation}
        WHERE {' AND '.join(key+'=?' for key in CATEGORY_FIELDS)} AND seller_id<>? AND time_index<?
        QUALIFY row_number() OVER(PARTITION BY item_id ORDER BY time_index DESC)=1
        ORDER BY item_id""",
        [*[own[key] for key in CATEGORY_FIELDS], own["seller_id"], cutoff],
    )
    candidates = [row for row in candidates if complete_category(row)]
    for row in [own, *candidates]:
        require(row["seller_id"] is not None and row["seller_id"] > 0, "Invalid seller identity")
        require(row["reference_price_time_index"] < cutoff, "Future reference price")
        price = row["reference_price_raw"]
        if row["price_status"] == "observed":
            require(
                price is not None and math.isfinite(price) and price > 0, "Invalid observed price"
            )
        else:
            require(
                row["price_status"] in ("invalid_price", "conflicting_prices") and price is None,
                "Unknown or inconsistent price status",
            )
    return own, candidates


def snapshot(c, cfg, item_id, cutoff):
    """Return every eligible product, not a selected competitor list or an Agent prompt."""
    _, candidates = public_candidates(c, cfg, item_id, cutoff)
    return {"item_id": item_id, "history_cutoff_exclusive": cutoff, "candidates": candidates}


def candidate_page(pool, *, after_item_id=0, page_size=20):
    """Page a candidate pool already ordered by ascending item ID."""
    require(
        type(after_item_id) is int and after_item_id >= 0, "Nonnegative integer cursor required"
    )
    require(type(page_size) is int and page_size > 0, "Positive integer page size required")
    remaining = [row for row in pool["candidates"] if row["item_id"] > after_item_id]
    page = remaining[:page_size]
    return {
        "item_id": pool["item_id"],
        "history_cutoff_exclusive": pool["history_cutoff_exclusive"],
        "total_candidates": len(pool["candidates"]),
        "after_item_id": after_item_id,
        "next_after_item_id": page[-1]["item_id"] if len(remaining) > page_size else None,
        "candidates": [dict(row) for row in page],
    }


def candidate_detail(pool, candidate_item_id):
    """Restrict detail lookup to the same server-bound focal item and cutoff snapshot."""
    require(
        type(candidate_item_id) is int and candidate_item_id > 0,
        "Positive integer candidate ID required",
    )
    for row in pool["candidates"]:
        if row["item_id"] == candidate_item_id:
            return {
                "item_id": pool["item_id"],
                "history_cutoff_exclusive": pool["history_cutoff_exclusive"],
                "candidate": dict(row),
            }
    raise PermissionError("Product is outside the eligible pre-cutoff public candidate pool")


def discover_candidates(pool, *, after_item_id=0, page_size=5, query="", brand_id=None):
    """Filter public text literally, then paginate; relevance remains the Seller's judgment."""
    require(type(page_size) is int and 1 <= page_size <= 100, "Page size must be between 1 and 100")
    require(isinstance(query, str), "Query must be text")
    require(brand_id is None or (type(brand_id) is int and brand_id > 0), "Invalid brand ID")
    normalize = lambda text: unicodedata.normalize("NFKC", text).casefold()
    terms = normalize(query).split()
    matched = []
    for row in pool["candidates"]:
        if brand_id is not None and row["brand_id"] != brand_id:
            continue
        text = normalize(" ".join(row[k] or "" for k in ("item_title", "brand_name")))
        if all(term in text for term in terms):
            matched.append(row)
    page = candidate_page(
        {**pool, "candidates": matched}, after_item_id=after_item_id, page_size=page_size
    )
    page["eligible_candidates"] = len(pool["candidates"])
    page["filters"] = {"query": query, "brand_id": brand_id}
    page["ordering"] = "item_id ascending; not a relevance or competitor ranking"
    page["matching"] = (
        "All whitespace-separated terms must occur in the title or brand name; literal matching."
    )
    return page


def write_csv(path, records, fields=PUBLIC_FIELDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--item-id", type=int, required=True)
    parser.add_argument("--cutoff", type=int, required=True)
    parser.add_argument("--action", choices=("export", "list", "detail"), default="export")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--after-item-id", type=int, default=0)
    parser.add_argument("--candidate-id", type=int)
    args = parser.parse_args()
    require(
        (args.action == "export") == (args.output is not None),
        "--output is required only for export",
    )
    require(
        (args.action == "detail") == (args.candidate_id is not None),
        "--candidate-id is required only for detail",
    )
    if args.output is not None:
        require(
            not args.output.exists(),
            "Use a new example directory; do not overwrite earlier evidence",
        )
    cfg = Config.load(args.config)
    c = connect(cfg)
    started = time.monotonic()
    try:
        pool = snapshot(c, cfg, args.item_id, args.cutoff)
    finally:
        c.close()
    if args.action == "detail":
        result = candidate_detail(pool, args.candidate_id)
    else:
        result = candidate_page(pool, after_item_id=args.after_item_id, page_size=args.page_size)
    if args.action == "export":
        audit = {
            "dataset": cfg.dataset,
            "eligible_candidates": len(pool["candidates"]),
            "eligibility_rule": "same complete category, different seller, observed strictly before cutoff",
            "ordering": "item_id ascending for pagination only; no ranking or automatic focus selection",
            "page_size": args.page_size,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        save_json(args.output / "candidate_pool.json", pool)
        save_json(args.output / "candidate_page.json", result)
        write_csv(args.output / "candidates.csv", pool["candidates"])
        print(json.dumps(audit, ensure_ascii=False), flush=True)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
