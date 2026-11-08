"""Read historical quotes, category snapshots and market price comparisons."""


import json
import duckdb
from copy import deepcopy
from collections import OrderedDict
from bisect import bisect_left
from itertools import groupby
from heapq import nlargest
from agent_simulation.data.common import iter_rows, rows, require, business_parts, literal, sha256
from agent_simulation.data.market.products import CATEGORY_FIELDS, PROFILE_FIELDS, complete_category
from agent_simulation.data.market.statistics import MARKET_NOTE, market_position, distribution, COUNTS
from pathlib import Path

def category_panels(data, categories, before):
    """Read bounded category batches under the existing 2 GB database memory limit."""
    for start in range(0, len(categories), 16):
        yield from _category_panels(data, categories[start:start+16], before)


def _category_panels(data, categories, before):
    """Yield one category's prefix counts at a time from each shared source scan."""
    if not categories:
        return
    fields = ','.join(CATEGORY_FIELDS)
    placeholders = ','.join('(?,?,?)' for _ in categories)
    cursor = data.connection.execute(f"""
          SELECT {fields},item_id,seller_id,time_index,price_status,
            CASE WHEN price_status='observed' AND isfinite(price_raw) AND price_raw>0
                 THEN price_raw END price_raw,{','.join(COUNTS)}
          FROM {business_relation(data, 'category')}
          WHERE ({fields}) IN (VALUES {placeholders}) AND time_index<?
          ORDER BY {fields},item_id,time_index
          """, [v for category in categories for v in category] + [before])
    records = iter_rows(cursor, size=4096)
    for category, observations in groupby(records, key=lambda p: tuple(p[k] for k in CATEGORY_FIELDS)):
        panel = []
        for item, history in groupby(observations, key=lambda p: p['item_id']):
            product = {'item_id': item, 'indices': [], **{k: [0] for k in COUNTS}}
            for row in history:
                product['indices'].append(row['time_index'])
                if row['price_raw'] is not None:
                    product['last_valid_price'] = row['price_raw']
                    product['last_valid_time'] = row['time_index']
                for metric in COUNTS:
                    product[metric].append(product[metric][-1] + row[metric])
            product.update(seller_id=row['seller_id'], price_raw=row['price_raw'],
                           price_status=row['price_status'], price_time_index=row['time_index'])
            panel.append(product)
        yield (before, *category), panel


def category_rows(data, profile, before, start):
    """Reuse prefix counts for unequal windows without rescanning raw records."""
    category = tuple(profile[k] for k in CATEGORY_FIELDS)
    key = (before, *category)
    if key not in data._market_panels:
        _, panel = next(category_panels(data, [category], before))
        # A bounded cache avoids retaining every category in a large shop.
        if len(data._market_panels) >= 2:
            data._market_panels.clear()
        data._market_panels[key] = panel
    result = []
    for product in data._market_panels[key]:
        position = bisect_left(product['indices'], start)
        visible = position < len(product['indices'])
        result.append({k: product[k] for k in
                       ('item_id', 'seller_id', 'price_raw', 'price_status', 'price_time_index')} |
                      {k: product[k][-1]-product[k][position] if visible else None for k in COUNTS} |
                      {'cumulative_'+k: product[k][-1] for k in COUNTS})
    return result


def owned_profiles(data, seller_id, before, item_ids=None):
    """Each product's start is derived from its own retained history window.

    This observation-based window is explicit, not a claimed optimal duration or calendar day.
    Caller-specified windows can override it through historical_position.
    """
    n = data.history_points
    selected = tuple(sorted(item_ids)) if item_ids is not None else None
    clause = " AND item_id IN (SELECT unnest(?::BIGINT[]))" if selected is not None else ""
    profiles = data.decision_cached(('profiles', seller_id, before, n, selected), lambda: rows(data.connection, f"""
        WITH ranked AS (
          SELECT *,row_number() OVER(PARTITION BY item_id ORDER BY time_index DESC) rn
          FROM {business_relation(data, 'seller')}
          WHERE seller_id=? AND time_index<? {clause}
        ), recent AS (
          SELECT *,min(time_index) OVER(PARTITION BY item_id) AS recent_start
          FROM ranked WHERE rn<=?
        ) SELECT * EXCLUDE(rn) FROM recent WHERE rn=1
        """, [seller_id, before] + ([list(selected)] if selected is not None else []) + [n]))
    # The summary already read these exact latest profiles; discovery need not scan
    # the target's full history again. Cache keys retain the exclusive cutoff.
    for profile in profiles:
        data.decision_cached(('product_profile', profile['item_id'], before), lambda: profile)
    return profiles


RECENT_MARKET_SPAN = 10000
SPARSE_PURCHASE_THRESHOLD = 5

PRICE_BANDS = ("(0,1000)", "[1000,5000)", "[5000,10000)", "[10000,+inf)", "unknown")


def purchase_bands(connection, seller, start, end, *, categories=None):
    """Only public purchase events are read. Group peers by category and price at purchase."""
    fields = ",".join(CATEGORY_FIELDS)
    selected = sorted(set(map(tuple, categories))) if categories is not None else None
    if selected == []:
        return {}
    clause = (f" AND ({fields}) IN (VALUES " + ','.join('(?,?,?)' for _ in selected) + ')'
              if selected is not None else '')
    records = rows(connection, f"""
        SELECT {fields}, CASE WHEN price_raw IS NULL OR NOT isfinite(price_raw) OR price_raw<=0 THEN 4
            WHEN price_raw<1000 THEN 0 WHEN price_raw<5000 THEN 1 WHEN price_raw<10000 THEN 2 ELSE 3 END band,
            count(*) purchases, count(DISTINCT item_id) purchased_products, count(DISTINCT seller_id) selling_shops
        FROM {purchase_relation(connection)}
        WHERE time_index>=? AND time_index<? AND seller_id<>? {clause}
        GROUP BY {fields},band""", [start, end, seller] +
            ([v for c in selected for v in c] if selected is not None else []))
    return {(tuple(r[k] for k in CATEGORY_FIELDS), r["band"]):
            {k:r[k] for k in ("purchases", "purchased_products", "selling_shops")} for r in records}


def band_summary(purchases, category, price, start, end, *, last_valid_reference=None):
    from bisect import bisect_right
    result = dict(clock="observed_time_index", start_inclusive=start, end_exclusive=end, price_raw=price)
    if price is None and last_valid_reference is not None:
        price = last_valid_reference["price_raw"]
        result.update(price_raw=price, price_basis="last_valid_observed",
                      price_time_index=last_valid_reference["time_index"])
    if price is None:
        return {**result, "status": "unknown_own_price"}
    i = bisect_right((1000, 5000, 10000), price)
    counts = purchases.get((tuple(category), i), dict(purchases=0, purchased_products=0, selling_shops=0))
    return {**result, "band": PRICE_BANDS[i], **counts,
            "evidence_status": "sparse" if counts["purchases"] < SPARSE_PURCHASE_THRESHOLD else "observed",
            "sparse_below_records": SPARSE_PURCHASE_THRESHOLD}


def observation_range(indices):
    values = list(indices)
    return [min(values), max(values)] if values else None


def extended_price_sample(peers, recent, before, start, minimum=20):
    """Supplement only sparse windows, using distinct products and complete time-index ties."""
    if len(recent) >= minimum:
        return None
    older = [p for p in peers if p['price_raw'] is not None and p['price_time_index'] < start]
    newest = nlargest(minimum-len(recent), (p['price_time_index'] for p in older))
    extra = [p for p in older if p['price_time_index'] >= newest[-1]] if newest else []
    sample = recent + extra
    stats = distribution([p['price_raw'] for p in sample], None)
    return {'target_products': minimum, 'recent_products': len(recent), 'added_products': len(extra),
            'products': len(sample), 'shops': len({p['seller_id'] for p in sample}),
            'target_reached': len(sample) >= minimum, 'cutoff_exclusive': before,
            'observation_range': observation_range(p['price_time_index'] for p in sample),
            **{k: stats[k] for k in ('p25','median','p75')}}


def historical_reference(panel, seller_id):
    valid = [p for p in panel if p['seller_id'] != seller_id and 'last_valid_price' in p]
    stats = distribution([p['last_valid_price'] for p in valid], None)
    return {'selection': 'last_valid_observation_per_peer_before_cutoff',
            'products': len(valid), 'shops': len({p['seller_id'] for p in valid}),
            'observation_range': observation_range(p['last_valid_time'] for p in valid),
            **{k: stats[k] for k in ('p25', 'median', 'p75')}}


def shop_period(data, seller_id, before):
    """One common 10,000-index window across all observed shop products, including unpriced ones."""
    start = max(0, before - RECENT_MARKET_SPAN)
    totals, = data.decision_cached(('shop_period', seller_id, before), lambda: rows(data.connection, f"""
        SELECT count(DISTINCT item_id) AS products_observed_before_cutoff,
               count(DISTINCT item_id) FILTER (WHERE time_index>=?) AS products_in_window,
               coalesce(sum(exposure_records) FILTER (WHERE time_index>=?),0) AS exposure_records,
               coalesce(sum(click_records) FILTER (WHERE time_index>=?),0) AS click_records,
               coalesce(sum(purchase_records) FILTER (WHERE time_index>=?),0) AS purchase_records
        FROM {business_relation(data, 'seller')} WHERE seller_id=? AND time_index<?
        """, [start]*4 + [seller_id, before]))
    return dict(clock='observed_time_index', start_inclusive=start, end_exclusive=before, **totals)


def historical_position(data, seller_id, item_id, before, start=None, *, profile=None, reference=None):
    data.window(before, 1)
    if profile is None:
        profiles = owned_profiles(data, seller_id, before, item_ids=[item_id])
        if not profiles:
            raise PermissionError('Market position requires a previously observed owned product')
        profile, = profiles
    require(profile['seller_id'] == seller_id, 'Foreign market profile')
    automatic = start is None
    if automatic:
        start = profile['recent_start']
    require(type(start) is int and 0 <= start < before, 'Invalid market window')
    if not complete_category(profile):
        return {'item_id': item_id, 'status': 'missing_category', 'semantics': MARKET_NOTE}
    key = (before, start, *(profile[k] for k in CATEGORY_FIELDS))
    if key not in data._market_rows:
        if len(data._market_rows) >= 2:
            data._market_rows.clear()
        data._market_rows[key] = category_rows(data, profile, before, start)
    records = data._market_rows[key]
    own = next(r for r in records if r['item_id'] == item_id)
    peers = [r for r in records if r['seller_id'] != seller_id]
    times = [r['price_time_index'] for r in peers if r['price_raw'] is not None]
    recent = [r for r in peers if r['price_raw'] is not None and r['exposure_records'] is not None]
    if reference is None:
        panel = data._market_panels.get((before, *key[2:]))
        if panel is None:
            _, panel = next(category_panels(data, [key[2:]], before))
        reference = historical_reference(panel, seller_id)
    recent_scope = {'products': len(recent), 'shops': len({r['seller_id'] for r in recent}),
                    'observation_range': observation_range(r['price_time_index'] for r in recent)}
    if len(recent) == 1:
        peer, = recent
        recent_scope['single_observed_offer'] = {k: peer[k] for k in
                                                ('item_id', 'price_raw', 'price_time_index')}
    return {'item_id': item_id, 'status': 'available' if peers else 'no_peers',
            'category_ids': [profile[k] for k in CATEGORY_FIELDS],
            'recent_price_scope': recent_scope, 'historical_reference': reference,
            'expanded_price_sample': extended_price_sample(peers, recent, before, start),
            'peer_price_observation_range': {'earliest_index': min(times) if times else None,
                                             'latest_index': max(times) if times else None},
            **market_position(own, peers, window={'start_inclusive': start, 'end_exclusive': before,
                              'selection': 'item_recent_history_span' if automatic else 'explicit'},
                              clock='observed_time_index'), 'semantics': MARKET_NOTE}


class QuoteSnapshot(list):
    """Immutable internal view, with a bounded delta for incremental consumers."""
    def __init__(self, products, revision, parent, changed, before):
        super().__init__(products[item] for item in sorted(products))
        self.revision, self.parent_revision = revision, parent
        self.changed, self.before = tuple(changed), before


class CategorySnapshots:
    """Bounded exact prefixes for explicit full-history queries."""
    def __init__(self):
        self.categories = OrderedDict()
        self.derived = OrderedDict()
        self.revision = 0

    def snapshot(self, category, before, read):
        cached = self.categories.get(category)
        if cached is not None and before == cached[0]:
            self.categories.move_to_end(category)
            return cached[2]
        if cached is None or before < cached[0]:
            products = {row['item_id']: row for row in read()}
            parent, changed = None, list(products.values())
        else:
            previous, prior_products, previous_view = cached
            products = prior_products.copy()
            parent = previous_view.revision
            changed = read(previous)
            for row in changed:
                prior = products.get(row['item_id'], {})
                if row['last_valid_price'] is None:
                    row['last_valid_price'] = prior.get('last_valid_price')
                    row['last_valid_time'] = prior.get('last_valid_time')
                products[row['item_id']] = row
        self.revision += 1
        view = QuoteSnapshot(products, self.revision, parent, changed, before)
        self.categories[category] = before, products, view
        self.categories.move_to_end(category)
        while len(self.categories) > 8 or (len(self.categories)>1 and
                sum(len(value[1]) for value in self.categories.values())>100000):
            key,_ = self.categories.popitem(last=False)
            self.derived.pop(key,None)
        if len(products)>100000:
            self.categories.pop(category,None); self.derived.pop(category,None)
        return view


def _quote_row(latest, last_valid_price, last_valid_time):
    profile = {k: latest[k] for k in PROFILE_FIELDS}
    index = latest.get('price_time_index', latest.get('time_index'))
    return {**profile, 'price_raw':latest['price_raw'], 'price_time_index':index,
            'price_status':latest['price_status'], 'last_valid_price':last_valid_price,
            'last_valid_time':last_valid_time, '_complete_category':complete_category(profile),
            '_public':{**profile, 'reference_price_raw':latest['price_raw'],
                       'reference_price_time_index':index, 'price_status':latest['price_status']}}


def category_snapshot(data, category, before, *, start=None):
    """Read latest quotes after a SQL time filter; full history remains explicit.

    A bounded window needs no older last-valid price or rolling expiry cache.
    The same per-decision result serves the market summary and competitors.
    """
    if start is not None:
        require(type(start) is int and 0 <= start < before, 'Invalid quote window')
    category = tuple(category)
    def read(after=None):
        lower = start if start is not None else after
        fields = ",".join(PROFILE_FIELDS)
        packed = ",".join(f"{field} := {field}" for field in PROFILE_FIELDS)
        valid = (", arg_max(price,time_index) AS last_valid_price, "
                 "max(time_index) FILTER (WHERE price IS NOT NULL) AS last_valid_time") if start is None else ""
        records = rows(data.connection, f"""
            WITH observations AS (
                SELECT {fields}, time_index, price_status,
                       CASE WHEN price_status='observed' AND isfinite(price_raw) AND price_raw>0
                            THEN price_raw END AS price
                FROM {business_relation(data, 'category')}
                WHERE {' AND '.join(field + '=?' for field in CATEGORY_FIELDS)} AND time_index<?
                  {'AND time_index>=?' if lower is not None else ''}
            )
            SELECT arg_max(struct_pack({packed}, price_raw := price,
                       price_time_index := time_index, price_status := price_status), time_index) AS latest {valid}
            FROM observations GROUP BY item_id ORDER BY item_id
            """, [*category, before] + ([lower] if lower is not None else []))
        return [_quote_row(r['latest'],r.get('last_valid_price'),r.get('last_valid_time')) for r in records]
    if start is not None:
        return data.decision_cached(('recent_category_snapshot', start, before, category), read)
    return data.decision_cached(('category_snapshot', before, category),
        lambda: data.category_snapshots.snapshot(category, before, read)
        if data.category_snapshots is not None else read())


def _decision_positions(data, grouped, seller_id, before):
    """Yield targets by category while their shared quote snapshot is recent."""
    start = max(0, before - RECENT_MARKET_SPAN)
    for category, profiles in grouped.items():
        def summarize():
            peers = [row for row in category_snapshot(data, category, before, start=start)
                     if row['seller_id'] != seller_id]
            recent = [r for r in peers if r["price_raw"] is not None]
            summary = {
                "status": "available" if peers else "no_recent_quotes",
                "category_ids": list(category),
                "window": {"start_inclusive": start, "end_exclusive": before},
                "recent_price_scope": {"products": len(recent), "shops": len({r['seller_id'] for r in recent}),
                                       "observation_range": observation_range(r['price_time_index'] for r in recent)},
                "price_raw": {"median": distribution([r['price_raw'] for r in recent], None)['median']},
            }
            return summary
        summary = data.decision_cached(('decision_summary', seller_id, category, before), summarize)
        for profile in profiles:
            item = profile["item_id"]
            yield item, dict(deepcopy(summary), item_id=item, semantics=MARKET_NOTE)


def _requested_profiles(data, seller_id, item_ids, before, profiles=None):
    data.window(before, 1)
    requested = set(item_ids)
    require(len(item_ids) == len(requested), 'Duplicate market IDs')
    if profiles is None:
        profiles = owned_profiles(data, seller_id, before, item_ids=item_ids)
    else:
        require(all(p['seller_id'] == seller_id and p['time_index'] < before for p in profiles),
                'Invalid observed market profiles')
    selected = {p['item_id']: p for p in profiles if p['item_id'] in requested}
    require(requested <= selected.keys(), 'Missing owned market product')
    return selected


def iter_decision_positions(data, seller_id, item_ids, before, *, profiles=None):
    """Let pricing read competitor examples before moving to the next category."""
    selected = _requested_profiles(data, seller_id, item_ids, before, profiles)
    grouped = {}
    for item, profile in selected.items():
        if not complete_category(profile):
            yield item, dict(item_id=item, status="missing_category", semantics=MARKET_NOTE)
            continue
        category = tuple(profile[k] for k in CATEGORY_FIELDS)
        grouped.setdefault(category, []).append(profile)
    for item, position in _decision_positions(data, grouped, seller_id, before):
        data.decision_cached(('product_profile', item, before), lambda: selected[item])
        yield item, position


def historical_positions(data, seller_id, item_ids, before, *, decision_only=False, profiles=None):
    if decision_only:
        return dict(iter_decision_positions(data, seller_id, item_ids, before, profiles=profiles))
    require(profiles is None, 'Invalid observed market profiles')
    selected = _requested_profiles(data, seller_id, item_ids, before)
    grouped, result = {}, {}
    for profile in selected.values():
        category = tuple(profile[k] for k in CATEGORY_FIELDS)
        grouped.setdefault(category, []).append(profile)
    # Category batches share source scans. No SQL is executed inside this loop:
    # explicit profiles/windows and the preloaded panel keep the streaming cursor valid.
    for key, panel in category_panels(data, list(grouped), before):
        data._market_panels = {key: panel}
        reference = historical_reference(panel, seller_id)
        for p in grouped[key[1:]]:
            value = historical_position(data, seller_id, p['item_id'], before,
                                        p['recent_start'], profile=p, reference=reference)
            if 'window' in value:
                value['window']['selection'] = 'item_recent_history_span'
            result[p['item_id']] = value
    # Missing category IDs cannot match SQL tuple equality.
    for item, profile in selected.items():
        if item not in result:
            result[item] = historical_position(data, seller_id, item, before,
                                               profile['recent_start'], profile=profile)
    return result



ALIAS = 'seller_market_cache'
TABLES = {'category': 'by_category', 'seller': 'by_seller'}


def source_hashes(config):
    return {p.name: sha256(p) for p in business_parts(config)}


def build_cache(config, directory):
    directory = Path(directory)
    require(not directory.exists(), 'Market cache destination already exists')
    sources = source_hashes(config)
    directory.mkdir(parents=True)
    database = directory / 'market.duckdb'
    source = literal(config.data / 'seller_business/part_*.parquet')
    counts = {}
    with duckdb.connect(str(database)) as connection:
        connection.execute("SET memory_limit='2GB'")
        connection.execute('SET threads=2')
        connection.execute('SET preserve_insertion_order=false')
        connection.execute('SET enable_progress_bar=false')
        for table, order in (
            ('by_category', 'category_level1_id,category_level2_id,category_level3_id,time_index,item_id'),
            ('by_seller', 'seller_id,time_index,item_id'),
        ):
            connection.execute(f'CREATE TABLE {table} AS SELECT * FROM read_parquet({source}) ORDER BY {order}')
            counts[table] = connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
        connection.execute('CHECKPOINT')
    require(source_hashes(config) == sources, 'Market sources changed during cache construction')
    require(counts['by_category'] == counts['by_seller'], 'Market table counts differ')
    manifest = dict(schema='seller-market-cache-v1', dataset=config.dataset, sources=sources,
                    database_sha256=sha256(database), rows=counts)
    # This manifest is the completion marker; interrupted databases are not usable.
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def attach_cache(connection, config, directory, *, verify_hashes=True):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    require(manifest['schema'] == 'seller-market-cache-v1', 'Invalid market cache schema')
    require(manifest['dataset'] == config.dataset, 'Market cache dataset differs')
    if verify_hashes:
        require(manifest['sources'] == source_hashes(config), 'Market cache sources differ')
        require(sha256(directory / 'market.duckdb') == manifest['database_sha256'], 'Market cache changed')
    connection.execute(f'ATTACH {literal(directory / "market.duckdb")} AS {ALIAS} (READ_ONLY)')


def business_relation(data, layout):
    require(layout in TABLES, 'Unknown market table layout')
    present = getattr(data, '_market_cache_present', None)
    if present is None:
        business_parts(data.config)
        present = bool(data.connection.execute(
            'SELECT count(*) FROM duckdb_databases() WHERE database_name=?', [ALIAS]
        ).fetchone()[0])
        data._market_cache_present = present
    if present:
        return f'{ALIAS}.{TABLES[layout]}'
    return f'read_parquet({literal(data.config.data / "seller_business/part_*.parquet")})'


def purchase_relation(connection):
    """Materialize public purchases once per read-only catalog connection.

    The temporary table belongs to the connection, so it cannot outlive or be
    reused across source catalogs. Every consumer must still apply its cutoff.
    """
    name = "context_public_purchases"
    try:
        connection.execute(f'SELECT 1 FROM temp.main.{name} LIMIT 0')
    except duckdb.CatalogException:
        connection.execute(f"""CREATE TEMP TABLE {name} AS
            SELECT item_id,seller_id,category_level1_id,category_level2_id,category_level3_id,
                   time_index,price_raw
            FROM ranking_observations JOIN products USING(item_id)
            WHERE is_purchased=1 ORDER BY time_index,item_id""")
    return name


def main():
    import argparse
    from contextlib import closing
    from agent_simulation.data.catalog import Catalog

    parser = argparse.ArgumentParser(description="Build an exact read-only market cache")
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / 'manifest.json').read_text())
    with closing(Catalog(manifest['catalog'])) as catalog:
        print(json.dumps(build_cache(catalog.config, args.output), indent=2))


if __name__ == '__main__':
    main()
