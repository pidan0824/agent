"""Sample Customer requests and their Seller supply within each source split.

Sampling limits supervision only. Customer units are whole requests, before
Query/Response expansion. Original temporal boundaries and full history remain.
"""
import argparse
from contextlib import closing
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import shutil
import tempfile

from agent_simulation.data.common import literal, require, sha256
from agent_simulation.data.catalog import Catalog
from agent_simulation.data.history_records import customer_history_limit
from agent_simulation.logs import write_report
from agent_simulation.datasets.labels import ordered_splits, release_splits

ROLES = ('customer', 'seller')
SCHEMA = 'sft-split-candidates-v2'


# History features shared across source splits.

def label_query(labels):
    """Tag source membership only for sampling; it is removed from exported candidates."""
    return " UNION ALL ".join(
        f"SELECT *, {literal(split)} AS sampling_split FROM read_parquet({literal(path)})"
        for split, path in labels.items()
    )


def quotas(counts, budget):
    """Largest-remainder allocation preserves population proportions to rounding."""
    require(type(budget) is int and 0 < budget <= sum(counts.values()), 'Invalid candidate budget')
    require(all(type(n) is int and n > 0 for n in counts.values()), 'Invalid stratum size')
    total = sum(counts.values())
    allocation = {k: budget*n//total for k,n in counts.items()}
    remaining = budget - sum(allocation.values())
    for k in sorted(counts, key=lambda k: (-(budget*counts[k] % total),k))[:remaining]:
        allocation[k] += 1
    return allocation


def seller_features(connection, labels, output):
    """Bound database memory by actor; all feature windows end before the label."""
    source = label_query(labels)
    for bucket in range(32):
        path = output / f'seller-{bucket:02d}.parquet'
        connection.execute(f"""COPY (
            WITH history AS (
                SELECT seller_id,item_id,time_index,category_level1_id,
                    count(*) OVER w AS prior_points,
                    last_value(CASE WHEN price_status='observed' AND isfinite(price_raw)
                        AND price_raw>0 THEN price_raw END IGNORE NULLS) OVER w AS prior_price
                FROM seller_history WHERE seller_id % 32 = {bucket}
                WINDOW w AS (PARTITION BY seller_id,item_id ORDER BY time_index
                    RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
            ) SELECT l.*,h.category_level1_id,h.prior_points,h.prior_price,
                CASE WHEN l.target_price_raw=h.prior_price THEN 'same'
                     WHEN l.target_price_raw>h.prior_price THEN 'up' ELSE 'down' END AS action,
                CASE WHEN h.prior_points<3 THEN 'short' ELSE 'established' END AS history_band,
                l.seller_id AS actor_id, l.sample_id AS record_id
            FROM ({source}) l JOIN history h
                ON l.seller_id=h.seller_id AND l.item_id=h.item_id
                AND l.history_cutoff_exclusive=h.time_index
            WHERE l.seller_id % 32 = {bucket}
        ) TO {literal(path)} (FORMAT PARQUET, COMPRESSION ZSTD)""")


def customer_features(connection, labels, output):
    """Compute all split features in one scan, preserving each request's history scope."""
    history_count = ("CASE WHEN split='train' THEN count(*) FILTER (WHERE split='train') OVER w "
                     "ELSE count(*) OVER w END")
    connection.execute(f"""COPY (
        WITH history AS (
            SELECT session_id, {history_count} AS prior_points
            FROM requests
            WINDOW w AS (PARTITION BY user_id ORDER BY time_index
                RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        ) SELECT l.*,h.prior_points,
            CASE WHEN len(l.purchased_item_ids)>0 THEN 'purchase'
                 WHEN len(l.clicked_item_ids)>0 THEN 'click_only' ELSE 'no_action' END AS action,
            CASE WHEN h.prior_points=0 THEN 'cold' WHEN h.prior_points<={customer_history_limit()} THEN 'recent'
                 ELSE 'retrieval' END AS history_band,
            l.user_id AS actor_id, CAST(l.session_id AS VARCHAR) AS record_id
        FROM ({label_query(labels)}) l JOIN history h USING(session_id)
    ) TO {literal(output / 'customer-00.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)""")


# Proportional quotas and deterministic within-stratum selection.

def prepare_population(connection, features, role, split, strata, expected_count):
    source = features / f'{role}-*.parquet'
    expression = "||':'||".join(f"coalesce(CAST({field} AS VARCHAR),'unknown')" for field in strata)
    connection.execute(f"""
        CREATE OR REPLACE TEMP VIEW candidates AS
        SELECT * EXCLUDE (sampling_split), {expression} AS stratum
        FROM read_parquet({literal(source)}) WHERE sampling_split={literal(split)}
    """)
    invalid_price_count = (
        'count(*) FILTER (WHERE prior_price IS NULL OR prior_price<=0 OR NOT isfinite(prior_price))'
        if role == 'seller' else '0'
    )
    count, distinct, bad = connection.execute(f"""
        SELECT count(*), count(DISTINCT record_id), {invalid_price_count} FROM candidates
    """).fetchone()
    require(count == distinct == expected_count, 'Feature join lost or duplicated labels')
    require(bad == 0, 'Missing earlier reliable Seller price')
    connection.execute('CREATE OR REPLACE TEMP TABLE selected_ids(record_id VARCHAR, selection_hash VARCHAR)')
    return dict(connection.execute('SELECT stratum,count(*) FROM candidates GROUP BY stratum').fetchall())


def customer_quotas(connection, pool, target_quotas, count, split):
    eligible_sizes = dict(connection.execute(
        f'SELECT stratum,count(*) FROM {pool} GROUP BY stratum'
    ).fetchall())
    require(sum(eligible_sizes.values()) >= count,
            f'{split}: too few Customer requests with Seller supply for the requested budget')
    eligible_quotas = {k: min(n, eligible_sizes.get(k, 0)) for k, n in target_quotas.items()}
    shortfall = count - sum(eligible_quotas.values())
    if shortfall:
        capacity = {k: n - eligible_quotas[k] for k, n in eligible_sizes.items() if n > eligible_quotas[k]}
        for k, n in quotas(capacity, shortfall).items():
            eligible_quotas[k] += n
    return eligible_quotas


def select_by_stratum(connection, path, sizes, remaining_quotas, count, seed, *, pool='candidates'):
    """Fill remaining quotas from the prepared pool and export the complete selection."""
    # Sort narrow IDs one stratum at a time, not millions of full label records.
    for stratum, quota in sorted(remaining_quotas.items()):
        if not quota:
            continue
        connection.execute(f"""INSERT INTO selected_ids
            WITH keyed AS (
                SELECT record_id,sha256(record_id||':'||CAST(? AS VARCHAR)) AS selection_hash
                FROM {pool} ANTI JOIN selected_ids USING(record_id) WHERE stratum=?
            ) SELECT record_id,selection_hash
            FROM keyed ORDER BY selection_hash,record_id LIMIT ?
        """, [seed,stratum,quota])
    connection.execute(f"""COPY (
        SELECT c.*,s.selection_hash FROM candidates c JOIN selected_ids s USING(record_id)
        ORDER BY actor_id,history_cutoff_exclusive,record_id
    ) TO {literal(path)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
    selected = dict(connection.execute(
        'SELECT stratum,count(*) FROM read_parquet(?) GROUP BY stratum', [str(path)]
    ).fetchall())
    require(sum(selected.values()) == count, 'Candidate count mismatch')
    return dict(file=path.name, sha256=sha256(path), records=count,
                available_by_stratum=sizes, selected_by_stratum={k: selected.get(k, 0) for k in sizes})


def prepare_supply(connection, features, seller_labels, split):
    """Find the first supervised quote in this split for each product."""
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE supply_quotes AS
        SELECT item_id, min(history_cutoff_exclusive) AS first_quote,
               arg_min(sample_id, (history_cutoff_exclusive, sample_id)) AS record_id
        FROM read_parquet(?) GROUP BY item_id
    """, [str(seller_labels)])
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE eligible_customer AS
        SELECT c.record_id FROM read_parquet(?) c
        WHERE c.sampling_split=? AND len(c.impressed_item_ids)>0
          AND NOT EXISTS (
              SELECT 1 FROM unnest(c.impressed_item_ids) AS shown(item_id)
              LEFT JOIN supply_quotes s USING(item_id)
              WHERE s.first_quote IS NULL OR s.first_quote>c.history_cutoff_exclusive
          )
    """, [str(features / 'customer-00.parquet'), split])


def reserve_supply(connection, customer_path, target_quotas, count, seed, split):
    connection.execute("""
        INSERT INTO selected_ids
        SELECT s.record_id,sha256(s.record_id||':'||CAST(? AS VARCHAR)) FROM supply_quotes s
        SEMI JOIN (
            SELECT DISTINCT unnest(impressed_item_ids) AS item_id FROM read_parquet(?)
        ) shown USING(item_id)
    """, [seed, str(customer_path)])
    reserved = dict(connection.execute(
        'SELECT stratum,count(*) FROM candidates SEMI JOIN selected_ids USING(record_id) GROUP BY stratum'
    ).fetchall())
    remaining = count - sum(reserved.values())
    require(remaining >= 0, f'{split}: required Seller supply exceeds the requested Seller budget')
    remaining_quotas = {k: max(0, n - reserved.get(k, 0)) for k, n in target_quotas.items()}
    if sum(remaining_quotas.values()) > remaining:
        unfilled = {k: n for k, n in remaining_quotas.items() if n}
        return quotas(unfilled, remaining) if remaining else {}
    return remaining_quotas


def supply_coverage(connection, customer_path, seller_path):
    """Check every exposure against the quotes that were actually selected."""
    requests, exposures, items, missing = connection.execute("""
        WITH shown AS (
            SELECT session_id, history_cutoff_exclusive, unnest(impressed_item_ids) AS item_id
            FROM read_parquet(?)
        ), selected AS (
            SELECT item_id, min(history_cutoff_exclusive) AS first_quote
            FROM read_parquet(?) GROUP BY item_id
        )
        SELECT count(DISTINCT session_id), count(*), count(DISTINCT item_id),
               count(*) FILTER (WHERE first_quote IS NULL OR first_quote>history_cutoff_exclusive)
        FROM shown LEFT JOIN selected USING(item_id)
    """, [str(customer_path), str(seller_path)]).fetchone()
    require(missing == 0, 'Customer exposure lacks selected Seller supply at its request index')
    return dict(requests=requests, exposures=exposures, products=items)


# Configuration and atomic candidate publication.

def read_config(path):
    config = json.loads(Path(path).read_text())
    require(config.get('schema') == 'sft-split-sampling-config-v1', 'Invalid sampling config')
    require(type(config.get('seed')) is int, 'Explicit integer seed required')
    ordered_splits(config['fractions'])
    for value in config['fractions'].values():
        require(type(value) in (float, int) and 0 < value <= 1, 'Fractions must be in (0,1]')
    require(set(config['strata']) == set(ROLES), 'Configure strata for both roles')
    for role, fields in config['strata'].items():
        allowed = {'action','history_band'} | ({'category_level1_id'} if role == 'seller' else set())
        require(isinstance(fields,list) and fields and len(fields) == len(set(fields))
                and set(fields) <= allowed, 'Unsupported stratum fields')
    return config


def fraction_count(population, fraction):
    """Round each split total to nearest integer; no forced rare-stratum oversampling."""
    return int((Decimal(population) * Decimal(str(fraction))).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def select(dataset, output, config_path):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    config = read_config(config_path)
    splits = release_splits(dataset)
    require(set(config['fractions']) == set(splits), 'Sampling splits differ from dataset release')
    require(not output.exists(), 'Choose a new candidate directory')
    manifest = json.loads((dataset/'manifest.json').read_text())
    require(manifest['schema'] == 'kuaisearch-on-demand-sft-v1', 'Expected complete frozen label pools')
    label_hashes = {}
    for entry in manifest['files']:
        path = dataset / entry['path']
        require(sha256(path) == entry['sha256'], 'Frozen labels changed')
        label_hashes[path] = entry['sha256']
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'-', dir=output.parent))
    try:
        roles = {role:{} for role in ROLES}
        supply = {}
        with closing(Catalog(manifest['catalog'])) as catalog:
            c = catalog.connection
            c.execute('SET threads=1')
            c.execute('SET preserve_insertion_order=false')
            temp = staging / 'working'
            features = temp / 'features'
            features.mkdir(parents=True)
            c.execute(f'SET temp_directory={literal(temp / "spill")}')
            labels = {
                role: {split: dataset / role / split / 'labels.parquet' for split in splits}
                for role in ROLES
            }
            customer_features(c, labels['customer'], features)
            seller_features(c, labels['seller'], features)
            for split in splits:
                fraction = config['fractions'][split]
                seed = config['seed']
                prepare_supply(c, features, labels['seller'][split], split)

                customer_population = manifest['counts']['customer'][split]
                customer_count = fraction_count(customer_population, fraction)
                customer_sizes = prepare_population(
                    c, features, 'customer', split, config['strata']['customer'], customer_population
                )
                customer_target = quotas(customer_sizes, customer_count) if customer_count else dict.fromkeys(customer_sizes, 0)
                customer_pool = '(SELECT c.* FROM candidates c SEMI JOIN eligible_customer USING(record_id))'
                eligible_quotas = customer_quotas(c, customer_pool, customer_target, customer_count, split)
                customer_path = staging / f'customer-{split}-candidates.parquet'
                customer = select_by_stratum(
                    c, customer_path, customer_sizes, eligible_quotas, customer_count, seed, pool=customer_pool
                )

                seller_population = manifest['counts']['seller'][split]
                seller_count = fraction_count(seller_population, fraction)
                seller_sizes = prepare_population(
                    c, features, 'seller', split, config['strata']['seller'], seller_population
                )
                seller_target = quotas(seller_sizes, seller_count) if seller_count else dict.fromkeys(seller_sizes, 0)
                remaining_quotas = reserve_supply(c, customer_path, seller_target, seller_count, seed, split)
                seller_path = staging / f'seller-{split}-candidates.parquet'
                seller = select_by_stratum(c, seller_path, seller_sizes, remaining_quotas, seller_count, seed)

                for role, entry, population in (
                    ('customer', customer, customer_population), ('seller', seller, seller_population)
                ):
                    entry.update(population_records=population, requested_fraction=fraction,
                                 actual_fraction=entry['records']/population if population else 0,
                                 source_labels_sha256=label_hashes[labels[role][split]])
                    roles[role][split] = entry
                    print(f"{role} {split}: {entry['records']}/{population}", flush=True)

                coverage = supply_coverage(c, customer_path, seller_path)
                require(coverage['requests'] == roles['customer'][split]['records'],
                        'Customer candidate has no displayed products')
                supply[split] = coverage
            for role in ROLES:
                paths=[str(staging/roles[role][split]['file']) for split in splits]
                n, unique=c.execute('SELECT count(*),count(distinct record_id) FROM read_parquet(?)',[paths]).fetchone()
                require(n == unique, 'Selected splits overlap')
        shutil.rmtree(temp)
        result = dict(feature_policy={"customer_recent_requests": customer_history_limit(),
                                      "seller_short_history_points": 3}, schema=SCHEMA, status='pending_context_validation', seed=config['seed'],
            config=config, dataset_manifest_sha256=sha256(dataset/'manifest.json'),
            roles=roles,
            sampling_unit={'customer':'complete request before Query/Response expansion','seller':'item-time price record'},
            selection=('Rounded split budgets, proportional quotas and SHA256(record_id:seed) order. '
                       'Select Customer requests first, then reserve their earliest Seller quotes.'),
            supply=dict(rule='Each displayed product has a selected Seller quote in the same split at or before the request index.',
                        splits=supply),
            history='Use all eligible prior history.',
            evaluation='Sample official test records within their source split.')
        write_report(staging/'manifest.json', result)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return result


def changed_seller_records(connection, keys_sql, parameters=()):
    """Compare target quotes with the strictly earlier latest valid quote."""
    return connection.execute(f"""
        WITH keys AS ({keys_sql}), targets AS (
            SELECT k.*, h.price_raw FROM keys k
            JOIN seller_history h USING (seller_id, item_id, time_index)
        ), earlier AS (
            SELECT h.seller_id, h.item_id, h.time_index, h.price_raw
            FROM seller_history h SEMI JOIN keys k USING (seller_id, item_id)
            WHERE h.price_status='observed' AND h.price_raw>0 AND isfinite(h.price_raw)
        )
        SELECT t.seller_id, t.item_id, t.time_index FROM targets t
        ASOF JOIN earlier h ON t.seller_id=h.seller_id AND t.item_id=h.item_id
                              AND t.time_index>h.time_index
        WHERE t.price_raw<>h.price_raw
    """, parameters).fetchall()


def select_seller_training(dataset, candidates, output, *, keep_ratio=4):
    """Thin only unchanged Seller Train labels before any context/model work."""
    dataset, candidates, output = map(lambda p: Path(p).resolve(), (dataset, candidates, output))
    require(type(keep_ratio) is int and keep_ratio >= 0, 'Seller keep ratio must be a nonnegative integer')
    require(not output.exists(), 'Choose a new candidate directory')
    source = json.loads(candidates.read_text())
    require(source['schema'] == SCHEMA, 'Expected split candidate manifest')
    require(source['dataset_manifest_sha256'] == sha256(dataset/'manifest.json'), 'Candidate dataset differs')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.'+output.name+'-', dir=output.parent) as temp:
        staging = Path(temp)/'release'
        staging.mkdir()
        roles = {role: {} for role in source['roles']}
        for role, splits in source['roles'].items():
            for split, entry in splits.items():
                if (role, split) != ('seller', 'train'):
                    shutil.copyfile(candidates.parent/entry['file'], staging/entry['file'])
                    roles[role][split] = dict(entry)
        filename = source['roles']['seller']['train']['file']
        manifest = json.loads((dataset/'manifest.json').read_text())
        with closing(Catalog(manifest['catalog'])) as catalog:
            c = catalog.connection
            c.execute('CREATE TEMP TABLE train_candidates AS SELECT * FROM read_parquet(?)',
                      [str(candidates.parent/filename)])
            changed = changed_seller_records(c, '''SELECT seller_id,item_id,
                history_cutoff_exclusive AS time_index FROM train_candidates''')
            require(changed, 'No observed price changes in the selected Seller Train window')
            c.execute('CREATE TEMP TABLE changed_seller(seller_id BIGINT,item_id BIGINT,time_index BIGINT)')
            c.executemany('INSERT INTO changed_seller VALUES (?,?,?)', changed)
            c.execute('''CREATE TEMP TABLE marked AS SELECT t.*, c.time_index IS NOT NULL AS price_changed
                FROM train_candidates t LEFT JOIN changed_seller c
                ON t.seller_id=c.seller_id AND t.item_id=c.item_id AND t.history_cutoff_exclusive=c.time_index''')
            total, changes = c.execute('SELECT count(*),count(*) FILTER (WHERE price_changed) FROM marked').fetchone()
            kept = min(total-changes, changes*keep_ratio)
            c.execute(f'''COPY (
                WITH kept_unchanged AS (
                    SELECT record_id FROM marked WHERE NOT price_changed
                    ORDER BY sha256(record_id || ':' || {literal(source['seed'])}),record_id LIMIT {kept}
                )
                SELECT * EXCLUDE(price_changed) FROM marked
                WHERE price_changed OR record_id IN (SELECT record_id FROM kept_unchanged)
                ORDER BY actor_id,history_cutoff_exclusive,item_id
            ) TO {literal(staging/filename)} (FORMAT PARQUET)''')
        roles['seller']['train'] = dict(file=filename, records=changes+kept, sha256=sha256(staging/filename))
        result = dict(schema=SCHEMA, seed=source['seed'], roles=roles,
                      dataset_manifest_sha256=source['dataset_manifest_sha256'],
                      source_candidates_sha256=sha256(candidates),
                      seller_train_sampling=dict(unchanged_per_changed=keep_ratio, source_records=total,
                          source_changed=changes, source_unchanged=total-changes,
                          kept_changed=changes, kept_unchanged=kept),
                      selection='All changed Seller Train quotes; unchanged quotes ordered by SHA256(record_id:seed). Customer and Test unchanged.',
                      history='Only supervision targets are filtered; all eligible earlier facts remain available.')
        write_report(staging/'manifest.json', result)
        staging.rename(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--config', type=Path, help='Build the original stratified candidate pool')
    mode.add_argument('--candidates', type=Path, help='Keep Customer/Test and thin unchanged Seller Train targets')
    parser.add_argument('--seller-keep-ratio', type=int, default=4,
                        help='Maximum unchanged Seller Train targets per changed target (default: 4)')
    args = parser.parse_args()
    if args.candidates:
        result = select_seller_training(args.dataset, args.candidates, args.output, keep_ratio=args.seller_keep_ratio)
        print(json.dumps(result['seller_train_sampling'], ensure_ascii=False))
    else:
        select(args.dataset, args.output, args.config)


if __name__ == '__main__':
    main()
