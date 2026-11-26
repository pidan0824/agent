"""Select representative Customer scenes from the complete official Train pool.

No LLM inference, prompt reconstruction or current-target-conditioned history.
Quality ranking concerns observed input completeness, not teacher agreement.
"""
import argparse
from contextlib import closing
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from agent_simulation.data.catalog import Catalog
from agent_simulation.data.common import literal, require, sha256


def rank_population(connection, labels, *, per_scene, seed):
    require(type(per_scene) is int and per_scene > 0, "Positive per-scene budget required")
    source = literal(labels)
    connection.execute(f"CREATE OR REPLACE TEMP VIEW selection_labels AS SELECT * FROM read_parquet({source})")
    count, unique = connection.execute(
        "SELECT count(*),count(DISTINCT session_id) FROM selection_labels").fetchone()
    require(count > 0 and count == unique, "Empty or duplicate Customer label pool")
    require(connection.execute("SELECT count(*) FROM selection_labels WHERE split IS DISTINCT FROM 'train'").fetchone()[0] == 0,
            "Scene selection accepts Train only")
    # Same-time records are excluded. Future/Test events cannot improve quality.
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE selection_history AS
        SELECT session_id, count(*) OVER w AS prior_requests,
            count(*) FILTER (WHERE len(clicked_item_ids)>0 OR len(purchased_item_ids)>0) OVER w AS prior_interactions
        FROM requests WHERE split='train'
        WINDOW w AS (PARTITION BY user_id ORDER BY time_index
                     RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    """)
    # Inspect every displayed item uniformly, never only the observed positives.
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE selection_titles AS
        SELECT l.session_id,count(*) FILTER (
            WHERE p.item_id IS NOT NULL AND nullif(trim(p.item_title),'') IS NOT NULL
              AND lower(trim(p.item_title)) NOT IN ('unknown','null')) AS titled_items
        FROM selection_labels l, unnest(l.impressed_item_ids) AS shown(item_id)
        LEFT JOIN products p USING(item_id) GROUP BY l.session_id
    """)
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE selection_features AS
        WITH features AS (
            SELECT l.session_id,l.user_id AS actor_id,l.history_cutoff_exclusive,
                coalesce(h.prior_requests,0) AS prior_requests,
                coalesce(h.prior_interactions,0) AS prior_interactions,
                len(l.impressed_item_ids) AS exposure_count,coalesce(t.titled_items,0) AS titled_items,
                CASE WHEN len(l.purchased_item_ids)>0 AND len(l.clicked_item_ids)>0 THEN 'click_purchase'
                     WHEN len(l.purchased_item_ids)>0 THEN 'purchase_only'
                     WHEN len(l.clicked_item_ids)>0 THEN 'click_only' ELSE 'no_action' END AS behavior,
                CASE WHEN len(list_distinct(list_concat(l.clicked_item_ids,l.purchased_item_ids)))>1
                     THEN 'multiple' ELSE 'zero_or_one' END AS interaction_band,
                CASE WHEN h.prior_requests=0 THEN 'cold'
                     WHEN h.prior_requests<3 THEN 'short' ELSE 'established' END AS history_band,
                CASE WHEN len(l.impressed_item_ids)<=12 THEN '1-12'
                     WHEN len(l.impressed_item_ids)<=32 THEN '13-32'
                     WHEN len(l.impressed_item_ids)<=128 THEN '33-128' ELSE '129+' END AS exposure_band,
                CASE WHEN h.session_id IS NULL THEN 'missing_train_request'
                     WHEN NOT EXISTS (SELECT 1 FROM users u WHERE u.user_id=l.user_id) THEN 'missing_user'
                     WHEN l.query_target IS NULL OR trim(l.query_target)='' THEN 'missing_query'
                     WHEN l.impressed_item_ids IS NULL OR len(l.impressed_item_ids)=0 THEN 'empty_exposure'
                     WHEN l.clicked_item_ids IS NULL OR l.purchased_item_ids IS NULL
                       OR NOT list_has_all(l.impressed_item_ids,l.clicked_item_ids)
                       OR NOT list_has_all(l.impressed_item_ids,l.purchased_item_ids) THEN 'invalid_action_ids'
                     WHEN len(list_distinct(l.impressed_item_ids))<>len(l.impressed_item_ids)
                       OR len(list_distinct(l.clicked_item_ids))<>len(l.clicked_item_ids)
                       OR len(list_distinct(l.purchased_item_ids))<>len(l.purchased_item_ids) THEN 'duplicate_item_ids'
                     WHEN coalesce(t.titled_items,0)=0 THEN 'no_exposure_titles'
                     ELSE NULL END AS exclusion_reason,
                sha256(CAST(l.session_id AS VARCHAR)||':'||CAST({int(seed)} AS VARCHAR)) AS selection_hash
            FROM selection_labels l LEFT JOIN selection_history h USING(session_id)
            LEFT JOIN selection_titles t USING(session_id)
        )
        SELECT *,behavior||'/'||interaction_band||'/'||history_band||'/'||exposure_band AS stratum,
            titled_items::DOUBLE/greatest(exposure_count,1) AS title_coverage,
            titled_items=exposure_count AS complete_titles
        FROM features
    """)
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE selection_ranked AS
        WITH actors AS (
            SELECT *,row_number() OVER (
                PARTITION BY stratum,actor_id ORDER BY title_coverage DESC,(prior_interactions>0) DESC,selection_hash,session_id
            ) AS actor_rank
            FROM selection_features WHERE exclusion_reason IS NULL
        )
        SELECT session_id,row_number() OVER (
            PARTITION BY stratum ORDER BY complete_titles DESC,actor_rank,
            title_coverage DESC,(prior_interactions>0) DESC,selection_hash,session_id
        ) AS scene_rank FROM actors
    """)
    connection.execute(f"""
        CREATE OR REPLACE TEMP TABLE scene_selection AS
        SELECT f.*,r.scene_rank,coalesce(r.scene_rank<={per_scene},false) AS selected
        FROM selection_features f LEFT JOIN selection_ranked r USING(session_id)
    """)
    strata = connection.execute("""
        SELECT stratum,count(*) AS total,count(*) FILTER (WHERE exclusion_reason IS NULL) AS eligible,
            count(*) FILTER (WHERE selected) AS selected,
            count(DISTINCT actor_id) FILTER (WHERE selected) AS selected_users
        FROM scene_selection GROUP BY stratum ORDER BY stratum
    """).fetchall()
    excluded = dict(connection.execute("""
        SELECT exclusion_reason,count(*) FROM scene_selection
        WHERE exclusion_reason IS NOT NULL GROUP BY exclusion_reason
    """).fetchall())
    return {"population": count, "selected": sum(r[3] for r in strata),
            "exclusions": excluded, "strata": [
                dict(zip(("stratum","total","eligible","selected","selected_users"), r)) for r in strata]}


def read_selection(directory, *, dataset, split):
    require(split == "train", "Training scene selection cannot be used for Test")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    require(manifest.get("schema") == "customer-train-scene-selection-v1", "Unknown scene selection")
    require(manifest.get("split") == "train" and manifest.get("population_scope") == "complete_official_train",
            "Selection is not complete-Train derived")
    require(manifest["dataset_manifest_sha256"] == sha256(Path(dataset) / "manifest.json"),
            "Selection label release differs")
    path = directory / "selected.parquet"
    require(sha256(path) == manifest["selected_sha256"], "Selected scene IDs changed")
    import duckdb
    with closing(duckdb.connect()) as connection:
        ids = [r[0] for r in connection.execute(
            "SELECT session_id FROM read_parquet(?) ORDER BY session_id", [str(path)]).fetchall()]
    require(ids and len(ids) == len(set(ids)) == manifest["selected"], "Invalid scene selection IDs")
    return ids, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-scene", type=int, required=True, help="Maximum original requests per behavior/history/exposure stratum")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    require(args.per_scene > 0 and not args.output.exists(), "Use a positive budget and new output directory")
    release = json.loads((args.dataset / "manifest.json").read_text())
    require(not release.get("time_window_experiment"), "Use the complete Lite Train release, not a time-window subset")
    require(release.get("dataset") == "KuaiSearch Lite", "This selector expects KuaiSearch Lite")
    labels = args.dataset / "customer/train/labels.parquet"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix=".customer-selection-") as temporary:
        stage = Path(temporary) / "selection"
        stage.mkdir()
        with closing(Catalog(release["catalog"])) as catalog:
            require(sha256(catalog.directory / "manifest.json") == release["catalog_manifest_sha256"],
                    "Catalog changed")
            require(not catalog.manifest.get("time_window_experiment"), "Use the complete source catalog")
            c = catalog.connection
            c.execute(f"SET temp_directory={literal(Path(temporary) / 'spill')}")
            # Exact set comparison avoids accepting a renamed/downsampled label release.
            expected = c.execute("SELECT count(*) FROM customer_splits WHERE assigned_split='train'").fetchone()[0]
            actual = c.execute("SELECT count(*),count(DISTINCT session_id) FROM read_parquet(?)",
                               [str(labels)]).fetchone()
            require(actual == (expected, expected), "Label pool does not cover complete official Train")
            missing = c.execute("""
                SELECT count(*) FROM (SELECT session_id FROM customer_splits WHERE assigned_split='train'
                  EXCEPT SELECT session_id FROM read_parquet(?))
            """, [str(labels)]).fetchone()[0]
            require(missing == 0, "Label pool differs from official Train membership")
            report = rank_population(c, labels, per_scene=args.per_scene, seed=args.seed)
            for filename, condition in (("candidates.parquet", ""), ("selected.parquet", "WHERE selected")):
                c.execute(f"COPY (SELECT * FROM scene_selection {condition} ORDER BY stratum,scene_rank,session_id) "
                          f"TO {literal(stage / filename)} (FORMAT PARQUET, COMPRESSION ZSTD)")
        report.update(schema="customer-train-scene-selection-v1", split="train",
            population_scope="complete_official_train", dataset=str(args.dataset.resolve()),
            dataset_manifest_sha256=sha256(args.dataset / "manifest.json"),
            catalog_manifest_sha256=release["catalog_manifest_sha256"],
            labels_sha256=sha256(labels), selected_sha256=sha256(stage / "selected.parquet"),
            candidates_sha256=sha256(stage / "candidates.parquet"), seed=args.seed,
            per_scene=args.per_scene, source_sha256=sha256(Path(__file__)),
            ranking="Complete exposure titles first, diverse users, title coverage, any strictly prior observed action, seeded tie break",
            scope="Input-quality scene candidates, not verified rationales or unbiased population samples",
            history="Strictly earlier Train requests only; cold/short history retained in separate strata",
            label_use="Observed behavior only stratifies Train; current targets never select visible history",
            remaining_checks="Actual prompt token budget and generated action/evidence checks remain required")
        (stage / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        stage.rename(args.output)
    print(json.dumps({k: report[k] for k in ("population", "selected", "exclusions")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
