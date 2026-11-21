"""Compare historical and supervised Seller predictors on one frozen evidence contract."""
import argparse
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from agent_simulation.data.catalog import Catalog, DataRepository
from agent_simulation.data.common import require, rows, sha256
from agent_simulation.data.market.quotes import RECENT_MARKET_SPAN, CategorySnapshots
from agent_simulation.datasets.labels import select_record_ids
from agent_simulation.evaluation.micro.behavior_prediction import summarize
from agent_simulation.logs import write_report


FEATURES = (
    "log_previous_price", "log_history_points", "last_log_change", "log_price_cv",
    "log_quote_gap", "recent_ctr", "recent_purchase_rate", "log_exposures",
    "log_market_products", "log_market_median_ratio", "market_percentile",
    "log_market_median_change", "log_history_median_ratio",
)
RULES = ("persistence", "rolling_median", "ewma", "competitor_relative", "competitor_quantile")
LEARNED = ("two_stage_linear", "gbdt")


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def evidence_features(history, current_market, previous_market, cutoff, *, window=10, alpha=0.3):
    """No current target is accepted by this function. Quotes are in original raw units."""
    require(window > 0 and 0 < alpha <= 1, "Invalid history window or EWMA alpha")
    require(all(row["time_index"] < cutoff for row in history), "Future history in baseline")
    history = sorted(history, key=lambda row: row["time_index"])
    valid = [r for r in history if r.get("price_status") == "observed" and positive(r["price_raw"])]
    require(bool(valid), "Seller baseline requires an earlier valid quote")
    reference = valid[-1]
    previous = reference["price_raw"]
    recent = history[-window:]
    prices = np.asarray([r["price_raw"] for r in valid[-window:]], dtype=float)
    current = np.asarray([p for p in current_market if positive(p)], dtype=float)
    earlier = np.asarray([p for p in previous_market if positive(p)], dtype=float)
    median = float(np.median(current)) if len(current) else None
    old_median = float(np.median(earlier)) if len(earlier) else None
    exposures = sum(r.get("exposure_records", 0) for r in recent)
    clicks = sum(r.get("click_records", 0) for r in recent)
    purchases = sum(r.get("purchase_records", 0) for r in recent)
    smooth = float(prices[0])
    for price in prices[1:]:
        smooth = alpha * float(price) + (1 - alpha) * smooth
    rank = lambda a: float((np.sum(a < previous) + .5 * np.sum(a == previous)) / len(a))
    comparable = bool(len(current) and len(earlier))
    rules = {
        "persistence": previous,
        "rolling_median": float(np.median(prices)),
        "ewma": smooth,
        "competitor_relative": previous * median / old_median if comparable else previous,
        "competitor_quantile": float(np.quantile(current, rank(earlier))) if comparable else previous,
    }
    values = [
        math.log(previous), math.log1p(len(valid)),
        math.log(prices[-1] / prices[-2]) if len(prices) > 1 else None,
        float(np.std(np.log(prices))), math.log1p(cutoff - reference["time_index"]),
        clicks / exposures if exposures else None, purchases / exposures if exposures else None,
        math.log1p(exposures), math.log1p(len(current)),
        math.log(median / previous) if median else None, rank(current) if len(current) else None,
        math.log(median / old_median) if comparable else None,
        math.log(float(np.median(prices)) / previous),
    ]
    return {"features": dict(zip(FEATURES, values)), "rules": rules,
            "previous_price_raw": previous, "previous_time_index": reference["time_index"],
            "history_points": len(valid), "recent_history_points": len(prices),
            "market_products": len(current), "previous_market_products": len(earlier),
            "competitor_fallback": not comparable}


def collect(dataset, split, *, count, seed, window, alpha, sample_ids=None):
    manifest = json.loads((dataset / "manifest.json").read_text())
    require(sha256(Path(manifest["catalog"]) / "manifest.json") == manifest["catalog_manifest_sha256"],
            "Baseline catalog differs from the frozen dataset")
    with closing(Catalog(manifest["catalog"])) as catalog:
        ids = sample_ids or select_record_ids(catalog.connection, dataset, "seller", split, count, seed)
        require(len(ids) == len(set(ids)), "Duplicate Seller selection")
        labels = rows(catalog.connection,
                      "SELECT * FROM read_parquet(?) WHERE sample_id IN (SELECT unnest(?))",
                      [str(dataset / "seller" / split / "labels.parquet"), ids])
        require({r["sample_id"] for r in labels} == set(ids), "Selected Seller IDs are outside this split")
        snapshots = CategorySnapshots()
        records = []
        for label in sorted(labels, key=lambda row: (row["history_cutoff_exclusive"], row["sample_id"])):
            seller, item, cutoff = (label[k] for k in ("seller_id", "item_id", "history_cutoff_exclusive"))
            repo = DataRepository(catalog.connection, catalog.config, cutoff, category_snapshots=snapshots)
            history = repo.records(item)
            require(all(row["seller_id"] == seller for row in history), "Foreign Seller history")
            valid = sorted((r for r in history if r["price_status"] == "observed" and positive(r["price_raw"])),
                           key=lambda row: row["time_index"])
            require(bool(valid), "Missing previous Seller quote")
            old_cutoff = valid[-1]["time_index"] + 1

            def market(before):
                pool = repo.competitor_pool(seller, item, before)
                return [r["reference_price_raw"] for r in pool
                        if max(0, before - RECENT_MARKET_SPAN) <= r["reference_price_time_index"] < before
                        and r["seller_id"] != seller and r["price_status"] == "observed"]

            facts = evidence_features(history, market(cutoff), market(old_cutoff), cutoff,
                                      window=window, alpha=alpha)
            identity = {"sample_id": label["sample_id"], "cutoff": cutoff, "facts": facts}
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
            records.append({"sample_id": label["sample_id"], "split": split, "seller_id": seller,
                            "item_id": item, "cutoff": cutoff, "evidence_sha256": fingerprint,
                            **facts, "target_price_raw": label["target_price_raw"]})
    return records


def matrix(records):
    return np.asarray([[r["features"][name] for name in FEATURES] for r in records], dtype=float)


class FittedBaselines:
    """Fit only train; the change classifier retains the observed class distribution."""
    def fit(self, records, *, seed=42, threshold=.5):
        require(records and all(r["split"] == "train" for r in records), "Fit requires only train records")
        require(0 < threshold < 1, "Change threshold must lie strictly between 0 and 1")
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.threshold = threshold
        x = matrix(records)
        y = np.log([r["target_price_raw"] / r["previous_price_raw"] for r in records])
        changed = np.asarray([r["target_price_raw"] != r["previous_price_raw"] for r in records])
        self.transform = make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                                     keep_empty_features=True), StandardScaler())
        z = self.transform.fit_transform(x)
        self.change_constant = bool(changed[0]) if np.unique(changed).size == 1 else None
        self.classifier = None if self.change_constant is not None else LogisticRegression(
            max_iter=500, random_state=seed).fit(z, changed)
        self.delta = Ridge(alpha=1.0).fit(z[changed], y[changed]) if changed.any() else None
        self.gbdt = HistGradientBoostingRegressor(random_state=seed, max_iter=100,
                                                  max_leaf_nodes=15).fit(x, y)
        return self

    def predict(self, records):
        require(bool(records), "No evaluation rows")
        x = matrix(records)
        z = self.transform.transform(x)
        probabilities = (self.classifier.predict_proba(z)[:, 1] if self.classifier is not None
                         else np.full(len(records), float(self.change_constant)))
        delta = self.delta.predict(z) if self.delta is not None else np.zeros(len(records))
        delta = np.where(probabilities >= self.threshold, delta, 0.0)
        previous = np.asarray([r["previous_price_raw"] for r in records])
        # The bound protects exp from overflow; it is not a learned pricing rule.
        return {"two_stage_linear": previous * np.exp(np.clip(delta, -20, 20)),
                "gbdt": previous * np.exp(np.clip(self.gbdt.predict(x), -20, 20))}


def prediction_rows(records, name, values):
    return [{"sample_id": r["sample_id"], "stage": "seller", "status": "ok",
             "baseline": name, "evidence_sha256": r["evidence_sha256"],
             "previous_price_raw": r["previous_price_raw"],
             "target": {"prices": [{"item_id": r["item_id"], "price_raw": r["target_price_raw"]}]},
             "prediction": {"prices": [{"item_id": r["item_id"], "price_raw": float(value)}]}}
            for r, value in zip(records, values, strict=True)]


def write_jsonl(path, records):
    with path.open("x") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--sample-id", action="append", help="Explicit test IDs, shared with the LLM comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=.3)
    parser.add_argument("--change-threshold", type=float, default=.5)
    parser.add_argument("--rules-only", action="store_true", help="Skip supervised models explicitly")
    args = parser.parse_args(argv)
    require(args.train_count > 0 and args.count > 0, "Positive bounded sample counts required")
    require(not args.output.exists(), "Use a new baseline output directory")
    args.output.mkdir(parents=True)
    manifest = {"schema": "seller-baselines-v1", "dataset_sha256": sha256(args.dataset / "manifest.json"),
                "seed": args.seed, "window": args.window, "alpha": args.alpha,
                "change_threshold": args.change_threshold, "features": list(FEATURES),
                "information_contract": "Prior owned observations and other shops' same-category quotes in the existing recent market window; strict cutoff; raw units. No future target in features.",
                "protocol": "Fit on official train only; fixed hyperparameters, test used only for scoring. Earlier test observations may enter later test history.",
                "comparison_limit": "Common accessible facts, not a claim that an autonomous agent read every fact. Align explicit sample IDs for model comparisons.",
                "status": "running"}
    write_report(args.output / "manifest.json", manifest)
    try:
        options = dict(seed=args.seed, window=args.window, alpha=args.alpha)
        test = collect(args.dataset, "test", count=args.count, sample_ids=args.sample_id, **options)
        write_jsonl(args.output / "test-features.jsonl", test)
        predictions = {name: [r["rules"][name] for r in test] for name in RULES}
        if not args.rules_only:
            train = collect(args.dataset, "train", count=args.train_count, **options)
            write_jsonl(args.output / "train-features.jsonl", train)
            fitted = FittedBaselines().fit(train, seed=args.seed, threshold=args.change_threshold)
            predictions.update(fitted.predict(test))
            import joblib
            joblib.dump({"features": FEATURES, "transform": fitted.transform,
                         "classifier": fitted.classifier, "change_constant": fitted.change_constant,
                         "delta": fitted.delta, "gbdt": fitted.gbdt, "threshold": fitted.threshold},
                        args.output / "fitted-models.joblib")
            manifest["train_ids"] = [r["sample_id"] for r in train]
        report = {}
        for name, values in predictions.items():
            records = prediction_rows(test, name, values)
            write_jsonl(args.output / (name + ".jsonl"), records)
            report[name] = summarize(records)
        report["coverage"] = {"test_rows": len(test),
                              "competitor_fallback_rows": sum(r["competitor_fallback"] for r in test)}
        write_report(args.output / "report.json", report)
        manifest.update(status="complete", test_ids=[r["sample_id"] for r in test])
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_report(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
