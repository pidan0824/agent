"""Request-level metrics; reconstructed shortlists never masquerade as full exposure."""
from collections import defaultdict


def request_metrics(records):
    groups = defaultdict(list)
    for row in records:
        if row["stage"] != "response":
            continue
        t = row["trajectory"]
        key = (row.get("variant", row.get("model")), row.get("mode", "fixed"),
               row.get("policy", "sampled"), row.get("response_style", row.get("style")),
               t.get("source_request_id", row.get("source_request_id")), row.get("tid", t.get("sample_index", 0)))
        groups[key].append(row)
    reports = []
    for key, rows in groups.items():
        target = rows[0]["trajectory"].get("source_response_target", rows[0]["trajectory"].get("target", rows[0].get("target")))
        if target is None:
            continue
        prediction = {f: set() for f in ("clicked_item_ids", "purchased_item_ids")}
        seen, duplicate = set(), False
        for row in rows:
            t = row["trajectory"]
            scene = t.get("scene_id", t.get("record_id"))
            duplicate |= scene in seen
            seen.add(scene)
            if t["status"] == "ok":
                for f in prediction:
                    prediction[f].update(t["final_action"][f])
        # Each positive must have its own child in an outcome-conditioned list.
        expected = max(1, len(set(target["clicked_item_ids"] + target["purchased_item_ids"]))) if key[2] == "sampled" else 1
        complete = len(rows) == expected and not duplicate and all(r["trajectory"]["status"] == "ok" for r in rows)
        metrics = {}
        for f, pred in prediction.items():
            actual = set(target[f]); hit = len(pred & actual)
            metrics[f] = {"precision": hit / len(pred) if pred else (1. if not actual else 0.),
                          "recall": hit / len(actual) if actual else None,
                          "f1": 2*hit/(len(pred)+len(actual)) if pred or actual else 1.,
                          "exact_match": complete and pred == actual}
        reports.append({"variant": key[0], "mode": key[1], "policy": key[2], "style": key[3],
            "source_request_id": key[4], "sample_index": key[5], "complete": complete,
            "protocol": "split_reconstructed" if key[2] == "sampled" else "full_exposure" if key[2] == "full" else "retrieved",
            "metrics": metrics, "prediction": {f: sorted(v) for f,v in prediction.items()},
            "target": target, "expected_scenes": expected, "observed_scenes": len(rows)})
    return reports


def response_metrics(rows):
    """All-attempt scene metrics. Invalid output never earns an empty-match reward.

    Rows contain target, prediction and status. Micro counts pool item IDs within
    each head across scenes; macro scores give each scene equal weight.
    """
    rows = list(rows)
    n = len(rows)
    valid = [r["status"] == "ok" for r in rows]
    report = {"scenes": n, "valid": sum(valid), "invalid": n - sum(valid), "by_head": {}}
    for field in ("clicked_item_ids", "purchased_item_ids"):
        tp = fp = fn = positive = predicted = false_action = 0
        macro, positive_macro, empty_correct = [], [], 0
        for row, ok in zip(rows, valid):
            actual = set(row["target"][field])
            pred = set(row["prediction"][field]) if ok else set()
            hit = len(actual & pred)
            tp += hit
            fp += len(pred - actual)
            fn += len(actual - pred)
            positive += bool(actual)
            predicted += bool(pred)
            false_action += not actual and bool(pred)
            empty_correct += ok and not actual and not pred
            score = (2 * hit / (len(actual) + len(pred)) if actual or pred else 1.) if ok else 0.
            macro.append(score)
            if actual:
                positive_macro.append(score)
        report["by_head"][field] = {
            "tp": tp, "fp": fp, "fn": fn, "positive_scenes": positive,
            "empty_scenes": n - positive, "predicted_action_scenes": predicted,
            "false_action_scenes": false_action,
            "micro_precision": tp / (tp + fp) if tp + fp else None,
            "micro_recall": tp / (tp + fn) if tp + fn else None,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "scene_macro_f1_empty_match_one": sum(macro) / n if n else None,
            "positive_scene_macro_f1": sum(positive_macro) / positive if positive else None,
            "false_action_rate_on_empty": false_action / (n - positive) if n > positive else None,
            "predicted_action_rate": predicted / n if n else None,
            "empty_accuracy_all_attempts": empty_correct / (n - positive) if n > positive else None,
        }
    tp, fp, fn = (sum(h[k] for h in report["by_head"].values()) for k in ("tp", "fp", "fn"))
    report["pooled_action_micro_f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    report["predicted_any_action_rate"] = sum(
        ok and any(row["prediction"][f] for f in report["by_head"])
        for row, ok in zip(rows, valid)) / n if n else None
    report["protocol"] = (
        "All attempts; item-level micro pools TP/FP/FN, scene macro weights children equally. "
        "Empty-empty earns 1 only in explicitly named macro/empty accuracy. "
        "Undefined ratios are null. Invalid outputs contribute FN for positives and zero macro, "
        "but are not counted as predicted or false actions; inspect invalid count.")
    return report


def response_with_baseline(rows):
    """Use identical targets for the actual arm and always-no-action reference."""
    rows = list(rows)
    empty = {"clicked_item_ids": [], "purchased_item_ids": []}
    return {"model": response_metrics(rows),
            "always_no_action": response_metrics(
                {"target": r["target"], "prediction": empty, "status": "ok"} for r in rows)}
