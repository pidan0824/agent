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
