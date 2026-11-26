"""Action contracts for conditional shortlists; full exposures retain multi-action."""
from agent_simulation.data.common import require


def single_item_actions(ids):
    actions = [{"clicked_item_ids": [], "purchased_item_ids": []}]
    for item in ids:
        for click, purchase in ((True, False), (False, True), (True, True)):
            actions.append({"clicked_item_ids": [item] if click else [],
                            "purchased_item_ids": [item] if purchase else []})
    return actions


def single_item_schema(ids):
    return {"anyOf": [{"const": action} for action in single_item_actions(ids)]}


def validate_request_splits(records):
    """Fail before export if sibling scenes/Query rows cross a source-request split."""
    seen = {}
    for row in records:
        t = row.get("trajectory", row)
        split = t["source_split"]
        require(row.get("source_split", split) == split, "Row and trajectory split mismatch")
        request_id = t.get("source_request_id", t.get("record_id"))
        require(request_id is not None, "Missing source request identity")
        # Namespace by dataset release, never by domain, scene, model, or policy.
        key = (t.get("dataset_manifest_sha256", "legacy"), str(request_id))
        require(key not in seen or seen[key] == split, "Source request crosses dataset splits")
        seen[key] = split
