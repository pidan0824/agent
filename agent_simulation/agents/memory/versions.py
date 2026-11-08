"""Version identity, visibility and retirement of long-term memory entries."""

from agent_simulation.data.common import require


def memory_id(entry):
    """Identify the one version an actor can save per reflection batch in this market."""
    return f"{entry['owner']}:{entry['event_count']}"


def _generated_at(entry):
    return entry["event_count"], entry["generated_step"], entry.get("generated_logical_time")


def _reached(boundary, cutoff):
    event_count, event_step, event_time = boundary
    count, step, logical_time = cutoff
    return (
        event_count <= count
        and event_step <= step
        and (logical_time is None or event_time is None or event_time <= logical_time)
    )


def active(entries, count, step, logical_time=None):
    """Use all available clocks, falling back to count and step for legacy records."""
    cutoff = (count, step, logical_time)
    return [
        e
        for e in entries
        if _reached(_generated_at(e), cutoff)
        and not (
            "retired_at_event_count" in e
            and _reached(
                (e["retired_at_event_count"], e.get("retired_step", float("inf")),
                 e.get("retired_logical_time")),
                cutoff,
            )
        )
    ]


def retire(entries, ids, *, count, step, replacement, source_ids, logical_time=None,
           context_source_ids=()):
    """Stage retirement of earlier visible versions in the proposed memory list."""
    updated = list(entries)
    positions = {memory_id(entry): index for index, entry in enumerate(updated)}
    targets = set(ids)
    require(len(positions) == len(updated), "Duplicate long-term memory ID")
    require(targets <= positions.keys(), "Retirement target does not exist")
    new_entry = None
    if replacement is not None:
        require(replacement in positions, "Replacement memory does not exist")
        new_entry = updated[positions[replacement]]
        require(_generated_at(new_entry) == (count, step, logical_time)
                and "retired_at_event_count" not in new_entry,
                "Replacement memory must be new in this batch")

    for target in targets:
        index = positions[target]
        entry = updated[index]
        require("retired_at_event_count" not in entry, "Retirement target is already retired")
        require(_reached(_generated_at(entry), (count - 1, step, logical_time)),
                "Retirement target is not an earlier visible memory")
        require(new_entry is None or new_entry["owner"] == entry["owner"],
                "Replacement and retirement target must have the same owner")
        updated[index] = {
            **entry,
            "retired_at_event_count": count,
            "retired_step": step,
            "retired_logical_time": logical_time,
            "replaced_by": replacement,
            "retirement_source_ids": list(source_ids),
            "retirement_context_source_ids": list(context_source_ids),
        }
    return updated
