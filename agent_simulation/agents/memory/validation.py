"""Shared grounded reflection output contract."""

from agent_simulation.data.common import require
from agent_simulation.agents.schemas import strict_json
from agent_simulation.agents.memory.versions import memory_id

MAX_MEMORY_SOURCES = 3


class ReflectionDraftError(ValueError):
    """A rejected model draft preserved with its validation error."""

    def __init__(self, output, reason, *, request=None):
        super().__init__(reason)
        self.output = output
        self.request = request
        self.audit_id = None


def quality_issues(result, new_ids, previous, *, action):
    """Check new evidence and a few explicit issues in an authorized draft."""
    if action in ("no_update", "retract"):
        return []
    content = result["content"].strip()
    issues = []
    if set(new_ids).isdisjoint(result["source_ids"]):
        issues.append("The update cites no new evidence.")
    if action == "add" and any(content == entry["content"].strip() for entry in previous):
        issues.append("Content exactly repeats an active memory.")
    return issues


def source_ids(events, previous):
    return {e["source_id"] for e in events} | {s for e in previous for s in e["source_ids"]}


def read_draft(raw, selected, previous, *, request=None):
    """Decode and validate the response once against authorized evidence and active versions."""
    sources = source_ids(selected, previous)
    memories = {memory_id(e) for e in previous}
    try:
        result = strict_json(raw) if isinstance(raw, str) else raw
        action = validate_draft(result, sources, memories)
    except ValueError as error:
        raise ReflectionDraftError(raw, str(error), request=request) from error
    return {key: value for key, value in result.items() if key != "action"}, action


def validate_draft(result, sources, memories):
    require(
        type(result) is dict
        and {"action", "content", "source_ids"} <= set(result)
        and set(result) <= {"action", "content", "source_ids", "supersedes"},
        "Invalid memory output schema",
    )
    require(type(result["content"]) is str, "Memory content must be a string")
    for name, allowed, message in (
        ("source_ids", sources, "Memory cites unavailable evidence"),
        ("supersedes", memories, "Can only revise supplied active memories"),
    ):
        values = result.get(name, [])
        require(
            type(values) is list
            and all(type(v) is str for v in values)
            and len(values) == len(set(values)),
            f"{name} must contain unique string IDs",
        )
        require(set(values) <= allowed, message)
    require(len(result["source_ids"]) <= MAX_MEMORY_SOURCES,
            "Use at most three supporting sources for one focused memory update.")
    return operation(result)


def operation(draft):
    """Check the declared operation against its content, references and replacements."""
    content = draft["content"]
    replacing = "supersedes" in draft
    require(not replacing or bool(draft["supersedes"]), "Omit empty supersedes.")
    require(not content or content.strip(), "Whitespace alone is not a summary.")
    action = draft["action"]
    require(action in ("no_update", "add", "revise", "retract"), "Unknown memory action.")
    require(replacing == (action in ("revise", "retract")),
            "Memory action does not match supersedes.")
    require(bool(content) == (action in ("add", "revise")),
            "Memory action does not match content.")
    if action in ("add", "revise"):
        require(draft["source_ids"], "A memory change requires cited evidence.")
    else:
        require(not draft["source_ids"], "This memory action does not take source_ids.")
    return action
