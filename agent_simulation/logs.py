"""Record model calls, Agent runs and JSON reports."""

from collections import Counter
import json
from pathlib import Path
from uuid import uuid4


def write_report(path, value):
    """Create parent directories and atomically replace a JSON report."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class ModelLog:
    """Append unedited model outputs before the model API parses them."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)

    def append(self, record):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encode(record) + "\n")


def model_totals(path):
    """Summarize persisted model calls without changing or repairing any record."""
    totals = {"model_calls": 0, "generated_tokens": 0, "model_seconds": 0.0}
    with path.open() as lines:
        for line in lines:
            call = json.loads(line)
            totals["model_calls"] += 1
            totals["generated_tokens"] += call["output_tokens"]
            totals["model_seconds"] += call["seconds"]
    return totals


class AgentRunLog:
    """Store Agent traces in the environment's existing SQLite connection."""

    def __init__(self, connection):
        self.db = connection
        with self.db:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS agent_runs ("
                "id TEXT PRIMARY KEY, role TEXT, actor INTEGER, status TEXT, trace TEXT, error TEXT)"
            )

    def start(self, role, actor):
        run = {
            "id": uuid4().hex,
            "role": role,
            "actor": actor,
            "model_calls": [],
            "tools": [],
            "memory_updates": [],
        }
        self.save(run, "running")
        return run

    def save(self, run, status, error=None):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO agent_runs VALUES (?,?,?,?,?,?)",
                (run["id"], run["role"], run["actor"], status, encode(run), error),
            )

    def tool_counts(self):
        counts = Counter()
        for (raw,) in self.db.execute("SELECT trace FROM agent_runs"):
            counts.update(entry["name"] for entry in json.loads(raw)["tools"])
        return dict(counts)
