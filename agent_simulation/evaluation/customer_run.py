"""Small run journals and deterministic post-generation sampling; no model management."""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import numpy as np

from agent_simulation.data.common import require
from agent_simulation.evaluation.trajectories import digest


@contextmanager
def run_lock(directory):
    with (Path(directory) / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another writer is using this run") from None
        yield


class RowJournal:
    """Persist completed units; discard only an interrupted final JSONL line."""
    def __init__(self, path):
        self.path = Path(path)
        self.rows = {}
        if self.path.exists():
            with self.path.open("r+b") as stream:
                while True:
                    start = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        stream.truncate(start)
                        break
                    row = json.loads(line)
                    key = row["task_id"]
                    require(key not in self.rows, "Duplicate journal task")
                    self.rows[key] = row
        self.stream = self.path.open("ab")

    def get(self, key):
        return deepcopy(self.rows.get(key))

    def append(self, row):
        key = row["task_id"]
        require(key not in self.rows, "Task already persisted")
        self.stream.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode())
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.rows[key] = deepcopy(row)

    def close(self):
        self.stream.close()


def task_key(row):
    return digest([row["cid"], row["variant"], row["tid"]])


def sample_probabilities(row, probabilities, seed, scorer):
    """A scene/variant/sample owns its RNG regardless of completion order."""
    from agent_simulation.evaluation.customer_filtering import probability_record
    seed_key = int(digest(["response-probability-v1", seed, task_key(row)])[:16], 16)
    t = row["trajectory"]
    values = probability_record(probabilities, t["candidate_item_ids"], t["target"],
                                rng=np.random.default_rng(seed_key))
    row.update(values, probability_sampling_seed=seed_key)
    for name in ("argmax", "sampled"):
        row[name + "_score"] = scorer.score("response", row[name + "_action"], t["target"])["score"]


class FrozenPredictionBinding:
    """Replay exactly the label-free context saved at generation time."""
    def __init__(self, row):
        from agent_simulation.evaluation.customer_evidence import encode_context
        context = row["probability_context"]
        require(digest(context) == row["probability_context_sha256"], "Predictor context changed")
        require(any(m.get("role") == "user" and m.get("content") == encode_context(context)
                    for m in row["trajectory"]["prompt"]), "Predictor context differs from original prompt")
        self.value = context
        self.displayed = [{"item_id": p["item_id"]} for p in context["candidate_context"]]
        require([p["item_id"] for p in self.displayed] == row["trajectory"]["candidate_item_ids"],
                "Predictor candidates differ from original trajectory")
        self.context = {"response_action_mode": row["trajectory"]["action_mode"]}

    def initial_context(self):
        return deepcopy(self.value)


def predictor_trajectory(row):
    # Final actions and observed targets must never enter the predictor input.
    t = row["trajectory"]
    return {"rationale": deepcopy(t.get("rationale", {})),
            "trace": {"tools": deepcopy(t["trace"].get("tools", []))}}
