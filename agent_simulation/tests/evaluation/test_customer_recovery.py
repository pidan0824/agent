"""Recovery and staged scoring through the real entry points, without model services."""
import asyncio
from contextlib import contextmanager, ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import httpx

from agent_simulation.evaluation import customer_curation as curation, customer_scoring as scoring
from agent_simulation.evaluation.customer_contract import single_item_actions
from agent_simulation.evaluation.customer_run import RowJournal, run_lock, sample_probabilities
from agent_simulation.evaluation.trajectories import OutcomeScorer
from agent_simulation.tests.evaluation.test_customer_generation import job
from agent_simulation.tests.ekb_fixture import rationale


@contextmanager
def fixture(root):
    prepared = {}
    for value in (job(11), job(12, empty=True)):
        identity, _, _, _, source, candidate, _, _ = value
        prepared[identity] = {"request": source, "query_enabled": True,
                             "response_scenes": [{"request": source, "candidate": candidate}]}
    for name in ("dataset", "scenes", "tokenizer"):
        (root / name).mkdir()
        (root / name / "manifest.json").write_text("{}")
    (root / "tokenizer/tokenizer_config.json").write_text("{}")
    state = SimpleNamespace(calls=0, checks=[], fail_at=None, score_calls=0, score_fail_at=None)
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
    data = Mock()
    data.tokenizer = tokenizer
    data.catalog.connection.execute.return_value.fetchall.return_value = [(i,) for i in prepared]
    class Client:
        def __init__(self, url, model, tokenizer, **kwargs):
            self.model, self.seed = model, kwargs.get("seed")
        def close(self): pass
        def check_models(self): state.checks.append(self.model)
        async def acomplete(self, value, *, call_id):
            state.calls += 1
            if state.calls == state.fail_at:
                raise httpx.ReadTimeout("fixture interruption")
            if "query" in value["final_schema"].get("properties", {}):
                output = {"query": "SECRET TARGET"}
            else:
                output = {"rationale": rationale(),
                          "action": {"clicked_item_ids": [], "purchased_item_ids": []}}
            await asyncio.sleep(0)
            return {"role": "assistant", "content": json.dumps(output), "tool_calls": []}
    def predict(binding, trajectory):
        state.score_calls += 1
        if state.score_calls == state.score_fail_at:
            raise httpx.ReadTimeout("predictor interrupted")
        assert "target" not in trajectory and "source_response_target" not in trajectory
        actions = single_item_actions([p["item_id"] for p in binding.displayed])
        return {"actions": actions, "probabilities": [1 / len(actions)] * len(actions)}
    with ExitStack() as stack:
        stack.enter_context(patch.object(curation, "SFTDataset", return_value=data))
        stack.enter_context(patch("agent_simulation.datasets.customer_scenes.read_scenes",
                                 return_value=(deepcopy(prepared), {})))
        for module in (curation, scoring):
            stack.enter_context(patch.object(module, "load_tokenizer", return_value=tokenizer))
            stack.enter_context(patch.object(module, "AgenticClient", Client))
            predictor = stack.enter_context(patch.object(module, "OptionPredictor"))
            predictor.return_value.score.side_effect = predict
        stack.enter_context(redirect_stdout(io.StringIO()))
        yield state


def argv(root, output, *extra):
    return ["--dataset", str(root / "dataset"), "--scenes", str(root / "scenes"),
            "--output", str(root / output), "--samples", "2", "--scene-concurrency", "1",
            "--sample-concurrency", "1", *extra]


def score_args(root, runs, output="scored", *extra):
    args = ["--output", str(root / output), "--predictor-url", "http://fixture",
            "--predictor-model", "fixed-predictor", "--predictor-tokenizer", str(root / "tokenizer"),
            "--concurrency", "1", *extra]
    for run in runs: args += ["--run", str(root / run)]
    return args


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class RecoveryTests(unittest.TestCase):
    def test_resume_reuses_samples_and_supervision_before_transport_failure(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with fixture(root) as state:
                state.fail_at = 3
                args = argv(root, "run", "--sft-only")
                with self.assertRaises(httpx.ReadTimeout): curation.main(args)
                self.assertEqual(len(rows(root / "run/generated.jsonl")), 2)
                self.assertEqual(len(rows(root / "run/observed-actions.jsonl")), 4)
                self.assertFalse((root / "run/completed.json").exists())
                state.fail_at = None
                curation.main(args + ["--resume", "--scene-concurrency", "2"])
                # SFT-only keeps observed Query rows without generating them: four successful calls plus one failure.
                self.assertEqual(state.calls, 5)
                completed = (root / "run/records.jsonl").read_bytes()
                state.checks.clear()
                curation.main(args + ["--resume"])
                self.assertEqual(state.calls, 5)
                self.assertEqual(state.checks, [])
                self.assertEqual(completed, (root / "run/records.jsonl").read_bytes())
                with self.assertRaisesRegex(ValueError, "Resume identity"):
                    curation.main(args + ["--resume", "--seed", "99"])

    def test_inline_predictor_failure_does_not_repeat_generation(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with fixture(root) as state:
                state.score_fail_at = 1
                args = argv(root, "run")
                with self.assertRaises(httpx.ReadTimeout): curation.main(args)
                saved = len(rows(root / "run/generated.jsonl"))
                self.assertGreater(saved, 0)
                state.score_fail_at = None
                curation.main(args + ["--resume"])
                self.assertEqual(state.calls, 16)
                self.assertEqual(len(rows(root / "run/generated.jsonl")), 16)

    def test_staged_scoring_matches_inline_and_only_connects_selected_generator(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with fixture(root) as state:
                curation.main(argv(root, "online"))
                state.checks.clear()
                for variant in ("strong", "weak"):
                    curation.main(argv(root, variant, "--defer-probabilities", "--generation-variants", variant))
                    self.assertFalse((root / variant / "preferences.jsonl").exists())
                self.assertEqual(state.checks, ["Qwen3-32B", "Qwen3.5-9B"])
                generation_calls = state.calls
                scoring.main(score_args(root, ["strong", "weak"]))
                self.assertEqual(state.calls, generation_calls)
                a = {r["task_id"]: r for r in rows(root / "online/records.jsonl")}
                b = {r["task_id"]: r for r in rows(root / "scored/records.jsonl")}
                self.assertEqual(set(a), set(b))
                for key in a:
                    self.assertEqual(a[key]["trajectory"], b[key]["trajectory"])
                    for field in ("prob", "sampled_action", "entropy", "logloss", "preference"):
                        if field in a[key]: self.assertEqual(a[key][field], b[key][field])
                before = (root / "scored/records.jsonl").read_bytes()
                old_scores = state.score_calls
                state.checks.clear()
                scoring.main(score_args(root, ["weak", "strong"], "scored", "--resume"))
                self.assertEqual(old_scores, state.score_calls)
                self.assertEqual(state.checks, [])
                self.assertEqual(before, (root / "scored/records.jsonl").read_bytes())
                with self.assertRaisesRegex(ValueError, "overlapping"):
                    scoring.load_runs([root / "strong", root / "strong"])
                with self.assertRaisesRegex(ValueError, "different frozen"):
                    # Different sampling settings cannot be merged.
                    curation.main(argv(root, "different", "--defer-probabilities",
                                       "--generation-variants", "weak", "--seed", "99"))
                    scoring.load_runs([root / "strong", root / "different"])

    def test_scoring_resume_keeps_completed_scores_and_original_transcripts(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with fixture(root) as state:
                curation.main(argv(root, "run", "--defer-probabilities"))
                original = (root / "run/records.jsonl").read_bytes()
                state.score_fail_at = 3
                with self.assertRaises(httpx.ReadTimeout):
                    scoring.main(score_args(root, ["run"]))
                self.assertEqual(len(rows(root / "scored/probabilities.jsonl")), 2)
                state.score_fail_at = None
                scoring.main(score_args(root, ["run"], "scored", "--resume"))
                self.assertEqual(state.score_calls, 9)  # Eight Response scores plus one failed request.
                self.assertEqual(original, (root / "run/records.jsonl").read_bytes())

    def test_staged_scoring_rejects_incomplete_runs_and_changed_snapshots(self):
        from agent_simulation.evaluation.customer_run import FrozenPredictionBinding
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with fixture(root):
                curation.main(argv(root, "run", "--defer-probabilities"))
                row = next(r for r in rows(root / "run/records.jsonl") if r["stage"] == "response")
                FrozenPredictionBinding(row)
                row["probability_context"]["query"] = "changed"
                with self.assertRaisesRegex(ValueError, "context changed"):
                    FrozenPredictionBinding(row)
                (root / "run/records.jsonl").write_text("")
                with self.assertRaisesRegex(ValueError, "incomplete or changed"):
                    scoring.load_runs([root / "run"])

    def test_journal_repairs_only_partial_tail_and_refuses_corrupt_complete_lines(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "journal.jsonl"
            path.write_bytes(b'{"task_id":"first"}\n{"task_')
            journal = RowJournal(path)
            self.assertEqual(set(journal.rows), {"first"})
            journal.append({"task_id": "second"})
            journal.close()
            self.assertEqual(len(rows(path)), 2)
            path.write_text('not json\n')
            with self.assertRaises(json.JSONDecodeError): RowJournal(path)
            with run_lock(Path(temp)):
                with self.assertRaisesRegex(ValueError, "Another writer"):
                    with run_lock(Path(temp)): pass

    def test_sampling_is_independent_of_row_order(self):
        source = {"cid": "scene", "variant": "strong", "tid": 0,
                  "trajectory": {"candidate_item_ids": [1, 2],
                                 "target": {"clicked_item_ids": [1], "purchased_item_ids": []}}}
        inputs = [dict(deepcopy(source), cid=str(i)) for i in range(20)]
        probabilities = {"actions": single_item_actions([1, 2]), "probabilities": [1/7] * 7}
        def run(values):
            result = {}
            for original in values:
                row = deepcopy(original)
                sample_probabilities(row, probabilities, 42, OutcomeScorer())
                result[row["cid"]] = row["sampled_action"]
            return result
        self.assertEqual(run(inputs), run(inputs[::-1]))
