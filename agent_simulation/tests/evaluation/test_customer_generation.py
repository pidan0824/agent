"""Offline regression checks for the formal lightweight curation entry."""
import asyncio
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent_simulation.agents.prompts.customer_ekb import concise_ekb_instructions
from agent_simulation.evaluation.customer_curation import CurationTools, main, parser
from agent_simulation.evaluation.customer_generation import (
    bounded_map, budget_scenes, resolve_generation_options, scene_evidence,
)
from agent_simulation.evaluation.trajectories import trajectory_messages
from agent_simulation.tests.evaluation.test_customer_agentic import request
from agent_simulation.tests.ekb_fixture import rationale


def options(*extra):
    args = parser().parse_args(["--dataset", "unused", "--output", "unused", *extra])
    resolve_generation_options(args)
    return args


def job(identity, *, duplicate=False, empty=False):
    source = request()
    source.update(session_id=identity, assigned_split="train")
    source["response_input"]["response_action_mode"] = "single_item"
    if duplicate:
        source["response_input"]["displayed_items"].append(
            {**source["response_input"]["displayed_items"][0], "item_id": 5})
    if empty:
        source["response_target"] = {"clicked_item_ids": [], "purchased_item_ids": []}
    ids = [p["item_id"] for p in source["response_input"]["displayed_items"]]
    candidate = {"scene_id": str(identity), "domain": "test", "candidate_count": len(ids),
                 "candidate_item_ids": ids, "original_item_ids": ids,
                 "source_request_id": identity, "target_item_id": None if empty else 4,
                 "source_response_target": deepcopy(source["response_target"])}
    return identity, 7, "response", "sampled", source, candidate, "fixed", "ekb"


class GenerationTests(unittest.TestCase):
    def test_preset_keeps_research_defaults_and_explicit_overrides(self):
        research = options()
        self.assertEqual((research.samples, research.temperature, research.max_new_tokens,
                          research.scene_concurrency), (10, .7, 2048, 1))
        light = options("--cot-preset", "lightweight", "--sft-only")
        self.assertEqual(light.history_token_budget, 2048)
        self.assertIsNone(research.history_token_budget)
        self.assertEqual((light.samples, light.temperature, light.max_new_tokens,
                          light.scene_concurrency), (1, 0., 768, 4))
        override = options("--cot-preset", "lightweight", "--sft-only",
                           "--samples", "2", "--temperature", ".3", "--max-new-tokens", "512",
                           "--scene-concurrency", "2")
        self.assertEqual((override.samples, override.temperature, override.max_new_tokens,
                          override.scene_concurrency), (2, .3, 512, 2))
        for flags in (["--cot-preset", "lightweight"],
                      ["--cot-preset", "lightweight", "--sft-only", "--modes", "autonomous"],
                      ["--teacher-budget", "1"], ["--scene-concurrency", "0"],
                      ["--cot-preset", "lightweight", "--sft-only", "--teacher-budget", "1", "--split", "test"]):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                options(*flags)

    def test_concise_prompt_changes_system_only_and_keeps_schema(self):
        source = request()
        before = deepcopy(source)
        old = CurationTools(source, "response", mode="fixed")
        new = CurationTools(source, "response", mode="fixed", concise=True)
        schema, prompt = trajectory_messages(old, role="customer", stage="response", theory=True)
        new_schema, new_prompt = trajectory_messages(new, role="customer", stage="response", theory=True)
        self.assertEqual(schema, new_schema)
        self.assertEqual(prompt[1:], new_prompt[1:])
        self.assertIn(concise_ekb_instructions(), new_prompt[0]["content"])
        self.assertLess(len(new_prompt[0]["content"]), len(prompt[0]["content"]))
        self.assertEqual(source, before)

    def test_input_priority_detects_full_attribute_collision_not_title_alone(self):
        value = job(1, duplicate=True)
        binding = CurationTools(value[4], "response", mode="fixed")
        info = scene_evidence(binding)
        self.assertEqual(info["unsupported_identical_groups"], [[4, 5]])
        changed = deepcopy(value[4])
        changed["response_target"] = {"clicked_item_ids": [], "purchased_item_ids": []}
        self.assertEqual(info, scene_evidence(CurationTools(changed, "response", mode="fixed")))
        changed["response_input"]["displayed_items"][1]["price_raw"] = 9
        self.assertEqual(scene_evidence(CurationTools(changed, "response", mode="fixed"))["identical_visible_groups"], [])
        changed = deepcopy(value[4])
        changed["response_input"]["short_term_memory"]["search_history"][0]["clicked_items"] = [{"item_id": 4}]
        info = scene_evidence(CurationTools(changed, "response", mode="fixed"))
        self.assertTrue(info["same_item_history"])
        self.assertEqual(info["unsupported_identical_groups"], [])
        self.assertEqual(info["identical_visible_groups"], [[4, 5]])

    def test_budget_prefers_evidence_preserves_strata_and_zero_consumes_all(self):
        values = [job(1, duplicate=True), job(2), job(3, empty=True)]
        frozen = deepcopy(values)
        selected, report = budget_scenes(iter(values), 2, 42, io.StringIO())
        self.assertEqual([v[0] for v in selected], [2, 3])
        self.assertEqual(report["considered"], 3)
        self.assertEqual(values, frozen)
        again, _ = budget_scenes(iter(values), 2, 42, io.StringIO())
        self.assertEqual(selected, again)
        consumed = []
        def items():
            for value in values:
                consumed.append(value[0])
                yield value
        selected, report = budget_scenes(items(), 0, 42, io.StringIO())
        self.assertEqual(consumed, [1, 2, 3])
        self.assertEqual(selected, [])
        self.assertEqual(report["selected"], 0)

    def test_scheduler_is_concurrent_bounded_lazy_and_ordered(self):
        async def run():
            active = peak = consumed = 0
            ready = asyncio.Event()
            def items():
                nonlocal consumed
                for i in range(9):
                    consumed += 1
                    yield i
            async def task(i):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                if active == 3:
                    ready.set()
                if i < 3:
                    await asyncio.wait_for(ready.wait(), 1)
                await asyncio.sleep(0)
                active -= 1
                return i
            results = []
            async for result in bounded_map(task, items(), 3):
                self.assertLessEqual(consumed - len(results), 3)
                results.append(result)
            self.assertEqual(results, list(range(9)))
            self.assertEqual(peak, 3)
        asyncio.run(run())

    def test_scheduler_cancels_other_tasks_on_failure(self):
        async def run():
            started = asyncio.Event()
            cancelled = []
            async def task(i):
                if i == 0:
                    await started.wait()
                    raise RuntimeError("broken")
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.append(i)
            with self.assertRaisesRegex(RuntimeError, "broken"):
                async for _ in bounded_map(task, range(20), 2):
                    pass
            self.assertEqual(cancelled, [1])
        asyncio.run(run())

    def test_formal_entry_offline_parallel_and_budgeted_export(self):
        from agent_simulation.evaluation import customer_curation as module
        prepared = {}
        for value in [job(11, duplicate=True), job(12), job(13, empty=True)]:
            identity, _, _, _, source, candidate, _, _ = value
            prepared[identity] = {"request": source, "query_enabled": True,
                                 "response_scenes": [{"request": source, "candidate": candidate}]}
        for budget, light, samples in ((None, True, 1), (2, True, 1), (0, True, 1), (None, False, 2)):
            with self.subTest(budget=budget, lightweight=light), TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                for name in ("dataset", "scenes"):
                    (root / name).mkdir()
                    (root / name / "manifest.json").write_text("{}")
                data = Mock()
                data.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
                data.catalog.connection.execute.return_value.fetchall.return_value = [(i,) for i in prepared]
                stack.enter_context(patch.object(module, "SFTDataset", return_value=data))
                stack.enter_context(patch.object(module, "load_tokenizer", return_value=object()))
                stack.enter_context(patch("agent_simulation.datasets.customer_scenes.read_scenes",
                                         return_value=(deepcopy(prepared), {})))
                predictor = stack.enter_context(patch.object(module, "OptionPredictor"))
                def probability_score(binding, trajectory):
                    self.assertNotIn("target", trajectory)
                    self.assertNotIn("source_response_target", trajectory)
                    raise module.ContextCapacityError("offline capacity fixture")
                predictor.return_value.score.side_effect = probability_score
                state = SimpleNamespace(active=0, peak=0, calls=0, clients=[])
                outer = self
                class Client:
                    def __init__(self, url, model, tokenizer, **kwargs):
                        self.model, self.seed = model, kwargs["seed"]
                        state.clients.append(self)
                        outer.assertEqual(kwargs["max_new_tokens"], 768 if light else 2048)
                        outer.assertEqual(kwargs["temperature"], 0 if light else .7)
                    def check_models(self): pass
                    def close(self): pass
                    async def acomplete(self, value, *, call_id):
                        state.active += 1
                        state.peak = max(state.active, state.peak)
                        state.calls += 1
                        outer.assertIn(self.seed, range(42, 42 + samples))
                        outer.assertEqual(concise_ekb_instructions() in value["messages"][0]["content"], light)
                        outer.assertNotIn("source_response_target", json.dumps(value))
                        await asyncio.sleep(.01)
                        state.active -= 1
                        if "query" in value["final_schema"].get("properties", {}):
                            return {"role": "assistant", "content": json.dumps({"query": "SECRET TARGET"}), "tool_calls": []}
                        return {"role": "assistant", "content": json.dumps({
                            "rationale": rationale(),
                            "action": {"clicked_item_ids": [], "purchased_item_ids": []}}), "tool_calls": []}
                stack.enter_context(patch.object(module, "AgenticClient", Client))
                argv = ["--dataset", str(root / "dataset"), "--scenes", str(root / "scenes"),
                        "--output", str(root / "output"), "--scene-concurrency", "2"]
                argv += ["--sft-only", "--cot-preset", "lightweight"] if light else ["--samples", "2"]
                if budget is not None:
                    argv += ["--teacher-budget", str(budget)]
                with redirect_stdout(io.StringIO()):
                    main(argv)
                expected = (3 if budget is None else budget) * samples * (1 if light else 4)
                self.assertEqual(state.calls, expected)
                self.assertEqual(state.peak, min(2 if light else 4, expected))
                if light:
                    predictor.assert_not_called()
                else:
                    self.assertEqual(predictor.return_value.score.call_count, expected // 2)
                rows = [json.loads(line) for line in (root / "output/records.jsonl").read_text().splitlines()]
                self.assertEqual([r["trajectory"]["source_request_id"] for r in rows],
                                 [i for i in ([11, 12, 13] if budget is None else [12, 13] if budget else [])
                                  for _ in range(samples * (1 if light else 4))])
                self.assertTrue(all(r["trajectory"]["status"] == "ok" for r in rows))
                observed = [json.loads(line) for line in (root / "output/observed-actions.jsonl").read_text().splitlines()]
                self.assertEqual(len(observed), 6)
                sft = [json.loads(line) for line in (root / "output/sft.jsonl").read_text().splitlines()]
                self.assertTrue(all(row in sft for row in observed))
                self.assertEqual((root / "output/preferences.jsonl").read_text(), "")
                self.assertEqual(len(state.clients), 0 if budget == 0 else 1 if light else 2)
                self.assertTrue(all(client.seed == 42 for client in state.clients))
