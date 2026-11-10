"""Incremental generation must match the independently queried reference builder."""

from copy import deepcopy
from collections import Counter
from itertools import groupby
from pathlib import Path
import tempfile
import unittest
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch
from agent_simulation.agents.retrieval.history import seller_history, seller_segment
from agent_simulation.agents.context.pricing import target_context
from agent_simulation.tests.fixtures import MarketFixture
from agent_simulation.datasets.samples.seller import SellerHistory
from agent_simulation.agents.context.pricing import build_pricing_context
from agent_simulation.agents.context.pricing import overview_context
from agent_simulation.datasets.samples.seller import sample_from_context
from agent_simulation.datasets.samples.seller import with_market_context
from agent_simulation.data.common import rows, literal


class StreamTests(MarketFixture, unittest.TestCase):
    def test_filtered_seller_targets_still_contribute_to_later_context(self):
        original = rows(
            self.c,
            f'SELECT * FROM read_parquet({literal(self.cfg.data / "seller_business/part_00.parquet")}) WHERE item_id=1 ORDER BY time_index LIMIT 1',
        )[0]
        stream = SellerHistory(10, "fixture")
        for index, split in [(1, None), (2, "train")]:
            with patch("agent_simulation.datasets.samples.seller.sample_from_context",
                       side_effect=AssertionError("Filtered price target was expanded")):
                self.assertEqual(list(stream.samples([
                    dict(original, time_index=index, assigned_split=split)
                ], split="validation")), [])
        samples = list(stream.samples([
            dict(original, time_index=3, assigned_split="validation")
        ], split="validation"))
        self.assertEqual(len(samples), 1)
        memory = samples[0]["input"]["short_term_memory"]
        self.assertEqual([e["time_index"] for e in memory], [1, 2])
        self.assertNotIn("coverage", samples[0])

    def test_shop_eligibility_uses_the_full_prefix_before_the_cohort(self):
        stream = SellerHistory(10, "fixture")
        stream.limit = 2
        a, b = self.data.records(1)[0], self.data.records(3)[0]
        for t in range(1, 7):
            cohort = [dict(a, time_index=t, assigned_split="train" if t > 1 else None),
                      dict(b, time_index=t, price_raw=200 if t == 3 else None,
                           price_status="observed" if t == 3 else "invalid_price")]
            samples = list(stream.samples(cohort))
            if samples:
                sample, = samples
                self.assertEqual([p["item_id"] for p in sample["input"]["owned_products"]], [1])
                self.assertNotIn("shop_summary", sample["input"])
                self.assertEqual(sorted(stream.valid_references), [1, 3] if t >= 3 else [1])
                self.assertEqual(sample["target"]["prices"][0]["item_id"], 1)

        # Item 3 remains in the history pool even when absent from target 1 input.
        sample, = list(stream.samples([dict(b, time_index=7, assigned_split="train")]))
        product, = sample["input"]["owned_products"]
        self.assertIsNone(product["last_observed_price_raw"])
        self.assertEqual(product["last_observed_time_index"], 6)
        self.assertEqual(product["last_valid_reference"], {"price_raw": 200, "time_index": 3})

    def test_equal_time_targets_only_use_prior_cohorts_and_long_history_matches(self):
        original = rows(
            self.c,
            f'SELECT * FROM read_parquet({literal(self.cfg.data / "seller_business/part_00.parquet")}) WHERE seller_id=10 ORDER BY time_index,item_id',
        )
        events = []
        for t in range(1, 25):
            for item in [1, 3]:
                r = deepcopy(next(x for x in original if x["item_id"] == item))
                r.update(
                    time_index=t,
                    price_raw=100 + t,
                    price_status="observed",
                    assigned_split="train" if t > 1 else None,
                )
                events.append(r)
        stream = SellerHistory(10, "fixture")
        past = []
        samples = []
        for cutoff, group in groupby(events, key=lambda r: r["time_index"]):
            cohort = list(group)
            actual = list(stream.samples(cohort))
            for target, sample in zip([r for r in cohort if r["assigned_split"]], actual):
                state, coverage = build_pricing_context(
                    past,
                    owner_id=10,
                    cutoff=cutoff,
                    observations_per_item=10,
                    public_pages=[],
                    requested_item_ids=[target["item_id"]],
                )
                expected = sample_from_context(
                    "fixture", target, target_context(overview_context(state, [target["item_id"]]), [target["item_id"]])
                )
                self.assertEqual(sample, expected)
                self.assertTrue(
                    all(
                        r["time_index"] < cutoff
                        for e in sample["input"]["short_term_memory"]
                        for r in e["content"]
                    )
                )
            samples.extend(actual)
            past.extend(cohort)
        self.assertEqual(len(samples), 46)

    def test_incremental_retrieval_matches_prefix_with_sparse_points_and_changed_titles(self):
        originals = rows(
            self.c,
            f'SELECT * FROM read_parquet({literal(self.cfg.data / "seller_business/part_00.parquet")}) WHERE seller_id=10 ORDER BY time_index,item_id',
        )
        profiles = {r["item_id"]: r for r in originals}
        events = []
        for step in range(1, 38):
            for item in [1, 3]:
                if item == 3 and step % 3:
                    continue
                row = deepcopy(profiles[item])
                row.update(
                    time_index=step * step + 2,
                    price_raw=None if step % 7 == 0 else 100 + step % 5,
                    price_status="invalid_price" if step % 7 == 0 else "observed",
                    click_records=int(step % 4 == 0),
                    purchase_records=int(step % 5 == 0),
                    item_title=f"商品{item} 描述版本{int(step>19)}",
                    assigned_split=None,
                )
                if step > 4 and row["price_status"] == "observed":
                    row["assigned_split"] = "train"
                events.append(row)

        def encode(text):
            return [float(n) + 1 for n in hashlib.sha256(text.encode()).digest()[:3]]

        for embedder in (None, encode):
            stream = SellerHistory(10, "fixture")
            past = []
            with patch(
                "agent_simulation.agents.retrieval.history.seller_segment",
                wraps=seller_segment,
            ) as prepare:
                for cutoff, group in groupby(events, key=lambda r: r["time_index"]):
                    cohort = list(group)
                    for sample in stream.samples(cohort):
                        focus = sample["input"]["requested_price_item_ids"]
                        actual = stream.retrieve(cutoff, focus, embedder=embedder)
                        prepared = prepare.call_count
                        self.assertEqual(stream.retrieve(cutoff, focus, embedder=embedder), actual)
                        self.assertEqual(prepare.call_count, prepared)
                        # Count only incremental preparation, excluding the independent replay.
                        with patch("agent_simulation.agents.retrieval.history.seller_segment", seller_segment):
                            expected = seller_history(
                                past,
                                seller_id=10,
                                cutoff=cutoff,
                                recent_n=10,
                                focus_ids=focus,
                                embedder=embedder,
                            )
                        self.assertEqual(actual, expected)
                        self.assertTrue(
                            all(
                                r["index"] < cutoff
                                for s in actual["segments"]
                                for r in s["records"]
                            )
                        )
                        if actual["segments"]:
                            actual["segments"][0]["records"][0]["index"] = 999999
                            self.assertEqual(
                                stream.retrieve(cutoff, focus, embedder=embedder), expected
                            )
                    past.extend(cohort)
                evicted = sum(max(0, count - 10) for count in Counter(r["item_id"] for r in past).values())
                # Recent chunks are new candidates; older chunks remain incremental.
                self.assertGreater(prepare.call_count, evicted)
                self.assertLessEqual(prepare.call_count, evicted + len(events) * 4)

    def test_partial_segments_are_searchable_and_same_index_is_excluded(self):
        original = rows(
            self.c,
            f'SELECT * FROM read_parquet({literal(self.cfg.data / "seller_business/part_00.parquet")}) WHERE item_id=1 ORDER BY time_index LIMIT 1',
        )[0]
        stream = SellerHistory(10, "fixture")
        for index in range(1, 15):
            row = dict(original, time_index=index, assigned_split="train" if index > 1 else None)
            for sample in stream.samples([row]):
                result = stream.retrieve(index, [1], embedder=None)
                lengths = [len(s["records"]) for s in result["segments"]]
                self.assertEqual(lengths, [index - 11] if index in (12, 13, 14) else [])
                self.assertTrue(
                    all(r["index"] < index for s in result["segments"] for r in s["records"])
                )
        segments = stream.windows[1].segments()
        self.assertEqual([len(s["records"]) for s in segments], [3, 1])

    def test_prepared_retrieval_avoids_another_prefix_query(self):
        past = rows(
            self.c,
            f'SELECT * FROM read_parquet({literal(self.cfg.data / "seller_business/part_00.parquet")}) WHERE seller_id=10 AND time_index<10 ORDER BY time_index,item_id',
        )
        target = dict(
            next(r for r in past if r["item_id"] == 1), time_index=10, assigned_split="test"
        )
        state, coverage = build_pricing_context(
            past,
            owner_id=10,
            cutoff=10,
            observations_per_item=10,
            public_pages=[],
            requested_item_ids=[1],
        )
        sample = sample_from_context("fixture", target, overview_context(state, [1]), render=False)
        self.assertNotIn("messages", sample)
        history = seller_history(
            past, seller_id=10, cutoff=10, recent_n=10, focus_ids=[1], embedder=None
        )
        market = {i: {"item_id": i, "semantics": "fixture", "status": "fixture"} for i in [1, 3]}
        catalog = SimpleNamespace(connection=self.c, config=self.cfg)
        with (
            patch(
                "agent_simulation.datasets.samples.seller.rows",
                side_effect=AssertionError("prefix reread"),
            ),
            patch("agent_simulation.agents.context.market.iter_decision_positions", return_value=market.items()),
        ):
            result = with_market_context(
                catalog, sample, retrieved_history=history, history_embedder=None
            )
        self.assertEqual(result["input"]["retrieved_context"]["history"], history)
        self.assertEqual(
            json.loads(result["messages"][-1]["content"]), result["target"]
        )

    def test_seller_capacity_rejection_preserves_history_and_other_errors_propagate(self):
        from agent_simulation.datasets.samples.seller import seller_records
        from agent_simulation.inference.requests import ContextCapacityError
        source = self.cfg.data / "seller_business/part_00.parquet"
        self.c.execute(f"CREATE VIEW seller_history AS SELECT * FROM read_parquet({literal(source)})")
        self.c.execute("""CREATE TABLE seller_targets AS SELECT seller_id,item_id,time_index,
                       'train' assigned_split FROM seller_history WHERE item_id=1 AND time_index>=4""")
        catalog = SimpleNamespace(connection=self.c, config=self.cfg)
        rejected, histories = [], []

        def prepare(catalog, sample, **kwargs):
            cutoff = sample["source"]["time_index"]
            state = kwargs["product_lookup"].__self__
            self.assertLess(state.last_time, cutoff)
            histories.append([r["time_index"] for r in state.windows[1].recent_rows()])
            if cutoff == 4:
                raise ContextCapacityError("fixture overflow")
            return sample

        with patch.object(SellerHistory, "retrieve", return_value={}), \
             patch("agent_simulation.datasets.samples.seller.with_market_context", side_effect=prepare):
            samples = list(seller_records(catalog, 0, 1,
                           on_capacity_error=lambda *args: rejected.append(args)))
        self.assertEqual([s["source"]["time_index"] for s in samples], [10])
        self.assertEqual(histories, [[2], [2, 4]])
        self.assertEqual(rejected[0][:2], ("seller:10:1:4", "train"))
        for error in (ContextCapacityError("overflow"), ValueError("bad facts")):
            with patch.object(SellerHistory, "retrieve", return_value={}), \
                 patch("agent_simulation.datasets.samples.seller.with_market_context", side_effect=error):
                if isinstance(error, ContextCapacityError):
                    with self.assertRaises(ContextCapacityError):
                        list(seller_records(catalog, 0, 1))
                else:
                    with self.assertRaisesRegex(ValueError, "bad facts"):
                        list(seller_records(catalog, 0, 1,
                             on_capacity_error=lambda *args: self.fail("Not a capacity failure")))
