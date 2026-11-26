"""Response-local content deduplication, history protection and frozen prices."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import duckdb

from agent_simulation.data.common import sha256
from agent_simulation.datasets.customer_candidate_sampling import (
    CANDIDATE_DEDUPLICATION, SampledCandidateBuilder, deduplicate_pool,
)
from agent_simulation.datasets import customer_scenes
from agent_simulation.evaluation.customer_curation import observed_action_rows, CurationTools
from agent_simulation.evaluation.customer_filtering import combine_observed_sft
from agent_simulation.tests.datasets.test_customer_sampling import Encoder, pool, request


def fixture():
    products, interactions = pool()
    for p in products:
        p.update(seller_id=7, seller_name="shop", brand_id=8, brand_name="brand",
                 category_level1_id=1, category_level1_name="food",
                 category_level2_id=2, category_level2_name="snack",
                 category_level3_id=3, category_level3_name="plum")
    by_id = {p["item_id"]: p for p in products}
    for item, original in ((11, 10), (13, 12)):
        by_id[item].update({k: v for k, v in by_id[original].items() if k != "item_id"})
    source = request({"clicked_item_ids": [10], "purchased_item_ids": [10]})
    source["response_input"]["displayed_items"] = [
        {**deepcopy(by_id[i]), "price_raw": 999.} for i in range(10, 22)]
    prices = {i: (50., 3) for i in range(10, 22)}
    return products, interactions, source, prices


class CandidateDeduplicationTests(unittest.TestCase):
    def test_target_kept_even_with_larger_id_and_price_time_is_not_a_content_difference(self):
        products, _, _, prices = fixture()
        prices[11] = (50., 1)
        by_id = {p["item_id"]: p for p in products}
        before = deepcopy((by_id, prices))
        ids, audit = deduplicate_pool([10, 11], by_id, prices, set(), target_item_id=11, seed=42)
        self.assertEqual(ids, [11])
        self.assertEqual(audit["removed_to_retained"], [{"removed": 10, "retained": 11}])
        self.assertEqual(audit["protocol"], CANDIDATE_DEDUPLICATION)
        self.assertEqual((by_id, prices), before)

    def test_different_attributes_and_missing_metadata_never_collapse(self):
        products, _, _, prices = fixture()
        original = {p["item_id"]: p for p in products}
        changes = {"seller_id": 9, "seller_name": "other shop", "item_title": "other",
                   "brand_id": 9, "brand_name": "other brand", "category_level1_id": 9,
                   "category_level1_name": "other", "category_level2_id": 9,
                   "category_level2_name": "other", "category_level3_id": 9,
                   "category_level3_name": "other"}
        for field, value in changes.items():
            with self.subTest(field=field):
                products = deepcopy(original)
                products[11][field] = value
                ids, _ = deduplicate_pool([10, 11], products, prices, set(), target_item_id=10, seed=1)
                self.assertEqual(ids, [10, 11])
        for missing in (None, "", "unknown"):
            products = deepcopy(original)
            products[10]["seller_name"] = products[11]["seller_name"] = missing
            ids, _ = deduplicate_pool([10, 11], products, prices, set(), target_item_id=10, seed=1)
            self.assertEqual(ids, [10, 11])

    def test_price_difference_unknown_status_and_legacy_callers(self):
        products, _, _, _ = fixture()
        products = {p["item_id"]: p for p in products}
        for prices, expected in (({10: (50., 3), 11: (51., 3)}, [10, 11]),
                                 ({10: (50., 3)}, [10, 11]), ({}, [10]), (None, [10, 11])):
            with self.subTest(prices=prices):
                ids, _ = deduplicate_pool([10, 11], products, prices, set(), target_item_id=10, seed=1)
                self.assertEqual(ids, expected)

    def test_all_sampling_routes_fill_from_remaining_unique_exposures(self):
        products, interactions, source, prices = fixture()
        before = deepcopy((products, interactions, source, prices))
        for method in ("random", "content", "collaborative"):
            with self.subTest(method=method):
                builder = SampledCandidateBuilder(products, interactions, Encoder(), methods=(method,),
                                                   min_items=8, max_items=8)
                scene = builder.build(source, price_observations=prices)
                ids = scene["candidate"]["candidate_item_ids"]
                self.assertEqual(len(ids), 8)
                self.assertIn(10, ids)
                self.assertNotIn(11, ids)
                self.assertLessEqual(len(set(ids) & {12, 13}), 1)
                self.assertTrue(set(ids) <= set(range(10, 22)))
                self.assertEqual(scene["request"]["response_target"], source["response_target"])
                self.assertEqual(scene["request"]["query_input"], source["query_input"])
                self.assertEqual(scene["candidate"]["scope"]["sampling_pool_size"], 10)
                self.assertEqual(scene, builder.build(source, price_observations=prices))
                visible = json.dumps(CurationTools(scene["request"], "response", mode="fixed").initial_context())
                self.assertNotIn("removed_to_retained", visible)
                self.assertNotIn("history_protected_groups", visible)
        self.assertEqual((products, interactions, source, prices), before)

    def test_own_prior_history_protects_the_group_other_users_and_future_do_not(self):
        for kind in ("prior", "other_user", "equal_time", "future", "context"):
            with self.subTest(kind=kind):
                products, interactions, source, prices = fixture()
                if kind == "context":
                    source["response_input"]["short_term_memory"]["search_history"] = [
                        {"session_id": 1, "time_index": 4, "clicked_items": [],
                         "purchased_items": [{"item_id": 10}], "query": "plum"}]
                else:
                    interactions.append({"user_id": 2 if kind == "other_user" else 1, "item_id": 10,
                                         "time_index": {"prior": 4, "other_user": 4, "equal_time": 20, "future": 21}[kind]})
                builder = SampledCandidateBuilder(products, interactions, None, methods=("random",),
                                                   min_items=11, max_items=11)
                scene = builder.build(source, price_observations=prices)
                audit = scene["candidate"]["scope"]["deduplication"]
                if kind in ("prior", "context"):
                    self.assertIn([10, 11], audit["history_protected_groups"])
                    self.assertTrue({10, 11} <= set(scene["candidate"]["candidate_item_ids"]))
                else:
                    self.assertNotIn([10, 11], audit["history_protected_groups"])
                    self.assertNotIn(11, scene["candidate"]["candidate_item_ids"])

    def test_background_interaction_protection_is_request_local(self):
        products, interactions, source, prices = fixture()
        source["response_input"]["behavioral_summary"] = {
            "observed_time_index": 4, "recent_clicked_item_ids": [11], "recent_purchased_item_ids": []}
        builder = SampledCandidateBuilder(products, interactions, None, methods=("random",),
                                           min_items=11, max_items=11)
        first = builder.build(source, price_observations=prices)
        self.assertTrue({10, 11} <= set(first["candidate"]["candidate_item_ids"]))
        other = deepcopy(source)
        other["session_id"] += 1
        other["response_input"].pop("behavioral_summary")
        second = builder.build(other, price_observations=prices)
        self.assertNotIn(11, second["candidate"]["candidate_item_ids"])
        for field, value in (("observed_time_index", 20), ("user_id", 2)):
            bad = deepcopy(source)
            bad["response_input"]["behavioral_summary"][field] = value
            with self.assertRaises(ValueError):
                builder.build(bad, price_observations=prices)

    def test_foreign_context_and_nonprior_prices_are_rejected(self):
        products, interactions, source, prices = fixture()
        builder = SampledCandidateBuilder(products, interactions, None, methods=("random",))
        bad = deepcopy(source)
        bad["response_input"]["short_term_memory"]["search_history"] = [
            {"time_index": 1, "user_id": 2, "clicked_items": [{"item_id": 10}], "purchased_items": []}]
        with self.assertRaisesRegex(ValueError, "Foreign Customer history"):
            builder.build(bad, price_observations=prices)
        for price, when in ((50., 20), (50., 21), (-1., 3), (float("nan"), 3)):
            bad_prices = {**prices, 10: (price, when)}
            with self.assertRaisesRegex(ValueError, "strictly-prior candidate prices"):
                builder.build(source, price_observations=bad_prices)

    def test_short_pool_preserves_positive_and_no_action_supervision(self):
        products, interactions, source, prices = fixture()
        source["response_input"]["displayed_items"] = source["response_input"]["displayed_items"][:2]
        prices = {i: prices[i] for i in (10, 11)}
        builder = SampledCandidateBuilder(products, interactions, None, methods=("random",),
                                           min_items=5, max_items=5)
        for target in ({"clicked_item_ids": [10], "purchased_item_ids": [10]},
                       {"clicked_item_ids": [], "purchased_item_ids": []}):
            source["response_target"] = deepcopy(target)
            before = deepcopy(source)
            children = builder.build_all(source, price_observations=prices)
            self.assertEqual(len(children), 1)
            self.assertTrue(children[0]["candidate"]["short_pool"])
            self.assertEqual(children[0]["candidate"]["candidate_count"], 1)
            self.assertEqual(children[0]["request"]["response_target"], target)
            observed = observed_action_rows(source, 100, "train", "release",
                bundle={"request": source, "query_enabled": True, "response_scenes": children})
            self.assertEqual(len(observed), 2)
            self.assertEqual(combine_observed_sft([], [], observed), observed)
            self.assertEqual(source, before)

    def catalog(self, products):
        connection = duckdb.connect()
        self.addCleanup(connection.close)
        connection.execute("CREATE TABLE products AS SELECT * FROM read_json_auto(?)", [self.products_file(products)])
        connection.execute("CREATE TABLE requests(session_id BIGINT,user_id BIGINT,time_index BIGINT,split VARCHAR,clicked_item_ids BIGINT[],purchased_item_ids BIGINT[])")
        connection.executemany("INSERT INTO requests VALUES (?,?,?,?,?,?)",
            [(i, 1, i, "train", [i] if i < 4 else [], []) for i in (1, 2, 3, 20, 21)])
        connection.execute("CREATE TABLE ranking_observations(session_id BIGINT,item_id BIGINT,price_raw DOUBLE)")
        connection.executemany("INSERT INTO ranking_observations VALUES (?,?,?)",
            [(1 if i == 11 else 3, i, 51. if i == 13 else 50.) for i in range(10, 22)]
            + [(20, 11, 700.), (21, 11, 800.)])
        return SimpleNamespace(connection=connection)

    def products_file(self, products):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "products.jsonl"
        path.write_text("".join(json.dumps(p) + "\n" for p in products))
        return str(path)

    def test_formal_preparation_prices_once_and_preserves_all_positive_ids(self):
        products, _, source, _ = fixture()
        next(p for p in products if p["item_id"] == 15)["item_title"] = "item14"
        source["response_target"] = {"clicked_item_ids": [10, 11, 14], "purchased_item_ids": [11]}
        original = deepcopy(source)
        catalog = self.catalog(products)
        with patch.object(customer_scenes, "prior_prices", wraps=customer_scenes.prior_prices) as lookup:
            bundles, skipped = customer_scenes.prepare_scenes(catalog, [source], None,
                methods=("random",), min_items=10, max_items=10)
        lookup.assert_called_once()
        self.assertFalse(skipped)
        children = bundles[0]["response_scenes"]
        self.assertEqual([c["candidate"]["target_item_id"] for c in children], [10, 11, 14])
        for child in children:
            chosen = child["candidate"]["target_item_id"]
            ids = set(child["candidate"]["candidate_item_ids"])
            self.assertEqual(ids & {10, 11, 14}, {chosen})
            self.assertTrue({12, 13} <= ids)  # Same current raw price, different prior prices.
            if chosen == 14:
                self.assertTrue(child["candidate"]["short_pool"])
            products = {p["item_id"]: p for p in child["request"]["response_input"]["displayed_items"]}
            if chosen in (10, 11):
                self.assertEqual(products[chosen]["price_raw"], 50.)
                self.assertEqual(products[chosen]["price_observed_time_index"], 1 if chosen == 11 else 3)
        self.assertEqual(source, original)
        observed = observed_action_rows(source, 100, "train", "release", bundle=bundles[0])
        self.assertEqual(len(observed), 4)
        self.assertEqual(combine_observed_sft([], [], observed), observed)

    def test_cli_records_policy_and_loader_rejects_mixed_children(self):
        products, _, source, _ = fixture()
        catalog = self.catalog(products)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            catalog.directory = root / "catalog"
            catalog.directory.mkdir()
            (catalog.directory / "manifest.json").write_text("{}")
            catalog.request = lambda identity, **kwargs: deepcopy(source)
            catalog.close = lambda: None
            dataset = root / "dataset"
            (dataset / "customer/train").mkdir(parents=True)
            catalog.connection.execute("COPY (SELECT 100 AS session_id) TO ? (FORMAT PARQUET)",
                                       [str(dataset / "customer/train/labels.parquet")])
            (dataset / "manifest.json").write_text(json.dumps({
                "catalog": str(catalog.directory),
                "catalog_manifest_sha256": sha256(catalog.directory / "manifest.json")}))
            output = root / "scenes"
            with patch.object(customer_scenes, "Catalog", return_value=catalog), redirect_stdout(io.StringIO()):
                customer_scenes.main(["--dataset", str(dataset), "--output", str(output),
                                      "--request-id", "100", "--methods", "random"])
            loaded, manifest = customer_scenes.read_scenes(output, dataset=dataset, split="train")
            self.assertEqual(manifest["candidate_deduplication"], CANDIDATE_DEDUPLICATION)
            child = loaded[100]["response_scenes"][0]
            self.assertNotIn(11, child["candidate"]["candidate_item_ids"])
            child["candidate"]["scope"]["deduplication"]["protocol"] = None
            (output / "scenes.jsonl").write_text(json.dumps(loaded[100]) + "\n")
            manifest["scenes_sha256"] = sha256(output / "scenes.jsonl")
            (output / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Child deduplication policy"):
                customer_scenes.read_scenes(output)
            manifest.pop("candidate_deduplication")
            (output / "manifest.json").write_text(json.dumps(manifest))
            customer_scenes.read_scenes(output)  # Legacy scene directories remain readable.
