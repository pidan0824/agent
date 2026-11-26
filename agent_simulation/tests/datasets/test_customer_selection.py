"""Full-pool, past-only, diverse and reproducible scene quality selection."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import duckdb

from agent_simulation.datasets.customer_selection import rank_population, read_selection, main
from agent_simulation.data.common import sha256


class SceneSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.c = duckdb.connect()
        self.c.execute("SET threads=2")
        self.c.execute("CREATE TABLE users(user_id BIGINT)")
        self.c.execute("INSERT INTO users VALUES (1),(2),(3),(4),(5)")
        self.c.execute("CREATE TABLE products(item_id BIGINT,item_title VARCHAR)")
        self.c.execute("INSERT INTO products VALUES (10,'杯子'),(11,'水杯'),(12,NULL)")
        self.c.execute("""CREATE TABLE requests(session_id BIGINT,user_id BIGINT,time_index BIGINT,
            split VARCHAR,clicked_item_ids BIGINT[],purchased_item_ids BIGINT[])""")
        self.c.execute("""CREATE TABLE labels(session_id BIGINT,user_id BIGINT,history_cutoff_exclusive BIGINT,
            split VARCHAR,query_target VARCHAR,impressed_item_ids BIGINT[],
            clicked_item_ids BIGINT[],purchased_item_ids BIGINT[])""")
        # 1 and 2 share an actor. No same-time request counts as history.
        for identity, actor, items, clicked, purchased in (
            (1,1,[10,11],[],[]), (2,1,[10,11],[],[]), (3,2,[10,11],[],[]),
            (4,3,[10,12],[],[]), (5,4,[10,11],[10],[]),
            (6,5,[10,11],[],[11]), (7,2,[12],[],[]),
        ):
            self.c.execute("INSERT INTO requests VALUES (?,?,10,'train',?,?)",
                           [identity,actor,clicked,purchased])
            self.c.execute("INSERT INTO labels VALUES (?,?,10,'train','杯子',?,?,?)",
                           [identity,actor,items,clicked,purchased])
        self.labels = self.root / "labels.parquet"
        self.save_labels()

    def tearDown(self):
        self.c.close()
        self.temp.cleanup()

    def save_labels(self):
        self.c.execute("COPY labels TO ? (FORMAT PARQUET)", [str(self.labels)])

    def selected(self):
        return self.c.execute("SELECT session_id,actor_id FROM scene_selection WHERE selected ORDER BY session_id").fetchall()

    def test_per_scene_quality_and_actor_coverage_keep_negative_and_purchase(self):
        report = rank_population(self.c,self.labels,per_scene=2,seed=42)
        selected = self.selected()
        self.assertEqual(report["population"],7)
        self.assertEqual(report["exclusions"],{"no_exposure_titles":1})
        self.assertEqual(len([r for r in selected if r[1] == 1]),1)
        self.assertIn((3,2),selected)
        self.assertNotIn((4,3),selected)
        self.assertIn((5,4),selected)
        self.assertIn((6,5),selected)
        self.assertEqual(self.c.execute("SELECT max(prior_requests) FROM scene_selection").fetchone()[0],0)

    def test_future_and_test_events_cannot_affect_selection(self):
        rank_population(self.c,self.labels,per_scene=2,seed=42)
        before = self.c.execute("SELECT * FROM scene_selection ORDER BY session_id").fetchall()
        self.c.execute("INSERT INTO requests VALUES (100,1,11,'train',[10],[10]),(101,1,1,'test',[10],[10])")
        rank_population(self.c,self.labels,per_scene=2,seed=42)
        self.assertEqual(before,self.c.execute("SELECT * FROM scene_selection ORDER BY session_id").fetchall())
        rank_population(self.c,self.labels,per_scene=2,seed=42)
        self.assertEqual(before,self.c.execute("SELECT * FROM scene_selection ORDER BY session_id").fetchall())

    def test_test_labels_and_duplicate_requests_are_rejected(self):
        self.c.execute("UPDATE labels SET split='test' WHERE session_id=1")
        self.labels.unlink()
        self.save_labels()
        with self.assertRaisesRegex(ValueError,"Train only"):
            rank_population(self.c,self.labels,per_scene=2,seed=42)
        self.c.execute("UPDATE labels SET split='train'")
        self.c.execute("INSERT INTO labels SELECT * FROM labels WHERE session_id=1")
        self.labels.unlink()
        self.save_labels()
        with self.assertRaisesRegex(ValueError,"duplicate"):
            rank_population(self.c,self.labels,per_scene=2,seed=42)

    def test_selection_identity_hash_and_test_guards(self):
        rank_population(self.c,self.labels,per_scene=2,seed=42)
        dataset = self.root / "dataset"
        dataset.mkdir()
        (dataset / "manifest.json").write_text("{}")
        selected = self.root / "selected.parquet"
        self.c.execute("COPY (SELECT * FROM scene_selection WHERE selected) TO ? (FORMAT PARQUET)",[str(selected)])
        manifest = dict(schema="customer-train-scene-selection-v1",split="train",
            population_scope="complete_official_train",dataset_manifest_sha256=sha256(dataset/"manifest.json"),
            selected_sha256=sha256(selected),selected=len(self.selected()))
        (self.root/"manifest.json").write_text(json.dumps(manifest))
        ids,_ = read_selection(self.root,dataset=dataset,split="train")
        self.assertEqual(ids,[i for i,_ in self.selected()])
        with self.assertRaisesRegex(ValueError,"Test"):
            read_selection(self.root,dataset=dataset,split="test")
        (dataset/"manifest.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,"release differs"):
            read_selection(self.root,dataset=dataset,split="train")

    def test_curation_consumes_selected_ids_without_random_redraw(self):
        from contextlib import redirect_stdout
        from copy import deepcopy
        import io
        from unittest.mock import Mock, patch
        from agent_simulation.evaluation.customer_curation import main as curate
        from agent_simulation.tests.evaluation.test_customer_agentic import request
        from agent_simulation.tests.ekb_fixture import rationale
        source = request()
        source["assigned_split"] = "train"
        class Client:
            model, seed = "fixture", 42
            def __init__(self, *args, **kwargs): pass
            def check_models(self): pass
            def close(self): pass
            async def acomplete(self, value, **kwargs):
                return {"role": "assistant", "content": json.dumps({
                    "rationale": rationale(), "action": source["response_target"]}), "tool_calls": []}
        data = Mock()
        data.catalog.request.side_effect = lambda identity, **kwargs: dict(deepcopy(source),session_id=identity)
        data.catalog.connection.execute.return_value.fetchall.return_value = [(11,),(12,)]
        dataset = self.root / "dataset"
        selection = self.root / "selection"
        dataset.mkdir()
        selection.mkdir()
        (dataset/"manifest.json").write_text("{}")
        (selection/"manifest.json").write_text("{}")
        with patch("agent_simulation.evaluation.customer_curation.SFTDataset",return_value=data), \
             patch("agent_simulation.evaluation.customer_curation.AgenticClient",side_effect=Client), \
             patch("agent_simulation.datasets.customer_selection.read_selection",
                   return_value=([11,12],{"population":539012})), \
             patch("agent_simulation.evaluation.customer_curation.select_record_ids",
                   side_effect=AssertionError("must not redraw")), redirect_stdout(io.StringIO()):
            curate(["--dataset",str(dataset),"--selection",str(selection),"--candidate-policies","full",
                    "--output",str(self.root/"out"),"--samples","1","--sft-only"])
        self.assertEqual([call.args[0] for call in data.catalog.request.call_args_list],[11,12])
        rows = [json.loads(line) for line in (self.root/"out/sft.jsonl").read_text().splitlines()]
        self.assertEqual({row["source_request_id"] for row in rows},{11,12})
        self.assertEqual(len(rows),6)

    def test_old_time_window_is_not_silently_used_as_full_train(self):
        (self.root/"manifest.json").write_text(json.dumps({"time_window_experiment":{"fraction":.05}}))
        with self.assertRaisesRegex(ValueError,"complete Lite Train"):
            main(["--dataset",str(self.root),"--output",str(self.root/"out"),"--per-scene","2"])
        self.assertFalse((self.root/"out").exists())


if __name__ == "__main__":
    unittest.main()
