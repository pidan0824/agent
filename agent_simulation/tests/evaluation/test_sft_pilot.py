"""Pilot protocol leakage, pair identity and held-out export guards."""
from copy import deepcopy
from contextlib import redirect_stdout
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import io
import json
import unittest

from experiments.cot_sft_pilot import decision_request,export,quality,dump,comparison
from agent_simulation.data.common import sha256
from agent_simulation.evaluation.customer_curation import CurationTools
from agent_simulation.tests.evaluation.test_customer_agentic import request
from agent_simulation.tests.ekb_fixture import rationale


def case(stage="response",partition="fit"):
    source=request()
    return dict(case_id=stage+":11",role="customer",stage=stage,actor_id=7,source_request_id=11,
        source_split="train",partition=partition,stratum="fixture",
        dataset_manifest_sha256="frozen",
        context=CurationTools(source,stage,mode="fixed").initial_context(),
        target=source[stage+"_target"])


class PilotTests(unittest.TestCase):
    def test_targets_never_change_any_generation_request(self):
        for stage in ("query","response"):
            original=case(stage)
            modified=deepcopy(original)
            modified["target"]={"SECRET_LABEL":"must not be seen"}
            for style in ("direct","concise","original"):
                self.assertEqual(decision_request(original,style),decision_request(modified,style))
                if stage=="query":
                    self.assertNotIn("SECRET TARGET",dump(decision_request(original,style)))
        original=case()
        for style in ("direct","concise","original"):
            self.assertEqual(decision_request(original,style)["tools"],[])

    def test_concise_preserves_action_space_and_is_shorter_than_original(self):
        c=case()
        short=decision_request(c,"concise")
        original=decision_request(c,"original")
        self.assertEqual(short["final_schema"],original["final_schema"])
        self.assertLess(len(short["messages"][0]["content"]),len(original["messages"][0]["content"]))
        self.assertEqual(short["messages"][-1],original["messages"][-1])

    def test_correct_action_without_any_evidence_reference_is_not_a_teacher_example(self):
        c=case()
        r=rationale()
        r["evidence_refs"]={stage:[] for stage in ("stimulus","knowledge","evaluation")}
        req=decision_request(c,"concise")
        messages=req["messages"]+[{"role":"assistant","content":dump({"action":c["target"],"rationale":r})}]
        self.assertFalse(quality(c,c["target"],r,messages,"concise")["accepted"])

    def test_sampled_pilot_reuses_production_inputs_without_duplicate_query_or_budget(self):
        from experiments.cot_sft_pilot import sampled_customer_cases
        from agent_simulation.evaluation.customer_curation import observed_action_rows
        from agent_simulation.tests.datasets.test_customer_sampling import request as sampled_request, pool, Encoder
        from agent_simulation.datasets.customer_candidate_sampling import SampledCandidateBuilder
        from agent_simulation.agents.schemas import validate_arguments
        products, interactions = pool()
        source = sampled_request()
        bundle = {"request": source, "query_enabled": True,
                  "response_scenes": SampledCandidateBuilder(products, interactions, Encoder()).build_all(source)}
        observed = observed_action_rows(source, 100, "train", "release", bundle=bundle)
        cases = sampled_customer_cases(bundle, "fit", "fixture", "release")
        self.assertEqual([c["stage"] for c in cases], ["query", "response", "response"])
        for c, row in zip(cases, observed):
            direct = decision_request(c, "direct")
            self.assertEqual(direct["messages"], row["messages"][:-1])
            validate_arguments(c["target"], direct["final_schema"])
            self.assertEqual(decision_request(c, "concise")["messages"][1:], direct["messages"][1:])
            changed = deepcopy(c)
            changed["target"] = changed["source_response_target"] = {"SECRET": "hidden"}
            self.assertEqual(decision_request(changed, "concise"), decision_request(c, "concise"))
        with self.assertRaises(ValueError):
            validate_arguments(source["response_target"], decision_request(cases[1], "direct")["final_schema"])
        with TemporaryDirectory() as d:
            root = Path(d)
            (root / "cases.jsonl").write_text("".join(dump(c)+"\n" for c in cases))
            (root / "manifest.json").write_text(dump({"status":"complete","cases_sha256":sha256(root/"cases.jsonl")}))
            with redirect_stdout(io.StringIO()):
                export(Namespace(cohorts=root, teachers=[], output=root/"export"))
            saved = [json.loads(line) for line in (root/"export/customer.jsonl").read_text().splitlines()]
            for row, original in zip(saved, observed):
                self.assertEqual(row["messages"][:-1], original["messages"][:-1])
                self.assertEqual(row["target"], original["target"])
                self.assertEqual(row["scene_id"], original["scene_id"])

    def test_sampled_capacity_filter_preserves_other_children_and_query(self):
        from unittest.mock import patch
        from experiments.customer_sampled_pilot import filter_sampled_cases
        from agent_simulation.tests.datasets.test_customer_sampling import request as sampled_request, pool, Encoder
        from agent_simulation.datasets.customer_candidate_sampling import SampledCandidateBuilder
        products, interactions = pool()
        source = sampled_request()
        bundle = {"request": source, "query_enabled": True,
                  "response_scenes": SampledCandidateBuilder(products, interactions, Encoder()).build_all(source)}
        parent = {100: {"partition":"fit", "stratum":"fixture"}}
        with patch("experiments.customer_sampled_pilot.prompt_tokens",
                   side_effect=[range(n) for n in (10,10,200,200,10,10)]) as tokens:
            scenes, cases, skipped = filter_sampled_cases([bundle], parent, "release", None, 100)
        self.assertEqual(tokens.call_count, 6)
        self.assertEqual([c["stage"] for c in cases], ["query","response"])
        self.assertTrue(scenes[0]["query_enabled"])
        self.assertEqual(len(scenes[0]["response_scenes"]), 1)
        self.assertEqual(cases[1]["target"], {"clicked_item_ids":[11],"purchased_item_ids":[11]})
        self.assertEqual(skipped[0]["scene_id"], "100:positive:10")
        self.assertEqual(skipped[0]["input_tokens"], {"direct":200,"concise":200})

    def test_observed_export_excludes_dev_and_audit(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            rows=[case("query","fit"),case("response","dev")]
            (root/"cases.jsonl").write_text("".join(dump(r)+"\n" for r in rows))
            (root/"manifest.json").write_text(dump({"status":"complete","cases_sha256":sha256(root/"cases.jsonl")}))
            out=root/"export"
            with redirect_stdout(io.StringIO()):
                export(Namespace(cohorts=root,teachers=[],output=out))
            stored=[json.loads(line) for line in (out/"customer.jsonl").read_text().splitlines()]
            self.assertEqual(len(stored),1)
            self.assertEqual(stored[0]["stage"],"query")
            self.assertEqual(json.loads(stored[0]["messages"][-1]["content"]),rows[0]["target"])


    def test_evaluation_includes_invalid_and_rejects_unpaired_decoding(self):
        from experiments.cot_sft_pilot import write_json
        from agent_simulation.evaluation.trajectories import digest
        with TemporaryDirectory() as d:
            root=Path(d)
            c=case(partition="dev")
            (root/"cases.jsonl").write_text(dump(c)+"\n")
            checksum=sha256(root/"cases.jsonl")
            write_json(root/"manifest.json",dict(status="complete",cases_sha256=checksum))
            identity=dict(style="direct",cases_sha256=checksum,ids=[c["case_id"]],
                base_config_sha256="same",max_new_tokens=256,temperature=0,seed=42,
                partition="dev",model="fixture")
            for name,status in (("zero","invalid"),("sft","ok")):
                path=root/name
                path.mkdir()
                write_json(path/"manifest.json",identity)
                row=dict(case_id=c["case_id"],status=status,
                    request_sha256=digest(decision_request(c,"direct")))
                if status=="ok":row["prediction"]=c["target"]
                (path/"predictions.jsonl").write_text(dump(row)+"\n")
            args=Namespace(cohorts=root,baseline=root/"zero",sft=root/"sft",output=root/"comparison.json")
            with redirect_stdout(io.StringIO()):
                comparison(args)
            report=json.loads(args.output.read_text())
            self.assertEqual(report["invalid"]["zero_shot"],1)
            self.assertEqual(report["customer_response"]["baseline_mean"],0)
            self.assertEqual(report["customer_response"]["sft_mean"],1)
            metrics=report["customer_response"]["action_metrics"]
            self.assertEqual(metrics["zero_shot"]["model"]["invalid"],1)
            self.assertEqual(metrics["sft"]["model"]["pooled_action_micro_f1"],1)
            self.assertEqual(metrics["sft"]["always_no_action"]["pooled_action_micro_f1"],0)
            self.assertIn("observed_positive",report["customer_response"]["by_head"]["purchased_item_ids"])
            identity["temperature"]=.7
            write_json(root/"sft/manifest.json",identity)
            with self.assertRaisesRegex(ValueError,"decoding"):
                comparison(args)

    def test_shortlist_scores_weight_parent_requests_equally(self):
        from experiments.cot_sft_pilot import write_json
        from agent_simulation.evaluation.trajectories import digest
        with TemporaryDirectory() as d:
            root=Path(d)
            cases=[]
            for i, parent in enumerate((11,11,12)):
                c=case(partition="dev")
                c.update(case_id=f"response:{i}",source_request_id=parent,action_mode="single_item",
                         target={"clicked_item_ids":[4],"purchased_item_ids":[4]})
                cases.append(c)
            (root/"cases.jsonl").write_text("".join(dump(c)+"\n" for c in cases))
            checksum=sha256(root/"cases.jsonl")
            write_json(root/"manifest.json",dict(status="complete",cases_sha256=checksum))
            identity=dict(style="direct",cases_sha256=checksum,ids=[c["case_id"] for c in cases],
                base_config_sha256="same",max_new_tokens=256,temperature=0,seed=42,partition="dev",model="fixture")
            for name in ("zero","sft"):
                path=root/name;path.mkdir()
                write_json(path/"manifest.json",identity)
                rows=[]
                for i,c in enumerate(cases):
                    rows.append(dict(case_id=c["case_id"],status="ok",
                        request_sha256=digest(decision_request(c,"direct")),
                        prediction={"clicked_item_ids":[4] if name=="sft" and i<2 else [],"purchased_item_ids":[]}))
                (path/"predictions.jsonl").write_text("".join(dump(r)+"\n" for r in rows))
            with redirect_stdout(io.StringIO()):
                comparison(Namespace(cohorts=root,baseline=root/"zero",sft=root/"sft",output=root/"comparison.json"))
            result=json.loads((root/"comparison.json").read_text())["customer_response"]
            self.assertEqual((result["n"],result["scene_n"]),(2,3))
            self.assertEqual(result["sft_mean"],.25)
            click=result["by_head"]["clicked_item_ids"]
            self.assertEqual((click["n"],click["scene_n"],click["sft_mean"]),(2,3,.5))
            positive=click["observed_positive"]
            self.assertEqual((positive["n"],positive["scene_n"],positive["sft_mean"]),(2,3,.5))

    def test_export_rejects_rewritten_teacher_input(self):
        from experiments.cot_sft_pilot import write_json
        from agent_simulation.evaluation.trajectories import digest
        with TemporaryDirectory() as d:
            root=Path(d)
            c=case()
            (root/"cases.jsonl").write_text(dump(c)+"\n")
            checksum=sha256(root/"cases.jsonl")
            write_json(root/"manifest.json",dict(status="complete",cases_sha256=checksum))
            teacher=root/"teacher";teacher.mkdir()
            write_json(teacher/"manifest.json",dict(partition="fit",cases_sha256=checksum,
                model="fixture",style="concise"))
            req=decision_request(c,"concise")
            messages=deepcopy(req["messages"])
            messages[0]["content"]+=" injected target"
            messages.append({"role":"assistant","content":dump({"action":c["target"],"rationale":rationale()})})
            row=dict(case_id=c["case_id"],status="ok",style="concise",request_sha256=digest(req),
                prediction=c["target"],rationale=rationale(),messages=messages)
            (teacher/"predictions.jsonl").write_text(dump(row)+"\n")
            with self.assertRaisesRegex(ValueError,"frozen case"):
                export(Namespace(cohorts=root,teachers=[teacher],output=root/"export"))


    def test_manual_semantic_failure_cannot_be_overridden_by_action_match(self):
        c=case()
        r=rationale()
        req=decision_request(c,"concise")
        refs=list(c["context"]["facts"])
        r["evidence_refs"]={"stimulus":refs[:1],"knowledge":[],"evaluation":[]}
        messages=req["messages"]+[{"role":"assistant","content":dump({"action":c["target"],"rationale":r})}]
        verdict=quality(c,c["target"],r,messages,"concise",{"verdict":"fail","reason":"Unsupported preference"})
        self.assertFalse(verdict["accepted"])
        self.assertIn("semantic_review_failed",verdict["evidence"]["reasons"])


    def test_generation_resume_checks_prompt_and_needs_no_service_when_complete(self):
        from unittest.mock import patch
        from experiments.cot_sft_pilot import generate,write_json
        from agent_simulation.evaluation.trajectories import digest
        with TemporaryDirectory() as d:
            root=Path(d);c=case()
            (root/"cases.jsonl").write_text(dump(c)+"\n")
            checksum=sha256(root/"cases.jsonl")
            write_json(root/"manifest.json",dict(status="complete",cases_sha256=checksum))
            tok=root/"tokenizer";tok.mkdir();(tok/"config.json").write_text("{}")
            out=root/"generated";out.mkdir()
            args=Namespace(cohorts=root,partition="fit",role="customer",stage="response",limit=0,
                seed=42,style="concise",model="fixture",tokenizer=tok,output=out,
                max_new_tokens=768,temperature=0,concurrency=1,url="unused")
            identity=dict(requests_sha256=digest({c["case_id"]:decision_request(c,"concise")}),
                cases_sha256=checksum,ids=[c["case_id"]],base_config_sha256=sha256(tok/"config.json"),
                model="fixture",tokenizer=str(tok.resolve()),style="concise",partition="fit",
                max_new_tokens=768,temperature=0,seed=42,concurrency=1,
                decoder_schema_sha256=sha256(Path(__file__).resolve().parents[2] / "inference/vllm.py"))
            write_json(out/"manifest.json",identity)
            (out/"predictions.jsonl").write_text(dump(dict(case_id=c["case_id"],status="invalid"))+"\n")
            with patch("experiments.cot_sft_pilot.AgenticClient",side_effect=AssertionError("No service needed")),redirect_stdout(io.StringIO()):
                generate(args)
            self.assertEqual(json.loads((out/"completed.json").read_text())["resumed_records"],1)
            with patch("experiments.cot_sft_pilot.decision_request",return_value={"changed":"prompt"}):
                with self.assertRaisesRegex(ValueError,"Resume identity mismatch"):
                    generate(args)


    def test_invalid_query_is_scored_zero_without_embedding_empty_text(self):
        from unittest.mock import patch
        from experiments.cot_sft_pilot import write_json
        from agent_simulation.evaluation.trajectories import digest
        class Encoder:
            def encode_many(self,texts):
                if any(not text.strip() for text in texts):
                    raise AssertionError("Empty embedding input")
                return [[1.,0.] for text in texts]
        with TemporaryDirectory() as d:
            root=Path(d);c=case("query","dev")
            (root/"cases.jsonl").write_text(dump(c)+"\n")
            checksum=sha256(root/"cases.jsonl")
            write_json(root/"manifest.json",dict(status="complete",cases_sha256=checksum))
            identity=dict(style="direct",cases_sha256=checksum,ids=[c["case_id"]],
                base_config_sha256="same",max_new_tokens=256,temperature=0,seed=42,partition="dev",model="fixture")
            for name,status in (("zero","invalid"),("sft","ok")):
                path=root/name;path.mkdir()
                write_json(path/"manifest.json",identity)
                row=dict(case_id=c["case_id"],status=status,request_sha256=digest(decision_request(c,"direct")))
                if status=="ok":row["prediction"]=c["target"]
                (path/"predictions.jsonl").write_text(dump(row)+"\n")
            args=Namespace(cohorts=root,baseline=root/"zero",sft=root/"sft",output=root/"comparison.json")
            with patch("agent_simulation.agents.retrieval.bge_encoder.bge_encoder",return_value=(Encoder(),"fixture")),redirect_stdout(io.StringIO()):
                comparison(args)
            report=json.loads(args.output.read_text())
            self.assertEqual(report["customer_query_cosine"]["baseline_mean"],0.)
            self.assertEqual(report["customer_query_cosine"]["sft_mean"],1.)
            self.assertEqual(report["invalid"]["zero_shot"],1)


if __name__=="__main__":
    unittest.main()
