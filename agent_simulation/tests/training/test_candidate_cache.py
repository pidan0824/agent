"""Selected records retain full history and exclude only complete oversized requests."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import duckdb
import numpy as np
from agent_simulation.data.common import literal, sha256
from agent_simulation.datasets.tokens import (
    bind_release, bind_split_candidates, build_splits, cache_tokenizer_digest, preflight,
)
from agent_simulation.datasets.tokens import PreparedRow
from agent_simulation.datasets.tokens import read_shard
from agent_simulation.datasets.tokens import validate_release, selected_identity
from agent_simulation.inference.requests import ContextCapacityError


class CandidateCacheTests(unittest.TestCase):
    def selected_fixture(self, root):
        from agent_simulation.datasets.tokens import write_shard
        dataset = root / 'dataset'
        dataset.mkdir()
        (dataset / 'facts').mkdir()
        (dataset / 'facts/customer_background.parquet').write_bytes(b'fixture-background')
        (dataset / 'manifest.json').write_text(json.dumps({'catalog': str(dataset), 'counts': {
            role: {'train': 1, 'test': 1} for role in ('customer', 'seller')}}))
        entries = {}
        with duckdb.connect() as connection:
            for split, record in (('train', '4'), ('test', '9')):
                candidates = root / f'customer-{split}-candidates.parquet'
                connection.execute(f"COPY (SELECT '{record}' record_id,7 actor_id,'purchase:recent' stratum) "
                                   f"TO {literal(candidates)} (FORMAT PARQUET)")
                entries[split] = dict(file=candidates.name, sha256=sha256(candidates),
                    available_by_stratum={'purchase:recent': 10}, selected_by_stratum={'purchase:recent': 1})
                base = root / 'customer' / split / '000'
                base.mkdir(parents=True)
                shard = base / '000000.npz'
                write_shard(shard, [PreparedRow(f'{stage}:{record}', np.array([1,2,3], dtype=np.int32), 2, 1.)
                                    for stage in ('query', 'response')])
                choices = base / 'selection.json'
                choices.write_text(json.dumps(dict(accepted=[record], excluded=[], candidate_records=1)))
                (base / 'complete.json').write_text(json.dumps(dict(rows=2, records=1, candidate_records=1,
                    excluded_records=0, selection_sha256=sha256(choices),
                    files=[dict(file=shard.name, sha256=sha256(shard), rows=2)])))
        selection = dict(schema='sft-split-candidates-v2', seed=42,
            dataset_manifest_sha256=sha256(dataset/'manifest.json'),
            roles={'customer': entries, 'seller': {'train': {}, 'test': {}}})
        candidate = root / 'candidate_manifest.json'
        candidate.write_text(json.dumps(selection))
        spec = selected_identity(dataset, selection, sha256(candidate), 42, 1)
        (root / 'identity.json').write_text(json.dumps(spec))
        return dataset, spec

    def test_build_preflight_can_verify_candidate_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, _ = self.selected_fixture(root)
            manifest = root / 'candidate_manifest.json'
            selection = json.loads(manifest.read_text())
            selection['roles'].pop('seller')
            manifest.write_text(json.dumps(selection))
            args = SimpleNamespace(dataset=dataset, tokenizer=root, candidates=manifest,
                                   market_cache=None, seed=42, buckets=1)
            candidate = root / 'customer-train-candidates.parquet'
            candidate.write_bytes(b'changed')
            with patch('agent_simulation.datasets.tokens.SFTDataset') as load, \
                 patch('agent_simulation.datasets.tokens.cache_tokenizer_digest', return_value='digest'):
                preflight(args)
                load.assert_not_called()
                args.verify = True
                with self.assertRaisesRegex(ValueError, 'Candidate file changed'):
                    preflight(args)

    def test_cache_rejects_a_changed_chat_template(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, spec = self.selected_fixture(root)
            tokenizer_path = root / 'tokenizer'
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(WordLevel({'[UNK]': 0}, unk_token='[UNK]')),
                unk_token='[UNK]', chat_template='{{ messages[0].content }}',
            )
            tokenizer.save_pretrained(tokenizer_path)
            spec['tokenizer_sha256'] = cache_tokenizer_digest(tokenizer_path)
            (root / 'identity.json').write_text(json.dumps(spec))
            (root / 'complete.json').write_text(json.dumps(spec))
            args = SimpleNamespace(output=root, candidates=root / 'candidate_manifest.json')
            bind_release(args, spec)
            self.assertEqual(validate_release(root, dataset, 42, tokenizer_path=tokenizer_path), spec)
            tokenizer.chat_template = 'Changed: {{ messages[0].content }}'
            tokenizer.save_pretrained(tokenizer_path)
            changed = {**spec, 'tokenizer_sha256': cache_tokenizer_digest(tokenizer_path)}
            with self.assertRaisesRegex(ValueError, 'chat template differs'):
                bind_release(args, changed)
            with self.assertRaisesRegex(ValueError, 'chat template differs'):
                validate_release(root, dataset, 42, tokenizer_path=tokenizer_path)
    def test_training_requires_finished_cache_with_bound_candidate_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset, spec = self.selected_fixture(root)
            with self.assertRaises(FileNotFoundError):
                validate_release(root, dataset, 42)
            (root/'complete.json').write_text(json.dumps(spec))
            self.assertEqual(validate_release(root, dataset, 42), spec)
            with self.assertRaises(ValueError):
                validate_release(root, dataset, 43)
            (root/'candidate_manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'manifest changed'):
                validate_release(root, dataset, 42)

    def test_obsolete_candidate_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'Expected split candidate manifest'):
                selected_identity(Path(temp), {'schema': 'obsolete'}, 'hash', 42, 1)

    def test_customer_filter_keeps_nonselected_prior_requests_and_equal_time_cohort(self):
        with tempfile.TemporaryDirectory() as temp, duckdb.connect() as c:
            path=Path(temp)/'selected.parquet'
            c.execute("CREATE TABLE requests AS SELECT 7 user_id,i session_id,i time_index FROM range(1,6) t(i)")
            c.execute("CREATE TABLE customer_splits AS SELECT session_id,'train' assigned_split FROM requests")
            c.execute(f"COPY (SELECT 4 session_id,7 actor_id,4 history_cutoff_exclusive,'4' record_id) TO {literal(path)} (FORMAT PARQUET)")
            self.assertEqual(bind_split_candidates(c,{'train':path},'customer',0,1),[('4','train')])
            self.assertEqual(c.execute('SELECT session_id FROM requests ORDER BY 1').fetchall(),[(1,),(2,),(3,),(4,)])
            self.assertEqual(c.execute('SELECT session_id FROM customer_splits WHERE assigned_split IS NOT NULL').fetchall(),[(4,)])

    def test_cached_requests_remain_history_for_later_new_decisions(self):
        with tempfile.TemporaryDirectory() as temp,duckdb.connect() as c:
            path=Path(temp)/'selected.parquet'
            c.execute("CREATE TABLE requests AS SELECT 7 user_id,i session_id,i time_index FROM range(1,6) t(i)")
            c.execute("CREATE TABLE customer_splits AS SELECT session_id,'train' assigned_split FROM requests")
            c.execute(f"COPY (SELECT session_id,7 actor_id,time_index history_cutoff_exclusive,CAST(session_id AS VARCHAR) record_id FROM requests WHERE session_id IN (2,4)) TO {literal(path)} (FORMAT PARQUET)")
            self.assertEqual(bind_split_candidates(c,{'train':path},'customer',0,1,['2']),[('2','train'),('4','train')])
            self.assertEqual(c.execute('SELECT session_id FROM requests ORDER BY 1').fetchall(),[(1,),(2,),(3,),(4,)])
            self.assertEqual(c.execute('SELECT session_id FROM customer_splits WHERE assigned_split IS NOT NULL').fetchall(),[(4,)])

    def test_seller_cache_preserves_history_resume_and_source_splits(self):
        from agent_simulation.datasets.tokens import bind_selected_history
        for cached in (False, True):
            for restored in ([], ['2'], ['2', '4']):
                with self.subTest(cached=cached, restored=restored), duckdb.connect() as c:
                    c.execute('''CREATE TABLE seller_history AS
                        SELECT seller_id,11 item_id,i time_index,i*100 price_raw
                        FROM (VALUES (7),(8)) s(seller_id),range(1,7) t(i)''')
                    c.execute("""CREATE TABLE seller_targets AS
                        SELECT seller_id,item_id,time_index,'train' assigned_split
                        FROM seller_history WHERE time_index IN (2,4,6)""")
                    c.execute("""CREATE TEMP TABLE selected_records AS
                        SELECT CAST(time_index AS VARCHAR) record_id,seller_id actor_id,
                            seller_id,item_id,time_index history_cutoff_exclusive,'test' cache_split
                        FROM seller_targets WHERE seller_id=7 AND time_index IN (2,4)""")
                    if cached:
                        c.execute("ATTACH ':memory:' AS seller_market_cache")
                        c.execute('''CREATE TABLE seller_market_cache.by_seller AS
                            SELECT * FROM seller_history ORDER BY seller_id,time_index,item_id''')
                    bind_selected_history(c, 'seller', restored)
                    history=c.execute('SELECT seller_id,time_index,price_raw FROM seller_history ORDER BY time_index').fetchall()
                    targets=c.execute('SELECT time_index,assigned_split FROM seller_targets ORDER BY time_index').fetchall()
                    if len(restored)==2:
                        self.assertEqual(history,[])
                        self.assertEqual(targets,[])
                    else:
                        self.assertEqual(history,[(7,i,i*100) for i in range(1,5)])
                        # Restore skips labels only, and candidate metadata cannot relabel source facts.
                        self.assertEqual(targets,[(i,'train') for i in (2,4) if str(i) not in restored])
                    view=c.execute("SELECT sql FROM duckdb_views() WHERE view_name='seller_history' AND temporary").fetchone()[0]
                    self.assertEqual('seller_market_cache.by_seller' in view,cached)

    def test_capacity_rejection_is_whole_record_and_other_errors_fail(self):
        for failure in (ContextCapacityError('too long'),ValueError('corrupt label')):
            with self.subTest(failure=type(failure)),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                args=SimpleNamespace(output=root,dataset=root,tokenizer=root,candidates=root/'manifest.json',
                                     buckets=1,reserve_gib=0)
                data=SimpleNamespace(catalog=SimpleNamespace(connection=SimpleNamespace(execute=lambda *a:None)),
                                     tokenizer=None,close=lambda:None)
                raw=[({'key':{'session_id':i}}, {}) for i in (1,2)]
                row=PreparedRow('query:2',np.array([1,2,3],dtype=np.int32),2,1.)
                with patch('agent_simulation.datasets.tokens.SFTDataset',return_value=data), \
                     patch('agent_simulation.datasets.tokens.bind_split_candidates',return_value=[('1','train'),('2','train')]), \
                     patch('agent_simulation.datasets.tokens.customer_records',return_value=iter(raw)), \
                     patch('agent_simulation.datasets.tokens.expand_record',
                           side_effect=[[{'split':'train'}] * 2, [{'split':'train'}]]), \
                     patch('agent_simulation.datasets.tokens.PreparedRow.from_sample',
                           side_effect=[row, failure, row]):
                    if type(failure) is ValueError:
                        with self.assertRaisesRegex(ValueError,'corrupt label'):build_splits(args, 'customer', 0, ('train',))['train']
                        self.assertFalse((root/'customer/train/000/complete.json').exists())
                        continue
                    result=build_splits(args, 'customer', 0, ('train',))['train']
                base=root/'customer/train/000'
                audit=json.loads((base/'selection.json').read_text())
                self.assertEqual(audit['accepted'],['2'])
                self.assertEqual(audit['excluded'][0]['record_id'],'1')
                logged=json.loads((base/'failures.jsonl').read_text())
                self.assertEqual(logged,audit['excluded'][0])
                self.assertEqual((logged['role'],logged['split']),('customer','train'))
                self.assertEqual((result['records'],result['candidate_records'],result['excluded_records']),(1,2,1))
                self.assertEqual([r.sample_id for r in read_shard(base/'000000.npz')],['query:2'])

    def test_late_response_overflow_discards_query_and_earlier_batches(self):
        from agent_simulation.datasets.labels import checked_record
        from agent_simulation.tests.fixtures import DisplayBudgetTokenizer
        query={'sample_id':'query:1','input':{},'target':{'query':'杯子'}}
        response={'sample_id':'response:1','input':{'displayed_items':[{'item_id':1},{'item_id':2}]},
                  'target':{'clicked_item_ids':[1],'purchased_item_ids':[]}}
        calls=[]
        def check(sample):
            calls.append(sample['sample_id'])
            if len(calls)==3:raise ContextCapacityError('long second batch')
            return sample
        data = SimpleNamespace(check_sample=check, tokenizer=DisplayBudgetTokenizer(1))
        with self.assertRaisesRegex(ContextCapacityError, 'second batch'):
            checked_record(data, 'customer', (query, response))
        self.assertEqual(calls,['query:1','response:1:batch:1','response:1:batch:2'])



if __name__=='__main__':unittest.main()
