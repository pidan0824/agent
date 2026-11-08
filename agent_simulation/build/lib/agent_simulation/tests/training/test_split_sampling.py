"""Proportional split sampling is stable and preserves complete decisions."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import duckdb
import numpy as np
from agent_simulation.data.common import literal, sha256
from agent_simulation.datasets.sampling import (
    customer_features, fraction_count, prepare_population, quotas, read_config, select_by_stratum,
)
from agent_simulation.datasets.tokens import selected_identity, validate_release
from agent_simulation.datasets.tokens import write_shard
from agent_simulation.datasets.tokens import build_splits
from agent_simulation.datasets.tokens import PreparedRow


class SplitSamplingTests(unittest.TestCase):
    def test_pipeline_extracts_each_role_once_and_keeps_split_membership(self):
        from agent_simulation.datasets import sampling as sampling
        with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as c:
            root = Path(tmp)
            dataset = root / 'dataset'
            dataset.mkdir()
            c.execute("CREATE TABLE requests AS SELECT i session_id,7 user_id,i time_index,"
                      "if(i<=4,'train','test') split FROM range(1,9) t(i)")
            c.execute("CREATE TABLE seller_history AS SELECT 1 seller_id,2 item_id,3 category_level1_id,"
                      "i time_index,100.+i price_raw,'observed' price_status FROM range(0,9) t(i)")
            c.execute("""INSERT INTO seller_history VALUES
                (3,4,3,2,100.,'observed'), (3,4,3,3,110.,'observed'),
                (2,3,3,6,100.,'observed'), (2,3,3,7,110.,'observed')""")
            files = []
            for role in ('customer', 'seller'):
                for split in ('train', 'test'):
                    path = dataset / role / split / 'labels.parquet'
                    path.parent.mkdir(parents=True)
                    if role == 'customer':
                        query = ("SELECT session_id,user_id,time_index history_cutoff_exclusive,"
                                 "CASE WHEN session_id IN (1,6,7) THEN [2,3] "
                                 "WHEN session_id IN (2,3,5) THEN [2,4] ELSE [2] END impressed_item_ids,"
                                 "[]::INT[] clicked_item_ids,[]::INT[] purchased_item_ids "
                                 f"FROM requests WHERE split='{split}'")
                    else:
                        bounds = 'time_index BETWEEN 1 AND 4' if split == 'train' else 'time_index BETWEEN 5 AND 8'
                        query = ("SELECT 'seller:'||seller_id||':'||item_id||':'||time_index sample_id,seller_id,item_id,"
                                 "time_index history_cutoff_exclusive,price_raw target_price_raw "
                                 f"FROM seller_history WHERE {bounds} AND "
                                 "(item_id=2 OR (item_id=4 AND time_index=3) OR (item_id=3 AND time_index=7))")
                    c.execute(f'COPY ({query}) TO {literal(path)} (FORMAT PARQUET)')
                    files.append(dict(path=str(path.relative_to(dataset)), sha256=sha256(path)))
            (dataset/'manifest.json').write_text(json.dumps(dict(schema='kuaisearch-on-demand-sft-v1',
                files=files, catalog='fixture',
                counts={'customer': {'train': 4, 'test': 4}, 'seller': {'train': 5, 'test': 5}})))
            config = root / 'sampling.json'
            config.write_text(json.dumps(dict(schema='sft-split-sampling-config-v1', seed=42,
                fractions={'train': .5, 'test': .5},
                strata={role: ['action', 'history_band'] for role in ('customer', 'seller')})))
            catalog = SimpleNamespace(connection=c, close=lambda: None)
            with patch.object(sampling, 'Catalog', return_value=catalog), \
                 patch.object(sampling, 'customer_features', wraps=sampling.customer_features) as customer, \
                 patch.object(sampling, 'seller_features', wraps=sampling.seller_features) as seller, \
                 patch('builtins.print'):
                result = sampling.select(dataset, root/'selected', config)
            self.assertEqual((customer.call_count, seller.call_count), (1, 1))
            for role, splits in result['roles'].items():
                for split, entry in splits.items():
                    cursor = c.execute('SELECT * FROM read_parquet(?)', [str(root/'selected'/entry['file'])])
                    names = [d[0] for d in cursor.description]
                    rows = [dict(zip(names, row)) for row in cursor.fetchall()]
                    self.assertNotIn('sampling_split', names)
                    self.assertEqual(len(rows), 2 if role == 'customer' else 3)
                    self.assertTrue(all((r['history_cutoff_exclusive'] <= 4) == (split == 'train') for r in rows))
                    if role == 'customer':
                        self.assertEqual({r['session_id'] for r in rows},
                                         {3, 4} if split == 'train' else {7, 8})
                    else:
                        required = {'seller:1:2:1', 'seller:3:4:3'} if split == 'train' else {
                            'seller:1:2:5', 'seller:2:3:7'}
                        self.assertLessEqual(required, {r['record_id'] for r in rows})
            self.assertEqual(result['supply']['splits'], {
                'train': {'requests': 2, 'exposures': 3, 'products': 2},
                'test': {'requests': 2, 'exposures': 3, 'products': 2},
            })

    def test_rounding_and_explicit_seed_config(self):
        self.assertEqual([fraction_count(n,.02) for n in (0,1,24,25,100,485111)], [0,0,0,1,2,9702])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json'
            good={'schema':'sft-split-sampling-config-v1','seed':42,
                  'fractions':dict(train=.02,test=.02),
                  'strata':dict(customer=['action','history_band'],seller=['action','category_level1_id'])}
            p.write_text(json.dumps(good));self.assertEqual(read_config(p),good)
            for change in ({'seed':None},{'fractions':{'train':.02}},
                           {'fractions':dict(train=.02,validation=.02,test=.02)},
                           {'fractions':dict(train=0,test=.02)}):
                p.write_text(json.dumps({**good,**change}))
                with self.assertRaises(ValueError):read_config(p)

    def test_test_history_counts_all_prior_but_train_excludes_official_test(self):
        with tempfile.TemporaryDirectory() as tmp,duckdb.connect() as c:
            root=Path(tmp)
            c.execute("CREATE TABLE requests AS SELECT * FROM (VALUES (1,7,1,'train'),(2,7,2,'test'),(3,7,3,'train'),(4,7,3,'test'),(5,7,4,'test')) x(session_id,user_id,time_index,split)")
            labels=root/'labels.parquet'
            c.execute(f"COPY (SELECT session_id,user_id,time_index history_cutoff_exclusive,[]::INT[] clicked_item_ids,[]::INT[] purchased_item_ids FROM requests) TO {literal(labels)} (FORMAT PARQUET)")
            customer_features(c,{'test':labels},root)
            actual=c.execute(f"SELECT session_id,prior_points FROM read_parquet({literal(root/'customer-00.parquet')}) ORDER BY 1").fetchall()
            self.assertEqual(actual,[(1,0),(2,1),(3,1),(4,2),(5,4)])

    def test_selection_reproducible_across_source_order_and_seed_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp,duckdb.connect() as c:
            root=Path(tmp);(root/'features').mkdir()
            feature=root/'features/customer-00.parquet'
            def write(order):
                c.execute(f"COPY (SELECT 'test' sampling_split,CAST(i AS VARCHAR) record_id,i%7 actor_id,i history_cutoff_exclusive,if(i%3=0,'click','none') AS action,if(i%2=0,'cold','recent') history_band FROM range(1,1001) t(i) ORDER BY i {order}) TO {literal(feature)} (FORMAT PARQUET)")
            def choose(seed,count=20):
                sizes = prepare_population(c, root/'features', 'customer', 'test', ('action', 'history_band'), 1000)
                path = root/'customer-test-candidates.parquet'
                allocation = quotas(sizes, count) if count else {}
                result = select_by_stratum(c, path, sizes, allocation, count, seed)
                ids=c.execute('SELECT record_id FROM read_parquet(?) ORDER BY record_id',[str(path)]).fetchall()
                path.unlink();return ids,result
            write('ASC');first,stats=choose(42)
            feature.unlink();write('DESC');second,_=choose(42);third,_=choose(43)
            self.assertEqual(first,second);self.assertNotEqual(first,third)
            self.assertEqual(sum(stats['selected_by_stratum'].values()),20)
            self.assertEqual(choose(42,0)[0],[])

    def test_all_splits_cache_audit_and_training_identity(self):
        with tempfile.TemporaryDirectory() as tmp,duckdb.connect() as c:
            root=Path(tmp);dataset=root/'dataset';dataset.mkdir()
            (dataset/'manifest.json').write_text(json.dumps({'counts':{
                role:{s:1 for s in ('train','test')} for role in ('customer','seller')}}))
            selection={'schema':'sft-split-candidates-v2','seed':42,
                'dataset_manifest_sha256':sha256(dataset/'manifest.json'),
                'roles':{'customer':{}}}
            for split,record in [('train','1'),('test','2')]:
                path=root/f'customer-{split}-candidates.parquet'
                c.execute(f"COPY (SELECT '{record}' record_id,7 actor_id,'none:cold' stratum) TO {literal(path)} (FORMAT PARQUET)")
                selection['roles']['customer'][split]={'file':path.name,'sha256':sha256(path),
                    'available_by_stratum':{'none:cold':50},'selected_by_stratum':{'none:cold':1}}
                base=root/'customer'/split/'000';base.mkdir(parents=True)
                shard=base/'000000.npz'
                write_shard(shard,[PreparedRow(f'{stage}:{record}',np.array([1,2,3],dtype=np.int32),2,1.) for stage in ('query','response')])
                choices=base/'selection.json';choices.write_text(json.dumps(dict(accepted=[record],excluded=[],candidate_records=1)))
                (base/'complete.json').write_text(json.dumps(dict(rows=2,records=1,candidate_records=1,excluded_records=0,
                    selection_sha256=sha256(choices),files=[dict(file=shard.name,sha256=sha256(shard),rows=2)])))
            manifest=root/'candidate_manifest.json';manifest.write_text(json.dumps(selection))
            spec=selected_identity(dataset,selection,sha256(manifest),42,1)
            for name in ('identity','complete'):(root/f'{name}.json').write_text(json.dumps(spec))
            self.assertEqual(validate_release(root,dataset,42),spec)
            with self.assertRaisesRegex(ValueError,'seed differs'):validate_release(root,dataset,43)
            args=SimpleNamespace(cache=root,dataset=dataset,candidates=root,tokenizer=root,role='customer')
            from agent_simulation.training.run import check_cached_data
            args.token_cache=root;args.seed=42;args.validation_samples=128
            data=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
            checked = check_cached_data(args, data, [], verify=True)
            self.assertTrue(checked['test_used'])
            self.assertEqual(checked['counts'],{'train':2,'test':2})
            self.assertEqual({r['sample_id'] for r in checked['checks']},
                             {'query:1','response:1','query:2','response:2'})

    def test_selected_eval_build_uses_eval_split_and_records_whole_exclusions(self):
        from agent_simulation.inference.requests import ContextCapacityError
        for split in ('test',):
            with self.subTest(split=split),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);args=SimpleNamespace(output=root,dataset=root,tokenizer=root,candidates=root/'manifest.json',buckets=1,reserve_gib=0)
                data=SimpleNamespace(catalog=SimpleNamespace(connection=SimpleNamespace(execute=lambda *a:None)),tokenizer=None,close=lambda:None)
                raw=[({'key':{'session_id':i}}, {}) for i in (1,2)]
                row=PreparedRow('query:2',np.array([1,2,3],dtype=np.int32),2,1.)
                with patch('agent_simulation.datasets.tokens.SFTDataset',return_value=data),patch('agent_simulation.datasets.tokens.bind_split_candidates',return_value=[('1',split),('2',split)]),patch('agent_simulation.datasets.tokens.customer_records',return_value=iter(raw)) as source,patch('agent_simulation.datasets.tokens.expand_record',side_effect=[[{'split':split}], [{'split':split}]]),patch('agent_simulation.datasets.tokens.PreparedRow.from_sample',side_effect=[ContextCapacityError('too long'),row]):
                    result=build_splits(args, 'customer', 0, (split,))[split]
                self.assertFalse(source.call_args.kwargs['include_history_ids'])
                self.assertEqual((result['records'],result['excluded_records']),(1,1))
                self.assertTrue((root/'customer'/split/'000/selection.json').exists())


if __name__=='__main__':unittest.main()
