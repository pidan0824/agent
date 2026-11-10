"""Merged split replay preserves cutoff filters, ordering and independent output commits."""
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import unittest
import duckdb
import numpy as np
from agent_simulation.data.common import literal
from agent_simulation.datasets.tokens import bind_split_candidates, build_splits
from agent_simulation.datasets.tokens import PreparedRow
from agent_simulation.inference.requests import ContextCapacityError


class CombinedCandidateTests(unittest.TestCase):
    def test_customer_union_keeps_full_history_and_only_selected_labels(self):
        with TemporaryDirectory() as directory, duckdb.connect() as c:
            c.execute("CREATE TABLE requests AS SELECT 7 user_id,i session_id,i time_index FROM range(1,9) t(i)")
            c.execute("CREATE TABLE customer_splits AS SELECT session_id, CASE WHEN session_id<4 THEN 'train' WHEN session_id<6 THEN 'validation' ELSE 'test' END assigned_split FROM requests")
            paths = {}
            for split, i in [('train',2),('validation',4),('test',6)]:
                path = Path(directory)/f'{split}.parquet'
                c.execute(f"COPY (SELECT 7 actor_id,? session_id,? history_cutoff_exclusive,? record_id) TO {literal(path)} (FORMAT PARQUET)", [i,i,str(i)])
                paths[split] = path
            self.assertEqual(bind_split_candidates(c,paths,'customer',0,1), [('2','train'),('4','validation'),('6','test')])
            self.assertEqual(c.execute('SELECT session_id FROM requests ORDER BY 1').fetchall(), [(i,) for i in range(1,7)])
            self.assertEqual(c.execute('SELECT session_id,assigned_split FROM customer_splits WHERE assigned_split IS NOT NULL ORDER BY 1').fetchall(), [(2,'train'),(4,'validation'),(6,'test')])

    def test_one_replay_writes_all_splits_and_capacity_failure_stays_in_its_split(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output=root/'out', candidates=root/'manifest.json', dataset=root/'data', tokenizer=root/'tok', buckets=1, reserve_gib=1)
            data = Mock(catalog=SimpleNamespace(connection=Mock()), tokenizer=object())
            splits = ('train','validation','test')
            expected = [(str(i), split) for i,split in enumerate(splits,1)]
            raw = [(dict(key={'session_id':i}),{}) for i in (1,2,3)]
            expanded = [[{'split': split}] for split in splits]
            rows = [PreparedRow('query:1', np.array([1,2,3],dtype=np.int32),2,1.),
                    ContextCapacityError('too long'),
                    PreparedRow('query:3', np.array([1,2,3],dtype=np.int32),2,1.)]
            with patch('agent_simulation.datasets.tokens.SFTDataset',return_value=data), \
                 patch('agent_simulation.datasets.tokens.bind_split_candidates',return_value=expected), \
                 patch('agent_simulation.datasets.tokens.customer_records',return_value=iter(raw)) as source, \
                 patch('agent_simulation.datasets.tokens.expand_record',side_effect=expanded), \
                 patch('agent_simulation.datasets.tokens.PreparedRow.from_sample',side_effect=rows):
                results = build_splits(args, 'customer', 0, splits)
            source.assert_called_once_with(data.catalog, 0, 1, include_history_ids=False)
            self.assertEqual([results[s]['records'] for s in splits],[1,0,1])
            self.assertEqual(results['validation']['excluded_records'],1)
            self.assertTrue((args.output/'customer/validation/000/failures.jsonl').exists())
            with patch('agent_simulation.datasets.tokens.SFTDataset',side_effect=AssertionError('already complete')):
                self.assertEqual(build_splits(args, 'customer', 0, splits),results)

    def test_seller_context_overflow_is_recorded_before_encoding_and_replay_continues(self):
        import json
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output=root/'out', candidates=root/'manifest.json',
                                   dataset=root/'data', tokenizer=root/'tok', buckets=1, reserve_gib=1)
            data = Mock(catalog=SimpleNamespace(connection=Mock()), tokenizer=object())
            expected = [('seller:1:1:2', 'train'), ('seller:1:1:3', 'test'),
                        ('seller:1:1:4', 'test')]

            def source(*args, on_capacity_error, **kwargs):
                on_capacity_error(*expected[0], ContextCapacityError('first too long'))
                yield dict(sample_id=expected[1][0], split='test')
                on_capacity_error(*expected[2], ContextCapacityError('last too long'))

            with patch('agent_simulation.datasets.tokens.SFTDataset', return_value=data), \
                 patch('agent_simulation.datasets.tokens.bind_split_candidates', return_value=expected), \
                 patch('agent_simulation.datasets.tokens.seller_records', side_effect=source), \
                 patch('agent_simulation.datasets.tokens.expand_record', side_effect=lambda role, raw, **kw: [raw]), \
                 patch('agent_simulation.datasets.tokens.PreparedRow.from_sample', return_value=
                       PreparedRow(expected[1][0], np.array([1,2,3], dtype=np.int32), 2, 1.)):
                result = build_splits(args, 'seller', 0, ('train', 'test'))
            self.assertEqual((result['train']['records'], result['test']['records']), (0, 1))
            self.assertEqual((result['train']['excluded_records'], result['test']['excluded_records']), (1, 1))
            for record, split in (expected[0], expected[2]):
                failure = json.loads((args.output/f'seller/{split}/000/failures.jsonl').read_text())
                self.assertEqual(failure['record_id'], record)
                self.assertEqual(failure['reason'], 'context_capacity')
