"""Candidate selection preserves temporal evidence and source proportions."""
import tempfile
from pathlib import Path
import unittest

import duckdb
from agent_simulation.datasets.sampling import (
    customer_features, prepare_population, quotas, select_by_stratum, seller_features,
)
from agent_simulation.data.common import literal


class CandidateSelectionTests(unittest.TestCase):
    def test_quotas_exact_and_bounded_for_rare_strata(self):
        for counts in ({'a': 1, 'b': 1000, 'c': 3}, {'a': 20, 'b': 20}):
            for budget in (1, 4, 30, sum(counts.values())):
                result = quotas(counts, budget)
                self.assertEqual(sum(result.values()), budget)
                self.assertTrue(all(0 <= result[k] <= counts[k] for k in counts))
                self.assertTrue(all(abs(result[k]-budget*counts[k]/sum(counts.values()))<1 for k in counts))
                self.assertEqual(result, quotas(dict(reversed(list(counts.items()))), budget))
        with self.assertRaises(ValueError):
            quotas({'a': 1}, 2)

    def test_customer_history_excludes_equal_time_and_official_test(self):
        with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as c:
            root = Path(tmp)
            c.execute("CREATE TABLE requests AS SELECT * FROM (VALUES (1,7,1,'train'),(2,7,2,'test'),(3,7,3,'train'),(4,7,3,'train'),(5,7,4,'train')) x(session_id,user_id,time_index,split)")
            labels = root/'labels.parquet'
            c.execute(f"COPY (SELECT session_id,user_id,time_index history_cutoff_exclusive,[]::INT[] clicked_item_ids,[]::INT[] purchased_item_ids FROM requests WHERE session_id>=3) TO {literal(labels)} (FORMAT PARQUET)")
            customer_features(c,{'train':labels},root)
            actual = c.execute(f"SELECT session_id,prior_points FROM read_parquet({literal(root/'customer-00.parquet')}) ORDER BY 1").fetchall()
            self.assertEqual(actual,[(3,1),(4,1),(5,3)])

    def test_seller_features_use_strictly_earlier_valid_price(self):
        with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as c:
            root = Path(tmp)
            c.execute("CREATE TABLE seller_history AS SELECT 1 seller_id,2 item_id,3 category_level1_id,* FROM (VALUES (1,100.,'observed'),(2,NULL,'invalid'),(3,150.,'observed'),(4,10.,'observed')) x(time_index,price_raw,price_status)")
            labels=root/'labels.parquet'
            c.execute(f"COPY (SELECT 'seller:1:2:3' sample_id,1 seller_id,2 item_id,3 history_cutoff_exclusive,150. target_price_raw) TO {literal(labels)} (FORMAT PARQUET)")
            seller_features(c,{'train':labels},root)
            actual=c.execute(f"SELECT prior_points,prior_price,action FROM read_parquet({literal(root/'seller-*.parquet')})").fetchall()
            self.assertEqual(actual,[(2,100.,'up')])

    def test_uniform_seeded_order_does_not_prioritize_actor_coverage(self):
        with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as c:
            root=Path(tmp);(root/'features').mkdir()
            c.execute(f"COPY (SELECT 'train' sampling_split,CAST(i AS VARCHAR) record_id,if(i<9,1,i) actor_id,i history_cutoff_exclusive,100.0 prior_price,'same' AS action,'established' AS history_band FROM range(1,12) t(i)) TO {literal(root/'features/seller-00.parquet')} (FORMAT PARQUET)")
            path = root / 'seller-train-candidates.parquet'

            def choose():
                sizes = prepare_population(c, root/'features', 'seller', 'train', ('action', 'history_band'), 11)
                select_by_stratum(c, path, sizes, quotas(sizes, 4), 4, 42)

            choose()
            first=c.execute(f"SELECT record_id FROM read_parquet({literal(path)})").fetchall()
            expected=c.execute(f"SELECT record_id FROM read_parquet({literal(root/'features/seller-00.parquet')}) ORDER BY sha256(record_id||':42'),record_id LIMIT 4").fetchall()
            self.assertEqual(set(first),set(expected))
            path.unlink()
            choose()
            second=c.execute(f"SELECT record_id FROM read_parquet({literal(path)})").fetchall()
            self.assertEqual(first,second)


if __name__=='__main__':
    unittest.main()
