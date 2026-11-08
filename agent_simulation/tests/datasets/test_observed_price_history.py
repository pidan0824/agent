"""Full-prefix price evidence must preserve unknowns, time boundaries and counts."""
from copy import deepcopy
import math
import random
import unittest

from agent_simulation.data.history_records import ObservedPriceHistory
from agent_simulation.datasets.samples.seller import SellerHistory
from agent_simulation.tests.fixtures import MarketFixture


class PriceHistoryTests(unittest.TestCase):
    def test_incremental_prefix_matches_independent_full_scan(self):
        rng = random.Random(42)
        history = ObservedPriceHistory(7, 3)
        rows = []
        for index in range(1, 151):
            price = rng.choice([None, -1, float('nan'), 50, 60, 70, 80, 90, 100])
            row = dict(seller_id=7,item_id=3,time_index=index*3,price_raw=price,
                price_status='observed',exposure_records=rng.randrange(1,4),
                click_records=rng.randrange(2),purchase_records=rng.randrange(2))
            history.append(row)
            rows.append(row)
            got = history.view(row['time_index']+1)
            valid = [r for r in rows if r['price_raw'] is not None and math.isfinite(r['price_raw']) and r['price_raw'] > 0]
            distinct = set(r['price_raw'] for r in valid)
            self.assertEqual(got['distinct_valid_prices'],len(distinct))
            self.assertEqual(got['unknown_price_observations'],len(rows)-len(valid))
            self.assertEqual(got['valid_quote_changes'],sum(a['price_raw']!=b['price_raw'] for a,b in zip(valid,valid[1:])))
            for key in ('exposure_records','click_records','purchase_records'):
                self.assertEqual(got['all_observations'][key],sum(r[key] for r in rows))
            for level in got['price_level_totals']:
                matching = [r for r in valid if r['price_raw']==level['price_raw']]
                self.assertEqual(level['observations'],len(matching))
                self.assertEqual(level['first_index'],matching[0]['time_index'])
                for key in ('exposure_records','click_records','purchase_records'):
                    self.assertEqual(level[key],sum(r[key] for r in matching))
            expected_runs=[]
            for r in rows:
                p=r['price_raw']
                p=p if p is not None and math.isfinite(p) and p>0 else None
                if not expected_runs or expected_runs[-1][0]!=p:
                    expected_runs.append([p,r['time_index'],r['time_index'],1])
                else:
                    expected_runs[-1][2]=r['time_index'];expected_runs[-1][3]+=1
            actual=[[r['price_raw'],r['first_index'],r['last_index'],r['observations']] for r in got['recent_observation_runs']]
            self.assertEqual(actual,expected_runs[-4:])
            self.assertEqual(got['omitted_earlier_runs'],max(0,len(expected_runs)-4))

    def test_unknown_breaks_run_and_views_are_immutable(self):
        history=ObservedPriceHistory(7,3)
        for t,p in enumerate([69,None,69,79,69],1):
            history.append(dict(seller_id=7,item_id=3,time_index=t,price_raw=p,
                price_status='observed',exposure_records=1,click_records=0,purchase_records=0))
        view=history.view(6)
        self.assertEqual(view['valid_quote_changes'],2)
        self.assertEqual(view['price_level_totals'][0]['observations'],3)
        self.assertEqual(view['omitted_earlier_runs'],1)
        view['all_observations']['exposure_records']=-1
        self.assertEqual(history.view(6)['all_observations']['exposure_records'],5)
        with self.assertRaisesRegex(ValueError,'Future'):
            history.view(5)


class StreamingPriceBoundaryTests(MarketFixture,unittest.TestCase):
    def test_target_and_same_time_other_product_are_never_counted(self):
        base=self.data.records(1)[0]
        state=SellerHistory(10,'fixture')
        for time in range(1,21):
            list(state.samples([dict(base,time_index=time,price_raw=69 if time<5 else 79)]))
        target=dict(base,time_index=21,price_raw=999999,assigned_split='train')
        pending=state.samples([target,dict(base,item_id=3,time_index=21,price_raw=888888)])
        sample=next(pending)
        value=sample['input']['owned_products'][0]['observed_price_history']
        self.assertEqual(value['all_observations']['observations'],20)
        self.assertEqual(value['max_price_raw'],79)
        self.assertEqual(value['price_level_totals'][1]['price_raw'],69)
        self.assertEqual(value['price_level_totals'][1]['observations'],4)
        saved=deepcopy(value)
        list(pending)
        self.assertEqual(value,saved)
        self.assertEqual(state.price_histories[1].view(22)['max_price_raw'],999999)
