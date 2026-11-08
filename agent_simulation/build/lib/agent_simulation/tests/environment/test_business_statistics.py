"""Window totals, paired conversion labels, missing prices and tool permissions."""
from copy import deepcopy
import json
import unittest

from agent_simulation.tests.fixtures import MarketFixture
from agent_simulation.data.market.statistics import METRICS


class BusinessStatisticsTests(MarketFixture, unittest.TestCase):
    def query(self, **overrides):
        args = dict(item_ids=[1], before_time_index=10, limit=1, metrics=list(METRICS))
        args.update(overrides)
        return self.runtime.execute('seller', 10, 'query_business', args)

    def paired_fixture(self):
        records = deepcopy(self.data.records(1))
        records[0].update(exposure_records=2, click_records=1, purchase_records=1,
                          price_raw=None, price_status='conflicting_prices')
        self.data._histories[1] = records
        self.c.execute('''CREATE TABLE ranking_observations AS SELECT * FROM (VALUES
            (1,2,NULL::DOUBLE,0,1), (1,2,100.0,1,0), (1,4,120.0,1,1), (1,10,999.0,1,1)
            ) t(item_id,time_index,price_raw,is_clicked,is_purchased)''')

    def test_full_window_independent_of_detail_limit_and_future_target(self):
        self.paired_fixture()
        result = self.query()
        self.assertEqual(len(result['series']), 1)
        self.assertEqual(result['summary_of_returned_series'][0]['estimated_units_sold'], 1)
        item = result['statistics']['items'][0]
        self.assertEqual((item['exposure_records'], item['click_records'], item['estimated_units_sold']), (3, 2, 2))
        metrics = item['metrics']
        self.assertEqual(metrics['purchase_rate']['value'], 2 / 3)
        self.assertEqual(metrics['click_through_rate']['value'], 2 / 3)
        # Purchase-without-click must not inflate the click-conditioned conversion rate.
        self.assertEqual(metrics['click_to_purchase_rate']['value'], 1 / 2)
        self.assertEqual(metrics['estimated_sales_amount_raw'], {
            'value': 120.0, 'price_covered_purchase_records': 1, 'purchase_records': 2, 'status': 'partial'})
        larger = self.query(limit=100)
        self.assertEqual(larger['statistics'], result['statistics'])
        empty = self.query(start_time_index=5, before_time_index=9)['statistics']['items'][0]
        self.assertIsNone(empty['metrics']['purchase_rate']['value'])
        self.assertEqual(empty['metrics']['estimated_sales_amount_raw']['value'], 0)

    def test_no_fabricated_joint_counts_when_only_aggregates_exist(self):
        item = self.query()['statistics']['items'][0]
        self.assertEqual(item['metrics']['click_to_purchase_rate']['status'], 'unavailable_joint_labels')
        self.assertIsNone(item['metrics']['click_to_purchase_rate']['value'])
        self.assertEqual(item['metrics']['estimated_sales_amount_raw']['value'], 120)

    def test_statistics_do_not_change_state_and_obey_permissions(self):
        self.paired_fixture()
        before = self.runtime.snapshot()
        self.query()
        self.assertEqual(self.runtime.snapshot(), before)
        with self.assertRaises(PermissionError):
            self.query(item_ids=[2])
        with self.assertRaises(ValueError):
            self.query(before_time_index=11)
        with self.assertRaises(ValueError):
            self.query(metrics=['profit'])
        with self.assertRaises(ValueError):
            self.query(metrics=['purchase_rate', 'purchase_rate'])
        with self.assertRaises(ValueError):
            self.query(start_simulation_step=2, before_simulation_step=1)

    def test_simulation_rates_use_settled_pairs_and_original_displayed_price(self):
        r = self.runtime
        r.execute('customer', 7, 'search_products', {'query': '保温杯'})
        r.execute('seller', 10, 'submit_prices', {'prices': [{'item_id': 1, 'price_raw': 90}]})
        r.execute('customer', 7, 'submit_response', {'clicked_item_ids': [], 'purchased_item_ids': [1]})
        cutoff = r.snapshot()['simulation_step'] + 1
        r.execute('customer', 8, 'search_products', {'query': '保温杯'})
        r.execute('customer', 8, 'submit_response', {'clicked_item_ids': [1], 'purchased_item_ids': [1]})
        result = self.query(start_simulation_step=2)
        self.assertEqual(len(result['simulation']['settled_series']), 1)
        item = result['simulation']['statistics']['items'][0]
        self.assertEqual((item['exposure_records'], item['estimated_units_sold']), (2, 2))
        self.assertEqual(item['metrics']['click_to_purchase_rate']['value'], 1)
        self.assertEqual(item['metrics']['purchase_rate']['value'], 1)
        self.assertEqual(item['metrics']['estimated_sales_amount_raw']['value'], 210)
        past = self.query(start_simulation_step=2, before_simulation_step=cutoff)
        past_item = past['simulation']['statistics']['items'][0]
        self.assertEqual(past_item['metrics']['estimated_sales_amount_raw']['value'], 120)
        self.assertIsNone(past_item['metrics']['click_to_purchase_rate']['value'])
        # A third pending exposure is not an observed zero response in this settled cohort.
        r.execute('customer', 7, 'search_products', {'query': '保温杯'})
        self.assertEqual(self.query(start_simulation_step=2)['simulation']['statistics']['items'], [item])

    def test_legacy_unknown_displayed_price_does_not_become_zero_revenue(self):
        state = self.runtime.snapshot()
        state["active_prices"]["2"]["price_raw"] = None
        with self.runtime.db:
            self.runtime.db.execute("UPDATE state SET value=?", (json.dumps(state),))
        self.runtime.execute('customer', 7, 'search_products', {'query': '保温杯'})
        self.runtime.execute('customer', 7, 'submit_response', {'clicked_item_ids': [], 'purchased_item_ids': [2]})
        result = self.runtime.execute('seller', 20, 'query_business', dict(
            item_ids=[2], before_time_index=10, limit=1, metrics=['estimated_sales_amount_raw']))
        amount = result['simulation']['statistics']['items'][0]['metrics']['estimated_sales_amount_raw']
        self.assertEqual(amount['status'], 'no_valid_price')
        self.assertIsNone(amount['value'])


if __name__ == '__main__':
    unittest.main()
