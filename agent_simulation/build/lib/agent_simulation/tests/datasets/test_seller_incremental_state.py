"""Target rendering and full-prefix histories must equal the full-shop reader."""
from copy import deepcopy
from unittest.mock import patch
import unittest

from agent_simulation.tests.fixtures import MarketFixture
from agent_simulation.datasets.samples.seller import SellerHistory, with_market_context, sample_from_context
from agent_simulation.agents.context.pricing import build_pricing_context, overview_context
from agent_simulation.agents.context.pricing import target_context
from agent_simulation.agents.retrieval.history import seller_history
from types import SimpleNamespace


class IncrementalSellerTests(MarketFixture, unittest.TestCase):
    def test_sparse_cohorts_invalid_prices_and_late_eligibility_keep_full_inputs(self):
        base = self.data.records(1)[0]
        stream, past = SellerHistory(10, 'fixture'), []
        stream.limit = 3
        for time in range(1, 24):
            cohort = []
            for item in range(1, 13):
                if item % 3 == 0 and time % 2:
                    continue
                valid = time % 7 != 0 and (item != 12 or time >= 10)
                cohort.append(dict(base, item_id=item, time_index=time,
                    item_title=f'商品{item}版本{time//9}', seller_name=f'店名{time}',
                    price_raw=100+item+time if valid else None,
                    price_status='observed' if valid else 'invalid_price',
                    exposure_records=10, click_records=time % 3, purchase_records=time % 2,
                    assigned_split='test' if time > 2 and valid and item in (1, 2) else None))
            with patch('agent_simulation.datasets.samples.seller.build_pricing_context',
                       wraps=build_pricing_context) as render:
                actual = list(stream.samples(cohort))
                if actual:
                    self.assertEqual({r['item_id'] for r in render.call_args.args[0]}, {1, 2})
                    self.assertLessEqual(len(render.call_args.args[0]), 6)
            eligible = {r['item_id'] for r in past if r['price_status'] == 'observed'}
            for sample in actual:
                item = sample['source']['item_id']
                target = next(r for r in cohort if r['item_id'] == item)
                state, coverage = build_pricing_context([r for r in past if r['item_id'] in eligible],
                    owner_id=10, cutoff=time, observations_per_item=3, public_pages=[], requested_item_ids=[item])
                expected = sample_from_context('fixture', target,
                    target_context(overview_context(state, [item]), [item]))
                self.assertEqual(sample, expected)
            if len(actual) == 2:
                saved = deepcopy(actual[1]['input']['owned_products'])
                actual[0]['input']['owned_products'][0]['observed_price_history']['all_observations']['observations'] = -1
                self.assertEqual(actual[1]['input']['owned_products'], saved)
            past.extend(cohort)

    def test_retrieved_owned_profiles_do_not_query_product_history_again(self):
        stream = SellerHistory(10, 'fixture')
        prior = [dict(self.data.records(item)[0], time_index=1) for item in (1, 3)]
        self.assertEqual(list(stream.samples(prior)), [])
        target = dict(prior[0], time_index=9, assigned_split='train')
        # Keep the generator suspended: no same-cohort information may be consumed.
        pending = stream.samples([target])
        sample = next(pending)
        history = seller_history(prior, seller_id=10, cutoff=9, recent_n=10,
                                 focus_ids=[1], embedder=None)
        self.assertEqual([s['item_id'] for s in history['segments']], [3])
        with patch('agent_simulation.agents.context.market.load_market_context'), \
             patch('agent_simulation.data.catalog.DataRepository.product', side_effect=AssertionError('duplicate SQL')):
            result = with_market_context(SimpleNamespace(connection=self.c, config=self.cfg), sample,
                retrieved_history=history, product_lookup=stream.product_profile, history_embedder=None)
        details = {p['item_id']:p for p in result['input']['product_details']}
        self.assertEqual(details[3]['item_title'], prior[1]['item_title'])
        details[3]['item_title'] = 'caller mutation'
        self.assertEqual(stream.product_profile(3)['item_title'], prior[1]['item_title'])
        self.assertEqual(list(pending), [])
