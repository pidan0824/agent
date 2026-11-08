"""Target-only input, causal retrieval, shared queries and atomic pricing reviews."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from agent_simulation.tests.fixtures import MarketFixture, ScriptedModel, TokenizerFixture, answer
from agent_simulation.agents.seller import SellerAgent
from agent_simulation.agents.context.pricing import PricingSnapshot
from agent_simulation.agents.context.market import load_market_context
from agent_simulation.agents.prompts.seller import decode_input
from agent_simulation.agents.retrieval.history import prepare_seller_history, seller_history, CachedHistoryEncoder
from agent_simulation.agents.retrieval.similarity import cosine_scores
from agent_simulation.simulation.episode import create_clock
from agent_simulation.data.market import quotes as market_queries


class TargetPricingTests(MarketFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.runtime.inputs.history_embedder = None

    def test_target_view_preserves_shop_period_and_does_not_mutate_snapshot(self):
        full, _ = self.runtime.inputs.seller(10, retrieve=False, observe=False, include_market=False)
        before = deepcopy(full)
        snapshot = PricingSnapshot(full)
        a, b = snapshot.view([1]), snapshot.view([3])
        self.assertEqual(a['requested_price_item_ids'], [1])
        self.assertEqual([p['item_id'] for p in a['owned_products']], [1])
        self.assertNotIn('shop_summary', a)
        self.assertEqual(a['shop_scope'], b['shop_scope'])
        self.assertEqual(set(a['simulation_state']['active_prices']), {'1'})
        self.assertTrue(all(r['item_id']==1 for e in a['short_term_memory'] for r in e['content']))
        a['owned_products'][0]['last_observed_price_raw'] = 900
        a['shop_scope']['recent_and_retrieved_history'] = 'changed'
        self.assertEqual(full, before)
        self.assertEqual(snapshot.view([1])['shop_scope'], full['shop_scope'])
        for invalid in ([2], [1,1], []):
            with self.assertRaises(ValueError):snapshot.view(invalid)

    def test_target_must_have_detail_in_the_prepared_overview(self):
        full, _ = self.runtime.inputs.seller(10, detail_item_ids=[1], retrieve=False,
                                            observe=False, include_market=False)
        snapshot = PricingSnapshot(full)
        self.assertEqual(snapshot.view([1])['history_presentation']['detail_item_ids'], [1])
        with self.assertRaisesRegex(ValueError, 'lacks prepared recent detail'):
            snapshot.view([3])

    def test_default_single_target_uses_frozen_prices_and_no_whole_shop_partition(self):
        baseline = self.runtime.snapshot()
        def quote(item, price):
            def respond(request, call_id):
                self.assertEqual(self.runtime.snapshot(), baseline)
                ctx = decode_input(request['messages'][1]['content'])
                self.assertEqual(ctx['decision']['requested_item_ids'], [item])
                self.assertEqual([p['item_id'] for p in ctx['current_state']['products']], [item])
                self.assertEqual(ctx['current_state']['products'][0]['active_quote']['price_raw'], baseline['active_prices'][str(item)]['price_raw'])
                self.assertEqual(request['action_output']['final_schemas'][0]['properties']['prices']['items']['properties']['item_id']['enum'], [item])
                self.assertEqual(request['action_output']['final_schemas'][0]['properties']['prices']['maxItems'], 1)
                return answer({'prices':[{'item_id':item,'price_raw':price}]})
            return respond
        model = ScriptedModel([quote(1,111),quote(3,222)])
        from agent_simulation.agents.retrieval import history
        with patch.object(history,'prepare_seller_history',wraps=history.prepare_seller_history) as index, \
             patch('agent_simulation.agents.context.pricing.PricingSnapshot', wraps=PricingSnapshot) as snapshot:
            run = SellerAgent(self.runtime,model,TokenizerFixture()).run(10)
        self.assertEqual(index.call_count,1)
        snapshot.assert_called_once()
        self.assertEqual(run['result']['reviewed_products'],2)
        self.assertNotIn('context_provenance', run)
        state = self.runtime.snapshot()
        self.assertEqual(state['simulation_step'],baseline['simulation_step']+1)
        self.assertEqual([r['price_raw'] for r in state['price_actions']],[111,222])
        self.assertEqual(len({r['simulation_step'] for r in state['price_actions']}),1)

    def test_later_error_discards_all_target_drafts(self):
        before = self.runtime.snapshot()
        model = ScriptedModel([answer({'prices':[{'item_id':1,'price_raw':111}]}),
                               answer({'prices':[{'item_id':1,'price_raw':999}]})])
        with self.assertRaises(ValueError):SellerAgent(self.runtime,model,TokenizerFixture()).run(10)
        self.assertEqual(self.runtime.snapshot(),before)

    def test_episode_default_targets_publish_all_sellers_before_customers(self):
        from agent_simulation.tests.fixtures import ProfileCatalogFixture
        from agent_simulation.agents.customer import CustomerAgent
        from agent_simulation.simulation.episode import Episode
        self.runtime.catalog = ProfileCatalogFixture(self.runtime)
        initial = self.runtime.snapshot()['active_prices']
        reviewed = []
        def quote(request, call_id):
            self.assertEqual(self.runtime.snapshot()['active_prices'], initial)
            context = decode_input(request['messages'][1]['content'])
            item, = context['decision']['requested_item_ids']
            reviewed.append(item)
            return answer({'prices':[{'item_id':item,'price_raw':1000+item}]})
        def wait(request, call_id):
            self.assertEqual(reviewed,[1,3,2,4])
            self.assertEqual({k:v['price_raw'] for k,v in self.runtime.snapshot()['active_prices'].items()},
                             {str(i):1000+i for i in (1,2,3,4)})
            return answer({'action':'no_action'})
        seller = SellerAgent(self.runtime,ScriptedModel([quote]*4),TokenizerFixture())
        customer = CustomerAgent(self.runtime,ScriptedModel([wait]*2),TokenizerFixture())
        episode = Episode(self.runtime,seller,customer,tick_size=1,max_ticks=1)
        self.assertEqual(episode.run()['status'],'complete')

    def test_next_snapshot_reuses_history_but_reads_new_active_prices(self):
        before = list(self.runtime.inputs.seller_targets(10,state=self.runtime.snapshot()))
        self.runtime.execute('seller',10,'submit_prices',{'prices':[{'item_id':1,'price_raw':777}]})
        after = list(self.runtime.inputs.seller_targets(10,state=self.runtime.snapshot()))
        self.assertEqual(before[0]['simulation_state']['active_prices']['1']['price_raw'],120)
        self.assertEqual(after[0]['simulation_state']['active_prices']['1']['price_raw'],777)
        self.assertEqual(before[0]['owned_products'][0]['last_observed_price_raw'],
                         after[0]['owned_products'][0]['last_observed_price_raw'])

    def test_partial_market_declares_scope_without_claiming_full_shop_history(self):
        from agent_simulation.environment.market import MarketEnvironment
        runtime = MarketEnvironment.create(self.path.with_name("subset.sqlite"), self.data, [1], {})
        self.addCleanup(runtime.close)
        runtime.inputs.history_embedder = None
        context, = list(runtime.inputs.seller_targets(10, state=runtime.snapshot()))
        self.assertEqual(context['requested_price_item_ids'], [1])
        self.assertNotIn('shop_summary', context)
        self.assertEqual(context['observed_shop_period']['products_observed_before_cutoff'], 2)
        self.assertIn('Registered products only', context['shop_scope']['recent_and_retrieved_history'])
        self.assertIn('outside this simulation', context['shop_scope']['observed_shop_period'])
        self.assertTrue(all(s['item_id'] == 1 for s in context['retrieved_context']['history']['segments']))

    def test_unknown_price_can_remain_unknown_in_default_target_mode(self):
        state = self.runtime.snapshot()
        state['active_prices']['3']['price_raw'] = None
        self.runtime.db.execute('UPDATE state SET value=?',(json.dumps(state),))
        self.runtime.db.commit()
        model = ScriptedModel([answer({'prices':[{'item_id':1,'price_raw':120}]}),
                               answer({'prices':[{'item_id':3,'price_raw':None}]})])
        result=SellerAgent(self.runtime,model,TokenizerFixture()).run(10)
        self.assertEqual(result['result']['reviewed_products'],2)
        self.assertEqual(self.runtime.snapshot(),state)

    def test_first_target_wait_is_whole_shop_wait_and_later_wait_is_rejected(self):
        state = self.runtime.snapshot()
        state['clock'] = create_clock(10,1,1)
        self.runtime.db.execute('UPDATE state SET value=?',(json.dumps(state),))
        self.runtime.db.commit()
        model = ScriptedModel([answer({'action':'no_action'})])
        result=SellerAgent(self.runtime,model,TokenizerFixture()).propose(10)
        self.assertEqual(result['result'],{'action':'no_action'})
        self.assertEqual(self.runtime.snapshot(),state)
        model = ScriptedModel([answer({'prices':[{'item_id':1,'price_raw':111}]}),answer({'action':'no_action'})])
        with self.assertRaises(ValueError):SellerAgent(self.runtime,model,TokenizerFixture()).propose(10)
        self.assertEqual(self.runtime.snapshot(),state)

    def test_wait_and_quote_cannot_be_combined(self):
        self.runtime.start_episode(create_clock(10, 1, 1))
        before = self.runtime.snapshot()
        mixed = answer({"action": "no_action", "prices": [{"item_id": 1, "price_raw": 111}]})

        model = ScriptedModel([mixed])
        with self.assertRaisesRegex(ValueError, "'action' was unexpected"):
            SellerAgent(self.runtime, model, TokenizerFixture()).propose(10)
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(self.runtime.snapshot(), before)

    def test_market_cache_reuses_exact_category_and_shop_but_not_different_cutoff(self):
        full,_=self.runtime.inputs.seller(10,retrieve=False,observe=False,include_market=False)
        snapshot=PricingSnapshot(full)
        self.data._decision_cache.clear()
        with patch.object(market_queries,'rows',wraps=market_queries.rows) as read:
            first=snapshot.view([1]);load_market_context(first,self.data)
            n=read.call_count
            second=snapshot.view([3]);load_market_context(second,self.data)
            # Only the second product's own window is new; category and shop are shared.
            self.assertEqual(read.call_count,n+1)
            again=snapshot.view([1]);load_market_context(again,self.data)
            self.assertEqual(read.call_count,n+1)
            self.assertEqual(first,again)
            first['owned_products'][0]['market_position']['recent']['products']=999
            pristine=snapshot.view([1]);load_market_context(pristine,self.data)
            self.assertNotEqual(pristine,first)
            market_queries.shop_period(self.data,10,9)
            self.assertEqual(read.call_count,n+2)
        self.assertEqual(second['owned_products'][0]['market_position']['category_ids'],[1,2,3])

    def test_prepared_history_retains_other_products_and_reuses_exact_vectors(self):
        observed=[]
        for item in (1,3):
            row=self.data.records(item)[0]
            observed.extend({**row,'time_index':t,'price_raw':100+item+t} for t in range(1,9))
        calls=[]
        def encoder(text):calls.append(text);return [len(text)%11+1.,1.]
        index=prepare_seller_history(observed,seller_id=10,cutoff=10,recent_n=2)
        for item in (1,3):
            actual=index.retrieve([item],embedder=encoder)
            expected=seller_history(observed,seller_id=10,cutoff=10,recent_n=2,focus_ids=[item],embedder=encoder)
            self.assertEqual(actual,expected)
            self.assertTrue(any(s['item_id']!=item for s in actual['segments']))
            self.assertTrue(all(r['index']<10 for s in actual['segments'] for r in s['records']))
        cached=CachedHistoryEncoder(encoder,max_entries=2)
        cached('a');n=len(calls);cached('a');self.assertEqual(len(calls),n)
        cached('b');cached('c');cached('a');self.assertEqual(len(cached.documents),2)


class ExactCosineTests(unittest.TestCase):
    def test_authorization_zero_vectors_and_scalar_reference(self):
        rng=np.random.default_rng(7);vectors=rng.normal(size=(30,12));vectors[0]=0
        q=rng.normal(size=12);meta=[{'owner':i%2,'time':i} for i in range(30)]
        actual=cosine_scores(q,vectors,metadata=meta,predicates=[('owner','==',0),('time','<',16)])
        self.assertEqual(set(actual),set(range(0,16,2)))
        for i,score in actual.items():
            expected=np.dot(q,vectors[i])/(np.linalg.norm(q)*np.linalg.norm(vectors[i])) if i else 0.
            self.assertEqual(score,expected)
        self.assertTrue(all(v==0 for v in cosine_scores(np.zeros(12),vectors).values()))
        self.assertEqual(cosine_scores([1],[[1]],metadata=[{}],predicates=[('owner','==',0)]),{})
        for values in ([[float('nan')]], [[1,2]]):
            with self.assertRaises(ValueError):cosine_scores([1],values)
        with self.assertRaises(ValueError):cosine_scores([1],[[1]],predicates=[('owner','contains',0)])
