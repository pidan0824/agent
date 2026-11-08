"""Decision evidence preserves ownership, separate clocks, citations and action scope."""
from copy import deepcopy
import json
import unittest
from agent_simulation.tests.data.fixtures import event
from agent_simulation.agents.context.pricing import build_pricing_context,overview_context
from agent_simulation.agents.context.pricing_evidence import SCHEMA, project_seller
from agent_simulation.agents.prompts.seller import pricing_messages,decode_input
from agent_simulation.agents.prompts.seller import seller_decision_messages
from agent_simulation.tests.seller_context_fixture import memory


def state():
    observed=[event(1,1,100),event(1,2,110),event(1,3,110),event(1,4,999)]
    feedback=[{**event(1,0,90),'request_id':1,'search_step':1,'settled_step':2}]
    ctx,_=build_pricing_context(observed,owner_id=10,cutoff=4,observations_per_item=3,
        public_pages=[],requested_item_ids=[1],simulated=feedback,simulation_cutoff=3,
        active_prices={'1':{'price_raw':80,'source':'simulation:submitted_price','simulation_step':2}})
    ctx=overview_context(ctx,[1])
    ctx['retrieved_context']={'history':{'segments':[dict(owner_id=10,item_id=1,
         clock='observed_time_index',records=[dict(index=1,price_raw=100,price_status='observed',
         exposure_records=1,click_records=0,purchase_records=0)])]}}
    return ctx


class SellerEvidenceTests(unittest.TestCase):
    def test_only_complete_recent_runs_are_deduplicated_with_separate_clocks(self):
        ctx=state(); before=deepcopy(ctx)
        view,_=project_seller(ctx)
        summaries={h['source']:h for h in view['history']['price_changes']}
        self.assertEqual(summaries['observed']['observation_range'],[1,3])
        self.assertEqual([r['price_raw'] for r in summaries['observed']['runs']],[100])
        self.assertEqual(summaries['observed']['recent_evidence_ref'],'R1:observed')
        self.assertEqual(summaries['settled']['runs'],[])
        self.assertEqual(summaries['settled']['recent_evidence_ref'],'R1:settled')
        self.assertEqual(ctx,before)

    def test_partial_recent_run_and_same_price_different_period_are_preserved(self):
        observed=[event(1,t,p) for t,p in enumerate([100,100,120,100,100,100],1)]
        ctx,_=build_pricing_context(observed,owner_id=10,cutoff=7,observations_per_item=2,
                                   public_pages=[],requested_item_ids=[1])
        ctx=overview_context(ctx,[1])
        view,_=project_seller(ctx)
        summary=view['history']['price_changes'][0]
        self.assertEqual(summary['observation_range'],[1,6])
        self.assertEqual(summary['observed_quote_changes'],2)
        self.assertEqual([(r['first_index'],r['last_index'],r['price_raw']) for r in summary['runs']],
                         [(1,2,100),(3,3,120),(4,6,100)])
        self.assertEqual(view['current_state']['products'][0]['recent'][0]['price_runs'][0]['observation_range'],[5,6])
        self.assertNotIn('recent_evidence_ref',summary)

    def test_future_or_inconsistent_summary_ranges_are_rejected(self):
        for field,update in [('all_observations',{'first_index':-1}),
                             ('recent_observation_runs',{'last_index':4})]:
            ctx=state(); h=ctx['owned_products'][0]['observed_price_history']
            target=h[field] if field=='all_observations' else h[field][0]
            target.update(update)
            with self.subTest(update=update),self.assertRaises(ValueError):project_seller(ctx)

    def test_summary_market_clocks_and_requested_ownership_are_checked(self):
        ctx=state();ctx['owned_products'][0]['observed_price_history']['clock']='simulation_settled_step'
        with self.assertRaisesRegex(ValueError,'clock mismatch'):project_seller(ctx)
        for ids in ([1,1],[999]):
            ctx=state();ctx['requested_price_item_ids']=ids
            with self.assertRaisesRegex(ValueError,'Requested product'):project_seller(ctx)
        ctx=state();ctx['owned_products'][0]['market_position']={'simulation':{
            'price_as_of':{'clock':'simulation_settled_step','as_of':3}}}
        with self.assertRaisesRegex(ValueError,'simulation market'):project_seller(ctx)

    def test_sft_and_runtime_share_projection_and_keep_action_protocol(self):
        ctx=state(); before=deepcopy(ctx)
        sft=pricing_messages(ctx,{'prices':[{'item_id':1,'price_raw':777}]})
        live=seller_decision_messages(ctx)
        self.assertEqual(sft[:2][1],live[1])
        self.assertNotIn('777',sft[1]['content'])
        self.assertEqual(ctx,before)
        v=decode_input(live[1]['content'])
        self.assertEqual(v['schema'], SCHEMA)
        saved = {**v, 'schema': 'seller-decision-evidence-v4'}
        with self.assertRaisesRegex(ValueError, "Unknown Seller input format"):
            decode_input(json.dumps(saved))
        self.assertNotIn('long_term_memory',v['history'])
        self.assertNotIn('indices',live[1]['content'])

    def test_same_integer_in_two_clocks_is_two_facts_and_active_price_is_separate(self):
        v,audit=project_seller(state())
        p=v['current_state']['products'][0]
        self.assertEqual(p['last_observed_quote']['price_raw'],110)
        self.assertEqual(p['active_quote']['price_raw'],80)
        self.assertEqual([r['source'] for r in p['recent']],['observed','settled'])
        self.assertEqual(p['recent'][1]['price_runs'][0]['price_raw'],90)
        summaries={s['source']:s for s in v['history']['price_changes']}
        self.assertEqual(summaries['observed']['observations'],3)
        self.assertEqual(summaries['settled']['observations'],1)
        keys=[tuple(k) for e in audit['evidence'].values() for k in e['record_keys']]
        self.assertIn(('observed_time_index',1,2),keys)
        self.assertIn(('simulation_settled_step',1,2),keys)
        self.assertNotIn('999',str(v))

    def test_memory_reference_resolves_after_projection_and_exact_citations_are_audited(self):
        ctx=state(); m=memory('A cited experience',step=2)
        m['evidence_times'][0]['same_observations_in_retrieved_history']=[{'segment_number':1,'record_number':1}]
        ctx['long_term_memory']=[m]
        v,audit=project_seller(ctx)
        shown=v['history']['long_term_memory'][0]
        self.assertEqual(shown['supporting_evidence_refs'],['H1'])
        self.assertNotIn('source_ids',shown)
        self.assertEqual(audit['memories'][0]['source_ids'],m['source_ids'])
        self.assertNotIn('content', audit['memories'][0])
        self.assertEqual(project_seller(ctx, include_audit=False), (v, None))
        self.assertEqual(audit['evidence']['H1']['record_keys'],[['observed_time_index',1,1]])
        m['evidence_times'][0]['same_observations_in_retrieved_history'][0]['record_number']=999
        with self.assertRaisesRegex(ValueError,'Unresolved'):project_seller(ctx)

    def test_future_foreign_uncited_memory_is_rejected(self):
        for update in [{'owner':'seller:20'},{'simulation_step':3},{'source_ids':[]},
                       {'evidence_times':[{'origin':'observed','time_index':4}]},
                       {'evidence_times':[{'origin':'simulated','settled_step':3}]}]:
            with self.subTest(update=update):
                ctx=state();ctx['long_term_memory']=[{**memory(step=2),**update}]
                with self.assertRaises(ValueError):project_seller(ctx)

    def test_fact_conflicts_and_future_rows_rejected(self):
        for update in [{'index':4},{'item_id':2},{'seller_id':20}]:
            ctx=state();ctx['retrieved_context']['history']['segments'][0]['records'][0].update(update)
            with self.assertRaises(ValueError):project_seller(ctx)
        ctx=state();seg=ctx['retrieved_context']['history']['segments'][0]
        seg['records'][0].update(index=2,price_raw=999)
        with self.assertRaisesRegex(ValueError,'Conflicting'):project_seller(ctx)

    def test_duplicate_record_is_not_counted_twice_and_memory_can_reference_recent(self):
        ctx=state();seg=ctx['retrieved_context']['history']['segments'][0]
        seg['records'][0].update(index=2,price_raw=110)
        m=memory(step=2);m['evidence_times'][0]['same_observations_in_retrieved_history']=[{'segment_number':1,'record_number':1}]
        ctx['long_term_memory']=[m]
        v,a=project_seller(ctx)
        self.assertEqual(v['history']['similar_evidence'],[])
        self.assertEqual(v['history']['long_term_memory'][0]['supporting_evidence_refs'],['R1:observed'])

    def test_equal_price_fragments_stay_separate_and_keep_their_citations(self):
        ctx = state()
        ctx['short_term_memory'] = []
        first = ctx['retrieved_context']['history']['segments'][0]
        row = first['records'][0]
        second = {**first, 'records': [dict(row, index=2), dict(row, index=3, price_raw=120)]}
        ctx['retrieved_context']['history']['segments'] = [first, second]
        m = memory(step=2)
        m['evidence_times'][0]['same_observations_in_retrieved_history'] = [
            {'segment_number': 2, 'record_number': 1},
        ]
        ctx['long_term_memory'] = [m]
        view, audit = project_seller(ctx)
        fragments = view['history']['similar_evidence']
        self.assertEqual([s['evidence_ref'] for s in fragments], ['H1', 'H2'])
        self.assertEqual([[r['price_raw'] for r in s['price_runs']] for s in fragments],
                         [[100], [100, 120]])
        self.assertEqual(view['history']['long_term_memory'][0]['supporting_evidence_refs'], ['H2'])
        self.assertEqual(audit['evidence']['H2']['path'], ['retrieved_context', 'history', 'segments', 1])

    def test_skipped_observation_breaks_a_run_and_out_of_order_rows_fail(self):
        ctx = state()
        ctx['short_term_memory'] = []
        first = ctx['retrieved_context']['history']['segments'][0]
        row = first['records'][0]
        second = {**first, 'records': [dict(row, index=i) for i in (0, 1, 2)]}
        ctx['retrieved_context']['history']['segments'] = [first, second]
        view, _ = project_seller(ctx)
        runs = view['history']['similar_evidence'][1]['price_runs']
        self.assertEqual([run['observation_range'] for run in runs], [[0, 0], [2, 2]])
        second['records'].reverse()
        with self.assertRaisesRegex(ValueError, 'chronological'):
            project_seller(ctx)

    def test_price_history_keeps_unknown_status_and_deduplicates_only_covered_runs(self):
        observed = [event(1, 1, 100), event(1, 2, None, status='invalid_price'),
                    event(1, 3, None, status='conflicting_prices'), event(1, 4, 120)]
        ctx, _ = build_pricing_context(observed, owner_id=10, cutoff=5, observations_per_item=2,
                                       public_pages=[], requested_item_ids=[1])
        view, _ = project_seller(ctx)
        summary, = view['history']['price_changes']
        self.assertEqual([(r['first_index'], r.get('price_status')) for r in summary['runs']],
                         [(1, None), (2, 'invalid_price')])
        # A partial view that ends before the latest source record cannot prove run coverage.
        ctx['short_term_memory'].pop()
        view, _ = project_seller(ctx)
        self.assertEqual(len(view['history']['price_changes'][0]['runs']), 4)

    def test_memory_can_cite_an_earlier_cutoff_but_not_a_future_cutoff_or_event(self):
        ctx = state()
        m = memory(step=2)
        ctx['long_term_memory'] = [m]
        evidence = m['evidence_times'][0]
        evidence['historical_cutoff_exclusive'] = 2
        view, _ = project_seller(ctx)
        self.assertEqual(len(view['history']['long_term_memory']), 1)
        for cutoff, index in [(5, 1), (2, 2)]:
            with self.subTest(cutoff=cutoff, index=index):
                evidence.update(historical_cutoff_exclusive=cutoff, time_index=index)
                with self.assertRaisesRegex(ValueError, 'Future memory citation cutoff'):
                    project_seller(ctx)
