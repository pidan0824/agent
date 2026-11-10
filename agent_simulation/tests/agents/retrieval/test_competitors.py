"""Public product selection must keep ownership, time and diversity boundaries."""
import unittest
from copy import deepcopy
from unittest.mock import patch

from agent_simulation.agents.retrieval.competitors import rank_competitors, competitor_examples
from agent_simulation.agents.retrieval.history import CachedHistoryEncoder
from agent_simulation.tests.fixtures import MarketFixture


def product(item, seller, title="绿色保温杯", time=99):
    return dict(item_id=item, seller_id=seller, item_title=title, brand_id=1, brand_name="品牌",
                category_level1_id=1, category_level1_name="家居", category_level2_id=2,
                category_level2_name="饮具", category_level3_id=3, category_level3_name="杯子",
                reference_price_raw=100, reference_price_time_index=time, price_status="observed")


class CompetitorTests(unittest.TestCase):
    def test_postings_reuse_preserves_exact_scores_across_quote_title_changes_and_rewind(self):
        from agent_simulation.agents.retrieval.competitors import CoarseFeatures,coarse_candidates,title_terms
        f=product(1,10);index=CoarseFeatures()
        first=[product(i,i+100,title=f'保温杯 abc{i%3}',time=i) for i in range(2,302)]
        changed=[dict(p,reference_price_raw=200,item_title='其他新品' if p['item_id']%7==0 else p['item_title']) for p in first]
        for pool in (first,changed,first[:90],first):
            index.prepare(pool)
            slots, values = index._recall(f)
            scores = dict(zip(slots, values))
            lex = [scores.get(index.entries[p['item_id']], 0.) for p in pool]
            a=title_terms(f['item_title'])
            expected=[]
            for p in pool:
                b=title_terms(p['item_title']);expected.append(len(a&b)/len(a|b))
            self.assertEqual(lex,expected)
            self.assertEqual(coarse_candidates(f,pool,features=index),coarse_candidates(f,pool))

    def test_bit_mask_similarity_is_exact_for_cjk_latin_and_repeated_tokens(self):
        from agent_simulation.agents.retrieval.competitors import title_similarity,title_terms
        texts=['绿色保温杯2025款','绿色绿色杯 2025','abc abc 123','ABC 124','汉字测试','', '杯 abc']
        for left in texts:
            for right in texts:
                a,b=title_terms(left),title_terms(right)
                self.assertEqual(title_similarity(left,right),len(a&b)/len(a|b) if a|b else 0.)

    def test_single_coarse_rank_prioritizes_text_and_bounds_diverse_candidates(self):
        from agent_simulation.agents.retrieval.competitors import coarse_candidates
        focal=product(1,10)
        peers=[dict(product(i,i+1000,time=20),reference_price_raw=1000) for i in range(2,702)]
        cheap=dict(product(900,1900,title='另一个杯子',time=30),brand_id=2,reference_price_raw=100)
        fresh=dict(product(901,1901,title='新商品',time=99),brand_id=2,reference_price_raw=999)
        candidates=[*peers,cheap,fresh]
        pool=coarse_candidates(focal,candidates)
        self.assertEqual(len(pool),64)
        self.assertTrue({900,901}.isdisjoint(p['item_id'] for p in pool))
        self.assertEqual(pool,coarse_candidates(focal,list(reversed(candidates))))
        tied=[dict(product(i,20 if i<12 else 30,time=i),reference_price_raw=100+(i%2)) for i in range(2,22)]
        pool=coarse_candidates(focal,tied)
        self.assertEqual([p['item_id'] for p in pool],list(range(21,1,-1)))
        selected,meta=rank_competitors(focal,[*peers,cheap,fresh,product(999,9999,time=100)],100,embedder=None)
        self.assertLessEqual(meta['coarse_candidates'],64)
        self.assertNotIn(999,[p['item_id'] for p in selected])

    def test_only_past_other_shops_same_category_without_final_count_limit(self):
        focal = product(1, 10)
        peers = [product(2, 10), product(3, 20, time=100), product(4, 30),
                 product(5, 30), product(6, 40), product(7, 50), product(8, 60)]
        peers[-1]["category_level3_id"] = 9
        original = deepcopy(peers)
        selected, meta = rank_competitors(focal, peers, 100, embedder=None)
        self.assertEqual({p["seller_id"] for p in selected}, {30, 40, 50})
        self.assertEqual(meta["eligible_products"], 4)
        self.assertEqual(len(selected), 4)
        self.assertEqual(peers, original)

    def test_freshness_breaks_equal_relevance_and_no_padding(self):
        selected, _ = rank_competitors(product(1, 10), [product(2, 20, time=2), product(3, 30)],
                                       100, embedder=None)
        self.assertEqual([p["item_id"] for p in selected], [3, 2])
        self.assertEqual(rank_competitors(product(1, 10), [], 100, embedder=None)[0], [])

    def test_semantics_use_existing_cached_encoder_with_bounded_candidates(self):
        encoded = []
        def encode(text):
            encoded.append(text)
            return [1, 0] if "保温" in text else [0, 1]
        cache = CachedHistoryEncoder(encode)
        peers = [product(i, i + 10, title=f"杯子{i}") for i in range(2, 102)]
        peers[10]["item_title"] = "绿色保温杯"
        selected, meta = rank_competitors(product(1, 10), peers, 100, embedder=cache)
        self.assertEqual(selected[0]["item_id"], peers[10]["item_id"])
        self.assertLessEqual(meta["encoded_candidates"], 64)
        before = len(encoded)
        rank_competitors(product(1, 10), peers, 100, embedder=cache)
        self.assertEqual(len(encoded) - before, 1)  # Query only; document vectors reused.


class CompetitorContextTests(MarketFixture, unittest.TestCase):
    def test_indexed_snapshot_matches_full_pool_on_rewind_and_reuses_unchanged_features(self):
        from agent_simulation.data.market.quotes import CategorySnapshots
        self.data.category_snapshots=CategorySnapshots()
        self.data.competitor_features=self.data.category_snapshots.derived
        for before in (10,4,10):
            actual=competitor_examples(self.data,10,1,before,0,embedder=None)
            focal=self.data.product(1,before)
            expected,meta=rank_competitors(focal,self.data.competitor_pool(10,1,before),before,embedder=None)
            self.assertEqual([p['item_id'] for p in actual['products']],[p['item_id'] for p in expected])
            self.assertEqual(actual['selection'],meta)
            index=self.data.competitor_features[next(reversed(self.data.competitor_features))]
            with patch.object(index,'_update',wraps=index._update) as update:
                competitor_examples(self.data,10,1,before,0,embedder=None)
                self.assertEqual(update.call_count,0)

    def test_default_competitors_exclude_old_quotes_without_full_history_fallback(self):
        from agent_simulation.data.catalog import DataRepository
        data=DataRepository(self.c,self.cfg,10010)
        with patch.object(data,'competitor_pool',side_effect=AssertionError('No full-history fallback')):
            result=competitor_examples(data,10,1,10010,10,embedder=None)
        self.assertEqual(result['products'],[])
        self.assertEqual(result['selection']['eligible_products'],0)
        self.assertTrue(data.competitor_pool(10,1,10010))
        # Own historical quotes still provide pricing supervision and price changes.
        self.assertTrue(data.records(1))

    def test_recent_observed_filter_keeps_live_market_prices_and_own_history(self):
        from pathlib import Path
        from agent_simulation.data.catalog import DataRepository
        from agent_simulation.environment.market import MarketEnvironment
        data=DataRepository(self.c,self.cfg,10010)
        runtime=MarketEnvironment.create(Path(self.temp.name)/'recent.sqlite',data,[1,2,3,4],{},history_per_item=2)
        try:
            runtime.inputs.history_embedder=None
            context=next(runtime.inputs.seller_targets(10,state=runtime.snapshot()))
            own=context['owned_products'][0];market=own['market_position']
            self.assertEqual(market['competitor_examples']['products'],[])
            self.assertEqual(market['recent']['products'],0)
            self.assertEqual(market['simulation']['median_active_price'],80)
            self.assertEqual(own['observed_price_history']['all_observations']['observations'],3)
        finally:runtime.close()

    def test_prequalified_pool_matches_filter_and_respects_rewind(self):
        from agent_simulation.data.market.products import complete_category
        for before in (10,4,10):
            expected=[p for p in self.data.public_pool(10,1,before)['candidates'] if complete_category(p)]
            self.assertEqual(self.data.competitor_pool(10,1,before),expected)
            self.assertTrue(all(p['seller_id']!=10 and p['reference_price_time_index']<before for p in expected))

    def test_public_pool_reuses_category_without_leaking_mutations_or_time(self):
        from agent_simulation.data.market.quotes import category_snapshot
        with patch('agent_simulation.data.market.quotes.category_snapshot', wraps=category_snapshot) as load:
            first = self.data.discover(10, [1, 3], 1, 10)
            expected = deepcopy(first['candidates'])
            first['candidates'][0]['item_title'] = 'changed'
            second = self.data.discover(10, [1, 3], 3, 10)
            self.assertEqual(second['candidates'], expected)
            self.assertEqual(load.call_count, 1)
            earlier = self.data.discover(10, [1, 3], 1, 4)
            self.assertEqual(load.call_count, 2)
            self.assertTrue(all(p['reference_price_time_index'] < 4 for p in earlier['candidates']))
            with self.assertRaises(ValueError):
                self.data.public_pool(20, 1, 10)

    def test_simulated_price_is_separate_and_private_traffic_not_exposed(self):
        state = self.runtime.snapshot()
        state["active_prices"]["2"]["price_raw"] = 123
        result = competitor_examples(self.data, 10, 1, 10, 0, embedder=None, simulation=state)
        card, = result["products"]
        self.assertEqual(card["item_id"], 2)
        self.assertIsNone(card["reference_price_raw"])
        self.assertEqual(card["simulation_price"]["price_raw"], 123)
        self.assertNotIn("exposure_records", card)
        self.assertNotIn("click_records", card)
        self.assertNotIn("source", card["simulation_price"])

class ProgressiveRankingTests(unittest.TestCase):
    def test_or_recall_keeps_future_and_changed_titles_out(self):
        from agent_simulation.agents.retrieval.competitors import CoarseFeatures,coarse_candidates
        from agent_simulation.data.market.quotes import QuoteSnapshot
        def row(p):
            return dict(item_id=p['item_id'],_public=p,_complete_category=True)
        focal=product(1,10,title='cup green')
        old=[product(i,i+10,title='cup' if i%2 else 'green',time=2) for i in range(2,90)]
        new=[dict(p,item_title='unrelated',reference_price_raw=200,reference_price_time_index=7)
             if p['item_id']<70 else p for p in old]
        # The first title matches each document by OR although no document has all words.
        index=CoarseFeatures()
        for revision,(pool,end,parent) in enumerate(((old,3,None),(new,8,1),(old,3,None)),1):
            rows={p['item_id']:row(p) for p in pool}
            snapshot=QuoteSnapshot(rows,revision,parent,list(rows.values()),end)
            selected,_=index.select_snapshot(focal,snapshot,end)
            self.assertEqual(selected,coarse_candidates(focal,pool))
            self.assertTrue(all(p['reference_price_time_index']<end for p in selected))
        self.assertEqual(index.last_work['zero_fallback_candidates'],0)

    def test_selection_equals_full_lexicographic_sort_with_time_ties_and_shop_caps(self):
        import random
        from collections import Counter
        from agent_simulation.agents.retrieval.competitors import coarse_candidates,title_similarity,CoarseFeatures
        rng=random.Random(42);features=CoarseFeatures()
        for run in range(12):
            focal=product(1,10,title=rng.choice(['保温杯','杯','abc 绿色','']),time=100)
            focal['reference_price_raw']=rng.choice([100,120,None])
            peers=[dict(product(i,20+rng.randrange(23),title=rng.choice(['保温杯','杯','abc 绿色','陌生商品','']),time=rng.randrange(1,100)),
                        reference_price_raw=rng.choice([50,80,100,120,200,None]),brand_id=rng.choice([None,1,2])) for i in range(2,1202)]
            def key(p):
                return (title_similarity(focal['item_title'],p['item_title']),int(p['brand_id']==1),
                        p['reference_price_time_index'],-p['item_id'])
            expected=[];shops=Counter()
            for p in sorted(peers,key=key,reverse=True):
                if shops[p['seller_id']]>=4:continue
                shops[p['seller_id']]+=1;expected.append(p)
                if len(expected)==64:break
            self.assertEqual(coarse_candidates(focal,peers,features=features),expected)
        # Strong matches fill the shortlist without a zero-overlap fallback.
        focal=product(1,10)
        pool=[product(i,i+100,title='保温杯') for i in range(2,102)]+[product(i,i+100,title='完全不同商品') for i in range(102,2102)]
        coarse_candidates(focal,pool,features=features)
        self.assertEqual(features.last_work['zero_fallback_candidates'],0)


class CompetitorStabilityTests(unittest.TestCase):
    def test_prices_do_not_change_shortlist_or_final_reference_identity(self):
        from agent_simulation.agents.retrieval.competitors import coarse_candidates
        focal = product(1, 10)
        peers = [dict(product(i, i + 100, time=i), reference_price_raw=i * 50)
                 for i in range(2, 90)]
        encoder = CachedHistoryEncoder(lambda text: [1., 0.])
        baseline = rank_competitors(focal, peers, 100, embedder=encoder)[0]
        shortlist = coarse_candidates(focal, peers)
        for price in (None, 1, 100, 100000):
            changed = dict(focal, reference_price_raw=price)
            self.assertEqual(coarse_candidates(changed, peers), shortlist)
            self.assertEqual(rank_competitors(changed, peers, 100, embedder=encoder)[0], baseline)
        repriced = [dict(p, reference_price_raw=1) for p in peers]
        self.assertEqual([p['item_id'] for p in rank_competitors(focal, repriced, 100, embedder=encoder)[0]],
                         [p['item_id'] for p in baseline])
        # Comparable, recent quotes survive even when the older price is closer.
        self.assertEqual([p['item_id'] for p in baseline], list(range(89,25,-1)))

    def test_snapshot_replacement_matches_rebuild_after_expiry_and_profile_changes(self):
        from agent_simulation.agents.retrieval.competitors import CoarseFeatures
        from agent_simulation.data.market.quotes import QuoteSnapshot
        def row(p, complete=True):
            return dict(item_id=p['item_id'], _public=p, _complete_category=complete)
        focal = product(1, 10)
        first = {i: row(product(i, i + 20, time=2)) for i in range(2, 90)}
        changed = {i: deepcopy(r) for i, r in first.items() if i >= 20}
        changed[20] = row(product(20, 10, time=3))  # Now owned by the focal shop.
        changed[21] = row(product(21, 41, time=3), False)
        changed[22] = row(product(22, 42, title='新标题', time=3))
        changed[23] = row(product(23, 43, time=10))  # Equal to cutoff is invisible.
        changed[24] = row(product(24, 44, time=11))
        a = QuoteSnapshot(first, 1, None, list(first.values()), 10)
        b = QuoteSnapshot(changed, 2, 1, [changed[i] for i in range(20, 25)], 12)
        features = CoarseFeatures()
        for snapshot in (a, b, [], a):
            selected, eligible = features.select_snapshot(focal, snapshot, 10)
            fresh, count = CoarseFeatures().select_snapshot(focal, snapshot, 10)
            self.assertEqual((selected, eligible), (fresh, count))
            expected = sum(r['_complete_category'] and r['_public']['seller_id'] != 10
                           and r['_public']['reference_price_time_index'] < 10 for r in snapshot)
            self.assertEqual(eligible, expected)
            if snapshot is b:
                self.assertTrue({*range(2, 22), 23, 24}.isdisjoint(p['item_id'] for p in selected))
            with patch.object(features, '_update', wraps=features._update) as update:
                self.assertEqual(features.select_snapshot(focal, snapshot, 10), (selected, eligible))
                update.assert_not_called()


class ModelCompetitorTests(unittest.TestCase):

    def test_quote_comparison_distinguishes_old_category_and_brand_references(self):
        from agent_simulation.agents.context import quote_comparison, market_comparisons
        market = {"window": {"start_inclusive": 100}, "recent": {
            "own_quote": {"price_raw": 3980, "observation_index": 90, "price_band": "above_p75"},
            "p25_price": 1099.25, "p75_price": 3867.5, "max_price": 28310,
            "median_price": 1684.5, "same_brand": {"brand_name": "雅鹿", "median": 3994.5}}}
        before = deepcopy(market)
        text = quote_comparison(market)
        self.assertIn("Latest observed quote before the market window: 3980", text)
        self.assertIn("above the middle half of category quotes", text)
        self.assertIn("0.36% below the 雅鹿 median (3994.5)", text)
        self.assertIn("brand_and_category", market_comparisons(market))
        self.assertEqual(market, before)
        self.assertEqual(quote_comparison({"recent": {"own_quote": {"price_raw": None}}}), "")

    def test_single_oversized_competitor_is_a_capacity_failure(self):
        from agent_simulation.agents.context import ContextModel
        from agent_simulation.inference.requests import ContextCapacityError
        from agent_simulation.tests.fixtures import ScriptedModel, TokenizerFixture
        client = ScriptedModel([])
        client.tokenizer = TokenizerFixture()
        model = ContextModel("fixture", client=client)
        with patch("agent_simulation.agents.context.input_token_budget", return_value=1), \
             patch("agent_simulation.agents.context.prompt_tokens", return_value=[1, 2]):
            with self.assertRaises(ContextCapacityError):
                model.competitors(product(1, 10), [product(2, 20)])
        self.assertEqual(client.requests, [])

    def test_model_can_select_zero_or_more_than_three_without_changing_quotes(self):
        from agent_simulation.agents.context import ContextModel
        from agent_simulation.tests.fixtures import ScriptedModel, TokenizerFixture, answer
        focal = dict(product(1, 10), item_title="适配iPhone 15的保护壳")
        candidates = [dict(product(i, 20), reference_price_raw=i * 10,
                           item_title=f"适配ｉＰｈｏｎｅ１５的保护壳，款式{i}") for i in range(2, 8)]
        candidates[4]["item_title"] = "适配iPhone 14的保护壳"
        before = deepcopy(candidates)
        for terms, proposed, ids in (
            (["iPhone 15", "保护壳"], [], []),
            (["iPhone 15", "保护壳"], [2, 3, 4, 5, 6], [2, 3, 4, 5]),
            (["iPhone 14", "保护壳"], [6], []),
        ):
            result = {"match_terms": terms, "selected_items": proposed, "market_overview": "middle_and_upper_end"}
            client = ScriptedModel([answer(result)])
            client.tokenizer = TokenizerFixture()
            model = ContextModel('fixture', competitor_candidates=5, client=client)
            market = {"window": {"start_inclusive": 0, "end_exclusive": 10},
                      "recent": {"products": 5, "p25_price": 20, "p75_price": 50, "max_price": 70,
                                 "own_quote": {"price_raw": 30, "price_band": "p25_to_p75"}}}
            selected, analysis = model.competitors(focal, candidates, market)
            self.assertEqual([p['item_id'] for p in selected], ids)
            self.assertEqual(analysis['candidates_considered'], 5)
            self.assertEqual(analysis['selected_count'], len(ids))
            self.assertNotIn('match_terms', analysis)
            self.assertEqual(analysis['selected_observed_price_raw_range'], [min(ids)*10, max(ids)*10] if ids else None)
            import json
            from agent_simulation.agents.prompts.context_format import unpack
            payload = json.loads(client.requests[0]['messages'][1]['content'])
            self.assertIn('$table', payload['candidates'])
            self.assertEqual(unpack(payload)['market'], market)
            self.assertEqual(analysis['market_overview'],
                         'The middle half of category quotes ranges from 20 to 50. '
                         'The highest quote (70) is 1.40 times the 75th percentile.')
            self.assertIn('within the middle half of category quotes [20, 50]', analysis['price_position'])
            self.assertEqual(unpack(payload)['candidates'], candidates[:5])
            self.assertEqual(len(client.requests), 1)
            from agent_simulation.inference.local.qwen import action_parser
            output = json.dumps(result, ensure_ascii=False)
            parser = action_parser(client.requests[0]['json_schema'], 1536, ''.join(set(output)))
            for char in output:
                self.assertIn(char, parser.get_allowed_characters())
                parser = parser.add_character(char)
            self.assertTrue(parser.can_end())
            self.assertEqual(selected, [p for p in candidates if p['item_id'] in ids])
        self.assertEqual(candidates, before)

    def test_short_carrier_name_still_requires_matching_face_value(self):
        from agent_simulation.agents.context import ContextModel
        from agent_simulation.tests.fixtures import ScriptedModel, TokenizerFixture, answer

        focal = dict(product(1, 10), item_title="中国移动100元话费充值")
        matching = dict(product(2, 20), item_title="移动话费100元")
        wrong_value = dict(product(3, 30), item_title="移动话费200元")
        client = ScriptedModel([answer({"match_terms": ["中国移动", "100元", "话费"],
                                        "selected_items": [2, 3], "market_overview": ""})])
        client.tokenizer = TokenizerFixture()
        selected, _ = ContextModel("fixture", client=client).competitors(focal, [matching, wrong_value])
        self.assertEqual(selected, [matching])
