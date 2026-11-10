"""Share preparation, not stage-specific ranking or mutable model inputs."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from agent_simulation.datasets.samples.customer import sample_from_request
from agent_simulation.agents.retrieval.history import prepare_customer_history, customer_history
from agent_simulation.agents.retrieval.retrieval_text import event_text


class Encoder:
    def __init__(self):
        self.documents, self.queries = [], []

    def __call__(self, text):
        self.documents.append(text)
        return [1., float(len(text) % 7 + 1)]

    def encode_query(self, text):
        self.queries.append(text)
        return [1., 2. if text == '保温杯' else 3.]


def request():
    events = [dict(user_id=7,session_id=i,time_index=i,query=f'旧搜索{i}',
                   clicked_items=[],purchased_items=[]) for i in range(1,41)]
    return dict(query_input={'profile':{'user_id':7},'history_cutoff_exclusive':50,
                            'short_term_memory':{'search_history':events},
                            'provenance':{'request_ids':list(range(1,41))}},
                query_target={'query':'保温杯'},assigned_split='train',
                response_input={'displayed_items':[{'item_id':101,'item_title':'保温杯'}]},
                response_target={'clicked_item_ids':[101],'purchased_item_ids':[]})


class SharedCustomerHistoryTests(unittest.TestCase):
    def test_response_batches_share_formatting_but_not_mutable_input_or_labels(self):
        from agent_simulation.datasets.samples.customer import split_supervision
        from agent_simulation.agents.prompts.customer import factor_history, messages
        from agent_simulation.tests.fixtures import DisplayBudgetTokenizer
        source = request()
        source['response_input']['displayed_items'] = [
            {'item_id': i, 'item_title': '保温杯'} for i in range(101, 230)]
        _, response = sample_from_request('fixture', 50, source, history_embedder=Encoder(), render=False)
        with patch('agent_simulation.agents.prompts.customer.factor_history', wraps=factor_history) as factor:
            batches = split_supervision(response, DisplayBudgetTokenizer(64))
        self.assertEqual(factor.call_count, 1)
        for batch in batches:
            self.assertEqual(batch['messages'], messages(batch['input'], batch['target'], response=True))
            self.assertEqual(batch['batching']['request_weight'], 1 / len(batches))
        expected = deepcopy(batches[1]['input'])
        batches[0]['input']['short_term_memory']['search_history'][0]['query'] = 'changed'
        self.assertEqual(batches[1]['input'], expected)

    def test_consecutive_decisions_only_prepare_newly_evicted_events(self):
        events = request()['query_input']['short_term_memory']['search_history']
        encoder = Encoder()
        with patch('agent_simulation.agents.retrieval.history.event_text', wraps=event_text) as render:
            prepared = prepare_customer_history(events[:20], (), cutoff=21, recent_n=10, user_id=7)
            prepared.retrieve(embedder=encoder)
            self.assertEqual((render.call_count, len(encoder.documents)), (10, 10))
            prepared.update(events[:21], (), cutoff=22)
            actual = prepared.retrieve('保温杯', embedder=encoder)
            self.assertEqual((render.call_count, len(encoder.documents)), (11, 11))
            prepared.update(events[:21], (), cutoff=23)
            prepared.retrieve('手机', embedder=encoder)
            self.assertEqual((render.call_count, len(encoder.documents)), (11, 11))
        cold = prepare_customer_history(events[:21], (), cutoff=22, recent_n=10, user_id=7)
        self.assertEqual(actual, cold.retrieve('保温杯', embedder=Encoder()))

    def test_corrections_and_rewind_invalidate_candidates_without_stale_facts(self):
        events = request()['query_input']['short_term_memory']['search_history']
        prepared = prepare_customer_history(events, (), cutoff=50, recent_n=10, user_id=7)
        corrected = deepcopy(events)
        corrected[0]['query'] = '修正搜索'
        prepared.update(corrected, (), cutoff=50)
        self.assertEqual(prepared.candidates[0]['records'][0]['query'], '修正搜索')
        simulated = [dict(user_id=7, settled_step=1, query='仿真', clicked_items=[], purchased_items=[])]
        prepared.update(corrected, simulated, cutoff=50, simulation_before=2)
        cold = prepare_customer_history(corrected, simulated, cutoff=50, recent_n=10, user_id=7, simulation_before=2)
        self.assertEqual(prepared.retrieve(embedder=None), cold.retrieve(embedder=None))
        prepared.update(events[:15], (), cutoff=16)
        cold = prepare_customer_history(events[:15], (), cutoff=16, recent_n=10, user_id=7)
        self.assertEqual(prepared.retrieve(embedder=None), cold.retrieve(embedder=None))

    def test_append_only_stream_reads_new_suffix_and_matches_full_snapshot(self):
        events = request()['query_input']['short_term_memory']['search_history']
        prepared = prepare_customer_history(events[:20], (), cutoff=21, recent_n=10, user_id=7)

        class NewSuffixOnly(list):
            def __getitem__(self, key):
                if isinstance(key, slice) and (key.start or 0) < 20:
                    raise AssertionError("Append-only update reread a stored prefix")
                return super().__getitem__(key)

        prepared.update(NewSuffixOnly(events[:21]), (), cutoff=22, append_only=True)
        cold = prepare_customer_history(events[:21], (), cutoff=22, recent_n=10, user_id=7)
        self.assertEqual(prepared.recent_history(), cold.recent_history())
        self.assertEqual(prepared.retrieve(embedder=None), cold.retrieve(embedder=None))
        before = prepared.retrieve(embedder=None)
        for history, cutoff in ((events[:20], 22), (events[:22], 22)):
            with self.assertRaises(ValueError):
                prepared.update(history, (), cutoff=cutoff, append_only=True)
            self.assertEqual(prepared.retrieve(embedder=None), before)
        # Inserting observed events before existing simulated history must rebuild.
        simulated = [dict(user_id=7, settled_step=1, query='仿真', clicked_items=[], purchased_items=[])]
        prepared.update(events[:21], simulated, cutoff=23, simulation_before=2, append_only=True)
        prepared.update(events[:22], simulated, cutoff=23, simulation_before=2, append_only=True)
        cold = prepare_customer_history(events[:22], simulated, cutoff=23, recent_n=10,
                                        user_id=7, simulation_before=2)
        self.assertEqual(prepared.recent_history(), cold.recent_history())
        self.assertEqual(prepared.retrieve(embedder=None), cold.retrieve(embedder=None))

    def test_pair_prepares_once_and_encodes_documents_once(self):
        source=request();encoder=Encoder()
        with patch('agent_simulation.datasets.samples.customer.prepare_customer_history',wraps=prepare_customer_history) as prepare, \
             patch('agent_simulation.agents.retrieval.history.event_text',wraps=event_text) as render:
            query,response=sample_from_request('fixture',50,source,history_embedder=encoder)
        self.assertEqual(prepare.call_count,1)
        self.assertEqual(render.call_count,30)
        self.assertEqual(len(encoder.documents),30)
        self.assertEqual(len(encoder.queries),2)
        self.assertNotIn('保温杯',encoder.queries[0])
        self.assertEqual(encoder.queries[1],'保温杯')
        for stage,index in [('query',0),('response',1)]:
            standalone=sample_from_request('fixture',50,source,stage=stage,history_embedder=Encoder())
            self.assertEqual(standalone[index],(query,response)[index])

    def test_deferred_rendering_only_builds_the_final_query_and_response_messages(self):
        from agent_simulation.datasets.samples.customer import split_supervision
        from agent_simulation.datasets.labels import expand_record
        from agent_simulation.agents.prompts.customer import encode_input
        from agent_simulation.tests.fixtures import TokenizerFixture
        source = request()
        source['response_input']['displayed_items'] = [
            {'item_id':i,'item_title':f'商品{i}'} for i in range(101,230)]
        eager_query, eager_response = sample_from_request('fixture',50,source,history_embedder=Encoder())
        tokenizer = TokenizerFixture()
        expected = split_supervision(eager_response, tokenizer)
        with patch('agent_simulation.agents.prompts.customer.encode_input', wraps=encode_input) as render:
            query,response = sample_from_request('fixture',50,source,history_embedder=Encoder(),render=False)
            self.assertNotIn('messages',query)
            self.assertNotIn('messages',response)
            render.assert_not_called()
            actual = list(expand_record('customer', (query, response), tokenizer=tokenizer))
        self.assertEqual(render.call_count, len(actual))
        self.assertEqual(actual, [eager_query, *expected])

    def test_response_query_cannot_change_query_input_or_label_independent_history(self):
        source=request();original=deepcopy(source)
        first=sample_from_request('fixture',50,source,history_embedder=Encoder())
        changed=deepcopy(source);changed['query_target']={'query':'手机壳'}
        changed['response_target']={'clicked_item_ids':[],'purchased_item_ids':[101]}
        second=sample_from_request('fixture',50,changed,history_embedder=Encoder())
        self.assertEqual(first[0]['input'],second[0]['input'])
        self.assertEqual(first[0]['messages'][:-1],second[0]['messages'][:-1])
        self.assertEqual(source,original)
        first[0]['input']['short_term_memory']['search_history'][0]['query']='mutated'
        self.assertNotEqual(first[0]['input']['short_term_memory'],first[1]['input']['short_term_memory'])

    def test_ranked_results_do_not_mutate_the_shared_pool(self):
        events=request()['query_input']['short_term_memory']['search_history']
        prepared=prepare_customer_history(events,(),cutoff=50,recent_n=30,user_id=7)
        first=prepared.retrieve('保温杯',embedder=None)
        expected=deepcopy(first)
        first['segments'][0]['records'][0]['query']='mutated'
        self.assertEqual(prepared.retrieve('保温杯',embedder=None),expected)
        self.assertTrue(all('search_text' in row and 'match' not in row for row in prepared.candidates))
        with self.assertRaisesRegex(ValueError,'snapshot differs'):
            customer_history(events,(),cutoff=51,recent_n=30,user_id=7,prepared=prepared)


if __name__=='__main__':unittest.main()


class ContextSummaryTests(unittest.TestCase):

    def test_joint_expansion_summarizes_identical_recent_history_once(self):
        from agent_simulation.datasets.labels import expand_record
        from agent_simulation.agents.prompts.customer import decode_input
        from agent_simulation.tests.fixtures import DisplayBudgetTokenizer
        source = request()
        source["response_input"]["displayed_items"] = [{"item_id": i} for i in range(101, 230)]
        record = sample_from_request("fixture", 50, source, history_embedder=None, render=False)
        before = deepcopy(record)

        class Summaries:
            def __init__(self):
                self.scopes = []
            def summarize(self, records, *, scope, compact=None):
                self.scopes.append(scope)
                return {"source": "model_summary", "text": scope} if records else None

        model = Summaries()
        with patch("agent_simulation.agents.context.context_model", return_value=model):
            samples = list(expand_record("customer", record, tokenizer=DisplayBudgetTokenizer(64)))
        self.assertEqual(model.scopes.count("Customer recent history"), 1)
        self.assertEqual(model.scopes.count("Customer retrieved older history"), 2)
        self.assertEqual(record, before)
        self.assertGreater(len(samples), 2)
        recent = [decode_input(sample["messages"][1]["content"])["short_term_memory"] for sample in samples]
        self.assertTrue(all(value == recent[0] for value in recent))
        self.assertEqual(sum(sample.get("batching", {}).get("request_weight", 0) for sample in samples), 1.)

    def model(self, reply, threshold=1):
        from agent_simulation.agents.context import ContextModel
        from agent_simulation.tests.fixtures import ScriptedModel, TokenizerFixture, answer
        client = ScriptedModel([answer(reply), answer(reply)])
        client.tokenizer = TokenizerFixture()
        return ContextModel("fixture", history_tokens=threshold, client=client), client

    def test_compact_history_avoids_a_model_call_and_preserves_every_event(self):
        import json
        from agent_simulation.agents.prompts.customer import messages, decode_input, _encoded_history
        product = {"item_id": 9, "item_title": "保温杯" * 100, "brand_name": "品牌", "price_raw": 100}
        events = [{"time_index": i, "query": "杯子", "clicked_items": [deepcopy(product)],
                   "purchased_items": []} for i in range(20)]
        state = {"profile": {"user_id": 1}, "short_term_memory": {"search_history": events}}
        model, client = self.model({"summary": "should not run"})
        encoded = {k: json.loads(v) for k, v in _encoded_history(
            {"short_term_memory": state["short_term_memory"]}).items()}
        small = len(model.tokenizer.encode(json.dumps(
            {"scope": "Customer recent history", "history": encoded},
            ensure_ascii=False, separators=(",", ":")), add_special_tokens=False))
        raw = sum(len(model.tokenizer.encode(json.dumps(e, ensure_ascii=False),
                                            add_special_tokens=False)) for e in events)
        self.assertLess(small, raw)
        model.history_tokens = (small + raw) // 2
        before = deepcopy(state)
        with patch("agent_simulation.agents.context.context_model", return_value=model):
            shown = decode_input(messages(state)[1]["content"])
        self.assertEqual(client.requests, [])
        self.assertEqual(shown, before)
        self.assertEqual(state, before)

    def test_title_sharing_keeps_old_titles_prices_and_background_sequences(self):
        from agent_simulation.agents.prompts.customer import messages, decode_input
        title = "同一商品的完整商品描述" * 20
        background = {"observed_time_index": 10, "recent_clicked_item_ids": [9, 9],
                      "recent_purchased_item_ids": [9], "statistics_30d": {"orders": 1},
                      "item_details": [{"item_id": 9, "item_title": title, "brand_name": "新品牌"}]}
        state = {"profile": {"user_id": 1}, "behavioral_summary": background,
                 "short_term_memory": {"search_history": [
                     {"time_index": i, "query": "杯子", "clicked_items": [
                         {"item_id": 9, "item_title": title, "brand_name": "旧品牌", "price_raw": i}],
                      "purchased_items": []} for i in range(3)]},
                 "displayed_items": [{"item_id": 9, "item_title": title, "price_raw": 100},
                                     {"item_id": 10, "item_title": "旧标题", "price_raw": None}]}
        state["short_term_memory"]["search_history"].append(
            {"time_index": 5, "query": "旧商品", "clicked_items": [
                {"item_id": 9, "item_title": "旧标题", "price_raw": None}], "purchased_items": []})
        before = deepcopy(state)
        with patch("agent_simulation.agents.context.context_model", return_value=None):
            result = messages(state, response=True)
        self.assertIn('"title_ref"', result[1]["content"])
        self.assertEqual(decode_input(result[1]["content"]), before)
        self.assertEqual(state, before)

    def test_short_history_skips_generation_and_preserves_prompt(self):
        from agent_simulation.agents.prompts.customer import messages
        _, sample = sample_from_request('fixture', 50, request(), history_embedder=None, render=False)
        before = deepcopy(sample['input'])
        expected = messages(before, response=True)
        model, client = self.model({'summary': 'unused'}, threshold=100000)
        with patch('agent_simulation.agents.context.context_model', return_value=model):
            self.assertEqual(messages(before, response=True), expected)
        self.assertEqual(client.requests, [])
        self.assertEqual(sample['input'], before)

    def test_long_history_keeps_background_and_labels_and_summarizes_once_across_batches(self):
        from agent_simulation.datasets.samples.customer import split_supervision
        from agent_simulation.agents.prompts.customer import decode_input
        from agent_simulation.tests.fixtures import DisplayBudgetTokenizer
        source = request()
        source['query_input']['behavioral_summary'] = {'observed_time_index': 40,
            'statistics_30d': {'orders': 7}, 'recent_purchased_item_ids': [9], 'item_details': []}
        source['response_input']['displayed_items'] = [{'item_id': i} for i in range(101, 230)]
        source['response_target']['purchased_item_ids'] = [229]
        _, sample = sample_from_request('fixture', 50, source, history_embedder=None, render=False)
        before = deepcopy(sample)
        model, client = self.model({'summary': '历史搜索涉及多个主题，没有购买记录。'})
        with patch('agent_simulation.agents.context.context_model', return_value=model):
            batches = split_supervision(sample, DisplayBudgetTokenizer(64))
        self.assertGreater(len(batches), 1)
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(sample, before)
        self.assertNotIn('response_target', client.requests[0]['messages'][1]['content'])
        self.assertEqual([i for batch in batches for i in batch['target']['purchased_item_ids']], [229])
        for batch in batches:
            view = decode_input(batch['messages'][1]['content'])
            self.assertEqual(view['behavioral_summary'], before['input']['behavioral_summary'])
            self.assertEqual(view['short_term_memory']['summary']['source'], 'model_summary')

    def test_rendered_response_reuses_summary_when_partitioned(self):
        from agent_simulation.datasets.samples.customer import split_supervision
        from agent_simulation.tests.fixtures import DisplayBudgetTokenizer
        source = request()
        source['response_input']['displayed_items'] = [{'item_id': i} for i in range(101, 230)]
        model, client = self.model({'summary': '已有历史摘要'})
        with patch('agent_simulation.agents.context.context_model', return_value=model):
            _, sample = sample_from_request('fixture', 50, source, stage='response', history_embedder=None)
            split_supervision(sample, DisplayBudgetTokenizer(64))
        self.assertEqual(len(client.requests), 2)

    def test_seller_summary_keeps_original_business_statistics_and_source(self):
        from agent_simulation.tests.seller_context_fixture import seller_state
        from agent_simulation.agents.prompts.seller import seller_messages, decode_input
        state = seller_state()
        state['owned_products'][0]['business_summary'] = {'observed_time_index': 2,
            'statistics_30d': {'exposures': 100, 'clicks': 7, 'orders': 2}}
        before = deepcopy(state)
        model, client = self.model({'summary': '旧报价与经历'})
        # The stub returns the same summary independently for older and recent histories.
        client.replies = iter([{'role': 'assistant', 'content': '{"summary":"旧报价与经历"}'}] * 4)
        with patch('agent_simulation.agents.context.context_model', return_value=model):
            view = decode_input(seller_messages(state)[1]['content'])
        self.assertEqual(state, before)
        self.assertEqual(view['current_state']['products'][0]['business_summary'],
                         before['owned_products'][0]['business_summary'])
        self.assertEqual(view['history']['summary']['source'], 'model_summary')

    def test_long_retrieved_history_does_not_replace_short_recent_history(self):
        import json
        from agent_simulation.agents.prompts.customer import prepare_history
        recent = {"search_history": [{"query": "杯子", "clicked_items": [], "purchased_items": []}]}
        state = {"short_term_memory": recent, "retrieved_context": {"history": {
            "segments": [{"records": [{"query": "旧经历" * 2000}]}]}}}
        before = deepcopy(state)
        model, client = self.model({"summary": "较早经历摘要"}, threshold=500)
        with patch('agent_simulation.agents.context.context_model', return_value=model):
            encoded = prepare_history(state)
        self.assertEqual(json.loads(encoded["short_term_memory"]), recent)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(json.loads(encoded["retrieved_context"])["history"]["summary"]["text"], "较早经历摘要")
        self.assertEqual(state, before)


    def test_runtime_summaries_keep_cited_customer_and_seller_records(self):
        from agent_simulation.tests.agents.test_customer_evidence import state as customer_fixture
        from agent_simulation.tests.agents.test_seller_evidence import state as seller_fixture, memory
        from agent_simulation.agents.prompts.customer import customer_decision_messages, decode_input as customer_decode
        from agent_simulation.agents.prompts.seller import seller_decision_messages, decode_input as seller_decode

        class LongHistory:
            def summarize(self, records, *, scope, compact=None):
                return {"source": "model_summary", "text": "历史摘要"} if records else None

        customer = customer_fixture()
        citation = {"same_observations_in_retrieved_history": [{"segment_number": 1, "record_number": 1}]}
        customer["long_term_memory"] = [{"content": "旧经历", "evidence_times": [citation]}]
        seller = seller_fixture()
        seller["short_term_memory"] = []
        first = seller["retrieved_context"]["history"]["segments"][0]
        second = {**first, "records": [dict(first["records"][0], index=2)]}
        seller["retrieved_context"]["history"]["segments"] = [first, second]
        m = memory(step=2)
        m["evidence_times"][0]["same_observations_in_retrieved_history"] = [
            {"segment_number": 2, "record_number": 1}]
        seller["long_term_memory"] = [m]
        with patch('agent_simulation.agents.context.context_model', return_value=LongHistory()):
            c = customer_decode(customer_decision_messages(customer)[1]["content"])
            s = seller_decode(seller_decision_messages(seller)[1]["content"])
        self.assertTrue(c["retrieved_context"]["history"]["segments"][0]["records"])
        self.assertEqual(c["long_term_memory"][0]["evidence_times"], [citation])
        self.assertEqual([r["evidence_ref"] for r in s["history"]["similar_evidence"]], ["H2"])
        self.assertEqual(s["history"]["long_term_memory"][0]["supporting_evidence_refs"], ["H2"])
        self.assertIn("summary", s["history"])
