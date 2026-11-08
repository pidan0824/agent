"""The two SFT stages share real memory; only Response can use the target query/list."""
from agent_simulation.agents.context.shopping import project_customer

from copy import deepcopy
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import duckdb
from agent_simulation.data.catalog import Catalog

from agent_simulation.datasets.samples.customer import customer_sample, sample_from_request, CustomerHistory
from agent_simulation.datasets.labels import customer_samples
from agent_simulation.datasets.labels import SFTDataset
from agent_simulation.tests.fixtures import TokenizerFixture
from agent_simulation.agents.prompts.customer import decode_input
from agent_simulation.agents.retrieval.history import PreparedCustomerHistory


class CustomerStageContextTests(unittest.TestCase):
    def test_stream_and_direct_search_all_prior_requests_outside_recent_window(self):
        with duckdb.connect() as con:
            con.execute("""CREATE TABLE requests AS SELECT 7 user_id,i session_id,i time_index,
                'train' split,'杯子' query,[1] impressed_item_ids,[]::INTEGER[] clicked_item_ids,
                []::INTEGER[] purchased_item_ids FROM range(1,104) t(i)""")
            con.execute("""CREATE TABLE customer_splits AS SELECT user_id,session_id,
                if(time_index=103,'validation','train') assigned_split FROM requests""")
            con.execute("CREATE TABLE users AS SELECT 7 user_id")
            con.execute("CREATE TABLE products AS SELECT 1 item_id")
            con.execute("CREATE TABLE ranking_observations AS SELECT session_id,1 item_id,100 price_raw FROM requests")
            catalog = Catalog.__new__(Catalog)
            catalog.connection, catalog.config = con, SimpleNamespace(dataset="fixture")
            with patch('agent_simulation.agents.retrieval.bge_encoder.default_encoder',
                       return_value=(lambda text: [1., 0.], 'fixture')):
                direct = customer_sample(catalog, 103, stage='query')[0]
                streamed, = customer_samples(catalog, 0, 1, split='validation', stage='query')
            self.assertEqual(direct['messages'], streamed['messages'])
            state = streamed['input']
            self.assertEqual(decode_input(streamed['messages'][1]['content']), project_customer(state))
            self.assertEqual([e['session_id'] for e in state['short_term_memory']['search_history']],
                             list(range(93, 103)))
            h = state['retrieved_context']['history']
            self.assertEqual((h['eligible_segments'], h['returned_segments']), (92, 5))
            self.assertTrue(all(s['end_inclusive'] < 93 for s in h['segments']))

    def test_history_uses_the_price_of_each_selected_exposure(self):
        with duckdb.connect() as con:
            con.execute("""CREATE TABLE requests AS SELECT 7 user_id,i session_id,i time_index,
                'train' split,'杯子' query,[8] impressed_item_ids,[8] clicked_item_ids,
                [8] purchased_item_ids FROM range(1,5) t(i)""")
            con.execute("CREATE TABLE customer_splits AS SELECT session_id,'train' assigned_split FROM requests")
            con.execute("CREATE TABLE users AS SELECT 7 user_id")
            con.execute("CREATE TABLE products AS SELECT 8 item_id")
            con.execute("""CREATE TABLE ranking_observations AS SELECT * FROM
                (VALUES (1,8,150),(2,8,0),(3,8,90),(4,8,999)) x(session_id,item_id,price_raw)""")
            catalog = Catalog.__new__(Catalog)
            catalog.connection, catalog.config = con, SimpleNamespace(dataset="fixture")
            direct = customer_sample(catalog, 4, stage="query")[0]
            data = SFTDataset.__new__(SFTDataset)
            data.catalog, data.tokenizer = catalog, TokenizerFixture()
            separate = [data.sample("query:4"), *data.request_samples(4, stage="response")]
            with patch.object(catalog, "request", wraps=catalog.request) as read:
                self.assertEqual(data.request_samples(4), separate)
                read.assert_called_once()
            rolled = list(customer_samples(catalog, 0, 1, stage="query"))[-1]
            self.assertEqual(direct["messages"], rolled["messages"])
            history = decode_input(rolled["messages"][1]["content"])["short_term_memory"]["search_history"]
            for action in ("clicked_items", "purchased_items"):
                self.assertEqual([(e["time_index"], e[action][0]["price_raw"]) for e in history],
                                 [(1, 150), (2, None), (3, 90)])

    def test_stage_selection_keeps_the_same_inputs_and_avoids_other_stage_retrieval(self):
        past = [{'session_id': i, 'time_index': i, 'query': f'搜索{i}',
                 'clicked_items': [], 'purchased_items': []} for i in range(1, 41)]
        request = {
            'query_input': {'profile': {'user_id': 7}, 'history_cutoff_exclusive': 50,
                            'short_term_memory': {'search_history': past},
                            'provenance': {'request_ids': list(range(1, 41))}},
            'query_target': {'query': '杯子'}, 'assigned_split': 'validation',
            'response_input': {'displayed_items': [{'item_id': 101, 'item_title': '杯子'}]},
            'response_target': {'clicked_item_ids': [101], 'purchased_item_ids': []},
        }
        expected = sample_from_request('fixture', 50, request, history_embedder=None)
        for stage, index in [('query', 0), ('response', 1)]:
            with patch.object(PreparedCustomerHistory, 'retrieve', autospec=True,
                              side_effect=PreparedCustomerHistory.retrieve) as calls:
                actual = sample_from_request('fixture', 50, request, stage=stage, history_embedder=None)
            self.assertEqual(calls.call_count, 1)
            self.assertEqual(actual[index], expected[index])
            self.assertIsNone(actual[1 - index])

    def test_filtered_cohorts_still_enter_later_history_without_building_prompts(self):
        history = CustomerHistory(7, 'fixture', include_history_ids=True)
        row = {
            'user_id': 7, 'session_id': 1, 'time_index': 1, 'split': 'train',
            'assigned_split': 'train', 'query': '之前搜索', 'user_profile': {'user_id': 7},
            'impressed_item_ids': [1, 2], 'clicked_item_ids': [1], 'purchased_item_ids': [],
            'displayed_products': [{'item_id': 1, 'item_title': '被点击商品'}],
        }
        class AppendOnly(list):
            def __iter__(self):
                raise AssertionError("Filtered request reread the user's whole history")

        history.train = AppendOnly()
        with patch('agent_simulation.datasets.samples.customer.sample_from_request',
                   side_effect=AssertionError('Filtered target was expanded')):
            self.assertEqual(list(history.samples([row], split='validation', stage='query')), [])
        history.train = history.train[:]
        next_row = {**row, 'session_id': 2, 'time_index': 2, 'assigned_split': 'validation'}
        peer = {**next_row, 'session_id': 3}
        later = {**next_row, 'session_id': 4, 'time_index': 3}
        with patch('agent_simulation.datasets.samples.customer.sample_from_request',
                   return_value=({}, None)) as build:
            list(history.samples([next_row, peer], split='validation', stage='query'))
            list(history.samples([later], split='validation', stage='query'))
        requests = [call.args[2] for call in build.call_args_list]
        self.assertEqual([r['query_input']['provenance']['request_ids'] for r in requests],
                         [[1], [1], [1, 2, 3]])
        self.assertEqual(requests[0]['response_input']['displayed_items'], [])

    def test_sql_query_only_stream_joins_history_choices_not_unused_exposures(self):
        with duckdb.connect() as connection:
            connection.execute("""CREATE TABLE requests AS SELECT 7 user_id,1 session_id,1 time_index,
                'train' split,'旧搜索' query,[1,2] impressed_item_ids,[1] clicked_item_ids,
                []::INTEGER[] purchased_item_ids UNION ALL SELECT 7,2,2,'train','杯子',[1,2],[1],[]""")
            connection.execute("""CREATE TABLE customer_splits AS SELECT session_id,user_id,time_index,
                split source_split,if(session_id=1,'train','validation') assigned_split FROM requests""")
            connection.execute("CREATE TABLE users AS SELECT 7 user_id")
            # Item 2 is deliberately absent: Query needs only the recorded historical choices.
            connection.execute("CREATE TABLE products AS SELECT 1 item_id,'杯子' item_title")
            connection.execute("CREATE TABLE ranking_observations AS SELECT 1 session_id,1 item_id,100 price_raw UNION ALL SELECT 2,1,100")
            catalog = SimpleNamespace(connection=connection, config=SimpleNamespace(dataset='fixture'))
            with patch('agent_simulation.datasets.labels.split_supervision',
                       side_effect=AssertionError('Query-only stream partitioned Response')):
                samples = list(customer_samples(catalog, 0, 1,
                                               split='validation', stage='query'))
            self.assertEqual([s['sample_id'] for s in samples], ['query:2'])
            self.assertEqual(samples[0]['input']['short_term_memory']['search_history'][0]['session_id'], 1)

    def test_response_stream_and_direct_sample_preserve_choices_with_unknown_prices(self):
        with duckdb.connect() as connection:
            connection.execute("""CREATE TABLE requests AS SELECT 7 user_id,1 session_id,1 time_index,
                'train' split,'杯子' query,[2,1,3,4,5] impressed_item_ids,[2] clicked_item_ids,
                [3] purchased_item_ids""")
            connection.execute("""CREATE TABLE customer_splits AS SELECT session_id,user_id,time_index,
                split source_split,split assigned_split FROM requests""")
            connection.execute("CREATE TABLE users AS SELECT 7 user_id")
            connection.execute("CREATE TABLE products AS SELECT unnest([1,2,3,4,5]) item_id")
            connection.execute("""CREATE TABLE ranking_observations AS SELECT 1 session_id,item_id,price_raw
                FROM (VALUES (1,125.5),(2,0),(3,-1),(4,NULL),(5,'Infinity'::DOUBLE)) p(item_id,price_raw)""")
            catalog = Catalog.__new__(Catalog)
            catalog.connection = connection
            catalog.config = SimpleNamespace(dataset="fixture")
            direct = customer_sample(catalog, 1, stage="response")[1]
            rolling, = customer_samples(catalog, 1, 2, split="train", stage="response")
            self.assertEqual(rolling["messages"], direct["messages"])
            items = decode_input(rolling["messages"][1]["content"])["displayed_items"]
            self.assertEqual([(p["item_id"], p["price_raw"]) for p in items],
                             [(2, None), (1, 125.5), (3, None), (4, None), (5, None)])
            self.assertEqual(rolling["target"], {"clicked_item_ids": [2], "purchased_item_ids": [3]})
            self.assertEqual(connection.execute(
                "SELECT price_raw FROM ranking_observations WHERE item_id=2").fetchone(), (0,))

    def test_both_stages_have_history_and_older_retrieval_without_target_leakage(self):
        past = [{'session_id': i, 'time_index': i, 'query': f'过去的搜索{i}',
                 'clicked_items': [], 'purchased_items': []} for i in range(1, 41)]
        request = {
            'query_input': {'profile': {'user_id': 7}, 'history_cutoff_exclusive': 50,
                            'short_term_memory': {'search_history': past},
                            'provenance': {'request_ids': list(range(1, 41))}},
            'query_target': {'query': '本次目标搜索'}, 'assigned_split': 'train',
            'response_input': {'displayed_items': [{'item_id': 101, 'item_title': '当前商品'}]},
            'response_target': {'clicked_item_ids': [101], 'purchased_item_ids': []}}
        original = deepcopy(request)
        with patch.object(PreparedCustomerHistory, 'retrieve', autospec=True,
                          side_effect=PreparedCustomerHistory.retrieve) as retrieve:
            query, response = sample_from_request(
                'explicit-test-fixture', 50, request, history_embedder=lambda text: [1., 0.]
            )
        self.assertEqual(retrieve.call_count, 2)
        self.assertEqual(retrieve.call_args_list[0].args[1], '')
        self.assertEqual(retrieve.call_args_list[1].args[1], '本次目标搜索')
        qi, ri = (decode_input(s['messages'][1]['content']) for s in (query, response))
        self.assertEqual(qi['profile'], ri['profile'])
        self.assertEqual(qi['short_term_memory'], ri['short_term_memory'])
        self.assertEqual(len(qi['short_term_memory']['search_history']), 10)
        for state in (qi, ri):
            self.assertNotIn('ranking', state['retrieved_context']['history'])
            self.assertTrue(all('match' not in seg for seg in state['retrieved_context']['history']['segments']))
            segments = state['retrieved_context']['history']['segments']
            self.assertTrue(segments)
            self.assertTrue(all(r['time_index'] < 31 for s in segments for r in s['records']))
            self.assertEqual(state['long_term_memory'], [])
        self.assertNotIn('query', qi)
        self.assertNotIn('displayed_items', qi)
        self.assertNotIn('本次目标搜索', query['messages'][1]['content'])
        self.assertEqual(ri['query'], query['target']['query'])
        self.assertEqual(ri['displayed_items'], request['response_input']['displayed_items'])
        self.assertNotIn('clicked_item_ids', response['messages'][1]['content'])
        self.assertEqual(request, original)
