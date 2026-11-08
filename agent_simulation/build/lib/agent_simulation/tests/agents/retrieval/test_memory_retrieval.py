"""Test retrieval text views and local Chinese BGE encoding."""

from agent_simulation import PROJECT_ROOT

from copy import deepcopy
import unittest

import numpy as np
from agent_simulation.agents.retrieval.bge_encoder import default_encoder, QUERY_INSTRUCTION
from agent_simulation.agents.retrieval.retrieval_text import event_text, query_text


class RetrievalViewTests(unittest.TestCase):
    def test_seller_query_uses_only_focal_description(self):
        products = [
            {"item_id": 1, "item_title": "隐形痘痘贴"},
            {"item_id": 2, "item_title": "无关产品" * 3000},
        ]
        context = {
            "owned_products": products,
            "history_presentation": {"detail_item_ids": [1]},
            "profile": {"secret_provenance": "not-an-embedding-field"},
        }
        before = deepcopy(context)
        text = query_text("seller", context)
        self.assertIn("隐形痘痘贴", text)
        self.assertNotIn("无关产品", text)
        self.assertNotIn("secret_provenance", text)
        self.assertEqual(context, before)
        context["history_presentation"]["detail_item_ids"] = []
        self.assertNotIn("隐形痘痘贴", query_text("seller", context))

    def test_summary_query_does_not_copy_unrelated_shop_products(self):
        profile = {
            "products": [
                {"item_id": 1, "item_title": "水杯"},
                {"item_id": 2, "item_title": "鞋子" * 1000},
            ]
        }
        event = {
            "content": [
                {
                    "item_id": 1,
                    "price_raw": None,
                    "price_status": "invalid_price",
                    "exposure_records": 1,
                    "click_records": 0,
                    "purchase_records": 1,
                }
            ]
        }
        before = deepcopy(event)
        text = query_text("seller", profile, event)
        self.assertIn("价格原始值：未知", text)
        self.assertIn("点击记录：0；购买记录：1", text)
        self.assertNotIn("鞋子", text)
        self.assertEqual(event, before)

    def test_customer_view_keeps_all_clicked_and_purchased_descriptions(self):
        event = {
            "content": {
                "session_id": 1,
                "time_index": 1,
                "query": "鞋",
                "clicked_items": [{"item_id": i, "item_title": f"商品{i}"} for i in range(20)],
                "purchased_items": [{"item_id": 99, "item_title": "购买商品99"}],
            }
        }
        before = deepcopy(event)
        text = event_text("customer", event)
        self.assertIn("商品0", text)
        self.assertIn("商品19", text)
        self.assertIn("购买商品99", text)
        self.assertEqual(event, before)

    def test_customer_query_excludes_current_exposure_list_and_uses_prior_history_for_search(self):
        history = {"query": "运动鞋", "clicked_items": [], "purchased_items": []}
        context = {
            "query": "金条",
            "displayed_items": [{"item_title": "不可混入检索查询"}],
            "short_term_memory": {"search_history": [history]},
        }
        text = query_text("customer", context)
        self.assertIn("金条", text)
        self.assertNotIn("运动鞋", text)
        self.assertNotIn("不可混入", text)
        del context["query"]
        self.assertIn("运动鞋", query_text("customer", context))
        self.assertEqual(query_text("seller", "同店其他商品的价格趋势"), "同店其他商品的价格趋势")


@unittest.skipUnless(
    (PROJECT_ROOT / "models/bge-small-zh-v1.5/model.safetensors").is_file(),
    "Local Chinese BGE weights are required",
)
class RealBGETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder, cls.identity = default_encoder()

    def test_every_token_is_encoded_across_chunk_boundaries(self):
        text = ("历史搜索运动鞋，点击透气轻便跑步鞋。" * 90) + "尾部记录：购买足金吊坠。"
        body = self.encoder.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        for query in (False, True):
            chunks = self.encoder.chunks(text, query=query)
            self.assertGreater(len(chunks), 1)
            self.assertEqual([i for p in chunks for i in p["body_ids"]], body)
            self.assertTrue(all(len(p["encoded"]["input_ids"]) <= 512 for p in chunks))
            self.assertEqual(chunks[-1]["end"], len(body))
            prefix = self.encoder.tokenizer.encode(QUERY_INSTRUCTION, add_special_tokens=False)
            if query:
                self.assertEqual(chunks[0]["encoded"]["input_ids"][1 : len(prefix) + 1], prefix)

    def test_real_vectors_are_normalized_and_include_the_tail(self):
        text = "用户曾搜索并点击运动鞋。" * 90 + "用户最后购买的是足金吊坠。"
        parts = self.encoder.chunks(text)
        full = np.asarray(self.encoder(text))
        first = self.encoder.tokenizer.decode(parts[0]["body_ids"])
        self.assertEqual(full.shape, (512,))
        self.assertTrue(np.isfinite(full).all())
        self.assertAlmostEqual(float(np.linalg.norm(full)), 1.0, places=5)
        np.testing.assert_array_equal(full, self.encoder(text))
        self.assertGreater(float(np.linalg.norm(full - np.asarray(self.encoder(first)))), 1e-3)
        self.assertFalse(self.encoder.model().training)
        self.assertFalse(any(p.requires_grad for p in self.encoder.model().parameters()))
        self.assertTrue(self.identity.startswith("bge-small-zh-v1.5:"))

    def test_chinese_retrieval_distinguishes_product_topics(self):
        q = np.asarray(self.encoder.encode_query("购买透气运动鞋"))
        related = np.asarray(self.encoder("用户点击了透气轻便的跑步运动鞋。"))
        other = np.asarray(self.encoder("用户购买了足金投资金条。"))
        self.assertGreater(float(q @ related), float(q @ other))


if __name__ == "__main__":
    unittest.main()
