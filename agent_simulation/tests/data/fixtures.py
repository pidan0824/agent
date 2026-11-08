"""Small artificial records for contract tests."""

import json
from pathlib import Path
import tempfile

from agent_simulation.data.common import Config, connect
from agent_simulation.data.build.seller_tables import sources, ranking, business
from agent_simulation.data.build.splits import build as split
from agent_simulation.data.build.catalog import build
from agent_simulation.data.catalog import Catalog
from agent_simulation.data.market.products import PROFILE_FIELDS


def event(item, time, price, status="observed", seller=10):
    return {
        **dict.fromkeys(PROFILE_FIELDS, None),
        "item_id": item,
        "seller_id": seller,
        "seller_name": "shop",
        "time_index": time,
        "price_raw": price,
        "price_status": status,
        "exposure_records": 1,
        "click_records": 0,
        "purchase_records": 0,
        "source_split": "train",
    }


class CatalogFixture:
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        products = [
            dict(
                item_id=i,
                seller_id=s,
                seller_name=f"shop{s}",
                item_title=t,
                brand_id=1,
                brand_name="品牌",
                category_level1_id=1,
                category_level1_name="美妆",
                category_level2_id=2,
                category_level2_name="护肤",
                category_level3_id=3,
                category_level3_name="贴片",
            )
            for i, s, t in [(1, 10, "痘痘贴"), (2, 10, "同店另一商品"), (3, 20, "公开竞品")]
        ]
        requests = []
        rank = []
        for t in range(1, 6):
            r = dict(
                session_id=t,
                user_id=1,
                time_index=t,
                query="浴室柜" if t == 1 else "痘痘贴",
                impressed_item_ids=[1, 2, 3],
                clicked_item_ids=[1] if t == 1 else [],
                purchased_item_ids=[],
                split="test" if t == 5 else "train",
            )
            requests.append(r)
            for i in r["impressed_item_ids"]:
                rank.append(
                    dict(
                        user_id=1,
                        recently_clicked_item_ids=[1],
                        recently_purchased_item_ids=[2, 2],
                        user_statistical_features=dict(user_show_cnt_30d_hist=20, user_click_cnt_30d_hist=4,
                                                       user_order_cnt_30d_hist=2, user_gmv_30d_hist=200),
                        session_id=t,
                        target_item_id=i,
                        time_index=t,
                        target_item_price=None if (i == 2 and t == 2) else 100 + i,
                        is_clicked=int(i in r["clicked_item_ids"]),
                        is_purchased=0,
                        split=r["split"],
                        target_item_statistical_features=dict(
                            item_show_cnt_30d_hist=1,
                            item_click_cnt_30d_hist=0,
                            item_order_cnt_30d_hist=0,
                        ),
                    )
                )
        paths = {}
        for key, data in [
            ("items", products),
            (
                "users",
                [
                    {
                        "user_id": 1,
                        "gender": "F",
                        "age_bucket": "31-40",
                        "fre_country": "中国",
                        "fre_province": "河北",
                        "fre_city": "唐山",
                    }
                ],
            ),
            ("recall", requests),
            ("ranking", rank),
        ]:
            path = cls.root / (key + ".jsonl")
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data))
            paths[key] = {"path": str(path)}
        cls.cfg = Config("fixture", cls.root / "business", paths, 2)
        for stage in (sources, ranking, business):
            c = connect(cls.cfg)
            try:
                stage(c, cls.cfg)
            finally:
                c.close()
        cls.splits = cls.root / "splits"
        split(cls.cfg, cls.splits)
        cls.output = cls.root / "catalog"
        build(cls.cfg, cls.root / "users.jsonl", cls.splits, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.catalog = Catalog(self.output)
        self.addCleanup(self.catalog.close)
