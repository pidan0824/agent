"""Shared market, tokenizer and model fixtures for tests."""

from copy import deepcopy

import hashlib
import json
from pathlib import Path
import tempfile

from agent_simulation.data.common import Config, connect, literal
from agent_simulation.data.market.products import PROFILE_FIELDS
from agent_simulation.data.catalog import DataRepository
from agent_simulation.environment.market import MarketEnvironment


class MarketFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cfg = Config("fixture", Path(self.temp.name), {}, 1)
        self.c = connect(self.cfg)
        profile = dict(
            zip(
                PROFILE_FIELDS,
                [1, 10, "shop", "绿色保温杯", 1, "品牌", 1, "家居", 2, "饮具", 3, "杯子"],
            )
        )
        events = []
        for item, seller, t, price, category in [
            (1, 10, 2, 100, 3),
            (1, 10, 4, 120, 3),
            (1, 10, 10, 999, 3),
            (2, 20, 3, 80, 3),
            (2, 20, 5, None, 3),
            (3, 10, 3, 200, 3),
            (4, 40, 3, 60, 9),
        ]:
            events.append(
                {
                    **profile,
                    "item_id": item,
                    "seller_id": seller,
                    "category_level3_id": category,
                    "time_index": t,
                    "price_raw": price,
                    "price_status": "observed" if price is not None else "invalid_price",
                    "exposure_records": 1,
                    "click_records": int(t == 4),
                    "purchase_records": int(t == 4),
                }
            )
        source = Path(self.temp.name) / "events.json"
        source.write_text(json.dumps(events, ensure_ascii=False))
        target = self.cfg.data / "seller_business/part_00.parquet"
        target.parent.mkdir(parents=True)
        self.c.execute(
            f"COPY (SELECT * FROM read_json_auto({literal(source)})) TO {literal(target)} (FORMAT PARQUET)"
        )
        self.data = DataRepository(self.c, self.cfg, 10)
        self.path = Path(self.temp.name) / "runtime.sqlite"
        histories = {
            7: [
                {
                    "session_id": 5,
                    "time_index": 2,
                    "query": "水杯",
                    "clicked_items": [],
                    "purchased_items": [],
                }
            ],
            8: [],
        }
        self.runtime = MarketEnvironment.create(
            self.path, self.data, [1, 2, 3, 4], histories, top_k=4, history_per_item=2
        )

    def tearDown(self):
        self.runtime.close()
        self.c.close()
        self.temp.cleanup()


from agent_simulation.agents.seller import SellerAgent
from agent_simulation.agents.customer import CustomerAgent


class AgentFixture(SellerAgent, CustomerAgent):
    """Combine Customer and Seller helpers for integration tests."""
    def seller(self, actor, **kwargs):
        return SellerAgent.run(self, actor, **kwargs)

    def customer(self, actor):
        return CustomerAgent.run(self, actor)


class TokenizerFixture:
    chat_template = "fixture-only"

    def apply_chat_template(self, messages, tools, **kwargs):
        return list(range(len(json.dumps([messages, tools])) // 4))

    def encode(self, text, *, add_special_tokens=False):
        return list(range(len(text) // 4))

class ScriptedModel:
    def __init__(self, replies):
        self.replies, self.requests = iter(replies), []

    def complete(self, request, *, call_id):
        self.requests.append(deepcopy(request))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply(request, call_id) if callable(reply) else deepcopy(reply)


def explicit_shop_model(runtime, replies):
    """Give one-batch episode fixtures explicit keep quotes for unchanged targets."""
    model = ScriptedModel(replies)
    complete = model.complete
    def respond(request, *, call_id):
        reply = complete(request, call_id=call_id)
        if reply.get("content") is not None:
            try:
                value = json.loads(reply["content"])
            except (TypeError, ValueError):
                return reply
            if isinstance(value, dict) and "prices" in value:
                prices = value["prices"]
                selected = {p["item_id"] for p in prices}
                quote = request["action_output"]["final_schemas"][0]
                targets = quote["properties"]["prices"]["items"]["properties"]["item_id"]["enum"]
                state = runtime.snapshot()
                prices.extend({"item_id": item, "price_raw": state["active_prices"][str(item)]["price_raw"]}
                              for item in targets if item not in selected)
                return answer(value)
        return reply
    model.complete = respond
    return model


def keep_prices(request):
    from agent_simulation.agents.prompts.seller import decode_input

    context = decode_input(request["messages"][1]["content"])
    return {"prices": [{"item_id": item["item_id"], "price_raw": item["active_quote"]["price_raw"]}
                       for item in context["current_state"]["products"]]}

def tool(name, arguments, identifier=None):
    def reply(request, call_id):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": identifier or call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }

    return reply

def answer(value):
    return {"role": "assistant", "content": json.dumps(value, ensure_ascii=False)}

class ProfileCatalogFixture:
    """Only replace catalog seed reads; settlement and query tools are real."""

    def __init__(self, runtime):
        self.state = runtime.snapshot()

    def customer_history(self, actor, cutoff, limit=None, official_split="train", *, all_prior=False):
        history = self.state["customers"][str(actor)]["observed_history"]
        return {
            "profile": {"user_id": actor, "gender": "fixture"},
            "short_term_memory": {
                "search_history": [e for e in history if e["time_index"] < cutoff][0 if all_prior else -limit:]
            },
        }

    def close(self):
        pass


def tiny_base(vocab=32):
    """Random one-layer model for tensor and checkpoint tests; no pretrained weights."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(11)
    config = Qwen2Config(
        vocab_size=vocab,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=32768,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(config)


def sample_identity(sample):
    """Bind the complete model payload, answer and original batching metadata."""
    value = {
        key: sample.get(key)
        for key in ("sample_id", "split", "messages", "tools", "target", "batching")
    }
    digest = hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {"sample_id": sample["sample_id"], "payload_sha256": digest}
