import json
import tempfile
import unittest
from pathlib import Path

import httpx

from agent_simulation.inference.agentic import AgenticClient
from agent_simulation.inference.client import ModelOutputError
from agent_simulation.inference.requests import ContextCapacityError


class Tokenizer:
    chat_template = "test"

    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(m.get("content") or "") for m in messages)))


class AgenticClientTest(unittest.TestCase):
    def client(self, message, reason="stop", **kwargs):
        self.requests = []

        def respond(request):
            self.requests.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [
                {"message": message, "finish_reason": reason}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 7}})

        client = AgenticClient("http://localhost/v1", "Qwen3-32B", Tokenizer(),
                               transport=httpx.MockTransport(respond), **kwargs)
        self.addCleanup(client.close)
        return client

    def test_native_tool_calls_and_multi_turn_messages_survive(self):
        call = {"id": "a", "type": "function", "function": {
            "name": "query_customer_history", "arguments": "{}"}}
        client = self.client({"role": "assistant", "content": None,
                              "tool_calls": [call]}, reason="tool_calls")
        messages = [{"role": "user", "content": "Choose"},
                    {"role": "assistant", "content": None, "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": "a", "content": "[]"}]
        tools = [{"type": "function", "function": {"name": "query_customer_history",
                  "parameters": {"type": "object", "properties": {}}}}]
        reply = client.complete({"messages": messages, "tools": tools}, call_id="test")
        self.assertEqual(reply["tool_calls"], [call])
        self.assertEqual(self.requests[0]["messages"], messages)
        self.assertEqual(self.requests[0]["tool_choice"], "auto")
        self.assertEqual(self.requests[0]["chat_template_kwargs"], {"enable_thinking": False})

    def test_truncated_response_is_logged_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.jsonl"
            client = self.client({"role": "assistant", "content": '{"query":'},
                                 reason="length", log_path=path)
            with self.assertRaises(ModelOutputError):
                client.complete({"messages": [{"role": "user", "content": "choose"}]},
                                call_id="partial")
            record = json.loads(path.read_text())
            self.assertFalse(record["finished"])
            self.assertEqual(record["output_tokens"], 7)
            self.assertEqual(record["finish_reason"], "length")

    def test_context_overflow_never_contacts_server(self):
        client = self.client({"role": "assistant", "content": "{}"})
        with self.assertRaises(ContextCapacityError):
            client.complete({"messages": [{"role": "user", "content": "x" * 33000}]},
                            call_id="overflow")
        self.assertEqual(self.requests, [])

    def test_resampled_seed_matches_logged_model_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calls.jsonl"
            client = self.client({"role": "assistant", "content": "{}"}, seed=7, log_path=path)
            client.seed = 8
            client.complete({"messages": [{"role": "user", "content": "choose"}]}, call_id="sample-2")
            record = json.loads(path.read_text())
            self.assertEqual(self.requests[0]["seed"], 8)
            self.assertEqual(record["seed"], 8)
            self.assertIn("seed=8", record["model"])

    def test_final_budget_retains_schema_but_disables_tool_choice(self):
        from agent_simulation.agents.agentic import AgenticOptions, budget_message
        client = self.client({"role": "assistant", "content": "{}"})
        tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
        client.complete({"messages": [{"role": "user", "content": "Choose"},
            budget_message(AgenticOptions(max_model_calls=1), available=False)], "tools": tools}, call_id="final")
        self.assertEqual(self.requests[0]["tools"], tools)
        self.assertEqual(self.requests[0]["tool_choice"], "none")

    def test_empty_and_invalid_envelopes_fail(self):
        for message, reason in [({"role": "assistant", "content": None}, "stop"),
                                ({"role": "assistant", "content": "{}"}, "tool_calls"),
                                ({"role": "user", "content": "{}"}, "stop")]:
            with self.subTest(message=message, reason=reason):
                client = self.client(message, reason)
                with self.assertRaises(ModelOutputError):
                    client.complete({"messages": [{"role": "user", "content": "choose"}]},
                                    call_id="invalid")


if __name__ == "__main__":
    unittest.main()
