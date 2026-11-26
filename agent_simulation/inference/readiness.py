"""Verify actual ordinary and JSON generation before accepting workload."""
import argparse
import json
import time
import urllib.request


def check(url, model, *, timeout=300):
    results = []
    for structured in (False, True):
        body = {"model": model, "messages": [
            {"role": "user", "content": 'Return exactly {"ok":true}.'}],
            "max_tokens": 32, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
        if structured:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "ready", "strict": True, "schema": {
                    "type": "object", "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"], "additionalProperties": False}}}
        started = time.monotonic()
        request = urllib.request.Request(url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        choice, = result["choices"]
        if choice["finish_reason"] != "stop" or json.loads(choice["message"]["content"]) != {"ok": True}:
            raise RuntimeError("Generation readiness probe returned invalid output")
        results.append({"structured": structured, "seconds": time.monotonic() - started,
                        "usage": result.get("usage")})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    print(json.dumps(check(args.url, args.model, timeout=args.timeout), ensure_ascii=False))


if __name__ == "__main__":
    main()
