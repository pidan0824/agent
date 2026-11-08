"""Export or watch committed Episode ticks from a read-only local market store."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit
from collections import defaultdict
from contextlib import closing


TEMPLATE = Path(__file__).with_name("templates") / "timeline.html"


def page(data, live=False, *, template=TEMPLATE):
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", r"\u003c")
    return (
        template.read_text().replace("__LIVE__", json.dumps(live)).replace("__EPISODE__", payload)
    )


def serve(store, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Host") not in (
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            ):
                self.send_error(403)
                return
            path = urlsplit(self.path).path
            if path not in ("/", "/episode"):
                self.send_error(404)
                return
            try:
                data = read_timeline(
                    store,
                    (
                        self.headers.get("If-None-Match", "").strip('"')
                        if path == "/episode"
                        else None
                    ),
                )
                if data is None:
                    self.send_response(304)
                    self.end_headers()
                    return
                html = path == "/"
                body = (page(data, True) if html else json.dumps(data, ensure_ascii=False)).encode()
            except (ValueError, KeyError, sqlite3.Error):
                self.send_error(503, "Episode snapshot unavailable")
                return
            self.send_response(200)
            self.send_header(
                "Content-Type", ("text/html" if html else "application/json") + "; charset=utf-8"
            )
            self.send_header("Cache-Control", "no-store")
            self.send_header("ETag", '"' + data["stamp"] + '"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"http://127.0.0.1:{server.server_port}/", flush=True)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--html", type=Path)
    mode.add_argument("--animation", type=Path, help="Export the offline animated replay")
    mode.add_argument("--serve", type=int, metavar="PORT")
    args = parser.parse_args()
    if args.serve is not None:
        serve(args.store, args.serve)
    else:
        output = args.animation or args.html
        if output.suffix.lower() != ".html":
            parser.error("Export to a .html file")
        template = TEMPLATE.with_name("animation.html") if args.animation else TEMPLATE
        output.write_text(page(read_timeline(args.store), template=template))


def read_timeline(store, previous=None):
    """Read one consistent snapshot, exposing only the events shown by the viewer."""
    with closing(sqlite3.connect(f"{store.resolve().as_uri()}?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        latest = db.execute(
            "SELECT tick,json_extract(trace,'$.status') FROM ticks ORDER BY tick DESC LIMIT 1"
        ).fetchone()
        stamp = f"{latest[0]}-{latest[1]}" if latest else "init"
        if stamp == previous:
            return None
        ticks = [json.loads(t[0]) for t in db.execute("SELECT trace FROM ticks ORDER BY tick")]
        state = json.loads(db.execute("SELECT value FROM state WHERE id=1").fetchone()[0])
        searches = {}
        for actor, value in db.execute(
            "SELECT actor,result FROM calls WHERE name='search_products' AND error IS NULL"
        ):
            listing = json.loads(value)
            searches[listing["request_id"]] = (actor, listing)
    changes = {(a["simulation_step"], a["item_id"]): a for a in state["price_actions"]}
    feedback = defaultdict(dict)
    for row in state["feedback"]:
        feedback[row["request_id"]][row["item_id"]] = row
    frames = []
    for tick in ticks:
        if tick["status"] != "complete":
            continue
        sellers = []
        for run in tick["sellers"]:
            applied = run.get("applied_result")
            prices = []
            if applied:
                for p in applied["updated"]:
                    a = changes[applied["simulation_step"], p["item_id"]]
                    prices.append(
                        {
                            "id": p["item_id"],
                            "title": state["products"][str(p["item_id"])]["item_title"],
                            "before": a["previous_price_raw"],
                            "after": a["price_raw"],
                        }
                    )
            sellers.append(
                {
                    "id": run["actor"],
                    "wait": run["result"].get("action") == "no_action",
                    "prices": prices,
                }
            )
        customers = []
        for run in tick["customers"]:
            actor = run["actor"]
            if "settled_result" not in run:
                action = run["result"]["action"]
                if action not in ("exit", "no_action"):
                    raise ValueError("Unknown Customer action in completed Tick")
                customers.append({"id": actor, "action": action})
                continue
            receipt = run["settled_result"]
            rid = receipt["request_id"]
            owner, listing = searches[rid]
            if owner != actor or listing["logical_time"] != tick["logical_time"]:
                raise ValueError("Search receipt belongs to a different actor or Tick")
            items = []
            for p in listing["displayed_items"]:
                f = feedback[rid][p["item_id"]]
                if (
                    f["settled_step"] != receipt["settled_step"]
                    or f["price_raw"] != p["price_raw"]
                    or f["seller_id"] != p["seller_id"]
                    or f["exposure_records"] != 1
                    or bool(f["click_records"]) != (p["item_id"] in receipt["clicked_item_ids"])
                    or bool(f["purchase_records"])
                    != (p["item_id"] in receipt["purchased_item_ids"])
                ):
                    raise ValueError("Displayed item and settlement disagree")
                items.append(
                    {
                        "id": p["item_id"],
                        "shop": p["seller_id"],
                        "title": p["item_title"],
                        "price": p["price_raw"],
                        "click": f["click_records"],
                        "buy": f["purchase_records"],
                    }
                )
            if len(items) != len(feedback[rid]):
                raise ValueError("Exposure count differs from displayed list size")
            customers.append(
                {
                    "id": actor,
                    "action": "search",
                    "request": rid,
                    "query": listing["query"],
                    "items": items,
                }
            )
        frames.append(
            {
                "tick": tick["tick"],
                "time": tick["logical_time"],
                "sellers": sellers,
                "customers": customers,
            }
        )
    return {
        "dataset": state["dataset"],
        "backend": state["search_config"]["backend"],
        "status": state["clock"]["status"],
        "end": state["clock"].get("termination_reason"),
        "stamp": stamp,
        "frames": frames,
    }


if __name__ == "__main__":
    main()
