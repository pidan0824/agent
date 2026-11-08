"""Run a configured Customer/Seller Episode on an isolated Lite market store."""

import argparse
from collections import defaultdict
from contextlib import ExitStack, closing, contextmanager
import json
import math
from pathlib import Path
import time
from agent_simulation.logs import model_totals, write_report

from agent_simulation.data.common import require, rows, sha256
from agent_simulation.data.catalog import Catalog
from agent_simulation.data.catalog import DataRepository
from agent_simulation.data.history_records import seller_history_policy
from agent_simulation.inference.local.checkpoints import verify_checkpoint
from agent_simulation.inference.local.qwen import DEFAULT_REVISION, QwenLocal
from agent_simulation.inference.requests import load_tokenizer
from agent_simulation.agents.memory.reflection import SummaryGenerator
from agent_simulation.environment.market import MarketEnvironment
from agent_simulation.simulation.episode import Episode
from agent_simulation.agents.customer import CustomerAgent
from agent_simulation.agents.seller import SellerAgent


@contextmanager
def load_default_model(args):
    """Own the shared inference process for the lifetime of one simulation run."""
    if getattr(args, 'inference_backend', 'transformers') == 'vllm':
        from agent_simulation.inference.vllm import server_client
        require(not args.customer_adapter and not args.seller_adapter,
                'Use served model names for vLLM adapters, not local adapter paths')
        tokenizer = load_tokenizer(str(args.model), revision=args.revision, local_files_only=True)
        with closing(server_client(args, tokenizer)) as backend:
            checkpoint = backend.check_models()
            yield backend, checkpoint
        return
    checkpoint = verify_checkpoint(args.model, args.revision) if getattr(args, "verify", False) else None
    from agent_simulation.inference.local.service import LocalInferenceService
    tokenizer = load_tokenizer(str(args.model), revision=args.revision, local_files_only=True)
    adapters = {role: path for role, path in (("customer", args.customer_adapter),
                                              ("seller", args.seller_adapter)) if path}
    service = LocalInferenceService(
        QwenLocal, dict(model_path=str(args.model), revision=args.revision, device=args.device,
                        dtype=args.dtype, max_new_tokens=args.max_new_tokens,
                        log_path=args.output / "model_calls.jsonl", role_adapters=adapters,
                        checkpoint=checkpoint,
                        prefix_cache_tokens=getattr(args, "prefix_cache_tokens", 2048),
                        prefix_cache_entries=getattr(args, "prefix_cache_entries", 4)),
        concurrency=args.max_concurrent_requests, timeout=args.request_timeout,
    )
    with closing(service):
        service.tokenizer = tokenizer
        service.max_new_tokens = args.max_new_tokens
        yield service, checkpoint


def _save_market_state(output, runtime, report):
    final = runtime.snapshot()
    write_report(output / "final_state.json", final)
    report.update(
        clock=final.get("clock"),
        feedback_events=len(final["feedback"]),
        price_actions=len(final["price_actions"]),
        long_term_entries={k: len(v) for k, v in final["long_term"].items()},
        tool_calls=dict(runtime.db.execute("SELECT name,count(*) FROM calls GROUP BY name")),
        agent_tool_calls=runtime.logs.tool_counts(),
    )


def _record_model_totals(output, report, injected):
    raw_path = output / "model_calls.jsonl"
    if raw_path.exists():
        report.update(model_totals(raw_path))
        report["real_model_inference"] = None if injected else report["model_calls"] > 0
    elif injected:
        report["real_model_inference"] = None


def _finish_run(output, report, started, model_resources, runtime, catalog, injected, primary):
    """Attempt each finalization step even if a previous step failed."""
    errors = []

    def attempt(stage, operation, *args):
        try:
            operation(*args)
        except BaseException as error:
            detail = f"{stage}: {type(error).__name__}: {error}"
            report.setdefault("finalization_errors", []).append(detail)
            if primary is None and not errors:
                report.update(status="failed", error=detail)
            else:
                (primary if primary is not None else errors[0]).add_note(detail)
            errors.append(error)

    attempt("close model", model_resources.close)
    if runtime is not None:
        attempt("save market state", _save_market_state, output, runtime, report)
        attempt("close market", runtime.close)
    attempt("summarize model calls", _record_model_totals, output, report, injected)
    if catalog is not None:
        attempt("close catalog", catalog.close)
    report["seconds"] = time.monotonic() - started
    attempt("write final report", write_report, output / "report.json", report)
    if primary is None and errors:
        raise errors[0]
    if primary is None:
        print(json.dumps(report, ensure_ascii=False), flush=True)


def run(args, *, load_model=None):
    """Load (client, checkpoint report); Agent workflows use only the client interface."""
    if load_model is None and getattr(args, "inference_backend", "transformers") != "vllm":
        require(all(getattr(args, name, None) is None
                    for name in ("served_model", "customer_served_model", "seller_served_model")),
                "Served model names require --inference-backend vllm")
    require(args.max_concurrent_requests > 0, "Expected positive max concurrent requests")
    require(math.isfinite(args.request_timeout) and args.request_timeout > 0,
            "Expected a finite positive request timeout")
    require(not args.output.exists(), "Use a new output directory for each independent run")
    args.output.mkdir(parents=True)
    report = {
        "status": "starting",
        "real_model_inference": False,
        "training": False,
        "search_backend": args.search_backend,
        "behavior_validated": False,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    write_report(args.output / "report.json", report)
    started = time.monotonic()
    catalog = runtime = model = None
    primary = None
    model_resources = ExitStack()
    try:
        search_pipeline = None
        if args.search_backend == "kuaisearch":
            from agent_simulation.platform.pipeline import load_pipeline

            require(args.platform_assets is not None, "KuaiSearch requires --platform-assets")
            search_pipeline = load_pipeline(args.platform_assets)
            search_config = {
                "backend": "kuaisearch",
                "candidate_size": args.candidate_size,
                "pipeline_id": search_pipeline.identity,
                "reranker_id": search_pipeline.reranker_id,
            }
            report["platform_assets_sha256"] = sha256(args.platform_assets)
        else:
            require(
                args.platform_assets is None,
                "--platform-assets requires --search-backend kuaisearch",
            )
            search_config = (
                {"backend": "random", "seed": args.seed, "candidate_size": args.candidate_size}
                if args.search_backend == "random"
                else {"backend": "bm25"}
            )
        catalog = Catalog(args.catalog)
        seller_ids = args.sellers or [args.seller]
        require(len(set(seller_ids)) == len(seller_ids), "Duplicate sellers")
        history = rows(
            catalog.connection,
            "SELECT * FROM seller_history WHERE seller_id IN (SELECT unnest(?)) AND time_index<? ORDER BY time_index,item_id",
            [seller_ids, args.cutoff],
        )
        require(
            {r["seller_id"] for r in history} == set(seller_ids),
            "A selected shop has no history before this cutoff",
        )
        cache = defaultdict(list)
        for record in history:
            cache[record["item_id"]].append(record)
        data = DataRepository(
            catalog.connection,
            catalog.config,
            args.cutoff,
            history_points=seller_history_policy()["observations_per_item"],
        )
        data._histories = dict(cache)
        require(len(set(args.customers)) == len(args.customers), "Duplicate customers")
        runtime = MarketEnvironment.create(
            args.output / "runtime.sqlite",
            data,
            sorted(cache),
            {u: [] for u in args.customers},
            top_k=args.top_k,
            catalog=catalog,
            search_config=search_config,
            search_pipeline=search_pipeline,
        )
        if args.history_retrieval == "bge":
            runtime.inputs.use_semantic_history()
        else:
            runtime.inputs.history_embedder = None
        initial = runtime.snapshot()
        write_report(args.output / "initial_state.json", initial)
        report.update(
            products=len(initial["products"]),
            sellers=sorted({p["seller_id"] for p in initial["products"].values()}),
            excluded_items_without_initial_price=initial["excluded_items_without_initial_price"],
            customers=args.customers,
            catalog_manifest_sha256=sha256(catalog.directory / "manifest.json"),
        )
        print("Loading model", flush=True)
        if load_model is None:
            model, report["checkpoint"] = model_resources.enter_context(load_default_model(args))
        else:
            model, report["checkpoint"] = load_model(args)
            close = getattr(model, "close", None)
            if callable(close):
                model_resources.callback(close)
        report.update(model_id=model.generator_id, status="running")
        write_report(args.output / "report.json", report)
        print("Loaded", model.generator_id, flush=True)
        memory = {"generate": SummaryGenerator(model), "model_id": model.generator_id}
        seller = SellerAgent(
            runtime,
            model,
            model.tokenizer,
            summary_models=memory,
            price_batch_size=getattr(args, "seller_price_batch_size", 1),
            price_concurrency=args.max_concurrent_requests,
        )
        customer = CustomerAgent(
            runtime,
            model,
            model.tokenizer,
            summary_models=memory,
            response_concurrency=args.max_concurrent_requests,
            response_batch_size=args.customer_response_batch_size,
        )
        if args.mode == "episode":
            episode = Episode(
                runtime, seller, customer, tick_size=args.tick_size, max_ticks=args.ticks,
                max_concurrent_requests=args.max_concurrent_requests,
            )
            episode.run(progress=lambda tick, status: print("Tick", tick, status, flush=True))
        else:
            # Isolate Customer integration at initialized prices, without scripted Seller actions.
            for actor in args.customers:
                print("Customer component", actor, "started", flush=True)
                customer.run(actor)
                print("Customer component", actor, "complete", flush=True)
        final = runtime.snapshot()
        require(
            all(
                final["customers"][str(u)]["observed_history"]
                == initial["customers"][str(u)]["observed_history"]
                for u in args.customers
            ),
            "Initialized real history changed",
        )
        if args.mode == "customer":
            require(not final["pending"], "Customer component left pending requests")
        report.update(status="complete", initialized_history_unchanged=True)
    except BaseException as error:
        primary = error
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _finish_run(args.output, report, started, model_resources, runtime, catalog,
                    model is not None and load_model is not None, primary)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    from agent_simulation.inference.vllm import add_server_arguments
    add_server_arguments(p)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True,
                   help="Local Qwen checkpoint; with vLLM, the matching local tokenizer directory")
    p.add_argument("--customer-adapter", type=Path, help="Customer Query/Response LoRA directory")
    p.add_argument("--seller-adapter", type=Path, help="Seller pricing LoRA directory")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--verify", action="store_true", help="Verify local model file hashes before loading")
    p.add_argument("--mode", choices=["episode", "customer"], default="episode")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    sellers = p.add_mutually_exclusive_group()
    sellers.add_argument("--seller", type=int, default=108629)
    sellers.add_argument(
        "--sellers",
        type=int,
        nargs="+",
        help="Register every observed product of each selected shop",
    )
    p.add_argument("--customers", type=int, nargs="+", required=True)
    p.add_argument("--cutoff", type=int, default=850769)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--search-backend", choices=["bm25", "random", "kuaisearch"], default="random")
    p.add_argument(
        "--platform-assets",
        type=Path,
        help="Manifest of trained official KuaiSearch assets; never trains or downloads",
    )
    p.add_argument("--candidate-size", type=int, default=20)
    p.add_argument(
        "--history-retrieval",
        choices=("structured", "bge"),
        default="bge",
        help="Fuse local Chinese BGE and structured history ranks; structured is an explicit ablation",
    )
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--tick-size", type=int, default=1)
    p.add_argument("--ticks", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    from agent_simulation.environment.batching import CUSTOMER_RESPONSE_BATCH_SIZE
    p.add_argument("--customer-response-batch-size", type=int, default=CUSTOMER_RESPONSE_BATCH_SIZE,
                   help="Items per Customer response; defaults to inference/model_policy.json")
    p.add_argument("--seller-price-batch-size", type=int, choices=range(1, 17), default=1,
                   help="Explicit targets per decision; default 1 matches single-product SFT")
    p.add_argument("--max-concurrent-requests", type=int, default=2,
                   help="Global model concurrency per phase, including Seller targets and Customer response batches; also the local Qwen batch limit")
    p.add_argument("--prefix-cache-tokens", type=int, default=2048,
                   help="Maximum exact prefix tokens per cached KV entry; 0 disables reuse")
    p.add_argument("--prefix-cache-entries", type=int, default=4,
                   help="Maximum model-local prefix KV entries, isolated by adapter")
    p.add_argument("--request-timeout", type=float, default=120,
                   help="Per inference request deadline in seconds, including the worker queue")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
