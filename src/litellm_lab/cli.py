"""Ollama discovery, model comparison, streaming and benchmark history."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from openai import AsyncOpenAI
from litellm_lab.history import encode, load, save


def endpoint(value, no_path=False):
    p = urlsplit(value.strip())
    if p.scheme not in ("http", "https") or not p.hostname or p.username or p.password or p.query or p.fragment:
        raise ValueError("Use an absolute HTTP(S) URL without credentials, query, or fragment.")
    if no_path and p.path not in ("", "/"):
        raise ValueError("Ollama URL must not contain a path.")
    return urlunsplit((p.scheme, p.netloc, p.path.rstrip("/"), "", ""))


def discover(url, timeout):
    url = endpoint(url, no_path=True)
    try:
        req = Request(url + "/api/tags", headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("Ollama query failed (" + type(exc).__name__ + ").") from None
    found = []
    for item in payload.get("models", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        found.append({"name": item["name"], "size_bytes": item.get("size"),
                      "parameter_size": details.get("parameter_size"),
                      "quantization": details.get("quantization_level")})
    return sorted(found, key=lambda row: row["name"].casefold())


def pct(values, q):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]


async def benchmark(args, model):
    key = args.api_key or os.getenv("LITELLM_API_KEY")
    if not key:
        raise RuntimeError("Set LITELLM_API_KEY or pass --api-key.")
    base = endpoint(args.base_url)
    client = AsyncOpenAI(api_key=key, base_url=base.rstrip("/") + "/", timeout=args.timeout, max_retries=0)
    gate = asyncio.Semaphore(args.concurrency)
    started = datetime.now(timezone.utc).isoformat()
    digest = hashlib.sha256(args.prompt.encode("utf-8")).hexdigest()
    t0 = time.perf_counter()

    async def one(i):
        async with gate:
            start = time.perf_counter()
            ttft = None
            try:
                request = dict(model=model, messages=[{"role": "user", "content": args.prompt}],
                               max_tokens=args.max_tokens, temperature=0)
                tokens, response_id = None, None
                if args.stream:
                    stream = await client.chat.completions.create(**request, stream=True,
                                                                  stream_options={"include_usage": True})
                    async for chunk in stream:
                        response_id = response_id or chunk.id
                        if chunk.choices and chunk.choices[0].delta.content and ttft is None:
                            ttft = time.perf_counter() - start
                        if chunk.usage and chunk.usage.completion_tokens is not None:
                            tokens = chunk.usage.completion_tokens
                else:
                    response = await client.chat.completions.create(**request)
                    response_id = response.id
                    tokens = response.usage.completion_tokens if response.usage else None
                elapsed = time.perf_counter() - start
                return {"type": "request", "model": model, "index": i, "ok": True,
                        "latency_seconds": round(elapsed, 6),
                        "first_token_seconds": round(ttft, 6) if ttft is not None else None,
                        "completion_tokens": tokens,
                        "tokens_per_second": round(tokens / elapsed, 3) if tokens and elapsed else None,
                        "response_id": response_id}
            except Exception as exc:
                return {"type": "request_error", "model": model, "index": i, "ok": False,
                        "latency_seconds": round(time.perf_counter() - start, 6),
                        "first_token_seconds": round(ttft, 6) if ttft is not None else None,
                        "error_type": type(exc).__name__}

    try:
        results = await asyncio.gather(*(one(i) for i in range(1, args.requests + 1)))
    finally:
        await client.close()

    elapsed = time.perf_counter() - t0
    ok = [x for x in results if x["ok"]]
    latency = [x["latency_seconds"] for x in ok]
    ttft_values = [x["first_token_seconds"] for x in ok if x["first_token_seconds"] is not None]
    summary = {"type": "summary", "started_at": started, "model": model, "base_url": base,
               "requests": args.requests, "concurrency": args.concurrency, "max_tokens": args.max_tokens,
               "stream": args.stream, "succeeded": len(ok), "failed": len(results) - len(ok),
               "wall_seconds": round(elapsed, 6),
               "requests_per_second": round(len(ok) / elapsed, 3) if elapsed else 0,
               "completion_tokens": sum(x["completion_tokens"] or 0 for x in ok),
               "mean_latency_seconds": round(statistics.mean(latency), 6) if latency else None,
               "p50_latency_seconds": round(pct(latency, .5), 6) if latency else None,
               "p95_latency_seconds": round(pct(latency, .95), 6) if latency else None,
               "mean_ttft_seconds": round(statistics.mean(ttft_values), 6) if ttft_values else None,
               "p50_ttft_seconds": round(pct(ttft_values, .5), 6) if ttft_values else None,
               "prompt_sha256": digest}
    return results, summary


def check_args(a):
    for name in ("requests", "concurrency", "max_tokens"):
        if getattr(a, name) < 1:
            raise SystemExit("--" + name.replace("_", "-") + " must be positive")
    if a.requests > 10000 or a.concurrency > 256 or a.max_tokens > 32768:
        raise SystemExit("Limits: requests <= 10000, concurrency <= 256, max tokens <= 32768")
    if not 0 < a.timeout <= 3600:
        raise SystemExit("Timeout must be greater than 0 and at most 3600 seconds")


def emit(events, target):
    content = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in events)
    if target:
        path = Path(target).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print("Wrote metrics to " + str(path))
    else:
        sys.stdout.write(content)


async def run(args):
    check_args(args)
    aliases = args.models if args.command == "compare" else [args.model]
    aliases = list(dict.fromkeys(x.strip() for x in aliases if x.strip()))
    if not aliases:
        print("Provide at least one model alias.", file=sys.stderr)
        return 2
    all_events, status = [], 0
    for alias in aliases:
        try:
            records, summary = await benchmark(args, alias)
        except (RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        save(summary, args.history_db)
        all_events.extend(records)
        all_events.append(summary)
        if summary["failed"]:
            status = 1
    emit(all_events, args.output)
    return status


def add_benchmark_options(p):
    p.add_argument("--base-url", default="http://127.0.0.1:4000/v1")
    p.add_argument("--api-key", help="Prefer LITELLM_API_KEY environment variable.")
    p.add_argument("--requests", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--prompt", default="In two short sentences, explain what a local language model is.")
    p.add_argument("--stream", action="store_true", help="Measure time to first non-empty content chunk.")
    p.add_argument("--output", help="Optional JSONL file.")
    p.add_argument("--history-db", help="SQLite path (default ~/.litellm-lab/history.sqlite3).")


def main():
    parser = argparse.ArgumentParser(prog="litellm-lab")
    subs = parser.add_subparsers(dest="command", required=True)
    bench = subs.add_parser("bench", help="Benchmark one configured model.")
    add_benchmark_options(bench)
    bench.add_argument("--model", default="local-llama")
    compare = subs.add_parser("compare", help="Compare model aliases sequentially with identical settings.")
    add_benchmark_options(compare)
    compare.add_argument("--models", nargs="+", required=True)
    models = subs.add_parser("models", help="Discover models on a local Ollama server.")
    models.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    models.add_argument("--timeout", type=float, default=5)
    hist = subs.add_parser("history", help="Print recent benchmark summaries.")
    hist.add_argument("--history-db")
    hist.add_argument("--limit", type=int, default=20)
    export = subs.add_parser("export-history", help="Export saved summaries.")
    export.add_argument("--history-db")
    export.add_argument("--format", choices=("csv", "json"), default="csv")
    export.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command in ("bench", "compare"):
        raise SystemExit(asyncio.run(run(args)))
    if args.command == "models":
        if not 0 < args.timeout <= 60:
            raise SystemExit("Timeout must be in (0, 60].")
        try:
            rows = discover(args.ollama_url, args.timeout)
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc))
        print(json.dumps(rows, indent=2, ensure_ascii=False) if rows else
              "No Ollama models found. Pull one with: ollama pull llama3.2")
        return
    if args.command == "history":
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        print(json.dumps(load(args.history_db, args.limit), indent=2, ensure_ascii=False))
        return
    rows = load(args.history_db, 50000)
    path = Path(args.output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encode(rows, args.format), encoding="utf-8")
    print("Exported " + str(len(rows)) + " runs to " + str(path))


if __name__ == "__main__":
    main()
