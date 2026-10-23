"""Concurrent, metrics-only benchmark runner for an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a nearest-rank percentile; None when there are no successful samples."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


async def run_benchmark(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.getenv("LITELLM_API_KEY")
    if not api_key:
        print("Set --api-key or LITELLM_API_KEY before running the benchmark.", file=sys.stderr)
        return 2

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=args.base_url.rstrip("/") + "/",
        timeout=args.timeout,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    started_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    run_start = time.perf_counter()

    async def one_request(index: int) -> dict[str, Any]:
        async with semaphore:
            request_start = time.perf_counter()
            try:
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": args.prompt}],
                    max_tokens=args.max_tokens,
                    temperature=0,
                )
                elapsed = time.perf_counter() - request_start
                usage = response.usage
                completion_tokens = usage.completion_tokens if usage else None
                return {
                    "type": "request",
                    "index": index,
                    "ok": True,
                    "latency_seconds": round(elapsed, 6),
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                    "completion_tokens": completion_tokens,
                    "tokens_per_second": (
                        round(completion_tokens / elapsed, 3)
                        if completion_tokens is not None and elapsed > 0
                        else None
                    ),
                    "response_id": response.id,
                }
            except Exception as exc:  # keep the run going without saving provider error details
                return {
                    "type": "request",
                    "index": index,
                    "ok": False,
                    "latency_seconds": round(time.perf_counter() - request_start, 6),
                    "error_type": type(exc).__name__,
                }

    try:
        results = await asyncio.gather(*(one_request(i + 1) for i in range(args.requests)))
    finally:
        await client.close()

    total_seconds = time.perf_counter() - run_start
    successful = [item for item in results if item["ok"]]
    latencies = [item["latency_seconds"] for item in successful]
    total_tokens = sum(item["completion_tokens"] or 0 for item in successful)
    summary = {
        "type": "summary",
        "started_at": started_at,
        "base_url": args.base_url,
        "model": args.model,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "succeeded": len(successful),
        "failed": len(results) - len(successful),
        "wall_seconds": round(total_seconds, 6),
        "requests_per_second": round(len(successful) / total_seconds, 3) if total_seconds else 0,
        "completion_tokens": total_tokens,
        "mean_latency_seconds": round(statistics.mean(latencies), 6) if latencies else None,
        "p50_latency_seconds": round(percentile(latencies, 0.50), 6) if latencies else None,
        "p95_latency_seconds": round(percentile(latencies, 0.95), 6) if latencies else None,
    }

    output = [*results, summary]
    encoded = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in output)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
        print(f"Wrote metrics to {destination}")
    else:
        sys.stdout.write(encoded)

    return 0 if summary["failed"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="litellm-lab",
        description="Benchmark an OpenAI-compatible LiteLLM endpoint.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    bench = subparsers.add_parser("bench", help="Run concurrent chat-completion requests.")
    bench.add_argument("--base-url", default="http://127.0.0.1:4000/v1")
    bench.add_argument("--model", default="local-llama")
    bench.add_argument("--api-key", help="Gateway key; prefer LITELLM_API_KEY in the environment.")
    bench.add_argument("--requests", type=int, default=10)
    bench.add_argument("--concurrency", type=int, default=2)
    bench.add_argument("--max-tokens", type=int, default=128)
    bench.add_argument("--timeout", type=float, default=120.0)
    bench.add_argument("--prompt", default="In two short sentences, explain what a local language model is.")
    bench.add_argument("--output", help="Optional JSONL path. Defaults to stdout.")
    bench.set_defaults(handler=run_benchmark)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "bench":
        for name in ("requests", "concurrency", "max_tokens"):
            if getattr(args, name) < 1:
                raise SystemExit(f"--{name.replace('_', '-')} must be at least 1")
        if args.requests > 10000:
            raise SystemExit("--requests cannot exceed 10000")
        if args.concurrency > 256:
            raise SystemExit("--concurrency cannot exceed 256")
        if args.timeout <= 0:
            raise SystemExit("--timeout must be greater than 0")
        raise SystemExit(asyncio.run(args.handler(args)))


if __name__ == "__main__":
    main()
