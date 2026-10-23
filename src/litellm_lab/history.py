"""Local metrics-only SQLite history."""

from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from pathlib import Path

FIELDS = ("started_at", "model", "base_url", "requests", "concurrency", "max_tokens",
          "stream", "succeeded", "failed", "wall_seconds", "requests_per_second",
          "completion_tokens", "mean_latency_seconds", "p50_latency_seconds",
          "p95_latency_seconds", "mean_ttft_seconds", "p50_ttft_seconds", "prompt_sha256")


def default_db_path():
    return Path(os.getenv("LITELLM_LAB_DB", "~/.litellm-lab/history.sqlite3")).expanduser()


def connect(path=None):
    target = Path(path).expanduser() if path else default_db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(target, timeout=5)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS benchmark_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL, model TEXT NOT NULL,
        base_url TEXT NOT NULL, requests INTEGER NOT NULL, concurrency INTEGER NOT NULL,
        max_tokens INTEGER NOT NULL, stream INTEGER NOT NULL, succeeded INTEGER NOT NULL,
        failed INTEGER NOT NULL, wall_seconds REAL NOT NULL, requests_per_second REAL NOT NULL,
        completion_tokens INTEGER NOT NULL, mean_latency_seconds REAL, p50_latency_seconds REAL,
        p95_latency_seconds REAL, mean_ttft_seconds REAL, p50_ttft_seconds REAL,
        prompt_sha256 TEXT NOT NULL)""")
    db.commit()
    return db


def save(summary, path=None):
    cols = ", ".join(FIELDS)
    marks = ", ".join("?" for _ in FIELDS)
    vals = [int(summary.get(k, False)) if k == "stream" else summary.get(k) for k in FIELDS]
    with connect(path) as db:
        db.execute("INSERT INTO benchmark_runs (" + cols + ") VALUES (" + marks + ")", vals)


def load(path=None, limit=5000):
    target = Path(path).expanduser() if path else default_db_path()
    if not target.exists():
        return []
    db = sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute("SELECT * FROM benchmark_runs ORDER BY started_at DESC, id DESC LIMIT ?",
                          (max(1, min(int(limit), 50000)),)).fetchall()
        return [dict(row) for row in rows]
    finally:
        db.close()


def encode(rows, fmt):
    if fmt == "json":
        return json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=["id", *FIELDS], extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()
