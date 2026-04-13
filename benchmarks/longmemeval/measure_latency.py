#!/usr/bin/env python3
"""Measure Memento recall latency on LongMemEval oracle.

For each question:
1. Build a fresh MemoryStore with that question's haystack sessions (ingestion
   time is NOT measured — we only care about retrieval).
2. Call store.recall() 5 times with warm cache, recording each call's wall time.
3. Record the minimum of those 5 as the question's recall latency (cache-warm,
   best-effort measurement of steady-state cost).

Prints percentile stats over all questions. Ingestion time is reported
separately for reference.

Usage:
    python measure_latency.py --variant oracle --sample 100
    python measure_latency.py --variant oracle  # full 500
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from memento import MemoryStore  # noqa: E402
from memento.config import (  # noqa: E402
    ConsolidationConfig,
    MementoConfig,
    RetrievalConfig,
)

from run_benchmark import (  # noqa: E402
    _stratified_sample,
    ingest_haystack,
    load_dataset,
)


def create_store() -> MemoryStore:
    return MemoryStore(
        MementoConfig(
            db_path=Path(":memory:"),
            retrieval=RetrievalConfig(default_token_budget=4000, max_hop_depth=3),
            consolidation=ConsolidationConfig(
                decay_interval_ingestions=999_999,
                full_interval_ingestions=999_999,
            ),
        )
    )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round(p / 100 * (len(values) - 1)))
    return values[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["oracle"], default="oracle")
    parser.add_argument("--sample", type=int, default=None,
                        help="Use a stratified sample of N questions (default: all 500)")
    parser.add_argument("--warm-calls", type=int, default=5,
                        help="Warm calls per question, min reported (default: 5)")
    args = parser.parse_args()

    dataset = load_dataset(args.variant)
    if args.sample and args.sample < len(dataset):
        dataset = _stratified_sample(dataset, args.sample)

    print(f"\nMemento recall latency — {args.variant}, {len(dataset)} questions")
    print(f"Reporting min of {args.warm_calls} warm calls per question\n")

    ingest_times: list[float] = []
    recall_times: list[float] = []

    for i, entry in enumerate(dataset):
        sessions = entry["haystack_sessions"]
        dates = entry["haystack_dates"]
        question = entry["question"]
        current_date = entry.get("question_date", "")
        as_of = (
            f"{current_date}T23:59:59+00:00"
            if current_date and "T" not in current_date
            else current_date
        )

        store = create_store()
        try:
            t_in = time.perf_counter()
            ingest_haystack(store, sessions, dates, progress=False)
            ingest_times.append(time.perf_counter() - t_in)

            # Warm the cache with one call, then time N calls and take the min
            store.recall(question, token_budget=4000, as_of=as_of or None)

            per_call = []
            for _ in range(args.warm_calls):
                t0 = time.perf_counter()
                store.recall(question, token_budget=4000, as_of=as_of or None)
                per_call.append(time.perf_counter() - t0)
            recall_times.append(min(per_call))
        finally:
            store.close()

        if (i + 1) % 25 == 0:
            rt_ms = [t * 1000 for t in recall_times]
            print(f"  {i+1}/{len(dataset)}  "
                  f"median={statistics.median(rt_ms):.1f}ms  "
                  f"p95={percentile(rt_ms, 95):.1f}ms",
                  flush=True)

    # ── Final stats ────────────────────────────────────────────
    rt_ms = [t * 1000 for t in recall_times]
    it_s = ingest_times

    print(f"\n{'='*60}")
    print(f"  MEMENTO RECALL LATENCY  (ms, over {len(rt_ms)} questions)")
    print(f"{'='*60}")
    print(f"  min       {min(rt_ms):.1f}")
    print(f"  p50       {percentile(rt_ms, 50):.1f}")
    print(f"  mean      {statistics.mean(rt_ms):.1f}")
    print(f"  p90       {percentile(rt_ms, 90):.1f}")
    print(f"  p95       {percentile(rt_ms, 95):.1f}")
    print(f"  p99       {percentile(rt_ms, 99):.1f}")
    print(f"  max       {max(rt_ms):.1f}")
    print()
    print(f"  INGESTION TIME (s per question, informational)")
    print(f"  median    {statistics.median(it_s):.2f}")
    print(f"  mean      {statistics.mean(it_s):.2f}")
    print(f"  max       {max(it_s):.2f}")
    print(f"{'='*60}\n")

    # Also dump raw numbers for the paper
    out = {
        "variant": args.variant,
        "questions": len(rt_ms),
        "warm_calls": args.warm_calls,
        "recall_ms": {
            "min": min(rt_ms),
            "p50": percentile(rt_ms, 50),
            "mean": statistics.mean(rt_ms),
            "p90": percentile(rt_ms, 90),
            "p95": percentile(rt_ms, 95),
            "p99": percentile(rt_ms, 99),
            "max": max(rt_ms),
        },
        "ingest_seconds": {
            "median": statistics.median(it_s),
            "mean": statistics.mean(it_s),
            "max": max(it_s),
        },
    }
    out_path = Path("latency_results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Raw numbers written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
