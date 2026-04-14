#!/usr/bin/env python3
"""Diagnose wrong answers from a LongMemEval run.

For each question judged wrong (verdict: no), re-runs the retrieval step
only (no answer-model call) and prints side-by-side:

    QUESTION:       <the question>
    REFERENCE:      <the correct answer>
    HYPOTHESIS:     <what the system actually said>
    RETRIEVED:      <what Memento pulled from the knowledge graph>

After printing all failures, classifies each one heuristically:

    - RETRIEVAL_MISS:    the reference answer text is NOT in the retrieved context
    - ANSWER_ERROR:      the reference answer IS in the retrieved context (so the
                         LLM had the info but still answered wrong)
    - JUDGE_DISAGREEMENT: (inspect manually) the hypothesis looks semantically
                         close to the reference but the judge said no

Usage:
    python diagnose_failures.py --eval-file <results.jsonl.eval-gpt-4o>
                                 --variant s
                                 [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from memento import MemoryStore  # noqa: E402
from memento.config import (  # noqa: E402
    ConsolidationConfig,
    MementoConfig,
    RetrievalConfig,
)

from run_benchmark import ingest_haystack, load_dataset  # noqa: E402

logger = logging.getLogger(__name__)


def create_store() -> MemoryStore:
    return MemoryStore(
        MementoConfig(
            db_path=Path(":memory:"),
            retrieval=RetrievalConfig(
                default_token_budget=4000, max_hop_depth=3
            ),
            consolidation=ConsolidationConfig(
                decay_interval_ingestions=999_999,
                full_interval_ingestions=999_999,
            ),
        )
    )


def classify(
    hypothesis: str, reference: str, retrieved: str
) -> str:
    """Heuristic classification of why the answer was wrong."""
    ref_lower = reference.lower().strip()
    retrieved_lower = retrieved.lower()
    hyp_lower = hypothesis.lower()

    # Does the retrieved context actually contain the reference answer text?
    # This isn't perfect — the answer might be implied, not quoted verbatim —
    # but it's a decent signal for flat factual questions.
    ref_words = [w for w in ref_lower.split() if len(w) > 3]
    ref_in_retrieval = (
        ref_lower in retrieved_lower
        or (len(ref_words) > 0 and sum(w in retrieved_lower for w in ref_words) / len(ref_words) > 0.7)
    )
    ref_in_hypothesis = (
        ref_lower in hyp_lower
        or (len(ref_words) > 0 and sum(w in hyp_lower for w in ref_words) / len(ref_words) > 0.7)
    )

    if ref_in_hypothesis and ref_in_retrieval:
        return "JUDGE_DISAGREEMENT"
    if ref_in_retrieval:
        return "ANSWER_ERROR"
    return "RETRIEVAL_MISS"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", required=True, help="Path to .eval-gpt-4o JSONL")
    parser.add_argument("--variant", choices=["oracle", "s", "m"], required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N wrong answers")
    args = parser.parse_args()

    # Load eval results
    with open(args.eval_file, encoding="utf-8") as f:
        results = [json.loads(line) for line in f if line.strip()]

    wrong = [r for r in results if r.get("verdict") == "no"]
    if args.limit:
        wrong = wrong[: args.limit]

    print(f"\nLoaded {len(results)} results, {len(wrong)} judged wrong")
    print(f"Variant: {args.variant}\n")

    if not wrong:
        print("No wrong answers to diagnose.")
        return

    # Load reference dataset
    dataset = load_dataset(args.variant)
    ref_map = {e["question_id"]: e for e in dataset}

    classifications: dict[str, int] = {
        "RETRIEVAL_MISS": 0,
        "ANSWER_ERROR": 0,
        "JUDGE_DISAGREEMENT": 0,
    }

    for i, entry in enumerate(wrong):
        qid = entry["question_id"]
        hypothesis = entry["hypothesis"]
        ref = ref_map.get(qid)
        if not ref:
            print(f"  [{i+1}/{len(wrong)}] {qid}  SKIP — not in reference dataset")
            continue

        question = ref["question"]
        reference = ref["answer"]
        qtype = ref.get("question_type", "?")
        sessions = ref["haystack_sessions"]
        dates = ref["haystack_dates"]
        current_date = ref.get("question_date", "")
        as_of = (
            f"{current_date}T23:59:59+00:00"
            if current_date and "T" not in current_date
            else current_date
        )

        # Rebuild the memory store and run retrieval only
        print(f"\n{'='*70}")
        print(f"  [{i+1}/{len(wrong)}] {qid}  type={qtype}")
        print(f"{'='*70}")
        print(f"QUESTION:   {question}")
        print(f"REFERENCE:  {reference}")
        print(f"HYPOTHESIS: {hypothesis[:300]}{'...' if len(hypothesis) > 300 else ''}")

        store = create_store()
        try:
            ingest_haystack(store, sessions, dates, progress=False)
            memory = store.recall(question, token_budget=4000, as_of=as_of or None)
            retrieved = memory.text
        except Exception as e:
            print(f"RETRIEVAL FAILED: {e}")
            continue
        finally:
            store.close()

        # Truncate long retrieval output for readability
        retrieved_preview = retrieved[:1500] + "..." if len(retrieved) > 1500 else retrieved
        print(f"\nRETRIEVED ({len(retrieved)} chars):")
        print(retrieved_preview)

        verdict = classify(hypothesis, reference, retrieved)
        classifications[verdict] += 1
        print(f"\n==> CLASSIFIED AS: {verdict}")

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY — {len(wrong)} wrong answers")
    print(f"{'='*70}")
    for k, v in classifications.items():
        pct = v / len(wrong) * 100 if wrong else 0
        print(f"  {k:22} {v:>3} ({pct:.1f}%)")
    print(f"{'='*70}\n")

    # Interpretation hint
    print("Interpretation:")
    print("  RETRIEVAL_MISS      → the graph didn't surface the right evidence")
    print("                        (retrieval problem — may need prompt tuning,")
    print("                         wider recall, or better entity extraction)")
    print("  ANSWER_ERROR        → retrieval was good but the LLM got it wrong")
    print("                        (answer-model problem — try a different model)")
    print("  JUDGE_DISAGREEMENT  → hypothesis looks right but judge said no")
    print("                        (inspect manually — may be a judging artifact)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
