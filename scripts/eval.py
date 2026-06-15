#!/usr/bin/env python
"""DocRAG evaluation CLI.

Examples:
    python scripts/eval.py                      # retrieval metrics (no LLM needed)
    python scripts/eval.py --samples data/sample --dataset data/eval/qa.json
    python scripts/eval.py --ragas              # also run RAGAS (needs LLM + [eval])
"""

from __future__ import annotations

import argparse
from pathlib import Path

from docrag.eval.dataset import DEFAULT_DATASET, load_dataset
from docrag.eval.harness import build_sample_retriever, evaluate_retrieval, run_ragas


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DocRAG retrieval quality.")
    parser.add_argument("--samples", default="data/sample", help="Directory of documents.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Gold QA JSON file.")
    parser.add_argument("--ragas", action="store_true", help="Also run RAGAS generation metrics.")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    retriever, indexed = build_sample_retriever(args.samples)
    report = evaluate_retrieval(retriever, dataset, indexed_chunks=indexed)

    print("\n=== Retrieval quality ===\n")
    print(report.as_markdown())
    print()
    for row in report.rows:
        status = "✓" if row["rr"] == 1.0 else ("·" if row["rr"] else "✗")
        print(f"  {status} rr={row['rr']:<5} {row['question']}")

    if args.ragas:
        print("\n=== RAGAS generation metrics ===\n")
        try:
            result = run_ragas(dataset, Path(args.samples))
            print(result)
        except RuntimeError as exc:
            print(f"Skipped: {exc}")


if __name__ == "__main__":
    main()
