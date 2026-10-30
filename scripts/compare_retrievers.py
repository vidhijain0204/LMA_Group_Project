
"""Compare TF-IDF, BM25, dense, and hybrid retrieval.

Uses the same annotated evaluation queries and corpus for all methods.

Run from the project root:
    python -m scripts.compare_retrievers --eval data/common/retrieval_eval.csv --k 5
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

from agents.course_retriever import CourseRetriever


METHODS = ("tfidf", "bm25", "dense", "hybrid")

def parse_gold(value):
    return {
        item.strip()
        for item in (value or "").split(";")
        if item.strip()
    }


def reciprocal_rank(results, gold):
    for rank, result in enumerate(results, start=1):
        if result.get("chunk_id") in gold:
            return 1.0 / rank
    return 0.0


def recall_at_k(results, gold, k):
    if not gold:
        return None

    retrieved = {
        result.get("chunk_id")
        for result in results[:k]
    }

    return len(retrieved & gold) / len(gold)


def main():
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF, BM25, and dense retrieval."
    )

    parser.add_argument(
        "--eval",
        default="data/common/retrieval_eval.csv",
    )
    parser.add_argument(
        "--corpus",
        default="data/common/corpus.jsonl",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output",
        default="reports/retrieval_comparison_all_methods.csv",
    )

    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k must be >= 1")

    # Load evaluation queries.
    with Path(args.eval).open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(
            f"No evaluation questions found in {args.eval}"
        )

    retriever = CourseRetriever(args.corpus)

    # Warm up dense retrieval before measuring query latency.
    # This includes model loading and initial corpus embedding generation
    # if the embedding cache does not already exist.
    print("Initializing dense retrieval...")
    warmup_start = time.perf_counter()

    retriever.search(
        "embedding model initialization",
        method="dense",
        limit=1,
    )

    dense_initialization_ms = (
        time.perf_counter() - warmup_start
    ) * 1000

    print(
        f"Dense initialization time: "
        f"{dense_initialization_ms:.3f} ms"
    )

    output_rows = []
    summary = {}

    for method in METHODS:
        reciprocal_ranks = []
        recalls = []
        latencies = []

        for row in rows:
            query = (row.get("query") or "").strip()

            if not query:
                continue

            # Measure retrieval after initialization.
            start = time.perf_counter()

            results = retriever.search(
                query,
                limit=args.k,
                method=method,
            )

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            latencies.append(latency_ms)

            gold = parse_gold(
                row.get("relevant_chunk_ids", "")
            )

            annotated = bool(gold)

            if annotated:
                rr = reciprocal_rank(results, gold)
                recall = recall_at_k(results, gold, args.k)

                reciprocal_ranks.append(rr)
                recalls.append(recall)
            else:
                rr = None
                recall = None

            output_rows.append({
                "query_id": row.get("query_id", ""),
                "method": method,
                "query": query,
                "gold_chunk_ids": ";".join(sorted(gold)),
                "top_chunk_ids": ";".join(
                    result.get("chunk_id", "")
                    for result in results
                ),
                "top_scores": ";".join(
                    f"{result.get('score', 0.0):.6f}"
                    for result in results
                ),
                "first_result_text": (
                    results[0].get("text", "")
                    if results else ""
                ),
                "first_result_source": (
                    results[0].get("source", {}).get("title", "")
                    if results else ""
                ),
                "latency_ms": f"{latency_ms:.3f}",
                f"recall_at_{args.k}": (
                    "" if recall is None else f"{recall:.4f}"
                ),
                "reciprocal_rank": (
                    "" if rr is None else f"{rr:.4f}"
                ),
                "annotated": str(annotated).lower(),
            })

        summary[method] = {
            "total": len(latencies),
            "annotated": len(recalls),
            "unannotated": len(latencies) - len(recalls),
            "recall": (
                statistics.mean(recalls)
                if recalls else None
            ),
            "mrr": (
                statistics.mean(reciprocal_ranks)
                if reciprocal_ranks else None
            ),
            "mean_latency": (
                statistics.mean(latencies)
                if latencies else None
            ),
            "median_latency": (
                statistics.median(latencies)
                if latencies else None
            ),
        }

    # Save detailed query-level results.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "query_id",
        "method",
        "query",
        "gold_chunk_ids",
        "top_chunk_ids",
        "top_scores",
        "first_result_text",
        "first_result_source",
        "latency_ms",
        f"recall_at_{args.k}",
        "reciprocal_rank",
        "annotated",
    ]

    with output.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(output_rows)

    # Print aggregate comparison.
    print(f"\nDetailed results saved to: {output}")
    print(f"Evaluation queries: {len(rows)}")
    print(f"Top K: {args.k}")

    print("\nAGGREGATE COMPARISON")
    print("=" * 75)

    for method, result in summary.items():
        print(f"\n{method.upper()}")
        print("-" * 40)

        print(f"Total queries:       {result['total']}")
        print(f"Annotated queries:   {result['annotated']}")
        print(f"Unannotated queries: {result['unannotated']}")

        if result["recall"] is not None:
            print(
                f"Recall@{args.k}:           "
                f"{result['recall']:.4f}"
            )
            print(
                f"MRR:                 "
                f"{result['mrr']:.4f}"
            )
        else:
            print("Recall/MRR unavailable.")

        print(
            f"Mean latency:        "
            f"{result['mean_latency']:.3f} ms"
        )
        print(
            f"Median latency:      "
            f"{result['median_latency']:.3f} ms"
        )


if __name__ == "__main__":
    main()
