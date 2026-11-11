
import csv
from collections import defaultdict
from pathlib import Path


INPUT_FILE = "reports/retrieval_comparison_all_methods.csv"


def parse_ids(value):
    return {
        item.strip()
        for item in (value or "").split(";")
        if item.strip()
    }


rows_by_query = defaultdict(dict)

with Path(INPUT_FILE).open(
    "r", encoding="utf-8", newline=""
) as file:
    for row in csv.DictReader(file):
        rows_by_query[row["query_id"]][row["method"]] = row


print("=" * 90)
print("DENSE RETRIEVAL ERROR ANALYSIS")
print("=" * 90)

for query_id, methods in rows_by_query.items():

    bm25 = methods.get("bm25")
    dense = methods.get("dense")

    if not bm25 or not dense:
        continue

    gold = parse_ids(bm25["gold_chunk_ids"])

    if not gold:
        continue

    bm25_ids = parse_ids(bm25["top_chunk_ids"])
    dense_ids = parse_ids(dense["top_chunk_ids"])

    bm25_hits = len(bm25_ids & gold)
    dense_hits = len(dense_ids & gold)

    if dense_hits < bm25_hits:
        print(f"\nQuery: {query_id}")
        print(f"Question: {dense['query']}")

        print(f"\nRelevant chunks: {sorted(gold)}")

        print(
            f"BM25 hits: {bm25_hits}/5 "
            f"-> {bm25['top_chunk_ids']}"
        )

        print(
            f"Dense hits: {dense_hits}/5 "
            f"-> {dense['top_chunk_ids']}"
        )

        print("\nDense first result:")
        print(dense["first_result_text"][:350])

        print("-" * 90)
