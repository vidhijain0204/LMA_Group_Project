
import csv
from collections import defaultdict
from pathlib import Path

INPUT_FILE = "reports/retrieval_comparison_rrf.csv"


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


def evaluate(row):
    gold = parse_ids(row["gold_chunk_ids"])
    retrieved = parse_ids(row["top_chunk_ids"])

    hits = len(gold & retrieved)

    reciprocal_rank = 0.0
    ranked_ids = [
        item.strip()
        for item in row["top_chunk_ids"].split(";")
        if item.strip()
    ]

    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in gold:
            reciprocal_rank = 1.0 / rank
            break

    return hits, reciprocal_rank


print("=" * 90)
print("HYBRID VS BM25: QUERY-LEVEL ANALYSIS")
print("=" * 90)

hybrid_better = []
bm25_better = []
ties = []

for query_id, methods in rows_by_query.items():

    bm25 = methods.get("bm25")
    hybrid = methods.get("hybrid")

    if not bm25 or not hybrid:
        continue

    gold = parse_ids(bm25["gold_chunk_ids"])

    if not gold:
        continue

    bm25_hits, bm25_mrr = evaluate(bm25)
    hybrid_hits, hybrid_mrr = evaluate(hybrid)

    comparison = {
        "query_id": query_id,
        "question": hybrid["query"],
        "bm25_hits": bm25_hits,
        "hybrid_hits": hybrid_hits,
        "bm25_mrr": bm25_mrr,
        "hybrid_mrr": hybrid_mrr,
        "bm25_ids": bm25["top_chunk_ids"],
        "hybrid_ids": hybrid["top_chunk_ids"],
        "gold": sorted(gold),
    }

    if hybrid_hits > bm25_hits:
        hybrid_better.append(comparison)
    elif bm25_hits > hybrid_hits:
        bm25_better.append(comparison)
    else:
        ties.append(comparison)


def display(title, cases):
    print(f"\n{title}: {len(cases)} queries")
    print("-" * 90)

    for item in cases:
        print(f"\n{item['query_id']}: {item['question']}")
        print(
            f"BM25:   {item['bm25_hits']}/"
            f"5 | MRR={item['bm25_mrr']:.3f}"
        )
        print(
            f"Hybrid: {item['hybrid_hits']}/"
            f"5 | MRR={item['hybrid_mrr']:.3f}"
        )
        print(f"Gold:   {item['gold']}")
        print(f"BM25:   {item['bm25_ids']}")
        print(f"Hybrid: {item['hybrid_ids']}")


display("HYBRID BETTER IN RECALL", hybrid_better)
display("BM25 BETTER IN RECALL", bm25_better)
display("RECALL TIES", ties)

print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)
