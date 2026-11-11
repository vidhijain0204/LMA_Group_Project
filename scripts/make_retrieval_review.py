import csv
import json
from pathlib import Path

CORPUS = Path("data/common/corpus.jsonl")
RESULTS = Path("reports/retrieval_comparison.csv")
OUTPUT = Path("reports/retrieval_review.csv")


def load_corpus():
    chunks = {}
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                chunks[chunk_id] = chunk
    return chunks


def main():
    chunks = load_corpus()
    rows = []

    with RESULTS.open(encoding="utf-8", newline="") as f:
        for result in csv.DictReader(f):
            ids = result["top_chunk_ids"].split(";")
            scores = result["top_scores"].split(";")

            for rank, chunk_id in enumerate(ids, start=1):
                chunk = chunks.get(chunk_id, {})

                rows.append({
                    "query_id": result["query_id"],
                    "query": result["query"],
                    "method": result["method"],
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "score": scores[rank - 1]
                    if rank - 1 < len(scores) else "",
                    "topic_id": chunk.get("topic_id", ""),
                    "type": chunk.get("type", ""),
                    "difficulty": chunk.get("difficulty", ""),
                    "source": chunk.get("source", ""),
                    "chunk_text": chunk.get("text", ""),
                    "relevant": "",
                })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {OUTPUT}")
    print(f"Candidate rows: {len(rows)}")
    print(f"Corpus chunks loaded: {len(chunks)}")


if __name__ == "__main__":
    main()
