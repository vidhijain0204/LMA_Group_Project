"""Create a retrieval-evaluation CSV from existing diagnosis questions JSON.

Usage:
  python scripts/prepare_retrieval_eval.py data/common/retrieval_evaulation.json
"""
import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "query_id", "query", "related_topics",
    "relevant_chunk_ids", "explanation_for_annotator",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Path to diagnosis questions JSON array")
    parser.add_argument("--output", default="data/common/retrieval_eval.csv")
    args = parser.parse_args()

    with Path(args.input_json).open("r", encoding="utf-8") as f:
        problems = json.load(f)
    if not isinstance(problems, list):
        raise ValueError("Expected the diagnosis file to contain a JSON array.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for p in problems:
            pid = str(p.get("problem_id", "")).strip()
            body = str(p.get("body", "")).strip()
            if not pid or not body:
                continue
            topics = p.get("related topics", [])
            if not isinstance(topics, list):
                topics = [str(topics)]
            writer.writerow({
                "query_id": pid,
                "query": body,
                "related_topics": " | ".join(map(str, topics)),
                "relevant_chunk_ids": "",
                "explanation_for_annotator": str(p.get("explanation", "")).strip(),
            })
            count += 1
    print(f"Created {output} with {count} questions.")
    print("Annotate relevant_chunk_ids using chunk IDs from corpus.jsonl.")

if __name__ == "__main__":
    main()
