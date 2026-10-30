
import pandas as pd
from pathlib import Path

review_path = Path("reports/retrieval_review.csv")
eval_path = Path("data/common/retrieval_eval.csv")

review = pd.read_csv(review_path)
evaluation = pd.read_csv(eval_path)

# Keep only chunks you validated as relevant.
relevant = review[
    review["relevant"].astype(str).str.strip().str.lower() == "yes"
]

# Collect relevant chunk IDs for each query.
gold = (
    relevant.groupby("query_id")["chunk_id"]
    .apply(lambda ids: ";".join(sorted(set(ids.astype(str)))))
    .reset_index(name="relevant_chunk_ids")
)

# Replace the existing relevance column, if present.
evaluation = evaluation.drop(
    columns=["relevant_chunk_ids"], errors="ignore"
)

evaluation = evaluation.merge(gold, on="query_id", how="left")
evaluation["relevant_chunk_ids"] = (
    evaluation["relevant_chunk_ids"].fillna("")
)

evaluation.to_csv(eval_path, index=False)

annotated = evaluation["relevant_chunk_ids"].ne("").sum()
print(f"Queries with at least one relevant chunk: {annotated}")
print(f"Total queries: {len(evaluation)}")
print(f"Updated: {eval_path}")
