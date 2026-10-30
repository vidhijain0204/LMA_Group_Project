
import json
import re
from pathlib import Path

import pandas as pd

from agents.course_retriever import CourseRetriever


EVAL_FILE = Path("data/common/retrieval_eval.csv")
TOPICS_FILE = Path("data/common/topics.json")
OUTPUT_FILE = Path("reports/mastery_ranking_evaluation.csv")

METHOD = "bm25"
TOP_K = 5

CONDITIONS = {
    "baseline": None,
    "beginner": 0.0,
    "intermediate": 0.5,
    "advanced": 0.9,
}


def normalize_topic_name(name):
    """Normalize topic names for matching."""
    name = str(name).replace(";", ",").strip().lower()
    name = re.sub(r"\s+", " ", name)
    return name


def load_topic_mapping():
    with TOPICS_FILE.open(encoding="utf-8") as f:
        topics = json.load(f)

    mapping = {}

    for topic in topics:
        topic_id = topic["topic_id"]
        name = topic["name"]

        mapping[normalize_topic_name(name)] = topic_id
        mapping[topic_id.lower()] = topic_id

    return mapping


def parse_topics(value, topic_mapping):
    """Convert evaluation topic names into corpus topic IDs."""
    if pd.isna(value) or not str(value).strip():
        return []

    # Evaluation topics are separated by |.
    names = str(value).split("|")

    topic_ids = []
    unknown_topics = []

    for name in names:
        normalized = normalize_topic_name(name)

        if normalized in topic_mapping:
            topic_ids.append(topic_mapping[normalized])
        else:
            unknown_topics.append(name.strip())

    if unknown_topics:
        raise ValueError(
            f"Unmapped topic names: {unknown_topics}. "
            "Check retrieval_eval.csv and topics.json."
        )

    return list(dict.fromkeys(topic_ids))


def target_difficulty(mastery):
    if mastery is None:
        return None
    if mastery < 0.4:
        return 1
    if mastery < 0.7:
        return 2
    return 3


def main():
    evaluation = pd.read_csv(EVAL_FILE)
    topic_mapping = load_topic_mapping()
    retriever = CourseRetriever()

    required_columns = {"query_id", "query", "related_topics"}
    missing = required_columns - set(evaluation.columns)

    if missing:
        raise ValueError(
            f"Missing evaluation columns: {sorted(missing)}"
        )

    output_rows = []

    for _, row in evaluation.iterrows():
        query_id = str(row["query_id"])
        query = str(row["query"])

        topics = parse_topics(
            row["related_topics"],
            topic_mapping,
        )

        for condition, mastery in CONDITIONS.items():
            mastery_scores = (
                {topic_id: mastery for topic_id in topics}
                if mastery is not None
                else None
            )

            results = retriever.search(
                query=query,
                topic_ids=topics or None,
                mastery_scores=mastery_scores,
                limit=TOP_K,
                method=METHOD,
            )

            difficulties = [
                result["difficulty"]
                for result in results
                if isinstance(result["difficulty"], (int, float))
            ]

            target = target_difficulty(mastery)

            mean_difficulty = (
                sum(difficulties) / len(difficulties)
                if difficulties else None
            )

            match_rate = (
                sum(d == target for d in difficulties) / len(difficulties)
                if target is not None and difficulties else None
            )

            output_rows.append({
                "query_id": query_id,
                "query": query,
                "topics": ";".join(topics),
                "condition": condition,
                "mastery": mastery,
                "target_difficulty": target,
                "retrieved_count": len(results),
                "chunk_ids": ";".join(
                    result["chunk_id"] for result in results
                ),
                "difficulties": ";".join(
                    str(result["difficulty"]) for result in results
                ),
                "mean_difficulty": mean_difficulty,
                "target_match_rate": match_rate,
            })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(output_rows)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Detailed results saved to: {OUTPUT_FILE}")
    print(f"Queries evaluated: {evaluation['query_id'].nunique()}")
    print(f"Retrieval method: {METHOD}")
    print(f"Top K: {TOP_K}")

    print("\nAggregate results:")
    summary = (
        results_df.groupby("condition")
        .agg(
            mean_difficulty=("mean_difficulty", "mean"),
            target_match_rate=("target_match_rate", "mean"),
            mean_retrieved_count=("retrieved_count", "mean"),
        )
        .round(4)
    )

    print(summary.to_string())

    print("\nTopic mapping check:")
    print("All evaluation topic names mapped successfully.")


if __name__ == "__main__":
    main()
