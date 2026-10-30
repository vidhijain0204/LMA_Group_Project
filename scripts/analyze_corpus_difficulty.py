
import json
from collections import Counter, defaultdict

CORPUS_FILE = "data/common/corpus.jsonl"

topic_counts = defaultdict(Counter)
global_counts = Counter()

with open(CORPUS_FILE, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        document = json.loads(line)

        topic = document.get("topic_id")
        difficulty = document.get("difficulty")

        if difficulty is None:
            continue

        topic_counts[topic][difficulty] += 1
        global_counts[difficulty] += 1

print("GLOBAL DIFFICULTY DISTRIBUTION")
print("-" * 40)

total = sum(global_counts.values())

for difficulty in sorted(global_counts):
    count = global_counts[difficulty]
    print(
        f"Difficulty {difficulty}: {count} chunks "
        f"({count / total * 100:.1f}%)"
    )

print("\nPER-TOPIC DIFFICULTY DISTRIBUTION")
print("-" * 40)

for topic in sorted(topic_counts):
    counts = topic_counts[topic]
    print(
        f"{topic}: "
        f"D1={counts[1]}, "
        f"D2={counts[2]}, "
        f"D3={counts[3]}"
    )
