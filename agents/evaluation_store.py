"""Saves the students' answers to the "why did you get these wrong?" form (the evaluation labels).

Where it fits: after a quiz, app.py shows an optional form where the student says why they got
each question wrong and which subtopic caused most of their mistakes. app.py saves the answers
with append("labels", ...). scripts/eval_data.py reads them back as the ground truth ("true
root cause") for scripts/evaluate.py and scripts/fit_weights.py.

The labels are kept apart from the knowledge base, in data/evaluation/labels.jsonl:
one JSON object per line, only ever appended, so a crash never corrupts earlier records.
Expert labels imported with scripts/evaluate.py --import-labels go to a separate file in the
same folder (read() also accepts a file path).

Main contents: append(), read(), labelled_quiz_ids() and next_anonymous_id().
"""

import json
import os
import re
from datetime import datetime
from typing import Iterable, List, Optional

EVALUATION_DIR = "data/evaluation"
# Short name -> file name inside EVALUATION_DIR.
FILES = {
    "labels": "labels.jsonl",
}


def path_of(kind: str) -> str:
    """Full path of a stored file, e.g. path_of("labels") -> data/evaluation/labels.jsonl."""
    return os.path.join(EVALUATION_DIR, FILES[kind])


def append(kind: str, record: dict) -> dict:
    """Add one record (with a "created_at" time stamp) as a new line. Returns the saved record."""
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    record = {"created_at": datetime.now().isoformat(timespec="seconds"), **record}
    with open(path_of(kind), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read(kind_or_path: str) -> List[dict]:
    """All records of a file, given its short name ("labels") or a path to any .jsonl file.

    A missing file gives an empty list; lines that are not valid JSON are skipped.
    """
    path = path_of(kind_or_path) if kind_or_path in FILES else kind_or_path
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue            # a half-written last line
    return records


def labelled_quiz_ids(student_id: Optional[str] = None) -> set:
    """Ids of the quizzes that already have a label (for one student, or for everyone).

    app.py uses this so the form is not shown again for a quiz that was already answered.
    """
    return {r.get("quiz_id") for r in read("labels") if not student_id or r.get("student_id") == student_id}


def next_anonymous_id(existing_ids: Iterable[str]) -> str:
    """Next anonymous student id: S101, S102, ... (one above every existing id, at least S101).

    Used by the "New anonymous student" button in app.py.
    """
    ids = set(existing_ids)
    numbers = [int(m.group(1)) for i in ids if (m := re.fullmatch(r"S(\d+)", str(i)))]
    return f"S{max(numbers + [100]) + 1:03d}"
