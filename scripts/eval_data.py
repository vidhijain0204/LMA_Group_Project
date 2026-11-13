"""Shared data loader for scripts/evaluate.py and scripts/fit_weights.py.

Why it exists: both scripts need the same data, so the loading code lives here once.
It is not run on its own; the two scripts import it.

It reads:
    - the students' records from data/skb_database.json (the SKB),
    - every quiz in them, with the root cause each method picked,
    - the labels: the students' answers to the "why did you get these wrong?" form
      (saved by agents/evaluation_store.py) and any imported expert labels.
The labels are the ground truth (the "true" root-cause subtopic) the methods are compared with.

Importing this module also adds the project root to sys.path, so the scripts can import agents.*.
"""

import json
import os
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from agents import evaluation_store                     # noqa: E402
from agents.orchestrator import SKB_FILE_PATH, read_skb_database  # noqa: E402

EXPERT_LABELS = os.path.join(evaluation_store.EVALUATION_DIR, "expert_labels.jsonl")


def students(db_path: str = SKB_FILE_PATH) -> List[dict]:
    """All student records from the SKB database."""
    return read_skb_database(db_path)


def quizzes(records: List[dict]) -> List[dict]:
    """One entry per quiz, oldest first.

    A quiz on several topics is stored as one assessment event per topic; these are merged by
    quiz_id. Each entry has the responses, topics and, if one was found, the root cause, the
    method comparison, the hypothesis candidates and the settings used.
    """
    by_id: Dict[str, dict] = {}
    for kb in records:
        for event in kb.get("pedagogy_history", []):
            if event.get("event") != "assessment" or not event.get("quiz_id"):
                continue
            quiz = by_id.setdefault(event["quiz_id"], {
                "quiz_id": event["quiz_id"], "student_id": kb.get("student_id"),
                "created_at": event.get("created_at"), "responses": [], "topics": [],
                "retest_of": event.get("retest_of"),
            })
            quiz["responses"].extend(event.get("responses", []))
            quiz["topics"].append(event.get("topic_id"))
            if event.get("root_cause"):
                quiz["root_cause"] = event["root_cause"]
                quiz["comparison"] = event.get("root_cause_comparison") or {}
                quiz["candidates"] = event.get("hypothesis_candidates") or []
                quiz["settings"] = event.get("root_cause_settings") or {}
    return sorted(by_id.values(), key=lambda q: q.get("created_at") or "")


def labels() -> Dict[str, dict]:
    """quiz_id -> label. Expert labels (imported CSV) win over the student's own answer."""
    result: Dict[str, dict] = {}
    for record in evaluation_store.read("labels"):
        if record.get("quiz_id"):
            result[record["quiz_id"]] = record
    for record in evaluation_store.read(EXPERT_LABELS):
        if record.get("quiz_id") and (record.get("true_subtopic") or record.get("true_misconception")):
            result[record["quiz_id"]] = {**result.get(record["quiz_id"], {}), **record, "source": "expert"}
    return result


def true_subtopic(label: Optional[dict]) -> Optional[str]:
    """The true root-cause subtopic of a label, or None (no label, or the student was not sure)."""
    return (label or {}).get("true_subtopic") or None
