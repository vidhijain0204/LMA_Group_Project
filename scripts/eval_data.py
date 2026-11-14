"""Load the study data for scripts/evaluate.py and scripts/fit_weights.py."""

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
    return read_skb_database(db_path)


def quizzes(records: List[dict]) -> List[dict]:
    """One entry per quiz (its per-topic assessment events merged), oldest first."""
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
    return (label or {}).get("true_subtopic") or None
