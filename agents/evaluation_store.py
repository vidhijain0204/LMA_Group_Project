"""Study data collected in the app, kept apart from the knowledge base.

    data/evaluation/labels.jsonl     the student's own reason for each wrong quiz answer, and the
                                     subtopic they think caused most mistakes (ground truth for
                                     the root-cause evaluation; expert labels can be added with
                                     scripts/evaluate.py --import-labels)
    data/evaluation/feedback.jsonl   end-of-session ratings and comments
    data/evaluation/consent.jsonl    one line per student id that agreed to take part
    data/backups/<time>/             copies of the KB, traces and evaluation files

One JSON object per line, appended, so a crash never corrupts earlier records.
"""

import json
import os
import re
import shutil
from datetime import datetime
from typing import Iterable, List, Optional

EVALUATION_DIR = "data/evaluation"
BACKUP_DIR = "data/backups"
KEEP_BACKUPS = 30
FILES = {
    "labels": "labels.jsonl",
    "feedback": "feedback.jsonl",
    "consent": "consent.jsonl",
}


def path_of(kind: str) -> str:
    return os.path.join(EVALUATION_DIR, FILES[kind])


def append(kind: str, record: dict) -> dict:
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    record = {"created_at": datetime.now().isoformat(timespec="seconds"), **record}
    with open(path_of(kind), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read(kind_or_path: str) -> List[dict]:
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


def has_consent(student_id: str) -> bool:
    return any(r.get("student_id") == student_id and r.get("consent") for r in read("consent"))


def labelled_quiz_ids(student_id: Optional[str] = None) -> set:
    return {r.get("quiz_id") for r in read("labels") if not student_id or r.get("student_id") == student_id}


def next_anonymous_id(existing_ids: Iterable[str]) -> str:
    """S101, S102, ... : the next number above every id in the KB and the consent file."""
    ids = set(existing_ids) | {r.get("student_id", "") for r in read("consent")}
    numbers = [int(m.group(1)) for i in ids if (m := re.fullmatch(r"S(\d+)", str(i)))]
    return f"S{max(numbers + [100]) + 1:03d}"


def backup_data(sources: Iterable[str] = ("data/skb_database.json", "data/traces", EVALUATION_DIR),
                reason: str = "manual") -> Optional[str]:
    """Copy the study data to data/backups/<time>-<reason>/; keeps the newest KEEP_BACKUPS."""
    existing = [s for s in sources if os.path.exists(s)]
    if not existing:
        return None
    target = os.path.join(BACKUP_DIR, f"{datetime.now():%Y%m%d-%H%M%S}-{reason}")
    os.makedirs(target, exist_ok=True)
    for source in existing:
        destination = os.path.join(target, os.path.basename(source.rstrip("/")))
        if os.path.isdir(source):
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
    backups = sorted(d for d in os.listdir(BACKUP_DIR) if os.path.isdir(os.path.join(BACKUP_DIR, d)))
    for old in backups[:-KEEP_BACKUPS]:
        shutil.rmtree(os.path.join(BACKUP_DIR, old), ignore_errors=True)
    return target
