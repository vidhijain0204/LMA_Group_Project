"""Diagnosis agent: the one-time placement test that creates a new student's knowledge base.

Where it fits: when a student has no entry in data/skb_database.json yet, the orchestrator sends
them to the "diagnosis" node in workflow.py. That node shows get_questions() to the student and
passes their answers to administer_test(), which scores them and saves the first SKB entry.

What it stores per student (see models/skb_schema.py):
    topics                 accuracy per topic (0-1), used as mastery
    skills                 accuracy per subtopic (0-1)
    active_misconceptions  misconceptions linked to the questions answered wrongly, with counts
    diagnosis_responses    one record per question; the root-cause agent later treats the
                           diagnosis as the student's first quiz
Topics scoring below DEFAULT_THRESHOLD become weaknesses, the rest strengths.
"""

import json
import os
import re
from typing import Any, Dict, List

from models.skb_schema import StudentKnowledgeBase

DEFAULT_TOPICS_FILEPATH = "data/common/topics.json"
DEFAULT_MCQS_FILEPATH = "data/diagnosis/static_diagnosis_mcq_list.json"
DEFAULT_DB_FILEPATH = "data/skb_database.json"
DEFAULT_THRESHOLD = 0.7   # topic accuracy at or above this counts as a strength


def _field(question: dict, *names, default=None):
    """Read a question field that may be spelled differently across question files.

    Example: _field(q, "correct_option", "correct option") returns the first name that exists.
    """
    for name in names:
        if name in question:
            return question[name]
    return default

class DiagnosisAgent:
    """Gives the diagnosis test, scores it and saves the student's first knowledge base."""

    def __init__(
        self,
        topics_filepath: str = DEFAULT_TOPICS_FILEPATH,
        mcqs_filepath: str = DEFAULT_MCQS_FILEPATH,
        db_filepath: str = DEFAULT_DB_FILEPATH,
    ):
        """Load the topic catalog and the diagnosis questions.

        Also builds two lookups: topic name -> topic id, and misconception id -> description.
        """
        self.db_filepath = db_filepath
        self.threshold = DEFAULT_THRESHOLD
        self.topics = self._load_json(topics_filepath)
        self.mcqs = self._load_json(mcqs_filepath)

        self.topic_name_to_id = {t.get("name"): t.get("topic_id") for t in self.topics if "name" in t}
        self.misconception_map = {}
        for t in self.topics:
            for m in t.get("common_misconceptions", []):
                self.misconception_map[m.get("id")] = m.get("description")

    def _load_json(self, filepath: str) -> list:
        """Read a JSON file, or return an empty list if the file does not exist."""
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_db(self) -> list:
        """All student records from the SKB database file."""
        from agents.orchestrator import read_skb_database   # local import: avoids an import cycle
        return read_skb_database(self.db_filepath)

    def student_exists(self, student_id: str) -> bool:
        """True if this student already has an SKB entry (so the diagnosis was already taken)."""
        return any(e.get("student_id") == student_id for e in self._load_db())

    def new_student_id(self) -> str:
        """Next free id of the form S001, S002, ... (one above the highest existing number)."""
        numbers = [
            int(m.group(1))
            for e in self._load_db()
            if (m := re.fullmatch(r"S(\d+)", e.get("student_id", "")))
        ]
        return f"S{max(numbers, default=0) + 1:03d}"

    def get_questions(self) -> List[Dict[str, Any]]:
        """The diagnosis questions as shown to the student, without the correct answers.

        Field names are normalised, and options stored as a JSON string are decoded.
        """
        student_view = []
        for q in self.mcqs:
            raw_options = q.get("options", q.get("option", []))
            options = json.loads(raw_options) if isinstance(raw_options, str) else raw_options

            student_view.append({
                "problem_id": q.get("prob_id", q.get("problem_id")),
                "body": q.get("prob_body", q.get("body")),
                "options": options,
                "type": q.get("type", "conceptual"),
                "related_topics": _field(q, "related_topics", "related topics", default=[]),
                "related_subtopics": _field(q, "related_subtopics", "related subtopics", default=[]),
                "difficulty": q.get("difficulty", 0.0),
            })
        return student_view

    def administer_test(self, student_id: str, student_answers: Dict[str, Any]) -> StudentKnowledgeBase:
        """Score the diagnosis answers, build the student's SKB and save it.

        Args:
            student_id: the new student's id.
            student_answers: problem id -> chosen option.
        Returns: the new StudentKnowledgeBase.
        Raises ValueError if the student already has an SKB (the diagnosis is taken only once).
        """
        if self.student_exists(student_id):
            raise ValueError(f"{student_id} already has a KB entry. Diagnosis is taken only once.")

        # [correct, total] counters per topic and per subtopic.
        topic_stats = {t["topic_id"]: [0, 0] for t in self.topics if "topic_id" in t}
        subtopic_stats = {}
        misconceptions = {}
        responses = []

        for q in self.mcqs:
            problem_id = q.get("prob_id", q.get("problem_id"))
            correct_val = _field(q, "correct_option", "correct option")
            submitted_val = student_answers.get(problem_id)

            # Some questions accept several correct options.
            if isinstance(correct_val, list):
                is_correct = submitted_val in correct_val
            else:
                is_correct = (submitted_val == correct_val) and (submitted_val is not None)

            for topic_identifier in _field(q, "related_topics", "related topics", default=[]):
                # Questions may list a topic by name or by id; map names to ids.
                t_id = self.topic_name_to_id.get(topic_identifier, topic_identifier)
                bucket = topic_stats.setdefault(t_id, [0, 0])
                bucket[1] += 1
                bucket[0] += int(is_correct)

            responses.append({
                "prob_id": problem_id,
                "chosen": submitted_val,
                "is_correct": bool(is_correct),
                "subtopics": list(_field(q, "related_subtopics", "related subtopics", default=[])),
            })
            for sub_id in _field(q, "related_subtopics", "related subtopics", default=[]):
                sub_bucket = subtopic_stats.setdefault(sub_id, [0, 0])
                sub_bucket[1] += 1
                sub_bucket[0] += int(is_correct)

            if not is_correct:
                for m_tag in q.get("misconception_tags", []):
                    desc = self.misconception_map.get(m_tag, m_tag)
                    misconceptions[desc] = misconceptions.get(desc, 0) + 1

        def to_prob(stats: Dict[str, List[int]]) -> Dict[str, float]:
            """Turn [correct, total] counters into accuracies rounded to 2 decimals."""
            return {k: round(c / t, 2) if t else 0.0 for k, (c, t) in stats.items()}

        skb = StudentKnowledgeBase(
            student_id=student_id,
            topics=to_prob(topic_stats),
            skills=to_prob(subtopic_stats),  
            active_misconceptions=misconceptions,
            diagnosis_responses=responses,
        )
        skb.update_categorization(threshold=self.threshold)
        self._save(skb)
        return skb

    def _save(self, skb: StudentKnowledgeBase):
        """Write the student's SKB into the database file, replacing any old entry for them.

        Writes to a temporary file first and then swaps it in, so a crash cannot leave a
        half-written database.
        """
        db = [e for e in self._load_db() if e.get("student_id") != skb.student_id]
        db.append(skb.model_dump(mode="json"))

        folder = os.path.dirname(self.db_filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)
        tmp_path = self.db_filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2)
        os.replace(tmp_path, self.db_filepath)
        print(f"[Diagnosis] Saved KB for {skb.student_id} to {self.db_filepath}")