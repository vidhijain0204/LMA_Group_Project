import json
import os
import re
from typing import Any, Dict, List

from models.skb_schema import StudentKnowledgeBase

DEFAULT_TOPICS_FILEPATH = "data/common/topics.json"
DEFAULT_MCQS_FILEPATH = "data/diagnosis/static_diagnosis_mcq_list.json"
DEFAULT_DB_FILEPATH = "data/skb_database.json"
DEFAULT_THRESHOLD = 0.7

class DiagnosisAgent:
    def __init__(
        self,
        topics_filepath: str = DEFAULT_TOPICS_FILEPATH,
        mcqs_filepath: str = DEFAULT_MCQS_FILEPATH,
        db_filepath: str = DEFAULT_DB_FILEPATH,
    ):
        self.db_filepath = db_filepath
        self.threshold = DEFAULT_THRESHOLD
        self.topics = self._load_json(topics_filepath)
        self.mcqs = self._load_json(mcqs_filepath)

        # Lookup map for both name -> topic_id and topic_id -> topic_id
        self.topic_name_to_id = {t.get("name"): t.get("topic_id") for t in self.topics if "name" in t}

    def _load_json(self, filepath: str) -> list:
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_db(self) -> list:
        if not os.path.exists(self.db_filepath) or os.path.getsize(self.db_filepath) == 0:
            return []
        with open(self.db_filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def student_exists(self, student_id: str) -> bool:
        return any(e.get("student_id") == student_id for e in self._load_db())

    def new_student_id(self) -> str:
        numbers = [
            int(m.group(1))
            for e in self._load_db()
            if (m := re.fullmatch(r"S(\d+)", e.get("student_id", "")))
        ]
        return f"S{max(numbers, default=0) + 1:03d}"

    def get_questions(self) -> List[Dict[str, Any]]:
        """
        Returns student-facing question objects adhering to the schema.
        Omits answer key, explanations, and internal tags.
        """
        student_view = []
        for q in self.mcqs:
            # Handle options whether provided as list or serialized JSON string
            raw_options = q.get("option", [])
            options = json.loads(raw_options) if isinstance(raw_options, str) else raw_options

            student_view.append({
                "problem_id": q.get("problem_id"),
                "body": q.get("body"),
                "option": options,
                "type": q.get("type", "single_choice"),
                "related topics": q.get("related topics", []),
            })
        return student_view

    def administer_test(self, student_id: str, student_answers: Dict[str, Any]) -> StudentKnowledgeBase:
        if self.student_exists(student_id):
            raise ValueError(f"{student_id} already has a KB entry. Diagnosis is taken only once.")

        # Initialize tracking for all defined topics
        topic_stats = {t["topic_id"]: [0, 0] for t in self.topics if "topic_id" in t}
        skill_stats = {}
        misconceptions = {}

        for q in self.mcqs:
            problem_id = q.get("problem_id")
            correct_val = q.get("correct option")
            submitted_val = student_answers.get(problem_id)

            # Support both direct string matching and array-based answers
            if isinstance(correct_val, list):
                is_correct = submitted_val in correct_val
            else:
                is_correct = (submitted_val == correct_val) and (submitted_val is not None)

            # Aggregate stats across all related topics
            related_topics = q.get("related topics", [])
            for topic_identifier in related_topics:
                # Map topic name to topic_id if possible, otherwise retain identifier
                t_id = self.topic_name_to_id.get(topic_identifier, topic_identifier)
                bucket = topic_stats.setdefault(t_id, [0, 0])
                bucket[1] += 1
                bucket[0] += int(is_correct)

            # Skill tracking (if present in question payload)
            if q.get("primary_skill_id"):
                skill_bucket = skill_stats.setdefault(q["primary_skill_id"], [0, 0])
                skill_bucket[1] += 1
                skill_bucket[0] += int(is_correct)

            # Misconception tracking on incorrect answer
            if not is_correct:
                for tag in q.get("misconception_tags", []):
                    misconceptions[tag] = misconceptions.get(tag, 0) + 1

        def to_prob(stats: Dict[str, List[int]]) -> Dict[str, float]:
            return {k: round(c / t, 2) if t else 0.0 for k, (c, t) in stats.items()}

        skb = StudentKnowledgeBase(
            student_id=student_id,
            topics=to_prob(topic_stats),
            skills=to_prob(skill_stats),
            active_misconceptions=misconceptions,
        )
        skb.update_categorization(threshold=self.threshold)
        self._save(skb)
        return skb

    def _save(self, skb: StudentKnowledgeBase):
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