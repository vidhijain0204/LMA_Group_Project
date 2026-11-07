import json
from typing import Any, Dict, List


class ErrorAnalysisAgent:
    """Summarize incorrect quiz answers as misconception or subtopic signals."""

    def __init__(self, topics_filepath: str = "data/common/topics.json"):
        with open(topics_filepath, "r", encoding="utf-8") as topic_file:
            topics = json.load(topic_file)

        self.misconceptions = {}
        self.subtopics = {}
        for topic in topics:
            for misconception in topic.get("common_misconceptions", []):
                self.misconceptions[misconception["id"]] = misconception["description"]
            for subtopic in topic.get("subconcepts", []):
                self.subtopics[subtopic["id"]] = subtopic["name"]

    def analyze(
        self,
        questions: List[Dict[str, Any]],
        answers: Dict[str, Any],
        topic_id: str = "",
    ) -> dict:
        correct_count = 0
        topic_correct_count = 0
        topic_total = 0
        errors = []
        for question in questions:
            question_id = question.get("prob_id", question.get("problem_id"))
            is_correct = answers.get(question_id) == question.get("correct_option")
            related_topics = question.get("related_topics", [])
            related_subtopics = question.get("related_subtopics", [])
            topic_relevant = (
                not topic_id
                or not (related_topics or related_subtopics)
                or topic_id in related_topics
                or any(subtopic.startswith(f"{topic_id}.") for subtopic in related_subtopics)
            )
            if topic_relevant:
                topic_total += 1
                topic_correct_count += int(is_correct)
            if is_correct:
                correct_count += 1
                continue
            if topic_id and not topic_relevant:
                continue

            tags = question.get("misconception_tags", [])
            if tags:
                signals = [
                    (tag, self.misconceptions.get(tag, tag))
                    for tag in tags
                ]
            else:
                relevant_subtopics = [
                    subtopic
                    for subtopic in related_subtopics
                    if not topic_id or subtopic.startswith(f"{topic_id}.")
                ]
                if relevant_subtopics:
                    signals = [
                        (subtopic, self.subtopics.get(subtopic, subtopic))
                        for subtopic in relevant_subtopics
                    ]
                else:
                    relevant_topics = [
                        topic
                        for topic in related_topics
                        if not topic_id or topic == topic_id
                    ]
                    signals = [
                        (topic, f"Questions related to {topic}")
                        for topic in relevant_topics
                    ]

            for signal_id, description in signals:
                errors.append({
                    "question_id": question_id,
                    "signal_id": signal_id,
                    "description": description,
                })

        total = len(questions)
        return {
            "correct_count": correct_count,
            "total": total,
            "accuracy": round(correct_count / total, 2) if total else 0.0,
            "topic_accuracy": (
                round(topic_correct_count / topic_total, 2)
                if topic_total
                else None
            ),
            "errors": errors,
        }
