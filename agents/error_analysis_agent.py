"""Error analysis agent: grades a quiz for one topic and lists an "error signal" per wrong answer.

Where it fits: after a quiz, the assessment node in workflow.py splits the questions by topic
and calls ErrorAnalysisAgent.analyze() once per topic. The error signals are logged in the trace
and saved in the student's SKB (pedagogy history and recurring learning signals), where the
tutor can use them later. The root cause itself is found afterwards by root_cause_agent.py.

An error signal says what a wrong answer points to, using the most specific information the
question has: its misconception tags, else its subtopics, else its topics.
"""

import json
from typing import Any, Dict, List


class ErrorAnalysisAgent:
    """Grades quiz answers and turns each wrong answer into misconception or subtopic signals."""

    def __init__(self, topics_filepath: str = "data/common/topics.json"):
        """Load misconception and subtopic names from the topic catalog (id -> readable text)."""
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
        """Grade the answers and collect an error signal for each wrong answer.

        Args:
            questions: the quiz questions (with "correct_option" and their tags).
            answers: question id -> chosen option.
            topic_id: if given, only wrong answers about this topic produce signals, and
                topic_accuracy is measured on this topic's questions only.
        Returns: {"correct_count", "total", "accuracy", "topic_accuracy", "errors"}, where each
        error is {"question_id", "signal_id", "description"}.
        """
        correct_count = 0
        topic_correct_count = 0
        topic_total = 0
        errors = []
        for question in questions:
            question_id = question.get("prob_id", question.get("problem_id"))
            is_correct = answers.get(question_id) == question.get("correct_option")
            related_topics = question.get("related_topics", [])
            related_subtopics = question.get("related_subtopics", [])
            # A question with no topic tags at all counts as relevant to any topic.
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

            # Most specific signal first: misconception tags, then subtopics, then topics.
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
