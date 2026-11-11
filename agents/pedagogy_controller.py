from collections import Counter
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional, Tuple


STRATEGY_INSTRUCTIONS = {
    "hints": (
        "Give one targeted hint and a guiding question. Do not reveal the full solution "
        "unless the student asks for it."
    ),
    "step_scaffolding": (
        "Break the task into small, ordered steps. Explain the first step clearly and "
        "show how each step leads to the next."
    ),
    "explicit_explanation": (
        "Explain the concept directly, including the key rule, why it works, and a concise "
        "example."
    ),
    "worked_example": (
        "Walk through a complete example line by line, explaining the reasoning at each "
        "step, then offer a similar problem to try."
    ),
    "self_explanation": (
        "Ask the student to predict or explain a key step before giving the answer. Provide "
        "a focused prompt and a small clue rather than a full solution."
    ),
    "contrastive_feedback": (
        "Compare a correct example with a plausible incorrect one. Explain the difference "
        "and directly address the student's recurring error."
    ),
}

STRATEGY_CYCLE = tuple(STRATEGY_INSTRUCTIONS)
MIN_STRATEGY_ASSESSMENTS = 2
LOW_ACCURACY_THRESHOLD = 0.6


def resolve_topic_id(topic_focus: str) -> str:
    """Resolve a topic name or ID using the local topic catalog when available."""
    query = str(topic_focus or "").strip()
    if not query:
        return "unknown"

    try:
        import json

        with open("data/common/topics.json", "r", encoding="utf-8") as topic_file:
            topics = json.load(topic_file)
    except FileNotFoundError:
        return query

    lowered = query.lower()
    for topic in topics:
        topic_id = str(topic.get("topic_id", ""))
        name = str(topic.get("name", ""))
        if lowered == topic_id.lower() or lowered == name.lower():
            return topic_id
    for topic in topics:
        topic_id = str(topic.get("topic_id", ""))
        name = str(topic.get("name", ""))
        if topic_id and topic_id.lower() in lowered:
            return topic_id
        if lowered and lowered in name.lower():
            return topic_id
    return query


def _signals_recur(assessments: List[dict]) -> bool:
    recent = assessments[-3:]
    counts = Counter(
        signal_id
        for assessment in recent
        for signal_id in set(assessment.get("error_signals", []))
    )
    return any(count >= 2 for count in counts.values())


def recent_error_context(student_kb: Optional[dict], topic_id: str) -> List[str]:
    assessments = [
        event
        for event in (student_kb or {}).get("pedagogy_history", [])
        if event.get("event") == "assessment" and event.get("topic_id") == topic_id
    ][-3:]
    counts = Counter(
        signal_id
        for assessment in assessments
        for signal_id in set(assessment.get("error_signals", []))
    )
    descriptions = (student_kb or {}).get("learning_signal_descriptions", {})
    return [
        descriptions.get(signal_id, signal_id)
        for signal_id, _ in counts.most_common()
        if counts[signal_id] >= 2
    ]


def latest_root_cause(student_kb: Optional[dict], topic_id: str) -> Optional[dict]:
    """The most recent root cause found for a quiz on this topic (or located in this topic)."""
    for entry in reversed((student_kb or {}).get("root_cause_history", [])):
        if topic_id in (entry.get("quiz_topic_id"), entry.get("topic_id")):
            return entry
    return None


def choose_teaching_strategy(
    student_kb: Optional[dict],
    topic_id: str,
    mastery_score: float,
) -> Tuple[str, str]:
    """Select a strategy, changing it only after repeated assessment evidence."""
    history = (student_kb or {}).get("pedagogy_history", [])
    topic_history = [event for event in history if event.get("topic_id") == topic_id]
    lessons = [event for event in topic_history if event.get("event") == "lesson"]
    assessments = [
        event
        for event in topic_history
        if event.get("event") == "assessment" and event.get("accuracy") is not None
    ]

    if assessments and _signals_recur(assessments):
        current_strategy = lessons[-1]["strategy"] if lessons else None
        if current_strategy != "contrastive_feedback":
            return "contrastive_feedback", "Recurring errors on this topic call for a direct comparison."

    current_strategy = lessons[-1]["strategy"] if lessons else None
    if current_strategy:
        strategy_results = [
            event["accuracy"]
            for event in assessments
            if event.get("strategy") == current_strategy
        ]
        if (
            len(strategy_results) >= MIN_STRATEGY_ASSESSMENTS
            and sum(strategy_results[-MIN_STRATEGY_ASSESSMENTS:])
            / MIN_STRATEGY_ASSESSMENTS
            < LOW_ACCURACY_THRESHOLD
        ):
            next_index = (STRATEGY_CYCLE.index(current_strategy) + 1) % len(STRATEGY_CYCLE)
            return STRATEGY_CYCLE[next_index], (
                f"Assessment accuracy stayed below {LOW_ACCURACY_THRESHOLD:.0%} "
                f"across {MIN_STRATEGY_ASSESSMENTS} checks."
            )
        return current_strategy, "Continue the current strategy; evidence does not justify a switch."

    if mastery_score < 0.4:
        return "step_scaffolding", "Low mastery calls for guided, incremental steps."
    if mastery_score >= 0.8:
        return "self_explanation", "Strong mastery calls for retrieval and explanation practice."
    return "explicit_explanation", "Use a direct explanation while initial evidence is collected."


def record_lesson_strategy(
    student_kb: dict,
    topic_id: str,
    strategy: str,
    reason: str,
) -> dict:
    updated = deepcopy(student_kb)
    history = updated.setdefault("pedagogy_history", [])
    history.append({
        "event": "lesson",
        "topic_id": topic_id,
        "strategy": strategy,
        "reason": reason,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    updated["pedagogy_history"] = history[-100:]
    updated["latest_update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return updated


def record_assessment_outcome(
    student_kb: dict,
    topic_id: str,
    accuracy: Optional[float],
    error_signals: List[str],
    error_descriptions: Optional[Dict[str, str]] = None,
    question_results: Optional[List[dict]] = None,
    root_cause: Optional[dict] = None,
) -> dict:
    """Record a quiz outcome. `question_results` keeps the option chosen for every question
    (needed for counter-evidence and repeat checks later); `root_cause` is the hypothesis the
    RootCauseAgent selected for this quiz."""
    updated = deepcopy(student_kb)
    history = updated.setdefault("pedagogy_history", [])
    matched_lesson = next(
        (
            event
            for event in reversed(history)
            if event.get("event") == "lesson"
            and event.get("topic_id") == topic_id
            and "assessment_accuracy" not in event
        ),
        None,
    )
    strategy = matched_lesson.get("strategy") if matched_lesson else None
    if matched_lesson:
        matched_lesson["assessment_accuracy"] = accuracy
        matched_lesson["error_signals"] = list(dict.fromkeys(error_signals))

    assessment_event = {
        "event": "assessment",
        "topic_id": topic_id,
        "strategy": strategy,
        "accuracy": accuracy,
        "error_signals": list(dict.fromkeys(error_signals)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if question_results is not None:
        assessment_event["responses"] = [
            {
                "prob_id": result.get("prob_id"),
                "chosen": result.get("chosen"),
                "correct": result.get("correct"),
                "is_correct": bool(result.get("is_correct")),
                "subtopics": list(result.get("subtopics", [])),
            }
            for result in question_results
        ]
    if root_cause:
        assessment_event["root_cause"] = root_cause
        root_causes = updated.setdefault("root_cause_history", [])
        root_causes.append({
            **root_cause,
            "quiz_topic_id": topic_id,
            "created_at": assessment_event["created_at"],
        })
        updated["root_cause_history"] = root_causes[-50:]
    history.append(assessment_event)
    updated["pedagogy_history"] = history[-100:]

    signal_counts = updated.setdefault("learning_signals", {})
    signal_descriptions = updated.setdefault("learning_signal_descriptions", {})
    for signal_id in set(error_signals):
        signal_counts[signal_id] = signal_counts.get(signal_id, 0) + 1
        if error_descriptions and signal_id in error_descriptions:
            signal_descriptions[signal_id] = error_descriptions[signal_id]

    if accuracy is not None:
        topics = updated.setdefault("topics", {})
        if topic_id in topics:
            topics[topic_id] = round(float(topics[topic_id]) * 0.8 + accuracy * 0.2, 2)
        else:
            topics[topic_id] = round(accuracy, 2)
        updated["strengths"] = [
            key for key, score in topics.items() if float(score) >= 0.7
        ]
        updated["weaknesses"] = [
            key for key, score in topics.items() if float(score) < 0.7
        ]
    updated["latest_update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return updated
