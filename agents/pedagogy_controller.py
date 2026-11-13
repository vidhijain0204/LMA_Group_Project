"""Pedagogy controller: chooses how the tutor teaches and keeps the student's scores up to date.

Where it fits in the flow:
- tutor_agent.py calls `choose_teaching_strategy()` before each reply and
  `record_lesson_strategy()` after it, so every lesson is logged in the student's SKB
  (data/skb_database.json) under "pedagogy_history".
- workflow.py calls `record_assessment_outcome()` and `update_subtopic_mastery()` after a quiz
  has been graded (error_analysis_agent.py) and explained (root_cause_agent.py).
- tutor_agent.py also reads `recent_error_context()` and `latest_root_cause()` to tell the LLM
  which errors keep coming back and what the root cause of the last quiz was.

Main contents:
- STRATEGY_INSTRUCTIONS: the six teaching strategies and the text sent to the LLM for each one.
- choose_teaching_strategy(): keep the current strategy, or switch it when quizzes show it is
  not working (accuracy below 60% on 2 quizzes, or the same errors repeating).
- record_assessment_outcome(): store a quiz result and update the topic mastery score.
- update_subtopic_mastery(): per-subtopic mastery, updated after every quiz answer.
- resolve_topic_id(), compact_pedagogy_history(): small helpers used by other modules too.
"""

from collections import Counter
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# Teaching strategies the tutor can use. The text is added to the tutor's system prompt.
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

# Order in which strategies are tried when the current one fails (wraps around at the end).
STRATEGY_CYCLE = tuple(STRATEGY_INSTRUCTIONS)
# Switch strategy only if the average accuracy of the last 2 quizzes taught with it is below 60%.
MIN_STRATEGY_ASSESSMENTS = 2
LOW_ACCURACY_THRESHOLD = 0.6


def resolve_topic_id(topic_focus: str) -> str:
    """Turn a topic name or id (e.g. "Loops" or "T06") into its topic id using topics.json.

    Tries an exact id/name match first, then a partial match. Returns "unknown" for empty input
    and the input unchanged if nothing matches or the file is missing.
    """
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
    """True if any error signal appears in at least 2 of the last 3 quizzes."""
    recent = assessments[-3:]
    counts = Counter(
        signal_id
        for assessment in recent
        for signal_id in set(assessment.get("error_signals", []))
    )
    return any(count >= 2 for count in counts.values())


def recent_error_context(student_kb: Optional[dict], topic_id: str) -> List[str]:
    """Return descriptions of the error signals seen in at least 2 of the last 3 quizzes on
    this topic, most frequent first. The tutor puts them in its prompt."""
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
    """Return the most recent root cause found for a quiz on this topic, or one located in
    this topic. None if there is none."""
    for entry in reversed((student_kb or {}).get("root_cause_history", [])):
        if topic_id in (entry.get("quiz_topic_id"), entry.get("topic_id")):
            return entry
    return None


def choose_teaching_strategy(
    student_kb: Optional[dict],
    topic_id: str,
    mastery_score: float,
) -> Tuple[str, str]:
    """Pick the teaching strategy for the next tutor reply on a topic.

    Rules, checked in order:
    1. The same error signals keep coming back -> "contrastive_feedback" (if not already used).
    2. A strategy is already in use -> keep it, unless its last 2 quizzes averaged below 60%,
       then move to the next strategy in STRATEGY_CYCLE.
    3. No lesson yet -> choose from the mastery score (low: step_scaffolding,
       high: self_explanation, otherwise explicit_explanation).

    Returns:
        (strategy name, short reason shown in the UI and saved in the SKB)
    """
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


def _same_lesson(previous: dict, event: dict) -> bool:
    """True if `event` continues the `previous` lesson: same topic and strategy, and no quiz
    has graded the previous lesson yet."""
    return (
        previous.get("event") == "lesson" and event.get("event") == "lesson"
        and previous.get("topic_id") == event.get("topic_id")
        and previous.get("strategy") == event.get("strategy")
        and "assessment_accuracy" not in previous
    )


def compact_pedagogy_history(history: List[dict]) -> List[dict]:
    """Merge back-to-back lesson events with the same topic and strategy into one event
    with a "turns" count (older KBs stored one lesson event per tutor message)."""
    compacted: List[dict] = []
    for event in history:
        if compacted and _same_lesson(compacted[-1], event):
            merged = {**compacted[-1], **event}
            merged["created_at"] = compacted[-1].get("created_at", event.get("created_at"))
            merged["last_at"] = event.get("last_at") or event.get("created_at")
            merged["turns"] = compacted[-1].get("turns", 1) + event.get("turns", 1)
            compacted[-1] = merged
        else:
            compacted.append(dict(event))
    return compacted


def record_lesson_strategy(
    student_kb: dict,
    topic_id: str,
    strategy: str,
    reason: str,
    remediation_for: Optional[str] = None,
) -> dict:
    """Log a tutor lesson in the SKB's "pedagogy_history" and return the updated copy.

    One lesson event per teaching block: further messages on the same topic with the same
    strategy (and no quiz in between) only increase its "turns" count. `remediation_for` is the
    quiz id when the lesson was started by the remediation button; it always starts a new event.
    """
    updated = deepcopy(student_kb)
    history = compact_pedagogy_history(updated.get("pedagogy_history", []))
    now = datetime.now().isoformat(timespec="seconds")
    event = {"event": "lesson", "topic_id": topic_id, "strategy": strategy, "reason": reason, "created_at": now}
    if remediation_for:
        event["remediation_for"] = remediation_for      # a lesson started by the remediation button
    if history and _same_lesson(history[-1], event) and not remediation_for:
        history[-1].update(reason=reason, last_at=now, turns=history[-1].get("turns", 1) + 1)
    else:
        history.append({**event, "turns": 1})
    updated["pedagogy_history"] = history[-100:]   # keep only the newest 100 events
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
    """Save a graded quiz in the SKB and update the student's topic scores. Called by workflow.py.

    What it does:
    - marks every lesson on this topic since the last quiz with this quiz's accuracy
      (this is how a strategy is judged later in choose_teaching_strategy),
    - appends an "assessment" event (with per-question answers and the root cause),
    - counts each error signal in "learning_signals",
    - updates the topic mastery: new = 0.8 * old + 0.2 * accuracy (first quiz: just accuracy),
      and rebuilds strengths (>= 0.7) and weaknesses (< 0.7).

    Args:
        accuracy: share of correct answers (0..1), or None if it could not be computed.
        error_signals / error_descriptions: error ids from the error analysis and their text.
        question_results: the option chosen for every question (needed for counter-evidence
            and repeat checks later).
        root_cause: the hypothesis root_cause_agent.py selected for this quiz.
    Returns:
        The updated copy of the SKB.
    """
    updated = deepcopy(student_kb)
    history = updated.setdefault("pedagogy_history", [])
    # Every lesson on this topic since the last quiz is evaluated by this quiz.
    pending_lessons = [
        event for event in history
        if event.get("event") == "lesson"
        and event.get("topic_id") == topic_id
        and "assessment_accuracy" not in event
    ]
    strategy = pending_lessons[-1].get("strategy") if pending_lessons else None
    for lesson in pending_lessons:
        lesson["assessment_accuracy"] = accuracy
        lesson["error_signals"] = list(dict.fromkeys(error_signals))

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
                "misconception": result.get("misconception"),
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


# --------------------------------------------------------------- subtopic mastery
# Each quiz answer moves the mastery of every subtopic the question tests towards 1 (correct)
# or 0 (wrong): m <- m + RATE * (outcome - m). This is an exponential moving average, the
# simplest form of knowledge tracing; it keeps mastery per subtopic instead of per topic, so the
# root-cause agent (PM', lowest-mastery method) can tell the subtopics of one topic apart.
SUBTOPIC_LEARNING_RATE = 0.3
DEFAULT_SUBTOPIC_MASTERY = 0.5


def update_subtopic_mastery(
    student_kb: dict, responses: List[dict], rate: float = SUBTOPIC_LEARNING_RATE
) -> Tuple[dict, Dict[str, Dict[str, float]]]:
    """Update the mastery of every subtopic tested in a quiz (stored under "skills" in the SKB).

    A subtopic with no mastery yet starts from its topic's score, or 0.5.
    Returns the updated SKB copy and {subtopic: {"before", "after"}} for the subtopics touched.
    """
    updated = deepcopy(student_kb)
    skills = updated.setdefault("skills", {})
    topics = updated.get("topics", {})
    changes: Dict[str, Dict[str, float]] = {}
    for response in responses:
        outcome = 1.0 if response.get("is_correct") else 0.0
        for sid in response.get("subtopics", []):
            start = skills.get(sid, topics.get(str(sid).split(".")[0], DEFAULT_SUBTOPIC_MASTERY))
            changes.setdefault(sid, {"before": round(float(start), 2)})
            skills[sid] = round(float(start) + rate * (outcome - float(start)), 3)
            changes[sid]["after"] = round(skills[sid], 2)
    return updated, changes
