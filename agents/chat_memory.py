"""Short-term chat memory: the latest 3 exchanges per student, stored as summaries.

Each exchange (student message + tutor reply, or a quiz) is kept in the student's knowledge
base under "chat_memory" as a small summary instead of the full text:

    lesson: {"type": "lesson", "time", "topic_id", "strategy", "mode", "verdict",
             "student": <message, max 200 chars>,
             "covered": <section headings / key terms of the reply, max 250 chars>,
             "question": <the question the tutor ended with, or "">}
    quiz:   {"type": "quiz", "time", "topics", "score", "reason", "student"}

Only the latest CHAT_MEMORY_SIZE entries are kept. These summaries are what the orchestrator
and the tutor see as earlier messages, so prompts stay short (useful for a 4B model).
The tutor's closing question is kept word for word because the follow-up detection needs it.
"""

import re
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage

CHAT_MEMORY_SIZE = 3
STUDENT_CHARS = 200
COVERED_CHARS = 250
QUESTION_CHARS = 200

CODE_BLOCK = re.compile(r"```.*?(```|$)", re.DOTALL)
BOLD = re.compile(r"\*\*(.+?)\*\*")
SUMMARY_PREFIX = "[Summary of earlier tutor reply]"


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _unique(items: List[str]) -> List[str]:
    seen, result = set(), []
    for item in items:
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def closing_question(tutor_text: str) -> str:
    """The question the reply ends with (one of its last 3 non-heading lines), or ""."""
    lines = [
        line.strip() for line in CODE_BLOCK.sub("", str(tutor_text or "")).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ][-3:]
    question = next((line for line in reversed(lines) if "?" in line), "")
    return _clip(question, QUESTION_CHARS)


def key_points(tutor_text: str) -> str:
    """What the reply covered: its section headings, else bold terms, else its first sentence."""
    text = CODE_BLOCK.sub(" ", str(tutor_text or ""))
    headings = [
        line.lstrip("#").strip(" :*")
        for line in text.splitlines() if line.lstrip().startswith("#")
    ]
    points = _unique(headings) or _unique(m.strip() for m in BOLD.findall(text))
    # No "?" here: only the closing question may look like a question to the follow-up check.
    points = [p.replace("?", "") for p in points]
    if not points:
        first = re.split(r"(?<=[.!?])\s", " ".join(text.split()), maxsplit=1)[0]
        points = [first.replace("?", "")] if first else []
    has_code = "```" in str(tutor_text or "")
    summary = "; ".join(points)
    if has_code:
        summary = f"{summary} (with code example)" if summary else "code example"
    return _clip(summary, COVERED_CHARS)


def lesson_entry(student_text: str, tutor_text: str, topic_id: str, strategy: str,
                 mode: str = "new_request", verdict: Optional[str] = None) -> dict:
    return {
        "type": "lesson",
        "time": datetime.now().isoformat(timespec="seconds"),
        "topic_id": topic_id,
        "strategy": strategy,
        "mode": mode,
        "verdict": verdict,
        "student": _clip(student_text, STUDENT_CHARS),
        "covered": key_points(tutor_text),
        "question": closing_question(tutor_text),
    }


def quiz_entry(student_text: str, topics: List[str], score: str, reason: str) -> dict:
    return {
        "type": "quiz",
        "time": datetime.now().isoformat(timespec="seconds"),
        "topics": list(topics or []),
        "score": score,
        "reason": _clip(reason, COVERED_CHARS),
        "student": _clip(student_text, STUDENT_CHARS),
    }


def remember(student_kb: Optional[dict], entry: dict) -> Optional[dict]:
    """Add an exchange to the KB, keeping only the latest CHAT_MEMORY_SIZE."""
    if not student_kb:
        return student_kb
    memory = list(student_kb.get("chat_memory") or []) + [entry]
    student_kb["chat_memory"] = memory[-CHAT_MEMORY_SIZE:]
    return student_kb


def summary_text(entry: dict) -> str:
    """The tutor side of a stored exchange, as shown to the LLMs."""
    if entry.get("type") == "quiz":
        return (f"[Summary of earlier quiz] Topics: {', '.join(entry.get('topics') or [])}; "
                f"score {entry.get('score')}; main error reason: {entry.get('reason') or 'none'}.")
    lines = [f"{SUMMARY_PREFIX} Topic {entry.get('topic_id')}, strategy {entry.get('strategy')}."]
    if entry.get("verdict"):
        lines.append(f"Verdict on the student's answer: {entry['verdict']}.")
    if entry.get("covered"):
        lines.append(f"Covered: {entry['covered']}")
    if entry.get("question"):
        lines.append(f"Asked the student: {entry['question']}")   # last line, for follow-ups
    return "\n".join(lines)


def history_messages(student_kb: Optional[dict]) -> list:
    """Stored exchanges as chat messages (student message, summarised tutor reply)."""
    messages = []
    for entry in (student_kb or {}).get("chat_memory") or []:
        messages.append(HumanMessage(content=entry.get("student") or "(no message)"))
        messages.append(AIMessage(content=summary_text(entry)))
    return messages
