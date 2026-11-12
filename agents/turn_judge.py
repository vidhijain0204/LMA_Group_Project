"""Turn judge: one short LLM call that classifies a student message before it is routed.

    kind       "answer"        the message answers the question the tutor just asked
               "new_question"  the student asks something new instead of answering
               "other"         thanks, greetings, quiz requests, ...
    verdict    "correct" / "partial" / "incorrect" for an answer, else None
    confused   the student says or shows they do not understand
    asks_hint  the student asks for a hint, the answer or the solution

The judge replies in JSON (temperature 0, "judge" profile). If the call fails or the JSON cannot
be parsed, rule-based values are used instead (source "rules"). The rule values are always kept
next to the judge's ("rule_flags") so the two can be compared during evaluation.

Set TUTOR_LLM_JUDGE=0 to use only the rules (no extra LLM call per message).
"""

import json
import os
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.struggle_tracker import detect_struggle

KINDS = ("answer", "new_question", "other")
VERDICTS = ("correct", "partial", "incorrect")

JUDGE_SYSTEM_PROMPT = """You analyse ONE message that a student sent to a Python tutor.
Reply with a single JSON object and nothing else:
{"kind": "answer" | "new_question" | "other",
 "verdict": "correct" | "partial" | "incorrect" | null,
 "confused": true | false,
 "asks_hint": true | false}

Definitions:
- kind "answer": the message tries to answer the tutor's question below (even if wrong, unsure or very short, e.g. "None", "it prints 5").
- kind "new_question": the student asks about something instead of answering (e.g. "what is the difference between sort and count?").
- kind "other": anything else (thanks, greetings, asking for a quiz).
- If the tutor asked no question, kind is never "answer".
- verdict: only when kind is "answer": is the answer correct Python knowledge for that question? Use "partial" when it is incomplete. Otherwise null.
- confused: the student says or clearly shows they do not understand or are stuck.
- asks_hint: the student asks for a hint, the answer or the solution."""

JUDGE_USER_TEMPLATE = """Tutor's last question to the student: {question}

Student's message: {message}"""

QUESTION_START = re.compile(
    r"^\s*(what|why|how|when|where|which|who|can|could|should|is|are|does|do|explain|tell me|difference)\b",
    re.IGNORECASE,
)
QUIZ_REQUEST = re.compile(r"\b(quiz|test me|assess\w*|exam|ready for)\b", re.IGNORECASE)


def judge_enabled() -> bool:
    return os.environ.get("TUTOR_LLM_JUDGE", "1") != "0"


def rule_analysis(question: str, message: str) -> dict:
    """Rule-based fallback (also stored for comparison)."""
    flags = detect_struggle(message)
    text = str(message or "").strip()
    if QUIZ_REQUEST.search(text):
        kind = "other"
    elif not question:
        kind = "new_question"
    elif QUESTION_START.match(text) and (len(text.split()) > 4 or (text.endswith("?") and len(text.split()) > 3)):
        kind = "new_question"           # "what is the difference between sort and count method"
    else:
        kind = "answer"
    return {"kind": kind, "verdict": None, "confused": flags["confusion"], "asks_hint": flags["hint_request"]}


def parse_judgement(raw: str) -> Optional[dict]:
    content = re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.DOTALL)
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(content[start:end + 1])
    except ValueError:
        return None
    kind = str(data.get("kind", "")).strip().lower()
    if kind not in KINDS:
        return None
    verdict = data.get("verdict")
    verdict = str(verdict).strip().lower() if verdict else None
    return {
        "kind": kind,
        "verdict": verdict if kind == "answer" and verdict in VERDICTS else None,
        "confused": bool(data.get("confused")) if not isinstance(data.get("confused"), str)
        else data["confused"].lower() == "true",
        "asks_hint": bool(data.get("asks_hint")) if not isinstance(data.get("asks_hint"), str)
        else data["asks_hint"].lower() == "true",
    }


def analyze_turn(llm, question: str, message: str) -> dict:
    """Judge's analysis of the message, falling back to the rules."""
    rules = rule_analysis(question, message)
    result = None
    error = None
    if llm is not None and judge_enabled():
        try:
            response = llm.invoke([
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=JUDGE_USER_TEMPLATE.format(
                    question=question or "(none - the tutor asked no question)", message=message)),
            ])
            result = parse_judgement(getattr(response, "content", response))
            if result is None:
                error = "unparseable judge output"
        except Exception as exc:              # the judge must never break the tutor
            error = f"{type(exc).__name__}: {exc}"
    if result is not None and not question and result["kind"] == "answer":
        result["kind"] = "new_question"
    analysis = dict(result or rules)
    analysis["source"] = "llm" if result is not None else "rules"
    analysis["rule_flags"] = rules
    if error:
        analysis["judge_error"] = error[:200]
    return analysis
