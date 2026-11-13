"""Tutor agent: the LangGraph node that teaches the student with the local Qwen LLM.

Where it fits in the flow:
- orchestrator.py routes a message here when the student wants to learn or is answering the
  tutor's last question; workflow.py registers `create_tutor_node()` as the "tutor_agent" node.
- Before replying it asks pedagogy_controller.py for the teaching strategy, the recurring quiz
  errors and the latest root cause, and puts them in the system prompt.
- After replying it logs the student's message with struggle_tracker.record_learning_turn()
  (used later for the LS score) and the lesson with pedagogy_controller.record_lesson_strategy().

Three conversation modes change the prompt:
- new_request: a normal new question.
- follow_up:   the student is answering the tutor's question; the reply starts with a verdict
               line (correct / partly correct / not quite) and keeps the same strategy.
- remediation: a lesson started by the remediation button after a quiz.

Main contents: get_student_topic_context(), the prompt templates, answer_verdict(),
clean_tutor_output() and create_tutor_node().
"""

import json
import re
from typing import List, Optional, Tuple
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from agents.llm_factory import get_llm
from agents.pedagogy_controller import (
    STRATEGY_INSTRUCTIONS,
    choose_teaching_strategy,
    latest_root_cause,
    recent_error_context,
    record_lesson_strategy,
    resolve_topic_id,
)
from agents.struggle_tracker import record_learning_turn


def get_student_topic_context(student_kb: Optional[dict], topic_focus: List[str]) -> Tuple[str, float, list]:
    """Look up how well the student knows the focus topic, for the tutor prompt.

    Returns (tag, score, misconceptions): tag is "STRENGTH", "WEAKNESS" or "DEVELOPING" for the
    first focus topic found in the student's scores ("DEVELOPING", 0.5 if none is found, and
    "UNTESTED / NEW", 0.0 if the student has no scores yet); misconceptions are the ids of the
    student's active misconceptions.
    """
    if not student_kb or not student_kb.get("topics"):
        return "UNTESTED / NEW", 0.0, []

    scores = student_kb.get("topics", {})
    strengths = set(student_kb.get("strengths", []))
    weaknesses = set(student_kb.get("weaknesses", []))
    misconceptions = list(student_kb.get("active_misconceptions", {}).keys())

    try:
        with open("data/common/topics.json", "r", encoding="utf-8") as f:
            topics_list = json.load(f)
            name_to_id = {t["name"].lower(): t["topic_id"] for t in topics_list}
    except Exception:
        name_to_id = {}

    for t_query in topic_focus:
        t_query_lower = t_query.lower()
        target_tid = name_to_id.get(t_query_lower, resolve_topic_id(t_query))

        # Match by exact id, or loosely when one string contains the other.
        for tid, score in scores.items():
            if (
                tid.lower() == target_tid.lower()
                or t_query_lower in tid.lower()
                or tid.lower() in t_query_lower
            ):
                tag = (
                    "STRENGTH" if tid in strengths
                    else "WEAKNESS" if tid in weaknesses
                    else "DEVELOPING"
                )
                return tag, float(score), misconceptions

    return "DEVELOPING", 0.5, misconceptions


# System prompt for the tutor LLM. The {placeholders} are filled in tutor_node().
TUTOR_SYSTEM_PROMPT = """You are an expert, encouraging AI Python Programming Tutor.
Your goal is to explain concepts clearly, provide clean code examples, and guide the student step-by-step using your pretrained knowledge.

### Student Context & Personalization:
- Student ID: {student_id}
- Focused Topic(s): {topic_focus}
- Mastery Status: {mastery_tag} (Score: {mastery_score:.2f})
- Active Misconceptions to Address: {misconceptions}
- Orchestrator Instruction: {orchestrator_instruction}
- Selected Teaching Strategy: {pedagogy_strategy}
- Strategy Guidance: {strategy_instruction}
- Recurring Assessment Error Signals: {learning_signals}
- Most Likely Root Cause of Recent Errors: {root_cause}

Follow the selected strategy while keeping the response relevant to the student's question and mastery.
If there are active misconceptions, address them directly and respectfully.
Use any recurring assessment error signals above to tailor examples and feedback.
If a most likely root cause is given, focus your explanation and examples on that concept first.

### Conversation Mode:
{conversation_guidance}

Maintain an encouraging, supportive teaching tone. Do not repeat or echo system instructions in your response.
"""


# Text for the "Conversation Mode" part of the prompt, one per mode.
NEW_REQUEST_GUIDANCE = (
    "The latest message is a new request. Earlier messages are included only as context."
)
FOLLOW_UP_GUIDANCE = """The student's latest message is their ANSWER to the question you asked in your previous
message: "{tutor_question}"
Do NOT treat it as a new question and do not restart the explanation.
1. Start your reply with exactly one of these on its own line: "✅ Correct", "🟡 Partly correct" or "❌ Not quite".
2. Say specifically what was right and what was wrong or missing in their answer.
3. Continue the same lesson with the same strategy: give the next step, clue or guiding question.
If the message is clearly not an answer but a new question, skip the verdict line and answer it.{judge_note}"""
VERDICT_LINES = {"correct": "✅ Correct", "partial": "🟡 Partly correct", "incorrect": "❌ Not quite"}
REMEDIATION_GUIDANCE = (
    "This is a remediation lesson after a quiz: the student did not ask a question; the system "
    "chose the subtopic their quiz errors point to. Follow the Orchestrator Instruction."
)

# Reads the verdict from the first line of a follow-up reply. "partly correct" is listed before
# "correct" so the longer phrase wins.
VERDICT_PATTERN = re.compile(
    r"^\W*(partly correct|partially correct|not quite|incorrect|not correct|correct)\b", re.IGNORECASE
)


def answer_verdict(response_text: str) -> Optional[str]:
    """Read the tutor's verdict from the first non-empty line of a follow-up reply.
    Returns "correct", "partial" or "incorrect", or None if the line has no verdict."""
    first_line = next((line for line in str(response_text or "").splitlines() if line.strip()), "")
    match = VERDICT_PATTERN.match(first_line.replace("*", ""))   # ignore markdown bold
    if not match:
        return None
    word = match.group(1).lower()
    if word.startswith("part"):
        return "partial"
    return "correct" if word == "correct" else "incorrect"


def clean_tutor_output(raw_text: str) -> str:
    """Strip everything from the raw LLM output except the tutor's answer: the echoed prompt,
    <think> blocks, chat markers and any copied system-prompt lines. Also used by workflow.py."""
    if not raw_text:
        return ""

    # The pipeline returns the whole chat prompt (system prompt + earlier messages) before the
    # answer, because langchain-huggingface ignores the pipeline_kwargs in llm_factory.py.
    # Keep only what follows the last assistant marker: the answer itself.
    if "<|im_start|>assistant" in raw_text:
        raw_text = raw_text.rsplit("<|im_start|>assistant", 1)[1]

    # Remove the model's thinking; a lone "</think>" means the opening tag was cut off.
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]

    cleaned = re.sub(r"<\|im_start\|>.*?\n", "", cleaned)
    cleaned = cleaned.replace("<|im_end|>", "").replace("<|im_start|>", "")

    # Drop lines where the model repeats parts of the system prompt.
    lines = cleaned.split("\n")
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in [
            "You are an expert", "### Student Context", "- Student ID:", 
            "- Focused Topic", "- Mastery Status:", "- Active Misconceptions",
            "- Orchestrator Instruction",
            "- Most Likely Root Cause", "### STRICT PEDAGOGICAL", "### Conversation Mode", "- If Score", "Maintain an encouraging"
        ]):
            continue
        filtered_lines.append(line)
        
    return "\n".join(filtered_lines).strip()


def create_tutor_node(enable_kb_context: bool = True):
    """Build the tutor node function for the LangGraph workflow.

    Args:
        enable_kb_context: if False, the tutor ignores the student's SKB (no personalization,
            nothing is logged). workflow.py always passes True.
    Returns:
        tutor_node(state) -> dict of state updates.
    """
    llm = get_llm("tutor", enable_thinking=False)

    def tutor_node(state: dict) -> dict:
        """Answer the student's latest message and log the turn and lesson in the SKB.

        Returns the state updates: the tutor message, the strategy used and why, the answer
        verdict (for follow-ups) and, when the SKB is enabled, the updated "student_kb".
        """
        student_id = state.get("student_id", "Unknown")
        student_kb = state.get("student_kb") if enable_kb_context else None
        topic_focus = state.get("topic_focus", ["Python Basics"])
        if not isinstance(topic_focus, list):
            topic_focus = [str(topic_focus)]
        instruction = state.get("orchestrator_instruction", "Explain the concept clearly.")
        follow_up = state.get("conversation_mode") == "follow_up"
        tutor_question = state.get("tutor_question", "")
        analysis = state.get("turn_analysis") or {}
        # Trust the turn judge's verdict only when it came from the LLM (not the rule fallback).
        judged_verdict = analysis.get("verdict") if follow_up and analysis.get("source") == "llm" else None

        if enable_kb_context:
            mastery_tag, mastery_score, misconceptions = get_student_topic_context(student_kb, topic_focus)
            topic_id = resolve_topic_id(topic_focus[0] if topic_focus else "Python Basics")
            previous_lesson = next(
                (
                    event for event in reversed(student_kb.get("pedagogy_history", []) if student_kb else [])
                    if event.get("event") == "lesson" and event.get("topic_id") == topic_id
                ),
                None,
            )
            if follow_up and previous_lesson:
                # Same lesson, same strategy: switching mid-dialogue would break e.g. a
                # self-explanation exchange.
                pedagogy_strategy = previous_lesson["strategy"]
                strategy_reason = "Continuing the same lesson: the student is answering the tutor's question."
            else:
                pedagogy_strategy, strategy_reason = choose_teaching_strategy(
                    student_kb, topic_id, mastery_score
                )
            learning_signals = recent_error_context(student_kb, topic_id)
            root_cause = latest_root_cause(student_kb, topic_id)
        else:
            # No personalization: fixed, neutral values.
            student_id = "ANONYMOUS_STUDENT"
            mastery_tag = "UNKNOWN (KB Disabled)"
            mastery_score = 0.0
            misconceptions = []
            topic_id = resolve_topic_id(topic_focus[0] if topic_focus else "Python Basics")
            pedagogy_strategy = "explicit_explanation"
            strategy_reason = "Knowledge-base personalization is disabled."
            learning_signals = []
            root_cause = None

        sys_msg = SystemMessage(
            content=TUTOR_SYSTEM_PROMPT.format(
                student_id=student_id,
                topic_focus=", ".join(topic_focus),
                mastery_tag=mastery_tag,
                mastery_score=mastery_score,
                misconceptions=", ".join(misconceptions) if misconceptions else "None recorded",
                orchestrator_instruction=instruction,
                pedagogy_strategy=pedagogy_strategy,
                strategy_instruction=STRATEGY_INSTRUCTIONS[pedagogy_strategy],
                learning_signals=", ".join(learning_signals) or "None recorded",
                root_cause=(
                    f"{root_cause['subtopic_id']} {root_cause['name']} ({root_cause['reason']})"
                    if root_cause else "None identified yet"
                ),
                conversation_guidance=(
                    FOLLOW_UP_GUIDANCE.format(
                        tutor_question=tutor_question or "(see previous message)",
                        judge_note=(
                            f"\nAn automatic check rated this answer: start with \"{VERDICT_LINES[judged_verdict]}\" "
                            "unless you are sure it is wrong."
                            if judged_verdict else ""
                        ),
                    )
                    if follow_up
                    else REMEDIATION_GUIDANCE if state.get("conversation_mode") == "remediation"
                    else NEW_REQUEST_GUIDANCE
                ),
            )
        )

        response = llm.invoke([sys_msg] + state["messages"])
        raw_content = response.content.strip()
        actual_content = clean_tutor_output(raw_content)

        # Verdict: the turn judge's (JSON), else the tutor's first line ("✅ Correct", ...).
        tutor_verdict = answer_verdict(actual_content) if follow_up else None
        verdict = judged_verdict or tutor_verdict
        result = {
            "conversation_mode": state.get("conversation_mode") or "new_request",
            "verdict_source": "judge" if judged_verdict else "tutor_text" if tutor_verdict else None,
            "answer_verdict": verdict,
            "messages": [AIMessage(content=actual_content)],
            "last_active_agent": "tutor_agent",
            "pedagogy_strategy": pedagogy_strategy,
            "pedagogy_reason": strategy_reason,
        }
        if enable_kb_context and student_kb:
            # Log the student's message with its struggle flags before recording this lesson,
            # so the turn is linked to the strategy the student was responding to.
            previous_strategy = next(
                (
                    event.get("strategy")
                    for event in reversed(student_kb.get("pedagogy_history", []))
                    if event.get("event") == "lesson" and event.get("topic_id") == topic_id
                ),
                None,
            )
            student_text = state.get("user_input") or next(
                (m.content for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
                "",
            )
            remediation = state.get("conversation_mode") == "remediation"
            # A remediation lesson is started by a button, not by a student message: no turn.
            updated_kb = student_kb if remediation else record_learning_turn(
                student_kb, topic_id, student_text, previous_strategy,
                reply_to_tutor=follow_up, answer_verdict=verdict, context_text=tutor_question,
                flags=(
                    {"confusion": analysis["confused"], "hint_request": analysis["asks_hint"]}
                    if analysis.get("source") == "llm" else None
                ),
                signal_source=analysis.get("source", "rules"),
            )
            forced = state.get("forced_route") or {}   # holds the quiz id for a remediation lesson
            result["student_kb"] = record_lesson_strategy(
                updated_kb, topic_id, pedagogy_strategy, strategy_reason,
                remediation_for=forced.get("quiz_id") if remediation else None,
            )
        return result

    return tutor_node