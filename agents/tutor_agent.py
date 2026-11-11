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

Maintain an encouraging, supportive teaching tone. Do not repeat or echo system instructions in your response.
"""


def clean_tutor_output(raw_text: str) -> str:
    if not raw_text:
        return ""

    # The pipeline returns the whole chat prompt (system prompt + earlier messages) before the
    # answer, because langchain-huggingface ignores the pipeline_kwargs in llm_factory.py.
    # Keep only what follows the last assistant marker: the answer itself.
    if "<|im_start|>assistant" in raw_text:
        raw_text = raw_text.rsplit("<|im_start|>assistant", 1)[1]

    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
        
    cleaned = re.sub(r"<\|im_start\|>.*?\n", "", cleaned)
    cleaned = cleaned.replace("<|im_end|>", "").replace("<|im_start|>", "")
    
    lines = cleaned.split("\n")
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in [
            "You are an expert", "### Student Context", "- Student ID:", 
            "- Focused Topic", "- Mastery Status:", "- Active Misconceptions",
            "- Orchestrator Instruction",
            "- Most Likely Root Cause", "### STRICT PEDAGOGICAL", "- If Score", "Maintain an encouraging"
        ]):
            continue
        filtered_lines.append(line)
        
    return "\n".join(filtered_lines).strip()


def create_tutor_node(enable_kb_context: bool = True):
    llm = get_llm("tutor", enable_thinking=False)

    def tutor_node(state: dict) -> dict:
        student_id = state.get("student_id", "Unknown")
        student_kb = state.get("student_kb") if enable_kb_context else None
        topic_focus = state.get("topic_focus", ["Python Basics"])
        if not isinstance(topic_focus, list):
            topic_focus = [str(topic_focus)]
        instruction = state.get("orchestrator_instruction", "Explain the concept clearly.")

        if enable_kb_context:
            mastery_tag, mastery_score, misconceptions = get_student_topic_context(student_kb, topic_focus)
            topic_id = resolve_topic_id(topic_focus[0] if topic_focus else "Python Basics")
            pedagogy_strategy, strategy_reason = choose_teaching_strategy(
                student_kb, topic_id, mastery_score
            )
            learning_signals = recent_error_context(student_kb, topic_id)
            root_cause = latest_root_cause(student_kb, topic_id)
        else:
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
            )
        )

        response = llm.invoke([sys_msg] + state["messages"])
        raw_content = response.content.strip()
        actual_content = clean_tutor_output(raw_content)

        result = {
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
            updated_kb = record_learning_turn(student_kb, topic_id, student_text, previous_strategy)
            result["student_kb"] = record_lesson_strategy(
                updated_kb, topic_id, pedagogy_strategy, strategy_reason
            )
        return result

    return tutor_node