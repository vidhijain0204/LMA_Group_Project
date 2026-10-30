import json
import re
from typing import List, Optional, Tuple
from langchain_core.messages import SystemMessage, AIMessage

from agents.llm_factory import get_llm


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
        target_tid = name_to_id.get(t_query_lower, t_query_lower)

        for tid, score in scores.items():
            if tid.lower() == target_tid or t_query_lower in tid.lower() or tid.lower() in t_query_lower:
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

### STRICT PEDAGOGICAL RULES BASED ON SCORE:
- If Score < 0.4 (WEAKNESS/NEW): Do NOT give standard explanations. Use a simple, real-world physical analogy first, break down the syntax piece-by-piece, and use a very basic code example. Proactively warn against any listed misconceptions.
- If Score ≥ 0.8 (STRENGTH): SKIP all basic definitions. Immediately present an advanced edge case, a common bug/pitfall, or an optimization challenge.
- If Score is DEVELOPING (0.4 to 0.79): Provide standard explanations, intermediate code, and clear guidance.

Maintain an encouraging, supportive teaching tone. Do not repeat or echo system instructions in your response.
"""


def clean_tutor_output(raw_text: str) -> str:
    if not raw_text:
        return ""
    
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
            "- Orchestrator Instruction", "### STRICT PEDAGOGICAL", "- If Score", "Maintain an encouraging"
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
        instruction = state.get("orchestrator_instruction", "Explain the concept clearly.")

        if enable_kb_context:
            mastery_tag, mastery_score, misconceptions = get_student_topic_context(student_kb, topic_focus)
        else:
            student_id = "ANONYMOUS_STUDENT"
            mastery_tag = "UNKNOWN (KB Disabled)"
            mastery_score = 0.0
            misconceptions = []

        sys_msg = SystemMessage(
            content=TUTOR_SYSTEM_PROMPT.format(
                student_id=student_id,
                topic_focus=", ".join(topic_focus),
                mastery_tag=mastery_tag,
                mastery_score=mastery_score,
                misconceptions=", ".join(misconceptions) if misconceptions else "None recorded",
                orchestrator_instruction=instruction,
            )
        )

        response = llm.invoke([sys_msg] + state["messages"])
        raw_content = response.content.strip()
        actual_content = clean_tutor_output(raw_content)

        return {
            "messages": [AIMessage(content=actual_content)],
            "last_active_agent": "tutor_agent",
        }

    return tutor_node