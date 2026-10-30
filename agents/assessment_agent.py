import json
import os
import random
import re
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, AIMessage

from agents.llm_factory import get_llm

# The quiz bank (455 MCQs over all 13 topics). Its format - prob_id, prob_body, options as a
# {"A": ...} dict, correct_option as a letter - is what the quiz form in app.py and the grading in
# workflow.py expect. (The diagnosis questions use a different format and are only for diagnosis.)
MCQS_FILEPATH = "data/common/mcqs_list.json"
TOPICS_FILEPATH = "data/common/topics.json"


def load_all_mcqs() -> List[Dict[str, Any]]:
    if not os.path.exists(MCQS_FILEPATH):
        return []
    with open(MCQS_FILEPATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_topic_id_from_name(topic_query: str) -> str:
    """Resolve a topic given as "T08", "T08: Lists, Tuples, and Sets", "Lists, Tuples, and Sets"
    or a word from the topic name such as "lists" to its topic id. Returns the input unchanged
    if nothing matches."""
    query = str(topic_query or "").strip()
    match = re.search(r"\bT\d{2}\b", query, re.IGNORECASE)   # the orchestrator sends "T08: <name>"
    if match:
        return match.group(0).upper()
    if not os.path.exists(TOPICS_FILEPATH):
        return query
    try:
        with open(TOPICS_FILEPATH, "r", encoding="utf-8") as f:
            topics = json.load(f)
    except Exception:
        return query
    lowered = query.lower()
    for t in topics:
        if lowered in (t["topic_id"].lower(), t["name"].lower()):
            return t["topic_id"]
    for t in topics:                       # partial match, e.g. "lists" -> "Lists, Tuples, and Sets"
        if lowered and (lowered in t["name"].lower() or t["name"].lower() in lowered):
            return t["topic_id"]
    return query


def determine_target_difficulty(student_kb: Optional[dict], topic_focus: str) -> float:
    if not student_kb or not student_kb.get("topics"):
        return 0.3  

    scores = student_kb.get("topics", {})
    strengths = set(student_kb.get("strengths", []))
    weaknesses = set(student_kb.get("weaknesses", []))

    score = 0.5
    for tid, s in scores.items():
        if tid.lower() in topic_focus.lower() or topic_focus.lower() in tid.lower():
            score = float(s)
            break

    if score >= 0.8 or topic_focus in strengths:
        return 0.9  
    elif score < 0.4 or topic_focus in weaknesses:
        return 0.2  
    else:
        return 0.5  


def select_adaptive_mcqs(topic_focus: str, student_kb: Optional[dict], limit: int = 10) -> List[Dict[str, Any]]:
    all_mcqs = load_all_mcqs()
    target_tid = get_topic_id_from_name(topic_focus)
    target_diff = determine_target_difficulty(student_kb, topic_focus)

    filtered = []
    for q in all_mcqs:
        related_tops = [str(t).lower() for t in q.get("related_topics", [])]
        related_subs = [str(s).lower() for s in q.get("related_subtopics", [])]
        
        if target_tid.lower() in related_tops or any(target_tid.lower() in sub for sub in related_subs) or not related_tops:
            filtered.append(q)

    if len(filtered) < limit:
        filtered = all_mcqs

    filtered.sort(key=lambda x: abs(float(x.get("difficulty", 0.5)) - target_diff))

    selected = filtered[:limit * 2]
    random.shuffle(selected)
    
    final_selection = selected[:limit]
    
    if len(final_selection) < limit and len(all_mcqs) >= limit:
        remaining = [q for q in all_mcqs if q not in final_selection]
        final_selection.extend(random.sample(remaining, min(limit - len(final_selection), len(remaining))))

    return final_selection


def clean_llm_output(raw_text: str) -> str:
    """Return only the model's answer.

    langchain-huggingface ignores the pipeline_kwargs set in llm_factory.py, so the pipeline returns
    the full text: the whole chat prompt (<|im_start|>system ... ) followed by the answer, which may
    also contain a <think> block. Keep what follows the last assistant marker, without the thinking."""
    text = raw_text or ""
    if "<|im_start|>assistant" in text:
        text = text.rsplit("<|im_start|>assistant", 1)[1]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "<think>" in text:                  # thinking that was cut off before it closed
        text = text.split("<think>")[0]
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    return text.strip()


def keep_introduction_only(text: str) -> str:
    """The questions are shown by the quiz form, so drop anything from the point where the model
    starts writing questions of its own (e.g. "Question 1", "**Q2**", "1. What ...")."""
    question_start = re.compile(r"^\s*(#+\s*)?(\*\*)?\s*(Question\s*\d+|Q\s*\d+\b|\d+[.)]\s)", re.IGNORECASE)
    kept = []
    for line in text.split("\n"):
        if question_start.match(line):
            break
        kept.append(line)
    return "\n".join(kept).strip()


ASSESSMENT_SYSTEM_PROMPT = """You are an expert AI Assessment Coach.
The student is taking a personalized quiz on: {topic_focus}.
Student Mastery Score: {mastery_score} (Target Test Difficulty: {difficulty_label}).
Orchestrator Directive: {orchestrator_instruction}

Introduce the assessment encouragingly, explaining that this quiz is tailored to challenge them appropriately based on their current knowledge profile.
Write only a short introduction of 2-3 sentences. Do NOT write any quiz questions or answers yourself:
the questions are shown to the student separately.
"""


def create_assessment_node():
    llm = get_llm("quiz", enable_thinking=False)

    def assessment_node(state: dict) -> dict:
        student_id = state.get("student_id", "Unknown")
        student_kb = state.get("student_kb")
        topic_focus_list = state.get("topic_focus", ["Python Basics"])
        topic_focus = topic_focus_list[0] if topic_focus_list else "Python Basics"
        instruction = state.get("orchestrator_instruction", "Provide a practice quiz.")

        scores = student_kb.get("topics", {}) if student_kb else {}
        topic_id = get_topic_id_from_name(topic_focus)
        score = scores.get(topic_id, 0.5)
        
        if score >= 0.8:
            diff_label = "Advanced / Hard"
        elif score < 0.4:
            diff_label = "Foundational / Easy"
        else:
            diff_label = "Intermediate / Medium"

        quiz_questions = select_adaptive_mcqs(topic_focus, student_kb, limit=10)

        sys_msg = SystemMessage(
            content=ASSESSMENT_SYSTEM_PROMPT.format(
                topic_focus=topic_focus,
                mastery_score=score,
                difficulty_label=diff_label,
                orchestrator_instruction=instruction,
            )
        )

        response = llm.invoke([sys_msg] + state["messages"])
        intro_text = keep_introduction_only(clean_llm_output(response.content))
        if not intro_text:
            intro_text = (f"Here is a quiz tailored to your current level on {topic_focus}. "
                          "Take your time and good luck!")

        print(f"\n🎯 [ASSESSMENT AGENT] Generated 10 MCQs for Topic: {topic_focus} | Target Difficulty: {diff_label}")

        return {
            "messages": [AIMessage(content=intro_text)],
            "active_quiz_questions": quiz_questions,  
            "last_active_agent": "assessment_agent",
        }

    return assessment_node