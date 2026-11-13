"""Assessment agent: the LangGraph node that builds a personalized multiple-choice quiz.

Where it fits in the flow:
- orchestrator.py routes a message here when the student asks for a quiz (or a re-test after a
  remediation lesson); workflow.py registers `create_assessment_node()` as "assessment_agent".
- The node picks questions from the MCQ bank with `select_adaptive_mcqs()` and asks the LLM only
  for a short introduction. The questions themselves are shown by the quiz form in app.py.
- After the student submits, workflow.py grades the answers and passes them on to
  error_analysis_agent.py and root_cause_agent.py (not this file).

How questions are chosen (select_adaptive_mcqs):
- each focus topic gets an equal share; inside a topic the weakest subtopics (lowest mastery
  in the SKB) get the questions, 2 per subtopic,
- a re-test puts all questions on the given subtopics,
- questions close to the target difficulty and not seen before are preferred.
"""

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
    """Load the whole MCQ bank (an empty list if the file is missing). Also used by workflow.py."""
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
    """Pick the question difficulty (0..1) to aim for on a topic, from the student's score:
    0.9 for strong topics (score >= 0.8), 0.2 for weak ones (< 0.4), else 0.5.
    A student with no scores yet gets 0.3 (fairly easy)."""
    if not student_kb or not student_kb.get("topics"):
        return 0.3

    scores = student_kb.get("topics", {})
    strengths = set(student_kb.get("strengths", []))
    weaknesses = set(student_kb.get("weaknesses", []))

    score = 0.5   # default when the topic has no score yet
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


def _covers_topic(question: dict, topic_id: str) -> bool:
    """True if the question is tagged with the topic or with one of its subtopics."""
    return topic_id in question.get("related_topics", []) or any(
        str(sub).startswith(f"{topic_id}.") for sub in question.get("related_subtopics", [])
    )


def split_quota(limit: int, parts: int) -> List[int]:
    """Split `limit` questions as evenly as possible into `parts`: 10 over 3 -> [4, 3, 3]."""
    return [limit // parts + (i < limit % parts) for i in range(parts)]


# At least this many questions per focus subtopic, so one lucky guess or slip does not decide
# the counter-evidence (CE) of a subtopic.
QUESTIONS_PER_SUBTOPIC = 2
RETEST_SIZE = 6   # number of questions in a re-test after remediation (a normal quiz has 10)


def seen_question_ids(student_kb: Optional[dict]) -> set:
    """Return the ids of the questions this student already answered in earlier quizzes."""
    return {
        response.get("prob_id")
        for event in (student_kb or {}).get("pedagogy_history", [])
        if event.get("event") == "assessment"
        for response in event.get("responses", [])
    }


def subtopic_mastery(student_kb: Optional[dict], subtopic_id: str) -> float:
    """The student's mastery of a subtopic; falls back to the topic score, then to 0.5."""
    kb = student_kb or {}
    return float(kb.get("skills", {}).get(subtopic_id, kb.get("topics", {}).get(subtopic_id.split(".")[0], 0.5)))


def select_adaptive_mcqs(topic_focus, student_kb: Optional[dict], limit: int = 10,
                         target_subtopics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Pick `limit` questions from the MCQ bank for this student.

    - Each topic in `topic_focus` gets an equal share; inside a topic the share goes to its
      weakest subtopics (lowest subtopic mastery), QUESTIONS_PER_SUBTOPIC questions each.
    - `target_subtopics` (a re-test after remediation) puts all questions on those subtopics.
    - Questions near the topic's target difficulty are preferred, and questions the student has
      not answered before are preferred over repeats.
    - If there are still too few questions, the rest is filled from the chosen topics and then
      from the whole bank.
    """
    all_mcqs = load_all_mcqs()
    focus = topic_focus if isinstance(topic_focus, list) else [topic_focus]
    topic_ids = list(dict.fromkeys(get_topic_id_from_name(t) for t in focus if t)) or ["T01"]
    seen = seen_question_ids(student_kb)

    selected, used = [], set()

    def take(pool, count, target_diff):
        """Add the `count` best unused questions from `pool` to the quiz."""
        pool = [q for q in pool if q["prob_id"] not in used]
        random.shuffle(pool)       # random order among equally good questions
        # Unseen questions first, then closest to the target difficulty (rounded to 0.1 so
        # that near-equal questions stay in random order).
        pool.sort(key=lambda q: (q["prob_id"] in seen, round(abs(float(q.get("difficulty", 0.5)) - target_diff), 1)))
        for q in pool[:count]:
            selected.append(q)
            used.add(q["prob_id"])

    def with_subtopic(sid):
        """All bank questions tagged with subtopic `sid`."""
        return [q for q in all_mcqs if sid in q.get("related_subtopics", [])]

    plan = []      # (subtopic, number of questions, target difficulty)
    if target_subtopics:
        for sid, quota in zip(target_subtopics, split_quota(limit, len(target_subtopics))):
            plan.append((sid, quota, determine_target_difficulty(student_kb, sid.split(".")[0])))
    else:
        for topic_id, quota in zip(topic_ids, split_quota(limit, len(topic_ids))):
            difficulty = determine_target_difficulty(student_kb, topic_id)
            # The topic's subtopics, weakest first (ties in random order), keeping only those
            # with enough questions in the bank.
            subtopics = sorted(
                {sid for q in all_mcqs for sid in q.get("related_subtopics", []) if sid.startswith(f"{topic_id}.")},
                key=lambda sid: (subtopic_mastery(student_kb, sid), random.random()),
            )
            subtopics = [sid for sid in subtopics if len(with_subtopic(sid)) >= QUESTIONS_PER_SUBTOPIC]
            count = max(1, min(len(subtopics), quota // QUESTIONS_PER_SUBTOPIC))   # subtopics to cover
            if not subtopics:   # no tagged subtopics: take any questions on the topic
                take([q for q in all_mcqs if _covers_topic(q, topic_id)], quota, difficulty)
                continue
            for sid, share in zip(subtopics[:count], split_quota(quota, count)):
                plan.append((sid, share, difficulty))
    for sid, quota, difficulty in plan:
        take(with_subtopic(sid), quota, difficulty)

    # Too few questions: fill from the chosen topics, then from the whole bank.
    if len(selected) < limit:
        take([q for q in all_mcqs if any(_covers_topic(q, t) for t in topic_ids)], limit - len(selected), 0.5)
    if len(selected) < limit:
        take(all_mcqs, limit - len(selected), 0.5)
    return selected


def clean_llm_output(raw_text: str) -> str:
    """Return only the model's answer from the raw LLM output.

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


# System prompt for the quiz introduction. The LLM writes only the intro, never the questions.
ASSESSMENT_SYSTEM_PROMPT = """You are an expert AI Assessment Coach.
The student is taking a personalized quiz on: {topic_focus}.\n(If several topics are listed, the quiz mixes questions from all of them.)
Student Mastery Score: {mastery_score} (Target Test Difficulty: {difficulty_label}).
Orchestrator Directive: {orchestrator_instruction}

Introduce the assessment encouragingly, explaining that this quiz is tailored to challenge them appropriately based on their current knowledge profile.
Write only a short introduction of 2-3 sentences. Do NOT write any quiz questions or answers yourself:
the questions are shown to the student separately.
"""


def create_assessment_node():
    """Build the assessment node function for the LangGraph workflow."""
    llm = get_llm("quiz", enable_thinking=False)

    def assessment_node(state: dict) -> dict:
        """Select the quiz questions and write a short introduction.

        Returns the state updates: the intro message and "active_quiz_questions", which app.py
        shows as the quiz form. `state["quiz_subtopics"]` set means a re-test on those subtopics.
        """
        student_id = state.get("student_id", "Unknown")
        student_kb = state.get("student_kb")
        topic_focus_list = state.get("topic_focus") or ["Python Basics"]
        if not isinstance(topic_focus_list, list):
            topic_focus_list = [str(topic_focus_list)]
        instruction = state.get("orchestrator_instruction", "Provide a practice quiz.")

        scores = student_kb.get("topics", {}) if student_kb else {}
        topic_ids = list(dict.fromkeys(get_topic_id_from_name(t) for t in topic_focus_list))
        topic_focus = ", ".join(topic_ids)
        # Average mastery over the quiz topics (0.5 for an unscored topic); only used for the
        # difficulty label in the intro prompt.
        score = round(sum(float(scores.get(t, 0.5)) for t in topic_ids) / len(topic_ids), 2)

        if score >= 0.8:
            diff_label = "Advanced / Hard"
        elif score < 0.4:
            diff_label = "Foundational / Easy"
        else:
            diff_label = "Intermediate / Medium"

        target_subtopics = state.get("quiz_subtopics") or None
        quiz_questions = select_adaptive_mcqs(
            topic_ids, student_kb, limit=RETEST_SIZE if target_subtopics else 10,
            target_subtopics=target_subtopics,
        )

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
        if not intro_text:   # fallback if the model wrote nothing usable
            intro_text = (f"Here is a quiz tailored to your current level on {topic_focus}. "
                          "Take your time and good luck!")

        print(f"\n🎯 [ASSESSMENT AGENT] Generated {len(quiz_questions)} MCQs for topics: {topic_focus} | Target Difficulty: {diff_label}")

        return {
            "messages": [AIMessage(content=intro_text)],
            "active_quiz_questions": quiz_questions,  
            "last_active_agent": "assessment_agent",
        }

    return assessment_node