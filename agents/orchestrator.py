import json
from datetime import datetime
import os
import re
from typing import Any, List, Optional, Tuple, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command

from agents.pedagogy_controller import resolve_topic_id
from agents.struggle_tracker import (
    RECENT_TURNS_FOR_QUIZ, current_lesson_topic, recent_learning_topics, topic_evidence, topics_mentioned,
)



SKB_FILE_PATH = "data/skb_database.json"
TOPICS_FILE_PATH = "data/common/topics.json"

# Names of the graph nodes the orchestrator can route to. workflow.py registers its nodes under
# these exact names, so the routing decision and the graph always agree.
DIAGNOSIS_NODE = "diagnosis"
TUTOR_NODE = "tutor_agent"
ASSESSMENT_NODE = "assessment_agent"
ROUTABLE_NODES = (DIAGNOSIS_NODE, TUTOR_NODE, ASSESSMENT_NODE)


def normalize_route(label) -> str:
    """Map the routing label written by the LLM onto one of the graph node names.

    The prompt asks for "diagnosis", "tutor_agent" or "assessment_quiz", but a small model may
    also answer "diagnosis_agent", "quiz", "assessment", "tutor", ... Diagnosis is checked first
    so that a label like "diagnostic test" is not mistaken for a quiz. Anything that is not
    recognised goes to the tutor, the safest default.
    """
    label = str(label or "").strip().lower()
    if "diag" in label:
        return DIAGNOSIS_NODE
    if any(word in label for word in ("assess", "quiz", "test")):
        return ASSESSMENT_NODE
    return TUTOR_NODE


class AgentState(TypedDict):
    student_id: str
    messages: List[Any]
    student_kb: Optional[dict]
    topic_focus: List[str]
    last_active_agent: Optional[str]
    orchestrator_instruction: str
    orchestrator_reasoning: str
    conversation_mode: str
    tutor_question: str


def read_skb_database(db_path: str = SKB_FILE_PATH) -> List[dict]:
    """All student records, as a list. A missing, empty or whitespace-only file is an empty
    database. An unreadable file is renamed to <name>.corrupt-<time> (so the next save does
    not overwrite it silently) and treated as empty. A single-record dict is wrapped in a list."""
    if not os.path.exists(db_path):
        return []
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            return []
        data = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        backup = f"{db_path}.corrupt-{datetime.now():%Y%m%d-%H%M%S}"
        os.replace(db_path, backup)
        print(f"[KB] {db_path} is not valid JSON ({error}); moved it to {backup} and started empty.")
        return []
    if isinstance(data, dict):
        data = [data] if data.get("student_id") else []
    if not isinstance(data, list):
        print(f"[KB] Unsupported format in {db_path} ({type(data).__name__}); treating it as empty.")
        return []
    return [record for record in data if isinstance(record, dict)]


def load_student_kb(student_id: str, db_path: str = SKB_FILE_PATH) -> Optional[dict]:
    return next((r for r in read_skb_database(db_path) if r.get("student_id") == student_id), None)


ORCHESTRATOR_SYSTEM_PROMPT = """You are the Master Orchestrator for an AI Python Tutoring System.
Your job is to analyze the student's message, student knowledge base, and route to the correct agent.

### Available Agents & Intents:
1. `diagnosis`: Use ONLY if the student is brand new and needs their initial onboarding diagnostic test.
2. `tutor_agent`: Use if the student asks for concept explanations, syntax help, or coding guidance.
3. `assessment_quiz`: Use if the student asks for a quiz, test, challenge or assessment, says they are ready to be assessed, or wants to check their mastery/scores on a topic.

Earlier messages of the conversation are included: use them to understand follow-up questions
("explain that again", "what about the second example?"). If the tutor's last message asked the
student a question and the student answers it, route to `tutor_agent` with the same topic.

### Topic Catalog Reference:
{topics_summary}

Respond in strict JSON format with keys:
- `next_agent`: ("diagnosis" | "tutor_agent" | "assessment_quiz")
- `topic_focus`: [list of relevant topic names or IDs]
- `instruction`: [brief directive for the target agent]
- `reasoning`: [short explanation of routing decision]
"""


def choose_quiz_topics(user_text: str, llm_topics, student_kb: Optional[dict]) -> Tuple[List[str], str]:
    """Topics for a quiz request, and where they came from.

    1. Topics the student names in the message ("quiz on lists and loops").
    2. Otherwise ("ready for assessment"), the topics of the last 3 tutor conversations.
    3. Otherwise, whatever topic the LLM suggested.
    """
    named = topics_mentioned(user_text)
    if named:
        return named, f"topics named in the request: {', '.join(named)}"
    recent = recent_learning_topics(student_kb, RECENT_TURNS_FOR_QUIZ)
    if recent:
        return recent, (
            f"no topic named, so the topics of the last {RECENT_TURNS_FOR_QUIZ} tutor "
            f"conversations: {', '.join(recent)}"
        )
    llm_topics = llm_topics if isinstance(llm_topics, list) else [llm_topics]
    return [str(t) for t in llm_topics if t] or ["T01"], "no topic named and no learning history, so the LLM's topic"


QUIZ_REQUEST = re.compile(r"\b(quiz|test me|assess\w*|exam|ready for)\b", re.IGNORECASE)
FOLLOW_UP_MODE = "follow_up"
FOLLOW_UP_INSTRUCTION = (
    "The student is answering the question you asked in your previous message. Evaluate that "
    "answer and continue the same lesson; do not treat it as a new question."
)


def detect_follow_up(state: dict, student_kb: Optional[dict]) -> Optional[dict]:
    """Is the new message the student's reply to a question the tutor just asked?

    True when all of these hold:
    - the last thing that happened for this student was a tutor lesson (not a quiz),
    - the previous chat message is from the tutor and ends with a question,
    - the new message does not request a quiz and does not name a different topic.
    Returns the lesson to continue ({"topic_id", "strategy", "tutor_question"}) or None.
    """
    messages = state.get("messages", [])
    if len(messages) < 2 or not isinstance(messages[-2], AIMessage):
        return None
    tutor_message = str(messages[-2].content)
    # The tutor must END with a question (headings like "What is a list?" don't count).
    closing_lines = [
        line for line in tutor_message.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ][-3:]
    if not any("?" in line for line in closing_lines):
        return None
    history = (student_kb or {}).get("pedagogy_history", [])
    if not history or history[-1].get("event") != "lesson":
        return None
    lesson = history[-1]
    text = state.get("user_input") or str(messages[-1].content)
    if QUIZ_REQUEST.search(text):
        return None
    named = topics_mentioned(text)
    if named and lesson.get("topic_id") not in named:
        return None
    question = next(line.strip() for line in reversed(closing_lines) if "?" in line)
    return {"topic_id": lesson.get("topic_id"), "strategy": lesson.get("strategy"), "tutor_question": question[:300]}


def choose_tutor_topic(
    user_text: str, llm_topics, student_kb: Optional[dict], in_conversation: bool
) -> Tuple[List[str], str]:
    """Topic for a tutor request, and where it came from.

    1. A topic the student names ("explain dictionaries").
    2. Keyword evidence ("difference between sort and count" -> T08). When the conversation's
       current topic is among the best matches, it is kept ("count" is both a string and a
       list method; while learning lists it means the list method).
    3. No evidence ("can you give another example?"): stay on the conversation's topic.
    4. Otherwise the topic the LLM suggested.
    """
    previous = current_lesson_topic(student_kb) if in_conversation else None
    named = topics_mentioned(user_text)
    if named:
        return named, f"topic named in the message: {', '.join(named)}"
    evidence = topic_evidence(user_text)
    if evidence:
        best = max(evidence.values())
        leaders = sorted(tid for tid, score in evidence.items() if score == best)
        if previous in leaders:
            return [previous], f"keywords point at {', '.join(leaders)}; kept the conversation's topic {previous}"
        return [leaders[0]], f"keywords in the message point at {leaders[0]} ({evidence})"
    if previous:
        return [previous], f"no topic in the message; continuing the conversation's topic {previous}"
    llm_topics = llm_topics if isinstance(llm_topics, list) else [llm_topics]
    resolved = [resolve_topic_id(t) for t in llm_topics if t]
    resolved = [t for t in resolved if re.fullmatch(r"T\d{2}", t)]
    return (resolved or ["T01"]), "no topic in the message and no ongoing lesson; the LLM's topic"


def parse_decision(raw_content: str) -> dict:
    """Extract the JSON routing decision from the LLM output (thinking and code fences removed)."""
    content = raw_content.replace("\xa0", " ")
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if "<think>" in content:                      # thinking cut off before it closed
        content = content.split("<think>")[0].strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0].strip()
    json_start, json_end = content.find("{"), content.rfind("}")
    if json_start == -1 or json_end == -1:
        raise ValueError("No JSON object found in LLM response")
    decision = json.loads(content[json_start:json_end + 1])
    if not {"next_agent", "topic_focus", "instruction", "reasoning"}.issubset(decision):
        raise ValueError("Missing required JSON keys")
    return decision


def create_orchestrator_node(llm=None):
    """Create the orchestrator node. An LLM can be injected for testing; otherwise the local
    LLM is loaded."""
    if llm is None:
        from agents.llm_factory import get_llm
        llm = get_llm("router", enable_thinking=True)

    topics_summary = "Python Basics, Control Flow, Data Structures, Functions, OOP"
    if os.path.exists(TOPICS_FILE_PATH):
        try:
            with open(TOPICS_FILE_PATH, "r", encoding="utf-8") as f:
                topics_summary = ", ".join(f"{t['topic_id']}: {t['name']}" for t in json.load(f))
        except Exception:
            pass

    def orchestrator_node(state: AgentState) -> Command:
        student_id = state.get("student_id", "S001")
        # workflow.start() has already looked the student up (None = not in the database);
        # read the file only if the caller did not provide the record.
        student_kb = state["student_kb"] if "student_kb" in state else load_student_kb(student_id)

        # A student without a knowledge base record has never taken the diagnostic test.
        # Onboard them first, whatever they typed.
        if not student_kb:
            print(f"\n🧭 [ORCHESTRATOR] New student {student_id} (no knowledge base) -> {DIAGNOSIS_NODE}")
            return Command(
                goto=DIAGNOSIS_NODE,
                update={
                    "student_id": student_id,
                    "student_kb": None,
                    "topic_focus": [],
                    "orchestrator_instruction": "Run the onboarding diagnostic test.",
                    "orchestrator_reasoning": "New student: no knowledge base record found.",
                    "last_active_agent": "orchestrator",
                },
            )

        # The student is answering the tutor's last question: continue that lesson directly,
        # without asking the LLM (which would treat the answer as a brand-new request).
        follow_up = detect_follow_up(state, student_kb)
        if follow_up:
            reasoning = (
                f"Student is replying to the tutor's question on {follow_up['topic_id']} "
                f"(strategy: {follow_up['strategy']}); continuing the same lesson."
            )
            print(f"\n🧭 [ORCHESTRATOR] Follow-up reply -> {TUTOR_NODE} | {reasoning}")
            return Command(
                goto=TUTOR_NODE,
                update={
                    "student_id": student_id,
                    "student_kb": student_kb,
                    "topic_focus": [follow_up["topic_id"]],
                    "orchestrator_instruction": FOLLOW_UP_INSTRUCTION,
                    "orchestrator_reasoning": reasoning,
                    "conversation_mode": FOLLOW_UP_MODE,
                    "tutor_question": follow_up["tutor_question"],
                    "last_active_agent": "orchestrator",
                },
            )

        sys_msg = SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT.format(topics_summary=topics_summary))
        # state["messages"] holds the recent chat history followed by the current message.
        response = llm.invoke([sys_msg] + state["messages"])
        raw_content = response.content.strip()
        try:
            decision = parse_decision(raw_content)
            print("✅ Orchestrator Decision:", decision)
        except Exception as parse_err:
            print(f"⚠️ [ORCHESTRATOR PARSE ERROR]: {parse_err} | Raw: {raw_content}")
            decision = {
                "next_agent": "tutor_agent",
                "topic_focus": ["Variables and Data Types"],
                "instruction": "Explain concept clearly.",
                "reasoning": "Fallback routing due to parse error.",
            }

        # Translate the LLM's label into a node that exists in the graph. Existing students have
        # already taken the diagnosis (it is allowed only once), so that falls back to the tutor.
        next_agent = normalize_route(decision.get("next_agent"))
        if next_agent == DIAGNOSIS_NODE:
            next_agent = TUTOR_NODE

        topic_focus = decision.get("topic_focus") or ["Variables and Data Types"]
        if not isinstance(topic_focus, list):
            topic_focus = [str(topic_focus)]
        instruction = decision.get("instruction", "Explain concept.")
        reasoning = decision.get("reasoning", "Direct routing.")

        user_text = state.get("user_input") or next(
            (m.content for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)), ""
        )
        if next_agent == ASSESSMENT_NODE:
            topic_focus, source = choose_quiz_topics(user_text, topic_focus, student_kb)
            reasoning = f"{reasoning} | Quiz topics: {source}"
        else:
            # The LLM's topic is only a last resort: a small model (or the parse-error fallback)
            # often picks an unrelated topic for a follow-up question.
            in_conversation = len(state.get("messages", [])) > 1
            llm_topic = topic_focus
            topic_focus, source = choose_tutor_topic(user_text, llm_topic, student_kb, in_conversation)
            reasoning = f"{reasoning} | Topic: {source} (LLM suggested {llm_topic})"

        print(f"\n🧭 [ORCHESTRATOR] Routing to -> {next_agent} | Topic: {topic_focus} | Reason: {reasoning}")
        return Command(
            goto=next_agent,
            update={
                "student_id": student_id,
                "student_kb": student_kb,
                "topic_focus": topic_focus,
                "orchestrator_instruction": instruction,
                "orchestrator_reasoning": reasoning,
                "conversation_mode": "new_request",
                "tutor_question": "",
                "last_active_agent": "orchestrator",
            },
        )

    return orchestrator_node
