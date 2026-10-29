# =====================================================================
# FILE PATHS & CONFIGURATION
# =====================================================================
TOPICS_FILE_PATH = "data/common/topics.json"
SKB_FILE_PATH = "data/skb_database.json"

import json
import re
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.types import Command

from agents.llm_factory import get_llm


# =====================================================================
# DYNAMIC JSON LOADERS & MAPPINGS
# =====================================================================
def load_topics(filepath: str = TOPICS_FILE_PATH) -> List[Dict[str, str]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_topic_maps(filepath: str = TOPICS_FILE_PATH):
    topics = load_topics(filepath)
    id_to_name = {t["topic_id"]: t["name"] for t in topics}
    name_to_id = {t["name"].lower(): t["topic_id"] for t in topics}
    return id_to_name, name_to_id


def load_student_kb(student_id: str, filepath: str = SKB_FILE_PATH) -> Optional[dict]:
    path = Path(filepath)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        for record in data:
            if record.get("student_id") == student_id:
                return record
        return None
    elif isinstance(data, dict) and data.get("student_id") == student_id:
        return data
    return None


# =====================================================================
# LANGGRAPH SHARED STATE
# =====================================================================
class AgentState(TypedDict):
    student_id: str
    messages: Annotated[List[BaseMessage], add_messages]
    student_kb: Optional[dict]
    topic_focus: List[str]  # Human-readable topic names
    last_active_agent: Optional[
        Literal["diagnosis_agent", "tutor_agent", "assessment_agent"]
    ]
    orchestrator_instruction: str
    orchestrator_reasoning: str


# =====================================================================
# PROMPT FORMATTER
# =====================================================================
def format_orchestrator_context(
    topics_list: List[dict], student_kb: Optional[dict]
) -> tuple[str, str]:
    catalog_lines = [f"- `{t['topic_id']}`: {t['name']}" for t in topics_list]
    catalog_str = "\n".join(catalog_lines)

    if not student_kb or not student_kb.get("topics"):
        return catalog_str, "NO STUDENT KB FOUND (New / Cold Start Student)."

    scores = student_kb.get("topics", {})
    strengths = set(student_kb.get("strengths", []))
    weaknesses = set(student_kb.get("weaknesses", []))

    kb_lines = [
        f"Student ID: {student_kb.get('student_id')}",
        f"Strengths: {student_kb.get('strengths', [])}",
        f"Weaknesses: {student_kb.get('weaknesses', [])}",
        "Per-Topic Status:"
    ]

    for t in topics_list:
        tid = t["topic_id"]
        score = scores.get(tid, 0.0)
        tag = (
            "STRENGTH" if tid in strengths
            else "WEAKNESS" if tid in weaknesses
            else "UNTESTED/DEVELOPING"
        )
        kb_lines.append(f"  * {tid} ({t['name']}): {tag} (Score: {score:.2f})")

    return catalog_str, "\n".join(kb_lines)


# =====================================================================
# ORCHESTRATOR NODE FACTORY
# =====================================================================
ORCHESTRATOR_SYSTEM_PROMPT = """You are the Central Orchestrator of a Multi-Agent Tutoring System.
Analyze the user's input, student background, and choose between the active specialist agents: `tutor_agent` or `assessment_agent`.

### Available Topics:
{topics_catalog}

### Student KB Summary:
{formatted_kb}

### Last Active Agent: {last_active_agent}

### Selection Guidelines:
1. Choose `tutor_agent` if the user is asking for conceptual explanations, examples, or help understanding how something works.
2. Choose `assessment_agent` if the user explicitly asks for a practice challenge, coding quiz, test, or exam.

### Response Format:
AGENT: [tutor_agent OR assessment_agent]
TOPIC_NAMES: [Name of topic 1, Name of topic 2]
REASONING: [Clear explanation of why this agent was chosen based on student status and user input]
INSTRUCTION: [Short directive for the chosen agent]
"""


def create_orchestrator_node():
    llm = get_llm("router", enable_thinking=True)
    valid_agents = ["diagnosis_agent", "tutor_agent", "assessment_agent"]
    id_to_name, name_to_id = get_topic_maps(TOPICS_FILE_PATH)

    def orchestrator_node(
        state: AgentState,
    ) -> Command[Literal["diagnosis_agent", "tutor_agent", "assessment_agent"]]:
        topics_list = load_topics(TOPICS_FILE_PATH)
        
        student_id = state.get("student_id")
        student_kb = state.get("student_kb")
        if not student_kb and student_id:
            student_kb = load_student_kb(student_id, SKB_FILE_PATH)

        # -------------------------------------------------------------
        # RULE 1: If student is new (not in database), call DIAGNOSIS
        # -------------------------------------------------------------
        if not student_kb:
            print(f"[Orchestrator Rule] New student ID '{student_id}' not found in database -> Routing to Diagnosis")
            matched_names = [topics_list[0]["name"]]
            return Command(
                goto="diagnosis_agent",
                update={
                    "topic_focus": matched_names,
                    "orchestrator_instruction": "Initialize baseline student record via diagnostic assessment.",
                    "orchestrator_reasoning": f"Rule Trigger: Student ID '{student_id}' is brand new (not found in student_kb.json). Diagnosis routine is called to initialize profile.",
                    "student_kb": None,
                },
            )

        latest_user_msg = ""
        if state.get("messages"):
            latest_user_msg = state["messages"][-1].content.lower()

        # -------------------------------------------------------------
        # RULE 2: Check if existing student is asking for test/quiz -> ASSESSMENT
        # -------------------------------------------------------------
        assessment_keywords = ["quiz", "test", "exam", "challenge", "practice problem", "assessment"]
        if any(kw in latest_user_msg for kw in assessment_keywords):
            matched_names = [t["name"] for t in topics_list if t["topic_id"].lower() in latest_user_msg or t["name"].lower() in latest_user_msg]
            if not matched_names:
                matched_names = [topics_list[0]["name"]]
            return Command(
                goto="assessment_agent",
                update={
                    "topic_focus": matched_names,
                    "orchestrator_instruction": "User requested practice or quiz. Generate targeted assessment.",
                    "orchestrator_reasoning": "Rule Trigger: Existing student explicitly used assessment/quiz keywords in their request.",
                    "student_kb": student_kb,
                },
            )

        # -------------------------------------------------------------
        # RULE 3: Otherwise, leverage SLM intelligence for Tutor/Assessment
        # -------------------------------------------------------------
        catalog_str, formatted_kb = format_orchestrator_context(
            topics_list, student_kb
        )
        last_agent = state.get("last_active_agent") or "None"

        sys_msg = SystemMessage(
            content=ORCHESTRATOR_SYSTEM_PROMPT.format(
                topics_catalog=catalog_str,
                formatted_kb=formatted_kb,
                last_active_agent=last_agent,
            )
        )

        response = llm.invoke([sys_msg] + state["messages"])
        content = response.content.strip()

        # Strip out Qwen3 thinking block
        content_cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        chosen_agent = "tutor_agent"  # default fallback
        for agent in valid_agents:
            if agent in content_cleaned:
                chosen_agent = agent
                break

        found_names = []
        for t in topics_list:
            if t["topic_id"] in content_cleaned or t["name"].lower() in content_cleaned.lower():
                found_names.append(t["name"])
        
        if not found_names:
            found_names = [topics_list[0]["name"]]

        reasoning = "Orchestrator selected based on user intent and student profile."
        instruction = "Proceed with requested student support."
        
        for line in content_cleaned.split("\n"):
            line_str = line.strip()
            if line_str.startswith("REASONING:"):
                reasoning = line_str.split("REASONING:", 1)[1].strip()
            elif line_str.startswith("INSTRUCTION:"):
                instruction = line_str.split("INSTRUCTION:", 1)[1].strip()

        return Command(
            goto=chosen_agent,
            update={
                "topic_focus": found_names,
                "orchestrator_instruction": instruction,
                "orchestrator_reasoning": reasoning,
                "student_kb": student_kb,
            },
        )

    return orchestrator_node