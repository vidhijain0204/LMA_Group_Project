import json
import os
import re
from typing import List, Optional, TypedDict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import Command

from agents.llm_factory import get_llm

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


def load_student_kb(student_id: str, db_path: str = SKB_FILE_PATH) -> Optional[dict]:
    if not os.path.exists(db_path):
        return None
    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for record in data:
            if record.get("student_id") == student_id:
                return record
    elif isinstance(data, dict) and data.get("student_id") == student_id:
        return data
    return None


ORCHESTRATOR_SYSTEM_PROMPT = """You are the Master Orchestrator for an AI Python Tutoring System.
Your job is to analyze the student's message, student knowledge base, and route to the correct agent.

### Available Agents & Intents:
1. `diagnosis`: Use ONLY if the student is brand new and needs their initial onboarding diagnostic test.
2. `tutor_agent`: Use if the student asks for concept explanations, syntax help, or coding guidance.
3. `assessment_quiz`: Use if the student asks for a quiz, test, challenge, or wants to check their mastery/scores on a topic.

### Topic Catalog Reference:
{topics_summary}

Respond in strict JSON format with keys:
- `next_agent`: ("diagnosis" | "tutor_agent" | "assessment_quiz")
- `topic_focus`: [list of relevant topic names or IDs]
- `instruction`: [brief directive for the target agent]
- `reasoning`: [short explanation of routing decision]
"""


def create_orchestrator_node():
    llm = get_llm("router", enable_thinking=True)

    topics_summary = "Python Basics, Control Flow, Data Structures, Functions, OOP"
    if os.path.exists(TOPICS_FILE_PATH):
        try:
            with open(TOPICS_FILE_PATH, "r", encoding="utf-8") as f:
                t_list = json.load(f)
                topics_summary = ", ".join([f"{t['topic_id']}: {t['name']}" for t in t_list])
        except Exception:
            pass

        
    def orchestrator_node(state: AgentState) -> Command:
        student_id = state.get("student_id", "S001")
        # workflow.start() has already looked the student up (None = not in the database);
        # read the file only if the caller did not provide the record.
        student_kb = state["student_kb"] if "student_kb" in state else load_student_kb(student_id)

        # A student without a knowledge base record has never taken the diagnostic test.
        # Onboard them first, whatever they typed: this is a fact from the database, not
        # something the LLM can infer from the message.
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

        sys_msg = SystemMessage(
            content=ORCHESTRATOR_SYSTEM_PROMPT.format(
                topics_summary=topics_summary
            )
        )

        response = llm.invoke([sys_msg] + state["messages"])
        raw_content = response.content.strip()

        try:
            content = raw_content.replace("\xa0", " ")

            # Remove reasoning blocks
            content = re.sub(
                r"<think>.*?</think>",
                "",
                content,
                flags=re.DOTALL
            ).strip()

            # Handle incomplete/unclosed think blocks
            if "<think>" in content:
                content = content.split("<think>")[0].strip()

            # Extract JSON from markdown code blocks
            if "```json" in content:
                content = content.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in content:
                content = content.split("```", 1)[1].split("```", 1)[0].strip()

            # Extract JSON object
            json_start = content.find("{")
            json_end = content.rfind("}")

            if json_start == -1 or json_end == -1:
                raise ValueError("No JSON object found in LLM response")

            content = content[json_start:json_end + 1]

            decision = json.loads(content)

            # Validate required fields
            required_keys = {
                "next_agent",
                "topic_focus",
                "instruction",
                "reasoning"
            }

            if not required_keys.issubset(decision.keys()):
                raise ValueError("Missing required JSON keys")

            print("✅ Orchestrator Decision:", decision)

        except Exception as parse_err:
            print(
                f"⚠️ [ORCHESTRATOR PARSE ERROR]: "
                f"{parse_err} | Raw: {raw_content}"
            )

            decision = {
                "next_agent": "tutor_agent",
                "topic_focus": ["Variables and Data Types"],
                "instruction": "Explain concept clearly.",
                "reasoning": "Fallback routing due to parse error."
            }

        # Continue with your existing routing logic here


        # Translate the LLM's label into a node that exists in the graph.
        next_agent = normalize_route(decision.get("next_agent"))

        # Existing students have already taken the diagnosis, and DiagnosisAgent refuses a
        # second attempt, so a "diagnosis" decision for them falls back to the tutor.
        if next_agent == DIAGNOSIS_NODE:
            next_agent = TUTOR_NODE

        topic_focus = decision.get("topic_focus", ["Variables and Data Types"])
        instruction = decision.get("instruction", "Explain concept.")
        reasoning = decision.get("reasoning", "Direct routing.")

        print(f"\n🧭 [ORCHESTRATOR] Routing to -> {next_agent} | Topic: {topic_focus} | Reason: {reasoning}")

        return Command(
            goto=next_agent,
            update={
                "student_id": student_id,
                "student_kb": student_kb,
                "topic_focus": topic_focus,
                "orchestrator_instruction": instruction,
                "orchestrator_reasoning": reasoning,
                "last_active_agent": "orchestrator"
            }
        )

    return orchestrator_node