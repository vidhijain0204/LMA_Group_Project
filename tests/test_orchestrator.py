# =====================================================================
# TEST HARNESS FOR THE ORCHESTRATOR AGENT (`tests/test_orchestrator.py`)
# =====================================================================
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from agents.orchestrator import create_orchestrator_node, AgentState, SKB_FILE_PATH


# =====================================================================
# MOCK SUB-AGENT NODES (Prints Routing Decision, Topic, and Reasoning)
# =====================================================================
def diagnosis_node(state: AgentState):
    print(f"\n🟢 [ROUTED TO: DIAGNOSIS ROUTINE / INITIALIZER]")
    print(f"   - Student ID: {state['student_id']} (Status: New Student)")
    print(f"   - Orchestrator Summary / Reasoning: {state.get('orchestrator_reasoning', 'N/A')}")
    print(f"   - Instruction: {state['orchestrator_instruction']}")
    return {"last_active_agent": "diagnosis_agent"}


def tutor_node(state: AgentState):
    print(f"\n🟡 [ROUTED TO: TUTOR AGENT]")
    print(f"   - Student ID: {state['student_id']} (Status: Existing Student)")
    print(f"   - Topic Focus: {state['topic_focus']}")
    print(f"   - Orchestrator Summary / Reasoning: {state.get('orchestrator_reasoning', 'N/A')}")
    print(f"   - Instruction: {state['orchestrator_instruction']}")
    return {"last_active_agent": "tutor_agent"}


def assessment_node(state: AgentState):
    print(f"\n🔵 [ROUTED TO: ASSESSMENT AGENT]")
    print(f"   - Student ID: {state['student_id']} (Status: Existing Student)")
    print(f"   - Topic Focus: {state['topic_focus']}")
    print(f"   - Orchestrator Summary / Reasoning: {state.get('orchestrator_reasoning', 'N/A')}")
    print(f"   - Instruction: {state['orchestrator_instruction']}")
    return {"last_active_agent": "assessment_node"}


# =====================================================================
# BUILD TEST GRAPH
# =====================================================================
def build_test_graph():
    builder = StateGraph(AgentState)

    builder.add_node("orchestrator", create_orchestrator_node())
    builder.add_node("diagnosis_agent", diagnosis_node)
    builder.add_node("tutor_agent", tutor_node)
    builder.add_node("assessment_agent", assessment_node)

    builder.add_edge(START, "orchestrator")

    builder.add_edge("diagnosis_agent", END)
    builder.add_edge("tutor_agent", END)
    builder.add_edge("assessment_agent", END)

    return builder.compile()


def get_student_record(student_id: str):
    skb_path = Path(SKB_FILE_PATH)
    if not skb_path.exists():
        return None
    with open(skb_path, "r", encoding="utf-8") as f:
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
# 5 VARIETY TEST CASES
# =====================================================================
def run_tests():
    graph = build_test_graph()

    test_cases = [
        {
            "id": "Test 1: New Student Cold Start (Not in DB)",
            "student_id": "S001",  # Not in database
            "user_input": "I am new and want to learn Python.",
            "description": "New student ID not found in KB. Expected: diagnosis_agent",
        },
        {
            "id": "Test 2: Existing Student Conceptual Doubt",
            "student_id": "S001",  # Existing student
            "user_input": "I am struggling to understand how 'if, elif, else' conditional statements work. Can you explain?",
            "description": "Existing student asking for explanation. Expected: tutor_agent",
        },
        {
            "id": "Test 3: Existing Student Explicit Quiz Request",
            "student_id": "S002",  # Existing student
            "user_input": "I feel confident about Variables and Data Types. Give me a coding quiz to test my skills.",
            "description": "Explicit request for test/quiz. Expected: assessment_agent",
        },
        {
            "id": "Test 4: Another New Student Check",
            "student_id": "S888",  # Not in database
            "user_input": "Can you test my skills?",
            "description": "New student ID not found in KB. Expected: diagnosis_agent",
        },
        {
            "id": "Test 5: Existing Student General Learning Request",
            "student_id": "S003",  # Existing student
            "user_input": "How do Python functions and return statements work?",
            "description": "Existing student asking about functions. Expected: tutor_agent",
        },
    ]

    print("=========================================================")
    print("🚀 RUNNING 5 VARIETY TEST CASES FOR ORCHESTRATOR ROUTING")
    print("=========================================================")

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n---------------------------------------------------------")
        print(f"Test Case {idx}: {tc['id']}")
        print(f"Student ID: {tc['student_id']}")
        print(f"Description: {tc['description']}")
        print(f"User Input: \"{tc['user_input']}\"")

        student_kb_record = get_student_record(tc["student_id"])

        state = {
            "student_id": tc["student_id"],
            "messages": [HumanMessage(content=tc["user_input"])],
            "student_kb": student_kb_record,
            "topic_focus": [],
            "last_active_agent": None,
            "orchestrator_instruction": "",
            "orchestrator_reasoning": "",
        }

        try:
            graph.invoke(state)
        except Exception as e:
            print(f"❌ Test Failed with Exception: {e}")

    print(f"\n=========================================================")
    print("✅ ALL TEST CASES COMPLETED")
    print("=========================================================")


if __name__ == "__main__":
    run_tests()