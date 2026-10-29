# =====================================================================
# TEST HARNESS FOR THE TUTOR AGENT (`tests/test_tutor.py`)
# =====================================================================
import sys
from pathlib import Path

# Automatically add the project root directory to Python's path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from langchain_core.messages import HumanMessage
from agents.tutor_agent import create_tutor_node
from agents.orchestrator import SKB_FILE_PATH


def run_tutor_evaluation():
    print("=========================================================")
    print("🚀 RUNNING TUTOR AGENT EVALUATION (With KB vs. Without KB)")
    print("=========================================================")

    # Load sample student KB for S001
    skb_path = Path(SKB_FILE_PATH)
    student_kb_data = None
    if skb_path.exists():
        with open(skb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                student_kb_data = next((r for r in data if r.get("student_id") == "S001"), None)
            elif isinstance(data, dict):
                student_kb_data = data

    state = {
        "student_id": "S001",
        "student_kb": student_kb_data,
        "topic_focus": ["Conditional Statements"],
        "orchestrator_instruction": "Explain how if/else works with an example.",
        "messages": [HumanMessage(content="Can you explain how if, elif, else statements work in Python?")]
    }

    print(f"\n--- 1. TESTING WITHOUT KB (Anonymous / Default Context) ---")
    tutor_without_kb = create_tutor_node(enable_kb_context=False)
    tutor_without_kb(state)

    print(f"\n--- 2. TESTING WITH KB (Personalized Student Context) ---")
    tutor_with_kb = create_tutor_node(enable_kb_context=True)
    tutor_with_kb(state)

    print("\n=========================================================")
    print("✅ TUTOR EVALUATION COMPLETED")
    print("=========================================================")


if __name__ == "__main__":
    run_tutor_evaluation()