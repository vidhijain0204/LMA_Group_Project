"""Offline demo of the root-cause hypothesis step (no LLM server needed, nothing is saved).

    python demo_root_cause.py [STUDENT_ID] [TOPIC_ID]

It loads the student's knowledge base (or a blank one), simulates a few tutor messages that show
struggle, takes a 10-question quiz from the real question bank on TOPIC_ID answered wrongly on
purpose for one subtopic, and prints the three hypotheses with every parameter.
"""

import json
import sys
from copy import deepcopy

from agents.orchestrator import SKB_FILE_PATH, load_student_kb
from agents.pedagogy_controller import record_lesson_strategy
from agents.root_cause_agent import RootCauseAgent, format_hypotheses, format_root_cause, summarize_comparison
from agents.struggle_tracker import load_catalog, record_learning_turn

student_id = sys.argv[1] if len(sys.argv) > 1 else "S001"
topic_id = sys.argv[2] if len(sys.argv) > 2 else "T13"

kb = deepcopy(load_student_kb(student_id, SKB_FILE_PATH) or {"student_id": student_id, "topics": {}})
print(f"Student {student_id} | KB mastery of {topic_id}: {kb.get('topics', {}).get(topic_id, 'n/a')}")

# The quiz: 10 real questions on the topic. The simulated weak subtopic is the topic's subtopic
# that appears most often in them.
with open("data/common/mcqs_list.json", encoding="utf-8") as bank_file:
    bank = [q for q in json.load(bank_file) if topic_id in q.get("related_topics", [])][:10]
in_topic = [s for q in bank for s in q["related_subtopics"] if s.startswith(f"{topic_id}.")]
target = max(set(in_topic), key=in_topic.count)
target_name = load_catalog()["subtopic_names"][target]

# 1. Simulated learning conversation about the weak subtopic (what the tutor would log).
conversation = [
    ("explicit_explanation", f"Can you explain {target_name}?"),
    ("self_explanation", f"I'm confused about {target_name}, it doesn't make sense"),
    ("self_explanation", f"still not working, just tell me the answer for {target_name}"),
]
for strategy, text in conversation:
    kb = record_lesson_strategy(kb, topic_id, strategy, "demo")
    kb = record_learning_turn(kb, topic_id, text, previous_strategy=strategy)
    turn = kb["learning_conversations"][-1]
    print(f"  logged: {text!r} -> subtopics={turn['subtopic_ids']} "
          f"confusion={turn['confusion']} hint={turn['hint_request']}")

# 2. Answer wrongly every question that tests the weak subtopic, and correctly everything else.
answers = {}
for q in bank:
    wrong_option = next(o for o in q["options"] if o != q["correct_option"])
    answers[q["prob_id"]] = wrong_option if target in q["related_subtopics"] else q["correct_option"]
print(f"\nQuiz: {len(bank)} questions, simulated weakness in {target}; "
      f"wrong answers: {sum(answers[q['prob_id']] != q['correct_option'] for q in bank)}")

# 3. Hypotheses and root cause, then what every method / weight set picks on the same quiz.
agent = RootCauseAgent()
result = agent.analyze(bank, answers, kb)
print("\n" + format_hypotheses(result))
for h in result["hypotheses"]:
    print(f"   {h['subtopic_id']} struggle components: {h['struggle_components']}")
print("\n=> " + format_root_cause(result["root_cause"]))

print("\nMethod / weight comparison (simulated true cause: " + target + "):")
for group, picks in summarize_comparison(agent.compare(bank, answers, kb)).items():
    for name, sid in picks.items():
        print(f"   {group:<10} {name:<32} -> {sid}  {'OK' if sid == target else ''}")
