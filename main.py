import time
from transformers import set_seed
from agents.tutor_agent import TutorAgent

DB = "data/skb_database.json"
question = "Why does my file get erased when I open it?"

cases = [
    ("WITHOUT SKB (student S1000)", "S1000", None),
    ("WITH SKB (student S999)", "S999", None),
    ("WITH SKB + focus_topic T13 (File Handling)", "S999", "T13"),
]

tutor = TutorAgent(db_filepath=DB)

for title, student, focus in cases:
    set_seed(42)
    start = time.time()
    answer = tutor.explain(student, question, focus_topic=focus)
    print(f"\n{'=' * 70}\n{title}   ({time.time() - start:.1f}s)\n{'=' * 70}")
    print(answer)