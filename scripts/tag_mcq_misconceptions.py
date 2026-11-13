"""One-time data preparation: label each wrong quiz option with the misconception it reveals.

Why it exists: a wrong answer tells us which subtopic is weak, but the chosen wrong option often
shows more: the wrong belief behind it (e.g. "sort() returns the sorted list"). With these
labels the root-cause agent can name the student's misconception, not just the weak subtopic.

When to run it: once, to prepare the quiz bank (and again with --redo if the questions or the
misconception list change). The default method asks the local LLM (needs the GPU);
--method lexical needs no model but is much weaker.

It adds to each question in data/common/mcqs_list.json:
    "option_misconceptions": {"B": "sort_returns_none", "C": null, "D": "append_vs_extend"}
    "misconception_tag_source": "llm" | "lexical"
The misconception ids come from data/common/topics.json ("common_misconceptions" of the
question's topics).

Run from the project root:
    python scripts/tag_mcq_misconceptions.py --limit 10 --dry-run   # try 10 questions, write nothing
    python scripts/tag_mcq_misconceptions.py                        # tag all untagged questions (LLM)
    python scripts/tag_mcq_misconceptions.py --method lexical       # no model: word-overlap matching (weaker)
    python scripts/tag_mcq_misconceptions.py --redo                 # re-tag questions already tagged

A review sheet is always written to data/evaluation/mcq_misconception_review.csv: check a sample
by hand and fix wrong tags in mcqs_list.json (and report the checked accuracy).
"""

import argparse
import csv
import json
import os
import re
import sys

# Make the project root importable so "agents" can be found when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.struggle_tracker import load_catalog, tokenize  # noqa: E402

MCQS_FILEPATH = "data/common/mcqs_list.json"
REVIEW_FILEPATH = "data/evaluation/mcq_misconception_review.csv"
SAVE_EVERY = 20   # save progress to mcqs_list.json after every N questions

PROMPT = """You label multiple-choice Python questions for a tutoring system.
For EACH WRONG option, decide which misconception (from the list) a student who picks it most
likely has. Use null when no listed misconception fits (e.g. a random or careless choice).

Question: {question}
Options:
{options}
Correct option: {correct}
Explanation: {explanation}

Misconceptions (id: description):
{candidates}

Reply with one JSON object mapping each wrong option letter to a misconception id or null,
for example {{"B": "sort_returns_none", "C": null, "D": "append_vs_extend"}}. JSON only."""


def candidate_misconceptions(question: dict, catalog: dict) -> list:
    """Misconception ids that could apply: those of the question's topics (and its subtopics' topics)."""
    topics = list(dict.fromkeys(
        list(question.get("related_topics", []))
        + [str(s).split(".")[0] for s in question.get("related_subtopics", [])]
    ))
    return [m for t in topics for m in catalog["topic_misconceptions"].get(t, [])]


def tag_lexical(question: dict, candidates: list, catalog: dict) -> dict:
    """Tag wrong options by word overlap with each misconception (no model needed).

    An option gets a tag only when the best match shares at least 2 words and is clearly better
    than the second best. Weak (most options get null); prefer the LLM method when possible.
    Returns {option letter: misconception id or None} for every wrong option.
    """
    tags = {}
    for letter, text in question.get("options", {}).items():
        if letter == question.get("correct_option"):
            continue
        words = tokenize(text)
        scored = sorted(
            ((len(words & tokenize(catalog["misconceptions"][m] + " " + m.replace("_", " "))), m) for m in candidates),
            reverse=True,
        )
        clear = scored and scored[0][0] >= 2 and (len(scored) == 1 or scored[0][0] > scored[1][0])
        tags[letter] = scored[0][1] if clear else None
    return tags


def tag_llm(question: dict, candidates: list, catalog: dict, llm) -> dict:
    """Ask the LLM which misconception each wrong option reveals.

    Answers that are not in the candidate list become None.
    Returns {option letter: misconception id or None} for every wrong option.
    """
    from langchain_core.messages import HumanMessage
    prompt = PROMPT.format(
        question=question.get("prob_body", ""),
        options="\n".join(f"{k}: {v}" for k, v in question.get("options", {}).items()),
        correct=question.get("correct_option"),
        explanation=question.get("explanations", ""),
        candidates="\n".join(f"{m}: {catalog['misconceptions'][m]}" for m in candidates),
    )
    raw = llm.invoke([HumanMessage(content=prompt)]).content
    # Drop any <think> block and keep only the JSON object.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    start, end = raw.find("{"), raw.rfind("}")
    data = json.loads(raw[start:end + 1]) if start != -1 and end != -1 else {}
    tags = {}
    for letter in question.get("options", {}):
        if letter == question.get("correct_option"):
            continue
        value = data.get(letter)
        tags[letter] = value if value in candidates else None
    return tags


def main() -> None:
    """Tag the questions, save them to mcqs_list.json (unless --dry-run) and write the review sheet."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=("llm", "lexical"), default="llm")
    parser.add_argument("--limit", type=int, default=None, help="tag at most N questions")
    parser.add_argument("--redo", action="store_true", help="also re-tag questions that already have tags")
    parser.add_argument("--dry-run", action="store_true", help="only write the review CSV")
    args = parser.parse_args()

    catalog = load_catalog()
    with open(MCQS_FILEPATH, "r", encoding="utf-8") as f:
        questions = json.load(f)
    todo = [q for q in questions if args.redo or "option_misconceptions" not in q]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} question(s) to tag with method '{args.method}'.")
    if not todo:
        return

    llm = None
    if args.method == "llm":
        from agents.llm_factory import get_llm
        llm = get_llm("judge", enable_thinking=False)

    rows, failures = [], 0
    for index, question in enumerate(todo, 1):
        candidates = candidate_misconceptions(question, catalog)
        try:
            tags = (tag_llm(question, candidates, catalog, llm) if llm
                    else tag_lexical(question, candidates, catalog)) if candidates else {}
        except Exception as error:             # keep going; the question stays untagged
            failures += 1
            print(f"  {question['prob_id']}: failed ({error})")
            continue
        if not args.dry_run:
            question["option_misconceptions"] = tags
            question["misconception_tag_source"] = args.method
        for letter, tag in tags.items():
            rows.append({
                "prob_id": question["prob_id"], "question": question.get("prob_body", "")[:200],
                "option": letter, "option_text": question["options"].get(letter, ""),
                "correct_option": question.get("correct_option"), "misconception": tag or "",
                "description": catalog["misconceptions"].get(tag, "") if tag else "", "ok (y/n)": "",
            })
        print(f"[{index}/{len(todo)}] {question['prob_id']}: {tags}")
        if not args.dry_run and index % SAVE_EVERY == 0:
            with open(MCQS_FILEPATH, "w", encoding="utf-8") as f:
                json.dump(questions, f, indent=2, ensure_ascii=False)

    if not args.dry_run:
        with open(MCQS_FILEPATH, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
    os.makedirs(os.path.dirname(REVIEW_FILEPATH), exist_ok=True)
    with open(REVIEW_FILEPATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["prob_id"])
        writer.writeheader()
        writer.writerows(rows)
    tagged = sum(bool(r["misconception"]) for r in rows)
    print(f"\nWrong options: {len(rows)}, with a misconception: {tagged}, failed questions: {failures}.")
    print(f"Review sheet: {REVIEW_FILEPATH}" + ("" if not args.dry_run else " (dry run: mcqs_list.json unchanged)"))


if __name__ == "__main__":
    main()
