"""Evaluation report for the study.

    python scripts/evaluate.py                       # report -> data/evaluation/report/
    python scripts/evaluate.py --export-labeling     # CSV for teammates to label quizzes by hand
    python scripts/evaluate.py --import-labels FILE  # read that CSV back as expert labels

Sections of the report (data/evaluation/report/report.md, plus one CSV per table):
  1. Data collected
  2. Root-cause accuracy per method, weight preset and ablation, against the true subtopic
     (expert label if there is one, else the student's own answer in the app)
  3. Agreement between methods (no labels needed)
  4. Misconception check: did students confirm the misconception the system named?
  5. Learning gain after remediation (re-test before/after), per root-cause method
  6. Turn judge (LLM) vs rules on the struggle flags
  7. Response time per agent, from the traces
  8. Student feedback
"""

import argparse
import csv
import json
import os
import statistics
from collections import Counter, defaultdict
from itertools import combinations

from eval_data import EXPERT_LABELS, labels, quizzes, students, true_subtopic
from agents import evaluation_store
from agents.struggle_tracker import load_catalog
from agents.workflow import TRACE_DIR

REPORT_DIR = os.path.join(evaluation_store.EVALUATION_DIR, "report")
LABELING_CSV = os.path.join(evaluation_store.EVALUATION_DIR, "labeling_sheet.csv")


def pct(value):
    return "-" if value is None else f"{value:.0%}"


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


class Report:
    def __init__(self):
        self.lines = ["# Evaluation report", ""]
        os.makedirs(REPORT_DIR, exist_ok=True)

    def section(self, title, note=""):
        self.lines += ["", f"## {title}", ""] + ([note, ""] if note else [])

    def table(self, name, rows):
        if not rows:
            self.lines.append("_No data yet._")
            return
        headers = list(rows[0])
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("|" + "---|" * len(headers))
        for row in rows:
            self.lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
        with open(os.path.join(REPORT_DIR, f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    def text(self, line):
        self.lines.append(line)

    def save(self):
        path = os.path.join(REPORT_DIR, "report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
        print("\n".join(self.lines))
        print(f"\nSaved: {path} (+ CSV per table in {REPORT_DIR})")


def picks(quiz):
    """{"Method: no_kb": sid, "Preset: balanced": sid, "Ablation: ...": sid} for one quiz."""
    comparison = quiz.get("comparison") or {}
    result = {}
    for group, label in (("methods", "Method"), ("presets", "Preset"), ("ablations", "Ablation")):
        for name, sid in (comparison.get(group) or {}).items():
            result[f"{label}: {name}"] = sid
    return result


def root_cause_accuracy(report, all_quizzes, label_by_quiz):
    labelled = [q for q in all_quizzes if q.get("comparison") and true_subtopic(label_by_quiz.get(q["quiz_id"]))]
    report.section("2. Root-cause accuracy", (
        f"{len(labelled)} labelled quizzes. Top-1 = the pick is the true subtopic; same topic = the pick "
        "is in the true subtopic's topic; top-3 (hypothesis only) = the truth is among the 3 candidates."))
    hits, topic_hits = defaultdict(list), defaultdict(list)
    for quiz in labelled:
        truth = true_subtopic(label_by_quiz[quiz["quiz_id"]])
        for name, sid in picks(quiz).items():
            hits[name].append(sid == truth)
            topic_hits[name].append(bool(sid) and sid.split(".")[0] == truth.split(".")[0])
    top3 = [any(c["subtopic_id"] == true_subtopic(label_by_quiz[q["quiz_id"]]) for c in q.get("candidates", []))
            for q in labelled]
    rows = [{"Method / weights": name, "Quizzes": len(h), "Top-1 accuracy": pct(mean(h)),
             "Same-topic accuracy": pct(mean(topic_hits[name]))} for name, h in hits.items()]
    report.table("root_cause_accuracy", rows)
    report.text(f"\nHypothesis top-3 coverage (truth among candidates): {pct(mean(top3))}")
    sources = Counter(label_by_quiz[q["quiz_id"]].get("source", "student") for q in labelled)
    report.text(f"Label sources: {dict(sources)}")

    base = next((n for n in hits if n.startswith("Method: hypothesis")), None)
    if base:
        report.text("\n**Ablations** (accuracy change when one parameter is removed from the base preset):")
        rows = [{"Ablation": name, "Accuracy": pct(mean(h)),
                 "Change vs hypothesis": f"{(mean(h) - mean(hits[base])) * 100:+.0f} pts"}
                for name, h in hits.items() if name.startswith("Ablation") and h]
        report.table("ablation_accuracy", rows)


def agreement(report, all_quizzes):
    analysed = [q for q in all_quizzes if q.get("comparison")]
    report.section("3. Agreement between methods", f"{len(analysed)} quizzes with a root cause; no labels needed.")
    names = sorted({n for q in analysed for n in picks(q) if not n.startswith("Ablation")})
    rows = []
    for a, b in combinations(names, 2):
        same = [picks(q).get(a) == picks(q).get(b) for q in analysed if a in picks(q) and b in picks(q)]
        rows.append({"A": a, "B": b, "Same pick": pct(mean(same)), "Quizzes": len(same)})
    report.table("method_agreement", rows)
    changed = defaultdict(list)
    for q in analysed:
        p = picks(q)
        base = p.get("Method: hypothesis")
        for name, sid in p.items():
            if name.startswith("Ablation"):
                changed[name].append(sid != base)
    report.text("\n**How often removing a parameter changes the pick:**")
    report.table("ablation_changes", [{"Ablation": n, "Pick changed": pct(mean(v))} for n, v in changed.items()])


def misconception_check(report, all_quizzes, label_by_quiz):
    report.section("4. Misconception check", (
        "For wrong answers whose option is tagged with a misconception: how often the student, in the "
        "'why was this wrong?' form, chose that misconception as their reason."))
    tagged = confirmed = 0
    for quiz in all_quizzes:
        label = label_by_quiz.get(quiz["quiz_id"]) or {}
        reasons = {r.get("prob_id"): r for r in label.get("per_question", [])}
        for response in quiz["responses"]:
            if response.get("misconception") and response.get("prob_id") in reasons:
                tagged += 1
                confirmed += reasons[response["prob_id"]].get("misconception") == response["misconception"]
    report.text(f"Tagged wrong answers with a student reason: {tagged}; confirmed: {confirmed}"
                + (f" ({confirmed / tagged:.0%})" if tagged else ""))
    reasons = Counter(r.get("reason") for label in label_by_quiz.values() for r in label.get("per_question", []))
    report.text(f"\nReasons students gave for wrong answers: {dict(reasons)}")


def learning_gain(report, records):
    retests = [dict(r, student_id=kb.get("student_id")) for kb in records for r in kb.get("retests", [])]
    report.section("5. Learning gain after remediation", (
        "Accuracy on the root-cause subtopic in the original quiz (before) and in the re-test after the "
        "review lesson (after). Compare methods: a better root cause should lead to a larger gain."))
    groups = defaultdict(list)
    for r in retests:
        groups[r.get("root_cause_method") or "unknown"].append(r)
    rows = [{"Root-cause method": method, "Re-tests": len(items),
             "Students": len({r["student_id"] for r in items}),
             "Before": pct(mean(r.get("accuracy_before") for r in items)),
             "After": pct(mean(r.get("accuracy_after") for r in items)),
             "Mean gain": "-" if mean(r.get("gain") for r in items) is None
             else f"{mean(r.get('gain') for r in items) * 100:+.0f} pts",
             "Review turns": f"{mean(r.get('remediation_turns') for r in items) or 0:.1f}"}
            for method, items in groups.items()]
    report.table("learning_gain", rows)


def judge_vs_rules(report, records):
    turns = [t for kb in records for t in kb.get("learning_conversations", []) if t.get("rule_flags")]
    llm_turns = [t for t in turns if t.get("signal_source") == "llm"]
    report.section("6. Turn judge vs rules", (
        f"{len(turns)} logged student turns, {len(llm_turns)} judged by the LLM (the rest fell back to rules)."))
    rows = []
    for flag in ("confusion", "hint_request"):
        pairs = [(bool(t.get(flag)), bool(t["rule_flags"].get(flag))) for t in llm_turns]
        rows.append({
            "Flag": flag, "Agreement": pct(mean(a == b for a, b in pairs)),
            "LLM yes": sum(a for a, _ in pairs), "Rules yes": sum(b for _, b in pairs),
            "LLM only": sum(a and not b for a, b in pairs), "Rules only": sum(b and not a for a, b in pairs),
        })
    report.table("judge_vs_rules", rows)
    verdicts = Counter(t.get("answer_verdict") for t in turns if t.get("reply_to_tutor"))
    report.text(f"\nVerdicts on answers to the tutor's questions: {dict(verdicts)}")


def latency(report):
    report.section("7. Response time per agent", "From data/traces (one line per request).")
    per_agent, per_request = defaultdict(list), []
    if os.path.isdir(TRACE_DIR):
        for name in os.listdir(TRACE_DIR):
            for record in evaluation_store.read(os.path.join(TRACE_DIR, name)):
                total = 0
                for step in record.get("trace", []):
                    if step.get("duration_ms") is not None:
                        per_agent[step["agent"]].append(step["duration_ms"])
                        total += step["duration_ms"]
                if total:
                    per_request.append(total)

    def row(name, values):
        values = sorted(values)
        return {"Agent": name, "Calls": len(values), "Mean (s)": f"{statistics.mean(values) / 1000:.1f}",
                "Median (s)": f"{statistics.median(values) / 1000:.1f}",
                "90th pct (s)": f"{values[int(0.9 * (len(values) - 1))] / 1000:.1f}"}

    rows = [row(agent, values) for agent, values in sorted(per_agent.items())]
    if per_request:
        rows.append(row("Whole request", per_request))
    report.table("latency", rows)


def feedback(report):
    records = evaluation_store.read("feedback")
    report.section("8. Student feedback", f"{len(records)} responses, ratings 1 (disagree) to 5 (agree).")
    keys = sorted({k for r in records for k in r.get("ratings", {})})
    report.table("feedback", [{"Statement": k, "Mean": f"{mean(r['ratings'].get(k) for r in records):.2f}",
                               "Responses": sum(k in r.get("ratings", {}) for r in records)} for k in keys])
    comments = [r.get("comment") for r in records if r.get("comment")]
    if comments:
        report.text("\nComments:\n" + "\n".join(f"- {c}" for c in comments))


def export_labeling(all_quizzes):
    names = load_catalog()["subtopic_names"]
    rows = []
    for quiz in all_quizzes:
        if not quiz.get("comparison"):
            continue
        wrong = [r for r in quiz["responses"] if not r.get("is_correct")]
        rows.append({
            "quiz_id": quiz["quiz_id"], "student_id": quiz["student_id"],
            "candidates": "; ".join(f"{c['subtopic_id']} {names.get(c['subtopic_id'], '')}" for c in quiz.get("candidates", [])),
            "wrong_answers": "; ".join(f"{r['prob_id']} chose {r.get('chosen')} ({', '.join(r.get('subtopics', []))})" for r in wrong),
            "true_subtopic": "", "true_misconception": "", "labeller": "", "notes": "",
        })
    os.makedirs(os.path.dirname(LABELING_CSV), exist_ok=True)
    with open(LABELING_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["quiz_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} quizzes to {LABELING_CSV}. The method picks are left out on purpose so the "
          "labeller is not biased. Read the student's chat/trace, fill true_subtopic (e.g. T08.6) and "
          "true_misconception (an id from topics.json), then run --import-labels.")


def import_labels(path):
    count = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("true_subtopic", "").strip() or row.get("true_misconception", "").strip():
                _append_expert(row)
                count += 1
    print(f"Imported {count} expert labels into {EXPERT_LABELS}.")


def _append_expert(row):
    os.makedirs(os.path.dirname(EXPERT_LABELS), exist_ok=True)
    with open(EXPERT_LABELS, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "quiz_id": row["quiz_id"], "student_id": row.get("student_id"), "source": "expert",
            "true_subtopic": row.get("true_subtopic", "").strip() or None,
            "true_misconception": row.get("true_misconception", "").strip() or None,
            "labeller": row.get("labeller", "").strip(), "notes": row.get("notes", "").strip(),
        }) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-labeling", action="store_true")
    parser.add_argument("--import-labels", metavar="CSV")
    args = parser.parse_args()

    records = students()
    all_quizzes = quizzes(records)
    if args.export_labeling:
        export_labeling(all_quizzes)
        return
    if args.import_labels:
        import_labels(args.import_labels)
        return

    label_by_quiz = labels()
    report = Report()
    report.section("1. Data collected")
    turns = sum(len(kb.get("learning_conversations", [])) for kb in records)
    report.table("data_overview", [{
        "Students": len(records), "Quizzes": len(all_quizzes),
        "Quizzes with a root cause": sum(bool(q.get("comparison")) for q in all_quizzes),
        "Labelled quizzes": sum(q["quiz_id"] in label_by_quiz for q in all_quizzes),
        "Re-tests": sum(len(kb.get("retests", [])) for kb in records),
        "Student turns": turns,
        "Consents": len(evaluation_store.read("consent")),
    }])
    root_cause_accuracy(report, all_quizzes, label_by_quiz)
    agreement(report, all_quizzes)
    misconception_check(report, all_quizzes, label_by_quiz)
    learning_gain(report, records)
    judge_vs_rules(report, records)
    latency(report)
    feedback(report)
    report.save()


if __name__ == "__main__":
    main()
