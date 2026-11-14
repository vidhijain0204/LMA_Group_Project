"""Compare the four retrieval methods (TF-IDF, BM25, dense, hybrid) and one or more embedding models.

    python scripts/compare_methods.py                       # everything (run it on the GPU machine)
    python scripts/compare_methods.py --lexical-only        # TF-IDF and BM25 only, no embedding model
    python scripts/compare_methods.py --models Qwen/Qwen3-Embedding-0.6B

Three test sets, all searched without topic information so that the methods are compared on their own:
  A  22 annotated queries (data/common/retrieval_eval.csv): the relevant chunks are known exactly.
  B  the quiz bank (data/common/mcqs_list.json): a chunk counts as relevant if it shares a subtopic
     with the question. Labels are weak, but there are hundreds of questions.
  C  student-style questions (data/evaluation/student_questions.csv): relevant = the right topic.

For every method the report gives the results, and for every method except BM25 the difference from BM25
with a 95% bootstrap interval, one primary measure per set (A: Recall@k, B: MRR, C: precision@k).
The recommendation follows a fixed rule: a method replaces BM25 only if it is reliably better on at least
one primary measure (interval above zero) and reliably worse on none. Otherwise BM25 stays.

Writes data/evaluation/method_comparison/report.md plus CSV files. Nothing else is modified.
"""
import argparse
import csv
import gc
import json
import os
import random
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CORPUS = "data/common/corpus.jsonl"
ANNOTATED = "data/common/retrieval_eval.csv"
QUIZ_BANK = "data/common/mcqs_list.json"
STUDENT_QUESTIONS = "data/evaluation/student_questions.csv"
DEFAULT_MODELS = ["Qwen/Qwen3-Embedding-0.6B"]
PRIMARY = {"A": "recall", "B": "rr", "C": "precision"}   # the measure used for the verdicts, per set
SET_NAMES = {"A": "annotated queries", "B": "quiz bank (weak labels)", "C": "student-style questions"}


# ------------------------------------------------------------------------------------ data
def short(model_name):
    return model_name.split("/")[-1]


def read_csv(path):
    with open(os.path.join(ROOT, path), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_sets(bank_limit):
    corpus = [json.loads(line) for line in open(os.path.join(ROOT, CORPUS), encoding="utf-8") if line.strip()]
    chunk_subs = {c["chunk_id"]: set(c.get("subconcept_ids") or []) for c in corpus}
    chunk_topic = {c["chunk_id"]: c["topic_id"] for c in corpus}
    sets = {"A": [], "B": [], "C": []}
    for i, row in enumerate(read_csv(ANNOTATED)):
        if row.get("relevant_chunk_ids", "").strip():
            sets["A"].append({"id": row.get("query_id") or f"A{i:02d}", "query": row["query"], "gold": set(row["relevant_chunk_ids"].split(";"))})
    with open(os.path.join(ROOT, QUIZ_BANK), encoding="utf-8") as f:
        for q in json.load(f):
            subtopics = set(q.get("related_subtopics") or [])
            if subtopics and any(subs & subtopics for subs in chunk_subs.values()):   # skip questions no chunk could answer
                sets["B"].append({"id": q["prob_id"], "query": q["prob_body"], "subtopics": subtopics})
    if bank_limit:
        sets["B"] = sets["B"][:bank_limit]
    if os.path.exists(os.path.join(ROOT, STUDENT_QUESTIONS)):
        for i, row in enumerate(read_csv(STUDENT_QUESTIONS)):
            sets["C"].append({"id": f"C{i:02d}", "query": row["question"], "topics": set(row["topic_ids"].split("|"))})
    return sets, chunk_subs, chunk_topic


# ------------------------------------------------------------------------------------ metrics
def query_metrics(set_name, item, ids, chunk_subs, chunk_topic, k):
    """Metrics of one ranked list of chunk ids for one query."""
    if set_name == "A":
        gold = item["gold"]
        return {"recall": len(set(ids[:k]) & gold) / len(gold), "hit": float(bool(set(ids[:k]) & gold)),
                "rr": next((1 / (i + 1) for i, c in enumerate(ids[:10]) if c in gold), 0.0)}
    if set_name == "B":
        flags = [bool(chunk_subs[c] & item["subtopics"]) for c in ids]
    else:
        flags = [chunk_topic[c] in item["topics"] for c in ids]
    return {"hit": float(any(flags[:k])), "precision": sum(flags[:k]) / k,
            "rr": next((1 / (i + 1) for i, ok in enumerate(flags[:10]) if ok), 0.0)}


def evaluate_method(retriever, method, sets, chunk_subs, chunk_topic, k):
    """Run every query of every set; returns {set: [metrics per query]} and the latency of each search."""
    results, latencies = {s: [] for s in sets}, []
    for set_name, items in sets.items():
        for item in items:
            if method in ("dense", "hybrid"):
                retriever.clear_query_cache()          # time the real cost of embedding a new question
            start = time.perf_counter()
            found = retriever.search(item["query"], limit=10, method=method, include_context=False)
            latencies.append((time.perf_counter() - start) * 1000)
            results[set_name].append(query_metrics(set_name, item, [r["chunk_id"] for r in found], chunk_subs, chunk_topic, k))
    return results, latencies


def topic_boost_precision(retriever, method, items, chunk_topic, k):
    """Share of on-topic chunks when the orchestrator's topic is used as a soft preference: right, and wrong on 1 question in 5."""
    all_topics = sorted(set(chunk_topic.values()))
    right, wrong = [], []
    for i, item in enumerate(items):
        true_topic = sorted(item["topics"])[0]
        for bucket, topic in ((right, true_topic),
                              (wrong, next(t for t in all_topics if t not in item["topics"]) if i % 5 == 0 else true_topic)):
            retriever.clear_query_cache()
            found = retriever.search(item["query"], limit=k, method=method, preferred_topics=[topic], include_context=False)
            bucket.append(statistics.mean([chunk_topic[r["chunk_id"]] in item["topics"] for r in found]) if found else 0.0)
    return statistics.mean(right), statistics.mean(wrong)


def mean_of(rows, key):
    return statistics.mean(r[key] for r in rows) if rows else float("nan")


def paired_interval(values, baseline, resamples, seed=0):
    """Mean difference (method minus BM25) with a 95% bootstrap interval over the queries."""
    diffs = [a - b for a, b in zip(values, baseline)]
    rng, n = random.Random(seed), len(diffs)
    means = sorted(statistics.mean(rng.choices(diffs, k=n)) for _ in range(resamples))
    return statistics.mean(diffs), means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def verdict(low, high):
    return "better" if low > 0 else "worse" if high < 0 else "no reliable difference"


def recommend(paired, latency):
    """Apply the fixed rule. paired: {method: {set: (diff, low, high)}}."""
    better = {m: sum(1 for s in v if v[s][1] > 0) for m, v in paired.items()}
    worse = {m: sum(1 for s in v if v[s][2] < 0) for m, v in paired.items()}
    candidates = [m for m in paired if worse[m] == 0 and better[m] >= 1]
    if not candidates:
        reliably_worse = [m for m in paired if worse[m] > 0]
        text = ("**Keep BM25 as the default.** No other method is reliably better than BM25 on any primary measure "
                "without being reliably worse on another.")
        if reliably_worse:
            text += " Reliably worse on at least one measure: " + ", ".join(reliably_worse) + "."
        return text
    best = max(candidates, key=lambda m: (better[m], sum(v[0] for v in paired[m].values())))
    return (f"**Use {best}.** It is reliably better than BM25 on {better[best]} of {len(paired[best])} primary measures and "
            f"reliably worse on none. Its median search time is {latency[best]:.1f} ms against {latency['bm25']:.1f} ms for BM25, "
            f"and it needs the embedding model loaded next to the tutor model.")


# ------------------------------------------------------------------------------------ report
def table(rows, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(str(row[c]) for c in columns) + " |" for row in rows]
    return "\n".join(lines)


def write_csv(out_dir, name, rows):
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))   # rows of different sets have different columns
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)


def free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Compare retrieval methods and embedding models")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="comma-separated embedding models")
    parser.add_argument("--lexical-only", action="store_true", help="only TF-IDF and BM25 (no embedding model)")
    parser.add_argument("--k", type=int, default=4, help="chunks given to the tutor (default 4)")
    parser.add_argument("--bank-limit", type=int, default=None, help="use only the first N quiz-bank questions")
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples (default 2000)")
    parser.add_argument("--device", default=None, help="device for the embedding model, e.g. cuda or cpu")
    parser.add_argument("--no-instruction-ablation", action="store_true", help="skip the Qwen3 instruction variants")
    parser.add_argument("--out", default="data/evaluation/method_comparison")
    args = parser.parse_args()

    from agents.course_retriever import CourseRetriever

    out_dir = os.path.join(ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    sets, chunk_subs, chunk_topic = load_sets(args.bank_limit)
    sets = {s: items for s, items in sets.items() if items}
    print("test sets: " + ", ".join(f"{s} = {len(items)} {SET_NAMES[s]}" for s, items in sets.items()))
    corpus_path = os.path.join(ROOT, CORPUS)
    cache_dir = os.path.join(ROOT, ".cache", "course_retriever")

    results, latency, boost, ablation, skipped, per_query = {}, {}, {}, [], [], []

    def record(label, method, retriever, store_boost=True):
        print(f"  evaluating {label} ...")
        res, lat = evaluate_method(retriever, method, sets, chunk_subs, chunk_topic, args.k)
        results[label], latency[label] = res, statistics.median(lat)
        if store_boost and "C" in sets:
            boost[label] = topic_boost_precision(retriever, method, sets["C"], chunk_topic, args.k)
        for s, rows in res.items():
            for item, row in zip(sets[s], rows):
                per_query.append({"set": s, "query_id": item["id"], "method": label, **{k: round(v, 4) for k, v in row.items()}})
        return res

    lexical = CourseRetriever(corpus_path, embedding_cache_dir=cache_dir)
    for method in ("tfidf", "bm25"):
        record(method, method, lexical)

    if not args.lexical_only:
        for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
            print(f"embedding model {model_name}")
            retriever = None
            try:
                retriever = CourseRetriever(corpus_path, embedding_model_name=model_name, embedding_cache_dir=cache_dir,
                                            embedding_device=args.device)
                start = time.perf_counter()
                retriever.search("warm up question", method="dense")      # loads the model and embeds the chunks once
                print(f"  model loaded and {len(retriever.documents)} chunks embedded in {time.perf_counter() - start:.0f} s")
                for method in ("dense", "hybrid"):
                    record(f"{method} ({short(model_name)})", method, retriever)
                if "qwen3-embedding" in model_name.lower() and not args.no_instruction_ablation:
                    default = (retriever.embedding_query_prompt, retriever.embedding_query_prompt_name)
                    for name, prompt, prompt_name in (("tutoring instruction (used above)", default[0], default[1]),
                                                      ("built-in 'query' prompt", None, "query"),
                                                      ("no instruction", None, None)):
                        retriever.embedding_query_prompt, retriever.embedding_query_prompt_name = prompt, prompt_name
                        retriever.clear_query_cache()
                        for method in ("dense", "hybrid"):
                            label = f"{method} ({short(model_name)})"
                            if name.startswith("tutoring"):
                                res = results[label]                      # already evaluated above: reuse it
                            else:
                                print(f"  instruction variant: {name} / {method}")
                                res, _ = evaluate_method(retriever, method, sets, chunk_subs, chunk_topic, args.k)
                            ablation.append({"instruction": name, "method": label,
                                             **{f"{s} {PRIMARY[s]}": f"{mean_of(res[s], PRIMARY[s]):.3f}" for s in sets}})
                    retriever.embedding_query_prompt, retriever.embedding_query_prompt_name = default
            except Exception as error:                                  # a missing library or a failed download must not lose the rest
                print(f"  SKIPPED {model_name}: {type(error).__name__}: {error}")
                skipped.append((model_name, f"{type(error).__name__}: {error}"))
            finally:
                del retriever
                free_memory()

    # ---- summary table
    k = args.k
    columns = ["method"]
    if "A" in sets:
        columns += [f"A recall@{k}", f"A hit@{k}", "A MRR"]
    if "B" in sets:
        columns += [f"B hit@{k}", f"B precision@{k}", "B MRR"]
    if "C" in sets:
        columns += [f"C precision@{k}", f"C hit@{k}"]
    columns += ["median ms"]
    summary = []
    for label, res in results.items():
        row = {"method": label, "median ms": f"{latency[label]:.1f}"}
        if "A" in sets:
            row.update({f"A recall@{k}": f"{mean_of(res['A'], 'recall'):.3f}", f"A hit@{k}": f"{mean_of(res['A'], 'hit'):.3f}", "A MRR": f"{mean_of(res['A'], 'rr'):.3f}"})
        if "B" in sets:
            row.update({f"B hit@{k}": f"{mean_of(res['B'], 'hit'):.3f}", f"B precision@{k}": f"{mean_of(res['B'], 'precision'):.3f}", "B MRR": f"{mean_of(res['B'], 'rr'):.3f}"})
        if "C" in sets:
            row.update({f"C precision@{k}": f"{mean_of(res['C'], 'precision'):.3f}", f"C hit@{k}": f"{mean_of(res['C'], 'hit'):.3f}"})
        summary.append(row)

    # ---- paired comparison with BM25
    paired, paired_rows = {}, []
    for label, res in results.items():
        if label == "bm25":
            continue
        paired[label], row = {}, {"method": label}
        for s in sets:
            key = PRIMARY[s]
            diff, low, high = paired_interval([r[key] for r in res[s]], [r[key] for r in results["bm25"][s]], args.bootstrap)
            paired[label][s] = (diff, low, high)
            row[f"{s} {key}@{k}" if key in ("recall", "precision") else f"{s} MRR"] = f"{diff:+.3f} [{low:+.3f}, {high:+.3f}] {verdict(low, high)}"
        paired_rows.append(row)

    # ---- the report
    n = {s: len(items) for s, items in sets.items()}
    report = ["# Retrieval method comparison",
              f"Generated {time.strftime('%Y-%m-%d %H:%M')}. k = {k} (the number of chunks given to the tutor). "
              f"Test sets: " + ", ".join(f"{s} = {n[s]} {SET_NAMES[s]}" for s in sets) + ".",
              "\n## 1. Results\n\nAll searches without topic information. Higher is better for every column except time.\n",
              table(summary, columns)]
    if paired_rows:
        report += ["\n## 2. Difference from BM25 (95% bootstrap interval)\n",
                   "Each cell is the mean difference from BM25 on one primary measure, the interval, and a verdict. "
                   "'better' or 'worse' only when the interval excludes zero. Set A has few queries, so its intervals are wide.\n",
                   table(paired_rows, list(paired_rows[0].keys()))]
    if boost:
        rows = []
        for label, (right, wrong) in boost.items():
            res_c = results[label]["C"]
            rows.append({"method": label, "no topic": f"{mean_of(res_c, 'precision'):.0%}", "topic right (soft boost)": f"{right:.0%}",
                         "topic wrong on 1 in 5": f"{wrong:.0%}"})
        report += ["\n## 3. Using the orchestrator's topic as a soft preference (student-style questions)\n",
                   "Share of the retrieved chunks that are on-topic. This is how the tutor calls the retriever.\n",
                   table(rows, ["method", "no topic", "topic right (soft boost)", "topic wrong on 1 in 5"])]
        write_csv(out_dir, "topic_boost.csv", rows)
    if ablation:
        report += ["\n## 4. Qwen3 query instruction\n",
                   "Only the question is changed; the chunks are embedded without an instruction. The first variant is the one "
                   "used in the tables above and was chosen before running. Picking the best of the three on this data "
                   "would be slightly optimistic.\n",
                   table(ablation, list(ablation[0].keys()))]
        write_csv(out_dir, "instruction_ablation.csv", ablation)
    advice = recommend(paired, latency) if paired else "Only lexical methods were run; nothing to compare with BM25."
    if skipped:
        advice += ("\n\n**This only covers the methods that ran.** Not evaluated: " + ", ".join(name for name, _ in skipped)
                   + " (see Skipped below).")
    elif args.lexical_only:
        advice += "\n\n**Dense and hybrid were not evaluated** (--lexical-only)."
    report += ["\n## 5. Recommendation\n", advice]
    if skipped:
        report += ["\n## Skipped\n"] + [f"- {name}: {why}" for name, why in skipped]
    report += ["\n## How to read this\n",
               "- Differences smaller than the intervals are noise. With 22 annotated queries one query moves a score by about 4 points.",
               "- Set B labels are weak (sharing a subtopic does not prove a chunk answers the question), so use it to compare methods, not as an absolute score.",
               "- Dense and hybrid need the embedding model next to the tutor model on the GPU, and a search takes longer than BM25."]

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    write_csv(out_dir, "summary.csv", summary)
    write_csv(out_dir, "paired_vs_bm25.csv", paired_rows)
    write_csv(out_dir, "per_query.csv", per_query)
    print("\n" + "\n".join(report[2:]))
    print(f"\nReport and CSV files written to {args.out}/")


if __name__ == "__main__":
    main()
