#!/usr/bin/env python3
"""Check how well the corpus matches the concept map (topics.json), at three levels.

Level 1  IDs      every topic, subconcept and misconception has chunks, and no chunk uses an id
                  that is not in the map
Level 2  content  a chunk labelled with a subconcept visibly contains that subconcept (SIGNAL)
Level 3  order    a chunk's code does not need a topic that the map teaches later (CONSTRUCT and
                  the prerequisites in topics.json)

SIGNAL and CONSTRUCT are also used by finalize_corpus.py to correct labels and to fill the
`uses_later_topics` field, so the check and the corpus are built from the same rules.

Usage:
    python3 check_alignment.py --map topics.json --corpus corpus_final.jsonl [--show 4]
"""
import argparse
import collections
import json
import re

# What a chunk about each subconcept must visibly contain. Deliberately loose: a chunk that does not
# match is very likely mislabelled.
SIGNAL = {
    "T01.1": r"interpret|compil|machine|execut|python (is|program)|source code",
    "T01.2": r">>>|interactive|script|\.py\b|prompt|shell|terminal|command",
    "T01.3": r"statement|indent|syntax|colon|block|keyword|reserved",
    "T01.4": r"#|comment",
    "T01.5": r"error|traceback|exception",
    "T02.1": r"\w\s*=\s*\S|variable",
    "T02.2": r"\bint\b|float|\bstr\b|\bbool\b|integer|string|boolean|\btype",
    "T02.3": r"\b(int|float|str|bool)\(|conver",
    "T02.4": r"type\(|dynamic|<class '",
    "T02.5": r"reassign|updat|variable|\w\s*=\s*\w",
    "T03.1": r"print\(",
    "T03.2": r"\bf['\"]|format|\{[^}]*\}",
    "T03.3": r"input\(",
    "T03.4": r"\b(int|float)\(",
    "T03.5": r"split\(",
    "T04.1": r"\*\*|//|%|[+\-*/]|operator",
    "T04.2": r"==|!=|<=|>=|\s<\s|\s>\s|compar",
    "T04.3": r"\b(and|or|not)\b",
    "T04.4": r"[+\-*/]=|updat",
    "T04.5": r"preced|order of operations|parenthes",
    "T04.6": r"\bin\b|\bis\b",
    "T05.1": r"\bif\b",
    "T05.2": r"\belse\b",
    "T05.3": r"\belif\b|chained",
    "T05.4": r"nested|\n[ \t]{4,}if\b",
    "T05.5": r"\b(and|or|not)\b",
    "T05.6": r"truthy|falsy|bool\(|empty|\bNone\b|\bTrue\b|\bFalse\b",
    "T06.1": r"range\(",
    "T06.2": r"\bfor\b",
    "T06.3": r"\bwhile\b",
    "T06.4": r"\bbreak\b|\bcontinue\b",
    "T06.5": r"nested|\n[ \t]{4,}(for|while)\b",
    "T06.6": r"accumul|counter|count|total|sum|largest|smallest|max|min",
    "T07.1": r"\[-?\d+\]|index",
    "T07.2": r"\[\s*-?\w*\s*:\s*-?\w*\s*\]|slic",
    "T07.3": r"immutable|item assignment",
    "T07.4": r"\.(upper|lower|strip|rstrip|lstrip|split|join|replace|find|startswith|endswith|capitalize)\(|method",
    "T07.5": r"\bin\b|\.count\(|\bfor\b|find",
    "T07.6": r"\\n|\\t|escape|triple|\"\"\"|'''|newline",
    "T08.1": r"\[|list",
    "T08.2": r"\.(append|extend|insert|remove|pop|sort)\(|method",
    "T08.3": r"tuple",
    "T08.4": r"\bsets?\b|set\(",
    "T08.5": r"\bin\b|len\(|\bfor\b",
    "T08.6": r"alias|cop(y|ies)|\[:\]|same object|\bis\b|id\(|reference",
    "T09.1": r"\{|dict",
    "T09.2": r"\w\[[^\]\n]+\]\s*=[^=]|\bdel\b|pop\(|update\(|add",
    "T09.3": r"\.get\(|\bin\b|KeyError",
    "T09.4": r"\.(items|keys|values)\(|\bfor\b",
    "T09.5": r"hashab|immutable|tuple|key",
    "T09.6": r"count|nested|histogram|get\(|\+= ?1",
    "T10.1": r"\bdef\b|function|\b\w+\(\)",
    "T10.2": r"parameter|argument|def \w+\([^)]+\)",
    "T10.3": r"\breturn\b|return value",
    "T10.4": r"default|keyword|def \w+\([^)]*=",
    "T10.5": r"local|global|scope",
    "T10.6": r"function|def\b",
    "T11.1": r"\[[^\[\]\n]*\bfor\b[^\[\]\n]*\bin\b|comprehension",
    "T11.2": r"\[[^\[\]\n]*\bfor\b[^\[\]\n]*\bif\b",
    "T11.3": r"\bif\b[^\n]*\belse\b[^\n]*\bfor\b",
    "T11.4": r"\bfor\b[^\]\n]*\bfor\b|nested",
    "T11.5": r"append|comprehension",
    "T12.1": r"Error|Exception|Traceback",
    "T12.2": r"\btry\b|\bexcept\b",
    "T12.3": r"except\s+\(?\w+(Error|Exception)",
    "T12.4": r"\bfinally\b|\belse\b",
    "T12.5": r"\braise\b",
    "T12.6": r"input|valid|ValueError",
    "T13.1": r"open\(|mode|close",
    "T13.2": r"\bwith\b",
    "T13.3": r"\.read(line|lines)?\(|for \w+ in \w+|\bread",
    "T13.4": r"\.write\(|['\"][wa]['\"]|append",
    "T13.5": r"path|FileNotFoundError|No such file|director|cwd",
    "T13.6": r"line|strip|split|startswith",
}

# Code constructs that belong to a topic (searched in code blocks only).
CONSTRUCT = {
    "T06": r"(?m)^\s*(>>> |\.\.\. )?(for|while)\b[^\n]*:",                   # loops
    "T10": r"(?m)^\s*(>>> )?def\s+\w+\(",                                    # function definitions
    "T11": r"\[[^\[\]\n]*\bfor\b[^\[\]\n]*\bin\b",                           # list comprehensions
    "T12": r"(?m)^\s*(>>> )?(try|except)\b",                                 # exception handling
    "T13": r"\bopen\(",                                                      # files
    "T09": r"\.(items|keys|values)\(\)|\{\s*['\"]?\w+['\"]?\s*:\s*[^{}]",    # dictionaries
}

FENCE = re.compile(r"```(\w*)\n(.*?)\n```", re.S)


def body(record):
    """Chunk text without the '**Section > Subsection**' breadcrumb line."""
    return record["text"].split("\n\n", 1)[-1]


def check_ids(rows, topics, show):
    print("=== LEVEL 1: IDs ===")
    topic_ids = [t["topic_id"] for t in topics]
    sub_ids = [s["id"] for t in topics for s in t["subconcepts"]]
    tag_ids = [m["id"] for t in topics for m in t["common_misconceptions"]]
    unknown = [r["chunk_id"] for r in rows if r["topic_id"] not in topic_ids
               or any(s not in sub_ids for s in r["subconcept_ids"])
               or any(m not in tag_ids for m in r["misconception_tags"])]
    print(f"chunks pointing to ids not in the map: {len(unknown)} {unknown[:show]}")

    per_topic = collections.Counter(r["topic_id"] for r in rows)
    per_sub = collections.Counter(s for r in rows for s in r["subconcept_ids"])
    per_tag = collections.Counter(m for r in rows for m in r["misconception_tags"])
    print(f"topics with chunks: {sum(1 for t in topic_ids if per_topic[t])}/{len(topic_ids)} | "
          f"subconcepts: {sum(1 for s in sub_ids if per_sub[s])}/{len(sub_ids)} | "
          f"misconceptions: {sum(1 for m in tag_ids if per_tag[m])}/{len(tag_ids)}")

    print(f"{'topic':6}{'chunks':>7}{'subconcept min/max':>20}{'expl':>6}{'exmpl':>6}{'mistake':>8}{'debug':>6}  difficulty 1/2/3")
    for t in topics:
        chunks = [r for r in rows if r["topic_id"] == t["topic_id"]]
        counts = [per_sub[s["id"]] for s in t["subconcepts"]]
        types = collections.Counter(r["type"] for r in chunks)
        levels = collections.Counter(r["difficulty"] for r in chunks)
        print(f"{t['topic_id']:6}{len(chunks):7}{min(counts):>12}/{max(counts):<7}{types['explanation']:6}"
              f"{types['example']:6}{types['common_mistake']:8}{types['debugging']:6}  "
              f"{levels[1]}/{levels[2]}/{levels[3]}")
    print(f"subconcepts with fewer than 3 chunks: {[(s, per_sub[s]) for s in sub_ids if per_sub[s] < 3]}")
    no_sub = [r["chunk_id"] for r in rows if not r["subconcept_ids"]]
    print(f"chunks with a topic but no subconcept: {len(no_sub)} {no_sub[:show]}")
    return per_sub


def check_content(rows, topics, per_sub, show):
    print("\n=== LEVEL 2: content matches the label ===")
    names = {s["id"]: s["name"] for t in topics for s in t["subconcepts"]}
    missing, checked = collections.defaultdict(list), 0
    for r in rows:
        for s in r["subconcept_ids"]:
            checked += 1
            if not re.search(SIGNAL[s], body(r), re.I):
                missing[s].append(r["chunk_id"])
    n_missing = sum(len(v) for v in missing.values())
    print(f"(chunk, subconcept) labels checked: {checked} | label without visible signal: "
          f"{n_missing} ({100 * n_missing / checked:.1f}%)")
    for s, ids in sorted(missing.items(), key=lambda kv: -len(kv[1])):
        print(f"   {s} {names[s][:42]:42} {len(ids):3} of {per_sub[s]:3}  {ids[:show]}")


def check_order(rows, topics, show):
    print("\n=== LEVEL 3: teaching order (prerequisites) ===")
    by_id = {t["topic_id"]: t for t in topics}

    def prerequisites(topic, seen=None):
        seen = set() if seen is None else seen
        for p in by_id[topic]["prerequisites"]:
            if p not in seen:
                seen.add(p)
                prerequisites(p, seen)
        return seen
    allowed = {t: prerequisites(t) | {t} for t in by_id}

    ahead = collections.defaultdict(list)
    for r in rows:
        code = "\n".join(m.group(2) for m in FENCE.finditer(r["text"]) if m.group(1) != "text")
        for topic, pattern in CONSTRUCT.items():
            if topic not in allowed[r["topic_id"]] and re.search(pattern, code):
                ahead[(r["topic_id"], topic)].append(r["chunk_id"])
    print(f"chunks whose code uses a topic not yet taught (per the map's prerequisites): "
          f"{len({c for ids in ahead.values() for c in ids})} of {len(rows)}")
    for (topic, used), ids in sorted(ahead.items()):
        print(f"   {topic} chunk uses {used} ({by_id[used]['name'][:28]}): {len(ids):3}  {ids[:show]}")


def main():
    ap = argparse.ArgumentParser(description="Check the corpus against the concept map")
    ap.add_argument("--map", default="topics.json")
    ap.add_argument("--corpus", default="corpus_final.jsonl")
    ap.add_argument("--show", type=int, default=4, help="example chunk ids to list per finding")
    args = ap.parse_args()
    topics = json.load(open(args.map))
    rows = [json.loads(line) for line in open(args.corpus, encoding="utf-8") if line.strip()]
    per_sub = check_ids(rows, topics, args.show)
    check_content(rows, topics, per_sub, args.show)
    check_order(rows, topics, args.show)


if __name__ == "__main__":
    main()