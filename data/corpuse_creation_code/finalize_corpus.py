#!/usr/bin/env python3
"""Turn the scraped corpus into the final, teaching-ready corpus.

Input is the output of `scrape_corpus.py`; output is `corpus_final.jsonl` plus a short report.
Every change made to a chunk is recorded in its `edits` field.

Steps
  1. Repair chunk boundaries: an "Output:" block that starts a chunk is moved back to the chunk
     holding its code.
  2. Clean each chunk:
       - drop chunks that are off-topic or need material we do not have
       - remove exercise prompts, IPython-only cells and references to the source books
       - fix code that is wrong for Python 3.10+ (Python 2 output, 3.14/3.15-only syntax, known errata)
       - fix output that only a Jupyter notebook would show
       - add a note to examples that read a textbook data file
  3. Split chunks that are too long for embedding models.
  4. Add the hand-written FINALIZE_CHUNKS from build_original_chunks.py (examples are executed first).
  5. Correct subconcept labels that the chunk's content does not support (rules from check_alignment.py).
  6. Add difficulty, misconception tags, context links, near-duplicate groups and teaching-order fields.

Usage:
    python3 finalize_corpus.py --in corpus_out/corpus_chunks.jsonl --out corpus_final.jsonl \
                               --report finalize_report.md [--topics topics.json]
"""
import argparse
import ast
import builtins
import collections
import contextlib
import hashlib
import io
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from original_chunks import FINALIZE_CHUNKS
from LGP.corpuse_creation_code.check_alignment import CONSTRUCT, SIGNAL

FENCE = re.compile(r"```(\w*)\n(.*?)\n```", re.S)


# =============================================================================================
# Text helpers
# =============================================================================================
def words(s):
    return len(s.split())


def split_header(text):
    """Split the '**Section > Subsection**' breadcrumb line from the chunk body."""
    if text.startswith("**") and "\n\n" in text:
        return tuple(text.split("\n\n", 1))
    return "", text


def tokenize(body):
    """Split a chunk body into [kind, text, lang] tokens; kind is 'text', 'code' or 'output'."""
    tokens, pos = [], 0
    for m in FENCE.finditer(body):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body[pos:m.start()]) if p.strip()]
        is_output = bool(paragraphs) and paragraphs[-1] == "Output:" and m.group(1) == "text"
        if is_output:
            paragraphs = paragraphs[:-1]
        tokens += [["text", p, ""] for p in paragraphs]
        tokens.append(["output" if is_output else "code", m.group(2), m.group(1) or "python"])
        pos = m.end()
    tokens += [["text", p.strip(), ""] for p in re.split(r"\n\s*\n", body[pos:]) if p.strip()]
    return tokens


def render(tokens):
    parts = []
    for kind, text, lang in tokens:
        if kind == "text":
            parts.append(text)
        elif kind == "output":
            parts.append(f"Output:\n```text\n{text}\n```")
        else:
            parts.append(f"```{lang}\n{text}\n```")
    return "\n\n".join(parts)


def sentences(text):
    return re.split(r"(?<=[.!?])\s+(?=[A-Z`(\"“0-9])", text)


def capitalise_sentences(text):
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def content_hash(text):
    return hashlib.sha256(re.sub(r"\s+", " ", text.lower()).encode()).hexdigest()[:16]


# =============================================================================================
# Cleaning rules
# =============================================================================================
# Chunks removed entirely: (pattern in body, reason)
DROP_CHUNK = [
    (r"zip\(\*matrix\)", "off-topic aside (zip/star-unpacking)"),
    (r"from structshape import", "needs a helper module from the book's website"),
    (r"You don.t even need to know Python to use this program", "narrative about a later program, not teaching"),
    (r"isatty\(\)|seek\(0, 2\)", "advanced file-object methods (seek/tell/isatty), outside the concept map"),
]
DROP_CHUNK = [(re.compile(p), why) for p, why in DROP_CHUNK]

# Paragraphs that are exercises; the code block that follows them belongs to the exercise too.
EXERCISE = re.compile(r"^(\*\*)?Exercise\s*\d*\s*[:.]", re.I)

# Sentences that only make sense inside the original book or website are removed.
OUTSIDE_REFERENCE = re.compile(
    r"ask a virtual assistant|py4e\.com|screencast|https?://|www\.\w|download the (source )?code"
    r"|(see|try) the (first |second |third |last |next )?exercises?\b|exercises? at the end"
    r"|end of (this|the) chapter|get through Chapter|Chapter \d+ of this book"
    r"|Jupyter notebook, you can use shell commands", re.I)

# Phrases rewritten so the text reads as standalone teaching material: (pattern, replacement)
REWRITES = [
    (r"\s*\((https?://|www\.)[^)]*\)", ""),
    (r"\bIn Chapter \d+(\.\d+)*,?\s+", "Earlier, "),
    (r"\s+(from|in|of) Chapter \d+(\.\d+)*", ""),
    (r"\(see Section [\d.]+\)\s*", ""),
    (r"\b[Ss]ection \d+(\.\d+)*", "an earlier section"),
    (r"\b[Ii]n the previous chapter\b", "Earlier"),
    (r"\b(in )?(the )?(next|later|future|following) chapters?\b", "later"),
    (r"\bin this chapter\b", "here"),
    (r"\bthis chapter\b", "this topic"),
    (r"\bthis book\b", "this tutorial"),
    (r"appears at the end of a cell", "is the last thing you type at the interactive prompt"),
    (r"\bJupyter( notebook)?\b", "the interactive Python prompt"),
    (r"python3\.1[45]", "python3"),
    (r",?\s*including comprehensions with unpacking", ""),
]
REWRITES = [(re.compile(p), r) for p, r in REWRITES]

# Code fixes so every example is correct on Python 3.10+: (pattern, replacement)
CODE_FIXES = [
    (r"(?m)^\s*#\s*Code:\s*https?://\S+\s*$\n?", ""),                         # download links in comments
    (r"python3\.1[45]", "python3"),                                          # version-specific command
    (r"except\s+(\w+(?:\s*,\s*\w+)+)\s*:", r"except (\1):"),                 # 3.14+ unparenthesised except
    (r"<type '(\w+)'>", r"<class '\1'>"),                                    # Python 2 type display
    (r"(>>> print\(sum\(nums\)/len\(nums\)\)\n)25\b", r"\g<1>25.666666666666668"),  # Python 2 integer division
    (r"(>>> print\(t\[0\]\)\n)'a'", r"\1a"),                                 # print() shows no quotes
    (r"(>>> t3 = t1 \+ \[3\]\n>>> print\(t3\)\n)\[1, 2, 3\](?=\n)", r"\1[1, 2, 3, 3]"),  # py4e erratum
]
CODE_FIXES = [(re.compile(p), r) for p, r in CODE_FIXES]
DOUBLE_QUOTED_LIST = re.compile(r'(?m)^\[(?:"[^"\'\\\n]*"(?:, )?)+\]$')      # Python shows lists with ' quotes
COMPREHENSION_UNPACKING = re.compile(r"[\[{(]\s*\*{1,2}[\w\[\]().]+\s+for\b")  # 3.15-only syntax

# Output that only a Jupyter notebook shows, rewritten to what the normal Python prompt shows.
OUTPUT_FIXES = [
    (r"<function __main__\.(\w+)\([^)]*\)>", r"<function \1 at 0x...>"),
    (r"<function (\w+)\(x, /\)>", r"<built-in function \1>"),
    (r"<reversed at (0x[0-9a-f]+)>", r"<reversed object at \1>"),
    (r"Cell In\[\d+\], line (\d+)", r'File "<stdin>", line \1'),
    (r"\n{3,}", "\n"),
]
OUTPUT_FIXES = [(re.compile(p), r) for p, r in OUTPUT_FIXES]
TYPE_NAMES = {"int", "float", "str", "bool", "tuple", "list", "dict", "set", "NoneType", "function", "range"}

# py4e explains Python 2 division; only the Python 3 behaviour is kept.
PY2_DIVISION_PARAGRAPH = "The division operator in Python 2"
PY2_DIVISION_TEXT = [
    ("There has been a change in the division operator between Python 2 and Python 3. ", ""),
    ("To obtain the same answer in Python 3 use floored ( `//` integer) division.",
     "To get a whole-number result, use floored (`//` integer) division:"),
    ("In Python 3 integer division functions much more as you would expect if you entered the expression on a calculator.",
     "Ordinary division with `/` works the way it does on a calculator."),
]

DATA_FILE = re.compile(r"\b([\w-]+\.(?:txt|csv))\b")


def remove_comprehension_unpacking(code, edits):
    """Delete doctest examples using 3.15-only comprehension unpacking, with their output and setup lines."""
    kept, skipping, fed_names = [], False, set()
    for line in code.split("\n"):
        if line.startswith(">>> ") and COMPREHENSION_UNPACKING.search(line):
            skipping = True
            fed_names |= set(re.findall(r"\bin\s+(\w+)", line))
            edits.append("removed Python 3.15-only comprehension unpacking example")
            continue
        if skipping and (line.startswith(">>> ") or not line.strip()):
            skipping = False
        if not skipping:
            kept.append(line)
    if fed_names:  # a setup line like `>>> fruits = [...]` that only fed the removed example
        remaining = "\n".join(kept)
        kept = [ln for ln in kept
                if not ((m := re.match(r">>> (\w+)\s*=", ln)) and m.group(1) in fed_names
                        and remaining.count(m.group(1)) == 1)]
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def clean_text(text, edits):
    """Remove sentences that point outside the corpus and rewrite book-specific phrases."""
    original = text
    if "\n" in text:  # lists: filter sentence by sentence on each line
        text = "\n".join(" ".join(s for s in sentences(line) if not OUTSIDE_REFERENCE.search(s))
                         for line in text.split("\n"))
    else:
        text = " ".join(s for s in sentences(text) if not OUTSIDE_REFERENCE.search(s))
    if len([s for s in sentences(original) if not OUTSIDE_REFERENCE.search(s)]) != len(sentences(original)):
        edits.append("removed sentence referring outside the corpus")
    for pattern, replacement in REWRITES:
        text, n = pattern.subn(replacement, text)
        if n:
            edits.append("rewrote book-specific reference")
    if "\n" not in text:
        text = capitalise_sentences(re.sub(r"\s{2,}", " ", text).strip())
    return text


def clean_code(code, edits):
    before = code
    for pattern, replacement in CODE_FIXES:
        code = pattern.sub(replacement, code)
    if COMPREHENSION_UNPACKING.search(code):
        code = remove_comprehension_unpacking(code, edits)
    code = DOUBLE_QUOTED_LIST.sub(lambda m: m.group(0).replace('"', "'"), code)
    if code != before:
        edits.append("fixed code/output for Python 3.10+")
    return code


def clean_output(output, edits):
    before = output
    for pattern, replacement in OUTPUT_FIXES:
        output = pattern.sub(replacement, output)
    if output != before:
        edits.append("fixed notebook-only output")
    return output


def clean_chunk(record):
    """Return (cleaned record, None) or (None, reason the chunk was dropped)."""
    header, body = split_header(record["text"])
    for pattern, reason in DROP_CHUNK:
        if pattern.search(body):
            return None, reason
    edits = []

    # Exercises (and the code that belongs to them) and IPython-only cells (and their output).
    tokens, in_exercise = [], False
    for tok in tokenize(body):
        if tok[0] == "text":
            in_exercise = bool(EXERCISE.match(tok[1]))
            if in_exercise:
                edits.append("removed exercise prompt")
                continue
        elif in_exercise:
            continue
        if tok[0] == "code" and re.search(r"(?m)^\s*[%!]\w", tok[1]):
            edits.append("removed IPython-only cell")
            in_exercise = True
            continue
        tokens.append(tok)

    for tok in tokens:
        if tok[0] == "text":
            tok[1] = clean_text(tok[1], edits)
        elif tok[0] == "code":
            tok[1] = clean_code(tok[1], edits)
        else:
            tok[1] = clean_output(tok[1], edits)

    # A notebook shows `int` for type(...); the Python prompt shows <class 'int'>.
    for prev, tok in zip(tokens, tokens[1:]):
        if tok[0] == "output" and prev[0] == "code" and tok[1].strip() in TYPE_NAMES \
                and re.search(r"type\([^\n]*\)\s*$", prev[1].strip().split("\n")[-1]):
            tok[1] = f"<class '{tok[1].strip()}'>"
            edits.append("fixed notebook-only output")

    # Python 2 division example: drop its paragraph and code block, reword the surrounding text.
    for i, tok in enumerate(tokens):
        if tok and tok[0] == "text" and tok[1].startswith(PY2_DIVISION_PARAGRAPH):
            tokens[i] = None
            if i + 1 < len(tokens) and tokens[i + 1] and tokens[i + 1][0] == "code":
                tokens[i + 1] = None
            edits.append("removed Python 2 division example")
    tokens = [t for t in tokens if t]
    for tok in tokens:
        if tok[0] == "text":
            for old, new in PY2_DIVISION_TEXT:
                tok[1] = tok[1].replace(old, new)

    # Drop empty paragraphs and any output whose code was removed.
    tokens = [t for t in tokens if not (t[0] == "text" and not t[1].strip())]
    tokens = [t for i, t in enumerate(tokens) if not (t[0] == "output" and (i == 0 or tokens[i - 1][0] != "code"))]

    new_body = render(tokens)
    if words(new_body) < words(body) and words(new_body) < 25:
        return None, f"only {words(new_body)} words left after removing exercises/references"

    cleaned = dict(record)
    cleaned.pop("_pre_edits", None)
    cleaned["text"] = f"{header}\n\n{new_body}" if header else new_body
    cleaned["edits"] = sorted(set(edits + record.get("_pre_edits", [])))
    files = sorted(set(DATA_FILE.findall(new_body)))
    if files and re.search(r"open\(|\.txt|\.csv", new_body):
        cleaned["requires_files"] = files
        cleaned["text"] += ("\n\nNote: this example reads " + ", ".join(f"`{f}`" for f in files) +
                            ", a sample data file from the original textbook. To try it yourself, create a small "
                            "text file with that name (any lines of text will do) in the same folder as your "
                            "program, or use your own file.")
        cleaned["edits"].append("added note: example needs a data file")
    else:
        cleaned["requires_files"] = []
    return cleaned, None


def move_orphaned_outputs(rows):
    """A chunk that starts with an "Output:" block: give the block back to the previous chunk.
    If almost nothing is left, merge the rest into the previous chunk. Returns the ids merged away."""
    merged = set()
    for prev, row in zip(rows, rows[1:]):
        if row["source"].get("url") != prev["source"].get("url") or prev["chunk_id"] in merged:
            continue
        header, body = split_header(row["text"])
        m = re.match(r"\s*Output:\n```text\n(.*?)\n```\s*", body, re.S)
        if not m:
            continue
        prev["text"] = prev["text"].rstrip() + f"\n\nOutput:\n```text\n{m.group(1)}\n```"
        prev.setdefault("_pre_edits", []).append("received its output block back from the next chunk")
        rest = body[m.end():].strip()
        if words(rest) < 25:
            if rest:
                prev["text"] += "\n\n" + rest
            merged.add(row["chunk_id"])
        else:
            row["text"] = f"{header}\n\n{rest}" if header else rest
            row.setdefault("_pre_edits", []).append("moved orphaned output block to previous chunk")
    return merged


# =============================================================================================
# Splitting oversized chunks
# =============================================================================================
def make_part(record, header, n, body):
    part = dict(record)
    part["chunk_id"] = f"{record['chunk_id']}_p{n}"
    part["text"] = (f"{header} (part {n})\n\n" if header else "") + body
    part["edits"] = sorted(set(record.get("edits", []) + ["split oversized chunk"]))
    return part


def split_oversized(record, limit=450):
    """Split chunks over `limit` words: at a paragraph boundary after the middle or, for one long
    list paragraph, at list items (and very long items at sentence boundaries)."""
    if record["word_count"] <= limit:
        return [record]
    header, body = split_header(record["text"])
    tokens = tokenize(body)
    total, running, cut = sum(words(t[1]) for t in tokens), 0, None
    for i, tok in enumerate(tokens):
        running += words(tok[1])
        if running >= total / 2 and i + 1 < len(tokens) and tokens[i + 1][0] == "text":
            cut = i + 1
            break
    if cut is not None:
        return [make_part(record, header, n, render(seg)) for n, seg in enumerate((tokens[:cut], tokens[cut:]), 1)]

    if len(tokens) == 1 and tokens[0][0] == "text":
        budget = limit * 0.75
        items = []
        for item in re.split(r"\n(?=\d+\.\s)", tokens[0][1]):
            if words(item) <= budget:
                items.append(item)
                continue
            current = []
            for sentence in sentences(item):
                if current and words(" ".join(current + [sentence])) > budget:
                    items.append(" ".join(current))
                    current = []
                current.append(sentence)
            items.append(" ".join(current))
        groups, current = [], []
        for item in items:
            if current and words("\n".join(current + [item])) > budget:
                groups.append(current)
                current = []
            current.append(item)
        groups.append(current)
        if len(groups) > 1:
            return [make_part(record, header, n, "\n".join(g)) for n, g in enumerate(groups, 1)]
    return [record]


# =============================================================================================
# Hand-written chunks added after cleaning
# =============================================================================================
def run_examples(text):
    """Execute python blocks in a temporary folder; every "Output:" block must match the real output."""
    problems, namespace, cwd = [], {}, os.getcwd()
    normalise = lambda s: "\n".join(line.rstrip() for line in s.strip().splitlines())
    with tempfile.TemporaryDirectory() as folder:
        os.chdir(folder)
        try:
            for m in re.finditer(r"```python\n(.*?)\n```", text, re.S):
                printed = io.StringIO()
                try:
                    with contextlib.redirect_stdout(printed):
                        exec(m.group(1), namespace)
                except Exception as e:  # noqa: BLE001
                    problems.append(f"raised {type(e).__name__}: {e}")
                    continue
                shown = re.compile(r"\n\nOutput:\n```text\n(.*?)\n```", re.S).match(text, m.end())
                if shown and normalise(shown.group(1)) != normalise(printed.getvalue()):
                    problems.append(f"output mismatch: {normalise(shown.group(1))!r} vs {normalise(printed.getvalue())!r}")
        finally:
            os.chdir(cwd)
    return problems


def build_authored(topics, existing):
    """Turn FINALIZE_CHUNKS into corpus records; ids continue the existing `T##_orig_###` numbering."""
    topic_name = {t["topic_id"]: t["name"] for t in topics}
    known = {r["content_hash"] for r in existing}
    counter = collections.Counter()
    for r in existing:
        m = re.match(r"(T\d\d)_orig_(\d+)", r["chunk_id"])
        if m:
            counter[m.group(1)] = max(counter[m.group(1)], int(m.group(2)))
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = []
    for chunk in FINALIZE_CHUNKS:
        if chunk["verify"]:
            problems = run_examples(chunk["text"])
            if problems:
                raise SystemExit(f"authored chunk '{chunk['title']}' failed verification: {problems}")
        text = f"**{topic_name[chunk['topic']]} > {chunk['title']}**\n\n{chunk['text']}"
        if content_hash(text) in known:
            continue
        counter[chunk["topic"]] += 1
        code_words = sum(words(m.group(2)) for m in FENCE.finditer(text) if m.group(1) != "text")
        records.append({
            "chunk_id": f"{chunk['topic']}_orig_{counter[chunk['topic']]:03d}", "topic_id": chunk["topic"],
            "subconcept_ids": chunk["subs"], "type": chunk["type"], "difficulty": chunk["difficulty"],
            "misconception_tags": chunk["tags"], "text": text,
            "source": {"id": "original", "name": "Original teaching material written for this project", "url": None,
                       "page_title": chunk["title"], "section_path": [topic_name[chunk["topic"]], chunk["title"]],
                       "license": "Project-original (team to choose a licence)",
                       "attribution": "Written for the Personalized Python Tutor project", "redistributable": True},
            "scraped_at": timestamp, "word_count": words(text), "has_code": code_words > 0,
            "mapping": {"method": "authored", "confidence": 1.0, "candidate_topics": None, "scores": {}},
            "flags": [], "content_hash": content_hash(text), "edits": [], "requires_files": [],
        })
    return records


# =============================================================================================
# Labels: subconcept correction, difficulty, misconception tags
# =============================================================================================
def correct_labels(rows, topic_of):
    """Keep only subconcept labels whose signal (check_alignment.SIGNAL) is visible in the chunk."""
    stats = collections.Counter()
    for r in rows:
        if r["source"]["id"] == "original":
            continue
        body = split_header(r["text"])[1]
        labels = r["subconcept_ids"]
        supported = [s for s in labels if re.search(SIGNAL[s], body, re.I)]
        new = labels
        if labels and supported and len(supported) < len(labels):
            new = supported
            stats["removed unsupported second label"] += 1
        elif labels and not supported:
            if "T06.1" in labels and re.search(SIGNAL["T06.2"], body, re.I):
                new = ["T06.2"]  # a for loop without range() is "iterating over a sequence"
                stats["for-loop without range(): T06.1 -> T06.2"] += 1
            elif r["type"] == "debugging":
                new = []         # generic debugging advice is not about one subconcept
                stats["generic debugging section: subconcept cleared"] += 1
        if new != labels:
            r["subconcept_ids"] = new
            r["edits"] = sorted(set(r["edits"] + ["subconcept labels corrected to match content"]))
            if not new and "no_subconcept" not in r.get("flags", []):
                r.setdefault("flags", []).append("no_subconcept")
        if new and all(topic_of[s] != r["topic_id"] for s in new):
            r["edits"].append(f"topic changed {r['topic_id']} -> {topic_of[new[0]]} to match its subconcepts")
            r["topic_id"] = topic_of[new[0]]
            stats["topic changed to match subconcepts"] += 1
    return stats


# Difficulty 1-3 by each subconcept's position in the course; a chunk takes its hardest subconcept.
SUBCONCEPT_DIFFICULTY = {
    "T01.1": 1, "T01.2": 1, "T01.3": 1, "T01.4": 1, "T01.5": 2,
    "T02.1": 1, "T02.2": 1, "T02.3": 1, "T02.4": 2, "T02.5": 1,
    "T03.1": 1, "T03.2": 2, "T03.3": 1, "T03.4": 1, "T03.5": 2,
    "T04.1": 1, "T04.2": 1, "T04.3": 2, "T04.4": 1, "T04.5": 2, "T04.6": 2,
    "T05.1": 1, "T05.2": 1, "T05.3": 2, "T05.4": 2, "T05.5": 2, "T05.6": 2,
    "T06.1": 1, "T06.2": 1, "T06.3": 2, "T06.4": 2, "T06.5": 3, "T06.6": 2,
    "T07.1": 1, "T07.2": 2, "T07.3": 2, "T07.4": 2, "T07.5": 2, "T07.6": 2,
    "T08.1": 1, "T08.2": 2, "T08.3": 2, "T08.4": 2, "T08.5": 1, "T08.6": 3,
    "T09.1": 2, "T09.2": 2, "T09.3": 2, "T09.4": 2, "T09.5": 3, "T09.6": 3,
    "T10.1": 2, "T10.2": 2, "T10.3": 2, "T10.4": 3, "T10.5": 3, "T10.6": 3,
    "T11.1": 2, "T11.2": 2, "T11.3": 3, "T11.4": 3, "T11.5": 3,
    "T12.1": 2, "T12.2": 2, "T12.3": 2, "T12.4": 3, "T12.5": 3, "T12.6": 3,
    "T13.1": 2, "T13.2": 2, "T13.3": 2, "T13.4": 2, "T13.5": 2, "T13.6": 3,
}
TOPIC_DIFFICULTY = {"T01": 1, "T02": 1, "T03": 1, "T04": 1, "T05": 2, "T06": 2, "T07": 2,
                    "T08": 2, "T09": 2, "T10": 2, "T11": 3, "T12": 2, "T13": 2}

# Misconception tags for scraped chunks: (tag, topics where the rule applies, pattern in the body).
# Patterns are deliberately narrow: a tag means "this chunk explains material relevant to the mistake".
TAG_RULES = [
    ("indentation_syntax_error", {"T01", "T05"}, r"IndentationError|indentation (is|matters|error)"),
    ("case_sensitivity_error", {"T01", "T02"}, r"case[- ]sensitive"),
    ("ignoring_traceback", {"T01", "T12"}, r"[Tt]raceback"),
    ("invalid_variable_name", {"T01", "T02"}, r"illegal (variable )?name|is a keyword|reserved words?"),
    ("assignment_as_equation", {"T02"}, r"not (the same as|a statement of) (mathematical )?equality|is not (an )?equal"),
    ("type_mismatch_concat", {"T02", "T04", "T07"}, r"can only concatenate str|unsupported operand type"),
    ("use_before_assignment", {"T02", "T10"}, r"NameError: name '\w+' is not defined"),
    ("string_number_confusion", {"T02", "T03"}, r"'\d+'\s*\+\s*'\d+'|int\('\d+'\)|\"\d+\"\s*\+"),
    ("input_returns_string", {"T03"}, r"input[^.\n]{0,80}(returns?|as) (a |the )?string|int\(input|float\(input"),
    ("fstring_missing_prefix", {"T03"}, r"prefix(ing)? the string with `?f`?|prepend\w* an? `?f`?|`f` before"),
    ("division_type_confusion", {"T02", "T04"}, r"floor(ed)? division|integer division|`//`"),
    ("assign_vs_compare", {"T04", "T05"}, r"single equal sign|`=` is an assignment|assignment operator"),
    ("precedence_error", {"T04"}, r"order of operations|precedence|PEMDAS"),
    ("is_vs_equals", {"T04", "T08"}, r"`is` operator|\bidentical\b|same object"),
    ("modulo_misunderstanding", {"T04"}, r"modulus|modulo|remainder"),
    ("missing_colon", {"T01", "T05", "T10"}, r"\bcolon\b"),
    ("block_scope_indentation", {"T05"}, r"IndentationError|indent\w*[^.\n]{0,80}\b(block|body|belongs?|part of)\b"),
    ("condition_order_error", {"T05"}, r"checked in order|only the first true branch|first true condition"),
    ("off_by_one_range", {"T06", "T07", "T08"}, r"off[- ]by[- ]one|up to,? but not including|but not including"),
    ("infinite_loop_no_update", {"T06"}, r"infinite loop"),
    ("modify_list_while_iterating", {"T06", "T08"}, r"(modif|chang|remov|delet)\w*[^.\n]{0,50}(while|during)[^.\n]{0,25}(iterat|loop)"),
    ("break_continue_confusion", {"T06"}, r"`?break`?[^.\n]{0,80}`?continue`?|`?continue`?[^.\n]{0,80}`?break`?"),
    ("string_immutability_error", {"T07"}, r"immutable|does not support item assignment"),
    ("string_index_out_of_range", {"T07"}, r"IndexError: string index out of range"),
    ("slice_end_exclusive", {"T07", "T08"}, r"up to,? but not including|but not including the|excluding the"),
    ("method_result_not_assigned", {"T07"}, r"returns? a new string|do(es)?n.t modify the (original )?string"),
    ("case_sensitive_comparison", {"T04", "T07"}, r"uppercase letters? (come|are) before|upper[- ]?case[^.\n]{0,40}lower[- ]?case[^.\n]{0,40}(before|compar)"),
    ("list_aliasing", {"T08"}, r"\balias"),
    ("sort_returns_none", {"T08"}, r"return(s)? `?None`?[^.\n]{0,60}(sort|modif)|(sort|append)\w*[^.\n]{0,60}return(s)? `?None"),
    ("append_vs_extend", {"T08"}, r"(?s)(?=.*\bextend\b)(?=.*\bappend\b)"),
    ("tuple_mutation_error", {"T08"}, r"tuples? (are|is) immutable|'tuple' object does not support"),
    ("single_element_tuple", {"T08"}, r"single element|trailing comma|final comma"),
    ("set_ordering_assumption", {"T08"}, r"\bunordered\b|no (particular |guaranteed )?order"),
    ("list_index_out_of_range", {"T08"}, r"IndexError: list index out of range"),
    ("remove_vs_pop", {"T08"}, r"(?s)(?=.*\bpop\b)(?=.*\bremove\b)"),
    ("keyerror_missing_key", {"T09"}, r"KeyError"),
    ("unhashable_key", {"T08", "T09"}, r"unhashable|\bhashable\b"),
    ("iterate_gives_keys", {"T09"}, r"(travers|loop|iterat)\w*[^.\n]{0,30}\b(the )?keys\b|\bfor key in\b"),
    ("print_instead_of_return", {"T10"}, r"void function|fruitful function|displays? .{0,40}but (does not|doesn.t) return"),
    ("missing_return_none", {"T10"}, r"return(s)? `?None`?|special value `?None"),
    ("local_scope_access", {"T10"}, r"\blocal\b[^.\n]{0,60}(variable|scope|symbol table)"),
    ("mutable_default_argument", {"T10"}, r"default value is evaluated only once|mutable object such as a list"),
    ("argument_count_mismatch", {"T10"}, r"missing \d+ required positional|takes \d+ positional arguments? but"),
    ("bare_except", {"T12"}, r"except:\s*$|omit the exception name"),
    ("else_finally_confusion", {"T12"}, r"(?s)(?=.*\bfinally\b)(?=.*\belse\b)"),
    ("syntax_vs_runtime_error", {"T01", "T12"}, r"(?s)(?=.*[Ss]yntax [Ee]rror)(?=.*(exception|runtime error))"),
    ("wrong_exception_type", {"T12"}, r"specific exceptions?|which exceptions?"),
    ("forgot_to_close", {"T13"}, r"\.close\(\)|close the file"),
    ("write_mode_overwrites", {"T13"}, r"('w'|write mode)[^.\n]{0,120}(clear|eras|overwrit|old data|replac)"),
    ("newline_not_stripped", {"T13"}, r"rstrip|newline"),
    ("wrong_file_path", {"T13", "T12"}, r"FileNotFoundError|No such file"),
    ("read_returns_string", {"T13"}, r"read\(\)[^.\n]{0,100}(one|single|entire|whole)[^.\n]{0,30}string"),
]
TAG_RULES = [(tag, topics, re.compile(p, re.M)) for tag, topics, p in TAG_RULES]


def add_labels(r):
    """Difficulty, misconception tags and runnability for one record."""
    if r.get("difficulty") is None:
        levels = [SUBCONCEPT_DIFFICULTY[s] for s in r["subconcept_ids"] if s in SUBCONCEPT_DIFFICULTY]
        r["difficulty"] = max(levels) if levels else TOPIC_DIFFICULTY[r["topic_id"]]
        r["difficulty_source"] = "curriculum_default"
    else:
        r.setdefault("difficulty_source", "authored")
    if r["source"]["id"] == "original":
        r["misconception_tag_source"] = "authored"
    else:
        body = split_header(r["text"])[1]
        r["misconception_tags"] = [tag for tag, topics, rx in TAG_RULES if r["topic_id"] in topics and rx.search(body)]
        r["misconception_tag_source"] = "pattern" if r["misconception_tags"] else "none"
    r["runnable_as_shown"] = not r["requires_files"]


# =============================================================================================
# Links between chunks
# =============================================================================================
BUILTIN_NAMES = set(dir(builtins)) | {"__name__", "_"}
DANGLING_START = re.compile(
    r"^(it|this|that|these|those|here|in the (function|previous)|the (above|previous|same))\b[^.]{0,60}"
    r"\b(function|code|example|program|version|loop|call|like this)", re.I)


def defined_and_used_names(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(), set()
    defined, used = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (defined if isinstance(node.ctx, (ast.Store, ast.Del)) else used).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    return defined, used - BUILTIN_NAMES


def chunk_code(r):
    """All code of a chunk as plain Python (doctest prompts removed, outputs skipped)."""
    parts = []
    for m in FENCE.finditer(r["text"]):
        if m.group(1) == "text":
            continue
        code = m.group(2)
        if code.lstrip().startswith((">>>", "...")):
            code = "\n".join(line[4:] for line in code.splitlines() if line.startswith((">>> ", "... ")))
        parts.append(code)
    return "\n".join(parts)


def add_links(rows):
    """prev/next reading order per page, and the earlier chunks that define names a chunk's code uses."""
    pages = collections.OrderedDict()
    for r in rows:
        pages.setdefault((r["source"]["id"], r["source"].get("url") or r["topic_id"]), []).append(r)
    for page in pages.values():
        names = [defined_and_used_names(chunk_code(r)) for r in page]
        for i, r in enumerate(page):
            r["prev_chunk_id"] = page[i - 1]["chunk_id"] if i > 0 else None
            r["next_chunk_id"] = page[i + 1]["chunk_id"] if i + 1 < len(page) else None
            context, missing = [], set()
            defined_here, used = names[i]
            for name in sorted(used - defined_here):
                for j in range(i - 1, -1, -1):
                    if name in names[j][0]:
                        if page[j]["chunk_id"] not in context:
                            context.append(page[j]["chunk_id"])
                        break
                else:
                    missing.add(name)
            if DANGLING_START.match(split_header(r["text"])[1].strip()) and i > 0 \
                    and page[i - 1]["chunk_id"] not in context:
                context.append(page[i - 1]["chunk_id"])
            r["context_chunk_ids"] = context[:3]
            r["needs_context"] = bool(context)
            r["undefined_names"] = sorted(missing)


def add_duplicate_groups(rows, threshold=0.8):
    """Chunks whose TF-IDF cosine similarity exceeds `threshold` share a `duplicate_group` id."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(TfidfVectorizer(stop_words="english")
                                   .fit_transform([split_header(r["text"])[1] for r in rows]))
    parent = list(range(len(rows)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if similarity[i, j] > threshold:
                parent[root(i)] = root(j)
    groups = collections.defaultdict(list)
    for i in range(len(rows)):
        groups[root(i)].append(i)
    n = 0
    for members in groups.values():
        group_id = f"dup_{n:03d}" if len(members) > 1 else None
        n += len(members) > 1
        for i in members:
            rows[i]["duplicate_group"] = group_id
    return n


def add_teaching_order(rows, topics):
    """`uses_later_topics`: topics a chunk's code needs that the concept map teaches after its own topic."""
    by_id = {t["topic_id"]: t for t in topics}

    def prerequisites(topic, seen=None):
        seen = set() if seen is None else seen
        for p in by_id[topic]["prerequisites"]:
            if p not in seen:
                seen.add(p)
                prerequisites(p, seen)
        return seen
    allowed = {t: prerequisites(t) | {t} for t in by_id}
    for r in rows:
        code = "\n".join(m.group(2) for m in FENCE.finditer(r["text"]) if m.group(1) != "text")
        used = {topic for topic, rx in CONSTRUCT.items() if re.search(rx, code)}
        r["uses_topics"] = sorted(used | {r["topic_id"]})
        r["uses_later_topics"] = sorted(t for t in used if t not in allowed[r["topic_id"]])


# =============================================================================================
# Main
# =============================================================================================
def write_report(path, source, n_in, rows, dropped, added, edit_counts, label_stats, n_groups, topics):
    coverage = collections.Counter(s for r in rows for s in r["subconcept_ids"])
    subconcepts = [s["id"] for t in topics for s in t["subconcepts"]]
    tags = [m["id"] for t in topics for m in t["common_misconceptions"]]
    tag_coverage = collections.Counter(t for r in rows for t in r["misconception_tags"])
    lines = [
        "# Finalize report", "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from `{source}`.", "",
        f"- chunks in: **{n_in}**, dropped: **{len(dropped)}**, added (authored): **{len(added)}**, out: **{len(rows)}**",
        f"- subconcepts covered: **{sum(1 for s in subconcepts if coverage[s])}/{len(subconcepts)}** "
        f"(min chunks per subconcept: {min(coverage[s] for s in subconcepts)})",
        f"- misconception tags with at least one chunk: **{len(tag_coverage)}/{len(tags)}**",
        f"- examples needing a data file (`runnable_as_shown=false`): **{sum(bool(r['requires_files']) for r in rows)}**",
        f"- chunks to show with context (`context_chunk_ids`): **{sum(r['needs_context'] for r in rows)}**",
        f"- chunks using a topic taught later (`uses_later_topics`): **{sum(bool(r['uses_later_topics']) for r in rows)}**",
        f"- near-duplicate groups (`duplicate_group`): **{n_groups}**",
        f"- label corrections: {dict(label_stats)}", "",
        "## Dropped chunks", "", *[f"- `{cid}`: {why}" for cid, why in dropped], "",
        "## Edits applied (number of chunks)", "", *[f"- {k}: {v}" for k, v in edit_counts.most_common()],
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[4:12]))


def main():
    here = Path(__file__).resolve().parent
    default_topics = here / "topics.json" if (here / "topics.json").exists() else here.parent / "topics.json"
    ap = argparse.ArgumentParser(description="Clean, label and link the scraped corpus")
    ap.add_argument("--in", dest="inp", default="corpus_out/corpus_chunks.jsonl")
    ap.add_argument("--topics", default=str(default_topics))
    ap.add_argument("--out", default="corpus_final.jsonl")
    ap.add_argument("--report", default="finalize_report.md")
    args = ap.parse_args()

    topics = json.loads(Path(args.topics).read_text())
    rows = [json.loads(line) for line in open(args.inp, encoding="utf-8") if line.strip()]
    n_in = len(rows)

    merged = move_orphaned_outputs(rows)
    rows = [r for r in rows if r["chunk_id"] not in merged]
    dropped = [(cid, "merged into previous chunk (was an orphaned output)") for cid in sorted(merged)]

    kept, edit_counts = [], collections.Counter()
    for r in rows:
        cleaned, reason = clean_chunk(r)
        if cleaned is None:
            dropped.append((r["chunk_id"], reason))
            continue
        edit_counts.update(cleaned["edits"])
        kept.extend(split_oversized(cleaned))

    added = build_authored(topics, kept)
    rows = kept + added

    topic_of = {s["id"]: t["topic_id"] for t in topics for s in t["subconcepts"]}
    label_stats = correct_labels(rows, topic_of)
    for r in rows:
        r["word_count"] = words(r["text"])
        r["has_code"] = any(m.group(1) != "text" and m.group(2).split() for m in FENCE.finditer(r["text"]))
        r["content_hash"] = content_hash(r["text"])
        add_labels(r)
    add_links(rows)
    n_groups = add_duplicate_groups(rows)
    add_teaching_order(rows, topics)

    ids = {r["chunk_id"] for r in rows}
    assert len(ids) == len(rows), "duplicate chunk ids"
    assert all(c is None or c in ids for r in rows for c in [r["prev_chunk_id"], r["next_chunk_id"], *r["context_chunk_ids"]])

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_report(args.report, args.inp, n_in, rows, dropped, added, edit_counts, label_stats, n_groups, topics)
    print(f"\nwrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()