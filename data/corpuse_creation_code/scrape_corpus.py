#!/usr/bin/env python3
"""Scrape openly licensed Python tutorials into labelled teaching chunks.

Pipeline
  fetch    only the pages listed in sources_manifest.json (no crawling); polite, cached, robots.txt-aware
  extract  the main content: headings, prose, lists and code; navigation, ads and skipped sections removed
  chunk    ~220-word chunks that keep an explanation together with the code that follows it
  label    each chunk gets a topic and subconcepts from topics.json (heading rules first, then keywords)
  merge    the verified hand-written gap chunks from original_chunks.jsonl (see build_original_chunks.py)

Output (in --out, default ./corpus_out)
  corpus_chunks.jsonl   accepted chunks (input for finalize_corpus.py)
  review_queue.jsonl    chunks whose topic could not be assigned confidently (excluded)
  sources_log.json      one record per page: status, title, headings seen, number of chunks
  ATTRIBUTION.md        licences and credits of the sources used
  raw/                  cached HTML, so re-runs with --offline do not touch the network

Usage:
    python3 scrape_corpus.py --contact you@example.com           # first run (downloads pages)
    python3 scrape_corpus.py --contact you@example.com --limit 1 # smoke test: one page per source
    python3 scrape_corpus.py --offline                           # re-process cached pages only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import textwrap
import time
import urllib.robotparser as robotparser
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment, NavigableString, Tag

BOT_NAME = "PythonTutorCorpusBot"
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
NON_CONTENT_TAGS = ["script", "style", "noscript", "nav", "footer", "aside", "form", "iframe", "svg", "button"]
INLINE_CODE_TAGS = {"code", "tt", "kbd", "samp"}
NUMBER_PREFIX = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[A-Z]\.)\s+")   # "4.2. " in "4.2. for Statements"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def words(s: str) -> int:
    return len(s.split())


# =============================================================================================
# Fetching
# =============================================================================================
MOJIBAKE = re.compile("[\u00c2\u00c3\u00e2][\u0080-\u00bf]")


def decode_html(content: bytes, declared: str | None) -> str:
    """Decode a page as UTF-8 first. `requests` assumes Latin-1 when a server sends no charset, which
    turns ’ into â€™; text that was decoded that way is repaired as well."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        meta = re.search(rb"<meta[^>]+charset=[\"']?([A-Za-z0-9_\-]+)", content[:4096], re.I)
        text = None
        for encoding in [meta.group(1).decode() if meta else None, declared, "cp1252"]:
            if not encoding:
                continue
            try:
                text = content.decode(encoding)
                break
            except (LookupError, UnicodeDecodeError):
                continue
        if text is None:
            text = content.decode("utf-8", errors="replace")
    if MOJIBAKE.search(text):
        try:
            text = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return text


class Fetcher:
    """Downloads pages once, then serves them from the cache. Waits between requests to the same host,
    identifies itself with a contact address, and obeys robots.txt."""

    def __init__(self, cache_dir: Path, delay: float, contact: str, offline: bool):
        self.cache_dir, self.delay, self.offline = cache_dir, delay, offline
        self.session = requests.Session()
        self.session.headers["User-Agent"] = f"{BOT_NAME}/1.0 (academic project; contact: {contact})"
        self.robots: dict = {}
        self.last_request: dict = {}

    def cache_path(self, source_id: str, url: str) -> Path:
        tail = re.sub(r"[^A-Za-z0-9]+", "_", urlparse(url).path.strip("/").split("/")[-1] or "index")[:60]
        return self.cache_dir / source_id / f"{tail}_{hashlib.sha1(url.encode()).hexdigest()[:8]}.html"

    def allowed(self, url: str) -> bool:
        """robots.txt check. A missing robots.txt allows everything; an unreachable one allows nothing."""
        parts = urlparse(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self.robots:
            try:
                r = self.session.get(base + "/robots.txt", timeout=15)
                if r.status_code == 200:
                    parser = robotparser.RobotFileParser()
                    parser.parse(r.text.splitlines())
                    self.robots[base] = parser
                else:
                    self.robots[base] = None if 400 <= r.status_code < 500 else False
            except requests.RequestException:
                self.robots[base] = False
        rules = self.robots[base]
        return True if rules is None else bool(rules) and rules.can_fetch(BOT_NAME, url)

    def get(self, source_id: str, url: str):
        """Return (html or None, metadata)."""
        path = self.cache_path(source_id, url)
        meta_path = path.with_suffix(".meta.json")
        if path.exists():
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            return decode_html(path.read_bytes(), None), {"status": 200, "cached": True,
                                                          "final_url": meta.get("final_url", url)}
        if self.offline:
            return None, {"status": None, "error": "not in cache (offline mode)"}
        if not self.allowed(url):
            return None, {"status": None, "error": "disallowed by robots.txt (or robots.txt unreachable)"}
        error = "unknown error"
        for attempt in range(3):
            host = urlparse(url).netloc
            wait = self.delay - (time.time() - self.last_request.get(host, 0))
            if wait > 0:
                time.sleep(wait)
            self.last_request[host] = time.time()
            try:
                r = self.session.get(url, timeout=30)
            except requests.RequestException as e:
                error = str(e)
                time.sleep(2 ** (attempt + 1))
                continue
            if r.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(r.content)
                meta_path.write_text(json.dumps({"final_url": r.url, "fetched_at": now()}))
                return decode_html(r.content, r.encoding), {"status": 200, "cached": False, "final_url": r.url}
            if r.status_code in (429, 500, 502, 503, 504):  # temporary problems: retry with back-off
                error = f"HTTP {r.status_code}"
                time.sleep(2 ** (attempt + 1))
                continue
            return None, {"status": r.status_code, "error": f"HTTP {r.status_code}"}
        return None, {"status": None, "error": error}


# =============================================================================================
# Extraction: HTML -> blocks of text and code, each with its heading path
# =============================================================================================
def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def inline(node) -> str:
    """Render inline HTML as text; inline code becomes `code`."""
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        return str(node)
    if node.name in {"script", "style", "img", "svg"}:
        return ""
    if node.name == "br":
        return " "
    if node.name in INLINE_CODE_TAGS:
        code = collapse_ws(node.get_text())
        return f"`{code}`" if code else ""
    return "".join(inline(c) for c in node.children)


def inline_children(tag, skip=()) -> str:
    return collapse_ws("".join(inline(c) for c in tag.children if not (isinstance(c, Tag) and c.name in skip)))


def heading_text(tag) -> str:
    return re.sub(r"\s*[¶#§]\s*$", "", inline_children(tag)).strip()   # drop permalink symbols


def code_text(node) -> str:
    for br in node.find_all("br"):
        br.replace_with("\n")
    lines = [line.rstrip() for line in node.get_text().replace("\xa0", " ").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return textwrap.dedent("\n".join(lines))


def list_markdown(node, depth=0) -> str:
    lines = []
    for i, item in enumerate(node.find_all("li", recursive=False)):
        bullet = "- " if node.name == "ul" else f"{i + 1}. "
        lines.append("  " * depth + bullet + inline_children(item, skip=("ul", "ol", "pre")))
        lines += [list_markdown(c, depth + 1) for c in item.children if isinstance(c, Tag) and c.name in ("ul", "ol")]
    return "\n".join(lines)


def definition_list_markdown(node) -> str:
    return "\n".join(f"{inline_children(c)}:" if c.name == "dt" else f"  {inline_children(c)}"
                     for c in node.find_all(["dt", "dd"], recursive=False))


def table_markdown(node) -> str:
    rows = [" | ".join(inline_children(cell) for cell in tr.find_all(["th", "td"])) for tr in node.find_all("tr")]
    return "\n".join(r for r in rows if r.strip(" |"))


def restore_protected_emails(soup):
    """py4e.com hides e-mail addresses from spam bots (Cloudflare): the address is stored XOR-encoded
    in hex, first byte = key, and the page shows "[email protected]". Decode it back."""
    def decode(hexstr):
        try:
            key = int(hexstr[:2], 16)
            return "".join(chr(int(hexstr[i:i + 2], 16) ^ key) for i in range(2, len(hexstr), 2))
        except (ValueError, TypeError):
            return None
    for el in soup.select("[data-cfemail]"):
        if email := decode(el.get("data-cfemail")):
            el.replace_with(email)
    for a in soup.select('a[href*="/cdn-cgi/l/email-protection#"]'):
        if email := decode(a["href"].split("#", 1)[1]):
            a.replace_with(email)


@dataclass
class Block:
    kind: str    # "text", "code" or "output"
    text: str
    path: tuple  # heading breadcrumb, e.g. ("4. More Control Flow Tools", "4.2. for Statements")


class BlockCollector:
    """Walks the page, tracks the current heading path and skips sections whose heading matches a skip rule."""

    def __init__(self, skip_re, ignore_re, drop_re):
        self.skip_re, self.ignore_re, self.drop_re = skip_re, ignore_re, drop_re
        self.stack, self.blocks, self.headings = [], [], []
        self.skip_level = None   # heading level of the section being skipped, if any

    def heading(self, level, text):
        if not text:
            return
        plain = NUMBER_PREFIX.sub("", text).replace("`", "")
        if self.ignore_re and (self.ignore_re.search(plain) or self.ignore_re.search(text)):
            return
        if self.skip_level is not None and level <= self.skip_level:
            self.skip_level = None
        if self.skip_level is None and self.skip_re and (self.skip_re.search(plain) or self.skip_re.search(text)):
            self.skip_level = level
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))
        self.headings.append((level, text, self.skip_level is not None))

    @property
    def path(self):
        return tuple(text for _, text in self.stack)

    def add(self, kind, text):
        if self.skip_level is not None or not text or not text.strip():
            return
        if kind == "text" and self.drop_re and self.drop_re.search(text):
            return
        self.blocks.append(Block(kind, text.strip() if kind == "text" else text, self.path))


def any_of(patterns):
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.I) if patterns else None


def extract(html: str, source: dict, extra_skip=()):
    """Return (page title, blocks, headings seen) for one page."""
    soup = BeautifulSoup(html, "html.parser")
    restore_protected_emails(soup)
    title = collapse_ws(soup.title.get_text()) if soup.title else ""
    for selector in source.get("remove_selectors", []):
        for el in soup.select(selector):
            if not el.decomposed:            # may already be gone with a removed parent
                el.decompose()
    root = next((el for sel in source.get("content_selectors", []) if (el := soup.select_one(sel)) is not None),
                soup.body or soup)
    for el in root.find_all(NON_CONTENT_TAGS):
        if not el.decomposed:
            el.decompose()
    collector = BlockCollector(any_of(list(source.get("skip_headings", [])) + list(extra_skip)),
                               any_of(source.get("ignore_headings", [])), any_of(source.get("drop_text", [])))
    walk(root, collector)
    return title, collector.blocks, collector.headings


def walk(node, out: BlockCollector):
    for child in node.children:
        if not isinstance(child, Tag):
            continue
        classes = child.get("class", []) or []
        if child.name in HEADINGS:
            out.heading(HEADINGS[child.name], heading_text(child))
        elif child.name == "p":
            out.add("text", inline_children(child))
        elif child.name in ("ul", "ol"):
            out.add("text", list_markdown(child))
        elif child.name == "pre":
            out.add("code", code_text(child))
        elif child.name == "dl":
            out.add("text", definition_list_markdown(child))
        elif child.name == "table":
            out.add("text", table_markdown(child))
        elif child.name == "div" and "cell_output" in classes:            # notebook cell output
            out.add("output", code_text(child))
        elif child.name == "div" and any(c.startswith("highlight") for c in classes):  # highlighted code
            out.add("code", code_text(child))
        elif child.name == "div" and "admonition" in classes:             # Sphinx "Note:" boxes
            title = child.find(class_="admonition-title")
            label = collapse_ws(title.get_text()) if title else "Note"
            if title:
                title.decompose()
            body = " ".join(inline_children(p) for p in child.find_all("p"))
            out.add("text", f"{label}: {body}" if body else "")
        else:
            walk(child, out)


# =============================================================================================
# Chunking
# =============================================================================================
def split_sentences(text, target):
    groups, current = [], []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if current and words(" ".join(current)) + words(sentence) > target:
            groups.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        groups.append(" ".join(current))
    return groups


def group_pieces(pieces, target, max_words):
    """Group (kind, text) pieces into chunks of about `target` words. Code is never split, and a short
    paragraph that introduces a code block moves with it into the next chunk."""
    expanded = []
    for kind, text in pieces:
        if kind == "text" and words(text) > max_words and "\n" not in text:
            expanded += [("text", s) for s in split_sentences(text, target)]
        else:
            expanded.append((kind, text))
    groups, current, size = [], [], 0
    for kind, text in expanded:
        if current and size + words(text) > target:
            if kind != "text" and current[-1][0] == "text" and len(current) > 1:
                intro = current.pop()
                groups.append(current)
                current, size = [intro], words(intro[1])
            else:
                groups.append(current)
                current, size = [], 0
        current.append((kind, text))
        size += words(text)
    if current:
        groups.append(current)
    return groups


def render(group):
    parts = []
    for kind, text in group:
        if kind == "code":
            parts.append(f"```python\n{text}\n```")
        elif kind == "output":
            parts.append(f"Output:\n```text\n{text}\n```")
        else:
            parts.append(text)
    return "\n\n".join(parts)


def make_chunks(blocks, target, max_words, min_words):
    """Turn a page's blocks into chunks; sections shorter than `min_words` are merged into the next one."""
    sections = []
    for b in blocks:
        if sections and sections[-1][0] == b.path:
            sections[-1][1].append(b)
        else:
            sections.append((b.path, [b]))
    if any(path for path, _ in sections):            # text before the first heading is page boilerplate
        sections = [s for s in sections if s[0]]
    has_subsections = any(len(path) > 1 for path, _ in sections)

    chunks, carry = [], []
    for path, section_blocks in sections:
        pieces = [(b.kind, b.text) for b in section_blocks]
        if sum(words(t) for _, t in pieces) < min_words and all(k == "text" for k, _ in pieces):
            if len(path) > 1:                        # tiny chapter-level intros are dropped
                carry.append((path, pieces))
            continue
        for i, group in enumerate(group_pieces(pieces, target, max_words)):
            if i == 0 and carry:
                prefix = []
                for carried_path, carried in carry:
                    prefix.append(("text", f"**{carried_path[-1] if carried_path else ''}**"))
                    prefix += carried
                group, carry = prefix + group, []
            chunks.append({"path": path, "group": group})
    if carry and chunks:                             # a tiny last section joins the last chunk
        for carried_path, carried in carry:
            chunks[-1]["group"].append(("text", f"**{carried_path[-1] if carried_path else ''}**"))
            chunks[-1]["group"] += carried

    result = []
    for c in chunks:
        code_words = sum(words(t) for k, t in c["group"] if k != "text")
        total_words = sum(words(t) for _, t in c["group"])
        if total_words < min_words and code_words == 0:
            continue
        breadcrumb = " > ".join(c["path"][-2:]) if c["path"] else ""
        text = (f"**{breadcrumb}**\n\n{render(c['group'])}" if breadcrumb else render(c["group"])).strip()
        result.append({"path": list(c["path"]), "intro": has_subsections and len(c["path"]) <= 1, "text": text,
                       "has_code": code_words > 0, "code_ratio": code_words / total_words if total_words else 0.0,
                       "words": words(text)})
    return result


def chunk_type(chunk) -> str:
    heading = chunk["path"][-1] if chunk["path"] else ""
    if re.match(r"^\s*(?:\d+(?:\.\d+)*\.?\s*)?Debugging\s*$", heading, re.I):
        return "debugging"
    return "example" if chunk["code_ratio"] >= 0.6 else "explanation"


# =============================================================================================
# Labelling: chunk -> topic and subconcepts
# =============================================================================================
STOP_WORDS = set("a an and the of in on for to with using use vs or by as at from into is are be it its that "
                 "this these those basic core common general more about over".split())
# Python keywords count only when they appear in code or a heading, not in ordinary prose ("for example").
SYNTAX_WORDS = {"for", "if", "elif", "else", "while", "try", "except", "finally", "def", "return", "with",
                "break", "continue", "raise", "lambda", "import", "class", "pass"}
WORD = re.compile(r"[a-z_][a-z0-9_]*")
CODE_SPAN = re.compile(r"```.*?```|`[^`\n]+`", re.S)


def stem(w: str) -> str:
    """Very small plural stemmer: dictionaries -> dictionary, loops -> loop."""
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


SYNTAX_STEMS = {stem(w) for w in SYNTAX_WORDS}

# Keywords in addition to the words of each topic / subconcept name in topics.json.
TOPIC_KEYWORDS = {
    "T01": ["python", "interpreter", "script", "shell", "indentation", "syntax", "comment", "traceback", "prompt",
            "error", "debug", "bug", "runtime", "semantic", "compiler"],
    "T02": ["variable", "assignment", "int", "float", "integer", "string", "boolean", "type", "conversion", "none"],
    "T03": ["print", "input", "output", "format", "formatted", "display"],
    "T04": ["operator", "arithmetic", "modulo", "floor", "comparison", "logical", "precedence", "augmented"],
    "T05": ["if", "elif", "else", "conditional", "condition", "branch", "truthy"],
    "T06": ["for", "while", "loop", "range", "iterate", "iteration", "break", "continue", "enumerate"],
    "T07": ["string", "slice", "slicing", "index", "immutable", "upper", "lower", "strip", "split", "join", "replace",
            "find", "character"],
    "T08": ["list", "tuple", "set", "append", "extend", "insert", "remove", "pop", "sort", "sorted", "sequence",
            "union", "intersection"],
    "T09": ["dictionary", "dict", "key", "value", "item", "get", "mapping", "hash"],
    "T10": ["def", "function", "return", "argument", "parameter", "scope", "local", "global", "default", "keyword",
            "docstring"],
    "T11": ["comprehension"],
    "T12": ["exception", "try", "except", "finally", "raise", "error", "handler"],
    "T13": ["file", "open", "read", "write", "close", "with", "readline", "path"],
}
SUBCONCEPT_KEYWORDS = {
    "T01.5": ["traceback", "error", "syntaxerror", "debug", "bug", "runtime", "semantic"],
    "T03.2": ["f-string", "fstring", "format"],
    "T04.1": ["modulo", "floor", "exponent"],
    "T06.1": ["range", "for"],
    "T06.3": ["while"],
    "T06.4": ["break", "continue"],
    "T07.2": ["slice", "slicing"],
    "T07.3": ["immutable", "immutability"],
    "T08.2": ["append", "extend", "insert", "remove", "pop", "sort"],
    "T08.3": ["tuple", "immutable", "immutability"],
    "T08.4": ["set", "union", "intersection"],
    "T09.3": ["get", "in", "keyerror"],
    "T09.4": ["items", "keys", "values"],
    "T10.3": ["return"],
    "T10.5": ["global", "local", "scope"],
    "T11.1": ["comprehension"],
    "T12.2": ["try", "except"],
    "T12.4": ["else", "finally"],
    "T12.5": ["raise"],
    "T13.2": ["with"],
    "T13.3": ["read", "readline", "readlines"],
    "T13.4": ["write", "append"],
}
# Code patterns that identify a subconcept precisely: (pattern, bonus). Used only to choose subconcepts.
SUBCONCEPT_PATTERNS = {
    "T04.4": [(r"(?<![=!<>])[+\-*/%]=(?!=)", 4)],
    "T06.5": [(r"(?m)^([ \t]*)(?:for|while)\b[^\n]*:[ \t]*\n(?:\1[ \t]+[^\n]*\n)*?\1[ \t]+(?:for|while)\b", 5)],
    "T09.2": [(r"\b\w+\[[^\]\n]+\]\s*=(?!=)", 3), (r"\bdel\s+\w+\[", 3), (r"\.(?:pop|update|setdefault)\(", 2)],
    "T09.5": [(r"hashab", 4), (r"keys?\b[^.\n]{0,40}\bimmutable|immutable[^.\n]{0,40}\bkeys?\b", 4)],
    "T11.1": [(r"\[[^\[\]\n]*\bfor\b[^\[\]\n]*\bin\b[^\[\]\n]*\]", 4)],
    "T11.2": [(r"\[[^\[\]\n]*\bfor\b[^\[\]\n]*\bif\b[^\[\]\n]*\]", 3)],
    "T11.3": [(r"\[[^\[\]\n]*\bif\b[^\[\]\n]*\belse\b[^\[\]\n]*\bfor\b[^\[\]\n]*\]", 4)],
    "T11.4": [(r"\[[^\[\]\n]*\bfor\b[^\[\]\n]*\bfor\b[^\[\]\n]*\]", 4)],
    "T12.6": [(r"except\s+ValueError", 4)],
    "T13.2": [(r"\bwith\s+open\s*\(", 5)],
}
SUBCONCEPT_PATTERNS = {k: [(re.compile(p), bonus) for p, bonus in v] for k, v in SUBCONCEPT_PATTERNS.items()}


def name_tokens(name):
    return [stem(w) for w in WORD.findall(name.lower()) if w not in STOP_WORDS and len(w) > 2]


class Labeller:
    """Scores a chunk against each candidate topic and its subconcepts using keyword counts.
    Heading words count 3x; counts are capped at 3; a keyword shared by several subconcepts of a
    topic is down-weighted, because it does not tell them apart."""

    def __init__(self, topics):
        self.topics = {t["topic_id"]: t for t in topics}
        self.topic_kw, self.sub_kw, self.sub_weight = {}, {}, {}
        for t in topics:
            tid = t["topic_id"]
            self.topic_kw[tid] = set(name_tokens(t["name"])) | {stem(w) for w in TOPIC_KEYWORDS.get(tid, [])}
            for s in t["subconcepts"]:
                self.sub_kw[s["id"]] = set(name_tokens(s["name"])) | {stem(w) for w in SUBCONCEPT_KEYWORDS.get(s["id"], [])}
            shared = Counter(k for s in t["subconcepts"] for k in self.sub_kw[s["id"]])
            for s in t["subconcepts"]:
                self.sub_weight[s["id"]] = {k: 1.0 / shared[k] for k in self.sub_kw[s["id"]]}

    @staticmethod
    def score(keywords, heading, prose, code, weights=None):
        total = 0.0
        for k in keywords:
            w = weights.get(k, 1.0) if weights else 1.0
            body = code[k] if k in SYNTAX_STEMS else prose[k] + code[k]
            total += w * (3 * min(heading[k], 1) + min(body, 3))
        return total

    def label(self, heading, body, candidates):
        count = lambda s: Counter(stem(w) for w in WORD.findall(s.lower()))
        h, prose, code = count(heading), count(CODE_SPAN.sub(" ", body)), count(" ".join(CODE_SPAN.findall(body)))
        candidates = [c for c in (candidates or []) if c in self.topic_kw] or list(self.topic_kw)

        sub_scores, topic_scores = {}, {}
        for tid in candidates:
            sub_scores[tid] = {s["id"]: self.score(self.sub_kw[s["id"]], h, prose, code, self.sub_weight[s["id"]])
                               for s in self.topics[tid]["subconcepts"]}
            topic_scores[tid] = self.score(self.topic_kw[tid], h, prose, code) + 0.5 * sum(sub_scores[tid].values())
        ranked = sorted(topic_scores.items(), key=lambda kv: kv[1], reverse=True)
        best_topic, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0

        if len(candidates) == 1:   # the page belongs to one topic: confidence = is there enough evidence?
            confidence = 0.9 if best >= 4 else 0.6 if best >= 2 else 0.3 if best >= 1 else 0.0
            method = "page_default+keywords"
        else:                      # several candidate topics: margin over the runner-up, scaled by evidence
            confidence = round(best / (best + second + 1e-9) * min(1.0, best / 6.0), 3) if best > 0 else 0.0
            method = "keywords"

        scores = dict(sub_scores[best_topic])
        for sid in scores:
            for pattern, bonus in SUBCONCEPT_PATTERNS.get(sid, []):
                if pattern.search(body):
                    scores[sid] += bonus
        top = max(scores.values()) if scores else 0
        chosen = [sid for sid, v in sorted(scores.items(), key=lambda kv: -kv[1]) if v >= max(2, 0.6 * top)][:2]
        return {"topic_id": best_topic, "subconcept_ids": chosen, "confidence": confidence, "method": method,
                "candidate_scores": {t: round(v, 1) for t, v in ranked}}


def label_chunk(labeller, chunk, page):
    """Explicit heading rules from the manifest win over keyword scoring."""
    heading = " ".join(chunk["path"][-2:])
    last = NUMBER_PREFIX.sub("", chunk["path"][-1]).replace("`", "") if chunk["path"] else ""
    rule = next((r for r in page.get("heading_map", []) if re.search(r["match"], last, re.I)), None)
    if rule is None:
        return labeller.label(heading, chunk["text"], page.get("topics")), None
    if rule.get("subconcepts") == "auto":   # rule fixes the topic, keywords choose the subconcepts
        auto = labeller.label(heading, chunk["text"], [rule["topic"]])
        return {"topic_id": rule["topic"], "subconcept_ids": auto["subconcept_ids"], "confidence": 1.0,
                "method": "manual_topic+keywords", "candidate_scores": auto["candidate_scores"]}, rule
    return {"topic_id": rule["topic"], "subconcept_ids": list(rule.get("subconcepts", [])), "confidence": 1.0,
            "method": "manual_rule", "candidate_scores": {}}, rule


def validate_manifest(manifest, topics):
    """Fail early on a typo in a heading rule instead of silently mislabelling chunks."""
    topic_ids = {t["topic_id"] for t in topics}
    sub_ids = {s["id"] for t in topics for s in t["subconcepts"]}
    for source in manifest["sources"]:
        rule_lists = [p.get("heading_map", []) for p in source.get("pages", [])]
        rule_lists += list(source.get("discover", {}).get("heading_map_by_group", {}).values())
        for rule in (r for rules in rule_lists for r in rules):
            try:
                re.compile(rule["match"])
            except re.error as e:
                sys.exit(f"manifest error ({source['id']}): bad regex {rule['match']!r}: {e}")
            if rule["topic"] not in topic_ids:
                sys.exit(f"manifest error ({source['id']}): unknown topic {rule['topic']}")
            subs = rule.get("subconcepts", [])
            for sid in [] if subs == "auto" else subs:
                if sid not in sub_ids:
                    sys.exit(f"manifest error ({source['id']}): unknown subconcept {sid}")


# =============================================================================================
# Main
# =============================================================================================
def discover_pages(source, fetcher):
    """Book chapters found on an index page; only chapter groups mapped to topics are kept."""
    cfg = source.get("discover")
    if not cfg:
        return []
    html, meta = fetcher.get(source["id"], cfg["index_url"])
    if html is None:
        print(f"  ! could not fetch index {cfg['index_url']}: {meta.get('error')}")
        return []
    link = re.compile(cfg["link_regex"])
    pages, seen = [], set()
    for a in BeautifulSoup(html, "html.parser").find_all("a", href=True):
        href = a["href"].split("#")[0]
        m = link.search(href)
        if not m:
            continue
        url, group = urljoin(meta.get("final_url", cfg["index_url"]), href), m.group(1)
        topics = cfg.get("topics_by_group", {}).get(group)
        if url in seen or not topics:
            continue
        seen.add(url)
        pages.append({"url": url, "topics": topics,
                      "heading_map": cfg.get("heading_map_by_group", {}).get(group, []),
                      "skip_headings": cfg.get("skip_headings_by_group", {}).get(group, [])})
    return pages


def drop_duplicates(rows, seen):
    """Keep the first chunk of each exact (whitespace- and case-insensitive) text; `seen` is shared."""
    unique = []
    for r in rows:
        if r["content_hash"] not in seen:
            seen.add(r["content_hash"])
            unique.append(r)
    return unique


def content_hash(text):
    return hashlib.sha256(re.sub(r"\s+", " ", text.lower()).strip().encode()).hexdigest()[:16]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    here = Path(__file__).resolve().parent
    default_topics = here / "topics.json" if (here / "topics.json").exists() else here.parent / "topics.json"
    ap = argparse.ArgumentParser(description="Scrape the sources in sources_manifest.json into labelled chunks")
    ap.add_argument("--manifest", default=str(here / "sources_manifest.json"))
    ap.add_argument("--topics", default=str(default_topics))
    ap.add_argument("--out", default="corpus_out")
    ap.add_argument("--contact", default="offline", help="your e-mail or project URL, sent in the User-Agent")
    ap.add_argument("--offline", action="store_true", help="use cached pages only")
    ap.add_argument("--limit", type=int, help="at most this many pages per source (smoke test)")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    topics = json.loads(Path(args.topics).read_text())
    settings = manifest["defaults"]
    validate_manifest(manifest, topics)
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    fetcher = Fetcher(out / "raw", settings["delay_seconds"], args.contact, args.offline)
    labeller = Labeller(topics)

    accepted, review, log, used_sources, n_intros_dropped = [], [], [], {}, 0
    for source in manifest["sources"]:
        if not source.get("enabled", True):
            continue
        pages = source.get("pages", []) + discover_pages(source, fetcher)
        pages = pages[:args.limit] if args.limit else pages
        print(f"[{source['id']}] {len(pages)} pages")
        for page in pages:
            html, meta = fetcher.get(source["id"], page["url"])
            entry = {"source_id": source["id"], "url": page["url"], **meta}
            log.append(entry)
            if html is None:
                print(f"  x {page['url']}  ({meta.get('error')})")
                continue
            title, blocks, headings = extract(html, source, page.get("skip_headings", []))
            chunks = make_chunks(blocks, settings["chunk_target_words"], settings["chunk_max_words"],
                                 settings["min_chunk_words"])
            entry.update({"title": title, "blocks": len(blocks), "chunks": len(chunks),
                          "headings": [f"{'  ' * (lv - 1)}{t}{'   [SKIPPED]' if skipped else ''}"
                                       for lv, t, skipped in headings]})
            used_sources[source["id"]] = source
            print(f"  + {page['url']}  -> {len(blocks)} blocks, {len(chunks)} chunks")

            for chunk in chunks:
                label, rule = label_chunk(labeller, chunk, page)
                # A chapter's opening text is kept only if it has code and clearly belongs to the topic.
                if chunk["intro"] and source.get("drop_intro") and rule is None:
                    evidence = label["candidate_scores"].get(label["topic_id"], 0.0)
                    if evidence < source.get("intro_min_evidence", 10) or not chunk["has_code"]:
                        n_intros_dropped += 1
                        continue
                record = {
                    "chunk_id": None, "topic_id": label["topic_id"], "subconcept_ids": label["subconcept_ids"],
                    "type": chunk_type(chunk), "difficulty": None, "misconception_tags": [], "text": chunk["text"],
                    "source": {"id": source["id"], "name": source["name"], "url": meta.get("final_url", page["url"]),
                               "page_title": title, "section_path": chunk["path"], "license": source["license"],
                               "attribution": source["attribution"], "redistributable": True},
                    "scraped_at": now(), "word_count": chunk["words"], "has_code": chunk["has_code"],
                    "mapping": {"method": label["method"], "confidence": label["confidence"],
                                "candidate_topics": page.get("topics"), "scores": label["candidate_scores"]},
                    "flags": (["no_subconcept"] if not label["subconcept_ids"] else [])
                             + (["low_confidence"] if label["confidence"] < 0.6 else []),
                    "content_hash": content_hash(chunk["text"]),
                    "_short": source.get("short", source["id"]),
                }
                (accepted if label["confidence"] >= settings["min_confidence"] else review).append(record)

    # Hand-written gap chunks (written and verified by build_original_chunks.py).
    original = here / "original_chunks.jsonl"
    if original.exists():
        for line in original.open(encoding="utf-8"):
            if line.strip():
                accepted.append({**json.loads(line), "_short": "orig"})
        used_sources["original"] = {"name": "Original teaching material written for this project",
                                    "attribution": "Written for the Personalized Python Tutor project",
                                    "license": "Project-original (team to choose a licence)"}

    # Exact duplicates, then ids like T06_tp3_004.
    seen = set()
    accepted, review = drop_duplicates(accepted, seen), drop_duplicates(review, seen)
    counter = Counter()
    for r in accepted:
        counter[(r["topic_id"], r["_short"])] += 1
        r["chunk_id"] = f"{r['topic_id']}_{r['_short']}_{counter[(r['topic_id'], r['_short'])]:03d}"
    for i, r in enumerate(review, 1):
        r["chunk_id"] = f"REVIEW_{r['_short']}_{i:04d}"
    for r in accepted + review:
        del r["_short"]

    write_jsonl(out / "corpus_chunks.jsonl", accepted)
    write_jsonl(out / "review_queue.jsonl", review)
    (out / "sources_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    attribution = ["# Attribution and licences", "", "Keep this file with any copy of the corpus.", ""]
    for s in used_sources.values():
        attribution += [f"## {s['name']}", f"- Credit: {s['attribution']}", f"- Licence: {s['license']}", ""]
    (out / "ATTRIBUTION.md").write_text("\n".join(attribution), encoding="utf-8")

    failed = sum(1 for e in log if e.get("error"))
    print(f"\nChapter intros dropped: {n_intros_dropped} | accepted: {len(accepted)} | "
          f"review queue: {len(review)} | pages failed: {failed}")


if __name__ == "__main__":
    main()