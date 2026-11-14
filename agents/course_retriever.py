"""Course material search over data/common/corpus.jsonl.

Given a student's question, find the most relevant chunks of course text (explanations,
examples, common mistakes) to give the tutor as reference material.

How a search works
    1. Lexical ranking (BM25) over an inverted index built once at start-up. A search only
       touches the chunks that contain a word of the question, so it takes well under a
       millisecond. "tfidf", "dense" and "hybrid" (BM25 + embeddings) are also available.
    2. Optional topic filter (the orchestrator's topic). If the topic has nothing relevant,
       the search falls back to the whole corpus instead of returning nothing.
    3. Personalisation: among the best candidates, chunks whose difficulty (1-3) fits the
       student's mastery of the topic are moved up.
    4. Safety checks using the corpus metadata: a chunk is moved down if its example uses
       topics the student is still weak in (uses_later_topics); only one chunk per
       duplicate_group is returned; chunks that need an earlier chunk to make sense
       (needs_context) carry that background with them.

No library is needed for the default method. Embeddings (dense/hybrid) need numpy and
sentence-transformers and are only loaded when one of those methods is used.
"""

import hashlib
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

DEFAULT_CORPUS_FILEPATH = "data/common/corpus.jsonl"
METHODS = ("tfidf", "bm25", "dense", "hybrid")

# A "word" is a run of lowercase letters, digits or underscores (so "list_append" stays one word).
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
TOPIC_ID_PATTERN = re.compile(r"\bT\d{2}\b", re.IGNORECASE)

# Words that say nothing about the topic of a question. Python keywords and operators such as
# "if", "for", "while", "in", "is", "not", "and", "or", "with" are deliberately NOT in this list.
STOPWORDS = frozenset(
    "a an the of to do does did done i me my mine you your we us our it its this that these "
    "those can could should would will may might must shall please help explain tell show "
    "teach about how what whats why when where which who whom there here be been being am are "
    "was were have has had having some any one more also so but just really very thanks thank "
    "hi hello hey ok okay".split()
)

WEAK_MASTERY = 0.4  # below this a topic counts as "not learned yet" for the later-topics check


def stem(word: str) -> str:
    """Very light stemming so "lists"/"list" and "dictionaries"/"dictionary" match."""
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


class CourseRetriever:
    """Local search engine over the course chunks in corpus.jsonl."""

    def __init__(
        self,
        corpus_filepath: str = DEFAULT_CORPUS_FILEPATH,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_cache_dir: str = ".cache/course_retriever",
        *,
        k1: float = 1.5,
        b: float = 0.75,
        use_stemming: bool = False,
        drop_stopwords: bool = True,
        heading_boost: int = 0,
        difficulty_weight: float = 0.5,
    ):
        self.corpus_filepath = corpus_filepath
        self.documents = self._load_documents(corpus_filepath)
        self.k1, self.b = k1, b
        self.use_stemming = use_stemming
        self.drop_stopwords = drop_stopwords
        self.heading_boost = heading_boost
        self.difficulty_weight = difficulty_weight

        self._by_id = {d.get("chunk_id"): i for i, d in enumerate(self.documents) if d.get("chunk_id")}
        self._topic_index: Dict[str, List[int]] = defaultdict(list)
        for index, document in enumerate(self.documents):
            self._topic_index[document.get("topic_id")].append(index)
        self._build_index()

        # Dense retrieval: the model and the embeddings are loaded lazily.
        self.embedding_model_name = embedding_model_name
        self.embedding_cache_dir = Path(embedding_cache_dir)
        self._embedding_model = None
        self._dense_embeddings = None

    # ------------------------------------------------------------------ loading and indexing
    @staticmethod
    def _load_documents(filepath: str) -> List[Dict[str, Any]]:
        """Read one JSON chunk per line; skip blank lines and chunks without text."""
        documents = []
        with Path(filepath).open("r", encoding="utf-8") as corpus_file:
            for line_number, line in enumerate(corpus_file, start=1):
                if not line.strip():
                    continue
                try:
                    document = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {filepath} at line {line_number}") from error
                if not isinstance(document.get("text"), str) or not document["text"].strip():
                    continue
                documents.append(document)
        return documents

    def _tokenize(self, text: str) -> List[str]:
        """Split text into lowercase words, without filler words, reduced to their stems."""
        tokens = TOKEN_PATTERN.findall(text.lower())
        if self.drop_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        if self.use_stemming:
            tokens = [stem(t) for t in tokens]
        return tokens

    def _build_index(self) -> None:
        """Count the words of every chunk once, so that each search is fast.

        postings: word -> [(chunk position, how often the word occurs in the chunk)]
        """
        self.postings: Dict[str, List[tuple]] = defaultdict(list)
        self.doc_length: List[int] = []
        for position, document in enumerate(self.documents):
            counts = Counter(self._tokenize(document["text"]))
            if self.heading_boost:
                source = document.get("source") or {}
                heading = " ".join(str(p) for p in (source.get("section_path") or []) if isinstance(p, str))
                for token in self._tokenize(heading):
                    counts[token] += self.heading_boost
            self.doc_length.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((position, frequency))

        total = len(self.documents)
        self.average_length = (sum(self.doc_length) / total) if total else 0.0
        # BM25 idf that never goes negative, so tiny corpora behave sensibly too.
        self.idf = {
            term: math.log(1 + (total - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in self.postings.items()
        }
        # TF-IDF cosine (method="tfidf"): chunk vector lengths, computed once.
        self._tfidf_idf = {t: math.log((1 + total) / (1 + len(p))) + 1 for t, p in self.postings.items()}
        norms = [0.0] * total
        for term, plist in self.postings.items():
            weight = self._tfidf_idf[term]
            for position, frequency in plist:
                norms[position] += (frequency * weight) ** 2
        self._tfidf_norm = [math.sqrt(n) for n in norms]

    # ------------------------------------------------------------------ lexical scoring
    def _bm25_scores(self, query_terms: Iterable[str], allowed: Optional[Set[int]]) -> Dict[int, float]:
        scores: Dict[int, float] = defaultdict(float)
        k1, b, average = self.k1, self.b, self.average_length or 1.0
        for term in set(query_terms):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for position, frequency in plist:
                if allowed is not None and position not in allowed:
                    continue
                length_norm = 1 - b + b * self.doc_length[position] / average
                scores[position] += idf * frequency * (k1 + 1) / (frequency + k1 * length_norm)
        return scores

    def _tfidf_scores(self, query_terms: List[str], allowed: Optional[Set[int]]) -> Dict[int, float]:
        counts = Counter(query_terms)
        weights = {t: c * self._tfidf_idf[t] for t, c in counts.items() if t in self._tfidf_idf}
        query_norm = math.sqrt(sum(w * w for w in weights.values()))
        if not query_norm:
            return {}
        scores: Dict[int, float] = defaultdict(float)
        for term, query_weight in weights.items():
            for position, frequency in self.postings[term]:
                if allowed is not None and position not in allowed:
                    continue
                scores[position] += query_weight * frequency * self._tfidf_idf[term]
        return {p: s / (query_norm * self._tfidf_norm[p]) for p, s in scores.items() if self._tfidf_norm[p]}

    # ------------------------------------------------------------------ dense scoring
    def _get_embedding_model(self):
        """Load the sentence-transformer model only when dense retrieval is used."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def _get_dense_embeddings(self):
        """Load the cached chunk embeddings, or compute and save them (once)."""
        import numpy as np

        if self._dense_embeddings is not None:
            return self._dense_embeddings

        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        corpus_hash = hashlib.sha256(Path(self.corpus_filepath).read_bytes()).hexdigest()[:16]
        model_hash = hashlib.sha256(self.embedding_model_name.encode("utf-8")).hexdigest()[:12]
        cache_file = self.embedding_cache_dir / f"embeddings_{model_hash}_{corpus_hash}.npy"

        if cache_file.exists():
            embeddings = np.load(cache_file, allow_pickle=False)
            if embeddings.shape[0] == len(self.documents):
                self._dense_embeddings = embeddings
                return embeddings

        embeddings = self._get_embedding_model().encode(
            [d["text"] for d in self.documents],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        np.save(cache_file, embeddings)
        self._dense_embeddings = embeddings
        return embeddings

    def _dense_scores(self, query: str, candidates: List[int]) -> Dict[int, float]:
        """Cosine similarity between the question and each candidate chunk, mapped to 0..1."""
        import numpy as np

        if not candidates:
            return {}
        model = self._get_embedding_model()
        corpus_embeddings = self._get_dense_embeddings()
        query_embedding = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )[0]
        similarities = np.dot(corpus_embeddings[candidates], query_embedding)
        return {index: float((s + 1.0) / 2.0) for index, s in zip(candidates, similarities)}

    @staticmethod
    def _fuse_ranks(*score_dicts: Dict[int, float], k: int = 60) -> Dict[int, float]:
        """Reciprocal Rank Fusion: combine several rankings using only the ranks."""
        fused: Dict[int, float] = defaultdict(float)
        for scores in score_dicts:
            ranked = sorted(scores, key=scores.get, reverse=True)
            for rank, position in enumerate(ranked, start=1):
                fused[position] += 1.0 / (k + rank)
        return fused

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def parse_topic_ids(topic_focus: Optional[Iterable[str]]) -> List[str]:
        """["T08: Lists, Tuples, and Sets", "t06"] -> ["T08", "T06"]. Unrecognised items are kept as given."""
        if not topic_focus:
            return []
        if isinstance(topic_focus, str):
            topic_focus = [topic_focus]
        parsed = []
        for item in topic_focus:
            match = TOPIC_ID_PATTERN.search(str(item))
            parsed.append(match.group(0).upper() if match else str(item))
        return parsed

    @staticmethod
    def _target_difficulty(mastery: Optional[float]) -> Optional[int]:
        """Map an SKB mastery score in [0, 1] to the target chunk difficulty."""
        if mastery is None:
            return None
        mastery = max(0.0, min(float(mastery), 1.0))
        if mastery < 0.4:
            return 1
        if mastery < 0.7:
            return 2
        return 3

    @staticmethod
    def _difficulty_fit(difficulty: Any, target_difficulty: Optional[int]) -> float:
        """1.0 for an exact match, 0.5 for one level away, 0.0 otherwise."""
        if target_difficulty is None or not isinstance(difficulty, (int, float)):
            return 0.0
        distance = min(abs(float(difficulty) - target_difficulty), 2.0)
        return 1.0 - distance / 2.0

    @staticmethod
    def _source_details(document: Dict[str, Any]) -> Dict[str, str]:
        """Title, section path and URL of the chunk's source page, for citations."""
        source = document.get("source")
        if not isinstance(source, dict):
            source = {}
        section_path = source.get("section_path")
        if not isinstance(section_path, list):
            section_path = []
        return {
            "title": source.get("page_title") or source.get("name") or "Course material",
            "section": " > ".join(str(part) for part in section_path),
            "url": source.get("url") or "",
        }

    # ------------------------------------------------------------------ search
    def _raw_scores(self, query: str, method: str, allowed: Optional[Set[int]]) -> Dict[int, float]:
        query_terms = self._tokenize(query)
        if method == "bm25":
            return self._bm25_scores(query_terms, allowed)
        if method == "tfidf":
            return self._tfidf_scores(query_terms, allowed)
        candidates = sorted(allowed) if allowed is not None else list(range(len(self.documents)))
        dense = self._dense_scores(query, candidates)
        if method == "dense":
            return dense
        return self._fuse_ranks(self._bm25_scores(query_terms, allowed), dense)  # hybrid

    def search(
        self,
        query: str,
        topic_ids: Optional[List[str]] = None,
        mastery_scores: Optional[Dict[str, float]] = None,
        limit: int = 4,
        method: str = "bm25",
        *,
        preferred_topics: Optional[List[str]] = None,
        topic_boost: float = 1.0,
        topic_fallback: bool = True,
        dedupe: bool = True,
        avoid_later_topics: bool = True,
        include_context: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return the `limit` most useful chunks for the question, best first.

        Args:
            query: the student's question (do not add topic names to it; use topic_ids).
            topic_ids: only search these topics. Accepts "T08" or "T08: Lists, Tuples, and Sets".
            mastery_scores: {topic_id: mastery 0..1} from the student's knowledge base. Used to
                prefer chunks of a suitable difficulty and to avoid chunks that use topics the
                student has not learned yet.
            method: "bm25" (default), "tfidf", "dense" or "hybrid".
            preferred_topics: a softer alternative to topic_ids: chunks of these topics get their
                score raised by `topic_boost` (doubled by default) but nothing is excluded, so a wrong
                guess of the topic cannot hide the right chunks.
            topic_fallback: if the topics have no relevant chunk, search the whole corpus
                (results then carry "outside_topic": True).
        """
        method = (method or "bm25").lower()
        if method not in METHODS:
            raise ValueError(f"Unknown retrieval method: {method!r}. Choose from {', '.join(METHODS)}.")
        if not isinstance(query, str) or not query.strip() or limit <= 0:
            return []

        topics = set(self.parse_topic_ids(topic_ids))
        allowed = {i for t in topics for i in self._topic_index.get(t, ())} if topics else None

        raw = self._raw_scores(query, method, allowed) if (allowed is None or allowed) else {}
        outside_topic = False
        if not raw and topics and topic_fallback:
            raw, outside_topic = self._raw_scores(query, method, None), True
        raw = {p: s for p, s in raw.items() if s > 0}
        if not raw:
            return []
        preferred = set(self.parse_topic_ids(preferred_topics))
        if preferred:
            raw = {
                p: s * (1.0 + topic_boost) if self.documents[p].get("topic_id") in preferred else s
                for p, s in raw.items()
            }

        # Re-rank a pool of the best candidates by difficulty fit and the later-topics check.
        pool = heapq.nlargest(max(limit * 3, 12), raw.items(), key=lambda item: item[1])
        low, high = min(s for _, s in pool), pool[0][1]
        # BM25/TF-IDF scores are compared with the best score. Embedding and fused scores sit in a
        # narrow band, so those are stretched to 0..1 first.
        stretch = method in ("dense", "hybrid")
        mastery_scores = mastery_scores or {}
        ranked = []
        for position, score in pool:
            document = self.documents[position]
            if stretch:
                relevance = (score - low) / (high - low) if high > low else 1.0
            else:
                relevance = score / high
            fit = self._difficulty_fit(
                document.get("difficulty"), self._target_difficulty(mastery_scores.get(document.get("topic_id")))
            )
            adjusted = relevance * (1.0 + self.difficulty_weight * fit)
            if avoid_later_topics and mastery_scores:
                later = document.get("uses_later_topics") or []
                if any(mastery_scores.get(t) is not None and mastery_scores[t] < WEAK_MASTERY for t in later):
                    adjusted *= 0.5
            ranked.append((adjusted, score, position))
        ranked.sort(key=lambda item: item[0], reverse=True)

        results, seen_groups, chosen = [], set(), set()
        for adjusted, score, position in ranked:
            document = self.documents[position]
            group = document.get("duplicate_group")
            if dedupe and group:
                if group in seen_groups:
                    continue
                seen_groups.add(group)
            chosen.add(document.get("chunk_id"))
            result = {
                "chunk_id": document.get("chunk_id", ""),
                "topic_id": document.get("topic_id", ""),
                "type": document.get("type", ""),
                "difficulty": document.get("difficulty"),
                "subconcept_ids": document.get("subconcept_ids") or [],
                "text": document["text"],
                "source": self._source_details(document),
                "score": adjusted,
                "relevance": score,
            }
            if outside_topic:
                result["outside_topic"] = True
            results.append((result, position))
            if len(results) == limit:
                break

        final = []
        for result, position in results:
            document = self.documents[position]
            if include_context and document.get("needs_context"):
                context = [
                    {"chunk_id": cid, "text": self.documents[self._by_id[cid]]["text"]}
                    for cid in (document.get("context_chunk_ids") or [])
                    if cid in self._by_id and cid not in chosen
                ][:2]
                if context:
                    result["context"] = context
            final.append(result)
        return final

    def __call__(self, query: str) -> str:
        """Search and return the results as one text block (usable as a simple retriever function)."""
        return self.format_context(self.search(query))

    @staticmethod
    def format_context(results: List[Dict[str, Any]]) -> str:
        """Turn search results into one prompt-ready text block, each chunk with its citation."""
        sections = []
        for result in results:
            source = result["source"]
            citation = source["title"]
            if source["section"]:
                citation += f" — {source['section']}"
            if source["url"]:
                citation += f" ({source['url']})"
            background = "".join(
                f"(Background for the chunk below, from [{c['chunk_id']}])\n{c['text']}\n\n"
                for c in result.get("context", [])
            )
            sections.append(
                f"{background}[{result['chunk_id']} | {result['topic_id']} | "
                f"{result['type']} | difficulty {result['difficulty']}]\n"
                f"Source: {citation}\n{result['text']}"
            )
        return "\n\n---\n\n".join(sections)
