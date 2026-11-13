"""Course material search (TF-IDF) over data/common/corpus.jsonl.

What it is for: given a student's question, find the most relevant chunks of course text
(explanations, examples) so they could be added to the tutor's prompt as reference material.

Status: NOT connected to the app at the moment. No other file imports it; the tutor answers
from the LLM alone. It is kept so it can be plugged into tutor_agent.py later.

How it ranks a chunk:
    1. Word similarity (TF-IDF cosine) between the question and the chunk - the main score.
    2. A small bonus (up to +10%) when the chunk's difficulty (1-3) fits the student's mastery
       of that topic, so weaker students see easier material first.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


DEFAULT_CORPUS_FILEPATH = "data/common/corpus.jsonl"
# A "word" is any run of lowercase letters, digits or underscores (so "list_append" stays one word).
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


class CourseRetriever:
    """Local TF-IDF search engine over the course chunks in corpus.jsonl."""

    def __init__(self, corpus_filepath: str = DEFAULT_CORPUS_FILEPATH):
        """Load all chunks and count their words once, so each search is fast."""
        self.corpus_filepath = corpus_filepath
        self.documents = self._load_documents(corpus_filepath)
        self.document_frequencies: Counter = Counter()
        self.document_terms: List[Counter] = []
        self._index_documents()

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
                    raise ValueError(
                        f"Invalid JSON in {filepath} at line {line_number}"
                    ) from error
                if not isinstance(document.get("text"), str) or not document["text"].strip():
                    continue
                documents.append(document)
        return documents

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Split text into lowercase words."""
        return TOKEN_PATTERN.findall(text.lower())

    def _index_documents(self) -> None:
        """Count the words of every chunk, and in how many chunks each word appears."""
        for document in self.documents:
            terms = Counter(self._tokenize(document["text"]))
            self.document_terms.append(terms)
            self.document_frequencies.update(terms.keys())

    def _idf(self, term: str) -> float:
        """Inverse document frequency: rare words get a high weight, common words a low one."""
        total_documents = len(self.documents)
        return math.log(
            (1 + total_documents) / (1 + self.document_frequencies[term])
        ) + 1

    def _similarity(self, query_terms: Counter, document_terms: Counter) -> float:
        """Cosine similarity (0..1) between the TF-IDF vectors of the question and a chunk."""
        query_weights = {
            term: count * self._idf(term) for term, count in query_terms.items()
        }
        document_weights = {
            term: count * self._idf(term) for term, count in document_terms.items()
        }
        dot_product = sum(
            weight * document_weights.get(term, 0.0)
            for term, weight in query_weights.items()
        )
        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
        document_norm = math.sqrt(
            sum(weight * weight for weight in document_weights.values())
        )
        if not query_norm or not document_norm:
            return 0.0
        return dot_product / (query_norm * document_norm)

    @staticmethod
    def _target_difficulty(mastery: Optional[float]) -> Optional[int]:
        """Map an SKB mastery score in [0, 1] to the target chunk difficulty."""
        if mastery is None:
            return None

        # Clamp, in case a stored score is slightly outside 0..1.
        mastery = max(0.0, min(float(mastery), 1.0))
        if mastery < 0.4:
            return 1
        if mastery < 0.7:
            return 2
        return 3

    @staticmethod
    def _difficulty_fit(
        difficulty: Any, target_difficulty: Optional[int]
    ) -> float:
        """Return 1.0 for an exact match, 0.5 for one level away, and 0.0 otherwise."""
        if target_difficulty is None or not isinstance(difficulty, (int, float)):
            return 0.0
        distance = min(abs(float(difficulty) - target_difficulty), 2.0)
        return 1.0 - distance / 2.0

    @staticmethod
    def _source_details(document: Dict[str, Any]) -> Dict[str, str]:
        """Title, section path and URL of the chunk's source page, for citations."""
        source = document.get("source") or {}
        section_path = source.get("section_path") or []
        return {
            "title": source.get("page_title") or source.get("name") or "Course material",
            "section": " > ".join(section_path),
            "url": source.get("url") or "",
        }

    def search(
        self,
        query: str,
        topic_ids: Optional[List[str]] = None,
        mastery_scores: Optional[Dict[str, float]] = None,
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        """Return the `limit` most relevant chunks for the question.

        Args:
            query: the student's question.
            topic_ids: only search chunks of these topics (all chunks if none match).
            mastery_scores: {topic_id: mastery 0..1}, used for the small difficulty bonus.
        Returns:
            A list of chunks (id, topic, difficulty, text, source, score), best first.
        """
        query_terms = Counter(self._tokenize(query))
        if not query_terms or limit <= 0:
            return []

        selected_topics: Set[str] = set(topic_ids or [])
        candidates = [
            index
            for index, document in enumerate(self.documents)
            if not selected_topics or document.get("topic_id") in selected_topics
        ]
        # No chunk in the requested topics: fall back to searching everything.
        if selected_topics and not candidates:
            candidates = list(range(len(self.documents)))

        mastery_scores = mastery_scores or {}
        results = []
        for index in candidates:
            document = self.documents[index]
            similarity = self._similarity(query_terms, self.document_terms[index])
            if similarity <= 0:
                continue

            target_difficulty = self._target_difficulty(
                mastery_scores.get(document.get("topic_id"))
            )
            difficulty = document.get("difficulty")
            difficulty_fit = self._difficulty_fit(difficulty, target_difficulty)

            # Similarity stays the main score: an exact difficulty match adds 10%, one level
            # away adds 5%, otherwise nothing.
            score = similarity * (1.0 + 0.10 * difficulty_fit)

            results.append({
                "chunk_id": document.get("chunk_id", ""),
                "topic_id": document.get("topic_id", ""),
                "type": document.get("type", ""),
                "difficulty": difficulty,
                "text": document["text"],
                "source": self._source_details(document),
                "score": score,
            })

        results.sort(key=lambda result: result["score"], reverse=True)
        return results[:limit]

    def __call__(self, query: str) -> str:
        """Search and return the results as one text block (usable as a simple retriever function)."""
        results = self.search(query)
        return self.format_context(results)

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
            sections.append(
                f"[{result['chunk_id']} | {result['topic_id']} | "
                f"{result['type']} | difficulty {result['difficulty']}]\n"
                f"Source: {citation}\n{result['text']}"
            )
        return "\n\n---\n\n".join(sections)
