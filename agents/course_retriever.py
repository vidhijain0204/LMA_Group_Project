import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


DEFAULT_CORPUS_FILEPATH = "data/common/corpus.jsonl"
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


class CourseRetriever:
    """Local TF-IDF retriever for the course chunks in corpus.jsonl."""

    def __init__(self, corpus_filepath: str = DEFAULT_CORPUS_FILEPATH):
        self.corpus_filepath = corpus_filepath
        self.documents = self._load_documents(corpus_filepath)
        self.document_frequencies: Counter = Counter()
        self.document_terms: List[Counter] = []
        self._index_documents()

    @staticmethod
    def _load_documents(filepath: str) -> List[Dict[str, Any]]:
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
        return TOKEN_PATTERN.findall(text.lower())

    def _index_documents(self) -> None:
        for document in self.documents:
            terms = Counter(self._tokenize(document["text"]))
            self.document_terms.append(terms)
            self.document_frequencies.update(terms.keys())

    def _idf(self, term: str) -> float:
        total_documents = len(self.documents)
        return math.log(
            (1 + total_documents) / (1 + self.document_frequencies[term])
        ) + 1

    def _similarity(self, query_terms: Counter, document_terms: Counter) -> float:
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

        # Be defensive if a stored score is slightly outside the expected range.
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
        """Return relevant chunks, using mastery as a bounded ranking preference.

        Lexical relevance remains the foundation of the score. A difficulty fit
        multiplies that score by at most 1.10, so personalization can reorder
        similarly relevant chunks without giving a weak keyword match a large
        fixed bonus. Misconception metadata is intentionally not used here.
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

            # Relevance-first ranking: exact difficulty match gives up to a 10%
            # uplift, one level away gives 5%, and a mismatch gives no uplift.
            # This avoids a fixed additive bonus overwhelming lexical relevance.
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
        """Keep compatibility with TutorAgent's original string retriever hook."""
        results = self.search(query)
        return self.format_context(results)

    @staticmethod
    def format_context(results: List[Dict[str, Any]]) -> str:
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
