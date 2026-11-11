import json
import math
import re
import hashlib
from collections import Counter
from pathlib import Path
from rank_bm25 import BM25Okapi
from typing import Any, Dict, List, Optional, Set


DEFAULT_CORPUS_FILEPATH = "data/common/corpus.jsonl"
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


class CourseRetriever:
    """Retrieve course chunks using F-IDF, BM25, dense, or hybrid retrieval."""

    def __init__(
        self,
        corpus_filepath: str = DEFAULT_CORPUS_FILEPATH,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_cache_dir: str = ".cache/course_retriever",
    ):
        self.corpus_filepath = corpus_filepath
        self.documents = self._load_documents(corpus_filepath)

        # Existing TF-IDF index
        self.document_frequencies: Counter = Counter()
        self.document_terms: List[Counter] = []
        self._index_documents()

        # Existing BM25 index
        tokenized_documents = [
            self._tokenize(document["text"])
            for document in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized_documents) if tokenized_documents else None

        # Dense retrieval: model and embeddings are loaded lazily
        self.embedding_model_name = embedding_model_name
        self.embedding_cache_dir = Path(embedding_cache_dir)
        self._embedding_model = None
        self._dense_embeddings = None

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

    def _get_embedding_model(self):
        """Load the sentence-transformer model only when dense retrieval is used."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                self.embedding_model_name
            )

        return self._embedding_model


    def _get_dense_embeddings(self):
        """Load cached corpus embeddings or generate and save them."""
        import numpy as np

        if self._dense_embeddings is not None:
            return self._dense_embeddings

        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache key changes if the corpus content or model name changes.
        corpus_hash = hashlib.sha256(
            Path(self.corpus_filepath).read_bytes()
        ).hexdigest()[:16]

        model_hash = hashlib.sha256(
            self.embedding_model_name.encode("utf-8")
        ).hexdigest()[:12]

        cache_file = (
            self.embedding_cache_dir
            / f"embeddings_{model_hash}_{corpus_hash}.npy"
        )

        if cache_file.exists():
            embeddings = np.load(cache_file, allow_pickle=False)

            if embeddings.shape[0] == len(self.documents):
                self._dense_embeddings = embeddings
                return embeddings

        # Generate embeddings only once for the entire corpus.
        model = self._get_embedding_model()

        texts = [
            document["text"]
            for document in self.documents
        ]

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        np.save(cache_file, embeddings)

        self._dense_embeddings = embeddings

        return embeddings


    def _dense_relevance_scores(self, query: str, candidates: List[int]):
        """Calculate cosine similarity between query and candidate chunks."""
        import numpy as np

        if not candidates:
            return {}

        model = self._get_embedding_model()
        corpus_embeddings = self._get_dense_embeddings()

        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        candidate_embeddings = corpus_embeddings[candidates]

        # Embeddings are normalized, so dot product equals cosine similarity.
        similarities = np.dot(candidate_embeddings, query_embedding)

        # Convert cosine similarity from [-1, 1] to [0, 1].
        relevance_scores = (similarities + 1.0) / 2.0

        return {
            index: float(score)
            for index, score in zip(candidates, relevance_scores)
    }

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

        if not isinstance(source, dict):
            source = {}

        section_path = source.get("section_path") or []

        if not isinstance(section_path, list):
            section_path = []

        return {
            "title": (
                source.get("page_title")
                or source.get("name")
                or "Course material"
            ),
            "section": " > ".join(str(part) for part in section_path),
            "url": source.get("url") or "",
    }

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: List[List[Dict[str, Any]]],
        k: int = 60,
    ) -> Dict[str, float]:
        """
        Combine multiple ranked result lists using Reciprocal Rank Fusion.

        Each result must contain a chunk_id.
        RRF uses ranks rather than the original retrieval scores.
        """
        fused_scores: Dict[str, float] = {}

        for ranked_results in ranked_lists:
            for rank, result in enumerate(ranked_results, start=1):
                chunk_id = result.get("chunk_id")

                if not chunk_id:
                    continue

                fused_scores[chunk_id] = (
                    fused_scores.get(chunk_id, 0.0)
                    + 1.0 / (k + rank)
                )

        return fused_scores

    def search(
        self,
        query: str,
        topic_ids: Optional[List[str]] = None,
        mastery_scores: Optional[Dict[str, float]] = None,
        limit: int = 4,
        method: str = "bm25",
    ) -> List[Dict[str, Any]]:
        """Retrieve course chunks using TF-IDF, BM25, or dense retrieval."""

        method = method.lower()

        if method not in {"tfidf", "bm25", "dense", "hybrid"}:
            raise ValueError(
                f"Unknown retrieval method: {method!r}. "
                "Choose 'tfidf', 'bm25', 'dense', or 'hybrid'."
            )

        if not query.strip() or limit <= 0:
            return []

        query_tokens = self._tokenize(query)
        query_terms = Counter(query_tokens)

        # Lexical methods require at least one recognized token.
        if method in {"tfidf", "bm25"} and not query_tokens:
            return []

        selected_topics: Set[str] = set(topic_ids or [])

        candidates = [
            index
            for index, document in enumerate(self.documents)
            if not selected_topics
            or document.get("topic_id") in selected_topics
        ]

        if selected_topics and not candidates:
            return []

        # Calculate relevance using the selected retrieval method.
        if method == "tfidf":
            relevance_scores = {
                index: self._similarity(
                    query_terms, self.document_terms[index]
                )
                for index in candidates
            }

        elif method == "bm25":
            if self.bm25 is None:
                return []

            raw_scores = self.bm25.get_scores(query_tokens)

            matching_candidates = [
                index
                for index in candidates
                if any(
                    term in self.document_terms[index]
                    for term in query_terms
                )
            ]

            if not matching_candidates:
                return []

            candidate_scores = {
                index: float(raw_scores[index])
                for index in matching_candidates
            }

            min_score = min(candidate_scores.values())
            max_score = max(candidate_scores.values())

            if max_score > min_score:
                relevance_scores = {
                    index: (score - min_score) / (max_score - min_score)
                    for index, score in candidate_scores.items()
                }
            else:
                relevance_scores = {
                    index: 1.0 for index in matching_candidates
                }

        elif method == "dense":
            relevance_scores = self._dense_relevance_scores(
                query, candidates
            )

        else:  # hybrid: BM25 + dense using RRF

            # Retrieve the complete ranked candidate lists from both methods.
            # Mastery is deliberately disabled here so it is not applied twice.
            component_limit = len(candidates)

            topic_filter = (
                list(selected_topics) if selected_topics else None
            )

            bm25_results = self.search(
                query=query,
                topic_ids=topic_filter,
                mastery_scores={},
                limit=component_limit,
                method="bm25",
            )

            dense_results = self.search(
                query=query,
                topic_ids=topic_filter,
                mastery_scores={},
                limit=component_limit,
                method="dense",
            )

            # Combine the two rankings using Reciprocal Rank Fusion.
            fused_scores = self._reciprocal_rank_fusion(
                [bm25_results, dense_results],
                k=60,
            )

            # Map fused chunk IDs back to their original corpus positions.
            chunk_to_index = {
                document.get("chunk_id"): index
                for index, document in enumerate(self.documents)
            }

            relevance_scores = {
                chunk_to_index[chunk_id]: score
                for chunk_id, score in fused_scores.items()
                if chunk_id in chunk_to_index
            }

        mastery_scores = mastery_scores or {}
        results = []

        for index, relevance in relevance_scores.items():
            if relevance <= 0:
                continue

            document = self.documents[index]

            target_difficulty = self._target_difficulty(
                mastery_scores.get(document.get("topic_id"))
            )

            difficulty_fit = self._difficulty_fit(
                document.get("difficulty"),
                target_difficulty,
            )

            # Relevance remains primary; mastery gives at most a 10% uplift.
            score = relevance * (1.0 + 0.10 * difficulty_fit)

            results.append({
                "chunk_id": document.get("chunk_id", ""),
                "topic_id": document.get("topic_id", ""),
                "type": document.get("type", ""),
                "difficulty": document.get("difficulty"),
                "text": document["text"],
                "source": self._source_details(document),
                "score": score,
            })

        results.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

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
