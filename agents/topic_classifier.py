"""Semantic topic / subtopic detection for student messages.

Keyword rules (struggle_tracker.topic_evidence) are precise but miss messages that use none of
the listed words ("how do I keep data after the program closes?" -> File Handling). This
classifier compares the message with labelled course material instead:

    examples: the course chunks (data/common/corpus.jsonl, labelled with topic and subtopics),
              the quiz questions (data/common/mcqs_list.json, phrased like student questions)
              and the topic catalog (names, descriptions, subtopics, misconceptions).

    prediction: the K most similar examples vote, weighted by similarity. The winner is used
                only when its share of the vote is at least MIN_SHARE (and the best match is
                similar enough), otherwise the classifier abstains.

Backends:
    "embedding"  a sentence-transformers model (default sentence-transformers/all-MiniLM-L6-v2,
                 about 90 MB). Used when it is already downloaded, or when TOPIC_EMBED_DOWNLOAD=1
                 allows downloading it. TOPIC_EMBED_MODEL picks another model, "none" disables it.
    "tfidf"      word TF-IDF, no download; weaker on paraphrases. Used otherwise.
"""

import json
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from agents.struggle_tracker import TOPICS_FILEPATH

CORPUS_FILEPATH = "data/common/corpus.jsonl"
MCQS_FILEPATH = "data/common/mcqs_list.json"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

K = 10
MIN_SHARE = {"embedding": 0.40, "tfidf": 0.45}
MIN_TOP_SIMILARITY = {"embedding": 0.35, "tfidf": 0.10}


def _examples(topics_filepath: str) -> List[Tuple[str, str, List[str]]]:
    """(text, topic id, subtopic ids) for every labelled example."""
    examples = []
    with open(topics_filepath, "r", encoding="utf-8") as f:
        for topic in json.load(f):
            tid = topic["topic_id"]
            examples.append((f"{topic.get('name', '')}. {topic.get('description', '')}", tid, []))
            for sub in topic.get("subconcepts", []):
                examples.append((sub["name"], tid, [sub["id"]]))
            for item in topic.get("common_misconceptions", []):
                examples.append((item.get("description", ""), tid, []))
    if os.path.exists(CORPUS_FILEPATH):
        with open(CORPUS_FILEPATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    examples.append((chunk.get("text", "")[:1000], chunk.get("topic_id", ""),
                                     list(chunk.get("subconcept_ids", []))))
    if os.path.exists(MCQS_FILEPATH):
        with open(MCQS_FILEPATH, "r", encoding="utf-8") as f:
            for question in json.load(f):
                subtopics = list(question.get("related_subtopics", []))
                if subtopics:      # the first subtopic is the question's main subject
                    examples.append((question.get("prob_body", ""), subtopics[0].split(".")[0], subtopics[:1]))
    return [e for e in examples if e[0].strip() and e[1]]


def _load_embedding_model():
    name = os.environ.get("TOPIC_EMBED_MODEL") or DEFAULT_EMBED_MODEL
    if name.lower() == "none":
        return None
    try:
        from sentence_transformers import SentenceTransformer
        allow_download = os.environ.get("TOPIC_EMBED_DOWNLOAD") == "1"
        return SentenceTransformer(name, local_files_only=not allow_download)
    except Exception as error:
        print(f"[TopicClassifier] Embedding model {name} not available ({type(error).__name__}); using TF-IDF. "
              f"Set TOPIC_EMBED_DOWNLOAD=1 once to download it.")
        return None


class TopicClassifier:
    def __init__(self, topics_filepath: str = TOPICS_FILEPATH):
        import numpy as np
        examples = _examples(topics_filepath)
        self.texts = [e[0] for e in examples]
        self.topics = [str(e[1]) for e in examples]
        self.subtopics = [e[2] for e in examples]
        self._model = _load_embedding_model()
        if self._model is not None:
            self.backend = "embedding"
            self.matrix = self._model.encode(self.texts, normalize_embeddings=True, batch_size=64)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.backend = "tfidf"
            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english",
                                               token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z_0-9]+\b")
            self.matrix = self._vectorizer.fit_transform(self.texts)

    def _similarities(self, text: str):
        import numpy as np
        if self.backend == "embedding":
            return self.matrix @ self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray((self.matrix @ self._vectorizer.transform([text]).T).todense()).ravel()

    def _vote(self, text: str, labels_of) -> Tuple[Optional[str], List[Tuple[str, float]]]:
        import numpy as np
        similarities = self._similarities(text)
        nearest = [i for i in np.argsort(-similarities)[:K * 3] if labels_of(i)][:K]
        votes: Dict[str, float] = {}
        for i in nearest:
            for label in labels_of(i):
                votes[label] = votes.get(label, 0.0) + max(float(similarities[i]), 0.0)
        total = sum(votes.values())
        if not total:
            return None, []
        ranking = sorted(((label, round(v / total, 2)) for label, v in votes.items()), key=lambda x: -x[1])[:3]
        confident = (
            ranking[0][1] >= MIN_SHARE[self.backend]
            and float(similarities[nearest[0]]) >= MIN_TOP_SIMILARITY[self.backend]
        )
        return (ranking[0][0] if confident else None), ranking

    def topic(self, text: str):
        return self._vote(text, lambda i: [self.topics[i]])

    def subtopic(self, text: str, topic_id: str):
        return self._vote(text, lambda i: [s for s in self.subtopics[i] if s.startswith(f"{topic_id}.")])


@lru_cache(maxsize=1)
def get_classifier() -> Optional[TopicClassifier]:
    try:
        return TopicClassifier()
    except Exception as error:                  # never break the tutor because of the classifier
        print(f"[TopicClassifier] Disabled: {error}")
        return None


def classify_topic(text: str) -> Tuple[Optional[str], List[Tuple[str, float]]]:
    """(topic id when confident else None, top-3 vote shares)."""
    classifier = get_classifier()
    if not classifier or not str(text or "").strip():
        return None, []
    return classifier.topic(text)


def classify_subtopic(text: str, topic_id: str) -> Optional[str]:
    classifier = get_classifier()
    if not classifier or not str(text or "").strip():
        return None
    return classifier.subtopic(text, topic_id)[0]
