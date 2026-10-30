import json
import tempfile
import unittest
from pathlib import Path

from agents.course_retriever import CourseRetriever


class FakeEmbeddingModel:
    """Deterministic embedding model for testing dense retrieval."""

    def __init__(self):
        self.document_encoding_calls = 0

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        import numpy as np

        if len(texts) > 1:
            self.document_encoding_calls += 1

        embeddings = []

        for text in texts:
            text = text.lower()

            if "string" in text or "immutable" in text:
                embeddings.append([0.0, 1.0])
            else:
                embeddings.append([1.0, 0.0])

        return np.array(embeddings, dtype=np.float32)


class CourseRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.corpus_path = Path(self.temp_dir.name) / "corpus.jsonl"
        documents = [
            {
                "chunk_id": "loops_intro",
                "topic_id": "T06",
                "type": "explanation",
                "difficulty": 1,
                "text": "A for loop repeats code for each item in a sequence.",
                "source": {
                    "name": "Python Tutorial",
                    "section_path": ["Control flow", "for statements"],
                    "url": "https://example.test/loops",
                },
            },
            {
                "chunk_id": "loops_advanced",
                "topic_id": "T06",
                "type": "example",
                "difficulty": 3,
                "text": "A for loop repeats code for each item in a sequence.",
                "source": {"name": "Python Tutorial"},
            },
            {
                "chunk_id": "strings",
                "topic_id": "T07",
                "type": "explanation",
                "difficulty": 1,
                "text": "Strings are immutable sequences of characters.",
                "source": {"name": "Python Tutorial"},
            },
        ]
        self.corpus_path.write_text(
            "\n".join(json.dumps(document) for document in documents),
            encoding="utf-8",
        )
        self.retriever = CourseRetriever(str(self.corpus_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_search_prefers_relevant_chunks_in_routed_topic(self):
        results = self.retriever.search(
            "How does a for loop repeat code?",
            topic_ids=["T06"],
        )
        self.assertTrue(results)
        self.assertEqual(results[0]["chunk_id"], "loops_intro")
        self.assertTrue(all(result["topic_id"] == "T06" for result in results))

    def test_mastery_personalizes_difficulty_ranking(self):
        question = "for loop iterable objects"
        beginner = self.retriever.search(
            question, topic_ids=["T06"], mastery_scores={"T06": 0.2}
        )
        advanced = self.retriever.search(
            question, topic_ids=["T06"], mastery_scores={"T06": 0.9}
        )
        self.assertEqual(beginner[0]["difficulty"], 1)
        self.assertEqual(advanced[0]["difficulty"], 3)

    def test_no_matching_terms_returns_no_course_citations(self):
        self.assertEqual(self.retriever.search("quantum entanglement"), [])

    def test_format_context_includes_source_attribution(self):
        results = self.retriever.search("for loop", topic_ids=["T06"])
        context = self.retriever.format_context(results)
        self.assertIn("Python Tutorial", context)
        self.assertIn("https://example.test/loops", context)

    def test_bm25_returns_matching_chunks(self):
        results = self.retriever.search(
            "loop iteration",
            topic_ids=["T06"],
            method="bm25",
            limit=4,
        )
        self.assertTrue(results)
        self.assertTrue(
            all(result["topic_id"] == "T06" for result in results)
        )


    def test_unknown_retrieval_method_is_rejected(self):
        with self.assertRaises(ValueError):
            self.retriever.search("loop iteration", method="unknown")

    def test_dense_retrieval_returns_semantically_matching_chunks(self):
        fake_model = FakeEmbeddingModel()

        self.retriever.embedding_cache_dir = (
            Path(self.temp_dir.name) / "embedding_cache"
        )

        self.retriever._get_embedding_model = lambda: fake_model

        results = self.retriever.search(
            "How can I repeatedly execute instructions?",
            method="dense",
            limit=2,
        )

        self.assertTrue(results)
        self.assertEqual(results[0]["chunk_id"], "loops_intro")


    def test_dense_retrieval_respects_topic_filter(self):
        fake_model = FakeEmbeddingModel()

        self.retriever.embedding_cache_dir = (
            Path(self.temp_dir.name) / "embedding_cache"
        )

        self.retriever._get_embedding_model = lambda: fake_model

        results = self.retriever.search(
            "How can I repeatedly execute instructions?",
            method="dense",
            topic_ids=["T07"],
        )

        self.assertTrue(results)
        self.assertTrue(
            all(result["topic_id"] == "T07" for result in results)
    )


    def test_dense_embeddings_are_cached(self):
        fake_model = FakeEmbeddingModel()

        self.retriever.embedding_cache_dir = (
            Path(self.temp_dir.name) / "embedding_cache"
        )

        self.retriever._get_embedding_model = lambda: fake_model

        self.retriever.search("for loop", method="dense")
        self.retriever.search("loop iteration", method="dense")

        # Corpus embeddings should be generated only once.
        self.assertEqual(fake_model.document_encoding_calls, 1)

    def test_default_search_uses_bm25(self):
        query = "Python list append method"

        default_results = self.retriever.search(query)
        bm25_results = self.retriever.search(query, method="bm25")

        self.assertEqual(default_results, bm25_results)


if __name__ == "__main__":
    unittest.main()
