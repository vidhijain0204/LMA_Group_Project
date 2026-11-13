import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.tutor_agent import create_tutor_node


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.search_call = None

    def search(self, query, limit, method):
        self.search_call = (query, limit, method)
        return self.results

    @staticmethod
    def format_context(results):
        return "\n".join(result["text"] for result in results)


class FakeLLM:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content="A for loop repeats code for each item.")


class TutorRagTests(unittest.TestCase):
    def setUp(self):
        self.chunk = {
            "chunk_id": "loops_intro",
            "text": "A for loop repeats code for each item in a sequence.",
            "source": {
                "title": "Python Tutorial",
                "section": "Control flow > for statements",
                "url": "https://example.test/loops",
            },
        }
        self.state = {
            "student_id": "S001",
            "student_kb": None,
            "topic_focus": ["Control Flow"],
            "orchestrator_instruction": "Explain for loops.",
            "messages": [
                HumanMessage(content="How does a for loop work?")
            ],
        }

    def test_tutor_retrieves_course_context_and_cites_it(self):
        retriever = FakeRetriever([self.chunk])
        llm = FakeLLM()

        with patch("agents.tutor_agent.get_llm", return_value=llm):
            tutor_node = create_tutor_node(retriever=retriever)
            result = tutor_node(self.state)

        self.assertEqual(
            retriever.search_call,
            ("How does a for loop work? Control Flow", 4, "bm25"),
        )
        system_prompt = next(
            message.content
            for message in llm.messages
            if isinstance(message, SystemMessage)
        )
        self.assertIn(self.chunk["text"], system_prompt)
        self.assertIn(
            "Course sources: [Python Tutorial — Control flow > for statements]"
            "(https://example.test/loops) [loops_intro]",
            result["messages"][0].content,
        )

    def test_tutor_notes_when_retrieval_returns_no_chunks(self):
        retriever = FakeRetriever([])
        llm = FakeLLM()

        with patch("agents.tutor_agent.get_llm", return_value=llm):
            tutor_node = create_tutor_node(retriever=retriever)
            result = tutor_node(self.state)

        system_prompt = next(
            message.content
            for message in llm.messages
            if isinstance(message, SystemMessage)
        )
        self.assertIn(
            "No matching course material was retrieved.",
            system_prompt,
        )
        self.assertNotIn("Course sources:", result["messages"][0].content)


if __name__ == "__main__":
    unittest.main()
