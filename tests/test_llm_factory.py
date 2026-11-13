import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from agents.llm_factory import _invoke_with_continuation


class LLMFactoryTests(unittest.TestCase):
    def test_continues_truncated_response_and_combines_usage(self):
        model = Mock()
        model.invoke.side_effect = [
            AIMessage(
                content="First part. ",
                response_metadata={"finish_reason": "length"},
                usage_metadata={
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "total_tokens": 5,
                },
            ),
            AIMessage(
                content="Second part.",
                response_metadata={"finish_reason": "stop"},
                usage_metadata={
                    "input_tokens": 4,
                    "output_tokens": 2,
                    "total_tokens": 6,
                },
            ),
        ]

        with patch("agents.llm_factory._get_chat_model", return_value=model):
            result = _invoke_with_continuation(
                [HumanMessage(content="Explain this.")], "tutor"
            )

        self.assertEqual(result.content, "First part. Second part.")
        self.assertEqual(result.response_metadata["continuation_count"], 1)
        self.assertEqual(result.usage_metadata["total_tokens"], 11)
        second_request = model.invoke.call_args_list[1].args[0]
        self.assertEqual(second_request[-2].content, "First part. ")
        self.assertTrue(second_request[-1].content.startswith("/no_think"))


if __name__ == "__main__":
    unittest.main()
