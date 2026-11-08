import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agents.error_analysis_agent import ErrorAnalysisAgent
from langchain_core.messages import AIMessage
from agents.pedagogy_controller import (
    choose_teaching_strategy,
    record_assessment_outcome,
    record_lesson_strategy,
)


class PedagogyControllerTests(unittest.TestCase):
    def test_initial_strategy_uses_mastery(self):
        self.assertEqual(
            choose_teaching_strategy({}, "T05", 0.2)[0],
            "step_scaffolding",
        )
        self.assertEqual(
            choose_teaching_strategy({}, "T05", 0.5)[0],
            "explicit_explanation",
        )
        self.assertEqual(
            choose_teaching_strategy({}, "T05", 0.9)[0],
            "self_explanation",
        )

    def test_switch_waits_for_two_low_assessments(self):
        kb = {"student_id": "S-test", "topics": {"T05": 0.5}}
        kb = record_lesson_strategy(kb, "T05", "explicit_explanation", "initial")
        kb = record_assessment_outcome(kb, "T05", 0.3, [])
        self.assertEqual(choose_teaching_strategy(kb, "T05", 0.5)[0], "explicit_explanation")

        kb = record_lesson_strategy(kb, "T05", "explicit_explanation", "continue")
        kb = record_assessment_outcome(kb, "T05", 0.4, [])
        self.assertEqual(choose_teaching_strategy(kb, "T05", 0.5)[0], "worked_example")

    def test_recurring_error_signal_prioritizes_contrastive_feedback(self):
        kb = {"student_id": "S-test", "topics": {"T05": 0.5}}
        kb = record_lesson_strategy(kb, "T05", "explicit_explanation", "initial")
        kb = record_assessment_outcome(
            kb, "T05", 0.8, ["T05.2"], {"T05.2": "if-else"}
        )
        kb = record_lesson_strategy(kb, "T05", "explicit_explanation", "continue")
        kb = record_assessment_outcome(
            kb, "T05", 0.8, ["T05.2"], {"T05.2": "if-else"}
        )
        self.assertEqual(choose_teaching_strategy(kb, "T05", 0.5)[0], "contrastive_feedback")
        self.assertEqual(kb["learning_signals"]["T05.2"], 2)
        self.assertIn("if-else", kb["learning_signal_descriptions"]["T05.2"])

    def test_error_analysis_uses_quiz_subtopic_metadata(self):
        analyzer = ErrorAnalysisAgent()
        result = analyzer.analyze(
            [{
                "prob_id": "test-q",
                "correct_option": "A",
                "related_topics": ["T05"],
                "related_subtopics": ["T05.2"],
            }],
            {"test-q": "B"},
            topic_id="T05",
        )
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["topic_accuracy"], 0.0)
        self.assertEqual(result["errors"][0]["signal_id"], "T05.2")
        self.assertEqual(result["errors"][0]["description"], "if-else")

    def test_error_analysis_does_not_assign_other_topic_errors(self):
        analyzer = ErrorAnalysisAgent()
        result = analyzer.analyze(
            [{
                "prob_id": "other-topic-q",
                "correct_option": "A",
                "related_topics": ["T02"],
                "related_subtopics": ["T02.1"],
            }],
            {"other-topic-q": "B"},
            topic_id="T05",
        )
        self.assertEqual(result["topic_accuracy"], None)
        self.assertEqual(result["errors"], [])

    def test_tutor_uses_and_records_selected_strategy(self):
        from langchain_core.messages import HumanMessage

        class MockTutor:
            def invoke(self, messages):
                self.messages = messages
                return AIMessage(content="Try separating the condition from its branches.")

        mock_tutor = MockTutor()
        kb = {"student_id": "S-test", "topics": {"T05": 0.3}}
        state = {
            "student_id": "S-test",
            "student_kb": kb,
            "topic_focus": ["T05"],
            "messages": [HumanMessage(content="How do if statements work?")],
        }

        with patch("agents.tutor_agent.get_llm", return_value=mock_tutor):
            from agents.tutor_agent import create_tutor_node

            result = create_tutor_node()(state)

        self.assertEqual(result["pedagogy_strategy"], "step_scaffolding")
        self.assertIn("step_scaffolding", mock_tutor.messages[0].content)
        self.assertEqual(
            result["student_kb"]["pedagogy_history"][-1]["strategy"],
            "step_scaffolding",
        )

    def test_workflow_logs_pedagogy_strategy_switch(self):
        from agents.workflow import AdaptiveTutorWorkflow

        workflow = AdaptiveTutorWorkflow.__new__(AdaptiveTutorWorkflow)
        workflow.tutor_node_fn = lambda state: {
            "messages": [AIMessage(content="A tutoring response.")],
            "student_kb": state["student_kb"],
            "pedagogy_strategy": "contrastive_feedback",
            "pedagogy_reason": "Recurring errors call for a direct comparison.",
        }
        workflow._save_student_kb = lambda student_kb: None
        state = {
            "topic_focus": ["T05"],
            "student_kb": {
                "pedagogy_history": [{
                    "event": "lesson",
                    "topic_id": "T05",
                    "strategy": "explicit_explanation",
                }],
            },
        }

        output = io.StringIO()
        with redirect_stdout(output):
            workflow.tutor_node(state)

        self.assertIn(
            "Teaching strategy changed: Explicit Explanation → Contrastive Feedback",
            output.getvalue(),
        )
        self.assertIn("Recurring errors call for a direct comparison.", output.getvalue())

    def test_workflow_persists_pedagogy_history_without_replacing_other_students(self):
        from agents.workflow import AdaptiveTutorWorkflow

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "skb.json"
            db_path.write_text(
                json.dumps([{"student_id": "S-other"}, {"student_id": "S-test"}]),
                encoding="utf-8",
            )
            workflow = AdaptiveTutorWorkflow.__new__(AdaptiveTutorWorkflow)
            workflow.db_filepath = str(db_path)
            updated_kb = {
                "student_id": "S-test",
                "pedagogy_history": [{"event": "lesson", "strategy": "hints"}],
            }

            workflow._save_student_kb(updated_kb)

            saved_records = json.loads(db_path.read_text(encoding="utf-8"))
            self.assertEqual(len(saved_records), 2)
            self.assertEqual(saved_records[0]["student_id"], "S-other")
            self.assertEqual(saved_records[1]["pedagogy_history"], updated_kb["pedagogy_history"])


if __name__ == "__main__":
    unittest.main()
