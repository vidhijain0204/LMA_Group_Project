import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from agents.pedagogy_controller import latest_root_cause, record_assessment_outcome
from agents.root_cause_agent import (
    PARAMS,
    WEIGHT_PRESETS,
    RootCauseAgent,
    ablation_weights,
    summarize_comparison,
)
from agents.struggle_tracker import (
    detect_struggle,
    learning_struggle,
    map_subtopics,
    record_learning_turn,
)


def question(prob_id, subtopics, correct="A"):
    return {
        "prob_id": prob_id,
        "options": {"A": "right", "B": "wrong"},
        "correct_option": correct,
        "related_topics": sorted({s.split(".")[0] for s in subtopics}),
        "related_subtopics": subtopics,
        "explanations": "",
    }


class StruggleTrackerTests(unittest.TestCase):
    def test_detects_confusion_and_hint_requests(self):
        self.assertTrue(detect_struggle("I'm still confused about this")["confusion"])
        self.assertTrue(detect_struggle("Why is my file empty after I write to it?")["confusion"])
        self.assertTrue(detect_struggle("just tell me the answer")["hint_request"])
        self.assertEqual(
            detect_struggle("Can you explain how with works?"),
            {"confusion": False, "hint_request": False},
        )

    def test_maps_message_to_distinctive_subtopic(self):
        self.assertIn("T13.1", map_subtopics("which file mode should I use to open it?", "T13"))
        self.assertIn("T13.2", map_subtopics("how does the with statement close the file?", "T13"))

    def test_struggle_turns_raise_learning_struggle(self):
        kb = {"student_id": "S-test", "topics": {"T13": 0.8}}
        self.assertEqual(learning_struggle(kb, "T13.1")["LS"], 0.0)
        for text in (
            "what does file mode w do?",
            "I'm confused, why is my file empty after opening in w mode?",
            "still not working, just tell me the answer for the mode",
        ):
            kb = record_learning_turn(kb, "T13", text, previous_strategy="self_explanation")
        result = learning_struggle(kb, "T13.1")
        self.assertGreater(result["LS"], 0.3)
        self.assertEqual(result["components"]["RA"], 1.0)
        self.assertGreater(result["components"]["WS"], 0.0)


class RootCauseAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = RootCauseAgent()
        self.kb = {"student_id": "S-test", "topics": {"T13": 0.6, "T12": 0.6, "T07": 0.6, "T03": 0.6}}

    def test_counter_evidence_separates_shared_tags(self):
        # T13.1 and T13.2 are both on the two wrong questions, but T13.2 is answered
        # correctly elsewhere, so T13.1 must be the root cause.
        questions = [
            question("q1", ["T13.1", "T13.2"]),
            question("q2", ["T13.1", "T13.2"]),
            question("q3", ["T13.2", "T13.3"]),
            question("q4", ["T13.2", "T13.4"]),
        ]
        answers = {"q1": "B", "q2": "B", "q3": "A", "q4": "A"}
        result = self.agent.analyze(questions, answers, self.kb)

        self.assertLessEqual(len(result["hypotheses"]), 3)
        self.assertEqual(result["root_cause"]["subtopic_id"], "T13.1")
        by_id = {h["subtopic_id"]: h for h in result["hypotheses"]}
        self.assertEqual(by_id["T13.1"]["params"]["CE"], 0.0)
        self.assertGreater(by_id["T13.2"]["params"]["CE"], 0.0)
        self.assertAlmostEqual(sum(h["probability"] for h in result["hypotheses"]), 1.0, places=1)

    def test_learning_struggle_breaks_tie_despite_good_mastery(self):
        questions = [question("q1", ["T13.1", "T13.2"])]
        kb = dict(self.kb, topics={**self.kb["topics"], "T13": 0.9})
        for text in (
            "I don't understand the with statement",
            "still confused about with, it doesn't make sense",
            "just tell me the answer for with",
        ):
            kb = record_learning_turn(kb, "T13", text)

        result = self.agent.analyze(questions, {"q1": "B"}, kb)
        top = result["hypotheses"][0]
        self.assertEqual(result["root_cause"]["subtopic_id"], "T13.2")
        # Fragile mastery: PM' is higher than the plain 1 - mastery gap.
        self.assertGreater(top["params"]["PM"], 0.1)

    def test_no_wrong_answers_means_no_root_cause(self):
        result = self.agent.analyze([question("q1", ["T13.1", "T13.2"])], {"q1": "A"}, self.kb)
        self.assertEqual(result["hypotheses"], [])
        self.assertIsNone(result["root_cause"])

    def test_outcome_stores_responses_and_root_cause(self):
        questions = [question("q1", ["T13.1", "T13.2"])]
        result = self.agent.analyze(questions, {"q1": "B"}, self.kb)
        kb = record_assessment_outcome(
            self.kb, "T13", 0.0, ["T13.1"],
            question_results=result["responses"], root_cause=result["root_cause"],
        )
        event = kb["pedagogy_history"][-1]
        self.assertEqual(event["responses"][0]["chosen"], "B")
        self.assertEqual(event["root_cause"], result["root_cause"])
        self.assertEqual(latest_root_cause(kb, "T13")["subtopic_id"], result["root_cause"]["subtopic_id"])

        # The stored responses feed the repeat parameter of the next quiz.
        again = self.agent.analyze(questions, {"q1": "B"}, kb)
        self.assertEqual(again["hypotheses"][0]["params"]["RP"], 1.0)


class MethodAndWeightTests(unittest.TestCase):
    def setUp(self):
        self.agent = RootCauseAgent()
        # T13.1 has the most wrong answers; T13.2 has the lowest stored mastery.
        self.questions = [
            question("q1", ["T13.1", "T13.3"]),
            question("q2", ["T13.1", "T13.4"]),
            question("q3", ["T13.2", "T13.5"]),
        ]
        self.answers = {"q1": "B", "q2": "B", "q3": "B"}
        self.kb = {"student_id": "S-test", "topics": {"T13": 0.6},
                   "skills": {"T13.1": 0.8, "T13.2": 0.1, "T13.3": 0.7, "T13.4": 0.7, "T13.5": 0.7}}

    def test_no_kb_picks_most_wrong_and_ignores_kb(self):
        result = self.agent.analyze(self.questions, self.answers, self.kb, method="no_kb")
        self.assertEqual(result["root_cause"]["subtopic_id"], "T13.1")
        self.assertEqual(result["root_cause"]["method"], "no_kb")
        self.assertEqual(result["hypotheses"], [])

    def test_kb_low_score_picks_lowest_mastery(self):
        result = self.agent.analyze(self.questions, self.answers, self.kb, method="kb_low_score")
        self.assertEqual(result["root_cause"]["subtopic_id"], "T13.2")
        self.assertIn("0.10", result["root_cause"]["reason"])

    def test_summary_is_topic_subtopic_reason(self):
        root = self.agent.analyze(self.questions, self.answers, self.kb)["root_cause"]
        self.assertTrue(root["summary"].startswith("T13 File Handling → "))
        self.assertEqual(root["summary"].count("→"), 2)

    def test_hypothesis_uses_only_the_five_parameters(self):
        result = self.agent.analyze(self.questions, self.answers, self.kb)
        for h in result["hypotheses"]:
            self.assertEqual(set(h["params"]), set(PARAMS))
            self.assertLessEqual(h["contributions"]["CE"], 0)

    def test_weights_change_the_scores(self):
        mastery_only = {key: 0.0 for key in PARAMS}
        mastery_only["PM"] = 1.0
        result = self.agent.analyze(self.questions, self.answers, self.kb, weights=mastery_only)
        self.assertEqual(result["preset"], "custom")
        # With only PM' (1 - mastery), the lowest-mastery candidate wins.
        top = result["hypotheses"][0]
        self.assertEqual(top["score"], top["params"]["PM"])
        self.assertEqual(top["mastery"], min(h["mastery"] for h in result["hypotheses"]))

    def test_ablation_drops_one_parameter_and_keeps_total(self):
        weights = ablation_weights(WEIGHT_PRESETS["balanced"], "LS")
        self.assertEqual(weights["LS"], 0.0)
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=3)

    def test_compare_runs_every_method_preset_and_ablation(self):
        comparison = self.agent.compare(self.questions, self.answers, self.kb)
        compact = summarize_comparison(comparison)
        self.assertEqual(set(compact["methods"]), {"no_kb", "kb_low_score", "hypothesis"})
        self.assertEqual(set(compact["presets"]), set(WEIGHT_PRESETS))
        self.assertEqual(len(compact["ablations"]), len(PARAMS))
        self.assertEqual(compact["methods"]["no_kb"], "T13.1")
        self.assertEqual(compact["methods"]["kb_low_score"], "T13.2")


class IntegrationTests(unittest.TestCase):
    def test_tutor_logs_learning_turn_and_uses_root_cause(self):
        class MockTutor:
            def invoke(self, messages):
                self.messages = messages
                return AIMessage(content="Let's look at file modes.")

        mock_tutor = MockTutor()
        kb = {
            "student_id": "S-test",
            "topics": {"T13": 0.8},
            "root_cause_history": [{
                "subtopic_id": "T13.1", "name": "Opening and closing files; file modes",
                "topic_id": "T13", "quiz_topic_id": "T13", "probability": 0.6, "reason": "test",
            }],
        }
        state = {
            "student_id": "S-test",
            "student_kb": kb,
            "topic_focus": ["T13"],
            "user_input": "I'm confused, why is my file empty after opening it in w mode?",
            "messages": [HumanMessage(content="I'm confused, why is my file empty after opening it in w mode?")],
        }
        with patch("agents.tutor_agent.get_llm", return_value=mock_tutor):
            from agents.tutor_agent import create_tutor_node

            result = create_tutor_node()(state)

        self.assertIn("T13.1 Opening and closing files; file modes", mock_tutor.messages[0].content)
        turn = result["student_kb"]["learning_conversations"][-1]
        self.assertTrue(turn["confusion"])
        self.assertIn("T13.1", turn["subtopic_ids"])

    def test_workflow_quiz_node_reports_root_cause(self):
        from agents.error_analysis_agent import ErrorAnalysisAgent
        from agents.workflow import AdaptiveTutorWorkflow

        questions = [question("q1", ["T13.1", "T13.2"]), question("q2", ["T13.2", "T13.3"])]
        workflow = AdaptiveTutorWorkflow.__new__(AdaptiveTutorWorkflow)
        workflow.assessment_node_fn = lambda state: {
            "messages": [AIMessage(content="Quiz intro")],
            "active_quiz_questions": questions,
        }
        workflow.error_analysis_agent = ErrorAnalysisAgent()
        workflow.root_cause_agent = RootCauseAgent()
        saved = []
        workflow._save_student_kb = saved.append
        state = {"topic_focus": ["T13"], "student_kb": {"student_id": "S-test", "topics": {"T13": 0.6}}}

        with patch("agents.workflow.interrupt", return_value={"q1": "B", "q2": "A"}), \
                patch("agents.workflow.load_all_mcqs", return_value=questions), \
                patch("builtins.print"):
            output = workflow.assessment_quiz_node(state)

        self.assertEqual(output["response_output"]["root_cause"]["subtopic_id"], "T13.1")
        self.assertEqual(saved[-1]["root_cause_history"][-1]["subtopic_id"], "T13.1")
        self.assertEqual(saved[-1]["pedagogy_history"][-1]["responses"][0]["chosen"], "B")
        self.assertIn("methods", saved[-1]["pedagogy_history"][-1]["root_cause_comparison"])
        agents = [step["agent"] for step in workflow._trace]
        for agent in ("Assessment Agent", "Error Analysis Agent", "Root Cause Agent", "Pedagogy Controller"):
            self.assertIn(agent, agents)


if __name__ == "__main__":
    unittest.main()
