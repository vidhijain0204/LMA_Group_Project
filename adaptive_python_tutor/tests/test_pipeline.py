from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path

from generation.build_dataset import DEFAULTS, _assign_difficulties, _parse_difficulty_distribution, build_dataset
from generation.generate_questions import generate_template_questions, validate_question
from generation.skills import SKILLS, Skill, validate_skill_graph
from generation.simulate_students import (
    create_student_profiles,
    response_probability,
    simulate_student_interactions,
)
from generation.update_mastery import LearningEvent, MASTERY_THRESHOLD, update_mastery
from generation.validate_questions import filter_executable_questions
from utils.validation import agent_history_as_of, read_csv_rows


class QuestionGenerationTests(unittest.TestCase):
    def test_question_schema_and_executable_answer_keys(self) -> None:
        questions, rejected, report = generate_template_questions(42, seed=7)
        self.assertEqual(rejected, [])
        self.assertEqual(report["accepted"], 42)
        self.assertEqual(len(questions), 42)
        known = {s.skill_id for s in SKILLS}
        for question in questions:
            self.assertEqual(validate_question(question, known), [])
        executable, rejected_executable = filter_executable_questions(questions)
        self.assertEqual(rejected_executable, [])
        self.assertEqual(len(executable), 42)
        types = {q["Problem Type"] for q in questions}
        self.assertTrue({"mcq_single", "mcq_multi", "code_trace", "fill_blank", "short_answer", "coding"} <= types)

    def test_skill_graph_rejects_cycles_and_missing_prerequisites(self) -> None:
        self.assertEqual(validate_skill_graph(SKILLS), [])
        cyclic = (
            Skill("A", "A", "", "test", "beginner", ("B",)),
            Skill("B", "B", "", "test", "beginner", ("A",)),
        )
        errors = validate_skill_graph(cyclic)
        self.assertTrue(any("cycle" in error for error in errors))
        missing = (Skill("A", "A", "", "test", "beginner", ("MISSING",)),)
        self.assertTrue(any("unknown prerequisite" in error for error in validate_skill_graph(missing)))

    def test_reproducible_template_questions(self) -> None:
        first, rejects_first, _ = generate_template_questions(50, seed=2026)
        second, rejects_second, _ = generate_template_questions(50, seed=2026)
        self.assertEqual(rejects_first, rejects_second)
        self.assertEqual(first, second)

    def test_difficulty_distribution_respects_skill_level_floor(self) -> None:
        questions, rejected, _ = generate_template_questions(81, seed=10)
        self.assertEqual(rejected, [])
        counts = _assign_difficulties(questions, _parse_difficulty_distribution({"beginner": 0.4, "intermediate": 0.4, "advanced": 0.2}), seed=42)
        level_rank = {"beginner": 0, "intermediate": 1, "advanced": 2}
        skill_rank = {skill.skill_id: level_rank[skill.level] for skill in SKILLS}
        for question in questions:
            self.assertGreaterEqual(level_rank[question["difficulty"]], skill_rank[question["primary_skill_id"]])
        self.assertEqual(sum(counts.values()), 81)
        self.assertGreater(counts["advanced"], 0)


class LearnerModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.questions, _, _ = generate_template_questions(35, seed=11)
        cls.profiles = create_student_profiles(6, seed=12)

    def test_response_probability_tracks_mastery(self) -> None:
        profile = self.profiles[1]
        question = next(q for q in self.questions if q["primary_skill_id"] == "SK001")
        low = {s.skill_id: 0.15 for s in SKILLS}
        high = {s.skill_id: 0.90 for s in SKILLS}
        self.assertGreater(response_probability(profile, question, high), response_probability(profile, question, low))

    def test_hint_and_answer_exposure_increase_response_probability(self) -> None:
        profile = self.profiles[2]
        question = next(q for q in self.questions if q["primary_skill_id"] == "SK001")
        mastery = {s.skill_id: 0.55 for s in SKILLS}
        base = response_probability(profile, question, mastery)
        hinted = response_probability(profile, question, mastery, used_hint=True)
        exposed = response_probability(profile, question, mastery, saw_answer=True)
        self.assertGreater(hinted, base)
        self.assertGreater(exposed, base)

    def test_mastery_updates_are_incremental_and_bounded(self) -> None:
        start = 0.45
        correct = update_mastery(start, LearningEvent(True, False, False, 0.25, 0.8, 0.7))
        incorrect = update_mastery(start, LearningEvent(False, False, False, 0.25, 0.8, 0.7))
        solution_view = update_mastery(start, LearningEvent(True, False, True, 0.25, 0.8, 0.7))
        self.assertGreater(correct, start)
        self.assertLess(correct, 1.0)
        self.assertGreater(incorrect, 0.0)
        self.assertLess(incorrect, start + 0.01)
        self.assertGreater(solution_view, start)
        self.assertLess(solution_view, correct)
        self.assertEqual(MASTERY_THRESHOLD, 0.70)

    def test_simulation_reproducible_and_chronological(self) -> None:
        questions = self.questions
        first = simulate_student_interactions(questions, 3, 8, 99, max_attempts=2)
        second = simulate_student_interactions(questions, 3, 8, 99, max_attempts=2)
        self.assertEqual(first, second)
        profiles, interactions, sequence, mastery = first
        self.assertEqual(len(profiles), 3)
        self.assertEqual(len(interactions), 24)
        by_user: dict[str, list[dict[str, str]]] = {}
        for event in interactions:
            by_user.setdefault(event["user_id"], []).append(event)
            self.assertLessEqual(event["introduced_at"], event["start_time"])
            self.assertIn(event["discrete_score"], (0, 1))
        for events in by_user.values():
            ordered = sorted(events, key=lambda row: (row["start_time"], row["id"]))
            self.assertEqual(events, ordered)
        self.assertGreater(len(sequence), 0)
        self.assertGreater(len(mastery), len(interactions))


class EndToEndTests(unittest.TestCase):
    def test_end_to_end_validation_and_leakage_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = dict(DEFAULTS)
            config.update({
                "output_dir": str(Path(temp_dir) / "dataset"),
                "num_students": 4,
                "num_questions": 35,
                "interactions_per_student": 7,
                "seed": 123,
                "generation_mode": "template",
                "max_attempts": 2,
            })
            result = build_dataset(config)
            self.assertEqual(result["validation"]["status"], "PASS", result["validation"]["errors"])
            out = Path(result["output_dir"])
            self.assertTrue((out / "validation_report.json").is_file())
            self.assertTrue((out / "PythonProblems.csv").is_file())
            self.assertTrue((out / "SkillMasteryHistory.csv").is_file())
            self.assertEqual(len(read_csv_rows(out / "PythonInteractions.csv")), 28)
            self.assertEqual(len(read_csv_rows(out / "AgentVisibleInteractions.csv")), 28)

            with (out / "AgentVisibleProblems.csv").open(encoding="utf-8", newline="") as handle:
                visible_problem_fields = set(csv.DictReader(handle).fieldnames or [])
            self.assertNotIn("reference_solution", visible_problem_fields)
            self.assertNotIn("answer_key_json", visible_problem_fields)
            with (out / "AgentVisibleInteractions.csv").open(encoding="utf-8", newline="") as handle:
                visible_interaction_fields = set(csv.DictReader(handle).fieldnames or [])
            self.assertNotIn("misconception_tag", visible_interaction_fields)
            self.assertNotIn("error_type", visible_interaction_fields)

            visible_events = read_csv_rows(out / "AgentVisibleInteractions.csv")
            first = min(visible_events, key=lambda row: row["start_time"])
            history = agent_history_as_of(visible_events, first["user_id"], first["start_time"])
            self.assertTrue(all(row["end_time"] <= first["start_time"] for row in history))
            self.assertNotIn(first["id"], {row["id"] for row in history})

    def test_output_directory_overwrite_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "existing"
            target.mkdir()
            (target / "keep.txt").write_text("do not delete", encoding="utf-8")
            config = dict(DEFAULTS)
            config.update({"output_dir": str(target), "num_students": 1, "num_questions": 5, "interactions_per_student": 1})
            with self.assertRaises(FileExistsError):
                build_dataset(config, overwrite=False)
            self.assertEqual((target / "keep.txt").read_text(encoding="utf-8"), "do not delete")

    def test_total_generation_failure_persists_rejection_and_failure_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(os.environ, {"PYTUTOR_LLM_API_KEY": "test-key"}):
            target = Path(temp_dir) / "failed_dataset"
            config = dict(DEFAULTS)
            config.update({"output_dir": str(target), "num_students": 1, "num_questions": 1,
                           "interactions_per_student": 1, "generation_mode": "llm", "llm_model": "mock-model"})
            generation_report = {"mode": "llm", "requested": 1, "generated": 0, "accepted": 0, "rejected": 1, "seed": 42}
            with patch("generation.build_dataset.generate_llm_questions", return_value=([], [{"candidate_id": "LLM000001", "reason": "mock failure"}], generation_report)):
                with self.assertRaisesRegex(RuntimeError, "reports were saved"):
                    build_dataset(config)
            saved_report = json.loads((target / "validation_report.json").read_text(encoding="utf-8"))
            rejected = read_csv_rows(target / "rejected_questions.csv")
            self.assertEqual(saved_report["status"], "FAIL")
            self.assertEqual(saved_report["checks_failed"], 1)
            self.assertEqual(rejected[0]["candidate_id"], "LLM000001")
            self.assertTrue((target / "dataset_metadata.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
