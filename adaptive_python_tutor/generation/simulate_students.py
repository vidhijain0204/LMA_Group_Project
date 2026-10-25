"""Synthetic student profiles and chronological interaction simulation."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from generation.skills import SKILLS, Skill
from generation.update_mastery import LearningEvent, MASTERY_THRESHOLD, decay_mastery, update_mastery
from utils.code_runner import CodeExecutionError, run_function_tests

STUDENT_COLUMNS = [
    "user_id", "archetype", "initial_mastery_json", "learning_rate", "response_consistency",
    "guessing_tendency", "misconception_tendency", "retention_rate_per_day", "hint_benefit",
    "worked_example_benefit", "hint_tendency", "answer_view_tendency", "transfer_penalty",
    "syntax_error_tendency", "misconception_profile_json", "generation_source",
]
INTERACTION_COLUMNS = [
    "id", "problem_id", "hint_count", "answer_text", "saw_answer", "discrete_score", "end_time", "user_id",
    "start_time", "attempt_number", "code_submission", "test_pass_rate", "tests_passed", "tests_total",
    "error_type", "misconception_tag", "response_time_seconds", "used_hint", "event_type", "generation_source",
    "selected_option_ids", "independent_attempt", "sequence_index", "introduced_at", "scoring_method",
]
AGENT_VISIBLE_INTERACTION_COLUMNS = [
    "id", "problem_id", "hint_count", "answer_text", "saw_answer", "discrete_score", "end_time", "user_id",
    "start_time", "attempt_number", "code_submission", "test_pass_rate", "tests_passed", "tests_total",
    "response_time_seconds", "used_hint", "event_type", "selected_option_ids", "independent_attempt",
    "sequence_index", "introduced_at",
]
LEARNING_SEQUENCE_COLUMNS = ["user_id", "problem_id", "sequence_index", "introduced_at", "primary_skill_id", "generation_source"]
MASTERY_COLUMNS = [
    "user_id", "skill_id", "timestamp", "true_mastery", "mastery_probability", "observation_count", "learning_events_count",
]

ARCHETYPES = (
    "beginner_gaps", "strong_fundamentals_weak_control_flow", "syntax_sensitive",
    "transfer_gap", "uneven_profile", "advanced_learner",
)
KNOWN_MISCONCEPTIONS = (
    "off_by_one", "assignment_rebinding", "comparison_logic", "type_conversion", "return_vs_print",
    "zero_based_indexing", "mutable_aliasing", "local_scope", "conditional_boundary", "range_stop_exclusive",
    "exception_type", "recursion_base_case", "comprehension_filter", "operator_precedence", "truthiness",
)


@dataclass
class StudentProfile:
    user_id: str
    archetype: str
    initial_mastery: dict[str, float]
    learning_rate: float
    response_consistency: float
    guessing_tendency: float
    misconception_tendency: float
    retention_rate_per_day: float
    hint_benefit: float
    worked_example_benefit: float
    hint_tendency: float
    answer_view_tendency: float
    transfer_penalty: float
    syntax_error_tendency: float
    misconception_profile: dict[str, float]

    def to_row(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "archetype": self.archetype,
            "initial_mastery_json": json.dumps(self.initial_mastery, sort_keys=True, separators=(",", ":")),
            "learning_rate": round(self.learning_rate, 6),
            "response_consistency": round(self.response_consistency, 6),
            "guessing_tendency": round(self.guessing_tendency, 6),
            "misconception_tendency": round(self.misconception_tendency, 6),
            "retention_rate_per_day": round(self.retention_rate_per_day, 6),
            "hint_benefit": round(self.hint_benefit, 6),
            "worked_example_benefit": round(self.worked_example_benefit, 6),
            "hint_tendency": round(self.hint_tendency, 6),
            "answer_view_tendency": round(self.answer_view_tendency, 6),
            "transfer_penalty": round(self.transfer_penalty, 6),
            "syntax_error_tendency": round(self.syntax_error_tendency, 6),
            "misconception_profile_json": json.dumps(self.misconception_profile, sort_keys=True, separators=(",", ":")),
            "generation_source": "simulator",
        }


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def create_student_profiles(num_students: int, seed: int, skills: tuple[Skill, ...] = SKILLS) -> list[StudentProfile]:
    """Create heterogeneous synthetic profiles with protected initial mastery."""
    if num_students < 1:
        raise ValueError("num_students must be at least 1")
    rng = random.Random(seed)
    profiles: list[StudentProfile] = []
    for i in range(num_students):
        archetype = ARCHETYPES[i % len(ARCHETYPES)]
        initial: dict[str, float] = {}
        for skill in skills:
            beginner_level = skill.level == "beginner"
            if archetype == "beginner_gaps":
                score = rng.uniform(0.08, 0.40)
            elif archetype == "strong_fundamentals_weak_control_flow":
                weak = skill.skill_id in {"SK009", "SK010", "SK011", "SK012", "SK017", "SK018", "SK021"}
                score = rng.uniform(0.15, 0.48) if weak else rng.uniform(0.67, 0.93) if beginner_level else rng.uniform(0.42, 0.72)
            elif archetype == "syntax_sensitive":
                score = rng.uniform(0.60, 0.88)
            elif archetype == "transfer_gap":
                score = rng.uniform(0.62, 0.88) if beginner_level else rng.uniform(0.42, 0.78)
            elif archetype == "uneven_profile":
                score = rng.uniform(0.08, 0.96)
            else:  # advanced learner
                score = rng.uniform(0.76, 0.97) if not beginner_level else rng.uniform(0.86, 0.99)
            initial[skill.skill_id] = round(score, 4)
        if archetype == "syntax_sensitive":
            response_consistency = rng.uniform(0.72, 0.90)
            syntax_error = rng.uniform(0.35, 0.60)
        else:
            response_consistency = rng.uniform(0.68, 0.96)
            syntax_error = rng.uniform(0.04, 0.22)
        if archetype == "transfer_gap":
            transfer_penalty = rng.uniform(0.22, 0.38)
        else:
            transfer_penalty = rng.uniform(0.02, 0.16)
        misconception_profile = {tag: round(rng.uniform(0.04, 0.42), 3) for tag in KNOWN_MISCONCEPTIONS}
        if archetype == "uneven_profile":
            for tag in rng.sample(list(KNOWN_MISCONCEPTIONS), k=3):
                misconception_profile[tag] = round(rng.uniform(0.45, 0.85), 3)
        if archetype == "beginner_gaps":
            for tag in rng.sample(list(KNOWN_MISCONCEPTIONS), k=4):
                misconception_profile[tag] = round(rng.uniform(0.45, 0.75), 3)
        profiles.append(StudentProfile(
            user_id=f"STU{i + 1:05d}", archetype=archetype, initial_mastery=initial,
            learning_rate=rng.uniform(0.10, 0.34), response_consistency=response_consistency,
            guessing_tendency=rng.uniform(0.08, 0.35), misconception_tendency=rng.uniform(0.12, 0.55),
            retention_rate_per_day=rng.uniform(0.965, 0.999), hint_benefit=rng.uniform(0.25, 0.95),
            worked_example_benefit=rng.uniform(0.20, 0.90), hint_tendency=rng.uniform(0.15, 0.70),
            answer_view_tendency=rng.uniform(0.02, 0.12), transfer_penalty=transfer_penalty,
            syntax_error_tendency=syntax_error, misconception_profile=misconception_profile,
        ))
    return profiles


def response_probability(
    profile: StudentProfile, question: dict[str, Any], mastery: dict[str, float],
    *, used_hint: bool = False, saw_answer: bool = False, prior_example_exposures: int = 0,
) -> float:
    """Return P(correct) using a documented probabilistic learner model.

    Ability is 72% primary-skill mastery and 28% mean secondary-skill mastery.
    Difficulty and relevant misconception evidence reduce ability. Response
    consistency mixes ability with a format-sensitive guessing baseline.
    """
    primary = str(question["primary_skill_id"])
    secondary = json.loads(question.get("secondary_skill_ids", "[]") or "[]")
    primary_mastery = mastery.get(primary, 0.0)
    secondary_mean = sum(mastery.get(s, 0.0) for s in secondary) / len(secondary) if secondary else primary_mastery
    combined_mastery = 0.72 * primary_mastery + 0.28 * secondary_mean
    difficulty_penalty = {"beginner": 0.04, "intermediate": 0.14, "advanced": 0.24}.get(question.get("difficulty"), 0.12)
    tags = json.loads(question.get("misconception_tags", "[]") or "[]")
    tag_match = max((profile.misconception_profile.get(tag, 0.10) for tag in tags), default=0.10)
    misconception_penalty = 0.14 * profile.misconception_tendency * tag_match
    syntax_penalty = 0.0
    if question.get("Problem Type") in {"coding", "debugging", "fill_blank"}:
        syntax_penalty = 0.22 * profile.syntax_error_tendency
    transfer_penalty = profile.transfer_penalty if question.get("Problem Type") in {"coding", "debugging"} else 0.0
    ability = _clamp(0.12 + 0.84 * combined_mastery - difficulty_penalty - misconception_penalty - syntax_penalty - transfer_penalty, 0.02, 0.98)
    qtype = question.get("Problem Type")
    if qtype == "mcq_single":
        try:
            guessing_baseline = 1.0 / max(2, len(json.loads(question.get("Multiple Choice Options", "[]"))))
        except (ValueError, TypeError):
            guessing_baseline = 0.25
    elif qtype == "mcq_multi":
        guessing_baseline = 0.08
    else:
        guessing_baseline = 0.025
    guessing_baseline = _clamp(guessing_baseline + 0.10 * profile.guessing_tendency, 0.02, 0.50)
    probability = profile.response_consistency * ability + (1.0 - profile.response_consistency) * guessing_baseline
    if used_hint:
        probability += 0.06 * profile.hint_benefit
    if saw_answer:
        probability += 0.18 * profile.worked_example_benefit
    if prior_example_exposures:
        probability += min(prior_example_exposures, 3) * 0.015 * profile.worked_example_benefit
    return _clamp(probability, 0.02, 0.98)


def _normalise_answer(value: str) -> str:
    return " ".join(value.strip().replace("\r\n", "\n").split()).casefold().strip(" .\n\t")


def _parse_json_field(question: dict[str, Any], field: str, default: Any) -> Any:
    value = question.get(field, "")
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _wrong_fill_answer(question: dict[str, Any], rng: random.Random) -> str:
    sid = question["primary_skill_id"]
    mapping = {
        "SK001": "0", "SK002": "type", "SK003": "str", "SK004": "/", "SK005": "or",
        "SK006": "upper", "SK007": "input", "SK008": "or", "SK009": "0", "SK010": "stop + 1",
        "SK011": "-=", "SK012": "1", "SK013": "1", "SK014": "stop + 1", "SK015": "list",
        "SK016": "'missing'", "SK017": "left", "SK018": "print", "SK019": "list", "SK020": "KeyError",
        "SK021": "n", "SK022": "close", "SK023": "self", "SK024": "extend", "SK025": "alias",
        "SK026": "=", "SK027": "O(1)",
    }
    return mapping.get(sid, "None")


def _wrong_short_answer(question: dict[str, Any], rng: random.Random) -> str:
    mapping = {
        "SK001": "Assignment compares two values but does not update a name.",
        "SK002": "Every value is stored as a string.",
        "SK003": "Type conversion always happens automatically and cannot fail.",
        "SK004": "The percent operator means exponentiation.",
        "SK005": "and is true when either operand is true.",
        "SK006": "Strings are mutable sequences and lower changes the object in place.",
        "SK007": "input returns an integer and print returns the displayed text.",
        "SK008": "A Boolean expression can only evaluate to a number.",
        "SK009": "Both the if branch and else branch always execute.",
        "SK010": "range includes the stop value and the accumulator resets every iteration.",
        "SK011": "A while loop always runs at least once and terminates automatically.",
        "SK012": "range includes the stop value by default.",
        "SK013": "The first list element has index one.",
        "SK014": "A slice always includes its stop index.",
        "SK015": "Sets preserve duplicates and tuples support in-place item assignment.",
        "SK016": "Accessing any missing dictionary key returns zero.",
        "SK017": "A parameter is a global variable defined by the caller.",
        "SK018": "print returns the displayed value to the caller.",
        "SK019": "A list comprehension always creates a dictionary.",
        "SK020": "Every runtime error is a SyntaxError.",
        "SK021": "A recursive function does not need a base case.",
        "SK022": "A context manager leaves the file open after the block.",
        "SK023": "self refers to the class definition, never an instance.",
        "SK024": "All Python objects are immutable.",
        "SK025": "Assigning alias = original always copies the list.",
        "SK026": "A correct algorithm can ignore the problem's state and stopping condition.",
        "SK027": "Big-O describes exact wall-clock time for any input.",
    }
    return mapping.get(question["primary_skill_id"], "Python automatically converts all values to strings.")


def _choose_misconception(profile: StudentProfile, question: dict[str, Any], rng: random.Random) -> str:
    tags = _parse_json_field(question, "misconception_tags", [])
    if not tags:
        return "general_concept_confusion"
    weights = [0.15 + profile.misconception_profile.get(tag, 0.10) for tag in tags]
    return rng.choices(tags, weights=weights, k=1)[0]


def _objective_score(question: dict[str, Any], response: dict[str, Any]) -> tuple[int | None, float | None, int | None, int | None, str]:
    """Score the observed response; coding items are evaluated against tests."""
    qtype = question["Problem Type"]
    if qtype in {"mcq_single", "mcq_multi"}:
        expected = sorted(_parse_json_field(question, "Multiple Choice Answers", []))
        observed = sorted(response.get("selected_option_ids", []))
        return int(expected == observed), None, None, None, "option_key"
    if qtype == "code_trace":
        return int(_normalise_answer(response["answer_text"]) == _normalise_answer(question.get("expected_output", ""))), None, None, None, "stdout_exact"
    if qtype in {"fill_blank", "debugging", "short_answer"}:
        expected = _parse_json_field(question, "Fill-in Answers", [])
        accepted = {_normalise_answer(str(x)) for x in expected}
        return int(_normalise_answer(response["answer_text"]) in accepted), None, None, None, "normalized_exact"
    if qtype == "coding":
        tests = _parse_json_field(question, "test_cases_json", [])
        try:
            passed, total, _details = run_function_tests(
                response.get("code_submission", ""), question.get("function_name", "solve"), tests, trusted=question.get("generation_source", "template") == "template",
            )
        except (CodeExecutionError, ValueError) as exc:
            passed, total = 0, len(tests)
        rate = passed / total if total else 0.0
        return int(total > 0 and passed == total), rate, passed, total, "executable_tests"
    return None, None, None, None, "unscored"


def _response_for_question(
    question: dict[str, Any], intend_correct: bool, rng: random.Random, profile: StudentProfile,
) -> dict[str, Any]:
    """Produce a response from deterministic, prevalidated answer templates."""
    qtype = question["Problem Type"]
    options = _parse_json_field(question, "Multiple Choice Options", [])
    correct_ids = _parse_json_field(question, "Multiple Choice Answers", [])
    fill_answers = _parse_json_field(question, "Fill-in Answers", [])
    templates = _parse_json_field(question, "response_templates_json", {})
    wrong_answers = templates.get("wrong_answers", []) if isinstance(templates, dict) else []
    response: dict[str, Any] = {"answer_text": "", "selected_option_ids": [], "code_submission": ""}
    if qtype in {"mcq_single", "mcq_multi"}:
        if intend_correct:
            selected = correct_ids[:]
        else:
            wrong_ids = [o["id"] for o in options if o.get("id") not in correct_ids]
            if qtype == "mcq_single":
                selected = [rng.choice(wrong_ids)] if wrong_ids else correct_ids[:]
            else:
                # At least one incorrect and not exactly the correct set.
                selected = [rng.choice(wrong_ids)] if wrong_ids else []
                if correct_ids and rng.random() < 0.25:
                    selected.append(rng.choice(correct_ids))
        by_id = {o["id"]: o["text"] for o in options}
        response["selected_option_ids"] = selected
        response["answer_text"] = " | ".join(str(by_id.get(item, item)) for item in selected)
    elif qtype == "code_trace":
        correct = question.get("expected_output", "")
        if intend_correct:
            response["answer_text"] = correct
        else:
            choices = [str(x) for x in wrong_answers if str(x) != correct]
            if choices:
                response["answer_text"] = rng.choice(choices)
            elif correct.strip() in {"True", "False"}:
                response["answer_text"] = "False" if correct.strip() == "True" else "True"
            elif correct.strip().lstrip("-").isdigit():
                response["answer_text"] = str(int(correct) + rng.choice([-1, 1]))
            else:
                response["answer_text"] = correct + "_"
    elif qtype == "fill_blank":
        correct = str(fill_answers[0]) if fill_answers else ""
        response["answer_text"] = correct if intend_correct else _wrong_fill_answer(question, rng)
    elif qtype == "debugging":
        if intend_correct:
            response["answer_text"] = str(fill_answers[0]) if fill_answers else ""
        else:
            buggy_lines = str(templates.get("buggy_code", "")).splitlines() if isinstance(templates, dict) else []
            fixed_lines = str(templates.get("fixed_code", "")).splitlines() if isinstance(templates, dict) else []
            differing_lines = [bad for index, bad in enumerate(buggy_lines) if index >= len(fixed_lines) or bad != fixed_lines[index]]
            response["answer_text"] = differing_lines[0].strip() if differing_lines else "if value = value:"
    elif qtype == "short_answer":
        response["answer_text"] = (str(fill_answers[0]) if fill_answers else "") if intend_correct else _wrong_short_answer(question, rng)
    elif qtype == "coding":
        if intend_correct:
            response["code_submission"] = question.get("reference_solution", "")
        else:
            response["code_submission"] = templates.get("buggy_solution", "") if isinstance(templates, dict) else ""
            if not response["code_submission"]:
                function_name = question.get("function_name") or "solve"
                response["code_submission"] = f"def {function_name}(*args, **kwargs):\n    return None\n"
        response["answer_text"] = "Code submitted"
    else:
        response["answer_text"] = ""
    return response


def _pick_question(
    questions: list[dict[str, Any]], mastery: dict[str, float], seen: set[str], rng: random.Random,
) -> dict[str, Any]:
    """Choose a problem with an adaptive weakness signal plus exploration."""
    unseen = [q for q in questions if q["problem_id"] not in seen]
    pool = unseen or questions
    weights: list[float] = []
    for question in pool:
        primary = question["primary_skill_id"]
        uncertainty = 4.0 * mastery.get(primary, 0.5) * (1.0 - mastery.get(primary, 0.5))
        weakness = 1.0 - mastery.get(primary, 0.5)
        weights.append(0.15 + 0.58 * weakness + 0.20 * uncertainty + rng.random() * 0.12)
    return rng.choices(pool, weights=weights, k=1)[0]


def _snapshot_mastery(
    user_id: str, timestamp: str, mastery: dict[str, float], observations: dict[str, int], learning_events: dict[str, int],
    skills: tuple[Skill, ...] = SKILLS,
) -> list[dict[str, Any]]:
    return [
        {
            "user_id": user_id,
            "skill_id": skill.skill_id,
            "timestamp": timestamp,
            "true_mastery": int(mastery[skill.skill_id] >= MASTERY_THRESHOLD),
            "mastery_probability": round(mastery[skill.skill_id], 6),
            "observation_count": observations[skill.skill_id],
            "learning_events_count": learning_events[skill.skill_id],
        }
        for skill in skills
    ]


def simulate_student_interactions(
    questions: list[dict[str, Any]], num_students: int, interactions_per_student: int, seed: int,
    *, max_attempts: int = 2, skills: tuple[Skill, ...] = SKILLS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return hidden profiles, interaction rows, sequence rows, and mastery snapshots."""
    if not questions:
        raise ValueError("At least one accepted question is required for simulation")
    if interactions_per_student < 1:
        raise ValueError("interactions_per_student must be at least 1")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    rng = random.Random(seed)
    profiles = create_student_profiles(num_students, seed + 101, skills)
    question_by_id = {q["problem_id"]: q for q in questions}
    interaction_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    mastery_rows: list[dict[str, Any]] = []
    for student_idx, profile in enumerate(profiles):
        mastery = dict(profile.initial_mastery)
        observations = {skill.skill_id: 0 for skill in skills}
        learning_event_counts = {skill.skill_id: 0 for skill in skills}
        base_time = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=rng.randint(0, 120), minutes=rng.randint(0, 1439))
        current_time = base_time
        previous_end = current_time
        snapshot_time = current_time.isoformat(timespec="seconds")
        mastery_rows.extend(_snapshot_mastery(profile.user_id, snapshot_time, mastery, observations, learning_event_counts, skills))
        seen_questions: set[str] = set()
        prior_exposure_by_skill: dict[str, int] = {}
        # Retry state is scoped to one assigned learning-sequence item.
        retry_queue: tuple[str, int, str, int] | None = None
        student_sequence = 0

        for event_index in range(interactions_per_student):
            if retry_queue is not None:
                problem_id, sequence_index, introduced_at, attempt_number = retry_queue
                question = question_by_id[problem_id]
                retry_queue = None
                is_new_assignment = False
            else:
                question = _pick_question(questions, mastery, seen_questions, rng)
                problem_id = question["problem_id"]
                attempt_number = 1
                is_new_assignment = True

            # Advance the synthetic clock; all recorded times are timezone-aware UTC.
            if event_index > 0:
                current_time = previous_end + timedelta(minutes=rng.randint(10, 180))
            start_time = current_time
            if is_new_assignment:
                student_sequence += 1
                sequence_index = student_sequence
                introduced_at = start_time.isoformat(timespec="seconds")
                seen_questions.add(problem_id)
                sequence_rows.append({
                    "user_id": profile.user_id, "problem_id": problem_id, "sequence_index": sequence_index,
                    "introduced_at": introduced_at, "primary_skill_id": question["primary_skill_id"], "generation_source": "simulator",
                })
            used_hint = rng.random() < profile.hint_tendency
            hint_count = rng.randint(1, 2) if used_hint else 0
            saw_answer = rng.random() < profile.answer_view_tendency
            p_correct = response_probability(
                profile, question, mastery, used_hint=used_hint, saw_answer=saw_answer,
                prior_example_exposures=prior_exposure_by_skill.get(question["primary_skill_id"], 0),
            )
            intended_correct = rng.random() < p_correct
            response = _response_for_question(question, intended_correct, rng, profile)
            score, pass_rate, tests_passed, tests_total, scoring_method = _objective_score(question, response)
            # Objective evaluation, rather than the latent intent, determines correctness.
            correct = bool(score == 1)
            elapsed_seconds = int(max(8, rng.gauss(75 + 60 * (1.0 - mastery[question["primary_skill_id"]]), 24)))
            end_time = start_time + timedelta(seconds=elapsed_seconds)
            tag = _choose_misconception(profile, question, rng) if not correct else ""
            error_type = tag if not correct else ""
            primary = question["primary_skill_id"]
            observations[primary] += 1
            covered = [primary] + [s for s in json.loads(question.get("secondary_skill_ids", "[]") or "[]") if s in mastery and s != primary]
            elapsed_hours = max(0.0, (end_time - previous_end).total_seconds() / 3600.0)
            for skill_id in mastery:
                mastery[skill_id] = decay_mastery(mastery[skill_id], profile.retention_rate_per_day, elapsed_hours)
            for covered_skill in covered:
                learning_event_counts[covered_skill] += 1
                event = LearningEvent(
                    correct=correct, used_hint=used_hint, saw_answer=saw_answer,
                    learning_rate=profile.learning_rate * (1.0 if covered_skill == primary else 0.35),
                    hint_benefit=profile.hint_benefit, worked_example_benefit=profile.worked_example_benefit,
                )
                mastery[covered_skill] = update_mastery(mastery[covered_skill], event)
            if saw_answer:
                prior_exposure_by_skill[primary] = prior_exposure_by_skill.get(primary, 0) + 1

            interaction_id = f"INT{student_idx + 1:05d}_{event_index + 1:05d}"
            interaction_rows.append({
                "id": interaction_id,
                "problem_id": problem_id,
                "hint_count": hint_count,
                "answer_text": response["answer_text"],
                "saw_answer": int(saw_answer),
                "discrete_score": score,
                "end_time": end_time.isoformat(timespec="seconds"),
                "user_id": profile.user_id,
                "start_time": start_time.isoformat(timespec="seconds"),
                "attempt_number": attempt_number,
                "code_submission": response["code_submission"],
                "test_pass_rate": "" if pass_rate is None else round(pass_rate, 6),
                "tests_passed": "" if tests_passed is None else tests_passed,
                "tests_total": "" if tests_total is None else tests_total,
                "error_type": error_type,
                "misconception_tag": tag,
                "response_time_seconds": elapsed_seconds,
                "used_hint": int(used_hint),
                "event_type": "answer_submitted",
                "generation_source": "simulator",
                "selected_option_ids": json.dumps(response["selected_option_ids"], separators=(",", ":")),
                "independent_attempt": int(not saw_answer),
                "sequence_index": sequence_index,
                "introduced_at": introduced_at,
                "scoring_method": scoring_method,
            })
            snapshot_time = end_time.isoformat(timespec="seconds")
            mastery_rows.extend(_snapshot_mastery(profile.user_id, snapshot_time, mastery, observations, learning_event_counts, skills))
            previous_end = end_time

            # Retry wrong answers occasionally, unless the solution was already viewed.
            if (not correct and not saw_answer and attempt_number < max_attempts and rng.random() < 0.24):
                retry_queue = (problem_id, sequence_index, introduced_at, attempt_number + 1)

    interaction_rows.sort(key=lambda row: (row["end_time"], row["user_id"], row["attempt_number"], row["id"]))
    mastery_rows.sort(key=lambda row: (row["user_id"], row["timestamp"], row["skill_id"]))
    sequence_rows.sort(key=lambda row: (row["user_id"], row["sequence_index"]))
    return [profile.to_row() for profile in profiles], interaction_rows, sequence_rows, mastery_rows
