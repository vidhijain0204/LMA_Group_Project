"""Semantic and executable quality checks for generated question records."""
from __future__ import annotations

import json
from typing import Any

from utils.code_runner import CodeExecutionError, run_function_tests, run_snippet


def validate_executable_question(question: dict[str, Any]) -> list[str]:
    """Check Python syntax/output and coding test consistency where applicable."""
    errors: list[str] = []
    qtype = question.get("Problem Type")
    try:
        if qtype in {"mcq_single", "code_trace", "fill_blank"}:
            source = question.get("trace_code", "") if qtype == "code_trace" else question.get("reference_solution", "")
            if not source.strip():
                errors.append(f"{qtype} requires a non-empty executable reference snippet")
            else:
                actual = run_snippet(source, trusted=question.get("generation_source", "template") == "template")
                expected = str(question.get("expected_output", "")).rstrip("\n")
                if actual != expected:
                    errors.append(f"executable output mismatch: expected {expected!r}, got {actual!r}")
                if qtype == "mcq_single":
                    options = json.loads(question["Multiple Choice Options"])
                    answers = json.loads(question["Multiple Choice Answers"])
                    answer_texts = [str(option["text"]) for option in options if option["id"] in answers]
                    if expected not in answer_texts:
                        errors.append("mcq_single answer key does not point to option matching executable output")
            if qtype == "code_trace":
                key = json.loads(question.get("answer_key_json", "{}"))
                if isinstance(key, dict) and key.get("expected_output") != question.get("expected_output", ""):
                    errors.append("code_trace answer_key_json expected_output must match expected_output")
        elif qtype == "debugging":
            compile(question.get("reference_solution", ""), f"<{question.get('problem_id', 'question')}>", "exec")
            answers = json.loads(question.get("Fill-in Answers", "[]"))
            if answers and str(answers[0]).strip() not in [line.strip() for line in question.get("reference_solution", "").splitlines()]:
                errors.append("debugging accepted corrected line must occur in reference_solution")
        elif qtype == "coding":
            tests = json.loads(question.get("test_cases_json", "[]"))
            passed, total, _details = run_function_tests(
                question.get("reference_solution", ""), question.get("function_name", "solve"), tests, trusted=question.get("generation_source", "template") == "template",
            )
            if total == 0 or passed != total:
                errors.append(f"reference solution passed {passed}/{total} executable tests")
    except (CodeExecutionError, SyntaxError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"executable validation failed: {type(exc).__name__}: {exc}")
    return errors


def filter_executable_questions(
    questions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return accepted questions and an explicit list of semantic rejections."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for question in questions:
        errors = validate_executable_question(question)
        if errors:
            rejected.append({"candidate_id": question.get("problem_id", "unknown"), "reason": "; ".join(errors)})
        else:
            accepted.append(question)
    return accepted, rejected
