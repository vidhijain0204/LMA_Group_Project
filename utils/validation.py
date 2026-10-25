"""Output data-integrity checks and leakage-safe history access helpers."""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from generation.generate_questions import AGENT_VISIBLE_PROBLEM_COLUMNS, PROBLEM_COLUMNS, validate_question
from generation.skills import SKILLS, validate_skill_graph
from generation.simulate_students import (
    AGENT_VISIBLE_INTERACTION_COLUMNS,
    INTERACTION_COLUMNS,
    LEARNING_SEQUENCE_COLUMNS,
    MASTERY_COLUMNS,
    STUDENT_COLUMNS,
)


EXPECTED_COLUMNS: dict[str, list[str]] = {
    "PythonProblems.csv": PROBLEM_COLUMNS,
    "AgentVisibleProblems.csv": AGENT_VISIBLE_PROBLEM_COLUMNS,
    "PythonSkills.csv": ["skill_id", "skill_name", "description", "category", "level", "prerequisite_skill_ids"],
    "SkillPrerequisites.csv": ["skill_id", "prerequisite_skill_id"],
    "Students.csv": STUDENT_COLUMNS,
    "PythonInteractions.csv": INTERACTION_COLUMNS,
    "AgentVisibleInteractions.csv": AGENT_VISIBLE_INTERACTION_COLUMNS,
    "LearningSequence.csv": LEARNING_SEQUENCE_COLUMNS,
    "SkillMasteryHistory.csv": MASTERY_COLUMNS,
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read UTF-8 CSV rows as dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def agent_history_as_of(
    interaction_rows: Iterable[dict[str, Any]], user_id: str, timestamp: str,
) -> list[dict[str, Any]]:
    """Return only agent-visible events for one user whose end_time is <= cutoff.

    Call this helper at each decision timestamp; the complete CSV contains later
    rows for offline evaluation and must not be sent to an agent wholesale.
    """
    cutoff = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    filtered: list[dict[str, Any]] = []
    for row in interaction_rows:
        if str(row.get("user_id")) != user_id:
            continue
        end = datetime.fromisoformat(str(row["end_time"]).replace("Z", "+00:00"))
        if end <= cutoff:
            safe_row = {key: value for key, value in row.items() if key not in {"error_type", "misconception_tag", "scoring_method"}}
            filtered.append(safe_row)
    filtered.sort(key=lambda row: (row["end_time"], row["id"]))
    return filtered


def validate_dataset_files(output_dir: str | Path) -> dict[str, Any]:
    """Validate schemas, identifiers, references, time ordering, JSON, and leakage."""
    root = Path(output_dir)
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    stats: dict[str, Any] = {}

    def record(name: str, passed: bool, message: str, count: int | None = None) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "message": message, "count": count})
        if not passed:
            errors.append(f"{name}: {message}")

    tables: dict[str, list[dict[str, str]]] = {}
    for filename, expected in EXPECTED_COLUMNS.items():
        path = root / filename
        if not path.is_file():
            record(f"file_exists:{filename}", False, "required CSV was not written")
            continue
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames or []
                rows = list(reader)
            tables[filename] = rows
            record(f"schema:{filename}", columns == expected, f"expected ordered columns {expected}; found {columns}", len(rows))
            stats[filename] = {"rows": len(rows), "columns": columns}
        except (OSError, csv.Error) as exc:
            record(f"read_csv:{filename}", False, str(exc))

    graph_errors = validate_skill_graph(SKILLS)
    record("skill_graph_acyclic_and_referential", not graph_errors, "; ".join(graph_errors) if graph_errors else "no missing references or cycles", len(SKILLS))
    skill_ids = {skill.skill_id for skill in SKILLS}
    problem_rows = tables.get("PythonProblems.csv", [])
    problem_ids = [row.get("problem_id", "") for row in problem_rows]
    record("problem_ids_unique", len(problem_ids) == len(set(problem_ids)) and all(problem_ids), "problem IDs must be non-empty and unique", len(problem_ids))
    question_errors: list[str] = []
    for row in problem_rows:
        schema_errors = validate_question(row, skill_ids)
        if schema_errors:
            question_errors.append(f"{row.get('problem_id')}: {'; '.join(schema_errors)}")
    record("question_schema_and_answer_keys", not question_errors, "; ".join(question_errors[:20]) if question_errors else "all question records passed schema checks", len(problem_rows))
    skills_rows = tables.get("PythonSkills.csv", [])
    file_skill_ids = {row.get("skill_id", "") for row in skills_rows}
    record("skill_ids_consistent", file_skill_ids == skill_ids, "PythonSkills.csv should match the canonical skill IDs", len(file_skill_ids))

    edge_rows = tables.get("SkillPrerequisites.csv", [])
    edge_set = {(row.get("skill_id", ""), row.get("prerequisite_skill_id", "")) for row in edge_rows}
    expected_edges = {(skill.skill_id, pre) for skill in SKILLS for pre in skill.prerequisite_skill_ids}
    record("prerequisite_edges_consistent", edge_set == expected_edges, "edge table must exactly represent the declared prerequisite graph", len(edge_rows))

    student_rows = tables.get("Students.csv", [])
    student_ids = [row.get("user_id", "") for row in student_rows]
    record("student_ids_unique", len(student_ids) == len(set(student_ids)) and all(student_ids), "student IDs must be non-empty and unique", len(student_ids))
    for row in student_rows:
        try:
            initial = json.loads(row["initial_mastery_json"])
            profile = json.loads(row["misconception_profile_json"])
            if set(initial) != skill_ids or any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in initial.values()):
                errors.append(f"{row['user_id']}: invalid initial mastery mapping")
            if not isinstance(profile, dict):
                errors.append(f"{row['user_id']}: misconception profile must be JSON object")
        except (json.JSONDecodeError, KeyError) as exc:
            errors.append(f"{row.get('user_id')}: malformed hidden student profile: {exc}")
    record("student_profile_json", not any("invalid initial mastery" in x or "misconception profile" in x or "malformed hidden student" in x for x in errors), "hidden profiles contain a probability for each skill", len(student_rows))

    interactions = tables.get("PythonInteractions.csv", [])
    visible_interactions = tables.get("AgentVisibleInteractions.csv", [])
    interaction_ids = [row.get("id", "") for row in interactions]
    record("interaction_ids_unique", len(interaction_ids) == len(set(interaction_ids)) and all(interaction_ids), "interaction IDs must be non-empty and unique", len(interactions))
    for row in interactions:
        if row.get("problem_id") not in set(problem_ids):
            errors.append(f"{row.get('id')}: unknown problem_id {row.get('problem_id')}")
        if row.get("user_id") not in set(student_ids):
            errors.append(f"{row.get('id')}: unknown user_id {row.get('user_id')}")
        if row.get("discrete_score") not in {"0", "1", ""}:
            errors.append(f"{row.get('id')}: discrete_score must be 0, 1, or empty")
        try:
            datetime.fromisoformat(row["start_time"].replace("Z", "+00:00"))
            datetime.fromisoformat(row["end_time"].replace("Z", "+00:00"))
            if datetime.fromisoformat(row["end_time"].replace("Z", "+00:00")) < datetime.fromisoformat(row["start_time"].replace("Z", "+00:00")):
                errors.append(f"{row.get('id')}: end_time precedes start_time")
            json.loads(row["selected_option_ids"] or "[]")
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"{row.get('id')}: malformed interaction time/JSON: {exc}")
    record("interaction_referential_integrity_and_fields", not any(": unknown problem_id" in x or ": unknown user_id" in x or "discrete_score must" in x or "end_time precedes" in x or "malformed interaction" in x for x in errors), "references, scores, timestamps, and selected option JSON are valid", len(interactions))

    seen_by_user: dict[str, set[str]] = {}
    previous_by_user: dict[str, str] = {}
    for row in sorted(interactions, key=lambda r: (r.get("user_id", ""), r.get("start_time", ""), r.get("id", ""))):
        uid, pid = row.get("user_id", ""), row.get("problem_id", "")
        previous = previous_by_user.get(uid)
        if previous and row.get("end_time", "") < previous:
            errors.append(f"{row.get('id')}: per-student chronology is not ordered")
        previous_by_user[uid] = row.get("end_time", "")
        introduced = row.get("introduced_at", "")
        try:
            if datetime.fromisoformat(introduced.replace("Z", "+00:00")) > datetime.fromisoformat(row["start_time"].replace("Z", "+00:00")):
                errors.append(f"{row.get('id')}: interaction occurred before question introduction")
        except (ValueError, KeyError):
            pass
        seen_by_user.setdefault(uid, set()).add(pid)
    record("chronology_and_question_introduction", not any("per-student chronology" in x or "before question introduction" in x for x in errors), "per-user event times are ordered and question introduction is not in the future", len(interactions))

    mastery_rows = tables.get("SkillMasteryHistory.csv", [])
    invalid_mastery: list[str] = []
    for row in mastery_rows:
        try:
            value = float(row["mastery_probability"])
            binary = int(row["true_mastery"])
            if not 0 <= value <= 1 or binary not in (0, 1) or row["skill_id"] not in skill_ids or row["user_id"] not in set(student_ids):
                invalid_mastery.append(f"{row.get('user_id')}/{row.get('skill_id')}")
            datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        except (ValueError, KeyError) as exc:
            invalid_mastery.append(f"{row.get('user_id')}: {exc}")
    record("mastery_history_valid", not invalid_mastery, "ground-truth mastery values are bounded and referentially valid", len(mastery_rows))

    # Data supplied to downstream agents must not contain target labels or answer keys.
    forbidden_problem_fields = {"reference_solution", "answer_key_json", "Fill-in Answers", "misconception_tags", "response_templates_json"}
    visible_problem_rows = tables.get("AgentVisibleProblems.csv", [])
    visible_problem_path = root / "AgentVisibleProblems.csv"
    if visible_problem_path.is_file():
        with visible_problem_path.open("r", encoding="utf-8-sig", newline="") as handle:
            visible_problem_columns = set(csv.DictReader(handle).fieldnames or [])
    else:
        visible_problem_columns = set()
    record("answer_key_leakage_prevention", not bool(forbidden_problem_fields & visible_problem_columns), "agent-visible problem schema excludes answer keys and simulator labels", len(visible_problem_rows))
    forbidden_interaction_fields = {"error_type", "misconception_tag", "scoring_method"}
    visible_interaction_path = root / "AgentVisibleInteractions.csv"
    if visible_interaction_path.is_file():
        with visible_interaction_path.open("r", encoding="utf-8-sig", newline="") as handle:
            visible_interaction_columns = set(csv.DictReader(handle).fieldnames or [])
    else:
        visible_interaction_columns = set()
    record("misconception_label_leakage_prevention", not bool(forbidden_interaction_fields & visible_interaction_columns), "agent-visible interaction schema excludes simulator-inferred error labels", len(visible_interactions))

    # Duplicates / distribution checks are diagnostics, not hard failures for tiny samples.
    type_counts = Counter(row.get("Problem Type", "") for row in problem_rows)
    difficulty_counts = Counter(row.get("difficulty", "") for row in problem_rows)
    skill_counts = Counter(row.get("primary_skill_id", "") for row in problem_rows)
    stats["question_type_counts"] = dict(type_counts)
    stats["difficulty_counts"] = dict(difficulty_counts)
    stats["primary_skill_counts"] = dict(skill_counts)
    stats["mastery_threshold"] = 0.70
    checks.append({"name": "distribution_summary", "status": "INFO", "message": "Question distribution is available for auditing; sparse skills are not an automatic failure for small samples.", "count": len(problem_rows)})
    unique_error_list = list(dict.fromkeys(errors))
    report = {
        "status": "PASS" if not unique_error_list else "FAIL",
        "checks_passed": sum(1 for check in checks if check["status"] == "PASS"),
        "checks_failed": sum(1 for check in checks if check["status"] == "FAIL"),
        "checks": checks,
        "stats": stats,
        "errors": unique_error_list,
        "notes": [
            "Students.csv and SkillMasteryHistory.csv are simulator-only; never provide them to the tutoring agents.",
            "PythonProblems.csv includes answers; use AgentVisibleProblems.csv for agent context.",
            "PythonInteractions.csv includes simulator annotations; use AgentVisibleInteractions.csv for agent context and filter by timestamp.",
        ],
    }
    return report
