"""Orchestrate question generation, student simulation, output writing, and validation."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from generation.generate_questions import (
    AGENT_VISIBLE_PROBLEM_COLUMNS,
    PROBLEM_COLUMNS,
    build_generation_plan,
    generate_llm_questions,
    generate_template_questions,
)
from generation.skills import SKILLS, validate_skill_graph
from generation.simulate_students import (
    AGENT_VISIBLE_INTERACTION_COLUMNS,
    INTERACTION_COLUMNS,
    LEARNING_SEQUENCE_COLUMNS,
    MASTERY_COLUMNS,
    STUDENT_COLUMNS,
    simulate_student_interactions,
)
from generation.validate_questions import filter_executable_questions
from utils.validation import validate_dataset_files

LOGGER = logging.getLogger("adaptive_python_tutor")

DEFAULTS: dict[str, Any] = {
    "output_dir": "data/generated",
    "num_students": 20,
    "num_questions": 100,
    "interactions_per_student": 50,
    "seed": 42,
    "generation_mode": "template",
    "max_attempts": 2,
    "skill_coverage": [],
    "difficulty_distribution": {"beginner": 0.4, "intermediate": 0.4, "advanced": 0.2},
    "minimum_prerequisite_coverage": 0.5,
    "llm_base_url": "",
    "llm_model": "",
    "llm_api_key_env": "PYTUTOR_LLM_API_KEY",
    "llm_timeout_seconds": 45.0,
    "llm_retry_limit": 2,
    "llm_batch_size": 8,
}

CSV_COLUMNS: dict[str, list[str]] = {
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


def _load_config(config_path: str | None) -> dict[str, Any]:
    config = dict(DEFAULTS)
    if not config_path:
        return config
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() == ".json":
            loaded = json.loads(raw)
        else:
            try:
                import yaml  # Optional dependency; only needed for YAML config files.
            except ImportError as exc:
                raise RuntimeError("Reading YAML config requires PyYAML. Install with `python -m pip install PyYAML`.") from exc
            loaded = yaml.safe_load(raw) or {}
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Could not parse configuration {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("The configuration root must be a mapping/object")
    # Accept either flat keys or the sample's llm: subsection.
    llm = loaded.pop("llm", {}) if "llm" in loaded else {}
    if llm and isinstance(llm, dict):
        mapping = {
            "base_url": "llm_base_url", "model": "llm_model", "api_key_env": "llm_api_key_env",
            "timeout_seconds": "llm_timeout_seconds", "retry_limit": "llm_retry_limit", "batch_size": "llm_batch_size",
        }
        for source, target in mapping.items():
            if source in llm:
                loaded[target] = llm[source]
    unknown = sorted(set(loaded) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    config.update(loaded)
    return config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a synthetic Adaptive Python Tutor dataset.")
    parser.add_argument("--config", help="Optional YAML/JSON configuration file")
    parser.add_argument("--output-dir")
    parser.add_argument("--num-students", type=int)
    parser.add_argument("--num-questions", type=int)
    parser.add_argument("--interactions-per-student", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--generation-mode", choices=["template", "llm"])
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--skill-coverage", help="Comma-separated skill IDs, categories, or exact skill names; default: all skills")
    parser.add_argument("--difficulty-distribution", help="Weights such as beginner:0.4,intermediate:0.4,advanced:0.2")
    parser.add_argument("--minimum-prerequisite-coverage", type=float, help="Required fraction [0,1] of direct prerequisite skills represented by questions")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-api-key-env")
    parser.add_argument("--llm-timeout-seconds", type=float)
    parser.add_argument("--llm-retry-limit", type=int)
    parser.add_argument("--llm-batch-size", type=int)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing dataset generated by this tool")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def _ensure_output_dir(path: Path, overwrite: bool) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    if not path.is_dir():
        raise NotADirectoryError(f"Output path exists and is not a directory: {path}")
    contents = list(path.iterdir())
    if not contents:
        return
    if not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}. Choose a new path or pass --overwrite.")
    marker = path / "dataset_metadata.json"
    if not marker.is_file():
        raise FileExistsError(f"Refusing to overwrite {path}: dataset_metadata.json marker not found.")
    for child in contents:
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Write stable UTF-8 CSV columns and JSON-serialized field values."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _skills_rows() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    skill_rows = [
        {
            "skill_id": s.skill_id, "skill_name": s.skill_name, "description": s.description,
            "category": s.category, "level": s.level,
            "prerequisite_skill_ids": json.dumps(list(s.prerequisite_skill_ids), separators=(",", ":")),
        }
        for s in SKILLS
    ]
    edges = [{"skill_id": s.skill_id, "prerequisite_skill_id": pre} for s in SKILLS for pre in s.prerequisite_skill_ids]
    return skill_rows, edges


def _agent_visible_problem_rows(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for question in questions:
        result.append({field: question.get(field, "") for field in AGENT_VISIBLE_PROBLEM_COLUMNS})
    return result


def _merge_question_rejections(*collections: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for collection in collections:
        merged.extend(collection)
    return merged


def _select_skills(selection: Any) -> list[Any]:
    """Resolve skill IDs, categories, or exact names into canonical skill rows."""
    if selection in (None, [], "", "all"):
        return list(SKILLS)
    if isinstance(selection, str):
        requested = [item.strip() for item in selection.split(",") if item.strip()]
    elif isinstance(selection, list):
        requested = [str(item).strip() for item in selection if str(item).strip()]
    else:
        raise ValueError("skill_coverage must be a list or comma-separated string")
    selected: list[Any] = []
    unknown: list[str] = []
    for item in requested:
        match = [s for s in SKILLS if item.casefold() in {s.skill_id.casefold(), s.category.casefold(), s.skill_name.casefold()}]
        if not match:
            unknown.append(item)
            continue
        for skill in match:
            if skill not in selected:
                selected.append(skill)
    if unknown:
        raise ValueError(f"Unknown skill coverage selector(s): {unknown}")
    if not selected:
        raise ValueError("skill_coverage selected no skills")
    return selected


def _parse_difficulty_distribution(raw: Any) -> dict[str, float]:
    """Normalize difficulty weights from a mapping or CLI string."""
    if isinstance(raw, str):
        parsed: dict[str, float] = {}
        for entry in raw.split(","):
            if not entry.strip():
                continue
            if ":" not in entry:
                raise ValueError("difficulty distribution entries must use name:weight")
            key, value = entry.split(":", 1)
            parsed[key.strip().lower()] = float(value)
        raw = parsed
    if not isinstance(raw, dict) or not raw:
        raise ValueError("difficulty_distribution must be a non-empty mapping")
    weights = {str(key).lower(): float(value) for key, value in raw.items()}
    if set(weights) - {"beginner", "intermediate", "advanced"}:
        raise ValueError("difficulty_distribution keys must be beginner, intermediate, or advanced")
    if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("difficulty weights must be non-negative and sum to a positive number")
    return {key: value / sum(weights.values()) for key, value in weights.items() if value > 0}


def _assign_difficulties(questions: list[dict[str, Any]], distribution: dict[str, float], seed: int) -> dict[str, int]:
    """Assign a weighted, reproducible distribution without undercutting skill level.

    A question cannot be labeled easier than its skill's curriculum level. If the
    requested weights are infeasible under that constraint, the realized counts
    are projected into the feasible range and reported to the caller.
    """
    import random
    count = len(questions)
    if count == 0:
        return {key: 0 for key in ("beginner", "intermediate", "advanced")}
    rank = {"beginner": 0, "intermediate": 1, "advanced": 2}
    skill_levels = {skill.skill_id: skill.level for skill in SKILLS}
    capacities = {
        level: sum(1 for question in questions if rank.get(skill_levels.get(question["primary_skill_id"], "beginner"), 0) <= rank[level])
        for level in ("beginner", "intermediate", "advanced")
    }
    exact = {key: distribution.get(key, 0.0) * count for key in ("beginner", "intermediate", "advanced")}
    target = {key: int(value) for key, value in exact.items()}
    remaining = count - sum(target.values())
    ranked = sorted(exact, key=lambda key: (exact[key] - target[key], key), reverse=True)
    for key in ranked[:remaining]:
        target[key] += 1

    # Beginner labels only fit beginner-level skills; intermediate labels fit
    # beginner/intermediate skills; advanced labels can fit any skill.
    beginner_count = min(target["beginner"], capacities["beginner"])
    intermediate_count = min(target["intermediate"], capacities["intermediate"] - beginner_count)
    advanced_count = count - beginner_count - intermediate_count
    if advanced_count < sum(1 for q in questions if skill_levels.get(q["primary_skill_id"]) == "advanced"):
        deficit = sum(1 for q in questions if skill_levels.get(q["primary_skill_id"]) == "advanced") - advanced_count
        reduce_intermediate = min(deficit, intermediate_count)
        intermediate_count -= reduce_intermediate
        deficit -= reduce_intermediate
        beginner_count -= min(deficit, beginner_count)
        advanced_count = count - beginner_count - intermediate_count

    labels = ["beginner"] * beginner_count + ["intermediate"] * intermediate_count + ["advanced"] * advanced_count
    rng = random.Random(seed + 303)
    rng.shuffle(labels)
    # Pair questions with difficulty-compatible labels. Shuffle within each tier,
    # then assign the least advanced compatible label to each question first.
    remaining_labels = labels[:]
    rng.shuffle(questions)
    assignments: dict[str, str] = {}
    for question in questions:
        minimum_rank = rank.get(skill_levels.get(question["primary_skill_id"], "beginner"), 0)
        compatible = [label for label in remaining_labels if rank[label] >= minimum_rank]
        if not compatible:
            raise ValueError("Could not assign a difficulty label consistent with skill curriculum levels")
        chosen = min(compatible, key=lambda label: (rank[label], rng.random()))
        remaining_labels.remove(chosen)
        assignments[question["problem_id"]] = chosen
    for question in questions:
        question["difficulty"] = assignments[question["problem_id"]]
    return {key: sum(1 for q in questions if q["difficulty"] == key) for key in ("beginner", "intermediate", "advanced")}


def _prerequisite_coverage(questions: list[dict[str, Any]], selected_skills: list[Any], minimum: float) -> dict[str, Any]:
    """Check whether direct prerequisite skills are represented in the question bank."""
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum_prerequisite_coverage must be between 0 and 1")
    skill_map = {skill.skill_id: skill for skill in SKILLS}
    required = {pre for skill in selected_skills for pre in skill.prerequisite_skill_ids}
    covered = {q["primary_skill_id"] for q in questions}
    missing = sorted(required - covered)
    ratio = 1.0 if not required else len(required & covered) / len(required)
    result = {"required_prerequisite_skill_count": len(required), "covered_prerequisite_skill_count": len(required & covered), "coverage_ratio": round(ratio, 6), "missing_skill_ids": missing, "minimum_required": minimum}
    if ratio < minimum:
        raise ValueError(f"Prerequisite coverage {ratio:.3f} is below minimum {minimum:.3f}; missing question coverage for {missing}")
    return result


def build_dataset(config: dict[str, Any], *, overwrite: bool = False) -> dict[str, Any]:
    """Build all artifacts and persist a machine-readable validation report."""
    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    _ensure_output_dir(output_dir, overwrite)
    skill_errors = validate_skill_graph(SKILLS)
    if skill_errors:
        raise ValueError("Invalid canonical skill graph: " + "; ".join(skill_errors))

    num_questions = int(config["num_questions"])
    seed = int(config["seed"])
    selected_skills = _select_skills(config.get("skill_coverage", []))
    difficulty_distribution = _parse_difficulty_distribution(config.get("difficulty_distribution", DEFAULTS["difficulty_distribution"]))
    minimum_prerequisite_coverage = float(config.get("minimum_prerequisite_coverage", 0.5))
    if not 0.0 <= minimum_prerequisite_coverage <= 1.0:
        raise ValueError("minimum_prerequisite_coverage must be between 0 and 1")
    if num_questions < 1 or int(config["num_students"]) < 1 or int(config["interactions_per_student"]) < 1:
        raise ValueError("num_questions, num_students, and interactions_per_student must all be positive")
    if int(config["max_attempts"]) < 1:
        raise ValueError("max_attempts must be at least 1")
    if float(config["llm_timeout_seconds"]) <= 0:
        raise ValueError("llm_timeout_seconds must be greater than 0")
    if int(config["llm_retry_limit"]) < 0:
        raise ValueError("llm_retry_limit must be non-negative")
    if int(config["llm_batch_size"]) < 1:
        raise ValueError("llm_batch_size must be at least 1")

    LOGGER.info("Generating %d Python question candidates in %s mode", num_questions, config["generation_mode"])
    if config["generation_mode"] == "template":
        question_candidates, rejected_schema, qgen_report = generate_template_questions(num_questions, seed, selected_skills)
    elif config["generation_mode"] == "llm":
        api_key_env = str(config["llm_api_key_env"])
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ValueError(f"LLM mode selected but environment variable {api_key_env!r} is empty")
        if not str(config["llm_model"]).strip():
            raise ValueError("LLM mode requires --llm-model or llm.model in the config")
        plan = build_generation_plan(num_questions, selected_skills)
        question_candidates, rejected_schema, qgen_report = generate_llm_questions(
            plan, base_url=str(config["llm_base_url"]), model=str(config["llm_model"]), api_key=api_key,
            timeout_seconds=float(config["llm_timeout_seconds"]), retry_limit=int(config["llm_retry_limit"]),
            batch_size=int(config["llm_batch_size"]), seed=seed,
        )
    else:
        raise ValueError(f"Unknown generation_mode: {config['generation_mode']}")

    questions, rejected_quality = filter_executable_questions(question_candidates)
    rejected = _merge_question_rejections(rejected_schema, rejected_quality)
    qgen_report.update({
        "candidate_count": qgen_report.get("generated", len(question_candidates)),
        "accepted": len(questions),
        "rejected": len(rejected),
        "requested_shortfall": max(0, num_questions - len(questions)),
        "executable_validation_rejections": len(rejected_quality),
    })
    if not questions:
        # Persist rejection evidence and schema-valid empty CSVs even when a
        # generation run cannot produce any accepted question candidates.
        skill_rows, edge_rows = _skills_rows()
        empty_tables: dict[str, list[dict[str, Any]]] = {filename: [] for filename in CSV_COLUMNS}
        empty_tables["PythonSkills.csv"] = skill_rows
        empty_tables["SkillPrerequisites.csv"] = edge_rows
        for filename, rows in empty_tables.items():
            _write_csv(output_dir / filename, rows, CSV_COLUMNS[filename])
        _write_csv(output_dir / "rejected_questions.csv", rejected, ["candidate_id", "reason"])
        _write_json(output_dir / "question_generation_report.json", qgen_report)
        metadata = {
            "project": "Adaptive Python Tutor Agent", "dataset_version": "0.1.0",
            "seed": seed, "configuration": config, "question_generation": qgen_report,
            "student_count": 0, "accepted_question_count": 0, "interaction_count": 0,
            "skill_count": len(SKILLS), "hidden_artifacts": ["Students.csv", "SkillMasteryHistory.csv", "PythonProblems.csv"],
            "agent_artifacts": ["AgentVisibleProblems.csv", "AgentVisibleInteractions.csv", "PythonSkills.csv", "SkillPrerequisites.csv"],
            "mastery_threshold": 0.70,
            "synthetic_data_notice": "Generated learners and events are simulated. Results evaluate simulator assumptions, not real-world learning efficacy.",
        }
        _write_json(output_dir / "dataset_metadata.json", metadata)
        failed_report = validate_dataset_files(output_dir)
        failed_check = {"name": "question_bank_nonempty", "status": "FAIL", "message": "No generated question candidate passed schema and executable validation", "count": 0}
        failed_report["checks"].append(failed_check)
        failed_report["checks_failed"] += 1
        failed_report["status"] = "FAIL"
        failed_report["errors"].append(failed_check["message"])
        failed_report["question_generation"] = qgen_report
        failed_report["rejected_question_records"] = len(rejected)
        _write_json(output_dir / "validation_report.json", failed_report)
        raise RuntimeError(f"No questions passed validation. Rejections and reports were saved to {output_dir}.")
    difficulty_counts = _assign_difficulties(questions, difficulty_distribution, seed)
    prerequisite_coverage = _prerequisite_coverage(questions, selected_skills, minimum_prerequisite_coverage)
    qgen_report["difficulty_counts"] = difficulty_counts
    qgen_report["skill_coverage_accepted"] = sorted({q["primary_skill_id"] for q in questions})
    qgen_report["prerequisite_coverage"] = prerequisite_coverage
    if len(questions) < num_questions:
        LOGGER.warning("Accepted %d/%d requested questions; see rejected_questions.csv", len(questions), num_questions)
    LOGGER.info("Accepted %d questions; rejected %d", len(questions), len(rejected))

    profiles, interactions, sequence, mastery_history = simulate_student_interactions(
        questions, int(config["num_students"]), int(config["interactions_per_student"]), seed + 7,
        max_attempts=int(config["max_attempts"]),
    )
    skill_rows, edge_rows = _skills_rows()

    tables: dict[str, list[dict[str, Any]]] = {
        "PythonProblems.csv": questions,
        "AgentVisibleProblems.csv": _agent_visible_problem_rows(questions),
        "PythonSkills.csv": skill_rows,
        "SkillPrerequisites.csv": edge_rows,
        "Students.csv": profiles,
        "PythonInteractions.csv": interactions,
        "AgentVisibleInteractions.csv": [{key: row.get(key, "") for key in AGENT_VISIBLE_INTERACTION_COLUMNS} for row in interactions],
        "LearningSequence.csv": sequence,
        "SkillMasteryHistory.csv": mastery_history,
    }
    LOGGER.info("Writing %d interactions for %d students", len(interactions), len(profiles))
    for filename, rows in tables.items():
        _write_csv(output_dir / filename, rows, CSV_COLUMNS[filename])
    _write_csv(output_dir / "rejected_questions.csv", rejected, ["candidate_id", "reason"])
    _write_json(output_dir / "question_generation_report.json", qgen_report)
    metadata = {
        "project": "Adaptive Python Tutor Agent",
        "dataset_version": "0.1.0",
        "seed": seed,
        "configuration": config,
        "question_generation": qgen_report,
        "student_count": len(profiles),
        "accepted_question_count": len(questions),
        "interaction_count": len(interactions),
        "skill_count": len(SKILLS),
        "hidden_artifacts": ["Students.csv", "SkillMasteryHistory.csv", "PythonProblems.csv"],
        "agent_artifacts": ["AgentVisibleProblems.csv", "AgentVisibleInteractions.csv", "PythonSkills.csv", "SkillPrerequisites.csv"],
        "mastery_threshold": 0.70,
        "synthetic_data_notice": "Generated learners and events are simulated. Results evaluate simulator assumptions, not real-world learning efficacy.",
    }
    _write_json(output_dir / "dataset_metadata.json", metadata)
    validation_report = validate_dataset_files(output_dir)
    validation_report["question_generation"] = qgen_report
    validation_report["rejected_question_records"] = len(rejected)
    _write_json(output_dir / "validation_report.json", validation_report)
    LOGGER.info(
        "Validation status: %s (%d checks passed, %d failed)",
        validation_report["status"], validation_report["checks_passed"], validation_report["checks_failed"],
    )
    if validation_report["status"] != "PASS":
        LOGGER.error("Validation errors: %s", validation_report["errors"][:10])
    return {"output_dir": str(output_dir), "question_generation": qgen_report, "validation": validation_report}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    try:
        config = _load_config(args.config)
        for key in DEFAULTS:
            arg_name = key
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                config[key] = getattr(args, arg_name)
        if args.skill_coverage is not None:
            config["skill_coverage"] = [item.strip() for item in args.skill_coverage.split(",") if item.strip()]
        if args.difficulty_distribution is not None:
            config["difficulty_distribution"] = args.difficulty_distribution
        if args.minimum_prerequisite_coverage is not None:
            config["minimum_prerequisite_coverage"] = args.minimum_prerequisite_coverage
        result = build_dataset(config, overwrite=args.overwrite)
        print(json.dumps({
            "output_dir": result["output_dir"],
            "questions": result["question_generation"],
            "validation_status": result["validation"]["status"],
            "validation_checks_failed": result["validation"]["checks_failed"],
        }, indent=2))
        return 0 if result["validation"]["status"] == "PASS" else 2
    except KeyboardInterrupt:
        LOGGER.error("Interrupted. Partial outputs may remain; rerun with --overwrite after checking the output directory.")
        return 130
    except Exception as exc:
        LOGGER.error("Dataset build failed: %s: %s", type(exc).__name__, exc)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.exception("Build traceback")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
