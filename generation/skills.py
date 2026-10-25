"""Canonical Python curriculum and its prerequisite DAG."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Skill:
    skill_id: str
    skill_name: str
    description: str
    category: str
    level: str
    prerequisite_skill_ids: tuple[str, ...] = ()


SKILLS: tuple[Skill, ...] = (
    Skill("SK001", "Variables and assignment", "Bind names to values and update bindings.", "fundamentals", "beginner"),
    Skill("SK002", "Primitive data types", "Recognize integers, floats, strings, and booleans.", "fundamentals", "beginner", ("SK001",)),
    Skill("SK003", "Type conversion", "Convert values explicitly with constructors such as int and str.", "fundamentals", "beginner", ("SK002",)),
    Skill("SK004", "Arithmetic operators", "Use arithmetic operators and evaluate expressions.", "expressions", "beginner", ("SK001", "SK002")),
    Skill("SK005", "Comparison and logical operators", "Combine comparisons with and, or, and not.", "expressions", "beginner", ("SK002", "SK004")),
    Skill("SK006", "Strings and string operations", "Index, slice, and transform immutable strings.", "data_structures", "beginner", ("SK001", "SK002")),
    Skill("SK007", "Input and output", "Understand input values and display formatted output.", "fundamentals", "beginner", ("SK001", "SK003", "SK006")),
    Skill("SK008", "Boolean expressions", "Evaluate truth-valued expressions and truthiness.", "expressions", "beginner", ("SK005",)),
    Skill("SK009", "Conditional statements", "Choose execution paths with if, elif, and else.", "control_flow", "beginner", ("SK008",)),
    Skill("SK010", "For loops", "Repeat operations over iterable values.", "control_flow", "intermediate", ("SK001", "SK005")),
    Skill("SK011", "While loops", "Repeat while a condition remains true and reason about termination.", "control_flow", "intermediate", ("SK008", "SK009")),
    Skill("SK012", "range and iteration", "Use range start, stop, and step semantics.", "control_flow", "intermediate", ("SK010",)),
    Skill("SK013", "Lists and indexing", "Create lists and use zero-based indexing safely.", "data_structures", "intermediate", ("SK002", "SK010")),
    Skill("SK014", "Slicing", "Select subsequences using start, stop, and step.", "data_structures", "intermediate", ("SK006", "SK013")),
    Skill("SK015", "Tuples and sets", "Understand tuple immutability and set uniqueness.", "data_structures", "intermediate", ("SK002", "SK013")),
    Skill("SK016", "Dictionaries", "Store and retrieve key-value pairs.", "data_structures", "intermediate", ("SK002",)),
    Skill("SK017", "Functions and parameters", "Define reusable functions with parameters.", "functions", "intermediate", ("SK001", "SK004", "SK009")),
    Skill("SK018", "Return values and scope", "Distinguish return from print and understand local scope.", "functions", "intermediate", ("SK017",)),
    Skill("SK019", "Comprehensions", "Construct lists or dictionaries concisely from iterables.", "functions", "intermediate", ("SK010", "SK013", "SK016")),
    Skill("SK020", "Exceptions and debugging", "Recognize common runtime errors and handle expected failures.", "debugging", "intermediate", ("SK002", "SK009")),
    Skill("SK021", "Recursion", "Use a base case and recursive calls to solve smaller subproblems.", "algorithms", "advanced", ("SK017", "SK018", "SK009")),
    Skill("SK022", "File handling", "Read text safely with context managers and process file contents.", "io", "advanced", ("SK007", "SK020")),
    Skill("SK023", "Object-oriented programming", "Define classes, instances, attributes, and methods.", "programming_paradigms", "advanced", ("SK017", "SK002")),
    Skill("SK024", "Mutable versus immutable objects", "Distinguish in-place mutation from creation of new values.", "semantics", "advanced", ("SK013", "SK015", "SK006")),
    Skill("SK025", "References and aliasing", "Reason about multiple names referring to the same mutable object.", "semantics", "advanced", ("SK013", "SK024")),
    Skill("SK026", "Algorithmic reasoning", "Translate a problem into a correct finite procedure.", "algorithms", "advanced", ("SK010", "SK013", "SK017")),
    Skill("SK027", "Time complexity fundamentals", "Classify simple algorithms by growth rate.", "algorithms", "advanced", ("SK010", "SK026")),
)


def skill_by_id() -> dict[str, Skill]:
    """Return the canonical skills indexed by stable ID."""
    return {skill.skill_id: skill for skill in SKILLS}


def validate_skill_graph(skills: Iterable[Skill] = SKILLS) -> list[str]:
    """Return graph integrity errors, including cycles and missing references."""
    skill_list = list(skills)
    ids = {item.skill_id for item in skill_list}
    errors: list[str] = []
    if len(ids) != len(skill_list):
        errors.append("skill_id values are not unique")
    for skill in skill_list:
        for prerequisite in skill.prerequisite_skill_ids:
            if prerequisite not in ids:
                errors.append(f"{skill.skill_id}: unknown prerequisite {prerequisite}")
            if prerequisite == skill.skill_id:
                errors.append(f"{skill.skill_id}: self prerequisite is not allowed")

    graph = {s.skill_id: [p for p in s.prerequisite_skill_ids if p in ids] for s in skill_list}
    state: dict[str, int] = {node: 0 for node in graph}  # 0 unseen, 1 visiting, 2 done

    def visit(node: str, path: list[str]) -> None:
        if state[node] == 1:
            errors.append("prerequisite cycle detected: " + " -> ".join(path + [node]))
            return
        if state[node] == 2:
            return
        state[node] = 1
        for parent in graph[node]:
            visit(parent, path + [node])
        state[node] = 2

    for node in graph:
        if state[node] == 0:
            visit(node, [])
    return errors
