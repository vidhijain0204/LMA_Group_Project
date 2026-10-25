"""Generate Python questions from deterministic templates or an optional LLM.

Structured values are serialized as JSON in CSV columns. The offline generator
has no network or third-party dependency.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from generation.skills import SKILLS, Skill
from utils.code_runner import run_snippet

QUESTION_TYPES = (
    "mcq_single",
    "mcq_multi",
    "code_trace",
    "fill_blank",
    "debugging",
    "short_answer",
    "coding",
)
DIFFICULTIES = ("beginner", "intermediate", "advanced")
PROBLEM_COLUMNS = [
    "problem_id", "Problem Body", "Problem Type", "Multiple Choice Options",
    "Multiple Choice Answers", "Fill-in Answers", "difficulty", "primary_skill_id",
    "secondary_skill_ids", "misconception_tags", "reference_solution", "explanation",
    "starter_code", "test_cases_json", "expected_output", "answer_key_json",
    "estimated_time_seconds", "generation_source", "trace_code", "function_name", "response_templates_json",
]
AGENT_VISIBLE_PROBLEM_COLUMNS = [
    "problem_id", "Problem Body", "Problem Type", "Multiple Choice Options",
    "difficulty", "primary_skill_id", "secondary_skill_ids", "estimated_time_seconds",
]

# A skill-specific tag is evidence vocabulary, not a claim about a real learner.
SKILL_TAGS: dict[str, list[str]] = {
    "SK001": ["assignment_rebinding", "variable_value_confusion"],
    "SK002": ["type_confusion", "bool_int_confusion"],
    "SK003": ["type_conversion", "string_number_confusion"],
    "SK004": ["operator_precedence", "integer_division_confusion"],
    "SK005": ["comparison_logic", "and_or_confusion"],
    "SK006": ["string_immutability", "string_indexing"],
    "SK007": ["input_is_string", "print_vs_return"],
    "SK008": ["truthiness", "boolean_logic"],
    "SK009": ["conditional_boundary", "branch_selection"],
    "SK010": ["off_by_one", "loop_accumulator"],
    "SK011": ["loop_termination", "while_condition"],
    "SK012": ["range_stop_exclusive", "range_step"],
    "SK013": ["zero_based_indexing", "index_out_of_range"],
    "SK014": ["slice_stop_exclusive", "slice_bounds"],
    "SK015": ["tuple_immutability", "set_uniqueness"],
    "SK016": ["dictionary_key_lookup", "missing_key"],
    "SK017": ["parameter_order", "function_call"],
    "SK018": ["return_vs_print", "local_scope"],
    "SK019": ["comprehension_expression", "comprehension_filter"],
    "SK020": ["exception_type", "debugging_cause"],
    "SK021": ["recursion_base_case", "recursive_step"],
    "SK022": ["file_context_manager", "line_processing"],
    "SK023": ["instance_attribute", "self_reference"],
    "SK024": ["mutable_immutable", "in_place_mutation"],
    "SK025": ["mutable_aliasing", "shallow_copy"],
    "SK026": ["algorithm_state", "loop_invariant"],
    "SK027": ["linear_complexity", "nested_loop_complexity"],
}

CODING_SKILLS = {"SK006", "SK010", "SK013", "SK016", "SK017", "SK021", "SK022", "SK026"}
DEBUGGING_SKILLS = {"SK009", "SK010", "SK011", "SK013", "SK018", "SK020", "SK021", "SK025"}


def _literal(value: Any) -> str:
    """Return a Python literal for simple generated values."""
    return repr(value)


def _case_for(skill: Skill, variant: int, rng: random.Random) -> dict[str, Any]:
    """Build one prevalidated concept case tied to a target skill."""
    sid = skill.skill_id
    a = rng.randint(2, 9)
    b = rng.randint(2, 8)
    word = rng.choice(["Python", "garden", "speech", "module", "coding", "orange", "window"])
    values = [rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)]
    n = rng.randint(3, 7)
    case: dict[str, Any] = {
        "concept_question": "Explain the main Python concept demonstrated by this example.",
        "concept_answer": "Trace the expression in order and apply Python's documented semantics.",
        "snippet": f"left = {a}\nright = {b}\nprint(left + right)",
        "output": str(a + b),
        "fill_snippet": f"left = ____\nright = {b}\nprint(left + right)",
        "fill_answer": str(a),
        "distractors": [str(a - b), str(a * b), str(a + b + 1)],
        "multi_true": ["Python evaluates the expression using the operator's defined semantics.", "A variable name refers to a value bound during execution."],
        "multi_false": ["Python variables must be declared with a fixed type before assignment.", "Every expression evaluates to a string.", "Whitespace changes the numeric value of an integer."],
        "buggy_code": "value = 3\nif value = 3:\n    print('match')",
        "fixed_code": "value = 3\nif value == 3:\n    print('match')",
        "fixed_line": "if value == 3:",
        "wrong_answers": ["It always evaluates to None.", "Python converts every value to a string automatically."],
        "explanation": "Use the code and Python's evaluation rules to justify the result.",
        "coding_prompt": "Write a function that returns the given value unchanged.",
        "starter_code": "def solve(value):\n    \"\"\"Return the supplied value.\"\"\"\n    pass\n",
        "reference_solution": "def solve(value):\n    return value\n",
        "function_name": "solve",
        "test_cases": [{"args": [a], "expected": a}, {"args": [word], "expected": word}],
        "buggy_solution": "def solve(value):\n    return None\n",
        "estimated_time_seconds": 90,
    }

    if sid == "SK001":
        start, delta = a, b
        case.update(
            concept_question="What value is bound to x after the reassignment?",
            concept_answer="Assignment rebinds the name to the value produced by the right-hand expression.",
            snippet=f"x = {start}\nx = x + {delta}\nprint(x)", output=str(start + delta),
            fill_snippet=f"x = ____\nx = x + {delta}\nprint(x)", fill_answer=str(start),
            distractors=[str(start), str(start + delta + 1), str(delta)],
            multi_true=["An assignment binds a name to a value.", "A later assignment can rebind an existing name."],
            multi_false=["A variable's value can never change.", "Assignment uses == to bind a value.", "Every name is global."],
            explanation="The second assignment reads the current x, adds delta, and rebinds x.",
        )
    elif sid == "SK002":
        kind = rng.choice([("int", a), ("float", float(a) + 0.5), ("str", word), ("bool", a > b)])
        case.update(
            concept_question="Which built-in type does this value have?",
            concept_answer=f"The value's type name is {kind[0]}.",
            snippet=f"value = {_literal(kind[1])}\nprint(type(value).__name__)", output=kind[0],
            fill_snippet=f"value = {_literal(kind[1])}\nprint(type(value).____)", fill_answer="__name__",
            distractors=["list", "str", "int"] if kind[0] not in {"str", "int"} else ["list", "tuple", "dict"],
            multi_true=["An int represents an integer value.", "A str represents text."],
            multi_false=["A float can only represent whole numbers.", "bool has no relationship to True and False.", "A variable's type is determined only by its name."],
            explanation="type(value).__name__ returns the built-in type's name.",
        )
    elif sid == "SK003":
        raw = str(a * 10 + b)
        case.update(
            concept_question="Why is explicit type conversion useful here?",
            concept_answer="int converts a string containing a valid integer representation into an integer.",
            snippet=f"raw = {_literal(raw)}\nvalue = int(raw) + {b}\nprint(value)", output=str(int(raw) + b),
            fill_snippet=f"raw = {_literal(raw)}\nvalue = ____(raw) + {b}\nprint(value)", fill_answer="int",
            distractors=[str(raw + str(b)), str(int(raw) - b), str(int(raw) * b)],
            multi_true=["int('12') produces the integer 12.", "Converting arbitrary non-numeric text with int can raise ValueError."],
            multi_false=["int('12') produces the string '12'.", "str(12) and int(12) have the same type.", "Type conversion never raises an exception."],
            explanation="The text is converted to an integer before arithmetic is performed.",
        )
    elif sid == "SK004":
        op = rng.choice(["+", "-", "*", "//", "%"])
        if op == "+": answer = a + b
        elif op == "-": answer = a - b
        elif op == "*": answer = a * b
        elif op == "//": answer = a // b
        else: answer = a % b
        case.update(
            concept_question=f"What does the {op!r} arithmetic operator do in this expression?",
            concept_answer={"+": "It adds the operands.", "-": "It subtracts the right operand from the left.", "*": "It multiplies the operands.", "//": "It performs floor division.", "%": "It returns the remainder."}[op],
            snippet=f"print({a} {op} {b})", output=str(answer),
            fill_snippet=f"print({a} ____ {b})", fill_answer=op,
            distractors=[str(a + b), str(a * b), str(a - b)],
            multi_true=["// performs floor division.", "% returns a remainder."],
            multi_false=["/ always returns an integer.", "* performs string comparison.", "Operator precedence is ignored."],
            explanation=f"Applying {op!r} to {a} and {b} gives {answer}.",
        )
    elif sid == "SK005":
        x, y = a, b
        correct = x > y and (x % 2 == 0)
        case.update(
            concept_question="How do comparison and logical operators combine in this expression?",
            concept_answer="Comparison expressions produce booleans, and and combines them using short-circuit Boolean logic.",
            snippet=f"x = {x}\ny = {y}\nprint(x > y and x % 2 == 0)", output=str(correct),
            fill_snippet=f"x = {x}\ny = {y}\nprint(x > y ____ x % 2 == 0)", fill_answer="and",
            distractors=[str(x > y), str(x % 2 == 0), str(x > y or x % 2 == 0)],
            multi_true=["Comparisons such as x > y produce booleans.", "and is true only when both operands are truthy."],
            multi_false=["= and == have the same meaning in expressions.", "or is true only when both operands are true.", "not adds two numeric values."],
            explanation="Both comparisons are evaluated according to and semantics; the result is a Boolean.",
        )
    elif sid == "SK006":
        piece = word[1:4]
        case.update(
            concept_question="What string is produced by lower() and slicing?",
            concept_answer="Strings are immutable; methods such as lower return a string value, and slicing selects a subsequence.",
            snippet=f"text = {_literal(word)}\nprint(text.lower())\nprint(text[1:4])",
            output=f"{word.lower()}\n{piece}",
            fill_snippet=f"text = {_literal(word)}\nprint(text.____())\nprint(text[1:4])", fill_answer="lower",
            distractors=[f"{word.upper()}\n{piece}", f"{word.lower()}\n{word[1:]}", f"{word}\n{piece}"],
            multi_true=["Strings are immutable sequences.", "The stop index in a slice is excluded."],
            multi_false=["lower() changes the original string in place.", "A string slice always includes its stop index.", "String indices start at one."],
            explanation="lower() returns lowercase text; [1:4] selects indices 1, 2, and 3.",
            coding_prompt="Implement normalize_word(text), returning text with surrounding whitespace removed and letters lowercased.",
            starter_code="def normalize_word(text):\n    pass\n",
            reference_solution="def normalize_word(text):\n    return text.strip().lower()\n",
            function_name="normalize_word",
            test_cases=[{"args": ["  PyThOn  "], "expected": "python"}, {"args": ["Speech"], "expected": "speech"}, {"args": ["  "], "expected": ""}],
            buggy_solution="def normalize_word(text):\n    return text.upper()\n",
        )
    elif sid == "SK007":
        name = rng.choice(["Mira", "Ari", "Sam", "Noor"])
        case.update(
            concept_question="What is the role of print in this example?",
            concept_answer="print writes a readable representation to standard output; input returns text unless it is converted.",
            snippet=f"name = {_literal(name)}\nprint('Hello, ' + name)", output=f"Hello, {name}",
            fill_snippet=f"name = {_literal(name)}\n____('Hello, ' + name)", fill_answer="print",
            distractors=[name, f"'Hello, {name}'", "None"],
            multi_true=["print writes output to standard output.", "input normally returns a str."],
            multi_false=["input automatically converts every response to int.", "print returns the displayed string.", "Output is stored in a variable named stdout by default."],
            explanation="The string expression is passed to print, which displays it.",
        )
    elif sid == "SK008":
        threshold = rng.randint(3, 7)
        case.update(
            concept_question="What does the Boolean expression evaluate to?",
            concept_answer="A Boolean expression evaluates to True or False based on the truth values of its parts.",
            snippet=f"value = {a}\nprint(value >= {threshold} and value < {threshold + 4})",
            output=str(a >= threshold and a < threshold + 4),
            fill_snippet=f"value = {a}\nprint(value >= {threshold} ____ value < {threshold + 4})", fill_answer="and",
            distractors=["True", "False", "None"],
            multi_true=["A Boolean expression evaluates to True or False.", "and requires both operands to be truthy for a truthy result."],
            multi_false=["False and True evaluates to True.", "Boolean expressions can only contain literals.", "not True evaluates to True."],
            explanation="Evaluate each comparison, then combine them using and.",
        )
    elif sid == "SK009":
        score = rng.randint(45, 95)
        threshold = 60
        output = "pass" if score >= threshold else "retry"
        case.update(
            concept_question="Which branch executes when the condition is evaluated?",
            concept_answer="An if statement executes the first branch whose condition is true; else runs if none is true.",
            snippet=f"score = {score}\nif score >= {threshold}:\n    print('pass')\nelse:\n    print('retry')", output=output,
            fill_snippet=f"score = {score}\nif score >= ____:\n    print('pass')\nelse:\n    print('retry')", fill_answer=str(threshold),
            distractors=["pass" if output == "retry" else "retry", "pass\nretry", "None"],
            multi_true=["The condition is evaluated before choosing a branch.", "else runs when no preceding if/elif condition was true."],
            multi_false=["Both if and else bodies always run.", "A comparison uses = for equality.", "Indentation has no effect on Python blocks."],
            explanation=f"The condition {score} >= {threshold} is {score >= threshold}, so the {output!r} branch runs.",
            buggy_code=f"score = {score}\nif score = {threshold}:\n    print('pass')\nelse:\n    print('retry')",
            fixed_code=f"score = {score}\nif score >= {threshold}:\n    print('pass')\nelse:\n    print('retry')", fixed_line=f"if score >= {threshold}:",
        )
    elif sid == "SK010":
        limit = n
        case.update(
            concept_question="How does the loop update the accumulator?",
            concept_answer="A for loop visits each value in the iterable once; the accumulator keeps the running total.",
            snippet=f"total = 0\nfor i in range({limit}):\n    total += i\nprint(total)", output=str(sum(range(limit))),
            fill_snippet=f"total = 0\nfor i in range(____):\n    total += i\nprint(total)", fill_answer=str(limit),
            distractors=[str(sum(range(limit - 1))), str(sum(range(limit + 1))), str(limit)],
            multi_true=["range(stop) excludes stop.", "A for loop can iterate over each item in an iterable."],
            multi_false=["range(stop) always includes stop.", "A for loop must run exactly once.", "continue exits the entire program."],
            explanation="range(n) yields 0 through n-1; total accumulates those values.",
            buggy_code=f"total = 0\nfor i in range({limit + 1}):\n    total += i\nprint(total)",
            fixed_code=f"total = 0\nfor i in range({limit}):\n    total += i\nprint(total)", fixed_line=f"for i in range({limit}):",
            coding_prompt="Implement sum_even(numbers), returning the sum of all even integers in the input list.",
            starter_code="def sum_even(numbers):\n    pass\n",
            reference_solution="def sum_even(numbers):\n    total = 0\n    for value in numbers:\n        if value % 2 == 0:\n            total += value\n    return total\n",
            function_name="sum_even",
            test_cases=[{"args": [[1, 2, 3, 4]], "expected": 6}, {"args": [[-2, 5, 8]], "expected": 6}, {"args": [[]], "expected": 0}],
            buggy_solution="def sum_even(numbers):\n    total = 0\n    for value in numbers:\n        if value % 2 == 1:\n            total += value\n    return total\n",
        )
    elif sid == "SK011":
        limit = rng.randint(3, 7)
        case.update(
            concept_question="What condition controls how long the while loop repeats?",
            concept_answer="A while loop repeats while its condition is true; its state must change to reach termination when needed.",
            snippet=f"count = 0\nwhile count < {limit}:\n    count += 1\nprint(count)", output=str(limit),
            fill_snippet=f"count = 0\nwhile count < {limit}:\n    count ____ 1\nprint(count)", fill_answer="+=",
            distractors=[str(limit - 1), str(limit + 1), "0"],
            multi_true=["A while loop checks its condition before each iteration.", "Updating state can make a while loop terminate."],
            multi_false=["A while loop always runs at least once.", "while conditions must be integers.", "A true condition immediately exits the loop."],
            explanation="The count increases until count < limit becomes false.",
            buggy_code=f"count = 0\nwhile count < {limit}:\n    count -= 1\nprint(count)",
            fixed_code=f"count = 0\nwhile count < {limit}:\n    count += 1\nprint(count)", fixed_line="count += 1",
        )
    elif sid == "SK012":
        start, stop, step = rng.randint(0, 2), rng.randint(7, 10), rng.choice([2, 3])
        seq = list(range(start, stop, step))
        case.update(
            concept_question="Which values are produced by range(start, stop, step)?",
            concept_answer="range includes start when reachable, excludes stop, and advances by step.",
            snippet=f"print(list(range({start}, {stop}, {step})))", output=repr(seq),
            fill_snippet=f"print(list(range({start}, {stop}, ____)))", fill_answer=str(step),
            distractors=[repr(list(range(start, stop + 1, step))), repr(list(range(start, stop, 1))), repr(list(range(start + step, stop, step)))],
            multi_true=["The stop value is excluded.", "The step controls the difference between successive values."],
            multi_false=["range always includes stop.", "range(5, 1) counts down by default.", "The step must be a float."],
            explanation="Construct the sequence from start by step increments strictly before stop.",
        )
    elif sid == "SK013":
        index = rng.randrange(len(values))
        case.update(
            concept_question="How does Python select an element from this list?",
            concept_answer="List indices are zero-based, so index 0 selects the first element.",
            snippet=f"values = {values!r}\nprint(values[{index}])", output=str(values[index]),
            fill_snippet=f"values = {values!r}\nprint(values[____])", fill_answer=str(index),
            distractors=[str(values[(index + 1) % len(values)]), str(values[0]), str(values[-1])],
            multi_true=["List indices start at zero.", "An invalid index can raise IndexError."],
            multi_false=["The first element is at index 1.", "Lists cannot contain repeated values.", "Indexing a list always returns a string."],
            explanation="The requested zero-based index is {index}; the element there is {values[index]}.",
            buggy_code=f"values = {values!r}\nprint(values[{min(index + 1, len(values) - 1)}])",
            fixed_code=f"values = {values!r}\nprint(values[{index}])", fixed_line=f"print(values[{index}])",
            coding_prompt="Implement second_or_none(values), returning the second element or None if fewer than two elements exist.",
            starter_code="def second_or_none(values):\n    pass\n",
            reference_solution="def second_or_none(values):\n    return values[1] if len(values) > 1 else None\n",
            function_name="second_or_none",
            test_cases=[{"args": [[4, 7, 9]], "expected": 7}, {"args": [[5]], "expected": None}, {"args": [[]], "expected": None}],
            buggy_solution="def second_or_none(values):\n    return values[0] if len(values) > 1 else None\n",
        )
    elif sid == "SK014":
        seq = ["a", "b", "c", "d", "e"]
        stop = rng.randint(2, 5)
        case.update(
            concept_question="Which subsequence does the slice select?",
            concept_answer="A slice includes its start index and excludes its stop index.",
            snippet=f"items = {seq!r}\nprint(items[1:{stop}])", output=repr(seq[1:stop]),
            fill_snippet=f"items = {seq!r}\nprint(items[1:____])", fill_answer=str(stop),
            distractors=[repr(seq[1:stop + 1]), repr(seq[:stop]), repr(seq[stop:])],
            multi_true=["A slice's stop index is excluded.", "Omitted slice bounds use sequence defaults."],
            multi_false=["A slice always mutates the original list.", "The stop index is always included.", "Negative indices are invalid."],
            explanation="items[1:stop] selects indices 1 up to but not including stop.",
        )
    elif sid == "SK015":
        seq = [a, b, a, n]
        case.update(
            concept_question="Why can a set have fewer elements than the source list?",
            concept_answer="Sets contain unique hashable elements; tuples are immutable sequences.",
            snippet=f"items = {seq!r}\nprint(len(set(items)))", output=str(len(set(seq))),
            fill_snippet=f"items = {seq!r}\nprint(len(____(items)))", fill_answer="set",
            distractors=[str(len(seq)), str(len(set(seq)) + 1), "0"],
            multi_true=["A set removes duplicate values.", "A tuple does not support item assignment."],
            multi_false=["A set preserves every duplicate.", "Tuples can always be changed in place.", "Set elements may be mutable lists."],
            explanation="Converting the list to a set retains one copy of each distinct integer.",
        )
    elif sid == "SK016":
        key = rng.choice(["math", "python", "speech", "ml"])
        score = rng.randint(50, 100)
        mapping = {key: score, "other": score - 5}
        case.update(
            concept_question="How does a dictionary retrieve the value associated with a key?",
            concept_answer="Dictionary lookup uses a key; an absent key raises KeyError unless get or a default is used.",
            snippet=f"scores = {mapping!r}\nprint(scores[{key!r}])", output=str(score),
            fill_snippet=f"scores = {mapping!r}\nprint(scores[____])", fill_answer=repr(key),
            distractors=[str(score - 5), repr(key), "None"],
            multi_true=["Dictionaries map keys to values.", "dict.get can return a default for a missing key."],
            multi_false=["Dictionary keys must be lists.", "Lookup by a missing key always returns zero.", "Dictionaries can store only numeric values."],
            explanation="The key maps directly to the stored score.",
            coding_prompt="Implement get_score(scores, name), returning the score or 0 if the name is absent.",
            starter_code="def get_score(scores, name):\n    pass\n",
            reference_solution="def get_score(scores, name):\n    return scores.get(name, 0)\n",
            function_name="get_score",
            test_cases=[{"args": [{"Ada": 9, "Lin": 8}, "Ada"], "expected": 9}, {"args": [{"Ada": 9}, "Noor"], "expected": 0}],
            buggy_solution="def get_score(scores, name):\n    return scores[name]\n",
        )
    elif sid == "SK017":
        case.update(
            concept_question="What is the purpose of a function parameter?",
            concept_answer="A parameter is a local name that receives an argument when a function is called.",
            snippet=f"def combine(left, right):\n    return left + right\n\nprint(combine({a}, {b}))", output=str(a + b),
            fill_snippet=f"def combine(left, right):\n    return left + right\n\nprint(combine({a}, ____))", fill_answer=str(b),
            distractors=[str(a), str(a * b), str(a + b + 1)],
            multi_true=["Parameters receive argument values at call time.", "A function can be called more than once."],
            multi_false=["All functions must accept exactly one parameter.", "A parameter is the same as a global variable.", "Function definitions execute their body immediately."],
            explanation="The call binds left to a and right to b, then returns their sum.",
            coding_prompt="Implement clamp(value, lower, upper), returning value constrained to the inclusive interval [lower, upper].",
            starter_code="def clamp(value, lower, upper):\n    pass\n",
            reference_solution="def clamp(value, lower, upper):\n    return max(lower, min(value, upper))\n",
            function_name="clamp",
            test_cases=[{"args": [5, 0, 10], "expected": 5}, {"args": [-2, 0, 10], "expected": 0}, {"args": [14, 0, 10], "expected": 10}],
            buggy_solution="def clamp(value, lower, upper):\n    return value\n",
        )
    elif sid == "SK018":
        value = a
        case.update(
            concept_question="What is the difference between returning and printing a value?",
            concept_answer="return sends a result to the caller; print displays text and does not return that displayed value.",
            snippet=f"def add_one(value):\n    result = value + 1\n    return result\n\nanswer = add_one({value})\nprint(answer)", output=str(value + 1),
            fill_snippet=f"def add_one(value):\n    result = value + 1\n    ____ result\n\nanswer = add_one({value})\nprint(answer)", fill_answer="return",
            distractors=[str(value), "None", str(value + 2)],
            multi_true=["A local name is normally available only in its scope.", "return passes a value back to the caller."],
            multi_false=["print returns the displayed text by default.", "A local variable is automatically global.", "A function can return only integers."],
            explanation="The function returns value + 1; the caller stores and prints it.",
            buggy_code=f"def add_one(value):\n    result = value + 1\n    print(result)\n\nanswer = add_one({value})\nprint(answer)",
            fixed_code=f"def add_one(value):\n    result = value + 1\n    return result\n\nanswer = add_one({value})\nprint(answer)", fixed_line="return result",
        )
    elif sid == "SK019":
        upper = rng.randint(4, 7)
        case.update(
            concept_question="How does the list comprehension construct its result?",
            concept_answer="A list comprehension evaluates an expression for each item and can optionally filter items with if.",
            snippet=f"squares = [x * x for x in range({upper})]\nprint(squares)", output=repr([x * x for x in range(upper)]),
            fill_snippet=f"squares = [x * x for x in ____({upper})]\nprint(squares)", fill_answer="range",
            distractors=[repr([x for x in range(upper)]), repr([x * x for x in range(upper + 1)]), repr([x * x for x in range(1, upper)])],
            multi_true=["The expression is evaluated for each item yielded by the iterable.", "An if clause can filter generated values."],
            multi_false=["A list comprehension always returns a set.", "The expression runs only once.", "Comprehensions cannot contain a for clause."],
            explanation="range(upper) yields 0 through upper-1 and each item is squared.",
        )
    elif sid == "SK020":
        raw = rng.choice(["abc", "4.2", "--"])
        case.update(
            concept_question="Which exception is raised when int receives this invalid integer string?",
            concept_answer="Invalid integer text raises ValueError; catch the specific expected error when recovery is appropriate.",
            snippet=f"text = {_literal(raw)}\ntry:\n    number = int(text)\n    print(number)\nexcept ValueError:\n    print('invalid')", output="invalid",
            fill_snippet=f"text = {_literal(raw)}\ntry:\n    number = int(text)\nexcept ____:\n    number = 0", fill_answer="ValueError",
            distractors=["KeyError", "IndexError", "NameError"],
            multi_true=["A try block can be paired with except handlers.", "Catching a specific exception makes the intended recovery clearer."],
            multi_false=["Every exception should be silently ignored.", "ValueError means a variable name is undefined.", "except must always appear before try."],
            explanation="int cannot parse this input as an integer, so the ValueError handler runs.",
            buggy_code=f"text = {_literal(raw)}\ntry:\n    number = int(text)\nexcept KeyError:\n    number = 0\nprint(number)",
            fixed_code=f"text = {_literal(raw)}\ntry:\n    number = int(text)\nexcept ValueError:\n    number = 0\nprint(number)", fixed_line="except ValueError:",
        )
    elif sid == "SK021":
        value = rng.randint(3, 6)
        case.update(
            concept_question="Why does this recursive function terminate?",
            concept_answer="A recursive function needs a base case and a recursive step that moves toward it.",
            snippet=f"def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nprint(factorial({value}))", output=str(__import__("math").factorial(value)),
            fill_snippet=f"def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(____)\n\nprint(factorial({value}))", fill_answer="n - 1",
            distractors=[str(value), str(__import__("math").factorial(value - 1)), "RecursionError"],
            multi_true=["A base case stops recursive calls.", "The recursive argument should approach the base case."],
            multi_false=["Every recursive function must call itself forever.", "A base case is optional when using return.", "Recursion cannot return a value."],
            explanation="Each call reduces n by one until n <= 1, then the products unwind.",
            buggy_code=f"def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n)\n\nprint(factorial({value}))",
            fixed_code=f"def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nprint(factorial({value}))", fixed_line="return n * factorial(n - 1)",
            coding_prompt="Implement factorial(n) recursively for non-negative integers.",
            starter_code="def factorial(n):\n    pass\n",
            reference_solution="def factorial(n):\n    if n < 0:\n        raise ValueError('n must be non-negative')\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n",
            function_name="factorial",
            test_cases=[{"args": [0], "expected": 1}, {"args": [1], "expected": 1}, {"args": [5], "expected": 120}],
            buggy_solution="def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n)\n",
        )
    elif sid == "SK022":
        lines = ["first line", "", "third line"]
        case.update(
            concept_question="Why is a context manager useful when working with files?",
            concept_answer="A with statement manages resource cleanup, and line processing should account for blank lines and newline characters.",
            snippet=f"from io import StringIO\ntext = {_literal(chr(10).join(lines))}\nwith StringIO(text) as handle:\n    print(handle.readline().strip())\n    print(handle.readline().strip() == '')",
            output="first line\nTrue",
            fill_snippet=f"from io import StringIO\ntext = {_literal(chr(10).join(lines))}\nwith StringIO(text) as handle:\n    print(handle.readline().____())", fill_answer="strip",
            distractors=["first line\nFalse", "third line\nTrue", "first line\nNone"],
            multi_true=["with can ensure a resource is closed after its block.", "readline returns the next line or an empty string at end-of-file."],
            multi_false=["A file handle should always be left open.", "readline returns all lines at once.", "strip writes text to a file."],
            explanation="StringIO provides a file-like stream; strip removes newline whitespace from the first line.",
            coding_prompt="Implement count_nonempty_lines(text), counting lines that contain non-whitespace characters.",
            starter_code="def count_nonempty_lines(text):\n    pass\n",
            reference_solution="def count_nonempty_lines(text):\n    return sum(1 for line in text.splitlines() if line.strip())\n",
            function_name="count_nonempty_lines",
            test_cases=[{"args": ["one\n\nthree\n"], "expected": 2}, {"args": ["  \n\t\n"], "expected": 0}, {"args": ["single"], "expected": 1}],
            buggy_solution="def count_nonempty_lines(text):\n    return len(text.splitlines())\n",
        )
    elif sid == "SK023":
        amount = rng.randint(2, 8)
        case.update(
            concept_question="What does self refer to inside an instance method?",
            concept_answer="self conventionally refers to the instance on which the method was called.",
            snippet=f"class Box:\n    def __init__(self, size):\n        self.size = size\n\nbox = Box({amount})\nprint(box.size)", output=str(amount),
            fill_snippet=f"class Box:\n    def __init__(self, size):\n        self.____ = size\n\nbox = Box({amount})\nprint(box.size)", fill_answer="size",
            distractors=[str(amount + 1), "None", "AttributeError"],
            multi_true=["An instance stores attributes on the object.", "self is the conventional first parameter of an instance method."],
            multi_false=["A class can have only one instance.", "self is a Python keyword that must be used verbatim in every method.", "__init__ must return the new instance explicitly."],
            explanation="The constructor stores size on the Box instance, and box.size reads it.",
        )
    elif sid == "SK024":
        amount = rng.randint(5, 12)
        case.update(
            concept_question="What does this example illustrate about mutation?",
            concept_answer="append mutates a list in place; assigning a new string creates a new string value because strings are immutable.",
            snippet=f"items = [{a}, {b}]\nitems.append({amount})\nprint(items)", output=repr([a, b, amount]),
            fill_snippet=f"items = [{a}, {b}]\nitems.____({amount})\nprint(items)", fill_answer="append",
            distractors=[repr([a, b]), repr([amount, a, b]), repr([a, b, amount, amount])],
            multi_true=["Lists are mutable.", "Strings are immutable sequences."],
            multi_false=["append returns a new list and leaves the original unchanged.", "Tuples support item assignment.", "Every Python object is immutable."],
            explanation="append changes the existing list by adding the new element.",
        )
    elif sid == "SK025":
        amount = rng.randint(5, 12)
        case.update(
            concept_question="Why does changing alias affect the value observed through original?",
            concept_answer="Both names refer to the same mutable list; mutating through either name changes that shared object.",
            snippet=f"original = [{a}, {b}]\nalias = original\nalias.append({amount})\nprint(original)", output=repr([a, b, amount]),
            fill_snippet=f"original = [{a}, {b}]\nalias = ____\nalias.append({amount})\nprint(original)", fill_answer="original",
            distractors=[repr([a, b]), repr([amount]), repr([a, b, amount, amount])],
            multi_true=["Assignment can make two names refer to the same list.", "A shallow copy can separate the outer list object."],
            multi_false=["alias = original always clones the list.", "append changes only the name alias.", "Lists are immutable."],
            explanation="No copy is made: original and alias refer to the same list object.",
            buggy_code=f"original = [{a}, {b}]\nalias = original\nalias.append({amount})\nprint(original)",
            fixed_code=f"original = [{a}, {b}]\nalias = original.copy()\nalias.append({amount})\nprint(original)", fixed_line="alias = original.copy()",
        )
    elif sid == "SK026":
        target = rng.randint(2, 8)
        seq = [rng.randint(1, 8) for _ in range(5)]
        case.update(
            concept_question="What state must an algorithm maintain to count target occurrences?",
            concept_answer="Maintain a counter, inspect each item, and increment only when it matches the target.",
            snippet=f"values = {seq!r}\ntarget = {target}\ncount = 0\nfor value in values:\n    if value == target:\n        count += 1\nprint(count)", output=str(seq.count(target)),
            fill_snippet=f"values = {seq!r}\ntarget = {target}\ncount = 0\nfor value in values:\n    if value == target:\n        count ____ 1\nprint(count)", fill_answer="+=",
            distractors=[str(len(seq)), str(seq.count(target) + 1), str(max(seq))],
            multi_true=["A counter can summarize how many matches have been seen.", "A loop invariant can describe what the counter represents after each iteration."],
            multi_false=["An algorithm must use recursion.", "A counter should increment on every item regardless of the condition.", "An algorithm cannot use loops."],
            explanation="The counter increases once for each element equal to target.",
            coding_prompt="Implement count_occurrences(values, target), returning how many values equal target.",
            starter_code="def count_occurrences(values, target):\n    pass\n",
            reference_solution="def count_occurrences(values, target):\n    count = 0\n    for value in values:\n        if value == target:\n            count += 1\n    return count\n",
            function_name="count_occurrences",
            test_cases=[{"args": [[1, 2, 1, 3], 1], "expected": 2}, {"args": [[2, 2], 3], "expected": 0}, {"args": [[], 0], "expected": 0}],
            buggy_solution="def count_occurrences(values, target):\n    return len(values)\n",
        )
    elif sid == "SK027":
        size = rng.randint(5, 12)
        case.update(
            concept_question="What is the time complexity of one loop that visits each of n items once?",
            concept_answer="A single constant-work pass over n items has linear time complexity, O(n).",
            snippet=f"n = {size}\nfor i in range(n):\n    pass\nprint('done')", output="done",
            fill_snippet=f"# A single pass through n items takes ____ time\nprint('done')", fill_answer="O(n)",
            distractors=["O(1)", "O(log n)", "O(n^2)"],
            multi_true=["A single pass with constant work per item is O(n).", "Two nested loops over n items often lead to O(n^2) work."],
            multi_false=["O(n) is constant time.", "Big-O gives exact wall-clock runtime.", "Every loop is O(1), regardless of its number of iterations."],
            explanation="The loop body executes n times with constant work per iteration; the growth is linear.",
        )

    # Every question case includes a known set of plausible incorrect responses.
    # The generator uses these pre-defined alternatives instead of fabricating code.
    case.setdefault("misconception_tags", SKILL_TAGS.get(sid, ["concept_confusion"]))
    case.setdefault("wrong_answers", case["distractors"][:])
    case["snippet"] = case["snippet"].replace("{index}", str(0))
    case["fill_snippet"] = case["fill_snippet"].replace("{index}", str(0))
    case["case_variant"] = variant
    return case


def _question_types_for(skill_id: str, skill_index: int, variant: int) -> str:
    """Cycle formats while reserving code-execution formats for suitable skills."""
    types = ["mcq_single", "mcq_multi", "code_trace", "fill_blank", "short_answer"]
    if skill_id in DEBUGGING_SKILLS:
        types.append("debugging")
    if skill_id in CODING_SKILLS:
        types.append("coding")
    return types[(variant + skill_index) % len(types)]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def make_template_question(skill: Skill, skill_index: int, variant: int, rng: random.Random, problem_id: str) -> dict[str, Any]:
    """Create one fully-structured, deterministic template question."""
    case = _case_for(skill, variant, rng)
    problem_type = _question_types_for(skill.skill_id, skill_index, variant)
    base = {
        "problem_id": problem_id,
        "Problem Body": "",
        "Problem Type": problem_type,
        "Multiple Choice Options": "[]",
        "Multiple Choice Answers": "[]",
        "Fill-in Answers": "[]",
        "difficulty": skill.level if skill.level in DIFFICULTIES else "beginner",
        "primary_skill_id": skill.skill_id,
        "secondary_skill_ids": _json(list(skill.prerequisite_skill_ids[:2])),
        "misconception_tags": _json(case["misconception_tags"]),
        "reference_solution": "",
        "explanation": case["explanation"],
        "starter_code": "",
        "test_cases_json": "[]",
        "expected_output": "",
        "answer_key_json": "{}",
        "estimated_time_seconds": int(case["estimated_time_seconds"] + (15 if skill.level == "intermediate" else 30 if skill.level == "advanced" else 0)),
        "generation_source": "template",
        "trace_code": case["snippet"],
        "function_name": "",
        "response_templates_json": _json({"wrong_answers": case["wrong_answers"], "buggy_code": case.get("buggy_code", ""), "fixed_code": case.get("fixed_code", ""), "fixed_line": case.get("fixed_line", ""), "buggy_solution": case.get("buggy_solution", "")}),
    }
    if problem_type == "mcq_single":
        body = f"What is printed by this Python code?\n```python\n{case['snippet']}\n```"
        options = [case["output"], *[v for v in case["distractors"] if v != case["output"]]]
        options = list(dict.fromkeys(options))[:4]
        while len(options) < 4:
            options.append(f"No output ({len(options)})")
        rng.shuffle(options)
        answer_ids = [chr(ord("A") + options.index(case["output"]))]
        option_objects = [{"id": chr(ord("A") + i), "text": str(option)} for i, option in enumerate(options)]
        base.update({
            "Problem Body": body,
            "Multiple Choice Options": _json(option_objects),
            "Multiple Choice Answers": _json(answer_ids),
            "reference_solution": case["snippet"],
            "expected_output": case["output"],
            "answer_key_json": _json({"correct_option_ids": answer_ids}),
        })
    elif problem_type == "mcq_multi":
        statements = list(case["multi_true"][:2]) + list(case["multi_false"][:3])
        statements = list(dict.fromkeys(statements))
        while len(statements) < 5:
            statements.append(f"This statement is a distractor for {skill.skill_name.lower()}.")
        statements = statements[:5]
        correct_texts = set(case["multi_true"][:2])
        rng.shuffle(statements)
        option_objects = [{"id": chr(ord("A") + i), "text": text} for i, text in enumerate(statements)]
        answer_ids = [option["id"] for option in option_objects if option["text"] in correct_texts]
        base.update({
            "Problem Body": f"Select all correct statements about {skill.skill_name.lower()}.",
            "Multiple Choice Options": _json(option_objects),
            "Multiple Choice Answers": _json(answer_ids),
            "reference_solution": "\n".join(case["multi_true"][:2]),
            "answer_key_json": _json({"correct_option_ids": answer_ids}),
        })
    elif problem_type == "code_trace":
        base.update({
            "Problem Body": f"Predict the exact standard output of this Python code.\n```python\n{case['snippet']}\n```",
            "reference_solution": case["snippet"],
            "expected_output": case["output"],
            "test_cases_json": _json([{"kind": "stdout", "expected": case["output"]}]),
            "answer_key_json": _json({"expected_output": case["output"]}),
        })
    elif problem_type == "fill_blank":
        filled_solution = case["fill_snippet"].replace("____", case["fill_answer"], 1)
        fill_output = run_snippet(filled_solution, trusted=True)
        base.update({
            "Problem Body": f"Complete the blank so the code behaves as intended.\n```python\n{case['fill_snippet']}\n```",
            "Fill-in Answers": _json([case["fill_answer"]]),
            "reference_solution": filled_solution,
            "expected_output": fill_output,
            "test_cases_json": _json([{"kind": "stdout", "expected": fill_output}]),
            "answer_key_json": _json({"accepted_answers": [case["fill_answer"]]}),
        })
    elif problem_type == "debugging":
        base.update({
            "Problem Body": f"Identify and correct the bug in this code. Submit the corrected line.\n```python\n{case['buggy_code']}\n```",
            "Fill-in Answers": _json([case["fixed_line"]]),
            "reference_solution": case["fixed_code"],
            "answer_key_json": _json({"accepted_answers": [case["fixed_line"]]}),
            "expected_output": "",
        })
    elif problem_type == "short_answer":
        base.update({
            "Problem Body": case["concept_question"],
            "Fill-in Answers": _json([case["concept_answer"]]),
            "reference_solution": case["concept_answer"],
            "answer_key_json": _json({"accepted_answers": [case["concept_answer"]]}),
        })
    elif problem_type == "coding":
        base.update({
            "Problem Body": case["coding_prompt"],
            "reference_solution": case["reference_solution"],
            "starter_code": case["starter_code"],
            "test_cases_json": _json(case["test_cases"]),
            "function_name": case["function_name"],
            "answer_key_json": _json({"function_name": case["function_name"], "test_cases": case["test_cases"]}),
            "expected_output": "",
            "trace_code": "",
        })
    else:  # pragma: no cover - guarded by QUESTION_TYPES
        raise ValueError(f"Unsupported question type: {problem_type}")
    return base


def validate_question(question: dict[str, Any], known_skill_ids: set[str]) -> list[str]:
    """Validate a candidate's essential schema and answer-key consistency."""
    errors: list[str] = []
    required = [
        "problem_id", "Problem Body", "Problem Type", "difficulty", "primary_skill_id",
        "reference_solution", "explanation", "answer_key_json", "estimated_time_seconds",
        "generation_source", "secondary_skill_ids", "misconception_tags",
    ]
    for field in required:
        if field not in question or question[field] in (None, ""):
            errors.append(f"missing required field: {field}")
    if question.get("Problem Type") not in QUESTION_TYPES:
        errors.append(f"unsupported Problem Type: {question.get('Problem Type')!r}")
    if question.get("difficulty") not in DIFFICULTIES:
        errors.append(f"invalid difficulty: {question.get('difficulty')!r}")
    if question.get("primary_skill_id") not in known_skill_ids:
        errors.append(f"unknown primary_skill_id: {question.get('primary_skill_id')!r}")
    try:
        options = json.loads(question.get("Multiple Choice Options", "[]"))
        answers = json.loads(question.get("Multiple Choice Answers", "[]"))
        fill_answers = json.loads(question.get("Fill-in Answers", "[]"))
        key = json.loads(question.get("answer_key_json", "{}"))
        secondary = json.loads(question.get("secondary_skill_ids", "[]"))
        tags = json.loads(question.get("misconception_tags", "[]"))
        tests = json.loads(question.get("test_cases_json", "[]"))
        templates = json.loads(question.get("response_templates_json", "{}"))
    except (json.JSONDecodeError, TypeError) as exc:
        errors.append(f"malformed JSON field: {exc}")
        return errors
    if not isinstance(options, list) or not isinstance(answers, list) or not isinstance(fill_answers, list):
        errors.append("option and answer fields must serialize JSON arrays")
    if not isinstance(key, dict):
        errors.append("answer_key_json must be a JSON object")
    if not isinstance(templates, dict):
        errors.append("response_templates_json must be a JSON object")
    if not isinstance(secondary, list) or any(item not in known_skill_ids for item in secondary):
        errors.append("secondary_skill_ids must be a JSON array of known skills")
    if not isinstance(tags, list) or not tags or not all(isinstance(tag, str) and tag.strip() for tag in tags):
        errors.append("misconception_tags must be a non-empty JSON array of strings")
    try:
        estimated_seconds = float(question.get("estimated_time_seconds", 0))
        if estimated_seconds <= 0:
            errors.append("estimated_time_seconds must be positive")
    except (ValueError, TypeError):
        errors.append("estimated_time_seconds must be numeric")
    if question.get("generation_source") not in {"template", "llm"}:
        errors.append("generation_source must be 'template' or 'llm'")
    qtype = question.get("Problem Type")
    if qtype in {"mcq_single", "code_trace", "fill_blank"} and "expected_output" not in question:
        errors.append(f"{qtype} requires expected_output")
    if qtype == "code_trace" and not question.get("trace_code", "").strip():
        errors.append("code_trace requires runnable trace_code")
    if qtype == "fill_blank" and "____" not in str(question.get("Problem Body", "")):
        errors.append("fill_blank question body must contain the blank marker ____")
    if qtype in {"mcq_single", "mcq_multi"}:
        if not options or not all(isinstance(o, dict) and "id" in o and "text" in o for o in options):
            errors.append("MCQ options must be non-empty JSON objects with id and text")
        ids = [str(o.get("id")) for o in options if isinstance(o, dict)]
        if len(ids) != len(set(ids)):
            errors.append("MCQ option IDs must be unique")
        if not answers or any(answer not in ids for answer in answers):
            errors.append("MCQ answer IDs must reference valid options")
        if qtype == "mcq_single" and len(answers) != 1:
            errors.append("mcq_single must have exactly one correct option")
        if qtype == "mcq_multi" and len(answers) < 2:
            errors.append("mcq_multi must have at least two correct options")
        if isinstance(key, dict) and key.get("correct_option_ids") != answers:
            errors.append("answer_key_json correct_option_ids must match Multiple Choice Answers")
    if qtype in {"fill_blank", "debugging", "short_answer"} and not fill_answers:
        errors.append(f"{qtype} requires Fill-in Answers")
    if qtype in {"fill_blank", "debugging", "short_answer"} and isinstance(key, dict):
        if key.get("accepted_answers") != fill_answers:
            errors.append("answer_key_json accepted_answers must match Fill-in Answers")
    if qtype == "coding":
        if not question.get("starter_code") or not question.get("function_name") or not tests:
            errors.append("coding questions require starter_code, function_name, and test cases")
        if not isinstance(tests, list) or any("args" not in test or "expected" not in test for test in tests):
            errors.append("coding test cases must have args and expected fields")
        if isinstance(key, dict) and (key.get("function_name") != question.get("function_name") or key.get("test_cases") != tests):
            errors.append("coding answer_key_json must match function_name and test_cases_json")
    if not isinstance(tests, list):
        errors.append("test_cases_json must be a JSON array")
    return errors


def build_generation_plan(num_questions: int, skills: Iterable[Skill] = SKILLS) -> list[tuple[Skill, int, str]]:
    """Create a balanced round-robin skill/type plan for either backend."""
    skill_list = list(skills)
    if not skill_list:
        raise ValueError("At least one skill must be selected")
    plan: list[tuple[Skill, int, str]] = []
    for index in range(num_questions):
        skill_index = index % len(skill_list)
        variant = index // len(skill_list)
        skill = skill_list[skill_index]
        plan.append((skill, variant, _question_types_for(skill.skill_id, skill_index, variant)))
    return plan


def _normalise_question_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def generate_template_questions(
    num_questions: int, seed: int, skills: Iterable[Skill] = SKILLS,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    """Generate candidates, reject invalid/duplicate questions, and report counts."""
    if num_questions < 1:
        raise ValueError("num_questions must be at least 1")
    selected_skills = list(skills)
    if not selected_skills:
        raise ValueError("At least one skill must be selected")
    rng = random.Random(seed)
    known_ids = {skill.skill_id for skill in SKILLS}
    plan = build_generation_plan(num_questions, selected_skills)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    type_counts: Counter[str] = Counter()
    generated_count = 0
    for idx, (skill, variant, _planned_type) in enumerate(plan, start=1):
        try:
            candidate = make_template_question(skill, selected_skills.index(skill), variant, rng, f"PY{idx:06d}")
        except Exception as exc:
            rejected.append({"candidate_id": f"PY{idx:06d}", "reason": f"candidate construction failed: {type(exc).__name__}: {exc}"})
            continue
        generated_count += 1
        errors = validate_question(candidate, known_ids)
        fingerprint = (candidate["Problem Type"], _normalise_question_text(candidate["Problem Body"]))
        if fingerprint in seen:
            errors.append("duplicate question body for the same type")
        if errors:
            rejected.append({"candidate_id": candidate.get("problem_id", f"candidate-{idx}"), "reason": "; ".join(errors)})
            continue
        seen.add(fingerprint)
        accepted.append(candidate)
        type_counts[candidate["Problem Type"]] += 1
    report = {
        "mode": "template",
        "requested": num_questions,
        "generated": generated_count,
        "attempted": len(plan),
        "rejected": len(rejected),
        "accepted": len(accepted),
        "question_type_counts": dict(type_counts),
        "skill_coverage_requested": [skill.skill_id for skill in selected_skills],
        "seed": seed,
    }
    return accepted, rejected, report


def generate_llm_questions(
    plan: list[tuple[Skill, int, str]],
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_seconds: float = 45.0,
    retry_limit: int = 2,
    batch_size: int = 8,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    """Generate question candidates using an OpenAI-compatible chat-completions API.

    The backend is optional and deliberately isolated here. Each response is
    schema-validated before acceptance; coding solutions are validated by the
    orchestrator's executable test stage. Failed candidates remain in the report.
    """
    from generation.llm_backend import request_question_batch

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    known_ids = {skill.skill_id for skill in SKILLS}
    raw_candidate_count = 0
    for start in range(0, len(plan), max(1, batch_size)):
        batch = plan[start:start + max(1, batch_size)]
        try:
            generated = request_question_batch(
                batch, base_url=base_url, model=model, api_key=api_key,
                timeout_seconds=timeout_seconds, retry_limit=retry_limit, seed=seed + start,
            )
        except Exception as exc:  # preserve the batch failure and continue no farther silently
            for offset, (skill, variant, qtype) in enumerate(batch):
                rejected.append({"candidate_id": f"LLM{start + offset + 1:06d}", "reason": f"LLM batch failed: {type(exc).__name__}: {exc}"})
            continue
        raw_candidate_count += len(generated)
        for offset, item in enumerate(generated):
            plan_idx = start + offset
            if offset >= len(batch):
                rejected.append({"candidate_id": f"LLM-EXTRA-{plan_idx + 1:06d}", "reason": "LLM returned more questions than requested"})
                continue
            if not isinstance(item, dict):
                rejected.append({"candidate_id": f"LLM{plan_idx + 1:06d}", "reason": f"LLM candidate must be a JSON object, received {type(item).__name__}"})
                continue
            skill, _variant, expected_type = batch[offset]
            item["problem_id"] = f"PY{plan_idx + 1:06d}"
            item["generation_source"] = "llm"
            item.setdefault("secondary_skill_ids", "[]")
            item.setdefault("misconception_tags", "[]")
            item.setdefault("Multiple Choice Options", "[]")
            item.setdefault("Multiple Choice Answers", "[]")
            item.setdefault("Fill-in Answers", "[]")
            item.setdefault("reference_solution", "")
            item.setdefault("explanation", "")
            item.setdefault("starter_code", "")
            item.setdefault("test_cases_json", "[]")
            item.setdefault("expected_output", "")
            item.setdefault("answer_key_json", "{}")
            item.setdefault("estimated_time_seconds", 120)
            item.setdefault("trace_code", "")
            item.setdefault("function_name", "")
            item.setdefault("response_templates_json", "{}")
            item["primary_skill_id"] = skill.skill_id
            if item.get("Problem Type") != expected_type:
                rejected.append({"candidate_id": item.get("problem_id", f"LLM{plan_idx + 1:06d}"), "reason": f"expected type {expected_type}, received {item.get('Problem Type')}"})
                continue
            errors = validate_question(item, known_ids)
            if errors:
                rejected.append({"candidate_id": item.get("problem_id", f"LLM{plan_idx + 1:06d}"), "reason": "; ".join(errors)})
                continue
            candidates.append(item)
        for missing_offset in range(len(generated), len(batch)):
            skill, _variant, expected_type = batch[missing_offset]
            rejected.append({"candidate_id": f"LLM{start + missing_offset + 1:06d}", "reason": f"LLM returned fewer questions than requested; missing {expected_type} for {skill.skill_id}"})
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for question in candidates:
        fingerprint = (question["Problem Type"], _normalise_question_text(question["Problem Body"]))
        if fingerprint in seen:
            rejected.append({"candidate_id": question["problem_id"], "reason": "duplicate question body for the same type"})
        else:
            seen.add(fingerprint)
            unique.append(question)
    type_counts = Counter(q["Problem Type"] for q in unique)
    report = {
        "mode": "llm", "requested": len(plan), "generated": raw_candidate_count,
        "rejected": len(rejected), "accepted": len(unique), "question_type_counts": dict(type_counts), "seed": seed,
    }
    return unique, rejected, report
