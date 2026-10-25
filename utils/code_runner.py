"""Timeout-bounded runner for snippets and callable solutions.

Trusted template code runs in-process for speed. Optional LLM-generated code is
run in a child process with a timeout. A subprocess timeout is not a security
sandbox; do not run arbitrary code in a privileged environment.
"""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from typing import Any


class CodeExecutionError(RuntimeError):
    """Raised when a snippet fails, times out, or returns an invalid result."""


def run_snippet(source: str, timeout_seconds: float = 2.0, *, trusted: bool = True) -> str:
    """Execute a Python snippet and return stdout, raising on non-zero exit."""
    if trusted:
        output = io.StringIO()
        namespace: dict[str, Any] = {"__name__": "__main__"}
        try:
            with contextlib.redirect_stdout(output):
                exec(compile(source, "<generated-question>", "exec"), namespace, namespace)
        except Exception as exc:
            raise CodeExecutionError(f"{type(exc).__name__}: {exc}") from exc
        return output.getvalue().rstrip("\n")
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", source], capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CodeExecutionError(f"snippet timed out after {timeout_seconds}s") from exc
    if completed.returncode != 0:
        raise CodeExecutionError((completed.stderr or "snippet failed").strip()[:1200])
    return completed.stdout.rstrip("\n")


def run_function_tests(
    source: str,
    function_name: str,
    test_cases: list[dict[str, Any]],
    timeout_seconds: float = 3.0,
    *,
    trusted: bool = True,
) -> tuple[int, int, list[dict[str, Any]]]:
    """Run callable tests and return passed, total, and structured result details."""
    if trusted:
        namespace: dict[str, Any] = {"__name__": "__generated_solution__"}
        try:
            exec(compile(source, "<generated-solution>", "exec"), namespace, namespace)
            function = namespace[function_name]
        except Exception as exc:
            raise CodeExecutionError(f"cannot load function {function_name}: {type(exc).__name__}: {exc}") from exc
        results: list[dict[str, Any]] = []
        for test in test_cases:
            try:
                actual = function(*test.get("args", []), **test.get("kwargs", {}))
                expected = test.get("expected")
                results.append({"ok": actual == expected, "actual": actual, "expected": expected})
            except Exception as exc:
                results.append({"ok": False, "error": f"{type(exc).__name__}: {exc}", "expected": test.get("expected")})
        return sum(result.get("ok") is True for result in results), len(test_cases), results

    harness = r'''
import json, sys
payload = json.load(sys.stdin)
namespace = {}
exec(compile(payload["source"], "<candidate>", "exec"), namespace, namespace)
function = namespace[payload["function_name"]]
results = []
for test in payload["test_cases"]:
    try:
        result = function(*test.get("args", []), **test.get("kwargs", {}))
        results.append({"ok": result == test.get("expected"), "actual": result, "expected": test.get("expected")})
    except Exception as error:
        results.append({"ok": False, "error": type(error).__name__ + ": " + str(error), "expected": test.get("expected")})
print(json.dumps(results, ensure_ascii=False, default=repr))
'''
    payload = json.dumps({"source": source, "function_name": function_name, "test_cases": test_cases})
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", harness], input=payload, capture_output=True,
            text=True, timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CodeExecutionError(f"test suite timed out after {timeout_seconds}s") from exc
    if completed.returncode != 0:
        raise CodeExecutionError((completed.stderr or "test harness failed").strip()[:1200])
    try:
        results = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CodeExecutionError("test harness produced malformed JSON") from exc
    passed = sum(1 for result in results if result.get("ok") is True)
    return passed, len(test_cases), results
