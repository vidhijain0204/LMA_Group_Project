"""Minimal configurable OpenAI-compatible question-generation client."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from generation.skills import Skill


def _schema_prompt(batch: list[tuple[Skill, int, str]]) -> str:
    specs = [
        {"primary_skill_id": s.skill_id, "skill_name": s.skill_name, "difficulty": s.level, "Problem Type": qtype}
        for s, _variant, qtype in batch
    ]
    return (
        f"Generate exactly {len(batch)} distinct Python programming questions. Return a JSON object with key `questions` "
        "whose value is an array in the same order as these specs: " + json.dumps(specs) + "\n"
        "Every question object must contain: problem_id, Problem Body, Problem Type, Multiple Choice Options, "
        "Multiple Choice Answers, Fill-in Answers, difficulty, primary_skill_id, secondary_skill_ids, misconception_tags, "
        "reference_solution, explanation, starter_code, test_cases_json, expected_output, answer_key_json, "
        "estimated_time_seconds, trace_code, function_name. Array/object fields must be valid JSON strings in the object, "
        "for example Multiple Choice Options='[{\"id\":\"A\",\"text\":\"...\"}]', not Python reprs. "
        "For mcq_single provide exactly one answer option ID; for mcq_multi at least two. Options are objects with id/text. "
        "For code_trace provide runnable trace_code and expected_output. For fill_blank, debugging, and short_answer provide "
        "Fill-in Answers. For coding provide starter_code, function_name, runnable reference_solution, and test_cases_json "
        "as a JSON array of {\"args\":[...],\"expected\":...} objects. Don't put answers in Problem Body. "
        "Use each spec's Problem Type, skill ID, and difficulty. Keep all questions unambiguous."
    )


def request_question_batch(
    batch: list[tuple[Skill, int, str]], *, base_url: str, model: str, api_key: str,
    timeout_seconds: float = 45.0, retry_limit: int = 2, seed: int = 42,
) -> list[Any]:
    """Call /chat/completions and parse a JSON array, retrying transient failures."""
    if not base_url.strip() or not model.strip() or not api_key.strip():
        raise ValueError("LLM mode requires base_url, model, and an API key")
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "Generate precise educational Python questions. Return strict JSON only."},
            {"role": "user", "content": _schema_prompt(batch)},
        ],
        "response_format": {"type": "json_object"},
        "seed": seed,
    }
    last_error: Exception | None = None
    for attempt in range(max(1, retry_limit + 1)):
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                envelope = json.loads(response.read().decode("utf-8"))
            content = envelope["choices"][0]["message"]["content"]
            decoded = json.loads(content)
            questions = decoded.get("questions") if isinstance(decoded, dict) else decoded
            if not isinstance(questions, list):
                raise ValueError("LLM response did not contain a JSON question array")
            return questions
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retry_limit:
                time.sleep(min(2 ** attempt, 4))
    raise RuntimeError(f"LLM request failed after {max(1, retry_limit + 1)} attempts: {last_error}")
