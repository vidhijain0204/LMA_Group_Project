"""Transparent learner mastery updates, separated from response simulation."""
from __future__ import annotations

from dataclasses import dataclass


MASTERY_THRESHOLD = 0.70


@dataclass(frozen=True)
class LearningEvent:
    correct: bool
    used_hint: bool
    saw_answer: bool
    learning_rate: float
    hint_benefit: float
    worked_example_benefit: float


def decay_mastery(mastery: float, retention_rate_per_day: float, elapsed_hours: float) -> float:
    """Apply exponential time decay; retention_rate_per_day is in [0, 1]."""
    if elapsed_hours <= 0:
        return min(1.0, max(0.0, mastery))
    retention_rate_per_day = min(1.0, max(0.0, retention_rate_per_day))
    decayed = mastery * (retention_rate_per_day ** (elapsed_hours / 24.0))
    return min(1.0, max(0.0, decayed))


def update_mastery(mastery: float, event: LearningEvent) -> float:
    """Update continuous mastery; solution exposure is not treated as mastery.

    Independent success earns a larger increment. Hints and worked examples
    provide smaller learning opportunities, and a single error cannot reset skill.
    """
    mastery = min(1.0, max(0.0, mastery))
    lr = min(1.0, max(0.0, event.learning_rate))
    if event.correct and not event.saw_answer:
        factor = 0.10 + (0.055 * event.hint_benefit if event.used_hint else 0.0)
    elif event.correct and event.saw_answer:
        factor = 0.025 * event.worked_example_benefit
    elif event.saw_answer:
        factor = 0.025 * event.worked_example_benefit
    elif event.used_hint:
        factor = 0.012 * event.hint_benefit
    else:
        factor = 0.002
    new_value = mastery + (lr * factor * (1.0 - mastery))
    return min(0.995, max(0.0, new_value))
