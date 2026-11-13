"""Find the most likely root cause of a student's quiz errors.

Where it fits: after a quiz, agents/workflow.py grades it (error_analysis_agent.py) and then
calls RootCauseAgent.analyze() (the method chosen in the app) and RootCauseAgent.compare()
(every method, for evaluation). The result names the subtopic (and, when known, the
misconception) that most likely caused the wrong answers; the app uses it for the
remediation lesson and the re-test. It reads the topic catalog and the LS value from
struggle_tracker.py and the student's record from the SKB (data/skb_database.json).

Every quiz question is tagged with two subtopics, so a wrong answer could be caused by either of
them. Three methods are implemented so they can be compared during evaluation:

    no_kb         Baseline without the knowledge base: the subtopic with the most wrong answers
                  in this quiz (ties broken by the lowest quiz score).
    kb_low_score  Uses the knowledge base directly: among the subtopics on wrongly answered
                  questions, the one with the lowest stored mastery.
    hypothesis    Hypothesis testing: the top 3 candidate subtopics (each one a "hypothesis" H)
                  are scored with five evidence parameters (all scaled to 0..1) and a weight set.

Hypothesis parameters:

    LS   Learning Struggle       the student struggled in the tutor conversation (struggle_tracker.py)
    CE   Counter-Evidence        H answered correctly elsewhere (this quiz + recent quizzes); subtracted
    RP   Repeat                  H was also wrong in recent quizzes
    PW   Prerequisite Weakness   a prerequisite topic of H is weak
    PM'  Prior Mastery gap       1 - mastery, raised when learning was a struggle (fragile mastery)

    PM'(H)    = min(1, (1 - M) + LAMBDA * LS * M)          M = stored mastery, LAMBDA = 0.5
    Score(H)  = wLS*LS - wCE*CE + wRP*RP + wPW*PW + wPM*PM'
    P(H)      = max(Score(H), 0) / sum of max(Score, 0) over the candidates

In words: each piece of evidence adds to the score, counter-evidence subtracts from it, and the
scores are turned into probabilities that add up to 1 (negative scores count as 0). The
candidate with the highest probability is the root cause.

The weights are equal (a plain average) unless custom weights are given. Weights fitted on
labelled quizzes by scripts/fit_weights.py are added as the "learned" preset and are used as
the starting values of the custom-weight sliders in the app.

History used for CE and RP: the recent quizzes plus the diagnosis test (its questions are tagged
with subtopics), so these parameters are informative from the first quiz on.

Main contents: weight presets and resolve_weights(), the RootCauseAgent class (grading, the
three methods, analyze() and compare()), and formatting helpers for the app and logs.

Misconceptions: when a quiz question tags its wrong options with the misconception they reveal
("option_misconceptions": {"B": "sort_returns_none"}, see scripts/tag_mcq_misconceptions.py),
the root cause also names the most frequent misconception among the student's wrong answers on
that subtopic - what the student most likely believes.
"""

import json
import os
from collections import Counter

from typing import Any, Dict, List, Optional

from agents.struggle_tracker import TOPICS_FILEPATH, learning_struggle, load_catalog

TOP_CANDIDATES = 3          # hypotheses scored by the hypothesis method
RECENT_ASSESSMENTS = 3      # how many earlier quizzes count for CE and RP
FRAGILITY_LAMBDA = 0.5      # LAMBDA in the PM' formula
DEFAULT_MASTERY = 0.5       # used when the SKB has no mastery value for a subtopic or topic

PARAMS = ("LS", "CE", "RP", "PW", "PM")
PARAM_LABELS = {
    "LS": "Learning Struggle",
    "CE": "Counter-Evidence",
    "RP": "Repeat",
    "PW": "Prerequisite Weakness",
    "PM": "Prior Mastery gap (PM')",
}
SHORT_LABELS = {"LS": "LS", "CE": "CE", "RP": "RP", "PW": "PW", "PM": "PM'"}
# CE is evidence *against* a hypothesis, so its weight is subtracted.
SIGNS = {"LS": 1, "CE": -1, "RP": 1, "PW": 1, "PM": 1}

# Weight magnitudes (the sign comes from SIGNS). By default every parameter gets the same
# weight, so the score is a plain average of the five evidence values. Custom weights can be
# set in the app sidebar instead.
EQUAL_WEIGHTS: Dict[str, float] = {"LS": 0.20, "CE": 0.20, "RP": 0.20, "PW": 0.20, "PM": 0.20}
WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {"equal": EQUAL_WEIGHTS}
DEFAULT_PRESET = "equal"

# Weights fitted on labelled quizzes (scripts/fit_weights.py) become the "learned" preset.
LEARNED_WEIGHTS_FILE = "data/evaluation/learned_weights.json"


def _load_learned_weights(path: str = LEARNED_WEIGHTS_FILE) -> Optional[Dict[str, float]]:
    """Read the learned weights file; None if it is missing, broken or lacks a parameter."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            weights = json.load(f).get("weights", {})
        return {key: float(weights[key]) for key in PARAMS} if all(k in weights for k in PARAMS) else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


if os.path.exists(LEARNED_WEIGHTS_FILE) and _load_learned_weights():
    WEIGHT_PRESETS["learned"] = _load_learned_weights()

METHODS = ("no_kb", "kb_low_score", "hypothesis")
METHOD_LABELS = {
    "no_kb": "Without KB (quiz errors only)",
    "kb_low_score": "With KB (lowest mastery subtopic)",
    "hypothesis": "Hypothesis testing (5 parameters)",
}
DEFAULT_METHOD = "hypothesis"

# How each parameter is described in the reason summary, when it supports the hypothesis.
EVIDENCE_PHRASES = {
    "LS": "the student struggled with it while learning with the tutor",
    "RP": "it was also answered wrongly in earlier quizzes or the diagnosis",
    "PW": "a prerequisite topic is weak",
    "PM": "prior mastery is low or fragile",
}


def _question_id(question: dict) -> str:
    """The question's id (question banks use either "prob_id" or "problem_id")."""
    return question.get("prob_id", question.get("problem_id"))


def resolve_weights(weights: Optional[Dict[str, float]] = None, preset: Optional[str] = None) -> Dict[str, float]:
    """Return a full weight dict for the five parameters.

    Starts from the named preset (or the default one if the name is unknown/None), then
    overwrites it with any values given in `weights` (e.g. the app's custom sliders).
    """
    base = dict(WEIGHT_PRESETS.get(preset or DEFAULT_PRESET, WEIGHT_PRESETS[DEFAULT_PRESET]))
    if weights:
        base.update({key: float(value) for key, value in weights.items() if key in PARAMS})
    return base


def ablation_weights(weights: Dict[str, float], drop: str) -> Dict[str, float]:
    """Copy of `weights` with parameter `drop` set to 0 and the others scaled up to keep the
    same total. Used in compare() to test how much each parameter matters."""
    total = sum(weights.values())
    kept = {key: value for key, value in weights.items() if key != drop}
    kept_total = sum(kept.values()) or 1.0
    result = {key: round(value * total / kept_total, 4) for key, value in kept.items()}
    result[drop] = 0.0
    return result


class RootCauseAgent:
    """Finds the subtopic (and misconception) behind a student's wrong quiz answers.

    Created once by workflow.py. analyze() runs one method; compare() runs all methods,
    weight presets and ablations on the same quiz for evaluation.
    """

    def __init__(self, topics_filepath: str = TOPICS_FILEPATH):
        """Load the topic catalog (names, subtopic -> topic map, prerequisites, misconceptions)."""
        self.topics_filepath = topics_filepath
        self.catalog = load_catalog(topics_filepath)

    # ----------------------------------------------------------------- helpers
    def topic_of(self, subtopic_id: str) -> str:
        """Topic id of a subtopic (falls back to the part before the dot, "T08.3" -> "T08")."""
        return self.catalog["subtopic_topic"].get(subtopic_id, subtopic_id.split(".")[0])

    def name_of(self, subtopic_id: str) -> str:
        """Readable name of a subtopic (its id if unknown)."""
        return self.catalog["subtopic_names"].get(subtopic_id, subtopic_id)

    def topic_name_of(self, topic_id: str) -> str:
        """Readable name of a topic (its id if unknown)."""
        return self.catalog.get("topic_names", {}).get(topic_id, topic_id)

    def mastery_of(self, student_kb: Optional[dict], subtopic_id: str) -> float:
        """Stored mastery (0..1) of a subtopic: the subtopic's own value from "skills" if
        present, else its topic's value from "topics", else DEFAULT_MASTERY."""
        kb = student_kb or {}
        if subtopic_id in kb.get("skills", {}):
            return float(kb["skills"][subtopic_id])
        return float(kb.get("topics", {}).get(self.topic_of(subtopic_id), DEFAULT_MASTERY))

    @staticmethod
    def grade(questions: List[dict], answers: Dict[str, Any]) -> List[dict]:
        """Turn a quiz and the student's answers into one record per question.

        Each record holds the chosen and correct option, whether it was right, the subtopics
        the question tests, the chosen option's text, the misconception that wrong option
        reveals (if tagged) and the question's explanation. Unanswered counts as wrong.
        """
        responses = []
        for question in questions:
            chosen = answers.get(_question_id(question))
            options = question.get("options", {}) if isinstance(question.get("options"), dict) else {}
            responses.append({
                "prob_id": _question_id(question),
                "chosen": chosen,
                "correct": question.get("correct_option"),
                "is_correct": chosen is not None and chosen == question.get("correct_option"),
                "subtopics": list(question.get("related_subtopics", [])),
                "chosen_text": str(options.get(chosen, "")),
                # The misconception this wrong option reveals, when the question bank tags it.
                "misconception": (
                    (question.get("option_misconceptions") or {}).get(chosen)
                    if chosen is not None and chosen != question.get("correct_option") else None
                ),
                "explanation": str(question.get("explanations", "")),
            })
        return responses

    @staticmethod
    def _assessments(student_kb: Optional[dict]) -> List[dict]:
        """Earlier quizzes from the SKB's pedagogy_history, oldest first. The diagnosis test
        (if stored) is added as the first one."""
        kb = student_kb or {}
        quizzes = [event for event in kb.get("pedagogy_history", []) if event.get("event") == "assessment"]
        if kb.get("diagnosis_responses"):
            quizzes.insert(0, {"event": "assessment", "topic_id": "diagnosis", "source": "diagnosis",
                               "responses": kb["diagnosis_responses"]})
        return quizzes

    def misconception_of(self, subtopic_id: str, responses: List[dict]) -> Optional[dict]:
        """Most frequent tagged misconception among the wrong answers on this subtopic.

        Returns {"id", "description", "count"} or None when no wrong answer is tagged.
        """
        counts = Counter(
            r["misconception"] for r in responses
            if not r["is_correct"] and r.get("misconception") and subtopic_id in r["subtopics"]
        )
        if not counts:
            return None
        misconception_id, count = counts.most_common(1)[0]
        return {
            "id": misconception_id,
            "description": self.catalog.get("misconceptions", {}).get(misconception_id, misconception_id),
            "count": count,
        }

    def _with_misconception(self, root: Optional[dict], responses: List[dict]) -> Optional[dict]:
        """Add the likely misconception to a root-cause dict and rebuild its summary line."""
        if root:
            misconception = self.misconception_of(root["subtopic_id"], responses)
            root["misconception"] = misconception
            root["summary"] = format_root_cause(root)
        return root

    @staticmethod
    def _subtopic_stats(responses: List[dict]) -> Dict[str, tuple]:
        """Count per subtopic in this quiz: {subtopic: (questions attempted, questions wrong)}."""
        stats = {}
        for response in responses:
            for sid in response["subtopics"]:
                attempted, wrong = stats.get(sid, (0, 0))
                stats[sid] = (attempted + 1, wrong + (not response["is_correct"]))
        return stats

    def _root(self, sid: str, probability: Optional[float], reason: str, method: str, **extra) -> dict:
        """Build the root-cause dict for subtopic `sid` (ids, names, probability, reason,
        method, any extra fields) with a one-line summary."""
        topic_id = self.topic_of(sid)
        root = {
            "subtopic_id": sid,
            "name": self.name_of(sid),
            "topic_id": topic_id,
            "topic_name": self.topic_name_of(topic_id),
            "probability": probability,
            "reason": reason,
            "method": method,
            **extra,
        }
        root["summary"] = format_root_cause(root)
        return root

    # -------------------------------------------------------------- parameters
    def counter_evidence(self, subtopic_id: str, responses: List[dict], past: List[dict]) -> float:
        """CE: share of questions on this subtopic answered correctly, in this quiz and the
        last RECENT_ASSESSMENTS quizzes. 0 if the subtopic was never tested."""
        tested = [r for r in responses if subtopic_id in r["subtopics"]]
        for event in past[-RECENT_ASSESSMENTS:]:
            tested += [r for r in event.get("responses", []) if subtopic_id in r.get("subtopics", [])]
        return sum(r.get("is_correct", False) for r in tested) / len(tested) if tested else 0.0

    def repeat(self, subtopic_id: str, past: List[dict]) -> float:
        """RP: share of the recent relevant quizzes in which this subtopic was also wrong.

        A quiz is relevant if it was on the subtopic's topic or contained a question on it
        (only the last RECENT_ASSESSMENTS relevant quizzes count). 0 if there are none.
        """
        topic_id = self.topic_of(subtopic_id)
        relevant = [
            event for event in past
            if event.get("topic_id") == topic_id
            or any(subtopic_id in r.get("subtopics", []) for r in event.get("responses", []))
        ][-RECENT_ASSESSMENTS:]
        if not relevant:
            return 0.0
        wrong_again = 0
        for event in relevant:
            if event.get("responses"):
                wrong_again += any(
                    subtopic_id in r.get("subtopics", []) and not r.get("is_correct")
                    for r in event["responses"]
                )
            else:
                # Events stored without per-question responses: use their list of error subtopics.
                wrong_again += subtopic_id in event.get("error_signals", [])
        return wrong_again / len(relevant)

    def prerequisite_weakness(self, subtopic_id: str, student_kb: Optional[dict]) -> float:
        """PW: 1 - mastery of the weakest prerequisite topic. 0 if the topic has no prerequisites."""
        prerequisites = self.catalog["prerequisites"].get(self.topic_of(subtopic_id), [])
        if not prerequisites:
            return 0.0
        topics = (student_kb or {}).get("topics", {})
        weakest = min(float(topics.get(p, DEFAULT_MASTERY)) for p in prerequisites)
        return max(0.0, 1.0 - weakest)

    # ------------------------------------------------------------------ method 1
    def analyze_no_kb(self, responses: List[dict]) -> Optional[dict]:
        """Method "no_kb": the subtopic with the most wrong answers in this quiz.

        Uses no student history. Probability = its share of all wrong answers.
        Returns None when every answer is correct.
        """
        stats = self._subtopic_stats(responses)
        suspects = [sid for sid, (_, wrong) in stats.items() if wrong]
        if not suspects:
            return None
        # Most wrong answers, then highest fraction wrong, then id (for a stable order).
        suspects.sort(key=lambda sid: (-stats[sid][1], -stats[sid][1] / stats[sid][0], sid))
        sid = suspects[0]
        attempted, wrong = stats[sid]
        total_wrong = sum(1 for r in responses if not r["is_correct"])
        return self._root(
            sid,
            round(wrong / total_wrong, 2) if total_wrong else None,
            f"it is tagged on {wrong} of the {total_wrong} wrong answers "
            f"({wrong}/{attempted} of its questions wrong); no student history used",
            "no_kb",
        )

    # ------------------------------------------------------------------ method 2
    def analyze_kb_low_score(self, responses: List[dict], student_kb: Optional[dict]) -> Optional[dict]:
        """Method "kb_low_score": among subtopics with a wrong answer, the one with the lowest
        stored mastery in the SKB. No probability is given. None when all answers are correct."""
        stats = self._subtopic_stats(responses)
        suspects = [sid for sid, (_, wrong) in stats.items() if wrong]
        if not suspects:
            return None
        # Lowest mastery, then most wrong answers, then id.
        suspects.sort(key=lambda sid: (self.mastery_of(student_kb, sid), -stats[sid][1], sid))
        sid = suspects[0]
        mastery = self.mastery_of(student_kb, sid)
        source = "subtopic" if sid in (student_kb or {}).get("skills", {}) else "topic"
        return self._root(
            sid,
            None,
            f"it has the lowest stored {source} mastery ({mastery:.2f}) among the subtopics "
            f"on the wrong answers",
            "kb_low_score",
            mastery=round(mastery, 2),
        )

    # ------------------------------------------------------------------ method 3
    def select_candidates(self, responses: List[dict], student_kb: Optional[dict]) -> List[str]:
        """Method "hypothesis", step 1: pick up to TOP_CANDIDATES subtopics with a wrong
        answer to test as hypotheses."""
        stats = self._subtopic_stats(responses)
        suspects = [sid for sid, (_, wrong) in stats.items() if wrong]
        suspects.sort(key=lambda sid: (
            -stats[sid][1] / stats[sid][0],      # highest fraction wrong (lowest quiz score) first
            -stats[sid][1],                      # then most wrong answers
            self.mastery_of(student_kb, sid),    # then lowest mastery
            sid,
        ))
        return suspects[:TOP_CANDIDATES]

    def compute_params(self, sid: str, responses: List[dict], past: List[dict], student_kb: Optional[dict]) -> dict:
        """Step 2: compute the five evidence values (LS, CE, RP, PW, PM') of one hypothesis.

        They do not depend on the weights, so they are computed once and can be re-scored
        with any weight set. Also returns the LS details for display.
        """
        struggle = learning_struggle(student_kb, sid, self.topics_filepath)
        mastery = self.mastery_of(student_kb, sid)
        params = {
            "LS": struggle["LS"],
            "CE": self.counter_evidence(sid, responses, past),
            "RP": self.repeat(sid, past),
            "PW": self.prerequisite_weakness(sid, student_kb),
            # PM' = mastery gap, raised when the mastery was reached with a struggle (fragile).
            "PM": min(1.0, (1 - mastery) + FRAGILITY_LAMBDA * struggle["LS"] * mastery),
        }
        return {
            "subtopic_id": sid,
            "name": self.name_of(sid),
            "topic_id": self.topic_of(sid),
            "mastery": round(mastery, 2),
            "params": {key: round(value, 2) for key, value in params.items()},
            "struggle_components": struggle["components"],
            "struggle_examples": struggle["examples"],
        }

    @staticmethod
    def score_hypotheses(raw: List[dict], weights: Dict[str, float]) -> List[dict]:
        """Step 3: score each hypothesis with a weight set and turn the scores into probabilities.

        contribution = sign * weight * value (CE has sign -1), score = sum of contributions,
        P = max(score, 0) / sum of positive scores (equal shares if no score is positive).
        Returns the hypotheses sorted from most to least likely.
        """
        hypotheses = []
        for h in raw:
            contributions = {
                key: round(SIGNS[key] * weights.get(key, 0.0) * h["params"][key], 3) for key in PARAMS
            }
            hypotheses.append({**h, "contributions": contributions, "score": round(sum(contributions.values()), 3)})
        positive_total = sum(max(h["score"], 0.0) for h in hypotheses)
        for h in hypotheses:
            h["probability"] = round(
                max(h["score"], 0.0) / positive_total if positive_total else 1 / len(hypotheses), 2
            )
        hypotheses.sort(key=lambda h: (-h["probability"], -h["score"]))
        return hypotheses

    def hypothesis_reason(self, top: dict) -> str:
        """Plain-English reason for the winning hypothesis: its (up to 3) largest positive
        contributions, plus a note when CE is 0 (never answered correctly elsewhere)."""
        drivers = [
            f"{EVIDENCE_PHRASES[key]} ({SHORT_LABELS[key]} {top['params'][key]:.2f})"
            for key, value in sorted(top["contributions"].items(), key=lambda item: -item[1])
            if value > 0 and key in EVIDENCE_PHRASES
        ][:3]
        if top["params"]["CE"] == 0:
            drivers.append("it was never answered correctly elsewhere (CE 0.00)")
        return "; ".join(drivers) or "highest combined evidence score"

    def root_from_hypotheses(self, hypotheses: List[dict], preset: Optional[str], weights: Dict[str, float]) -> Optional[dict]:
        """Root-cause dict for the most likely hypothesis (None if there are no hypotheses)."""
        if not hypotheses:
            return None
        top = hypotheses[0]
        return self._root(
            top["subtopic_id"], top["probability"], self.hypothesis_reason(top), "hypothesis",
            preset=preset or "custom", weights=weights,
        )

    # -------------------------------------------------------------------- main
    def analyze(
        self,
        questions: List[dict],
        answers: Dict[str, Any],
        student_kb: Optional[dict],
        method: str = DEFAULT_METHOD,
        preset: Optional[str] = DEFAULT_PRESET,
        weights: Optional[Dict[str, float]] = None,
    ) -> dict:
        """Find the root cause of one quiz with one method (main entry point).

        Args:
            questions / answers: the quiz questions and {question id: chosen option}.
            student_kb: the student's SKB record (history, mastery, conversations).
            method: "no_kb", "kb_low_score" or "hypothesis".
            preset / weights: weight preset name and optional custom weights (hypothesis only).

        Returns {"method", "root_cause", "responses", "hypotheses", ...}. `hypotheses` is
        filled only for the hypothesis method; root_cause is None if nothing was wrong.
        """
        responses = self.grade(questions, answers or {})
        if method == "no_kb":
            return {"method": method, "hypotheses": [], "responses": responses,
                    "root_cause": self._with_misconception(self.analyze_no_kb(responses), responses)}
        if method == "kb_low_score":
            return {"method": method, "hypotheses": [], "responses": responses,
                    "root_cause": self._with_misconception(
                        self.analyze_kb_low_score(responses, student_kb), responses)}

        past = self._assessments(student_kb)
        weights = resolve_weights(weights, preset)
        raw = [self.compute_params(sid, responses, past, student_kb)
               for sid in self.select_candidates(responses, student_kb)]
        hypotheses = self.score_hypotheses(raw, weights)
        return {
            "method": "hypothesis",
            # Report "custom" when the final weights differ from the named preset.
            "preset": preset if not weights or weights == WEIGHT_PRESETS.get(preset) else "custom",
            "weights": weights,
            "hypotheses": hypotheses,
            "root_cause": self._with_misconception(
                self.root_from_hypotheses(hypotheses, preset, weights), responses),
            "responses": responses,
        }

    def compare(
        self,
        questions: List[dict],
        answers: Dict[str, Any],
        student_kb: Optional[dict],
        base_preset: str = DEFAULT_PRESET,
    ) -> dict:
        """Run every method, every weight preset and every leave-one-out ablation of
        `base_preset` on the same quiz, for evaluation. Parameters are computed only once."""
        responses = self.grade(questions, answers or {})
        past = self._assessments(student_kb)
        raw = [self.compute_params(sid, responses, past, student_kb)
               for sid in self.select_candidates(responses, student_kb)]

        def pick(weights, preset):
            """Score the shared candidates with one weight set and return its root cause."""
            hyps = self.score_hypotheses(raw, weights)
            return {"weights": weights, "hypotheses": hyps,
                    "root_cause": self._with_misconception(
                        self.root_from_hypotheses(hyps, preset, weights), responses)}

        base = resolve_weights(preset=base_preset)
        return {
            # Raw parameters of the candidates: stored with the quiz so that weights can be
            # refitted later (scripts/fit_weights.py) without recomputing them.
            "candidates": [{"subtopic_id": h["subtopic_id"], "params": h["params"]} for h in raw],
            "methods": {
                "no_kb": self._with_misconception(self.analyze_no_kb(responses), responses),
                "kb_low_score": self._with_misconception(
                    self.analyze_kb_low_score(responses, student_kb), responses),
                "hypothesis": pick(base, base_preset)["root_cause"],
            },
            "presets": {name: pick(w, name) for name, w in WEIGHT_PRESETS.items()},
            "ablations": {f"{base_preset}-without-{p}": pick(ablation_weights(base, p), f"{base_preset}-without-{p}")
                          for p in PARAMS},
        }


def summarize_comparison(comparison: dict) -> Dict[str, Dict[str, Optional[str]]]:
    """Shrink compare() output to {group: {method or weight set: chosen subtopic id}} for logs."""
    def sid(root):
        """Subtopic id of a root cause, or None."""
        return root["subtopic_id"] if root else None
    return {
        "methods": {name: sid(root) for name, root in comparison["methods"].items()},
        "presets": {name: sid(entry["root_cause"]) for name, entry in comparison["presets"].items()},
        "ablations": {name: sid(entry["root_cause"]) for name, entry in comparison["ablations"].items()},
    }


def format_root_cause(root: Optional[dict]) -> str:
    """One readable line: topic -> subtopic -> reason (+ likely misconception, if known)."""
    if not root:
        return "No wrong answers, so no root cause."
    topic = f"{root['topic_id']} {root.get('topic_name') or ''}".strip()
    line = f"{topic} → {root['subtopic_id']} {root['name']} → {root['reason']}"
    misconception = root.get("misconception")
    if misconception:
        line += f"; likely misconception: {misconception['description']}"
    return line


def format_hypotheses(result: dict) -> str:
    """A text table of the hypotheses (parameters, score, probability) for the console logs."""
    lines = ["   Hypothesis                                   LS   CE   RP   PW   PM'  Score   P"]
    for h in result.get("hypotheses", []):
        p = h["params"]
        label = f"{h['subtopic_id']} {h['name']}"[:44]
        lines.append(
            f"   {label:<44} {p['LS']:.2f} {p['CE']:.2f} {p['RP']:.2f} {p['PW']:.2f} "
            f"{p['PM']:.2f} {h['score']:+.3f} {h['probability']:.2f}"
        )
    return "\n".join(lines)
