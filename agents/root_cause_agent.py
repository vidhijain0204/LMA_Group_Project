"""Find the most likely root cause of a student's quiz errors.

Every quiz question is tagged with two subtopics, so a wrong answer could be caused by either of
them. Three methods are implemented so they can be compared during evaluation:

    no_kb         Baseline without the knowledge base: the subtopic with the most wrong answers
                  in this quiz (ties broken by the lowest quiz score).
    kb_low_score  Uses the knowledge base directly: among the subtopics on wrongly answered
                  questions, the one with the lowest stored mastery.
    hypothesis    Hypothesis testing: the top 3 candidate subtopics are scored with five
                  evidence parameters (all scaled to 0..1) and a configurable weight set.

Hypothesis parameters:

    LS   Learning Struggle       the student struggled in the tutor conversation (struggle_tracker.py)
    CE   Counter-Evidence        H answered correctly elsewhere (this quiz + recent quizzes); subtracted
    RP   Repeat                  H was also wrong in recent quizzes
    PW   Prerequisite Weakness   a prerequisite topic of H is weak
    PM'  Prior Mastery gap       1 - mastery, raised when learning was a struggle (fragile mastery)

    PM'(H)    = min(1, (1 - M) + LAMBDA * LS * M)
    Score(H)  = wLS*LS - wCE*CE + wRP*RP + wPW*PW + wPM*PM'
    P(H)      = max(Score(H), 0) / sum of max(Score, 0) over the candidates

The weights come from WEIGHT_PRESETS (or any custom dict), so different combinations - and
leave-one-parameter-out ablations - can be evaluated against each other.
"""

from typing import Any, Dict, List, Optional

from agents.struggle_tracker import TOPICS_FILEPATH, learning_struggle, load_catalog

TOP_CANDIDATES = 3
RECENT_ASSESSMENTS = 3
FRAGILITY_LAMBDA = 0.5
DEFAULT_MASTERY = 0.5

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

# Weight magnitudes (the sign comes from SIGNS). Each preset sums to 1.
WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "balanced":       {"LS": 0.25, "CE": 0.20, "RP": 0.20, "PW": 0.15, "PM": 0.20},
    "equal":          {"LS": 0.20, "CE": 0.20, "RP": 0.20, "PW": 0.20, "PM": 0.20},
    "struggle_first": {"LS": 0.45, "CE": 0.15, "RP": 0.15, "PW": 0.10, "PM": 0.15},
    "history_first":  {"LS": 0.15, "CE": 0.25, "RP": 0.35, "PW": 0.10, "PM": 0.15},
    "mastery_first":  {"LS": 0.10, "CE": 0.15, "RP": 0.10, "PW": 0.25, "PM": 0.40},
}
DEFAULT_PRESET = "balanced"

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
    "RP": "the same mistake appeared in recent quizzes",
    "PW": "a prerequisite topic is weak",
    "PM": "prior mastery is low or fragile",
}


def _question_id(question: dict) -> str:
    return question.get("prob_id", question.get("problem_id"))


def resolve_weights(weights: Optional[Dict[str, float]] = None, preset: Optional[str] = None) -> Dict[str, float]:
    """A complete weight dict: explicit `weights` win, then the named preset, then the default."""
    base = dict(WEIGHT_PRESETS.get(preset or DEFAULT_PRESET, WEIGHT_PRESETS[DEFAULT_PRESET]))
    if weights:
        base.update({key: float(value) for key, value in weights.items() if key in PARAMS})
    return base


def ablation_weights(weights: Dict[str, float], drop: str) -> Dict[str, float]:
    """`weights` with one parameter switched off and the rest rescaled to the same total."""
    total = sum(weights.values())
    kept = {key: value for key, value in weights.items() if key != drop}
    kept_total = sum(kept.values()) or 1.0
    result = {key: round(value * total / kept_total, 4) for key, value in kept.items()}
    result[drop] = 0.0
    return result


class RootCauseAgent:
    def __init__(self, topics_filepath: str = TOPICS_FILEPATH):
        self.topics_filepath = topics_filepath
        self.catalog = load_catalog(topics_filepath)

    # ----------------------------------------------------------------- helpers
    def topic_of(self, subtopic_id: str) -> str:
        return self.catalog["subtopic_topic"].get(subtopic_id, subtopic_id.split(".")[0])

    def name_of(self, subtopic_id: str) -> str:
        return self.catalog["subtopic_names"].get(subtopic_id, subtopic_id)

    def topic_name_of(self, topic_id: str) -> str:
        return self.catalog.get("topic_names", {}).get(topic_id, topic_id)

    def mastery_of(self, student_kb: Optional[dict], subtopic_id: str) -> float:
        kb = student_kb or {}
        if subtopic_id in kb.get("skills", {}):
            return float(kb["skills"][subtopic_id])
        return float(kb.get("topics", {}).get(self.topic_of(subtopic_id), DEFAULT_MASTERY))

    @staticmethod
    def grade(questions: List[dict], answers: Dict[str, Any]) -> List[dict]:
        """One record per question: what the student chose and which subtopics it tests."""
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
                "explanation": str(question.get("explanations", "")),
            })
        return responses

    @staticmethod
    def _assessments(student_kb: Optional[dict]) -> List[dict]:
        return [
            event for event in (student_kb or {}).get("pedagogy_history", [])
            if event.get("event") == "assessment"
        ]

    @staticmethod
    def _subtopic_stats(responses: List[dict]) -> Dict[str, tuple]:
        """subtopic -> (questions attempted, questions wrong) in this quiz."""
        stats = {}
        for response in responses:
            for sid in response["subtopics"]:
                attempted, wrong = stats.get(sid, (0, 0))
                stats[sid] = (attempted + 1, wrong + (not response["is_correct"]))
        return stats

    def _root(self, sid: str, probability: Optional[float], reason: str, method: str, **extra) -> dict:
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
        tested = [r for r in responses if subtopic_id in r["subtopics"]]
        for event in past[-RECENT_ASSESSMENTS:]:
            tested += [r for r in event.get("responses", []) if subtopic_id in r.get("subtopics", [])]
        return sum(r.get("is_correct", False) for r in tested) / len(tested) if tested else 0.0

    def repeat(self, subtopic_id: str, past: List[dict]) -> float:
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
                wrong_again += subtopic_id in event.get("error_signals", [])
        return wrong_again / len(relevant)

    def prerequisite_weakness(self, subtopic_id: str, student_kb: Optional[dict]) -> float:
        prerequisites = self.catalog["prerequisites"].get(self.topic_of(subtopic_id), [])
        if not prerequisites:
            return 0.0
        topics = (student_kb or {}).get("topics", {})
        weakest = min(float(topics.get(p, DEFAULT_MASTERY)) for p in prerequisites)
        return max(0.0, 1.0 - weakest)

    # ------------------------------------------------------------------ method 1
    def analyze_no_kb(self, responses: List[dict]) -> Optional[dict]:
        stats = self._subtopic_stats(responses)
        suspects = [sid for sid, (_, wrong) in stats.items() if wrong]
        if not suspects:
            return None
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
        stats = self._subtopic_stats(responses)
        suspects = [sid for sid, (_, wrong) in stats.items() if wrong]
        if not suspects:
            return None
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
        stats = self._subtopic_stats(responses)
        suspects = [sid for sid, (_, wrong) in stats.items() if wrong]
        suspects.sort(key=lambda sid: (
            -stats[sid][1] / stats[sid][0],      # lowest quiz score first
            -stats[sid][1],                      # then most wrong answers
            self.mastery_of(student_kb, sid),    # then lowest mastery
            sid,
        ))
        return suspects[:TOP_CANDIDATES]

    def compute_params(self, sid: str, responses: List[dict], past: List[dict], student_kb: Optional[dict]) -> dict:
        """The five raw parameters of one hypothesis (independent of the weights)."""
        struggle = learning_struggle(student_kb, sid, self.topics_filepath)
        mastery = self.mastery_of(student_kb, sid)
        params = {
            "LS": struggle["LS"],
            "CE": self.counter_evidence(sid, responses, past),
            "RP": self.repeat(sid, past),
            "PW": self.prerequisite_weakness(sid, student_kb),
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
        """Apply a weight set to precomputed parameters and normalise to probabilities."""
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
        drivers = [
            f"{EVIDENCE_PHRASES[key]} ({SHORT_LABELS[key]} {top['params'][key]:.2f})"
            for key, value in sorted(top["contributions"].items(), key=lambda item: -item[1])
            if value > 0 and key in EVIDENCE_PHRASES
        ][:3]
        if top["params"]["CE"] == 0:
            drivers.append("it was never answered correctly elsewhere (CE 0.00)")
        return "; ".join(drivers) or "highest combined evidence score"

    def root_from_hypotheses(self, hypotheses: List[dict], preset: Optional[str], weights: Dict[str, float]) -> Optional[dict]:
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
        """Root cause with one method. `hypotheses` is filled only for the hypothesis method."""
        responses = self.grade(questions, answers or {})
        if method == "no_kb":
            return {"method": method, "hypotheses": [], "root_cause": self.analyze_no_kb(responses),
                    "responses": responses}
        if method == "kb_low_score":
            return {"method": method, "hypotheses": [],
                    "root_cause": self.analyze_kb_low_score(responses, student_kb), "responses": responses}

        past = self._assessments(student_kb)
        weights = resolve_weights(weights, preset)
        raw = [self.compute_params(sid, responses, past, student_kb)
               for sid in self.select_candidates(responses, student_kb)]
        hypotheses = self.score_hypotheses(raw, weights)
        return {
            "method": "hypothesis",
            "preset": preset if not weights or weights == WEIGHT_PRESETS.get(preset) else "custom",
            "weights": weights,
            "hypotheses": hypotheses,
            "root_cause": self.root_from_hypotheses(hypotheses, preset, weights),
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
            hyps = self.score_hypotheses(raw, weights)
            return {"weights": weights, "hypotheses": hyps,
                    "root_cause": self.root_from_hypotheses(hyps, preset, weights)}

        base = resolve_weights(preset=base_preset)
        return {
            "methods": {
                "no_kb": self.analyze_no_kb(responses),
                "kb_low_score": self.analyze_kb_low_score(responses, student_kb),
                "hypothesis": pick(base, base_preset)["root_cause"],
            },
            "presets": {name: pick(w, name) for name, w in WEIGHT_PRESETS.items()},
            "ablations": {f"{base_preset}-without-{p}": pick(ablation_weights(base, p), f"{base_preset}-without-{p}")
                          for p in PARAMS},
        }


def summarize_comparison(comparison: dict) -> Dict[str, Dict[str, Optional[str]]]:
    """compare() output reduced to {group: {method / weight set: picked subtopic id}}."""
    def sid(root):
        return root["subtopic_id"] if root else None
    return {
        "methods": {name: sid(root) for name, root in comparison["methods"].items()},
        "presets": {name: sid(entry["root_cause"]) for name, entry in comparison["presets"].items()},
        "ablations": {name: sid(entry["root_cause"]) for name, entry in comparison["ablations"].items()},
    }


def format_root_cause(root: Optional[dict]) -> str:
    """One line: topic -> subtopic -> reason."""
    if not root:
        return "No wrong answers, so no root cause."
    topic = f"{root['topic_id']} {root.get('topic_name') or ''}".strip()
    return f"{topic} → {root['subtopic_id']} {root['name']} → {root['reason']}"


def format_hypotheses(result: dict) -> str:
    """A console table of the hypotheses, for the workflow logs."""
    lines = ["   Hypothesis                                   LS   CE   RP   PW   PM'  Score   P"]
    for h in result.get("hypotheses", []):
        p = h["params"]
        label = f"{h['subtopic_id']} {h['name']}"[:44]
        lines.append(
            f"   {label:<44} {p['LS']:.2f} {p['CE']:.2f} {p['RP']:.2f} {p['PW']:.2f} "
            f"{p['PM']:.2f} {h['score']:+.3f} {h['probability']:.2f}"
        )
    return "\n".join(lines)
