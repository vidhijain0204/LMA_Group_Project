"""Find the most likely root cause of a student's quiz errors.

Every quiz question is tagged with two subtopics, so a wrong answer could be caused by either of
them. Instead of blaming every low-scoring subtopic, this agent:

1. picks the top 3 candidate subtopics (hypotheses) from the questions answered wrongly, and
2. scores each one with eight evidence parameters, all scaled to 0..1:

    QE   Quiz Error            share of this quiz's wrong answers that involve H
    CE   Counter-Evidence      H answered correctly elsewhere (this quiz + recent quizzes); subtracted
    DS   Distractor Signal     the chosen wrong option / explanation points at H rather than the other tag
    RP   Repeat                H was also wrong in recent quizzes
    ST   Survived Teaching     H's topic was already taught and H is still wrong
    LS   Learning Struggle     the student struggled in the tutor conversation (struggle_tracker.py)
    PM'  Prior Mastery gap     1 - mastery, raised when learning was a struggle (fragile mastery)
    PW   Prerequisite Weakness a prerequisite topic of H is weak

    PM'(H)    = (1 - M) + LAMBDA * LS * M
    Score(H)  = 0.25 QE - 0.20 CE + 0.15 DS + 0.10 RP + 0.10 ST + 0.10 LS + 0.05 PM' + 0.05 PW
    P(H)      = max(Score(H), 0) / sum of max(Score, 0) over the candidates

3. The root cause is the candidate with the highest P(H).
"""

from typing import Any, Dict, List, Optional

from agents.struggle_tracker import TOPICS_FILEPATH, learning_struggle, load_catalog, tokenize

TOP_CANDIDATES = 3
RECENT_ASSESSMENTS = 3
FRAGILITY_LAMBDA = 0.5
DEFAULT_MASTERY = 0.5
WEIGHTS = {
    "QE": 0.25,
    "CE": -0.20,
    "DS": 0.15,
    "RP": 0.10,
    "ST": 0.10,
    "LS": 0.10,
    "PM": 0.05,
    "PW": 0.05,
}
EVIDENCE_PHRASES = {
    "QE": "it explains most of the wrong answers in this quiz",
    "DS": "the wrong options chosen point at this concept",
    "RP": "the same mistake appeared in recent quizzes",
    "ST": "it is still wrong after being taught",
    "LS": "the student struggled with it while learning",
    "PM": "mastery is low or fragile",
    "PW": "a prerequisite topic is weak",
}


def _question_id(question: dict) -> str:
    return question.get("prob_id", question.get("problem_id"))


class RootCauseAgent:
    def __init__(self, topics_filepath: str = TOPICS_FILEPATH):
        self.topics_filepath = topics_filepath
        self.catalog = load_catalog(topics_filepath)

    # ----------------------------------------------------------------- helpers
    def topic_of(self, subtopic_id: str) -> str:
        return self.catalog["subtopic_topic"].get(subtopic_id, subtopic_id.split(".")[0])

    def name_of(self, subtopic_id: str) -> str:
        return self.catalog["subtopic_names"].get(subtopic_id, subtopic_id)

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

    # -------------------------------------------------------------- parameters
    def quiz_error(self, subtopic_id: str, wrong: List[dict]) -> float:
        return sum(subtopic_id in r["subtopics"] for r in wrong) / len(wrong) if wrong else 0.0

    def counter_evidence(self, subtopic_id: str, responses: List[dict], past: List[dict]) -> float:
        tested = [r for r in responses if subtopic_id in r["subtopics"]]
        for event in past[-RECENT_ASSESSMENTS:]:
            tested += [r for r in event.get("responses", []) if subtopic_id in r.get("subtopics", [])]
        return sum(r.get("is_correct", False) for r in tested) / len(tested) if tested else 0.0

    def distractor_signal(self, subtopic_id: str, wrong: List[dict]) -> float:
        """For each wrong answer involving H: does the chosen option plus the explanation share
        more words with H's name than with the other subtopic on the question?"""
        shares = []
        keywords = self.catalog["subtopic_keywords"]
        for response in wrong:
            if subtopic_id not in response["subtopics"]:
                continue
            tokens = tokenize(f"{response['chosen_text']} {response['explanation']}")
            own = len(tokens & keywords.get(subtopic_id, set()))
            other = max(
                (len(tokens & keywords.get(sid, set())) for sid in response["subtopics"] if sid != subtopic_id),
                default=0,
            )
            shares.append(own / (own + other) if own + other else 0.5)
        return sum(shares) / len(shares) if shares else 0.0

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

    def survived_teaching(self, subtopic_id: str, student_kb: Optional[dict]) -> float:
        topic_id = self.topic_of(subtopic_id)
        lessons = sum(
            1 for event in (student_kb or {}).get("pedagogy_history", [])
            if event.get("event") == "lesson" and event.get("topic_id") == topic_id
        )
        return min(1.0, lessons / 2)

    def prerequisite_weakness(self, subtopic_id: str, student_kb: Optional[dict]) -> float:
        prerequisites = self.catalog["prerequisites"].get(self.topic_of(subtopic_id), [])
        if not prerequisites:
            return 0.0
        topics = (student_kb or {}).get("topics", {})
        weakest = min(float(topics.get(p, DEFAULT_MASTERY)) for p in prerequisites)
        return max(0.0, 1.0 - weakest)

    # -------------------------------------------------------------------- main
    def select_candidates(self, responses: List[dict], student_kb: Optional[dict]) -> List[str]:
        stats = {}
        for response in responses:
            for sid in response["subtopics"]:
                attempted, wrong = stats.get(sid, (0, 0))
                stats[sid] = (attempted + 1, wrong + (not response["is_correct"]))
        suspects = [sid for sid, (_, wrong) in stats.items() if wrong]
        suspects.sort(key=lambda sid: (
            -stats[sid][1] / stats[sid][0],      # lowest quiz score first
            -stats[sid][1],                      # then most wrong answers
            self.mastery_of(student_kb, sid),    # then lowest mastery
            sid,
        ))
        return suspects[:TOP_CANDIDATES]

    def analyze(
        self,
        questions: List[dict],
        answers: Dict[str, Any],
        student_kb: Optional[dict],
    ) -> dict:
        responses = self.grade(questions, answers or {})
        wrong = [r for r in responses if not r["is_correct"]]
        past = self._assessments(student_kb)

        hypotheses = []
        for sid in self.select_candidates(responses, student_kb):
            struggle = learning_struggle(student_kb, sid, self.topics_filepath)
            mastery = self.mastery_of(student_kb, sid)
            params = {
                "QE": self.quiz_error(sid, wrong),
                "CE": self.counter_evidence(sid, responses, past),
                "DS": self.distractor_signal(sid, wrong),
                "RP": self.repeat(sid, past),
                "ST": self.survived_teaching(sid, student_kb),
                "LS": struggle["LS"],
                "PM": min(1.0, (1 - mastery) + FRAGILITY_LAMBDA * struggle["LS"] * mastery),
                "PW": self.prerequisite_weakness(sid, student_kb),
            }
            params = {key: round(value, 2) for key, value in params.items()}
            contributions = {key: round(WEIGHTS[key] * value, 3) for key, value in params.items()}
            hypotheses.append({
                "subtopic_id": sid,
                "name": self.name_of(sid),
                "topic_id": self.topic_of(sid),
                "mastery": round(mastery, 2),
                "params": params,
                "struggle_components": struggle["components"],
                "struggle_examples": struggle["examples"],
                "contributions": contributions,
                "score": round(sum(contributions.values()), 3),
            })

        positive_total = sum(max(h["score"], 0.0) for h in hypotheses)
        for h in hypotheses:
            h["probability"] = round(
                max(h["score"], 0.0) / positive_total if positive_total else 1 / len(hypotheses), 2
            )
        hypotheses.sort(key=lambda h: (-h["probability"], -h["score"]))

        root_cause = None
        if hypotheses:
            top = hypotheses[0]
            drivers = [
                EVIDENCE_PHRASES[key]
                for key, value in sorted(top["contributions"].items(), key=lambda item: -item[1])
                if value > 0 and key in EVIDENCE_PHRASES
            ][:3]
            root_cause = {
                "subtopic_id": top["subtopic_id"],
                "name": top["name"],
                "topic_id": top["topic_id"],
                "probability": top["probability"],
                "reason": "; ".join(drivers) or "highest combined evidence score",
            }
        return {"hypotheses": hypotheses, "root_cause": root_cause, "responses": responses}


def format_hypotheses(result: dict) -> str:
    """A console table of the hypotheses, for the workflow logs."""
    lines = ["   Hypothesis                                   QE   CE   DS   RP   ST   LS   PM'  PW   Score  P"]
    for h in result.get("hypotheses", []):
        p = h["params"]
        label = f"{h['subtopic_id']} {h['name']}"[:44]
        lines.append(
            f"   {label:<44} {p['QE']:.2f} {p['CE']:.2f} {p['DS']:.2f} {p['RP']:.2f} "
            f"{p['ST']:.2f} {p['LS']:.2f} {p['PM']:.2f} {p['PW']:.2f} {h['score']:+.3f} {h['probability']:.2f}"
        )
    return "\n".join(lines)
