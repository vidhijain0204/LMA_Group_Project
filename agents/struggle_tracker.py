"""Struggle tracker: records how much a student struggled while learning with the tutor.

Where it fits in the flow:
- tutor_agent.py calls `record_learning_turn()` after every tutor reply, so each student message
  is saved in the student's SKB (data/skb_database.json) under "learning_conversations".
- root_cause_agent.py calls `learning_struggle()` after a quiz to get the LS (Learning Struggle)
  evidence for each candidate subtopic.
- orchestrator.py uses the topic helpers at the bottom (`topics_mentioned`, `topic_evidence`,
  `recent_learning_topics`, `current_lesson_topic`) to decide which topic a message or quiz is about.
- `load_catalog()` reads data/common/topics.json and is shared by several other modules.

Each logged "learning turn" stores the topic, the subtopics the message mentions and two
struggle flags: confusion ("I don't get it") and hint request ("just tell me the answer").

The LS score for one subtopic is the plain average of six parts, each between 0 and 1:

    LS = average(CF, RA, TR, HR, WS, SW)

    CF  Confusion phrases     share of turns that say "I don't get it", "still not working", ...
    RA  Re-asking             the student asked about this subtopic 2+ times
    TR  Turns to resolve      many back-and-forth turns were needed on the topic
    HR  Hint requests         share of turns asking for a hint or the answer
    WS  Wrong self-explanation   the tutor judged the student's answer to its question partly
                                 correct or wrong, or after a hints/self-explanation lesson the
                                 student was still confused or asked for the answer
    SW  Strategy switches     the pedagogy controller had to change strategy on the topic

The detection here is rule-based (regular expressions and keyword overlap), so it is fast,
gives the same result every time and can be tested without the LLM. When the LLM turn judge
(turn_judge.py) is available, its flags can be passed in instead of the rule flags.
"""

import json
import re
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional

TOPICS_FILEPATH = "data/common/topics.json"
MAX_LOGGED_TURNS = 200   # only the newest 200 turns are kept, so the SKB does not grow forever

# Phrases that show the student is confused or stuck (sets the "confusion" flag).
CONFUSION_PATTERN = re.compile(
    r"\b(confus\w*|don'?t (get|understand)|do not (get|understand)|didn'?t (get|understand)|"
    r"not sure|i'?m lost|stuck|makes? no sense|doesn'?t make sense|can'?t understand|"
    r"cannot understand|not following|what do you mean|still (not|don'?t|confused|wrong|"
    r"getting|failing)|not working|doesn'?t work|isn'?t working|why (is|does|do|did) .{0,40}"
    r"(empty|error|wrong|fail|crash|not|none|erased|missing))\b",
    re.IGNORECASE,
)
# Phrases that ask for a hint or for the answer itself (sets the "hint_request" flag).
HINT_PATTERN = re.compile(
    r"\b(hint|just (tell|give|show) me|(tell|give|show) me the (answer|solution|code)|"
    r"what'?s the answer|what is the answer|solve it for me|do it for me)\b",
    re.IGNORECASE,
)

# Words too common to tell topics or subtopics apart; ignored when matching keywords.
STOPWORDS = {
    "and", "the", "for", "with", "how", "what", "are", "vs", "its", "into", "from", "that",
    "this", "their", "basic", "using", "use", "common", "python",
}
# Strategies that ask the student to explain or reason first; a confused reply after one of
# them counts as a failed self-explanation (WS).
SELF_EXPLANATION_STRATEGIES = {"self_explanation", "hints"}


def stem(word: str) -> str:
    """Cut common endings off a word so that "files"/"file", "modes"/"mode" and
    "closing"/"close" become the same word. Very simple on purpose (no NLP library)."""
    for suffix in ("ing", "ed", "s"):
        if len(word) > 4 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def tokenize(text: str) -> set:
    """Turn text into a set of stemmed lowercase words (3+ letters, stopwords removed),
    used for keyword matching against subtopic and topic names."""
    words = re.findall(r"[a-z_]+", str(text or "").lower())
    return {stem(word) for word in words if len(word) >= 3 and word not in STOPWORDS}


@lru_cache(maxsize=4)
def load_catalog(topics_filepath: str = TOPICS_FILEPATH) -> dict:
    """Read topics.json once and build lookup tables used across the project.

    Returns a dict with: subtopic and topic names, the topic of every subtopic, topic
    prerequisites, the subtopics of each topic, keywords per subtopic, misconception
    descriptions and the misconceptions of each topic. Cached, so the file is read only once.
    """
    with open(topics_filepath, "r", encoding="utf-8") as topic_file:
        topics = json.load(topic_file)

    subtopic_names, subtopic_topic, prerequisites, topic_subtopics, topic_names = {}, {}, {}, {}, {}
    misconceptions, topic_misconceptions = {}, {}
    for topic in topics:
        topic_id = topic["topic_id"]
        topic_misconceptions[topic_id] = []
        for item in topic.get("common_misconceptions", []):
            misconceptions[item["id"]] = item.get("description", item["id"])
            topic_misconceptions[topic_id].append(item["id"])
        topic_names[topic_id] = topic.get("name", topic_id)
        prerequisites[topic_id] = list(topic.get("prerequisites", []))
        topic_subtopics[topic_id] = []
        for subtopic in topic.get("subconcepts", []):
            subtopic_names[subtopic["id"]] = subtopic["name"]
            subtopic_topic[subtopic["id"]] = topic_id
            topic_subtopics[topic_id].append(subtopic["id"])

    # Keywords that tell the subtopics of one topic apart: drop words shared by most of them
    # (e.g. "file" in every File Handling subtopic). A word is kept only if at most
    # max(1, (n - 1) // 2) of the topic's n subtopic names contain it.
    distinctive_keywords = {}
    for topic_id, subtopic_ids in topic_subtopics.items():
        token_sets = {sid: tokenize(subtopic_names[sid]) for sid in subtopic_ids}
        for sid, tokens in token_sets.items():
            distinctive_keywords[sid] = {
                token for token in tokens
                if sum(token in other for other in token_sets.values()) <= max(1, (len(token_sets) - 1) // 2)
            }

    return {
        "subtopic_names": subtopic_names,
        "topic_names": topic_names,
        "subtopic_topic": subtopic_topic,
        "prerequisites": prerequisites,
        "topic_subtopics": topic_subtopics,
        "subtopic_keywords": {sid: tokenize(name) for sid, name in subtopic_names.items()},
        "distinctive_keywords": distinctive_keywords,
        "misconceptions": misconceptions,
        "topic_misconceptions": topic_misconceptions,
    }


def detect_struggle(text: str) -> Dict[str, bool]:
    """Rule-based struggle flags for one message: {"confusion": bool, "hint_request": bool}."""
    return {
        "confusion": bool(CONFUSION_PATTERN.search(text or "")),
        "hint_request": bool(HINT_PATTERN.search(text or "")),
    }


def map_subtopics(text: str, topic_id: str, topics_filepath: str = TOPICS_FILEPATH) -> List[str]:
    """Return the subtopics of `topic_id` whose distinctive keywords appear in the message.
    An empty list means the message is about the topic in general."""
    catalog = load_catalog(topics_filepath)
    tokens = tokenize(text)
    return [
        sid for sid in catalog["topic_subtopics"].get(topic_id, [])
        if tokens & catalog["distinctive_keywords"].get(sid, set())
    ]


def record_learning_turn(
    student_kb: dict,
    topic_id: str,
    student_text: str,
    previous_strategy: Optional[str] = None,
    topics_filepath: str = TOPICS_FILEPATH,
    reply_to_tutor: bool = False,
    answer_verdict: Optional[str] = None,
    context_text: str = "",
    flags: Optional[Dict[str, bool]] = None,
    signal_source: str = "rules",
) -> dict:
    """Save one student message sent to the tutor as a "learning turn" in the SKB.

    Called by tutor_agent.py after every tutor reply. The subtopic is found from the message's
    keywords, then from `context_text`, and finally from the semantic topic classifier.

    Args:
        student_kb: the student's SKB record (not modified; a copy is returned).
        previous_strategy: the teaching strategy the student was responding to (used for WS).
        reply_to_tutor / answer_verdict: whether the message answers the tutor's question, and
            the judgement of that answer ("correct" / "partial" / "incorrect").
        context_text: the tutor question being answered; helps find the subtopic when the
            answer itself names none ("it prints 5").
        flags: struggle flags from the LLM turn judge; if None, the rule-based flags are used.
        signal_source: where the flags came from ("rules" or "llm").
    Returns:
        A new SKB dict with the turn appended (unchanged if the message is empty).
    """
    updated = deepcopy(student_kb)
    if not str(student_text or "").strip():
        return updated
    rule_flags = detect_struggle(student_text)   # always stored, even when the LLM flags are used
    flags = flags or rule_flags
    subtopic_ids = (
        map_subtopics(student_text, topic_id, topics_filepath)
        or map_subtopics(context_text, topic_id, topics_filepath)
    )
    subtopic_source = "keywords" if subtopic_ids else None
    if not subtopic_ids:
        # No keyword match: ask the semantic classifier (it abstains when unsure).
        from agents.topic_classifier import classify_subtopic
        predicted = classify_subtopic(f"{context_text} {student_text}".strip(), topic_id)
        if predicted:
            subtopic_ids, subtopic_source = [predicted], "classifier"
    turns = updated.setdefault("learning_conversations", [])
    turns.append({
        "topic_id": topic_id,
        "subtopic_ids": subtopic_ids,
        "subtopic_source": subtopic_source,
        "text": str(student_text).strip()[:300],
        "confusion": bool(flags.get("confusion")),
        "hint_request": bool(flags.get("hint_request")),
        "signal_source": signal_source,
        "rule_flags": rule_flags,
        "after_strategy": previous_strategy,
        "reply_to_tutor": bool(reply_to_tutor),
        "answer_verdict": answer_verdict,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })
    updated["learning_conversations"] = turns[-MAX_LOGGED_TURNS:]
    return updated


def strategy_switches(student_kb: Optional[dict], topic_id: str) -> int:
    """Count how many times the teaching strategy changed between consecutive lessons on a topic."""
    strategies = [
        event.get("strategy")
        for event in (student_kb or {}).get("pedagogy_history", [])
        if event.get("event") == "lesson" and event.get("topic_id") == topic_id
    ]
    return sum(1 for before, after in zip(strategies, strategies[1:]) if before != after)


def learning_struggle(
    student_kb: Optional[dict],
    subtopic_id: str,
    topics_filepath: str = TOPICS_FILEPATH,
) -> dict:
    """Compute the LS (Learning Struggle) score and its six parts for one subtopic.

    Called by root_cause_agent.py. Uses the turns on the subtopic's topic that either name
    this subtopic or name no subtopic at all (general questions about the topic).

    Returns:
        {"LS": 0..1, "components": {CF, RA, TR, HR, WS, SW}, "turns": number of turns used,
         "examples": up to 2 recent messages that showed confusion or asked for a hint}
    """
    catalog = load_catalog(topics_filepath)
    topic_id = catalog["subtopic_topic"].get(subtopic_id, subtopic_id.split(".")[0])
    turns = [
        turn for turn in (student_kb or {}).get("learning_conversations", [])
        if turn.get("topic_id") == topic_id
        and (not turn.get("subtopic_ids") or subtopic_id in turn["subtopic_ids"])
    ]
    explicit_turns = [turn for turn in turns if subtopic_id in turn.get("subtopic_ids", [])]
    # Answers to the tutor's questions, plus any message after a hints / self-explanation lesson.
    eligible = [
        turn for turn in turns
        if turn.get("answer_verdict") or turn.get("after_strategy") in SELF_EXPLANATION_STRATEGIES
    ]

    count = len(turns)
    # RA: 0 for one explicit mention, 1.0 from three. TR: 0 up to 2 turns, 1.0 from 6 turns.
    # SW: 1.0 from two strategy switches. CF, HR and WS are simple shares of turns.
    components = {
        "CF": sum(turn.get("confusion", False) for turn in turns) / count if count else 0.0,
        "RA": min(1.0, max(0, len(explicit_turns) - 1) / 2),
        "TR": min(1.0, max(0, count - 2) / 4),
        "HR": sum(turn.get("hint_request", False) for turn in turns) / count if count else 0.0,
        "WS": (
            sum(
                turn.get("answer_verdict") in ("partial", "incorrect")
                or (turn.get("answer_verdict") != "correct"
                    and (turn.get("confusion", False) or turn.get("hint_request", False)))
                for turn in eligible
            ) / len(eligible)
            if eligible else 0.0
        ),
        "SW": min(1.0, strategy_switches(student_kb, topic_id) / 2),
    }
    components = {key: round(value, 2) for key, value in components.items()}
    return {
        "LS": round(sum(components.values()) / len(components), 2),
        "components": components,
        "turns": count,
        "examples": [turn["text"] for turn in turns if turn.get("confusion") or turn.get("hint_request")][-2:],
    }


# ------------------------------------------------------------------ quiz topic selection
# The helpers below are used by orchestrator.py to decide which topic(s) a quiz should cover.
RECENT_TURNS_FOR_QUIZ = 3   # "quiz me" with no topic named -> topics of the last 3 tutor turns


# How students name each topic. Plain "if" and "set" are left out on purpose: they are ordinary
# English words ("check if I'm ready", "a set of questions").
TOPIC_ALIASES = {
    "T01": r"introduction to python|interpreter",
    "T02": r"variables?|data ?types?",
    "T03": r"input|output",
    "T04": r"operators?",
    "T05": r"if[- /]?else|if[- ]statements?|elif|conditionals?|conditional statements?",
    "T06": r"loops?|while|for[- ]loops?",
    "T07": r"strings?",
    "T08": r"lists?(?! comprehensions?)|tuples?|sets",
    "T09": r"dict(ionary|ionaries|s)?",
    "T10": r"functions?|def",
    "T11": r"comprehensions?",
    "T12": r"exceptions?|try[- /]except|error handling",
    "T13": r"files?|file handling",
}
# Whole-word, case-insensitive regex for each topic's aliases.
_TOPIC_PATTERNS = {tid: re.compile(rf"\b(?:{alias})\b", re.IGNORECASE) for tid, alias in TOPIC_ALIASES.items()}


def topics_mentioned(text: str, topics_filepath: str = TOPICS_FILEPATH) -> List[str]:
    """Return the topic ids the message names explicitly, in the order they appear.
    Both ids ("T08") and names/aliases ("lists") count.
    Example: "quiz on lists and loops" -> [T08, T06]; "ready for assessment" -> []."""
    text = str(text or "")
    valid = load_catalog(topics_filepath)["topic_names"]
    positions = {}   # topic id -> position of its first mention, used for ordering
    for match in re.finditer(r"\bT\d{2}\b", text.upper()):
        if match.group(0) in valid:
            positions.setdefault(match.group(0), match.start())
    for tid, pattern in _TOPIC_PATTERNS.items():
        match = pattern.search(text)
        if match and tid in valid:
            positions.setdefault(tid, match.start())
    return sorted(positions, key=positions.get)


def recent_learning_topics(
    student_kb: Optional[dict],
    turns: int = RECENT_TURNS_FOR_QUIZ,
    topics_filepath: str = TOPICS_FILEPATH,
) -> List[str]:
    """Return the distinct topics of the student's last `turns` tutor messages, oldest first."""
    valid = load_catalog(topics_filepath)["topic_names"]
    recent = [
        turn.get("topic_id") for turn in (student_kb or {}).get("learning_conversations", [])
        if turn.get("topic_id") in valid
    ][-turns:]
    return list(dict.fromkeys(recent))


# ------------------------------------------------------------------ topic of a tutor request
# The helpers below are used by orchestrator.py to guess the topic of a tutor question.
# Concept, method and keyword names that point at a topic without naming it
# ("difference between sort and count" -> lists). A word may belong to several topics
# ("count", "slicing"); the conversation's current topic then breaks the tie.
TOPIC_HINTS = {
    "T01": "interpreter bytecode script shell pep indentation comment",
    "T02": "int float str bool string_type type casting cast variable assign assignment none integer",
    "T03": "print input format fstring f-string formatting",
    "T04": "operator arithmetic modulo floor division comparison logical precedence exponent",
    "T05": "condition conditional elif else nested ternary boolean",
    "T06": "loop while range break continue iterate iteration enumerate nested_loop",
    "T07": "string upper lower strip split replace find join startswith endswith concatenate "
           "count index slice slicing character",
    "T08": "list lists tuple tuples set sets append extend insert remove pop sort sorted count index "
           "reverse clear copy alias aliasing union intersection slice slicing immutable mutable",
    "T09": "dict dictionary dictionaries key keys value values items get update keyerror",
    "T10": "function def return parameter argument args kwargs recursion recursive lambda scope",
    "T11": "comprehension comprehensions",
    "T12": "exception exceptions try except finally raise error traceback valueerror typeerror "
           "zerodivisionerror",
    "T13": "file files open read write close readline readlines csv path",
}


# Everyday words that are also Python terms ("I don't GET it", "keep my DATA", "what does this
# VALUE mean"). They never count as topic evidence on their own; the topic classifier and the
# conversation's topic decide instead. Words are written in their stemmed form ("valu" = "value").
GENERIC_WORDS = {"get", "data", "program", "valu", "key", "item", "updat", "check", "work", "run",
                 "mak", "new", "creat", "number", "word", "nam", "tim", "thing", "exampl", "cod", "keys"}


@lru_cache(maxsize=4)
def _topic_evidence_words(topics_filepath: str = TOPICS_FILEPATH) -> Dict[str, set]:
    """Build, for each topic, the set of words that point at it: words from its own name and
    subtopic names that no other topic uses, plus its TOPIC_HINTS, minus GENERIC_WORDS. Cached."""
    catalog = load_catalog(topics_filepath)
    from_names = {
        tid: set().union(tokenize(catalog["topic_names"][tid]),
                         *(tokenize(catalog["subtopic_names"][sid]) for sid in sids))
        for tid, sids in catalog["topic_subtopics"].items()
    }
    words = {}
    for tid, tokens in from_names.items():
        distinctive = {t for t in tokens if sum(t in other for other in from_names.values()) == 1}
        words[tid] = (distinctive | tokenize(TOPIC_HINTS.get(tid, "").replace("_", " "))) - GENERIC_WORDS
    return words


def topic_evidence(text: str, topics_filepath: str = TOPICS_FILEPATH) -> Dict[str, int]:
    """Count how many words of the message point at each topic.
    Returns {topic_id: count}; topics with a count of 0 are left out."""
    tokens = tokenize(text)
    scores = {tid: len(tokens & words) for tid, words in _topic_evidence_words(topics_filepath).items()}
    return {tid: score for tid, score in scores.items() if score}


def current_lesson_topic(student_kb: Optional[dict]) -> Optional[str]:
    """Return the topic of the latest tutor lesson if the student's last recorded event was a
    lesson (not a quiz); otherwise None. Lets a follow-up message stay on the same topic."""
    history = (student_kb or {}).get("pedagogy_history", [])
    if history and history[-1].get("event") == "lesson":
        return history[-1].get("topic_id")
    return None
